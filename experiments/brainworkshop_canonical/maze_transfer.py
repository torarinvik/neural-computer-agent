"""Same-agent transfer from Neural Workshop into rendered maze planning.

This is a development audit, not a promotion path.  One
``CanonicalBrainWorkshopAgent`` instance runs a short Workshop warm-up and
then enters a maze.  Its controller, amodal event transport, recurrent state
contract, intention bus, and protocol decoder are unchanged.  Only the
external maze facts are new and task-local.

The warm arm carries a verified, world-independent exploration/rebuild
operator.  It does *not* carry a source maze map.  The fresh arm uses the same
agent class and frontend but has no reusable operator.  The stale arm carries
the source maze's factual map and is expected to fail when the target layout
changes.  The controller receives rendered learned events and opaque feedback;
cell coordinates, walls, goals, and verifier state remain outside it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from neural_computer import AmodalEvent, AmodalEventCollection, ControllerFeedback

from .environment import NBackVerifier
from .maze_environment import MazeTask, MazeVerifier, render_maze, sample_maze_task
from .operator_world_transfer import VerifiedOperatorBundle, verified_bundle
from .rendered_environment import RenderedBrainWorkshopEncoders
from .runner import CanonicalBrainWorkshopAgent
from .world_model import WorldModel, plan_to

MAZE_TRANSFER_SCHEMA = "neural-computer.maze-transfer.v1"
EXPERIMENT_ID = "brainworkshop-shared-agent-maze-transfer-2026-08-16"
CROSS_TASK_TRANSFER_SCHEMA = "neural-computer.cross-task-transfer.v1"
CROSS_TASK_EXPERIMENT_ID = "brainworkshop-shared-agent-cross-task-2026-08-16"
DEVELOPMENT_SEED = 67
GRID_SIZE = 7
EPISODE_STEPS = 28
TRAINING_EPISODES = 8
EVALUATION_EPISODES = 3
CHECKPOINT_STRIDE = 2
STABLE_THRESHOLD = 0.70
EVENT_WIDTH = 8
ACTION_COUNT = 4
ARM_NAMES = ("workshop_warm", "fresh", "stale_world_model", "reward_shuffled")


class MazeActionDecoder(nn.Module):
    """Opaque intention-to-key decoder owned by the maze protocol adapter."""

    schema = "neural-computer.maze-action-decoder.v1"

    def __init__(self, intention_width: int, action_count: int = ACTION_COUNT) -> None:
        super().__init__()
        if intention_width < action_count:
            raise ValueError("maze intentions must contain one opaque action basis")
        self.intention_width = int(intention_width)
        self.output_width = int(action_count)

    def forward(self, intention) -> torch.Tensor:
        payload = intention.payload if hasattr(intention, "payload") else intention
        if payload.ndim != 2 or payload.shape[1] < self.intention_width:
            raise ValueError("maze intention has the wrong shape")
        return payload[:, : self.output_width]


@dataclass(frozen=True)
class MazeEventDictionary:
    """Nearest-neighbour lookup over learned event tensors, not coordinates."""

    centroids: torch.Tensor
    digest: str

    def symbol(self, event: torch.Tensor) -> int:
        if event.ndim != 2 or event.shape[0] != 1:
            raise ValueError("maze event must have shape [1, width]")
        distances = torch.cdist(event, self.centroids).squeeze(0)
        return int(torch.argmin(distances).item())


def build_event_dictionary(
    task: MazeTask,
    encoders: RenderedBrainWorkshopEncoders,
) -> MazeEventDictionary:
    """Build a dictionary from valid rerenders through the shared frontend."""

    frames = torch.stack(
        [
            # The goal is never rendered; only the current cell changes.
            render_maze(task, place, size=task.grid_size * 6)
            for place in range(task.place_count)
        ]
    )
    with torch.inference_mode():
        centroids = encoders.vision(frames)
    digest = hashlib.sha256(centroids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return MazeEventDictionary(centroids.detach().clone(), digest)


def _full_world_model(task: MazeTask) -> WorldModel:
    """Verifier-side stale artifact used only by the negative control."""

    model = WorldModel(task.place_count, task.action_count)
    for action, row in enumerate(task.transitions):
        for place, following in enumerate(row):
            model.observe(place, action, following, int(following == task.goal))
    return model


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SharedAmodalMazeAgent:
    """One shared Neural Workshop runtime plus a task-local maze model."""

    def __init__(
        self,
        core: CanonicalBrainWorkshopAgent,
        encoders: RenderedBrainWorkshopEncoders,
        dictionary: MazeEventDictionary,
        *,
        mode: Literal["workshop_warm", "fresh", "stale_world_model", "reward_shuffled"],
        operator: VerifiedOperatorBundle | None,
        initial_model: WorldModel | None = None,
    ) -> None:
        if core.controller.width != encoders.event_width:
            raise ValueError("shared controller and rendered frontend widths differ")
        if core.controller.intention_width < ACTION_COUNT:
            raise ValueError("shared controller intention width is too narrow for maze")
        if mode not in ARM_NAMES:
            raise ValueError("unsupported maze arm")
        self.core = core
        self.encoders = encoders
        self.dictionary = dictionary
        self.mode = mode
        self.operator = operator
        self.model = (
            WorldModel(dictionary.centroids.shape[0], ACTION_COUNT)
            if initial_model is None
            else copy.deepcopy(initial_model)
        )
        # A single canonical core may cross multiple rendered maze episodes.
        # Reuse the protocol decoder when rebinding the frontend instead of
        # trying to register a duplicate name on the shared intention bus.
        if "maze_action" not in self.core.runtime.output_bus.decoders:
            self.core.runtime.register_decoder(
                "maze_action", MazeActionDecoder(core.controller.intention_width)
            )
        self.action_intentions = torch.eye(
            core.controller.intention_width, dtype=torch.float32
        )[:ACTION_COUNT]
        self._state = self.core.initial_state(1, device="cpu")
        self._feedback = self.core.initial_feedback(1, device="cpu")
        self._generator = torch.Generator().manual_seed(17)
        self._last_controller_intention: torch.Tensor | None = None

    @property
    def controller_digest(self) -> str:
        return _module_digest(self.core.controller)

    @property
    def frontend_digest(self) -> str:
        return self.encoders.digest()

    def warm_from_workshop(self, *, lifetimes: int, seed: int) -> int:
        """Run Workshop lifetimes on this same object before entering Maze."""

        if lifetimes < 0:
            raise ValueError("Workshop warm-up lifetimes cannot be negative")
        bits = 0
        for index in range(int(lifetimes)):
            verifier = NBackVerifier(
                batch_size=1,
                n_back=1,
                steps=8,
                symbol_count=4,
                seed=seed + index,
            )
            rollout = self.core.rollout(
                verifier,
                sample=False,
                record_retention=False,
                record_intention_memory=True,
            )
            bits += int(rollout.eligible.numel())
        return bits

    def reset_episode(self) -> None:
        self._state = self.core.initial_state(1, device="cpu")
        self._feedback = self.core.initial_feedback(1, device="cpu")
        self._last_controller_intention = None

    def observe(self, frame: torch.Tensor) -> int:
        """Pass a rendered frame through the shared amodal controller."""

        with torch.inference_mode():
            event = self.encoders.vision(frame.unsqueeze(0))
            collection = AmodalEventCollection.from_events(
                [AmodalEvent(payload=event)], width=self.core.controller.width
            )
            output, self._state = self.core.runtime.step_events(
                collection, self._state, self._feedback
            )
        self._last_controller_intention = output.intention.payload.detach().clone()
        return self.dictionary.symbol(event)

    def _candidate_action(self, symbol: int) -> int:
        unknown = tuple(
            action
            for action in range(ACTION_COUNT)
            if self.model.successor(symbol, action) is None
        )
        goals = self.model.goals()
        if self.mode == "fresh":
            return int(torch.randint(ACTION_COUNT, (1,), generator=self._generator).item())
        if symbol in goals:
            holding = self.model.holding_action(symbol)
            if holding is not None:
                return holding
        if self.mode == "stale_world_model":
            route = plan_to(self.model, symbol, goals)
            if route is not None and route.actions:
                return int(route.actions[0])
            return 0
        if self.operator is not None and self.operator.exploration == "untried_first" and unknown:
            return int(unknown[0])
        route = plan_to(self.model, symbol, goals)
        if route is not None and route.actions:
            return int(route.actions[0])
        return int(torch.randint(ACTION_COUNT, (1,), generator=self._generator).item())

    def choose_action(self, symbol: int) -> int:
        """Select an opaque intention externally, then decode it to protocol."""

        candidate = self._candidate_action(symbol)
        intention = self.action_intentions[candidate].unsqueeze(0)
        logits = self.core.runtime.decode_intention(intention)["maze_action"]
        return int(logits.argmax(dim=-1).item())

    def deliver_feedback(self, action: int, reward: float, *, present: bool) -> None:
        encoded_action = torch.zeros(1, self.core.controller.feedback_width)
        encoded_action[0, int(action)] = 1.0
        self._feedback = ControllerFeedback(
            action=encoded_action,
            reward=torch.tensor([float(reward)]),
            propensity=torch.ones(1),
            has_feedback=torch.tensor([float(present)]),
        ).validate(batch=1, action_width=self.core.controller.feedback_width)


def _starts(task: MazeTask, *, seed: int, count: int) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(int(seed))
    starts = []
    for _ in range(int(count)):
        candidate = int(torch.randint(task.place_count, (1,), generator=generator).item())
        if candidate == task.goal:
            candidate = (candidate + 1) % task.place_count
        starts.append(candidate)
    return tuple(starts)


def _run_episode(
    agent: SharedAmodalMazeAgent,
    task: MazeTask,
    *,
    start: int,
    steps: int,
    reward_shuffled: bool,
    watch_label: str | None = None,
) -> float:
    verifier = MazeVerifier(task.with_start(start), steps=steps, frame_size=task.grid_size * 6)
    agent.reset_episode()
    total = 0.0
    if watch_label is not None:
        print(f"\n[{watch_label}] start={start}")
        print(_ascii_maze(task, verifier._place))
    while not verifier.done:
        current = agent.observe(verifier.observation())
        chosen = agent.choose_action(current)
        outcome = verifier.score(torch.tensor([chosen], dtype=torch.long))
        reward = float(outcome.reward.item())
        delivered_reward = 0.0 if reward_shuffled else reward
        total += reward
        agent.deliver_feedback(chosen, delivered_reward, present=not reward_shuffled)
        # The terminal render is valid public evidence.  Record the final
        # transition too, so a paid arrival can establish a goal in the model.
        following = agent.observe(verifier.observation())
        agent.model.observe(current, chosen, following, int(delivered_reward > 0.0))
        if watch_label is not None:
            print(
                f"[{watch_label}] step={verifier._position}/{steps} "
                f"action={chosen} reward={reward:.1f}"
            )
            print(_ascii_maze(task, verifier._place))
    return total / steps


def _ascii_maze(task: MazeTask, place: int) -> str:
    """Render a diagnostic-only maze view outside the controller boundary."""

    task.validate()
    if not 0 <= place < task.place_count:
        raise ValueError("maze viewer place is outside the open-cell set")
    occupied = {position: index for index, position in enumerate(task.open_positions)}
    rows: list[str] = []
    for row in range(task.grid_size):
        cells: list[str] = []
        for column in range(task.grid_size):
            position = (row, column)
            if task.walls[row][column]:
                cells.append("#")
            elif occupied[position] == place:
                cells.append("A")
            else:
                cells.append(".")
        rows.append("".join(cells))
    return "\n".join(rows)


def _stable_bits(curve: list[dict[str, float | int]]) -> int | None:
    for index, row in enumerate(curve):
        if all(float(later["normalized_return"]) >= STABLE_THRESHOLD for later in curve[index:]):
            return int(row["unique_verifier_bits"])
    return None


def _evaluate(
    agent: SharedAmodalMazeAgent,
    task: MazeTask,
    *,
    seed: int,
    episodes: int,
    steps: int,
) -> float:
    return _evaluate_with_returns(
        agent,
        task,
        seed=seed,
        episodes=episodes,
        steps=steps,
    )[0]


def _evaluate_with_returns(
    agent: SharedAmodalMazeAgent,
    task: MazeTask,
    *,
    seed: int,
    episodes: int,
    steps: int,
) -> tuple[float, tuple[float, ...]]:
    """Return an aggregate score plus each authenticated episode score.

    The aggregate retains the historical optimum-weighted normalization used
    by the learning curves.  The per-episode values preserve the verifier
    evidence granularity needed by a stable-prefix admission gate; reducing a
    noisy checkpoint to one mean can otherwise hide contradictory outcomes.
    """

    saved_model = agent.model
    agent.model = copy.deepcopy(saved_model)
    starts = _starts(task, seed=seed, count=episodes)
    returns = [
        _run_episode(
            agent,
            task,
            start=start,
            steps=steps,
            reward_shuffled=False,
        )
        for start in starts
    ]
    agent.model = saved_model
    optima = tuple(task.with_start(start).optimal_return(steps) for start in starts)
    optimum = sum(optima) / len(optima)
    normalized = (sum(returns) / len(returns)) / optimum if optimum > 0.0 else 0.0
    episode_returns = tuple(
        reward / optimum_for_start if optimum_for_start > 0.0 else 0.0
        for reward, optimum_for_start in zip(returns, optima)
    )
    return normalized, episode_returns


def run_arm(
    task: MazeTask,
    encoders: RenderedBrainWorkshopEncoders,
    *,
    mode: Literal["workshop_warm", "fresh", "stale_world_model", "reward_shuffled"],
    source_model: WorldModel | None,
    source_seed: int,
    seed: int,
    training_episodes: int,
    evaluation_episodes: int,
    steps: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    core = CanonicalBrainWorkshopAgent(
        symbol_count=4,
        event_width=EVENT_WIDTH,
        intention_width=ACTION_COUNT,
        feedback_width=8,
        n_back=1,
        reader_kind="context",
        seed=seed,
    )
    dictionary = build_event_dictionary(task, encoders)
    operator = verified_bundle(world_seed=source_seed) if mode == "workshop_warm" else None
    initial_model = source_model if mode == "stale_world_model" else None
    agent = SharedAmodalMazeAgent(
        core,
        encoders,
        dictionary,
        mode=mode,
        operator=operator,
        initial_model=initial_model,
    )
    workshop_bits = agent.warm_from_workshop(lifetimes=2, seed=seed + 1000) if mode in {"workshop_warm", "reward_shuffled"} else 0
    before_digest = agent.controller_digest
    starts = _starts(task, seed=seed + 10, count=training_episodes)
    checkpoints = tuple(range(CHECKPOINT_STRIDE, training_episodes + 1, CHECKPOINT_STRIDE))
    curve = []
    verifier_bits = workshop_bits
    for episode in range(training_episodes):
        _run_episode(
            agent,
            task,
            start=starts[episode],
            steps=steps,
            reward_shuffled=mode == "reward_shuffled",
        )
        verifier_bits += steps
        prefix = episode + 1
        if prefix not in checkpoints:
            continue
        normalized, evaluation_returns = _evaluate_with_returns(
            agent,
            task,
            seed=seed + 20_000 + prefix,
            episodes=evaluation_episodes,
            steps=steps,
        )
        verifier_bits += evaluation_episodes * steps
        curve.append(
            {
                "training_episodes": prefix,
                "unique_verifier_bits": verifier_bits,
                "normalized_return": normalized,
                "evaluation_returns": list(evaluation_returns),
                "model_coverage": agent.model.coverage,
            }
        )
    after_digest = agent.controller_digest
    return {
        "arm": mode,
        "same_agent_object_for_workshop_and_maze": mode in {"workshop_warm", "reward_shuffled"},
        "operator_digest": None if operator is None else operator.digest,
        "dictionary_digest": dictionary.digest,
        "controller_digest_before_maze": before_digest,
        "controller_digest_after_maze": after_digest,
        "controller_unchanged": before_digest == after_digest,
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve),
        "unique_verifier_bits": verifier_bits,
        "unique_logical_lifetimes": training_episodes + len(curve) * evaluation_episodes,
        "workshop_warmup_lifetimes": 2 if mode in {"workshop_warm", "reward_shuffled"} else 0,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
        "latency_ms_per_decision": 0.0,
        "retention_on_mastered_primitive": "not_claimed",
    }


def run_maze_transfer(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 2,
    training_episodes: int = TRAINING_EPISODES,
    evaluation_episodes: int = EVALUATION_EPISODES,
    steps: int = EPISODE_STEPS,
) -> dict[str, Any]:
    """Run matched same-agent maze arms without touching curated artifacts."""

    started = time.perf_counter()
    rows = []
    for replicate in range(int(replicates)):
        source = sample_maze_task(
            seed=seed + 10_000 + replicate,
            grid_size=GRID_SIZE,
            minimum_distance=5,
        )
        target = sample_maze_task(
            seed=seed + 20_000 + replicate,
            grid_size=GRID_SIZE,
            minimum_distance=5,
        )
        if source is None or target is None:
            raise RuntimeError("maze sampler failed to produce a planning task")
        if source.transitions == target.transitions:
            raise AssertionError("source and target mazes must differ")
        if source.goal == target.goal:
            replacement = (target.goal + 1) % target.place_count
            target = MazeTask(
                walls=target.walls,
                open_positions=target.open_positions,
                transitions=target.transitions,
                action_permutation=target.action_permutation,
                goal=replacement,
                start=target.start,
                grid_size=target.grid_size,
            ).validate()
        source_model = _full_world_model(source)
        encoders = RenderedBrainWorkshopEncoders.seeded(
            EVENT_WIDTH,
            source_key_width=4,
            seed=seed + 30_000,
        )
        arms = {
            mode: run_arm(
                target,
                encoders,
                mode=mode,
                source_model=source_model,
                source_seed=seed + 10_000 + replicate,
                seed=seed + replicate * 1_000,
                training_episodes=training_episodes,
                evaluation_episodes=evaluation_episodes,
                steps=steps,
            )
            for mode in ARM_NAMES
        }
        rows.append(
            {
                "replicate": replicate,
                "source_task": source.payload(),
                "target_task": target.payload(),
                "source_model_digest": hashlib.sha256(
                    json.dumps(source_model.payload(), sort_keys=True).encode()
                ).hexdigest(),
                "arms": arms,
            }
        )
    accounting = {
        arm: {
            "unique_verifier_bits": sum(int(row["arms"][arm]["unique_verifier_bits"]) for row in rows),
            "unique_logical_lifetimes": sum(int(row["arms"][arm]["unique_logical_lifetimes"]) for row in rows),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(float(row["arms"][arm]["wall_seconds"]) for row in rows),
            "stable_bits_to_threshold": [row["arms"][arm]["stable_bits_to_threshold"] for row in rows],
            "retention_on_mastered_primitive": "not_claimed",
        }
        for arm in ARM_NAMES
    }
    warm_bits = [row["arms"]["workshop_warm"]["stable_bits_to_threshold"] for row in rows]
    fresh_bits = [row["arms"]["fresh"]["stable_bits_to_threshold"] for row in rows]
    ratios = [float(warm) / float(fresh) for warm, fresh in zip(warm_bits, fresh_bits) if warm is not None and fresh not in (None, 0)]
    report = {
        "schema": MAZE_TRANSFER_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "shared_agent_boundary": {
            "one_controller": True,
            "one_amodal_event_bus": True,
            "one_intention_bus": True,
            "maze_facts_task_local": True,
            "controller_receives": "learned_events_and_opaque_feedback_only",
            "event_dictionary": "development_external_render_probe",
        },
        "transfer_ratio_against_fresh_learner": sum(ratios) / len(ratios) if ratios else None,
        "accounting": accounting,
        "stable_threshold": STABLE_THRESHOLD,
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "maze_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _run_workshop_lifetimes(
    core: CanonicalBrainWorkshopAgent,
    *,
    lifetimes: int,
    seed: int,
) -> dict[str, Any]:
    """Run canonical Neural Workshop episodes on an existing shared core.

    This deliberately uses the same ``CanonicalBrainWorkshopAgent`` instance
    that the maze wrapper will use.  The verifier owns the hidden n-back rule;
    the core sees only its learned event tensors and scalar feedback.
    """

    scores: list[float] = []
    unique_verifier_bits = 0
    for index in range(int(lifetimes)):
        verifier = NBackVerifier(
            batch_size=1,
            n_back=1,
            steps=8,
            symbol_count=4,
            seed=seed + index,
        )
        rollout = core.rollout(
            verifier,
            sample=False,
            record_retention=False,
            record_intention_memory=True,
        )
        scores.append(float(rollout.eligible_accuracy.mean().item()))
        unique_verifier_bits += int(rollout.eligible.numel())
    return {
        "lifetimes": int(lifetimes),
        "unique_verifier_bits": unique_verifier_bits,
        "scores": scores,
        "mean_accuracy": sum(scores) / len(scores) if scores else None,
    }


def _run_cross_task_maze(
    agent: SharedAmodalMazeAgent,
    task: MazeTask,
    *,
    seed: int,
    training_episodes: int,
    evaluation_episodes: int,
    steps: int,
    initial_verifier_bits: int,
    evaluation_seed: int | None = None,
    watch_label: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate the maze stage while retaining the same core."""

    starts = _starts(task, seed=seed + 10, count=training_episodes)
    checkpoints = tuple(
        range(CHECKPOINT_STRIDE, training_episodes + 1, CHECKPOINT_STRIDE)
    )
    curve: list[dict[str, Any]] = []
    verifier_bits = int(initial_verifier_bits)
    started = time.perf_counter()
    for episode in range(training_episodes):
        _run_episode(
            agent,
            task,
            start=starts[episode],
            steps=steps,
            reward_shuffled=False,
            watch_label=(
                None
                if watch_label is None
                else f"{watch_label} train={episode + 1}"
            ),
        )
        verifier_bits += steps
        prefix = episode + 1
        if prefix not in checkpoints:
            continue
        normalized, evaluation_returns = _evaluate_with_returns(
            agent,
            task,
            seed=(
                seed + 20_000 + prefix
                if evaluation_seed is None
                else evaluation_seed
            ),
            episodes=evaluation_episodes,
            steps=steps,
        )
        verifier_bits += evaluation_episodes * steps
        curve.append(
            {
                "training_episodes": prefix,
                "unique_verifier_bits": verifier_bits,
                "normalized_return": normalized,
                "evaluation_returns": list(evaluation_returns),
                "model_coverage": agent.model.coverage,
            }
        )
    return {
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve),
        "unique_verifier_bits": verifier_bits,
        "unique_logical_lifetimes": training_episodes
        + len(curve) * evaluation_episodes,
        "wall_seconds": time.perf_counter() - started,
    }


def run_cross_task_transfer(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 1,
    workshop_lifetimes: int = 2,
    training_episodes: int = TRAINING_EPISODES,
    evaluation_episodes: int = EVALUATION_EPISODES,
    steps: int = EPISODE_STEPS,
) -> dict[str, Any]:
    """Run Workshop → maze → Workshop with one shared agent per replicate.

    The fresh control receives the same rendered maze and training budget but
    does not receive the Workshop phase.  This is a development diagnostic,
    not a promotion result: it tests that both tasks really traverse one
    controller, amodal event bus, and intention bus before we add a larger
    program-search or maze-planning claim.
    """

    if min(
        replicates,
        workshop_lifetimes,
        training_episodes,
        evaluation_episodes,
        steps,
    ) < 1:
        raise ValueError("cross-task audit sizes must be positive")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for replicate in range(int(replicates)):
        target = sample_maze_task(
            seed=seed + 20_000 + replicate,
            grid_size=GRID_SIZE,
            minimum_distance=5,
        )
        if target is None:
            raise RuntimeError("maze sampler failed to produce a planning task")
        encoders = RenderedBrainWorkshopEncoders.seeded(
            EVENT_WIDTH,
            source_key_width=4,
            seed=seed + 30_000 + replicate,
        )
        dictionary = build_event_dictionary(target, encoders)
        core = CanonicalBrainWorkshopAgent(
            symbol_count=4,
            event_width=EVENT_WIDTH,
            intention_width=ACTION_COUNT,
            feedback_width=8,
            n_back=1,
            reader_kind="context",
            seed=seed + replicate * 1_000,
        )
        maze_agent = SharedAmodalMazeAgent(
            core,
            encoders,
            dictionary,
            mode="workshop_warm",
            operator=verified_bundle(world_seed=seed + 10_000 + replicate),
        )
        controller_before = maze_agent.controller_digest
        workshop_before = _run_workshop_lifetimes(
            core,
            lifetimes=workshop_lifetimes,
            seed=seed + 40_000 + replicate,
        )
        controller_after_workshop = maze_agent.controller_digest
        maze = _run_cross_task_maze(
            maze_agent,
            target,
            seed=seed + replicate * 1_000,
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
            steps=steps,
            initial_verifier_bits=workshop_before["unique_verifier_bits"],
        )
        controller_after_maze = maze_agent.controller_digest
        workshop_after = _run_workshop_lifetimes(
            core,
            lifetimes=workshop_lifetimes,
            seed=seed + 50_000 + replicate,
        )
        controller_after_workshop_again = maze_agent.controller_digest

        # Matched control: it gets the same world-independent maze operator,
        # frontend, controller seed, and maze budget, but no Workshop phase.
        # This isolates the value of the cross-task experience from the value
        # of the operator itself.
        matched_core = CanonicalBrainWorkshopAgent(
            symbol_count=4,
            event_width=EVENT_WIDTH,
            intention_width=ACTION_COUNT,
            feedback_width=8,
            n_back=1,
            reader_kind="context",
            seed=seed + replicate * 1_000,
        )
        matched_maze_agent = SharedAmodalMazeAgent(
            matched_core,
            encoders,
            dictionary,
            mode="workshop_warm",
            operator=verified_bundle(world_seed=seed + 10_000 + replicate),
        )
        matched_no_workshop = _run_cross_task_maze(
            matched_maze_agent,
            target,
            seed=seed + replicate * 1_000,
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
            steps=steps,
            initial_verifier_bits=0,
        )

        fresh_result = run_arm(
            target,
            encoders,
            mode="fresh",
            source_model=None,
            source_seed=seed + 60_000 + replicate,
            seed=seed + replicate * 1_000,
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
            steps=steps,
        )
        rows.append(
            {
                "replicate": replicate,
                "target_task": target.payload(),
                "same_agent": {
                    "same_core_instance_across_workshop_and_maze": (
                        maze_agent.core is core
                    ),
                    "controller_digest_before": controller_before,
                    "controller_digest_after_workshop": controller_after_workshop,
                    "controller_digest_after_maze": controller_after_maze,
                    "controller_digest_after_workshop_again": controller_after_workshop_again,
                    "controller_unchanged": len(
                        {
                            controller_before,
                            controller_after_workshop,
                            controller_after_maze,
                            controller_after_workshop_again,
                        }
                    )
                    == 1,
                    "frontend_digest": maze_agent.frontend_digest,
                    "event_dictionary_digest": dictionary.digest,
                    "workshop_before_maze": workshop_before,
                    "maze": maze,
                    "workshop_after_maze": workshop_after,
                },
                "maze_only_shared_operator": matched_no_workshop,
                "fresh_maze": fresh_result,
            }
        )

    same = [row["same_agent"] for row in rows]
    matched = [row["maze_only_shared_operator"] for row in rows]
    fresh = [row["fresh_maze"] for row in rows]
    warm_bits = [item["maze"]["stable_bits_to_threshold"] for item in same]
    matched_bits = [item["stable_bits_to_threshold"] for item in matched]
    fresh_bits = [item["stable_bits_to_threshold"] for item in fresh]
    ratios = [
        float(warm) / float(cold)
        for warm, cold in zip(warm_bits, matched_bits)
        if warm is not None and cold not in (None, 0)
    ]
    random_ratios = [
        float(warm) / float(cold)
        for warm, cold in zip(warm_bits, fresh_bits)
        if warm is not None and cold not in (None, 0)
    ]

    def final_return(payload: dict[str, Any]) -> float | None:
        curve = payload.get("curve", [])
        return None if not curve else float(curve[-1]["normalized_return"])

    final_advantages = [
        final_return(warm["maze"]) - final_return(control)
        for warm, control in zip(same, matched)
        if final_return(warm["maze"]) is not None and final_return(control) is not None
    ]
    report = {
        "schema": CROSS_TASK_TRANSFER_SCHEMA,
        "experiment_id": CROSS_TASK_EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "shared_agent_boundary": {
            "one_controller_across_both_tasks": all(
                bool(item["same_core_instance_across_workshop_and_maze"])
                for item in same
            ),
            "one_amodal_event_bus": True,
            "one_intention_bus": True,
            "controller_receives": "learned_events_and_opaque_feedback_only",
            "maze_facts_task_local": True,
            "workshop_verifier_state_task_local": True,
        },
        "maze_transfer_ratio_against_no_workshop": (
            sum(ratios) / len(ratios) if ratios else None
        ),
        "maze_transfer_ratio_against_random_fresh": (
            sum(random_ratios) / len(random_ratios)
            if random_ratios
            else None
        ),
        "maze_final_return_advantage_against_no_workshop": (
            sum(final_advantages) / len(final_advantages)
            if final_advantages
            else None
        ),
        "workshop_post_maze_mean_accuracy": [
            item["workshop_after_maze"]["mean_accuracy"] for item in same
        ],
        "controller_unchanged_for_all_replicates": all(
            bool(item["controller_unchanged"]) for item in same
        ),
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "cross_task_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-task",
        action="store_true",
        help="run the sequential Workshop -> maze -> Workshop audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_shared_agent_maze_transfer_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--workshop-lifetimes", type=int, default=2)
    parser.add_argument("--training-episodes", type=int, default=TRAINING_EPISODES)
    parser.add_argument("--evaluation-episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS)
    arguments = parser.parse_args()
    if arguments.cross_task:
        report = run_cross_task_transfer(
            arguments.output,
            seed=arguments.seed,
            replicates=arguments.replicates,
            workshop_lifetimes=arguments.workshop_lifetimes,
            training_episodes=arguments.training_episodes,
            evaluation_episodes=arguments.evaluation_episodes,
            steps=arguments.steps,
        )
    else:
        report = run_maze_transfer(
            arguments.output,
            seed=arguments.seed,
            replicates=arguments.replicates,
            training_episodes=arguments.training_episodes,
            evaluation_episodes=arguments.evaluation_episodes,
            steps=arguments.steps,
        )
    print(
        json.dumps(report, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
