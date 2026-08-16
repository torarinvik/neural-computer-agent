"""Development-only closed-loop diagnostic for persistent causal identity v2.

The loop is intentionally narrow but real:

``rendered frame -> frozen learned events -> external identity artifact ->
policy-free amodal planner -> opaque decoder action -> marker transition ->
receipt-linked scalar feedback``.

The identity artifact is the only persistent component.  The controller and
curated bank remain untouched.  The feeder uses two non-overlapping position
bands so slot order changes between episodes without exposing coordinates to
the controller; positions are retained only by the rendered-world fixture and
scoring-side diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    ExternalCausalIdentityArtifact,
    ExternalCausalIdentityAssignment,
    ExternalModelBasedPlanner,
    LiveActionReceipt,
    LiveOutcomeEvent,
    PersistentCausalIdentityV2,
    PolicyFreeAmodalLiveMachine,
    PolicyFreeAmodalRuntime,
    ResolvedLiveOutcome,
)
from neural_computer.promotion import sha256_file

from .current_symbol_acquire import FRONTEND_SEED
from .object_scene import render_markers, scene_slots
from .rendered_environment import RenderedBrainWorkshopEncoders

EXPERIMENT_ID = "brainworkshop-persistent-identity-v2-2026-08-16"
PERSISTENT_IDENTITY_V2_EXPERIMENT_SCHEMA = (
    "neural-computer.persistent-identity-v2-closed-loop.v1"
)
DEVELOPMENT_SEED = 47
EVENT_WIDTH = 4
STATE_WIDTH = 12
INTENTION_WIDTH = 2
FRAME_SIZE = 36
EPISODE_STEPS = 8
EPISODES = 3
IDENTITY_MARGIN = 0.15
IDENTITY_MINIMUM_EVIDENCE = 0.2
IDENTITY_MINIMUM_SIMILARITY = 0.65
RECOVERY_EPISODES = 2


class _AdditiveFactualModel(nn.Module):
    state_width = STATE_WIDTH
    intention_width = INTENTION_WIDTH

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :INTENTION_WIDTH] += intention
        return result


class _Decision:
    def __init__(self, action: torch.Tensor, propensity: torch.Tensor) -> None:
        self.action = action
        self.propensity = propensity


class _CyclingDecoder:
    """A deterministic opaque decoder that supplies varied logged actions."""

    intention_width = INTENTION_WIDTH
    _pattern = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([0.0, -1.0]),
    )

    def __init__(self) -> None:
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def decide(self, intention, *, sample: bool = True):
        del sample
        action = self._pattern[self._step % len(self._pattern)].to(
            device=intention.payload.device,
            dtype=intention.payload.dtype,
        )
        self._step += 1
        return _Decision(
            action.unsqueeze(0).expand(intention.payload.shape[0], -1),
            torch.ones(intention.payload.shape[0], device=intention.payload.device),
        )


def _machine(
    *,
    identity_assignment: ExternalCausalIdentityAssignment | None,
    goal_state_candidates: torch.Tensor | None,
) -> PolicyFreeAmodalLiveMachine:
    controller = AmodalCognitiveController(
        width=EVENT_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=INTENTION_WIDTH,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
    )
    goal = torch.zeros(1, STATE_WIDTH)
    return PolicyFreeAmodalLiveMachine(
        policy_free,
        _CyclingDecoder(),
        goal_state=goal,
        goal_state_candidates=goal_state_candidates,
        identity_assignment=identity_assignment,
        candidate_intentions=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        output_key="persistent-identity-v2",
        sample=False,
    )


@dataclass
class _MarkerWorld:
    low: int
    high: int
    controlled: int
    distractor: int
    reverse: bool = False

    @property
    def controlled_slot(self) -> int:
        return int(self.controlled > self.distractor)

    def frame(self) -> torch.Tensor:
        return render_markers(
            (self.controlled, self.distractor),
            size=FRAME_SIZE,
        )

    def step(self, action: torch.Tensor) -> float:
        values = action.reshape(-1)
        direction = 1 if float(values[0]) > 0.0 or float(values[1]) > 0.0 else -1
        if self.reverse:
            direction *= -1
        width = self.high - self.low + 1
        self.controlled = self.low + ((self.controlled - self.low + direction) % width)
        return float(self.controlled == self.high)


def _pixel_events(
    encoders: RenderedBrainWorkshopEncoders,
    frame: torch.Tensor,
) -> AmodalEventCollection:
    # The diagnostic frontend is frozen.  Inference mode avoids constructing
    # an autograd graph for every rendered tick, which otherwise dominates
    # short closed-loop development runs without changing any tensor value.
    with torch.inference_mode():
        payload = encoders.vision(torch.stack(scene_slots(frame)))
    return AmodalEventCollection.from_events(
        [
            AmodalEvent(payload=payload[index : index + 1])
            for index in range(payload.shape[0])
        ],
        width=encoders.event_width,
    )


def _assignment_gate() -> ExternalCausalIdentityAssignment:
    return ExternalCausalIdentityAssignment(
        margin=IDENTITY_MARGIN,
        minimum_evidence=IDENTITY_MINIMUM_EVIDENCE,
    )


def _world_for(episode: int, *, reverse: bool = False) -> _MarkerWorld:
    if episode % 2 == 0:
        return _MarkerWorld(0, 2, 0, 7, reverse=reverse)
    return _MarkerWorld(5, 7, 7, 0, reverse=reverse)


def _receipt_outcome(
    pending: tuple[str, LiveActionReceipt, object, float] | None,
    *,
    now: float,
) -> tuple[str, ResolvedLiveOutcome] | None:
    if pending is None:
        return None
    owner, receipt, proposal, reward = pending
    outcome = LiveOutcomeEvent(
        receipt_id=receipt.receipt_id,
        reward=torch.tensor([reward]),
        present=torch.tensor([True]),
        observed_at=now,
    )
    return owner, ResolvedLiveOutcome(outcome, receipt, proposal)


def _run_arm(
    arm: str,
    encoders: RenderedBrainWorkshopEncoders,
    *,
    episodes: int = EPISODES,
    steps: int = EPISODE_STEPS,
    reverse_episode: int | None = None,
    persistent_factory=PersistentCausalIdentityV2,
    world_factory=_world_for,
) -> dict[str, Any]:
    if arm not in {"no_persistent", "episode_local", "persistent_v2", "oracle"}:
        raise ValueError(f"unknown persistent identity arm: {arm}")
    goals = torch.zeros(2, STATE_WIDTH)
    goals[0, 0] = 1.0
    goals[1, 0] = -1.0
    persistent = (
        persistent_factory(
            minimum_similarity=IDENTITY_MINIMUM_SIMILARITY,
            recovery_episodes=RECOVERY_EPISODES,
        )
        if arm == "persistent_v2"
        else None
    )
    local_artifact = ExternalCausalIdentityArtifact() if arm == "episode_local" else None
    persistent_model = persistent
    totals = {
        "reward": 0.0,
        "steps": 0,
        "primary_emissions": 0,
        "probe_emissions": 0,
        "identity_abstentions": 0,
        "confidently_wrong": 0,
    }
    episode_rows: list[dict[str, Any]] = []
    receipt_id = 0
    for episode in range(int(episodes)):
        world = world_factory(
            episode,
            reverse=reverse_episode is not None and episode == reverse_episode,
        )
        assignment_machine = _machine(
            identity_assignment=None if arm == "no_persistent" else _assignment_gate(),
            goal_state_candidates=None if arm == "no_persistent" else goals,
        )
        probe_machine = _machine(identity_assignment=None, goal_state_candidates=None)
        local_artifact = (
            ExternalCausalIdentityArtifact() if arm == "episode_local" else local_artifact
        )
        # Fixed-size buffers keep the short history view allocation-free.  The
        # earlier list/stack path copied the complete prefix on every tick,
        # turning a long development episode into an avoidable O(T^2) cost.
        event_history = torch.empty(
            steps, 2, EVENT_WIDTH, dtype=torch.float32
        )
        action_history = torch.empty(
            max(1, steps), INTENTION_WIDTH, dtype=torch.float32
        )
        pending: tuple[str, LiveActionReceipt, object, float] | None = None
        episode_row = {
            "episode": episode,
            "controlled_slot": world.controlled_slot,
            "reverse": world.reverse,
            "primary_emissions": 0,
            "probe_emissions": 0,
            "identity_abstentions": 0,
            "confidently_wrong": 0,
            "reward": 0.0,
            "statuses": [],
            "reasons": [],
            "selected_slots": [],
        }
        for step in range(int(steps)):
            events = _pixel_events(encoders, world.frame())
            event_history[step].copy_(events.payload.squeeze(0))
            history = event_history[: step + 1].unsqueeze(0)
            if arm == "oracle":
                identity_evidence = torch.zeros(1, 2)
                identity_evidence[0, world.controlled_slot] = 3.0
                episode_row["statuses"].append("oracle")
                episode_row["reasons"].append("evaluation_only")
            elif history.shape[1] >= 4:
                actions = action_history[:step].unsqueeze(0)
                if arm == "persistent_v2":
                    assert persistent is not None
                    persistent.resolve(history, actions, episode_id=episode)
                    identity_evidence = persistent.last_evidence
                    assert identity_evidence is not None
                    episode_row["statuses"].append(persistent.status)
                    episode_row["reasons"].append(persistent.reason)
                elif arm == "episode_local":
                    assert local_artifact is not None
                    identity_evidence = local_artifact.evidence(history, actions)
                    episode_row["statuses"].append("episode_local")
                    episode_row["reasons"].append("local_evidence")
                else:
                    identity_evidence = torch.zeros(1, 2)
            else:
                identity_evidence = torch.zeros(1, 2)
                if arm != "no_persistent":
                    episode_row["statuses"].append("warmup")
                    episode_row["reasons"].append("insufficient_history")

            routed = _receipt_outcome(pending, now=float(step))
            primary_outcomes = ()
            probe_outcomes = ()
            if routed is not None:
                owner, outcome = routed
                if owner == "primary":
                    primary_outcomes = (outcome,)
                else:
                    probe_outcomes = (outcome,)
            primary = assignment_machine.tick(
                events,
                primary_outcomes,
                now=float(step),
                elapsed=1.0,
                identity_evidence=(
                    None if arm == "no_persistent" else identity_evidence
                ),
            )
            probe = probe_machine.tick(
                events,
                probe_outcomes,
                now=float(step),
                elapsed=1.0,
            )
            if primary:
                owner, proposals = "primary", primary
                episode_row["primary_emissions"] += len(primary)
                totals["primary_emissions"] += len(primary)
            else:
                owner, proposals = "probe", probe
                episode_row["probe_emissions"] += len(probe)
                totals["probe_emissions"] += len(probe)
                if arm != "no_persistent":
                    episode_row["identity_abstentions"] += 1
                    totals["identity_abstentions"] += 1
            if not proposals:
                raise RuntimeError("closed-loop diagnostic lost its probe action")
            proposal = proposals[0]
            assignment = assignment_machine.last_identity_assignment
            if owner == "primary" and assignment is not None:
                selected = (
                    None
                    if bool(assignment.abstained[0])
                    else int(assignment.selected_slot[0].item())
                )
                episode_row["selected_slots"].append(selected)
                if selected is not None and selected != world.controlled_slot:
                    episode_row["confidently_wrong"] += 1
                    totals["confidently_wrong"] += 1
            reward = world.step(proposal.action)
            episode_row["reward"] += reward
            episode_row["steps"] = step + 1
            totals["reward"] += reward
            totals["steps"] += 1
            action_history[step].copy_(proposal.action.squeeze(0))
            receipt_id += 1
            receipt = LiveActionReceipt(
                receipt_id=receipt_id,
                action=proposal.action,
                propensity=proposal.propensity,
                output_key=proposal.output_key,
                emitted_at=float(step),
                model_version=proposal.model_version,
            )
            pending = (owner, receipt, proposal, reward)
        episode_rows.append(episode_row)
    result = {
        "arm": arm,
        "episodes": episode_rows,
        "integrated_return": totals["reward"] / max(1, totals["steps"]),
        "total_reward": totals["reward"],
        "steps": totals["steps"],
        "primary_emissions": totals["primary_emissions"],
        "probe_emissions": totals["probe_emissions"],
        "identity_abstentions": totals["identity_abstentions"],
        "abstention_rate": totals["identity_abstentions"] / max(1, totals["steps"]),
        "confidently_wrong": totals["confidently_wrong"],
        "persistent_status": None if persistent_model is None else persistent_model.status,
        "persistent_support": None if persistent_model is None else persistent_model.support,
        "quarantine_count": None
        if persistent_model is None
        else persistent_model.quarantine_count,
    }
    if persistent_model is not None:
        timeline = [
            (episode["episode"] * steps + step, status, reason)
            for episode in episode_rows
            for step, (status, reason) in enumerate(
                zip(episode["statuses"], episode["reasons"])
            )
        ]
        quarantined = [index for index, status, _ in timeline if status == "quarantined"]
        relearned = [
            index
            for index, _, reason in timeline
            if reason == "relearned_requires_confirmation"
        ]
        result["detection_delay_steps"] = (
            None if not quarantined else quarantined[0]
        )
        result["recovery_delay_steps"] = (
            None
            if not quarantined or not relearned
            else relearned[0] - quarantined[0]
        )
        result["post_recovery_return"] = (
            None
            if not relearned
            else sum(
                episode["reward"]
                for episode in episode_rows
                if episode["episode"] * steps >= relearned[0]
            )
            / max(
                1,
                sum(
                    episode["steps"]
                    for episode in episode_rows
                    if episode["episode"] * steps >= relearned[0]
                ),
            )
        )
    return result


def _control_report(
    encoders: RenderedBrainWorkshopEncoders,
    *,
    model_factory=PersistentCausalIdentityV2,
    world_factory=_world_for,
    controlled_slot: int = 0,
) -> dict[str, Any]:
    """Run controls against already-rendered learned histories."""

    base_world = world_factory(0)
    history: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    pattern = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0]),
        torch.tensor([0.0, -1.0]),
    )
    for step in range(8):
        history.append(_pixel_events(encoders, base_world.frame()).payload.squeeze(0))
        action = pattern[step % len(pattern)]
        actions.append(action)
        base_world.step(action)
    events = torch.stack(history).unsqueeze(0)
    action_tensor = torch.stack(actions[:-1]).unsqueeze(0)
    shuffled = action_tensor[:, torch.tensor([3, 0, 6, 2, 5, 1, 4])]
    model = model_factory(recovery_episodes=2)
    initial = model.resolve(events, action_tensor, episode_id=0)
    shuffled_result = model.resolve(events, shuffled, episode_id=1)
    shuffled_status = model.status
    missing = torch.ones(1, events.shape[1], events.shape[2], dtype=torch.bool)
    missing[:, 4, 0] = False
    missing_result = model.resolve(events, action_tensor, event_present=missing, episode_id=2)
    missing_status = model.status
    equivalent_model = model_factory()
    equivalent = events.clone()
    equivalent[:, :, 1] = equivalent[:, :, 0]
    equivalence_result = equivalent_model.resolve(equivalent, action_tensor, episode_id=0)
    partial = events.clone()
    distractor_slot = 1 - int(controlled_slot)
    partial[:, 1:, distractor_slot] = partial[:, 1:, controlled_slot]
    partial[:, 2::2, distractor_slot] = events[:, 2::2, distractor_slot]
    partial_model = model_factory()
    partial_result = partial_model.resolve(partial, action_tensor, episode_id=0)
    crossing = events.clone()
    crossing[:, 4:] = crossing[:, 4:, [1, 0]]
    crossing_model = model_factory()
    crossing_result = crossing_model.resolve(crossing, action_tensor, episode_id=0)
    birth_death_model = model_factory()
    birth_death_result = birth_death_model.resolve(
        events[:, :, controlled_slot : controlled_slot + 1],
        action_tensor,
        episode_id=0,
    )
    poisoned_model = model_factory()
    poisoned_events = events.clone()
    poisoned_events[:, :, controlled_slot] = events[:, :, distractor_slot]
    poisoned_events[:, :, distractor_slot] = events[:, :, controlled_slot]
    poisoned_model.resolve(poisoned_events, action_tensor, episode_id=0)
    poisoned_recovery = poisoned_model.resolve(events, action_tensor, episode_id=1)
    return {
        "action_shuffled_abstained": bool(shuffled_result.abstained[0]),
        "action_shuffled_status": shuffled_status,
        "missing_evidence_abstained": bool(missing_result.abstained[0]),
        "missing_evidence_status": missing_status,
        "exact_equivalence_abstained": bool(equivalence_result.abstained[0]),
        "exact_equivalence_support": equivalent_model.support,
        "partial_mimic_abstained": bool(partial_result.abstained[0]),
        "crossing_track_order_abstained": bool(crossing_result.abstained[0]),
        "birth_death_single_track_assigned": not bool(birth_death_result.abstained[0]),
        "poisoned_recovery_abstained": bool(poisoned_recovery.abstained[0]),
        "poisoned_recovery_status": poisoned_model.status,
        "initial_assignment": int(initial.selected_slot[0].item()),
    }


def run_persistent_identity_v2(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    steps: int = EPISODE_STEPS,
    episodes: int = EPISODES,
) -> dict[str, Any]:
    if steps < 4 or episodes < 2:
        raise ValueError("persistent identity v2 needs at least two four-step episodes")
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    encoders = RenderedBrainWorkshopEncoders.seeded(
        EVENT_WIDTH,
        source_key_width=4,
        seed=FRONTEND_SEED,
    )
    frontend_digest_before = encoders.digest()
    repository = Path(__file__).parents[2]
    bank_path = repository / "artifacts/checkpoints" / "AgentBrain.bank"
    bank_digest_before = sha256_file(bank_path) if bank_path.is_file() else None
    arms = {
        arm: _run_arm(arm, encoders, episodes=episodes, steps=steps)
        for arm in ("no_persistent", "episode_local", "persistent_v2", "oracle")
    }
    stale = _run_arm(
        "persistent_v2",
        encoders,
        episodes=episodes,
        steps=steps,
        reverse_episode=1,
    )
    controls = _control_report(encoders)
    frontend_digest_after = encoders.digest()
    bank_digest_after = sha256_file(bank_path) if bank_path.is_file() else None
    report = {
        "schema": PERSISTENT_IDENTITY_V2_EXPERIMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "steps": steps,
        "episodes": episodes,
        "artifact": PersistentCausalIdentityV2(
            minimum_similarity=IDENTITY_MINIMUM_SIMILARITY,
            recovery_episodes=RECOVERY_EPISODES,
        ).configuration(),
        "arms": arms,
        "persistent_advantage_over_episode_local": (
            arms["persistent_v2"]["integrated_return"]
            - arms["episode_local"]["integrated_return"]
        ),
        "stale_persistent_arm": stale,
        "controls": controls,
        "frontend_digest_before": frontend_digest_before,
        "frontend_digest_after": frontend_digest_after,
        "frontend_unchanged": frontend_digest_before == frontend_digest_after,
        "bank_digest_before": bank_digest_before,
        "bank_digest_after": bank_digest_after,
        "bank_unchanged": bank_digest_before == bank_digest_after,
        "unique_verifier_bits": int(steps * episodes * (len(arms) + 1)),
        "unique_logical_lifetimes": int(episodes * (len(arms) + 1)),
        "optimizer_updates": 0,
        "replayed_examples": int(steps * episodes * (len(arms) + 1)),
        "wall_seconds": time.perf_counter() - started,
        "claim_status": "development_closed_loop_diagnostic_not_promoted",
        "claim_boundary": (
            "The real amodal live tick contract, receipt-linked feedback, "
            "rendered learned events, and persistent causal identity state "
            "are exercised together. This is not a holdout or promotion: "
            "the marker fixture has fixed two-track binding within each "
            "episode, action decoding is deterministic, and no curated-bank "
            "or persistent self-model admission is claimed."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "persistent_identity_v2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "session_records"
        / "brainworkshop_persistent_identity_v2_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_persistent_identity_v2(
                arguments.output,
                seed=arguments.seed,
                steps=arguments.steps,
                episodes=arguments.episodes,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
