"""Test whether world-independent operators survive a change of dynamics.

The navigation records already show that a successor-feature library can pay
for a new goal in one world.  They do not show that the *way of learning and
using* that library is reusable in a different world.  This diagnostic keeps
those two claims apart.

Each world is a six-place ring with a different hidden ordering.  The event
stream exposes only the current place and the task target; the verifier emits
one binary arrival outcome per step.  A verified operator bundle carries no
place transitions or policies.  It contains only a control-flow contract:
probe unknown state/action cells, plan through known cells, and rebuild after
a contradiction.  The bundle is therefore reusable across worlds.

The controls are deliberately uncomfortable:

* ``fresh`` uses the same table and planner but has no reusable exploration
  operator;
* ``irrelevant`` inherits an unrelated random operator bundle;
* ``corrupted`` treats unknown cells as trustworthy self-loops;
* ``raw_successor`` carries the source world's successor features directly.

The last arm is expected to fail when the hidden ring changes.  No arm is
admitted to the curated bank by this experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .successor_features import (
    DEFAULT_DISCOUNT,
    generalised_policy_improvement,
    greedy_policy,
    successor_features,
)
from .world_model import WorldModel, plan_to

EXPERIMENT_ID = "brainworkshop-operator-world-transfer-2026-08-16"
OPERATOR_WORLD_TRANSFER_SCHEMA = "neural-computer.operator-world-transfer.v1"
OPERATOR_BUNDLE_SCHEMA = "neural-computer.verified-operator-bundle.v1"
OPERATOR_STAGER_SCHEMA = "neural-computer.verified-operator-stager.v1"
RAW_SUCCESSOR_SCHEMA = "neural-computer.raw-successor-artifact.v1"
DEVELOPMENT_SEED = 41
SOURCE_WORLD_SEED = 9000
TARGET_WORLD_SEED = 12000
WORLD_SEED_STRIDE = 37
PLACE_COUNT = 6
ACTION_COUNT = 3
EPISODE_STEPS = 16
TRAINING_EPISODES = 20
EVALUATION_EPISODES = 8
CHECKPOINT_STRIDE = 4
STABLE_THRESHOLD = 0.75
DISCOUNT = DEFAULT_DISCOUNT
ARM_NAMES = ("reusable", "fresh", "irrelevant", "corrupted", "raw_successor")


@dataclass(frozen=True)
class RingWorld:
    """A world with a stable protocol and a world-specific hidden ordering."""

    seed: int
    transitions: tuple[tuple[int, ...], ...]

    def task(self, *, start: int, goal: int):
        from .navigation_environment import NavigationTask

        return NavigationTask(
            transitions=self.transitions,
            goal=int(goal),
            start=int(start),
            place_count=PLACE_COUNT,
        ).validate()

    @property
    def digest(self) -> str:
        payload = json.dumps(self.transitions, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def sample_ring_world(seed: int) -> RingWorld:
    """Draw a connected ring; action 2 is the only universal hold action."""

    generator = torch.Generator().manual_seed(int(seed))
    order = [int(value) for value in torch.randperm(PLACE_COUNT, generator=generator)]
    forward = [0] * PLACE_COUNT
    backward = [0] * PLACE_COUNT
    for index, place in enumerate(order):
        forward[place] = order[(index + 1) % PLACE_COUNT]
        backward[place] = order[(index - 1) % PLACE_COUNT]
    world = RingWorld(
        seed=int(seed),
        transitions=(tuple(forward), tuple(backward), tuple(range(PLACE_COUNT))),
    )
    return world


@dataclass(frozen=True)
class VerifiedOperatorBundle:
    """A world-independent, versioned control-flow artifact."""

    schema: str = OPERATOR_BUNDLE_SCHEMA
    version: int = 1
    exploration: Literal["untried_first", "uniform"] = "untried_first"
    planning: Literal["safe_known_route", "unknown_self_loop"] = "safe_known_route"
    invalidation: Literal["rebuild_on_mismatch", "ignore_mismatch"] = (
        "rebuild_on_mismatch"
    )
    source_family: str = "ring-v1"
    verified_world_seed: int | None = None

    def validate(self) -> VerifiedOperatorBundle:
        if self.schema != OPERATOR_BUNDLE_SCHEMA or self.version != 1:
            raise ValueError("unsupported operator bundle")
        if self.source_family not in {"ring-v1", "unrelated-v0"}:
            raise ValueError("operator bundle is from an unrelated world family")
        if self.exploration not in {"untried_first", "uniform"}:
            raise ValueError("unknown exploration operator")
        if self.planning not in {"safe_known_route", "unknown_self_loop"}:
            raise ValueError("unknown planning operator")
        if self.invalidation not in {"rebuild_on_mismatch", "ignore_mismatch"}:
            raise ValueError("unknown invalidation operator")
        return self

    @property
    def digest(self) -> str:
        self.validate()
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {**self.__dict__, "digest": self.digest}


@dataclass(frozen=True)
class PlanningOperatorAdmission:
    """Receipt for a verifier-gated operator admission or rejection."""

    accepted: bool
    candidate_digest: str
    observations: int
    stable_prefix: int | None
    quarantined: bool
    reason: str
    schema: str = OPERATOR_STAGER_SCHEMA

    def validate(self) -> PlanningOperatorAdmission:
        if self.schema != OPERATOR_STAGER_SCHEMA:
            raise ValueError("unsupported operator admission schema")
        if len(self.candidate_digest) != 64:
            raise ValueError("operator admission digest is malformed")
        if self.observations < 0:
            raise ValueError("operator admission observation count is invalid")
        if self.stable_prefix is not None and not 0 <= self.stable_prefix < self.observations:
            raise ValueError("operator admission stable prefix is invalid")
        if not isinstance(self.quarantined, bool):
            raise TypeError("operator admission quarantine flag must be boolean")
        if not self.reason:
            raise ValueError("operator admission reason is missing")
        return self


class VerifiedPlanningOperatorStager:
    """Stage a reusable control-flow operator from scalar verifier evidence.

    The candidate is opaque to the stager.  Only the candidate digest and
    eligible scalar outcomes are retained.  Missing evidence is skipped,
    while a post-admission contradiction quarantines the candidate and freezes
    all later updates until a caller creates a fresh stager.
    """

    schema = OPERATOR_STAGER_SCHEMA

    def __init__(
        self,
        *,
        threshold: float = STABLE_THRESHOLD,
        min_observations: int = 2,
        min_stable_observations: int = 2,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("operator stager threshold must lie in [0, 1]")
        if min(min_observations, min_stable_observations) < 1:
            raise ValueError("operator stager observation gates must be positive")
        self.threshold = float(threshold)
        self.min_observations = int(min_observations)
        self.min_stable_observations = int(min_stable_observations)
        self._outcomes: dict[str, list[float]] = {}
        self._quarantined: set[str] = set()
        self._admitted: set[str] = set()

    def _digest(self, bundle: VerifiedOperatorBundle) -> str:
        return bundle.validate().digest

    def _stable_prefix(self, digest: str) -> int | None:
        outcomes = self._outcomes.get(digest, [])
        if len(outcomes) < self.min_observations:
            return None
        cumulative: list[float] = []
        total = 0.0
        for index, outcome in enumerate(outcomes):
            total += outcome
            cumulative.append(total / float(index + 1))
        for index, mean in enumerate(cumulative):
            if (
                mean >= self.threshold
                and len(cumulative) - index >= self.min_stable_observations
                and all(value >= self.threshold for value in cumulative[index:])
            ):
                return index
        return None

    def observe(
        self,
        bundle: VerifiedOperatorBundle,
        outcome: float,
        *,
        eligible: bool = True,
    ) -> None:
        digest = self._digest(bundle)
        if not isinstance(eligible, bool):
            raise TypeError("operator evidence eligibility must be boolean")
        if not isinstance(outcome, (int, float)) or not 0.0 <= float(outcome) <= 1.0:
            raise ValueError("operator verifier outcome must lie in [0, 1]")
        if digest in self._quarantined:
            return
        if not eligible:
            return
        value = float(outcome)
        if digest in self._admitted and value < self.threshold:
            self._quarantined.add(digest)
            return
        self._outcomes.setdefault(digest, []).append(value)

    def status(self, bundle: VerifiedOperatorBundle) -> PlanningOperatorAdmission:
        digest = self._digest(bundle)
        outcomes = self._outcomes.get(digest, [])
        stable = self._stable_prefix(digest)
        accepted = digest in self._admitted and digest not in self._quarantined
        reason = (
            "admitted"
            if accepted
            else "quarantined"
            if digest in self._quarantined
            else "stable-prefix-ready"
            if stable is not None
            else "insufficient-stable-evidence"
        )
        return PlanningOperatorAdmission(
            accepted=accepted,
            candidate_digest=digest,
            observations=len(outcomes),
            stable_prefix=stable,
            quarantined=digest in self._quarantined,
            reason=reason,
        ).validate()

    def admit_verified(
        self,
        bundle: VerifiedOperatorBundle,
        retention_probe: Callable[[VerifiedOperatorBundle], bool],
    ) -> PlanningOperatorAdmission:
        digest = self._digest(bundle)
        if not callable(retention_probe):
            raise TypeError("operator retention probe must be callable")
        status = self.status(bundle)
        if status.quarantined or status.stable_prefix is None:
            return PlanningOperatorAdmission(
                accepted=False,
                candidate_digest=digest,
                observations=status.observations,
                stable_prefix=status.stable_prefix,
                quarantined=status.quarantined,
                reason=status.reason,
            ).validate()
        if not bool(retention_probe(bundle)):
            return PlanningOperatorAdmission(
                accepted=False,
                candidate_digest=digest,
                observations=status.observations,
                stable_prefix=status.stable_prefix,
                quarantined=False,
                reason="retention-probe-rejected",
            ).validate()
        self._admitted.add(digest)
        return self.status(bundle)


@dataclass(frozen=True)
class RawSuccessorArtifact:
    """A source-world successor-feature artifact, intentionally not portable."""

    schema: str
    source_world_digest: str
    policies: tuple[tuple[int, ...], ...]
    psis: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]

    def validate(self) -> RawSuccessorArtifact:
        if self.schema != RAW_SUCCESSOR_SCHEMA:
            raise ValueError("unsupported successor artifact")
        if not self.policies or len(self.policies) != len(self.psis):
            raise ValueError("successor artifact is incomplete")
        return self

    @property
    def digest(self) -> str:
        self.validate()
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def verified_bundle(*, world_seed: int) -> VerifiedOperatorBundle:
    """Return the frozen bundle after a prior world has verified its contract."""

    return VerifiedOperatorBundle(verified_world_seed=int(world_seed)).validate()


def irrelevant_bundle() -> VerifiedOperatorBundle:
    """A syntactically valid artifact from a different operator family."""

    return VerifiedOperatorBundle(
        exploration="uniform",
        planning="safe_known_route",
        invalidation="ignore_mismatch",
        source_family="unrelated-v0",
        verified_world_seed=None,
    ).validate()


def corrupted_bundle(*, world_seed: int) -> VerifiedOperatorBundle:
    return VerifiedOperatorBundle(
        exploration="untried_first",
        planning="unknown_self_loop",
        invalidation="ignore_mismatch",
        verified_world_seed=int(world_seed),
    ).validate()


def _full_model(world: RingWorld) -> WorldModel:
    model = WorldModel(PLACE_COUNT, ACTION_COUNT)
    for action, row in enumerate(world.transitions):
        for place, following in enumerate(row):
            model.observe(place, action, following, 0)
    return model


def build_raw_successor_artifact(world: RingWorld) -> RawSuccessorArtifact:
    """Materialise source-world policies and occupancies for the negative arm."""

    model = _full_model(world)
    policies = []
    psis = []
    for goal in range(PLACE_COUNT):
        weights = torch.zeros(PLACE_COUNT, dtype=torch.float64)
        weights[goal] = 1.0
        policy = greedy_policy(model, weights, discount=DISCOUNT)
        psi = successor_features(model, policy, discount=DISCOUNT)
        policies.append(tuple(int(action) for action in policy))
        psis.append(tuple(tuple(tuple(float(value) for value in cell) for cell in row) for row in psi))
    return RawSuccessorArtifact(
        schema=RAW_SUCCESSOR_SCHEMA,
        source_world_digest=world.digest,
        policies=tuple(policies),
        psis=tuple(psis),
    ).validate()


def _random_tasks(world: RingWorld, *, seed: int, count: int):
    generator = torch.Generator().manual_seed(int(seed))
    tasks = []
    for _ in range(int(count)):
        start = int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item())
        goal = int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item())
        if goal == start:
            goal = (goal + 1) % PLACE_COUNT
        tasks.append(world.task(start=start, goal=goal))
    return tuple(tasks)


def _unknown_actions(model: WorldModel, place: int) -> tuple[int, ...]:
    return tuple(
        action
        for action in range(model.action_count)
        if not model.counts[action][int(place)]
    )


def observe_transition(
    model: WorldModel,
    place: int,
    action: int,
    following: int,
    reward: int,
    *,
    bundle: VerifiedOperatorBundle | None,
) -> None:
    """Apply a generic update operator, with optional contradiction reset."""

    expected = model.successor(place, action)
    if (
        expected is not None
        and int(expected) != int(following)
        and bundle is not None
        and bundle.invalidation == "rebuild_on_mismatch"
    ):
        model.counts = [
            [{} for _ in range(model.place_count)]
            for _ in range(model.action_count)
        ]
        model.rewarded.clear()
        model.visited.clear()
    model.observe(place, action, following, reward)


def _raw_action(artifact: RawSuccessorArtifact, *, place: int, goal: int) -> int:
    psis = [torch.tensor(psi, dtype=torch.float64) for psi in artifact.psis]
    weights = torch.zeros(PLACE_COUNT, dtype=torch.float64)
    weights[int(goal)] = 1.0
    return generalised_policy_improvement(psis, int(place), weights)


def _choose_action(
    model: WorldModel,
    *,
    place: int,
    goal: int,
    mode: str,
    bundle: VerifiedOperatorBundle | None,
    artifact: RawSuccessorArtifact | None,
    generator: torch.Generator,
) -> tuple[int, float]:
    """Choose an action and return decision latency in milliseconds."""

    started = time.perf_counter_ns()
    if mode == "raw_successor":
        if artifact is None:
            raise ValueError("raw successor arm needs an artifact")
        action = _raw_action(artifact, place=place, goal=goal)
    elif mode == "reusable" and bundle is not None and bundle.exploration == "untried_first":
        unknown = _unknown_actions(model, place)
        if int(place) == int(goal):
            holding = model.holding_action(place)
            action = 2 if holding is None else int(holding)
        elif unknown:
            action = int(unknown[0])
        else:
            route = plan_to(model, place, (goal,))
            action = (
                int(route.actions[0])
                if route is not None and route.actions
                else int(torch.randint(ACTION_COUNT, (1,), generator=generator).item())
            )
    elif bundle is not None and bundle.planning == "unknown_self_loop":
        # Corrupted artifact: it treats an untried cell as a trusted self-loop
        # and therefore keeps pressing the apparent hold action instead of
        # paying the evidence needed to discover the world.
        if _unknown_actions(model, place):
            action = 2
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            return action, elapsed
        weights = torch.zeros(PLACE_COUNT, dtype=torch.float64)
        weights[int(goal)] = 1.0
        action = int(greedy_policy(model, weights, discount=DISCOUNT)[int(place)])
    else:
        if int(place) == int(goal):
            holding = model.holding_action(place)
            action = 2 if holding is None else int(holding)
        else:
            route = plan_to(model, place, (goal,))
            action = (
                int(route.actions[0])
                if route is not None and route.actions
                else int(torch.randint(ACTION_COUNT, (1,), generator=generator).item())
            )
    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
    return action, elapsed


def _episode(
    world: RingWorld,
    model: WorldModel,
    task,
    *,
    mode: str,
    bundle: VerifiedOperatorBundle | None,
    artifact: RawSuccessorArtifact | None,
    seed: int,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(int(seed))
    place = int(task.start)
    total = 0.0
    latency = 0.0
    for _ in range(EPISODE_STEPS):
        action, elapsed = _choose_action(
            model,
            place=place,
            goal=int(task.goal),
            mode=mode,
            bundle=bundle,
            artifact=artifact,
            generator=generator,
        )
        following = int(world.transitions[action][place])
        reward = int(following == int(task.goal))
        observe_transition(
            model,
            place,
            action,
            following,
            reward,
            bundle=bundle,
        )
        place = following
        total += reward
        latency += elapsed
    return total / EPISODE_STEPS, latency / EPISODE_STEPS


def _evaluate(
    world: RingWorld,
    model: WorldModel,
    tasks,
    *,
    mode: str,
    bundle: VerifiedOperatorBundle | None,
    artifact: RawSuccessorArtifact | None,
    seed: int,
) -> tuple[float, float]:
    scores = []
    latency = []
    for index, task in enumerate(tasks):
        evaluation_model = copy.deepcopy(model)
        score, elapsed = _episode(
            world,
            evaluation_model,
            task,
            mode=mode,
            bundle=bundle,
            artifact=artifact,
            seed=seed + index,
        )
        optimal = task.optimal_return(EPISODE_STEPS)
        scores.append(score / optimal if optimal > 0.0 else 0.0)
        latency.append(elapsed)
    return sum(scores) / len(scores), sum(latency) / len(latency)


def _stable_bits(curve: list[dict[str, float | int]]) -> int | None:
    for index, row in enumerate(curve):
        if all(float(later["normalized_return"]) >= STABLE_THRESHOLD for later in curve[index:]):
            return int(row["unique_verifier_bits"])
    return None


def run_arm(
    world: RingWorld,
    *,
    mode: str,
    bundle: VerifiedOperatorBundle | None,
    artifact: RawSuccessorArtifact | None,
    seed: int,
    training_episodes: int = TRAINING_EPISODES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    if mode not in ARM_NAMES:
        raise ValueError(f"unknown arm: {mode}")
    started = time.perf_counter()
    model = WorldModel(PLACE_COUNT, ACTION_COUNT)
    checkpoints = tuple(range(CHECKPOINT_STRIDE, training_episodes + 1, CHECKPOINT_STRIDE))
    curve = []
    latency = []
    train_tasks = _random_tasks(world, seed=seed + 10, count=training_episodes)
    verifier_bits = 0
    for prefix in range(1, training_episodes + 1):
        _episode(
            world,
            model,
            train_tasks[prefix - 1],
            mode=mode,
            bundle=bundle,
            artifact=artifact,
            seed=seed + prefix,
        )
        verifier_bits += EPISODE_STEPS
        if prefix not in checkpoints:
            continue
        evaluation_tasks = _random_tasks(
            world,
            seed=seed + 100_000 + prefix * 1_000,
            count=evaluation_episodes,
        )
        score, decision_latency = _evaluate(
            world,
            model,
            evaluation_tasks,
            mode=mode,
            bundle=bundle,
            artifact=artifact,
            seed=seed + 200_000 + prefix * 1_000,
        )
        verifier_bits += evaluation_episodes * EPISODE_STEPS
        latency.append(decision_latency)
        curve.append(
            {
                "training_episodes": prefix,
                "unique_verifier_bits": verifier_bits,
                "normalized_return": score,
                "coverage": model.coverage,
            }
        )
    result = {
        "arm": mode,
        "bundle_digest": None if bundle is None else bundle.digest,
        "artifact_digest": None if artifact is None else artifact.digest,
        "final_model_coverage": model.coverage,
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve),
        "unique_verifier_bits": verifier_bits,
        "unique_logical_lifetimes": training_episodes + len(checkpoints) * evaluation_episodes,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
        "latency_ms_per_decision": sum(latency) / len(latency) if latency else 0.0,
        "retention_on_mastered_primitive": "not_claimed",
    }
    return result


def run_transfer(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 3,
    training_episodes: int = TRAINING_EPISODES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    """Run matched cross-world arms without touching any curated artifact."""

    started = time.perf_counter()
    rows = []
    for replicate in range(int(replicates)):
        source = sample_ring_world(SOURCE_WORLD_SEED + WORLD_SEED_STRIDE * replicate)
        target = sample_ring_world(TARGET_WORLD_SEED + WORLD_SEED_STRIDE * replicate)
        if source.transitions == target.transitions:
            raise AssertionError("source and target worlds must differ")
        candidate_bundle = verified_bundle(world_seed=source.seed)
        source_probe = run_arm(
            source,
            mode="reusable",
            bundle=candidate_bundle,
            artifact=None,
            seed=seed + 1_000 * replicate,
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes,
        )
        stager = VerifiedPlanningOperatorStager(
            threshold=STABLE_THRESHOLD,
            min_observations=2,
            min_stable_observations=2,
        )
        for checkpoint in source_probe["curve"]:
            stager.observe(candidate_bundle, checkpoint["normalized_return"])
        admission = stager.admit_verified(
            candidate_bundle,
            lambda retained, digest=candidate_bundle.digest: retained.digest == digest,
        )
        if not admission.accepted:
            raise RuntimeError("source-world operator failed its stable-prefix gate")
        bundle = candidate_bundle
        raw = build_raw_successor_artifact(source)
        arms = {
            "reusable": run_arm(
                target,
                mode="reusable",
                bundle=bundle,
                artifact=None,
                seed=seed + 1_000 * replicate,
                training_episodes=training_episodes,
                evaluation_episodes=evaluation_episodes,
            ),
            "fresh": run_arm(
                target,
                mode="fresh",
                bundle=None,
                artifact=None,
                seed=seed + 1_000 * replicate,
                training_episodes=training_episodes,
                evaluation_episodes=evaluation_episodes,
            ),
            "irrelevant": run_arm(
                target,
                mode="irrelevant",
                bundle=irrelevant_bundle(),
                artifact=None,
                seed=seed + 1_000 * replicate,
                training_episodes=training_episodes,
                evaluation_episodes=evaluation_episodes,
            ),
            "corrupted": run_arm(
                target,
                mode="corrupted",
                bundle=corrupted_bundle(world_seed=source.seed),
                artifact=None,
                seed=seed + 1_000 * replicate,
                training_episodes=training_episodes,
                evaluation_episodes=evaluation_episodes,
            ),
            "raw_successor": run_arm(
                target,
                mode="raw_successor",
                bundle=None,
                artifact=raw,
                seed=seed + 1_000 * replicate,
                training_episodes=training_episodes,
                evaluation_episodes=evaluation_episodes,
            ),
        }
        rows.append(
            {
                "replicate": replicate,
                "source_world_seed": source.seed,
                "target_world_seed": target.seed,
                "source_world_digest": source.digest,
                "target_world_digest": target.digest,
                "operator_bundle": bundle.payload(),
                "operator_admission": admission.__dict__,
                "operator_source_probe": source_probe,
                "raw_successor_digest": raw.digest,
                "arms": arms,
            }
        )
    reusable_bits = [row["arms"]["reusable"]["stable_bits_to_threshold"] for row in rows]
    fresh_bits = [row["arms"]["fresh"]["stable_bits_to_threshold"] for row in rows]
    ratios = [
        float(reusable) / float(fresh)
        for reusable, fresh in zip(reusable_bits, fresh_bits)
        if reusable is not None and fresh is not None and fresh > 0
    ]
    accounting = {
        arm: {
            "unique_verifier_bits": sum(
                int(row["arms"][arm]["unique_verifier_bits"]) for row in rows
            ),
            "unique_logical_lifetimes": sum(
                int(row["arms"][arm]["unique_logical_lifetimes"]) for row in rows
            ),
            "optimizer_updates": sum(
                int(row["arms"][arm]["optimizer_updates"]) for row in rows
            ),
            "replayed_examples": sum(
                int(row["arms"][arm]["replayed_examples"]) for row in rows
            ),
            "wall_seconds": sum(
                float(row["arms"][arm]["wall_seconds"]) for row in rows
            ),
            "latency_ms_per_decision": sum(
                float(row["arms"][arm]["latency_ms_per_decision"]) for row in rows
            )
            / len(rows),
            "stable_bits_to_threshold": [
                row["arms"][arm]["stable_bits_to_threshold"] for row in rows
            ],
            "retention_on_mastered_primitive": "not_claimed",
        }
        for arm in ARM_NAMES
    }
    report = {
        "schema": OPERATOR_WORLD_TRANSFER_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "operator_bundle_is_world_independent": True,
        "raw_successor_expected_to_transfer": False,
        "transfer_ratio_against_fresh_learner": sum(ratios) / len(ratios) if ratios else None,
        "accounting": accounting,
        "stable_threshold": STABLE_THRESHOLD,
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "operator_world_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_operator_world_transfer_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--training-episodes", type=int, default=TRAINING_EPISODES)
    parser.add_argument("--evaluation-episodes", type=int, default=EVALUATION_EPISODES)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_transfer(
                arguments.output,
                seed=arguments.seed,
                replicates=arguments.replicates,
                training_episodes=arguments.training_episodes,
                evaluation_episodes=arguments.evaluation_episodes,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
