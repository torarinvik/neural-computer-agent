"""Verifier-side objective for sample-efficient continual acquisition.

The controller never receives the private labels used here.  A training
campaign or population selector may use this score to choose checkpoints:

* positive gain on the new skill is rewarded;
* regressions on old skills are penalized;
* a replay-saving bonus is paid only when retention stays inside the gate.

Keeping this calculation outside the controller prevents it from becoming a
privileged shortcut or a reward-hacking channel.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ContinualObjective:
    """Auditable components of the continual-learning objective."""

    new_gain: float
    parent_gain: float
    causal_gain: float | None
    old_retention_deltas: tuple[float, ...]
    forgetting_penalty: float
    retained_old_mean: float
    replay_savings: float
    replay_efficiency_bonus: float
    score: float
    acquisition_gate_passed: bool
    retention_gate_passed: bool


def score_continual_acquisition(
        *,
        new_parent: float,
        new_child: float,
        new_causal_baseline: float | None = None,
        old_parent: Sequence[float],
        old_child: Sequence[float],
        replay_lifetimes: int,
        reference_replay_lifetimes: int,
        retention_tolerance: float = 0.02,
        new_weight: float = 1.0,
        forgetting_weight: float = 2.0,
        replay_weight: float = 0.25,
        minimum_new_gain: float = 0.0,
        ) -> ContinualObjective:
    """Compute a deterministic selection score from verifier measurements.

    ``reference_replay_lifetimes`` is a pre-registered comparison budget.  A
    checkpoint receives no replay bonus if it forgets beyond
    ``retention_tolerance``; reducing replay cannot compensate for forgetting.
    ``old_parent`` and ``old_child`` must be aligned by primitive.
    """
    if len(old_parent) != len(old_child) or not old_parent:
        raise ValueError("old-skill parent/child lists must be non-empty and aligned")
    if replay_lifetimes < 0 or reference_replay_lifetimes <= 0:
        raise ValueError("replay budgets must be non-negative and reference positive")
    if retention_tolerance < 0:
        raise ValueError("retention tolerance must be non-negative")
    if minimum_new_gain < 0:
        raise ValueError("minimum new gain must be non-negative")
    if new_weight < 0 or forgetting_weight < 0 or replay_weight < 0:
        raise ValueError("objective weights must be non-negative")

    deltas = tuple(float(child - parent)
                   for parent, child in zip(old_parent, old_child))
    forgetting = sum(max(0.0, -delta) for delta in deltas)
    gate_passed = all(delta >= -retention_tolerance for delta in deltas)
    retained_old = sum(float(value) for value in old_child) / len(old_child)
    replay_savings = max(
        0.0, 1.0 - float(replay_lifetimes) / reference_replay_lifetimes)
    parent_gain = float(new_child - new_parent)
    causal_gain = (
        None if new_causal_baseline is None
        else float(new_child - new_causal_baseline))
    # When a causal ablation is available, reward only the improvement that is
    # supported by both comparisons.  This prevents unrelated parameter drift
    # from masquerading as acquisition by the newly written skill.
    new_gain = (
        parent_gain if causal_gain is None
        else min(parent_gain, causal_gain))
    acquisition_gate_passed = new_gain > minimum_new_gain
    # Replay is useful only in service of acquiring something new.  Without
    # this gate, a do-nothing checkpoint could score well by preserving the
    # parent perfectly while claiming a zero-replay budget.
    replay_bonus = (
        retained_old * replay_savings
        if gate_passed and acquisition_gate_passed else 0.0)
    score = (
        new_weight * new_gain
        - forgetting_weight * forgetting
        + replay_weight * replay_bonus)
    return ContinualObjective(
        new_gain=new_gain,
        parent_gain=parent_gain,
        causal_gain=causal_gain,
        old_retention_deltas=deltas,
        forgetting_penalty=forgetting,
        retained_old_mean=retained_old,
        replay_savings=replay_savings,
        replay_efficiency_bonus=replay_bonus,
        score=score,
        acquisition_gate_passed=acquisition_gate_passed,
        retention_gate_passed=gate_passed,
    )


def _parse_old_pair(value: str) -> tuple[float, float]:
    try:
        name, parent, child = value.split(":", 2)
        del name  # The name is for the human-readable CLI input only.
        return float(parent), float(child)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "old skill must be NAME:PARENT_ACCURACY:CHILD_ACCURACY") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-parent", type=float, required=True)
    parser.add_argument("--new-child", type=float, required=True)
    parser.add_argument("--new-causal-baseline", type=float)
    parser.add_argument(
        "--old", action="append", type=_parse_old_pair, required=True,
        metavar="NAME:PARENT:CHILD",
        help="one verifier-measured old skill; repeat for every protected skill")
    parser.add_argument("--replay-lifetimes", type=int, required=True)
    parser.add_argument("--reference-replay-lifetimes", type=int, required=True)
    parser.add_argument("--retention-tolerance", type=float, default=0.02)
    parser.add_argument("--new-weight", type=float, default=1.0)
    parser.add_argument("--forgetting-weight", type=float, default=2.0)
    parser.add_argument("--replay-weight", type=float, default=0.25)
    parser.add_argument("--minimum-new-gain", type=float, default=0.0)
    args = parser.parse_args()
    result = score_continual_acquisition(
        new_parent=args.new_parent, new_child=args.new_child,
        new_causal_baseline=args.new_causal_baseline,
        old_parent=[pair[0] for pair in args.old],
        old_child=[pair[1] for pair in args.old],
        replay_lifetimes=args.replay_lifetimes,
        reference_replay_lifetimes=args.reference_replay_lifetimes,
        retention_tolerance=args.retention_tolerance,
        new_weight=args.new_weight,
        forgetting_weight=args.forgetting_weight,
        replay_weight=args.replay_weight,
        minimum_new_gain=args.minimum_new_gain,
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
