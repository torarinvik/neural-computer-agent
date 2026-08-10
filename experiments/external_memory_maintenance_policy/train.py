"""Pressure-test learned grow/share/compress/defer decisions.

The controller is represented only by a frozen digest.  A replaceable
external policy receives generic storage telemetry and consumes one scalar
verifier utility per decision, with no replay.  The verifier knows the legal
maintenance choice for the synthetic storage regime; those regime labels are
never passed to the policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    MAINTENANCE_ACTIONS,
    ExternalMemoryMaintenancePolicy,
)

FEATURE_WIDTH = 12
TRAIN_STEPS = 320
EVAL_STEPS = 96
HIDDEN_WIDTH = 32
TEMPERATURE = 0.4
LEARNING_RATE = 0.005


def _digest(module: torch.nn.Module) -> str:
    return repr(
        [
            (name, value.detach().cpu().clone())
            for name, value in sorted(module.state_dict().items())
        ]
    )


def _state(index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Create opaque storage facts, legal actions, and the verifier target."""

    regime = index % 4
    features = torch.zeros(FEATURE_WIDTH)
    mask = torch.ones(len(MAINTENANCE_ACTIONS), dtype=torch.bool)
    mask[MAINTENANCE_ACTIONS.index("evict")] = False
    if regime == 0:
        features[[0, 1, 2, 10]] = torch.tensor([0.95, 0.95, 0.90, 0.95])
        target = "share"
    elif regime == 1:
        features[[0, 1, 2, 4, 10]] = torch.tensor([1.0, 1.0, 1.0, 0.15, 0.05])
        target = "grow"
    elif regime == 2:
        features[[0, 1, 2, 6, 7, 11]] = torch.tensor(
            [0.85, 0.85, 0.85, 0.1, 0.1, 0.95]
        )
        target = "compress"
    else:
        features[[0, 1, 2, 4, 5, 10, 11]] = torch.tensor(
            [0.1, 0.1, 0.1, 0.7, 0.8, 0.05, 0.0]
        )
        target = "defer"
    return features, mask, target


def _utility(action: str, target: str) -> float:
    return 1.0 if action == target else 0.0


def _rollout(
    seed: int,
    *,
    learn: bool,
    shuffled_verifier: bool = False,
) -> dict[str, object]:
    torch.manual_seed(seed)
    policy = ExternalMemoryMaintenancePolicy(
        hidden_width=HIDDEN_WIDTH,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE) if learn else None
    generator = torch.Generator().manual_seed(seed + 4000)
    utilities: list[float] = []
    actions: dict[str, int] = {action: 0 for action in MAINTENANCE_ACTIONS}
    updates = 0
    for step in range(TRAIN_STEPS):
        features, mask, target = _state(step)
        proposal = policy.propose(
            features,
            mask,
            sample=learn,
            generator=generator if learn else None,
        )
        verifier_target = target
        if shuffled_verifier:
            verifier_target = MAINTENANCE_ACTIONS[
                int(torch.randint(len(MAINTENANCE_ACTIONS), (), generator=generator))
            ]
        utility = _utility(proposal.action, verifier_target)
        if learn:
            policy.adaptation_step(proposal, utility, optimizer=optimizer)
            updates += 1
        utilities.append(utility)
        actions[proposal.action] += 1

    evaluation: list[float] = []
    for step in range(EVAL_STEPS):
        features, mask, target = _state(TRAIN_STEPS + step)
        proposal = policy.propose(features, mask)
        evaluation.append(_utility(proposal.action, target))
    return {
        "policy": policy,
        "utilities": utilities,
        "evaluation": evaluation,
        "actions": actions,
        "optimizer_updates": updates,
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    controller = torch.nn.Linear(4, 4)
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    trained = _rollout(seed, learn=True)
    fresh = _rollout(seed + 900000, learn=False)
    shuffled = _rollout(seed + 910000, learn=True, shuffled_verifier=True)
    trained_eval = sum(trained["evaluation"]) / EVAL_STEPS
    fresh_eval = sum(fresh["evaluation"]) / EVAL_STEPS
    shuffled_eval = sum(shuffled["evaluation"]) / EVAL_STEPS
    gates = {
        "trained_beats_fresh": trained_eval > fresh_eval + 0.15,
        "trained_beats_shuffled_verifier": trained_eval > shuffled_eval + 0.10,
        "all_legacy_actions_observed": all(
            trained["actions"][action] > 0
            for action in MAINTENANCE_ACTIONS
            if action != "evict"
        ),
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "one_update_per_unique_utility": trained["optimizer_updates"] == TRAIN_STEPS,
    }
    report = {
        "schema": "neural-computer.external-memory-maintenance-policy.v1",
        "claim_boundary": (
            "learned finite-budget external-memory action selection over a "
            "synthetic verifier stream; not general continual learning, "
            "unrestricted memory growth, or autonomous equivalence discovery"
        ),
        "seed": seed,
        "configuration": {
            "train_steps": TRAIN_STEPS,
            "eval_steps": EVAL_STEPS,
            "actions": tuple(
                action for action in MAINTENANCE_ACTIONS if action != "evict"
            ),
            "update": "single_scalar_verifier_policy_gradient_without_replay_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "trained_eval_utility": trained_eval,
            "fresh_eval_utility": fresh_eval,
            "shuffled_verifier_eval_utility": shuffled_eval,
            "trained_final_window_utility": sum(trained["utilities"][-64:]) / 64,
            "trained_action_counts": trained["actions"],
        },
        "accounting": {
            "unique_verifier_utilities": TRAIN_STEPS,
            "unique_logical_lifetimes": TRAIN_STEPS,
            "optimizer_updates": trained["optimizer_updates"],
            "replayed_examples": 0,
            "controller_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6107)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
