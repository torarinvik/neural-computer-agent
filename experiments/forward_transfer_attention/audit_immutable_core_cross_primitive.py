"""Cross-primitive transfer audit for the immutable identify-then-act core.

This is deliberately a frozen-core, reward-only horse race.  The immutable
core and matched controls see identical rendered streams, attempted opaque
actions, and scalar verifier outcomes.  Private rules are used only by the
discarded evaluator.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .train import seed_everything
from .train_cross_primitive_transfer import (
    SPATIAL_TEST_START,
    SPATIAL_TRAIN_START,
    spatial_policy_sequences,
)
from .train_identify_then_act import (
    ActionHistoryCore,
    evaluate,
    fit_readout,
    make_readout,
)
from .train_third_primitive_compounding import (
    THIRD_TEST_START,
    THIRD_TRAIN_START,
    same_different_sequences,
)
from .train_zero_label_predictive_state import (
    TEST_PALETTES,
    TRAIN_PALETTES,
)


PREFIXES = (16, 32, 48, 64, 96, 128)
NULL_ACTION = 2


@torch.no_grad()
def frozen_event_state(
        core: ActionHistoryCore, frames: torch.Tensor,
        *, batch_size: int, device: torch.device) -> torch.Tensor:
    """Extract the final recurrent state without exposing private task state."""
    core.eval()
    outputs = []
    for start in range(0, frames.shape[0], batch_size):
        batch = frames[start:start + batch_size].to(device)
        previous = torch.full(
            batch.shape[:2], NULL_ACTION, dtype=torch.long, device=device)
        outputs.append(core.states(
            batch, previous, passive=False)[:, -1].cpu())
    return torch.cat(outputs).to(device)


def _load_arms(
        checkpoint: Path, *, device: torch.device
        ) -> dict[str, ActionHistoryCore]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    seed = int(payload["source"]["core_initialization_seed"])
    seed_everything(seed)
    fresh = ActionHistoryCore(64).to(device)
    fresh_state = copy.deepcopy(fresh.state_dict())

    immutable = ActionHistoryCore(64).to(device)
    immutable.load_state_dict(payload["core"])

    vision_only = ActionHistoryCore(64).to(device)
    vision_only.load_state_dict(fresh_state)
    vision_only.vision.load_state_dict(immutable.vision.state_dict())

    recurrent_only = ActionHistoryCore(64).to(device)
    recurrent_only.load_state_dict(fresh_state)
    recurrent_only.action_embedding.load_state_dict(
        immutable.action_embedding.state_dict())
    recurrent_only.recurrent.load_state_dict(immutable.recurrent.state_dict())

    return {
        "immutable_core263": immutable,
        "vision_only_transfer": vision_only,
        "recurrent_only_transfer": recurrent_only,
        "matched_fresh263": fresh,
    }


def _logged_binary_feedback(
        rules: torch.Tensor, *, seed: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    actions = torch.randint(
        0, 2, (rules.shape[0],), generator=generator)
    rewards = (actions == rules).float()
    order = torch.randperm(
        rules.shape[0], generator=torch.Generator().manual_seed(seed + 1))
    return actions[order], rewards[order], order


def _fit_arm(
        initial: dict[str, torch.Tensor], features: torch.Tensor,
        actions: torch.Tensor, rewards: torch.Tensor, *,
        prefix: int, updates: int, batch_size: int, seed: int):
    return fit_readout(
        initial, features[:prefix], actions[:prefix], rewards[:prefix],
        readout_kind="bottleneck", intention_width=64,
        updates=updates, batch_size=batch_size,
        learning_rate=3e-3, seed=seed)


def _accuracy_and_predictions(
        model: torch.nn.Module, features: torch.Tensor,
        rules: torch.Tensor) -> tuple[float, torch.Tensor]:
    result = evaluate(model, features, rules)
    return float(result["verified_accuracy"]), result["predictions"]


def _task_data(task: str, train_count: int, test_count: int):
    if task == "spatial":
        train, train_rules = spatial_policy_sequences(
            SPATIAL_TRAIN_START + 4_000_000, train_count,
            heldout=False, palettes=TRAIN_PALETTES)
        normal, rules = spatial_policy_sequences(
            SPATIAL_TEST_START + 4_000_000, test_count,
            heldout=True, palettes=TEST_PALETTES)
        counterfactual, counterfactual_rules = spatial_policy_sequences(
            SPATIAL_TEST_START + 4_000_000, test_count,
            heldout=True, palettes=TEST_PALETTES, mirror=True)
        missing = {
            "missing_feedback": spatial_policy_sequences(
                SPATIAL_TEST_START + 4_000_000, test_count,
                heldout=True, palettes=TEST_PALETTES,
                omit_feedback=True),
        }
    elif task == "same_different":
        train, train_rules = same_different_sequences(
            THIRD_TRAIN_START + 4_000_000, train_count,
            heldout=False, palettes=TRAIN_PALETTES)
        normal, rules = same_different_sequences(
            THIRD_TEST_START + 4_000_000, test_count,
            heldout=True, palettes=TEST_PALETTES)
        counterfactual, counterfactual_rules = same_different_sequences(
            THIRD_TEST_START + 4_000_000, test_count,
            heldout=True, palettes=TEST_PALETTES, counterfactual=True)
        missing = {
            "missing_first": same_different_sequences(
                THIRD_TEST_START + 4_000_000, test_count,
                heldout=True, palettes=TEST_PALETTES, omit_first=True),
            "missing_second": same_different_sequences(
                THIRD_TEST_START + 4_000_000, test_count,
                heldout=True, palettes=TEST_PALETTES, omit_second=True),
        }
    else:
        raise ValueError(task)
    return {
        "train": (train, train_rules),
        "normal": (normal, rules),
        "counterfactual": (counterfactual, counterfactual_rules),
        **missing,
    }


def _passes_behavior(point: dict[str, object]) -> bool:
    return bool(
        float(point["normal_accuracy"]) >= 0.75
        and float(point["counterfactual_accuracy"]) >= 0.70
        and float(point["counterfactual_flip_rate"]) >= 0.60
        and all(float(value) <= 0.60 for value in point["missing"].values()))


def _passes(
        point: dict[str, object], *,
        control_ceiling: float = 0.60) -> bool:
    return bool(
        _passes_behavior(point)
        and float(point["action_complement_accuracy"]) <= control_ceiling
        and float(point["reward_complement_accuracy"]) <= control_ceiling)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=("spatial", "same_different"),
        default=("spatial", "same_different"))
    parser.add_argument("--train-lifetimes", type=int, default=120)
    parser.add_argument("--test-lifetimes", type=int, default=240)
    parser.add_argument("--fit-updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.train_lifetimes % 6 or args.test_lifetimes % 6:
        raise ValueError(
            "lifetime counts must be divisible by six for palette balance")
    started = time.perf_counter()
    device = torch.device(args.device)
    arms = _load_arms(args.checkpoint, device=device)
    initial_states = {
        name: {
            key: value.detach().clone()
            for key, value in core.state_dict().items()}
        for name, core in arms.items()
    }

    report_tasks = {}
    for task_index, task in enumerate(args.tasks):
        data = _task_data(
            task, args.train_lifetimes, args.test_lifetimes)
        features = {
            arm: {
                split: frozen_event_state(
                    core, frames, batch_size=args.batch_size, device=device)
                for split, (frames, _) in data.items()
            }
            for arm, core in arms.items()
        }
        actions, rewards, order = _logged_binary_feedback(
            data["train"][1], seed=args.seed + 100 * task_index)
        seed_everything(args.seed + 10_000 + task_index)
        template = make_readout(
            "bottleneck", hidden=64, intention_width=64).to(device)
        initial_head = copy.deepcopy(template.state_dict())
        ordered_features = {
            name: value["train"][order.to(device)]
            for name, value in features.items()
        }
        task_arms = {}
        max_prefix = min(args.train_lifetimes, max(PREFIXES))
        prefixes = [value for value in PREFIXES if value <= max_prefix]
        if not prefixes or prefixes[-1] != max_prefix:
            prefixes.append(max_prefix)
        for name, split_features in features.items():
            curve = []
            for prefix in prefixes:
                fit_seed = (
                    args.seed + 20_000 + task_index * 1_000 + prefix)
                model = _fit_arm(
                    initial_head, ordered_features[name], actions, rewards,
                    prefix=prefix, updates=args.fit_updates,
                    batch_size=args.batch_size, seed=fit_seed)
                action_control = reward_control = None
                if name == "immutable_core263":
                    action_control = _fit_arm(
                        initial_head, ordered_features[name], 1 - actions,
                        rewards, prefix=prefix, updates=args.fit_updates,
                        batch_size=args.batch_size, seed=fit_seed)
                    reward_control = _fit_arm(
                        initial_head, ordered_features[name], actions,
                        1.0 - rewards, prefix=prefix,
                        updates=args.fit_updates, batch_size=args.batch_size,
                        seed=fit_seed)
                normal_accuracy, normal_predictions = (
                    _accuracy_and_predictions(
                        model, split_features["normal"],
                        data["normal"][1]))
                counterfactual_accuracy, counterfactual_predictions = (
                    _accuracy_and_predictions(
                        model, split_features["counterfactual"],
                        data["counterfactual"][1]))
                missing = {
                    split: _accuracy_and_predictions(
                        model, split_features[split], data[split][1])[0]
                    for split in data
                    if split.startswith("missing")
                }
                point = {
                    "unique_reward_bits": prefix,
                    "optimizer_updates": args.fit_updates,
                    "examples_processed": (
                        args.fit_updates * min(args.batch_size, prefix)),
                    "normal_accuracy": normal_accuracy,
                    "counterfactual_accuracy": counterfactual_accuracy,
                    "counterfactual_flip_rate": float(
                        (normal_predictions !=
                         counterfactual_predictions).float().mean()),
                    "missing": missing,
                    "action_complement_accuracy": (
                        _accuracy_and_predictions(
                            action_control, split_features["normal"],
                            data["normal"][1])[0]
                        if action_control is not None else None),
                    "reward_complement_accuracy": (
                        _accuracy_and_predictions(
                            reward_control, split_features["normal"],
                            data["normal"][1])[0]
                        if reward_control is not None else None),
                }
                point["passes_behavior_gates"] = _passes_behavior(point)
                point["passes_all_gates"] = (
                    _passes(point)
                    if name == "immutable_core263"
                    else point["passes_behavior_gates"])
                curve.append(point)
            stable = next((
                point["unique_reward_bits"]
                for index, point in enumerate(curve)
                if all(later["passes_all_gates"]
                       for later in curve[index:])
            ), None)
            task_arms[name] = {
                "curve": curve,
                "stable_bits_to_all_gates": stable,
            }
            print(json.dumps({
                "task": task,
                "arm": name,
                "stable_bits_to_all_gates": stable,
                "final": curve[-1],
            }, sort_keys=True), flush=True)
        candidate = task_arms["immutable_core263"]
        fresh = task_arms["matched_fresh263"]
        candidate_bits = candidate["stable_bits_to_all_gates"]
        fresh_bits = fresh["stable_bits_to_all_gates"]
        report_tasks[task] = {
            "arms": task_arms,
            "transfer_gate": {
                "immutable_reaches_causal_mastery": candidate_bits is not None,
                "fewer_bits_than_matched_fresh": (
                    candidate_bits is not None and (
                        fresh_bits is None or candidate_bits < fresh_bits)),
            },
        }

    retention = {
        name: all(
            torch.equal(initial_states[name][key], value)
            for key, value in core.state_dict().items())
        for name, core in arms.items()
    }
    cross_primitive_wins = sum(
        int(value["transfer_gate"]["fewer_bits_than_matched_fresh"])
        for value in report_tasks.values())
    report = {
        "schema": "immutable-core-cross-primitive-audit-v1",
        "claim_boundary": (
            "Frozen representation transfer across rendered primitive "
            "families; this is not yet continual-learning retention."),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb_stream", "recurrent_state",
            "attempted_opaque_action", "scalar_verified_outcome"],
        "verifier_private": [
            "spatial_relation", "same_different_relation",
            "counterfactual_pairing", "missing-evidence_variant"],
        "configuration": vars(args) | {
            "checkpoint": str(args.checkpoint),
            "report": str(args.report),
        },
        "tasks": report_tasks,
        "frozen_core_parameters_bit_identical": retention,
        "summary": {
            "cross_primitive_wins_over_matched_fresh": cross_primitive_wins,
            "tested_primitive_families": len(report_tasks),
            "advance_to_replication": (
                cross_primitive_wins >= 1
                and all(retention.values())),
        },
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "summary": report["summary"],
        "retention": retention,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
