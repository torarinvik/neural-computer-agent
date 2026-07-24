"""Tiny near-transfer ladder around the identify-then-act world.

The learner receives only the rendered stream, its own prior action, attempted
opaque answer, and scalar reward.  Private labels below exist solely in the
deterministic verifier and discarded evaluator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from .audit_immutable_core_cross_primitive import _load_arms
from .train import seed_everything
from .train_identify_then_act import (
    APPEARANCE_STYLES,
    PROTOCOLS,
    RELATION_AXES,
    TEST_START,
    TRAIN_START,
    decision_features,
    evaluate,
    fit_readout,
    identify_batch,
    make_readout,
)


TRAIN_OFFSET = 44_000_000
TEST_OFFSET = 46_000_000
PREFIXES = (8, 16, 24, 32, 40, 48, 64)


def private_label(
        data: dict[str, torch.Tensor], task: str) -> torch.Tensor:
    protocols = data["private_protocol_ids"].tolist()
    probes = data["probe_actions"].tolist()
    correct = data["correct_actions"].tolist()
    effect_right = torch.tensor([
        int(PROTOCOLS[protocol][probe] == 1)
        for protocol, probe in zip(protocols, probes)
    ])
    target_right = torch.tensor([
        int(PROTOCOLS[protocol][answer] == 1)
        for protocol, answer in zip(protocols, correct)
    ])
    if task == "target_side":
        return target_right
    if task == "effect_side":
        return effect_right
    if task == "effect_target_match":
        return (effect_right == target_right).long()
    raise ValueError(task)


def _counterfactual_kwargs(task: str) -> dict[str, bool]:
    if task == "target_side":
        return {"reverse_target": True}
    return {"swap_protocol": True}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_indices(
        labels: torch.Tensor, count: int, *, seed: int) -> torch.Tensor:
    if count % 2:
        raise ValueError("binary task count must be even")
    selected = []
    for value in (0, 1):
        candidates = torch.where(labels == value)[0]
        if candidates.numel() < count // 2:
            raise ValueError("candidate pool is too small for exact balance")
        order = torch.randperm(
            candidates.numel(),
            generator=torch.Generator().manual_seed(seed + value))
        selected.append(candidates[order[:count // 2]])
    combined = torch.cat(selected)
    return combined[torch.randperm(
        combined.numel(),
        generator=torch.Generator().manual_seed(seed + 2))]


def _subset_data(
        data: dict[str, torch.Tensor], indices: torch.Tensor
        ) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in data.items()}


def _passes_behavior(point: dict[str, object]) -> bool:
    return bool(
        float(point["normal_accuracy"]) >= 0.75
        and float(point["counterfactual_accuracy"]) >= 0.70
        and float(point["counterfactual_flip_rate"]) >= 0.60
        and all(float(value) <= 0.60 for value in point["missing"].values()))


@torch.no_grad()
def transfer_features(
        core: torch.nn.Module, data: dict[str, torch.Tensor], *,
        interface: str, device: torch.device) -> torch.Tensor:
    """Expose generic event snapshots without attaching semantic slot labels."""
    if interface not in (
            "decision", "event_vision", "decision_event_vision"):
        raise ValueError(interface)
    parts = []
    if interface in ("decision", "decision_event_vision"):
        parts.append(decision_features(
            core, data, passive=False, device=device))
    if interface in ("event_vision", "decision_event_vision"):
        frames = data["frames"][:, :3].to(device)
        parts.append(core.vision(frames.flatten(0, 1)).reshape(
            frames.shape[0], -1))
    return torch.cat(parts, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("target_side", "effect_side", "effect_target_match"),
        required=True)
    parser.add_argument("--train-lifetimes", type=int, default=64)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=941)
    parser.add_argument(
        "--data-offset", type=int, default=0,
        help="Disjoint logical/render stream offset for blind replication.")
    parser.add_argument(
        "--appearance-style", choices=APPEARANCE_STYLES,
        default="baseline",
        help=(
            "Change only public pixels while preserving the event protocol, "
            "opaque actions, scalar outcomes, and private verifier logic."))
    parser.add_argument(
        "--relation-axis", choices=RELATION_AXES, default="position",
        help=(
            "Render the controlled effect and target as either spatial "
            "position or color identity while preserving event structure."))
    parser.add_argument(
        "--feature-interface",
        choices=("decision", "event_vision", "decision_event_vision"),
        default="decision")
    parser.add_argument(
        "--readout-kind",
        choices=("bottleneck", "direct", "antisymmetric"),
        default="bottleneck")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    train_pool = identify_batch(
        TRAIN_START + TRAIN_OFFSET + args.data_offset,
        args.train_lifetimes * 2,
        heldout=False, appearance_style=args.appearance_style,
        relation_axis=args.relation_axis)
    train_indices = _balanced_indices(
        private_label(train_pool, args.task), args.train_lifetimes,
        seed=args.seed + 10)
    train = _subset_data(train_pool, train_indices)
    normal_pool = identify_batch(
        TEST_START + TEST_OFFSET + args.data_offset,
        args.test_lifetimes * 2, heldout=True,
        appearance_style=args.appearance_style,
        relation_axis=args.relation_axis)
    test_indices = _balanced_indices(
        private_label(normal_pool, args.task), args.test_lifetimes,
        seed=args.seed + 20)
    normal = _subset_data(normal_pool, test_indices)
    counterfactual = _subset_data(identify_batch(
        TEST_START + TEST_OFFSET + args.data_offset,
        args.test_lifetimes * 2, heldout=True,
        appearance_style=args.appearance_style,
        relation_axis=args.relation_axis,
        **_counterfactual_kwargs(args.task)), test_indices)
    datasets = {"train": train, "normal": normal,
                "counterfactual": counterfactual}
    if args.task != "target_side":
        datasets["missing_consequence"] = _subset_data(identify_batch(
            TEST_START + TEST_OFFSET + args.data_offset,
            args.test_lifetimes * 2,
            heldout=True, missing_consequence=True,
            appearance_style=args.appearance_style,
            relation_axis=args.relation_axis), test_indices)
        datasets["no_probe_effect"] = _subset_data(identify_batch(
            TEST_START + TEST_OFFSET + args.data_offset,
            args.test_lifetimes * 2,
            heldout=True, no_probe_effect=True,
            appearance_style=args.appearance_style,
            relation_axis=args.relation_axis), test_indices)
    labels = {
        name: private_label(data, args.task)
        for name, data in datasets.items()
    }
    arms = _load_arms(args.checkpoint, device=device)
    initial_states = {
        name: {key: value.detach().clone()
               for key, value in core.state_dict().items()}
        for name, core in arms.items()
    }
    features = {
        arm: {
            split: transfer_features(
                core, data, interface=args.feature_interface, device=device)
            for split, data in datasets.items()
        }
        for arm, core in arms.items()
    }
    attempted = torch.randint(
        0, 2, (args.train_lifetimes,),
        generator=torch.Generator().manual_seed(args.seed))
    rewards = (attempted == labels["train"]).float()
    order = torch.randperm(
        args.train_lifetimes,
        generator=torch.Generator().manual_seed(args.seed + 1))
    attempted = attempted[order]
    rewards = rewards[order]
    seed_everything(args.seed + 2)
    feature_width = next(iter(features.values()))["train"].shape[-1]
    template = make_readout(
        args.readout_kind, hidden=feature_width,
        intention_width=64).to(device)
    initial_head = copy.deepcopy(template.state_dict())
    prefixes = [value for value in PREFIXES
                if value <= args.train_lifetimes]
    if not prefixes or prefixes[-1] != args.train_lifetimes:
        prefixes.append(args.train_lifetimes)

    results = {}
    for name, split_features in features.items():
        ordered = split_features["train"][order.to(device)]
        curve = []
        for prefix in prefixes:
            fit_seed = args.seed + 100 + prefix
            model = fit_readout(
                initial_head, ordered[:prefix],
                attempted[:prefix], rewards[:prefix],
                readout_kind=args.readout_kind, intention_width=64,
                updates=args.fit_updates, batch_size=args.batch_size,
                learning_rate=3e-3, seed=fit_seed)
            action_control = reward_control = None
            if name == "immutable_core263":
                action_control = fit_readout(
                    initial_head, ordered[:prefix],
                    1 - attempted[:prefix], rewards[:prefix],
                    readout_kind=args.readout_kind, intention_width=64,
                    updates=args.fit_updates, batch_size=args.batch_size,
                    learning_rate=3e-3, seed=fit_seed)
                reward_control = fit_readout(
                    initial_head, ordered[:prefix],
                    attempted[:prefix], 1.0 - rewards[:prefix],
                    readout_kind=args.readout_kind, intention_width=64,
                    updates=args.fit_updates, batch_size=args.batch_size,
                    learning_rate=3e-3, seed=fit_seed)
            normal_result = evaluate(
                model, split_features["normal"], labels["normal"])
            counterfactual_result = evaluate(
                model, split_features["counterfactual"],
                labels["counterfactual"])
            missing = {
                split: float(evaluate(
                    model, split_features[split],
                    labels[split])["verified_accuracy"])
                for split in datasets if split not in (
                    "train", "normal", "counterfactual")
            }
            point = {
                "unique_reward_bits": prefix,
                "optimizer_updates": args.fit_updates,
                "examples_processed": (
                    args.fit_updates * min(args.batch_size, prefix)),
                "normal_accuracy": float(
                    normal_result["verified_accuracy"]),
                "counterfactual_accuracy": float(
                    counterfactual_result["verified_accuracy"]),
                "counterfactual_flip_rate": float(
                    (normal_result["predictions"] !=
                     counterfactual_result["predictions"]).float().mean()),
                "missing": missing,
                "action_complement_accuracy": (
                    float(evaluate(
                        action_control, split_features["normal"],
                        labels["normal"])["verified_accuracy"])
                    if action_control is not None else None),
                "reward_complement_accuracy": (
                    float(evaluate(
                        reward_control, split_features["normal"],
                        labels["normal"])["verified_accuracy"])
                    if reward_control is not None else None),
            }
            point["passes_behavior_gates"] = _passes_behavior(point)
            point["passes_all_gates"] = bool(
                point["passes_behavior_gates"]
                and (name != "immutable_core263" or (
                    point["action_complement_accuracy"] <= 0.60
                    and point["reward_complement_accuracy"] <= 0.60)))
            curve.append(point)
        stable = next((
            point["unique_reward_bits"]
            for index, point in enumerate(curve)
            if all(later["passes_all_gates"] for later in curve[index:])
        ), None)
        results[name] = {
            "curve": curve,
            "stable_bits_to_all_gates": stable,
        }
        print(json.dumps({
            "task": args.task, "arm": name, "stable": stable,
            "final": curve[-1],
        }, sort_keys=True), flush=True)

    candidate = results["immutable_core263"]["stable_bits_to_all_gates"]
    fresh = results["matched_fresh263"]["stable_bits_to_all_gates"]
    retention = {
        name: all(torch.equal(
            initial_states[name][key], value)
            for key, value in core.state_dict().items())
        for name, core in arms.items()
    }
    gate = {
        "candidate_reaches_causal_mastery": candidate is not None,
        "fewer_bits_than_matched_fresh": (
            candidate is not None
            and (fresh is None or candidate < fresh)),
        "all_cores_bit_identical": all(retention.values()),
    }
    gate["significant_near_transfer_breakthrough"] = bool(
        gate["candidate_reaches_causal_mastery"]
        and gate["fewer_bits_than_matched_fresh"]
        and gate["all_cores_bit_identical"])
    report = {
        "schema": "identify-near-transfer-ladder-v2",
        "claim_boundary": (
            "Reward-only near-transfer within shared identify event structure; "
            "public appearance may vary and private labels are evaluator-only."),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb_stream", "own_previous_action",
            "attempted_opaque_answer", "scalar_verified_outcome"],
        "verifier_private": [
            "protocol", "effect_direction", "target_direction",
            "logical_counterfactual_pairing"],
        "configuration": vars(args) | {
            "checkpoint": str(args.checkpoint),
            "report": str(args.report),
        },
        "checkpoint_sha256": _sha256(args.checkpoint),
        "experience_accounting": {
            "unique_environment_reward_bits_generated":
                args.train_lifetimes,
            "shared_outcome_stream_across_all_arms": True,
            "total_readout_optimizer_updates": (
                len(prefixes) * args.fit_updates * 6),
            "total_replayed_examples": sum(
                args.fit_updates * min(args.batch_size, prefix) * 6
                for prefix in prefixes),
            "candidate_models_per_prefix": 3,
            "control_arm_models_per_prefix": 1,
        },
        "arms": results,
        "frozen_core_parameters_bit_identical": retention,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
