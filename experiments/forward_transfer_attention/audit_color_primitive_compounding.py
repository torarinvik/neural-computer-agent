"""Reward-only compounding from acquired color primitives.

The learner receives rendered RGB events, attempted opaque binary answers, and
scalar verifier outcomes. Private identities and relation labels are used only
to generate outcomes and evaluate discarded heads. No semantic label is a
differentiable target.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.syllogimous_latent_agent.model import VisionEncoder

from .audit_identify_near_transfer import (
    PREFIXES,
    _balanced_indices,
    _subset_data,
    private_label,
    transfer_features,
)
from .audit_immutable_core_cross_primitive import _load_arms
from .train import seed_everything
from .train_identify_then_act import (
    evaluate,
    fit_readout,
    identify_batch,
    make_readout,
)


ATOM_TRAIN_START = 221_000_000
RELATION_TRAIN_START = 225_000_000
TEST_START = 229_000_000


def load_color_compounder_checkpoint(
        path: Path, *, device: torch.device
        ) -> dict[str, torch.nn.Module | dict[str, object]]:
    """Reconstruct the deployed modules from a curated milestone."""
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != "color-primitive-compounder-v1":
        raise ValueError("unsupported color compounder checkpoint schema")
    target_vision = VisionEncoder(64).to(device)
    effect_vision = VisionEncoder(64).to(device)
    target_head = make_readout("antisymmetric", 64 * 3, 64).to(device)
    effect_head = make_readout("antisymmetric", 64 * 3, 64).to(device)
    relation_head = make_readout("antisymmetric", 4, 16).to(device)
    modules = {
        "target_vision": target_vision,
        "effect_vision": effect_vision,
        "target_head": target_head,
        "effect_head": effect_head,
        "relation_head": relation_head,
    }
    for name, module in modules.items():
        module.load_state_dict(payload[name])
        module.eval()
    return {"source": payload["source"], **modules}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_data(
        start: int, count: int, *, task: str, heldout: bool,
        seed: int, data_offset: int = 0,
        **kwargs: object) -> dict[str, torch.Tensor]:
    pool = identify_batch(
        start + data_offset, count * 2, heldout=heldout, **kwargs)
    indices = _balanced_indices(
        private_label(pool, task), count, seed=seed)
    return _subset_data(pool, indices)


def _logged_outcomes(
        labels: torch.Tensor, *, seed: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attempted = torch.randint(
        0, 2, (labels.shape[0],),
        generator=torch.Generator().manual_seed(seed))
    rewards = (attempted == labels).float()
    order = torch.randperm(
        labels.shape[0],
        generator=torch.Generator().manual_seed(seed + 1))
    return attempted[order], rewards[order], order


def _decorrelated_control_rewards(
        ordered_target: torch.Tensor, ordered_effect: torch.Tensor,
        attempted: torch.Tensor, *,
        seed: int) -> torch.Tensor:
    """Randomize within every primitive quadrant with exact global balance."""
    if ordered_target.shape != ordered_effect.shape:
        raise ValueError("target/effect control labels must align")
    if ordered_target.shape[0] % 2:
        raise ValueError("decorrelated binary control requires an even count")
    generator = torch.Generator().manual_seed(seed)
    joint = ordered_effect * 2 + ordered_target
    groups = [torch.where(joint == value)[0] for value in range(4)]
    base_ones = sum(group.numel() // 2 for group in groups)
    extra_ones = ordered_target.shape[0] // 2 - base_ones
    odd_groups = [
        index for index, group in enumerate(groups)
        if group.numel() % 2]
    if extra_ones < 0 or extra_ones > len(odd_groups):
        raise AssertionError("quadrant balancing arithmetic is inconsistent")
    odd_order = torch.randperm(
        len(odd_groups), generator=generator).tolist()
    receive_extra = {
        odd_groups[index] for index in odd_order[:extra_ones]}
    shuffled = torch.empty_like(ordered_target)
    for group_index, indices in enumerate(groups):
        order = indices[torch.randperm(
            indices.numel(), generator=generator)]
        ones = indices.numel() // 2 + int(group_index in receive_extra)
        shuffled[order[:ones]] = 1
        shuffled[order[ones:]] = 0
    if torch.bincount(shuffled, minlength=2).tolist() != [
            ordered_target.shape[0] // 2] * 2:
        raise AssertionError("control labels are not globally balanced")
    return (attempted == shuffled).float()


def _fit_atom(
        core: torch.nn.Module, data: dict[str, torch.Tensor], *,
        task: str, updates: int, batch_size: int, seed: int,
        device: torch.device,
        initialization_seed: int | None = None,
        reward_permutation: torch.Tensor | None = None,
        ) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    features = transfer_features(
        core, data, interface="event_vision", device=device)
    labels = private_label(data, task)
    attempted, rewards, order = _logged_outcomes(labels, seed=seed)
    if reward_permutation is not None:
        rewards = rewards[reward_permutation]
    seed_everything(
        seed + 2 if initialization_seed is None
        else initialization_seed)
    initial_model = make_readout(
        "antisymmetric", features.shape[-1], 64).to(device)
    initial = copy.deepcopy(initial_model.state_dict())
    trained = fit_readout(
        initial, features[order.to(device)],
        attempted, rewards,
        readout_kind="antisymmetric", intention_width=64,
        updates=updates, batch_size=batch_size,
        learning_rate=3e-3, seed=seed + 3)
    return trained, initial


@torch.no_grad()
def _primitive_latent(
        target_head: torch.nn.Module, target_core: torch.nn.Module,
        effect_head: torch.nn.Module, effect_core: torch.nn.Module,
        data: dict[str, torch.Tensor], *,
        device: torch.device) -> torch.Tensor:
    target_features = transfer_features(
        target_core, data, interface="event_vision", device=device)
    effect_features = transfer_features(
        effect_core, data, interface="event_vision", device=device)
    return torch.cat([
        target_head(target_features),
        effect_head(effect_features),
    ], dim=-1)


def _evaluate_relation(
        model: torch.nn.Module,
        features: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor]) -> dict[str, object]:
    normal = evaluate(model, features["normal"], labels["normal"])
    result: dict[str, object] = {
        "normal_accuracy": float(normal["verified_accuracy"]),
        "missing": {
            name: float(evaluate(
                model, features[name], labels[name])["verified_accuracy"])
            for name in ("missing_consequence", "missing_target")
        },
    }
    for name in ("protocol_counterfactual", "target_counterfactual"):
        audit = evaluate(model, features[name], labels[name])
        result[f"{name}_accuracy"] = float(audit["verified_accuracy"])
        result[f"{name}_flip_rate"] = float(
            (normal["predictions"] != audit["predictions"]).float().mean())
    return result


def _passes_behavior(point: dict[str, object]) -> bool:
    return bool(
        float(point["normal_accuracy"]) >= 0.75
        and float(point["protocol_counterfactual_accuracy"]) >= 0.70
        and float(point["target_counterfactual_accuracy"]) >= 0.70
        and float(point["protocol_counterfactual_flip_rate"]) >= 0.60
        and float(point["target_counterfactual_flip_rate"]) >= 0.60
        and all(
            float(value) <= 0.60
            for value in point["missing"].values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--target-bits", type=int, default=64)
    parser.add_argument("--effect-bits", type=int, default=24)
    parser.add_argument("--relation-bits", type=int, default=64)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--atom-updates", type=int, default=128)
    parser.add_argument("--relation-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--target-initialization-seed", type=int)
    parser.add_argument("--effect-initialization-seed", type=int)
    parser.add_argument("--relation-initialization-seed", type=int)
    parser.add_argument("--data-offset", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device(args.device)
    arms = _load_arms(args.checkpoint, device=device)
    inherited = arms["immutable_core263"]
    fresh = arms["matched_fresh263"]
    initial_core_states = {
        name: {
            key: value.detach().clone()
            for key, value in core.state_dict().items()
        }
        for name, core in arms.items()
    }

    target_data = _balanced_data(
        ATOM_TRAIN_START, args.target_bits, task="target_side",
        heldout=False, seed=args.seed + 10,
        data_offset=args.data_offset, relation_axis="color_salient")
    effect_data = _balanced_data(
        ATOM_TRAIN_START + 1_000_000, args.effect_bits,
        task="effect_side", heldout=False, seed=args.seed + 20,
        data_offset=args.data_offset, relation_axis="color_salient")
    target_head, target_initial = _fit_atom(
        fresh, target_data, task="target_side",
        updates=args.atom_updates, batch_size=args.batch_size,
        seed=args.seed + 30, device=device,
        initialization_seed=args.target_initialization_seed)
    effect_head, effect_initial = _fit_atom(
        inherited, effect_data, task="effect_side",
        updates=args.atom_updates, batch_size=args.batch_size,
        seed=args.seed + 40, device=device,
        initialization_seed=args.effect_initialization_seed)
    target_untrained = make_readout(
        "antisymmetric", 64 * 3, 64).to(device)
    target_untrained.load_state_dict(target_initial)
    effect_untrained = make_readout(
        "antisymmetric", 64 * 3, 64).to(device)
    effect_untrained.load_state_dict(effect_initial)

    relation_train = _balanced_data(
        RELATION_TRAIN_START, args.relation_bits,
        task="effect_target_match", heldout=False,
        seed=args.seed + 50, data_offset=args.data_offset,
        relation_axis="color_salient")
    normal = _balanced_data(
        TEST_START, args.test_lifetimes,
        task="effect_target_match", heldout=True,
        seed=args.seed + 60, data_offset=args.data_offset,
        relation_axis="color_salient")
    # Every audit reuses the exact balanced logical indices selected above.
    normal_pool = identify_batch(
        TEST_START + args.data_offset, args.test_lifetimes * 2,
        heldout=True, relation_axis="color_salient")
    normal_indices = _balanced_indices(
        private_label(normal_pool, "effect_target_match"),
        args.test_lifetimes, seed=args.seed + 60)
    datasets = {
        "train": relation_train,
        "normal": normal,
        "protocol_counterfactual": _subset_data(identify_batch(
            TEST_START + args.data_offset, args.test_lifetimes * 2,
            heldout=True, relation_axis="color_salient",
            swap_protocol=True), normal_indices),
        "target_counterfactual": _subset_data(identify_batch(
            TEST_START + args.data_offset, args.test_lifetimes * 2,
            heldout=True, relation_axis="color_salient",
            reverse_target=True), normal_indices),
        "missing_consequence": _subset_data(identify_batch(
            TEST_START + args.data_offset, args.test_lifetimes * 2,
            heldout=True, relation_axis="color_salient",
            missing_consequence=True), normal_indices),
        "missing_target": _subset_data(identify_batch(
            TEST_START + args.data_offset, args.test_lifetimes * 2,
            heldout=True, relation_axis="color_salient",
            missing_target=True), normal_indices),
    }
    labels = {
        name: private_label(data, "effect_target_match")
        for name, data in datasets.items()
    }
    atom_test_labels = {
        "target": private_label(normal, "target_side"),
        "effect": private_label(normal, "effect_side"),
    }
    atom_accuracy = {
        "target": float(evaluate(
            target_head,
            transfer_features(
                fresh, normal, interface="event_vision", device=device),
            atom_test_labels["target"])["verified_accuracy"]),
        "effect": float(evaluate(
            effect_head,
            transfer_features(
                inherited, normal, interface="event_vision", device=device),
            atom_test_labels["effect"])["verified_accuracy"]),
    }

    head_arms = {
        "both_acquired": (target_head, effect_head),
        "target_only": (target_head, effect_untrained),
        "effect_only": (target_untrained, effect_head),
        "neither_acquired": (target_untrained, effect_untrained),
    }
    features = {
        arm: {
            split: _primitive_latent(
                target, fresh, effect, inherited, data, device=device)
            for split, data in datasets.items()
        }
        for arm, (target, effect) in head_arms.items()
    }
    attempted, rewards, order = _logged_outcomes(
        labels["train"], seed=args.seed + 70)
    prefixes = [
        value for value in PREFIXES if value <= args.relation_bits]
    if not prefixes or prefixes[-1] != args.relation_bits:
        prefixes.append(args.relation_bits)
    seed_everything(
        args.seed + 80
        if args.relation_initialization_seed is None
        else args.relation_initialization_seed)
    relation_initial = copy.deepcopy(make_readout(
        "antisymmetric", 4, 16).to(device).state_dict())

    results = {}
    relation_states: dict[int, dict[str, torch.Tensor]] = {}
    for arm, split_features in features.items():
        ordered = split_features["train"][order.to(device)]
        curve = []
        for prefix in prefixes:
            fit_seed = args.seed + 100 + prefix
            model = fit_readout(
                relation_initial, ordered[:prefix],
                attempted[:prefix], rewards[:prefix],
                readout_kind="antisymmetric", intention_width=16,
                updates=args.relation_updates,
                batch_size=args.batch_size,
                learning_rate=3e-3, seed=fit_seed)
            if arm == "both_acquired":
                relation_states[prefix] = copy.deepcopy(model.state_dict())
            point = {
                "unique_relation_reward_bits": prefix,
                "optimizer_updates": args.relation_updates,
                "examples_processed": (
                    args.relation_updates
                    * min(args.batch_size, prefix)),
                **_evaluate_relation(model, split_features, labels),
            }
            point["passes_behavior_gates"] = _passes_behavior(point)
            if arm == "both_acquired":
                action_control = fit_readout(
                    relation_initial, ordered[:prefix],
                    1 - attempted[:prefix], rewards[:prefix],
                    readout_kind="antisymmetric", intention_width=16,
                    updates=args.relation_updates,
                    batch_size=args.batch_size,
                    learning_rate=3e-3, seed=fit_seed)
                reward_control = fit_readout(
                    relation_initial, ordered[:prefix],
                    attempted[:prefix], 1.0 - rewards[:prefix],
                    readout_kind="antisymmetric", intention_width=16,
                    updates=args.relation_updates,
                    batch_size=args.batch_size,
                    learning_rate=3e-3, seed=fit_seed)
                point["action_complement_accuracy"] = float(evaluate(
                    action_control, split_features["normal"],
                    labels["normal"])["verified_accuracy"])
                point["reward_complement_accuracy"] = float(evaluate(
                    reward_control, split_features["normal"],
                    labels["normal"])["verified_accuracy"])
            else:
                point["action_complement_accuracy"] = None
                point["reward_complement_accuracy"] = None
            point["passes_all_gates"] = bool(
                point["passes_behavior_gates"]
                and (arm != "both_acquired" or (
                    point["action_complement_accuracy"] <= 0.60
                    and point["reward_complement_accuracy"] <= 0.60)))
            curve.append(point)
        stable = next((
            point["unique_relation_reward_bits"]
            for index, point in enumerate(curve)
            if all(later["passes_all_gates"] for later in curve[index:])
        ), None)
        results[arm] = {
            "curve": curve,
            "stable_relation_bits_to_all_gates": stable,
        }
        print(json.dumps({
            "arm": arm, "stable": stable, "final": curve[-1],
        }, sort_keys=True), flush=True)

    # Calibration: three label permutations with exactly 50% overlap with the
    # real relation. Private labels construct only this discarded control.
    shuffled_controls = []
    ordered_target = private_label(
        relation_train, "target_side")[order]
    ordered_effect = private_label(
        relation_train, "effect_side")[order]
    for control in range(3):
        shuffled_rewards = _decorrelated_control_rewards(
            ordered_target, ordered_effect, attempted,
            seed=args.seed + 900 + control)
        shuffled_model = fit_readout(
            relation_initial,
            features["both_acquired"]["train"][order.to(device)],
            attempted, shuffled_rewards,
            readout_kind="antisymmetric", intention_width=16,
            updates=args.relation_updates, batch_size=args.batch_size,
            learning_rate=3e-3, seed=args.seed + 910 + control)
        shuffled_metrics = _evaluate_relation(
            shuffled_model, features["both_acquired"], labels)
        shuffled_metrics["passes_behavior_gates"] = _passes_behavior(
            shuffled_metrics)
        shuffled_controls.append(shuffled_metrics)
    shuffled_normal = sorted(
        float(control["normal_accuracy"])
        for control in shuffled_controls)
    shuffled_median_accuracy = shuffled_normal[len(shuffled_normal) // 2]

    acquired = results["both_acquired"]["stable_relation_bits_to_all_gates"]
    fresh_control = results[
        "neither_acquired"]["stable_relation_bits_to_all_gates"]
    retention = {
        name: all(torch.equal(
            initial_core_states[name][key], value)
            for key, value in core.state_dict().items())
        for name, core in arms.items()
    }
    gate = {
        "acquired_primitives_reach_causal_mastery": acquired is not None,
        "fewer_relation_bits_than_unacquired_control": bool(
            acquired is not None
            and (fresh_control is None or acquired < fresh_control)),
        "shuffled_outcomes_have_no_causal_pass": not any(
            control["passes_behavior_gates"]
            for control in shuffled_controls),
        "shuffled_outcome_median_at_chance":
            shuffled_median_accuracy <= 0.60,
        "all_cores_bit_identical": all(retention.values()),
    }
    gate["significant_compounding_breakthrough"] = all(gate.values())
    report = {
        "schema": "color-primitive-compounding-v1",
        "claim_boundary": (
            "Reward-only color-primitive acquisition and reuse inside a "
            "shared event structure; continuous primitive latents may be "
            "useful before their standalone behavior is mastered."),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb_events", "attempted_opaque_binary_answer",
            "scalar_verified_outcome", "learned_primitive_latents",
        ],
        "verifier_private": [
            "target_identity", "effect_identity",
            "effect_target_relation", "counterfactual_pairing",
        ],
        "configuration": vars(args) | {
            "checkpoint": str(args.checkpoint),
            "report": str(args.report),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
        },
        "checkpoint_sha256": _sha256(args.checkpoint),
        "experience_accounting": {
            "target_atom_unique_reward_bits": args.target_bits,
            "effect_atom_unique_reward_bits": args.effect_bits,
            "relation_unique_reward_bits_generated": args.relation_bits,
            "relation_outcome_stream_shared_across_arms": True,
            "atom_optimizer_updates": args.atom_updates * 2,
            "relation_optimizer_updates_per_curve_point":
                args.relation_updates,
        },
        "atom_heldout_accuracy": atom_accuracy,
        "arms": results,
        "shuffled_outcome_controls": shuffled_controls,
        "shuffled_outcome_median_accuracy": shuffled_median_accuracy,
        "frozen_core_parameters_bit_identical": retention,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    if gate["significant_compounding_breakthrough"] and (
            args.checkpoint_out is not None):
        if acquired not in relation_states:
            raise AssertionError("stable relation state was not retained")
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "color-primitive-compounder-v1",
            "source": {
                "core_checkpoint": str(args.checkpoint),
                "core_checkpoint_sha256": _sha256(args.checkpoint),
                "configuration": report["configuration"],
                "stable_relation_reward_bits": acquired,
                "semantic_labels_used_for_training": False,
                "unattempted_action_labels_used_for_training": False,
            },
            "target_vision": fresh.vision.state_dict(),
            "effect_vision": inherited.vision.state_dict(),
            "target_head": target_head.state_dict(),
            "effect_head": effect_head.state_dict(),
            "relation_head": relation_states[acquired],
        }, args.checkpoint_out)
        report["output_checkpoint"] = {
            "path": str(args.checkpoint_out),
            "sha256": _sha256(args.checkpoint_out),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "atom_accuracy": atom_accuracy,
        "shuffled_outcome_controls": shuffled_controls,
        "shuffled_outcome_median_accuracy": shuffled_median_accuracy,
        "gate": gate,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
