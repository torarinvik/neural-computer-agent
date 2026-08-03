"""Compress natural memory streams with learned behavioral equivalence."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .audit_selective_disk import _support
from .environment import generate_lifetimes
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import evaluate, seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_conditional_memory_usage_prior import evaluate_conditional
from .train_continuous_memory_usage_prior import (
    evaluate_policy as evaluate_parent_continuous,
)
from .train_memory_replacement import _select_batch
from .train_natural_memory_equivalence import (
    counterfactual_reversal_audit,
    natural_equivalence_batch,
)


@torch.no_grad()
def natural_memory_streams(
        model: UnifiedCognitiveController, *, streams: int, length: int,
        seed: int, device: torch.device, heldout: bool,
        ) -> dict[str, torch.Tensor | int]:
    batch = _add_context_signatures(
        generate_lifetimes(
            streams * length, 3, seed=seed, heldout=heldout,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    keys, values, strengths = _support(model, batch, device=device)
    shape = (streams, length)
    return {
        "keys": keys.reshape(*shape, model.width),
        "values": values.reshape(*shape, model.width),
        "strengths": strengths.reshape(*shape),
        # Verifier-private diagnostic only; never consumed by a policy.
        "rule_bits": batch.rule_bits.reshape(*shape),
        "generated_contexts": streams * length,
    }


@torch.no_grad()
def consolidate(
        model: UnifiedCognitiveController,
        data: dict[str, torch.Tensor | int], *,
        capacity: int = 2, policy: str = "learned",
        invert_relation: bool = False,
        representatives_per_class: int = 1,
        ) -> dict[str, torch.Tensor]:
    """Online merge/store decisions using only controller-created latents."""
    values = data["values"]
    keys = data["keys"]
    strengths = data["strengths"]
    rules = data["rule_bits"]
    assert isinstance(values, torch.Tensor)
    assert isinstance(keys, torch.Tensor)
    assert isinstance(strengths, torch.Tensor)
    assert isinstance(rules, torch.Tensor)
    streams, length, width = values.shape
    if capacity < 1 or capacity > length:
        raise ValueError("capacity must fit the stream")
    if not 1 <= representatives_per_class <= capacity:
        raise ValueError(
            "representatives per class must fit the memory capacity")
    if policy in ("first", "last"):
        indices = (
            torch.arange(capacity, device=values.device)
            if policy == "first"
            else torch.arange(length - capacity, length, device=values.device))
        return {
            "keys": keys[:, indices].clone(),
            "values": values[:, indices].clone(),
            "usage": torch.ones(
                streams, capacity, device=values.device),
            "valid": torch.ones(
                streams, capacity, device=values.device, dtype=torch.bool),
            "rule_bits": rules[:, indices].clone(),
            "cluster_ids": torch.arange(
                capacity, device=values.device).expand(streams, -1).clone(),
            "representative_ranks": torch.zeros(
                streams, capacity, device=values.device, dtype=torch.long),
        }
    if policy not in ("learned", "uncalibrated", "cosine"):
        raise ValueError("unknown consolidation policy")

    bank_values = values.new_zeros(streams, capacity, width)
    bank_keys = keys.new_zeros(streams, capacity, width)
    bank_usage = values.new_zeros(streams, capacity)
    bank_rules = rules.new_full((streams, capacity), -1)
    bank_clusters = rules.new_full((streams, capacity), -1)
    representative_ranks = rules.new_full((streams, capacity), -1)
    valid = torch.zeros(
        streams, capacity, device=values.device, dtype=torch.bool)
    row = torch.arange(streams, device=values.device)
    for step in range(length):
        actual = values[:, step]
        decision = actual
        if step == 0:
            bank_values[:, 0] = actual
            bank_keys[:, 0] = keys[:, step]
            bank_usage[:, 0] = 1.0
            bank_rules[:, 0] = rules[:, step]
            bank_clusters[:, 0] = 0
            representative_ranks[:, 0] = 0
            valid[:, 0] = True
            continue
        if policy == "cosine":
            scores = torch.nn.functional.cosine_similarity(
                decision.unsqueeze(1), bank_values, dim=-1)
            equivalent = scores >= 0.40
        else:
            raw = model.memory_equivalence_logits(decision, bank_values)
            scores = (
                model.calibrated_memory_equivalence_logits(
                    decision, bank_values)
                if policy == "learned" else raw)
            if invert_relation:
                scores = -scores
            equivalent = scores >= 0.0
        equivalent = equivalent & valid
        scores = scores.masked_fill(~valid, float("-inf"))
        matched = equivalent.any(-1)
        matched_slot = scores.argmax(-1)
        # Behavioral equivalence is task-relative. Keeping a small diversity
        # reserve within each discovered class preserves appearance variation
        # that may become useful under a future distribution shift.
        retain_diverse = (
            matched
            & (equivalent.sum(-1) < representatives_per_class)
            & (~valid).any(-1))
        matched = matched & ~retain_diverse
        if bool(matched.any()):
            bank_usage[row[matched], matched_slot[matched]] += 1.0
        unmatched = ~matched
        if not bool(unmatched.any()):
            continue
        has_space = unmatched & (~valid).any(-1)
        if bool(has_space.any()):
            free = (~valid).to(torch.long).argmax(-1)
            target = free[has_space]
            bank_values[row[has_space], target] = actual[has_space]
            bank_keys[row[has_space], target] = keys[has_space, step]
            bank_usage[row[has_space], target] = 1.0
            bank_rules[row[has_space], target] = rules[has_space, step]
            next_cluster = bank_clusters.max(-1).values + 1
            bank_clusters[row[has_space], target] = next_cluster[has_space]
            representative_ranks[row[has_space], target] = 0
            diverse_with_space = retain_diverse & has_space
            if bool(diverse_with_space.any()):
                diverse_target = free[diverse_with_space]
                source = matched_slot[diverse_with_space]
                bank_clusters[
                    row[diverse_with_space], diverse_target
                ] = bank_clusters[row[diverse_with_space], source]
                representative_ranks[
                    row[diverse_with_space], diverse_target
                ] = equivalent[diverse_with_space].sum(-1)
            valid[row[has_space], target] = True
        full = unmatched & ~has_space
        if bool(full.any()):
            target = bank_usage[full].argmin(-1)
            bank_values[row[full], target] = actual[full]
            bank_keys[row[full], target] = keys[full, step]
            bank_usage[row[full], target] = 1.0
            bank_rules[row[full], target] = rules[full, step]
            bank_clusters[row[full], target] = (
                bank_clusters[full].max(-1).values + 1)
            representative_ranks[row[full], target] = 0
    return {
        "keys": bank_keys,
        "values": bank_values,
        "usage": bank_usage,
        "valid": valid,
        "rule_bits": bank_rules,
        "cluster_ids": bank_clusters,
        "representative_ranks": representative_ranks,
    }


@torch.no_grad()
def balanced_query_batch(
        model: UnifiedCognitiveController, *, streams: int, seed: int,
        device: torch.device, heldout: bool,
        ):
    pool = _add_context_signatures(
        generate_lifetimes(
            streams * 2, 3, seed=seed, heldout=heldout,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    first = torch.where(pool.rule_bits == 0)[0]
    second = torch.where(pool.rule_bits == 1)[0]
    if first.numel() != streams or second.numel() != streams:
        raise RuntimeError("balanced generator did not produce equal rules")
    batch = _select_batch(pool, torch.cat((first, second)))
    _, probes, _ = _support(model, batch, device=device)
    return batch, probes


@torch.no_grad()
def bank_behavior(
        model: UnifiedCognitiveController, bank: dict[str, torch.Tensor],
        *, seed: int, device: torch.device, heldout: bool,
        ) -> dict[str, float]:
    streams = bank["values"].shape[0]
    batch, probes = balanced_query_batch(
        model, streams=streams, seed=seed,
        device=device, heldout=heldout)
    values = bank["values"].repeat(2, 1, 1)
    valid = bank["valid"].repeat(2, 1)
    logits = model.calibrated_memory_equivalence_logits(probes, values)
    selected = logits.masked_fill(~valid, float("-inf")).argmax(-1)
    retrieved = values[
        torch.arange(values.shape[0], device=device), selected]
    outcomes = _outcomes(model, batch, retrieved, device=device).float()
    coverage = (
        (bank["rule_bits"] == 0).any(-1)
        & (bank["rule_bits"] == 1).any(-1))
    return {
        "visual_accuracy": float(outcomes.mean()),
        "rule_zero_accuracy": float(outcomes[:streams].mean()),
        "rule_one_accuracy": float(outcomes[streams:].mean()),
        "both_behaviors_retained": float(coverage.float().mean()),
        "mean_rows": float(bank["valid"].sum(-1).float().mean()),
    }


@torch.no_grad()
def evaluate_consolidation(
        model: UnifiedCognitiveController, *, streams: int, length: int,
        seed: int, device: torch.device,
        ) -> dict[str, object]:
    data = natural_memory_streams(
        model, streams=streams, length=length, seed=seed,
        device=device, heldout=True)
    policies = {
        "learned": consolidate(model, data, policy="learned"),
        "uncalibrated": consolidate(model, data, policy="uncalibrated"),
        "cosine": consolidate(model, data, policy="cosine"),
        "first_two": consolidate(model, data, policy="first"),
        "last_two": consolidate(model, data, policy="last"),
        "relation_inverted": consolidate(
            model, data, policy="learned", invert_relation=True),
    }
    full = {
        "keys": data["keys"],
        "values": data["values"],
        "usage": torch.ones_like(data["strengths"]),
        "valid": torch.ones_like(data["strengths"], dtype=torch.bool),
        "rule_bits": data["rule_bits"],
    }
    policies["uncompressed"] = full
    return {
        name: bank_behavior(
            model, bank, seed=seed + 20_000_000,
            device=device, heldout=True)
        for name, bank in policies.items()
    }


@torch.no_grad()
def physical_consolidation_audit(
        model: UnifiedCognitiveController, *, streams: int, length: int,
        seed: int, device: torch.device,
        ) -> dict[str, object]:
    data = natural_memory_streams(
        model, streams=streams, length=length, seed=seed,
        device=device, heldout=True)
    bank = consolidate(model, data, policy="learned")
    restored_values = []
    restored_valid = []
    restored_rules = []
    exact = 0
    compressed_bytes = full_bytes = 0
    with tempfile.TemporaryDirectory(
            prefix="equivalence-consolidation-") as root:
        directory = Path(root)
        for index in range(streams):
            compact = DiskLatentMemory(
                model.width, capacity=2, device=device)
            compact.commit(
                bank["keys"][index], bank["values"][index],
                bank["usage"][index], threshold=0.0)
            compact_path = directory / f"compact-{index:04d}.pt"
            compact.save(compact_path)
            restored = DiskLatentMemory.load(compact_path, device=device)
            exact += int(
                torch.equal(restored.store.keys, compact.store.keys)
                and torch.equal(restored.store.values, compact.store.values)
                and torch.equal(restored.store.usage, compact.store.usage))
            restored_values.append(restored.store.values[:2])
            restored_valid.append(restored.store.valid[:2])
            restored_rules.append(bank["rule_bits"][index])
            compressed_bytes += compact_path.stat().st_size

            full = DiskLatentMemory(
                model.width, capacity=length, device=device)
            full.commit(
                data["keys"][index], data["values"][index],
                torch.ones(length, device=device), threshold=0.0)
            full_path = directory / f"full-{index:04d}.pt"
            full.save(full_path)
            full_bytes += full_path.stat().st_size
    restored_bank = {
        "values": torch.stack(restored_values),
        "valid": torch.stack(restored_valid),
        "rule_bits": torch.stack(restored_rules),
    }
    behavior = bank_behavior(
        model, restored_bank, seed=seed + 20_000_000,
        device=device, heldout=True)
    return {
        **behavior,
        "banks": streams,
        "exact_reload_count": exact,
        "all_banks_reload_exactly": exact == streams,
        "compressed_bytes": compressed_bytes,
        "uncompressed_bytes": full_bytes,
        "byte_ratio": compressed_bytes / full_bytes,
        "logical_row_ratio": 2 / length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=20501)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-updates", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--test-streams", type=int, default=512)
    parser.add_argument("--stream-length", type=int, default=16)
    parser.add_argument("--physical-streams", type=int, default=128)
    parser.add_argument("--retention-count", type=int, default=256)
    parser.add_argument("--shuffle-rewards", action="store_true")
    args = parser.parse_args()
    if (
            args.steps < 1 or args.batch_size < 4
            or args.replay_updates < 1 or args.stream_length < 4):
        parser.error("training and stream budgets are too small")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"])
    if not configuration.get(
            "adaptive_memory_equivalence_calibration", False):
        configuration["adaptive_memory_equivalence_calibration"] = True
        model = UnifiedCognitiveController(**configuration).to(device)
        missing, unexpected = model.load_state_dict(
            payload["state_dict"], strict=False)
        expected = {
            "memory_equivalence_logit_scale",
            "memory_equivalence_logit_bias",
        }
        if set(missing) != expected or unexpected:
            raise ValueError(
                f"unexpected calibration expansion: "
                f"{missing=}, {unexpected=}")
    else:
        model = UnifiedCognitiveController(**configuration).to(device)
        model.load_state_dict(payload["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.memory_equivalence_logit_scale is not None
    assert model.memory_equivalence_logit_bias is not None
    trainable = [
        model.memory_equivalence_logit_scale,
        model.memory_equivalence_logit_bias,
    ]
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    preflight = evaluate_consolidation(
        model, streams=min(args.test_streams, 256),
        length=args.stream_length, seed=args.seed + 90_000_000,
        device=device)
    verifier_bits = logical_lifetimes = optimizer_updates = 0
    replayed_examples = 0
    started = time.perf_counter()
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 91_000_000)
    history = []
    for step in range(1, args.steps + 1):
        data = natural_equivalence_batch(
            model, count=args.batch_size,
            seed=args.seed * 1_000_000 + step,
            device=device, heldout=False, exact_fraction=0.0)
        features = data["sorted_values"].detach()
        probes = data["probe_values"].detach()
        outcomes = data["sorted_outcomes"].detach()
        verifier_bits += int(data["mining_verifier_bits"])
        logical_lifetimes += int(data["generated_contexts"])
        if args.shuffle_rewards:
            outcomes = outcomes.flatten()[
                torch.randperm(
                    outcomes.numel(), generator=reward_generator,
                    device=device)
            ].reshape_as(outcomes)
        for _ in range(args.replay_updates):
            logits = model.calibrated_memory_equivalence_logits(
                probes, features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, outcomes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            replayed_examples += outcomes.numel()
        with torch.no_grad():
            accuracy = float(
                ((logits >= 0) == outcomes.bool()).float().mean())
        history.append({
            "step": step,
            "verifier_bits": verifier_bits,
            "training_pair_accuracy": accuracy,
            "scale": float(
                model.memory_equivalence_logit_scale.detach()),
            "bias": float(model.memory_equivalence_logit_bias.detach()),
        })
    held_out = evaluate_consolidation(
        model, streams=args.test_streams, length=args.stream_length,
        seed=args.seed + 92_000_000, device=device)
    physical = physical_consolidation_audit(
        model, streams=args.physical_streams, length=args.stream_length,
        seed=args.seed + 93_000_000, device=device)
    reversal = counterfactual_reversal_audit(
        model, count=args.retention_count, seed=args.seed + 94_000_000,
        device=device)
    parent_continuous = evaluate_parent_continuous(
        model, count=args.test_streams, rows=4,
        seed=args.seed + 95_000_000, device=device,
        scale_cost=0.30, difficulty="separated")
    parent_conditional = evaluate_conditional(
        model, count=args.test_streams, seed=args.seed + 96_000_000,
        device=device)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 97_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 98_000_000, device=device,
        task="four_rule", feedback_trials=2)
    changed = sorted(
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu()))
    allowed = {
        "memory_equivalence_logit_scale",
        "memory_equivalence_logit_bias",
    }
    learned = held_out["learned"]
    gates = {
        "learned_visual_accuracy_at_least_98":
            learned["visual_accuracy"] >= 0.98,
        "learned_retains_both_at_least_98":
            learned["both_behaviors_retained"] >= 0.98,
        "within_one_point_of_uncompressed":
            learned["visual_accuracy"]
            >= held_out["uncompressed"]["visual_accuracy"] - 0.01,
        "beats_first_two_by_20_points":
            learned["visual_accuracy"]
            >= held_out["first_two"]["visual_accuracy"] + 0.20,
        "beats_uncalibrated_by_5_points":
            learned["both_behaviors_retained"]
            >= held_out["uncalibrated"]["both_behaviors_retained"] + 0.05,
        "relation_inversion_costs_20_coverage_points":
            learned["both_behaviors_retained"]
            >= held_out["relation_inverted"][
                "both_behaviors_retained"] + 0.20,
        "physical_accuracy_at_least_98":
            physical["visual_accuracy"] >= 0.98,
        "physical_reload_exact": physical["all_banks_reload_exactly"],
        "physical_rows_compressed_8x":
            physical["logical_row_ratio"] <= 0.125,
        "counterfactual_retrieval_retained":
            reversal["ordinary_accuracy"] >= 0.98
            and reversal["reversed_accuracy"] >= 0.98
            and reversal["selection_flip_rate"] >= 0.98,
        "parent_continuous_retained":
            parent_continuous["continuous"]["row_accuracy"] >= 0.95,
        "parent_conditional_retained":
            parent_conditional["learned"]["accuracy"] >= 0.95,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "only_calibration_changed": set(changed) <= allowed,
        "under_three_minutes": time.perf_counter() - started <= 180.0,
    }
    gates["accepted"] = all(gates.values())
    report = {
        "schema": "natural-equivalence-consolidation-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "learner_visible": [
            "controller_created_memory_values",
            "scalar_candidate_verifier_outcomes",
        ],
        "hidden_from_learner": [
            "rule_bits", "equivalence_labels",
            "correct_merge_or_store_action",
        ],
        "semantic_labels_used_for_training": False,
        "preflight": preflight,
        "history": history,
        "held_out": held_out,
        "physical": physical,
        "counterfactual_reversal": reversal,
        "retention": {
            "parent_continuous": parent_continuous,
            "parent_conditional": parent_conditional,
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "accounting": {
            "unique_verifier_bits": verifier_bits,
            "unique_logical_lifetimes": logical_lifetimes,
            "optimizer_updates": optimizer_updates,
            "replayed_pair_examples": replayed_examples,
            "wall_seconds": time.perf_counter() - started,
        },
        "changed_parameters": changed,
        "gates": gates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-controller-equivalence-consolidation-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_checkpoint": str(args.checkpoint_in),
            "report": report,
        }, args.checkpoint_out)
    print(json.dumps({
        "learned": held_out["learned"],
        "uncompressed": held_out["uncompressed"],
        "uncalibrated": held_out["uncalibrated"],
        "first_two": held_out["first_two"],
        "relation_inverted": held_out["relation_inverted"],
        "physical": physical,
        "history": history,
        "gates": gates,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
