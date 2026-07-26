"""Conservative incumbent/challenger adaptation from attempted outcomes only."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .probe_requery_operation import requery_batch
from .train import evaluate, seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import (
    ComputeAdvantageHead,
    attempted_advantage_target,
)
from .train_thought_compute_transfer import _metrics
from .verified_skill_store import VerifiedSkillStore


def paired_ips_improvement(
        incumbent_actions: torch.Tensor,
        challenger_actions: torch.Tensor,
        attempted_actions: torch.Tensor,
        attempted_utilities: torch.Tensor,
        context_features: torch.Tensor | None = None,
        *, propensity: float = 0.5, z: float = 1.96,
        baseline_mode: str = "global",
        ) -> dict[str, float]:
    """Return paired IPS challenger-minus-incumbent evidence."""
    if not 0 < propensity < 1:
        raise ValueError("propensity must be between zero and one")
    if not (
            incumbent_actions.shape == challenger_actions.shape
            == attempted_actions.shape == attempted_utilities.shape):
        raise ValueError("all logged tensors must have the same shape")
    if baseline_mode not in ("global", "context_crossfit"):
        raise ValueError(baseline_mode)
    if baseline_mode == "context_crossfit" and (
            context_features is None
            or context_features.shape[0] != attempted_utilities.shape[0]):
        raise ValueError(
            "context_crossfit requires one feature row per logged record")
    weights = torch.where(
        attempted_actions.bool(),
        torch.full_like(attempted_utilities, 1.0 / propensity),
        torch.full_like(attempted_utilities, 1.0 / (1.0 - propensity)))
    if baseline_mode == "global":
        baseline = torch.full_like(
            attempted_utilities, attempted_utilities.mean())
    else:
        assert context_features is not None
        baseline = cross_fitted_context_baseline(
            context_features, attempted_utilities)
    # Any context-only baseline is an unbiased control variate for a policy
    # difference because the two action-match indicators have equal expected
    # mass under the randomized logger.
    centered_utilities = attempted_utilities - baseline
    paired = weights * centered_utilities * (
        (attempted_actions == challenger_actions).to(attempted_utilities.dtype)
        - (attempted_actions == incumbent_actions).to(
            attempted_utilities.dtype))
    mean = float(paired.mean())
    standard_error = (
        float(paired.std(unbiased=True)) / math.sqrt(paired.numel())
        if paired.numel() > 1 else float("inf"))
    return {
        "estimated_improvement": mean,
        "standard_error": standard_error,
        "lower_95": mean - z * standard_error,
        "upper_95": mean + z * standard_error,
        "records": paired.numel(),
        "reward_baseline": float(attempted_utilities.mean()),
        "baseline_mode": baseline_mode,
        "baseline_residual_rms": float(
            centered_utilities.square().mean().sqrt()),
    }


def cross_fitted_context_baseline(
        features: torch.Tensor, outcomes: torch.Tensor,
        *, ridge: float = 1e-3) -> torch.Tensor:
    """Predict randomized-policy outcome with two-fold ridge cross-fitting."""
    if features.ndim != 2 or outcomes.ndim != 1:
        raise ValueError("features must be 2-D and outcomes 1-D")
    if features.shape[0] != outcomes.shape[0] or outcomes.numel() < 4:
        raise ValueError("cross-fitting requires at least four paired rows")
    design = torch.cat(
        (features, torch.ones(
            features.shape[0], 1,
            device=features.device, dtype=features.dtype)), dim=1)
    predictions = torch.empty_like(outcomes)
    indices = torch.arange(outcomes.numel(), device=outcomes.device)
    for heldout_parity in (0, 1):
        heldout = indices.remainder(2) == heldout_parity
        training = ~heldout
        x = design[training]
        y = outcomes[training]
        identity = torch.eye(
            x.shape[1], device=x.device, dtype=x.dtype)
        identity[-1, -1] = 0
        coefficients = torch.linalg.solve(
            x.T @ x + ridge * identity, x.T @ y)
        predictions[heldout] = design[heldout] @ coefficients
    return predictions


def _load_head(path: Path, device: torch.device) -> ComputeAdvantageHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    head = ComputeAdvantageHead(int(payload["head_hidden"])).to(device)
    head.load_state_dict(payload["head_state_dict"])
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7951)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--requery-cost", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--promotion-every", type=int, default=4)
    parser.add_argument(
        "--promotion-baseline",
        choices=("global", "context_crossfit"), default="global")
    parser.add_argument("--skill-store", type=Path)
    parser.add_argument("--parent-audit", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    mastered = _load_head(args.head_checkpoint, device)
    hidden = mastered.network[1].out_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        reset = ComputeAdvantageHead(hidden).to(device)
    arms = {}
    for name, initial in (("mastered", mastered), ("gap", reset)):
        arms[name] = {
            "incumbent": copy.deepcopy(initial),
            "challenger": copy.deepcopy(initial),
            "naive": copy.deepcopy(initial),
            "optimizer": None,
            "naive_optimizer": None,
            "logged": [],
            "promotions": [],
            "history": [],
        }
        arms[name]["optimizer"] = torch.optim.AdamW(
            arms[name]["challenger"].parameters(),
            lr=args.learning_rate, weight_decay=1e-4)
        arms[name]["naive_optimizer"] = torch.optim.AdamW(
            arms[name]["naive"].parameters(),
            lr=args.learning_rate, weight_decay=1e-4)

    test_features, test_first, test_second, _ = requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)

    def record(step: int) -> None:
        for arm in arms.values():
            arm["history"].append({
                "step": step,
                "unique_verifier_bits": step * args.batch_size,
                "incumbent": _metrics(
                    arm["incumbent"], test_features, test_first, test_second,
                    thought_cost=args.requery_cost),
                "challenger": _metrics(
                    arm["challenger"], test_features, test_first, test_second,
                    thought_cost=args.requery_cost),
                "naive": _metrics(
                    arm["naive"], test_features, test_first, test_second,
                    thought_cost=args.requery_cost),
            })

    record(0)
    action_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    baselines = {name: [0.0, 0] for name in arms}
    context_sum = torch.zeros(4, device=device)
    context_count = 0
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        features, first, second, _ = requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold)
        context_sum += features.sum(0)
        context_count += features.shape[0]
        attempted = torch.randint(
            0, 2, (args.batch_size,), generator=action_generator,
            device=device)
        utilities = torch.where(
            attempted.bool(), second - args.requery_cost, first)
        for name, arm in arms.items():
            baselines[name][0] += float(utilities.sum())
            baselines[name][1] += utilities.numel()
            targets = attempted_advantage_target(
                attempted, utilities,
                baseline=baselines[name][0] / baselines[name][1],
                propensity=0.5)
            for key, optimizer_key in (
                    ("challenger", "optimizer"),
                    ("naive", "naive_optimizer")):
                loss = nn.functional.smooth_l1_loss(
                    arm[key](features), targets)
                arm[optimizer_key].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(arm[key].parameters(), 1.0)
                arm[optimizer_key].step()
            arm["logged"].append((
                features.detach(), attempted.detach(), utilities.detach()))

        if step % args.promotion_every == 0:
            for arm in arms.values():
                features_log = torch.cat([row[0] for row in arm["logged"]])
                attempted_log = torch.cat([row[1] for row in arm["logged"]])
                utilities_log = torch.cat([row[2] for row in arm["logged"]])
                with torch.no_grad():
                    incumbent_actions = (
                        arm["incumbent"](features_log) > 0).long()
                    challenger_actions = (
                        arm["challenger"](features_log) > 0).long()
                evidence = paired_ips_improvement(
                    incumbent_actions, challenger_actions,
                    attempted_log, utilities_log, features_log,
                    baseline_mode=args.promotion_baseline)
                promoted = evidence["lower_95"] > 0
                if promoted:
                    arm["incumbent"].load_state_dict(
                        arm["challenger"].state_dict())
                    # Earlier records compared against the previous incumbent;
                    # restart the paired audit only after an actual promotion.
                    arm["logged"].clear()
                arm["promotions"].append({
                    "step": step, **evidence, "promoted": promoted})
        if step % 2 == 0 or step == args.steps:
            record(step)

    final = {
        name: arm["history"][-1] for name, arm in arms.items()}
    mastered_start = arms["mastered"]["history"][0]["incumbent"]
    mastered_final = final["mastered"]["incumbent"]
    gap_start = arms["gap"]["history"][0]["incumbent"]
    gap_final = final["gap"]["incumbent"]
    gate = {
        "mastered_incumbent_accuracy_retained":
            mastered_final["compute_choice_accuracy"] >= 0.65,
        "mastered_incumbent_utility_not_degraded": (
            mastered_final["verified_utility"]
            >= mastered_start["verified_utility"] - 0.005),
        "mastered_naive_is_destructive_control": (
            final["mastered"]["naive"]["verified_utility"]
            < mastered_final["verified_utility"] - 0.005),
        "gap_arm_promoted_with_positive_lower_bound": any(
            row["promoted"] and row["lower_95"] > 0
            for row in arms["gap"]["promotions"]),
        "gap_incumbent_utility_improved": (
            gap_final["verified_utility"]
            > gap_start["verified_utility"] + 0.02),
    }
    binary = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 93_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 94_000_000, device=device,
        task="four_rule", feedback_trials=2)
    gate["binary_retained"] = binary["gate"]["accepted"]
    gate["four_rule_retained"] = four_rule["gate"]["accepted"]
    persistence = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, arm in arms.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(arm["incumbent"].state_dict(), path)
            restored = ComputeAdvantageHead(hidden).to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            persistence[name] = torch.equal(
                arm["incumbent"](test_features),
                restored(test_features))
    gate["all_round_trips_exact"] = all(persistence.values())
    skill_store_report = None
    if args.skill_store is not None:
        if args.parent_audit is None:
            raise ValueError("--skill-store requires --parent-audit")
        audit = json.loads(args.parent_audit.read_text())
        parent_margin = min(
            row["verified_utility"] - row["strongest_fixed_utility"]
            for row in audit["streams"])
        promoted_rows = [
            row for row in arms["gap"]["promotions"] if row["promoted"]]
        if not promoted_rows:
            gate["persistent_skill_committed"] = False
        else:
            context_key = torch.nn.functional.normalize(
                context_sum / context_count, dim=0)
            store = VerifiedSkillStore(args.skill_store)
            parent_id = store.commit({
                "head_hidden": hidden,
                "head_state_dict": {
                    key: value.detach().cpu()
                    for key, value in mastered.state_dict().items()},
            }, context_key=context_key,
                lower_confidence_bound=parent_margin,
                verifier_bits=0, parent_id=None,
                provenance={
                    "kind": "fresh_stream_robust_audit",
                    "report": str(args.parent_audit),
                })
            last_promotion = promoted_rows[-1]
            child_id = store.commit({
                "head_hidden": hidden,
                "head_state_dict": {
                    key: value.detach().cpu()
                    for key, value in
                    arms["gap"]["incumbent"].state_dict().items()},
            }, context_key=context_key,
                lower_confidence_bound=last_promotion["lower_95"],
                verifier_bits=last_promotion["step"] * args.batch_size,
                parent_id=parent_id,
                provenance={
                    "kind": "attempted_outcome_safe_promotion",
                    "seed": args.seed,
                    "promotion_evidence": last_promotion,
                })
            fresh_store = VerifiedSkillStore(args.skill_store)
            loaded_parent = fresh_store.load(parent_id, device=device)
            loaded_child = fresh_store.load(child_id, device=device)
            restored_parent = ComputeAdvantageHead(hidden).to(device)
            restored_parent.load_state_dict(
                loaded_parent["payload"]["head_state_dict"])
            restored_child = ComputeAdvantageHead(hidden).to(device)
            restored_child.load_state_dict(
                loaded_child["payload"]["head_state_dict"])
            parent_exact = torch.equal(
                mastered(test_features), restored_parent(test_features))
            child_exact = torch.equal(
                arms["gap"]["incumbent"](test_features),
                restored_child(test_features))
            child_metrics = _metrics(
                restored_child, test_features, test_first, test_second,
                thought_cost=args.requery_cost)
            with tempfile.TemporaryDirectory() as corrupt_directory:
                corrupt_root = Path(corrupt_directory) / "store"
                shutil.copytree(args.skill_store, corrupt_root)
                corrupt_store = VerifiedSkillStore(corrupt_root)
                child_entry = next(
                    row for row in corrupt_store.entries()
                    if row["skill_id"] == child_id)
                child_path = corrupt_root / child_entry["file"]
                child_path.write_bytes(child_path.read_bytes() + b"corrupt")
                corruption_detected = False
                try:
                    corrupt_store.load(child_id)
                except ValueError:
                    corruption_detected = True
                parent_survives_corruption = (
                    corrupt_store.load(parent_id)["schema"]
                    == "verified-latent-skill-v1")
            skill_store_report = {
                "root": str(args.skill_store),
                "parent_id": parent_id,
                "child_id": child_id,
                "parent_margin": parent_margin,
                "child_lower_95": last_promotion["lower_95"],
                "entries": fresh_store.entries(),
                "parent_exact": parent_exact,
                "child_exact": child_exact,
                "child_metrics": child_metrics,
                "corruption_detected": corruption_detected,
                "parent_survives_child_corruption":
                    parent_survives_corruption,
            }
            gate["persistent_skill_committed"] = all((
                parent_exact, child_exact, corruption_detected,
                parent_survives_corruption,
                child_metrics["verified_utility"]
                == gap_final["verified_utility"],
            ))
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "safe-requery-adaptation-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "head_checkpoint": str(args.head_checkpoint),
            "report": str(args.report),
            "skill_store": (
                str(args.skill_store)
                if args.skill_store is not None else None),
            "parent_audit": (
                str(args.parent_audit)
                if args.parent_audit is not None else None),
        },
        "learner_visible": [
            "four_generic_memory_statistics", "attempted_action",
            "attempted_action_scalar_outcome", "logging_propensity_0_5",
        ],
        "hidden_from_learner_and_promotion": [
            "unattempted_outcome", "correct_compute_action",
            "correct_answer", "semantic_task_identity",
            "private_evaluation_metrics",
        ],
        "arms": {
            name: {
                "history": arm["history"],
                "promotion_evidence": arm["promotions"],
            }
            for name, arm in arms.items()
        },
        "final": final,
        "gate": gate,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "persistence_exact": persistence,
        "verified_skill_store": skill_store_report,
        "accounting": {
            "learner_visible_unique_verifier_bits":
                args.steps * args.batch_size,
            "optimizer_updates_per_challenger": args.steps,
            "private_test_both_action_bits": args.test_contexts * 2,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "promotions": {
            name: arm["promotions"] for name, arm in arms.items()},
        "final": final,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
