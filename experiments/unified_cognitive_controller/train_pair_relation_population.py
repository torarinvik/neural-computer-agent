"""Select a sample-efficient relation bridge from matched learning-rate clones.

Every clone receives the exact same sensory/reward stream.  A compact private
validation stream selects one retention-safe candidate; no validation answer
is used for gradient training.  The winner remains unpromoted until a separate
full causal audit on untouched seeds passes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from .audit_pair_relation_repertoire import _load
from .environment import generate_lifetimes
from .train import rollout


VALIDATION_SPECS = (
    # The final field is the first scorable trial. Hidden mapping has one
    # support outcome; the visible tasks are answerable immediately.
    ("dot_a", "pair_relation", "dot_pairs", 0),
    ("dot_b", "pair_relation", "dot_pairs", 0),
    ("bars", "pair_relation", "bars", 0),
    ("diamonds", "pair_relation", "diamonds", 0),
    ("binary_mapping", "binary_mapping", "bars", 1),
    ("visible_context", "visible_context", "bars", 0),
    ("visible_context_xor", "visible_context_xor", "bars", 0),
)
RETENTION_NAMES = (
    "bars", "diamonds", "binary_mapping",
    "visible_context", "visible_context_xor")


def _score_rewards(
        rewards: torch.Tensor, *, query_start: int) -> float:
    if not 0 <= query_start < rewards.shape[1]:
        raise ValueError("query start must leave a nonempty scoring suffix")
    return float(rewards[:, query_start:].float().mean())


@torch.no_grad()
def _accuracy(model, batch, *, query_start: int) -> float:
    result = rollout(
        model, batch, sample_actions=False, feedback_trials=1)
    return _score_rewards(result["rewards"], query_start=query_start)


def _select_winner(
        arms: list[dict[str, object]]) -> dict[str, object]:
    eligible = [arm for arm in arms if bool(arm["selection_eligible"])]
    if not eligible:
        raise RuntimeError(
            "no population arm passed mastery and retention floors")
    return max(
        eligible,
        key=lambda arm: (
            float(arm["dot_score"]),
            float(arm["minimum_retention_margin"]),
            -int(arm["consolidation_steps"]),
            -float(arm["learning_rate"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--learning-rates", default="0.005,0.006,0.007")
    parser.add_argument(
        "--arms",
        help=(
            "comma-separated learning-rate:consolidation-step pairs; "
            "overrides --learning-rates and --consolidation-steps"))
    parser.add_argument("--acquisition-steps", type=int, default=8)
    parser.add_argument("--consolidation-steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-batch-size", type=int, default=16)
    parser.add_argument("--validation-lifetimes", type=int, default=512)
    parser.add_argument("--retention-tolerance", type=float, default=0.025)
    parser.add_argument("--minimum-dot-score", type=float, default=0.95)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    learning_rates = tuple(
        float(value) for value in args.learning_rates.split(","))
    if (
            not learning_rates or any(value <= 0 for value in learning_rates)
            or len(set(learning_rates)) != len(learning_rates)):
        raise ValueError("learning rates must be unique and positive")
    if args.arms:
        arm_specs = tuple(
            (float(part.split(":")[0]), int(part.split(":")[1]))
            for part in args.arms.split(","))
    else:
        arm_specs = tuple(
            (learning_rate, args.consolidation_steps)
            for learning_rate in learning_rates)
    if (
            not arm_specs
            or any(rate <= 0 or steps < 0 for rate, steps in arm_specs)
            or len(set(arm_specs)) != len(arm_specs)):
        raise ValueError(
            "population arms must be unique positive-rate/nonnegative-step "
            "pairs")
    if args.validation_lifetimes < 2 or args.validation_lifetimes % 2:
        raise ValueError(
            "validation lifetimes must be positive and divisible by two")
    if not 0.0 <= args.retention_tolerance < 0.5:
        raise ValueError("retention tolerance is out of range")
    if not 0.5 < args.minimum_dot_score <= 1.0:
        raise ValueError("minimum dot score must be within (0.5, 1]")

    device = torch.device(args.device)
    work_dir = args.work_dir or (
        args.report.parent / f"{args.report.stem}_arms")
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    arm_artifacts = []
    for index, (learning_rate, consolidation_steps) in enumerate(
            arm_specs):
        tag = (
            f"arm{index}_lr{learning_rate:g}_c{consolidation_steps}")
        arm_report = work_dir / f"{tag}.json"
        arm_checkpoint = work_dir / f"{tag}.pt"
        command = [
            sys.executable, "-m",
            "experiments.unified_cognitive_controller."
            "train_pair_relation_appearance_bridge",
            "--parent", str(args.parent),
            "--report", str(arm_report),
            "--candidate-checkpoint-out", str(arm_checkpoint),
            "--skip-final-evaluation",
            "--seed", str(args.seed),
            "--steps", str(args.acquisition_steps),
            "--batch-size", str(args.batch_size),
            "--replay-batch-size", str(args.replay_batch_size),
            "--new-appearance", "dot_pairs",
            "--initialization", "experienced",
            "--retention-weight", "1.0",
            "--locality-weight", "0.001",
            "--learning-rate", str(learning_rate),
            "--consolidation-steps", str(consolidation_steps),
            "--consolidation-retention-weight", "4.0",
            "--device", str(device),
        ]
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True)
        arm_artifacts.append({
            "learning_rate": learning_rate,
            "consolidation_steps": consolidation_steps,
            "report_path": str(arm_report),
            "checkpoint_path": str(arm_checkpoint),
            "training_stdout": completed.stdout.strip(),
            "training_report": json.loads(arm_report.read_text()),
        })

    parent = _load(args.parent, device)
    models = [
        _load(Path(str(arm["checkpoint_path"])), device)
        for arm in arm_artifacts]
    validation_batches = {}
    query_starts = {}
    for index, (name, task, appearance, query_start) in enumerate(
            VALIDATION_SPECS):
        validation_batches[name] = generate_lifetimes(
            args.validation_lifetimes, 6,
            seed=args.seed + 120_000_000 + 1_000_000 * index,
            heldout=True, task=task, appearance=appearance,
            support_trials=1, device=device)
        query_starts[name] = query_start

    parent_retention = {
        name: _accuracy(
            parent, validation_batches[name],
            query_start=query_starts[name])
        for name in RETENTION_NAMES}
    arms = []
    for artifact, model in zip(arm_artifacts, models, strict=True):
        accuracy = {
            name: _accuracy(
                model, batch, query_start=query_starts[name])
            for name, batch in validation_batches.items()}
        margins = {
            name: accuracy[name] - parent_retention[name]
            for name in RETENTION_NAMES}
        retention_safe = all(
            accuracy[name] >= 0.90
            and margins[name] >= -args.retention_tolerance
            for name in RETENTION_NAMES)
        dot_score = min(accuracy["dot_a"], accuracy["dot_b"])
        training_report = artifact["training_report"]
        assert isinstance(training_report, dict)
        arms.append({
            "learning_rate": artifact["learning_rate"],
            "consolidation_steps": artifact["consolidation_steps"],
            "report_path": artifact["report_path"],
            "checkpoint_path": artifact["checkpoint_path"],
            "training_stdout": artifact["training_stdout"],
            "training_accounting": training_report["accounting"],
            "validation_accuracy": accuracy,
            "retention_margin": margins,
            "minimum_retention_margin": min(margins.values()),
            "retention_safe": retention_safe,
            "dot_score": dot_score,
            "selection_eligible": (
                retention_safe and dot_score >= args.minimum_dot_score),
        })
    training_bits = [
        int(arm["training_accounting"]["total_verifier_bits"])
        for arm in arms]
    validation_unique_bits = (
        len(VALIDATION_SPECS) * args.validation_lifetimes * 6)
    parent_validation_bits = (
        len(RETENTION_NAMES) * args.validation_lifetimes * 6)
    accounting = {
        "shared_unique_training_verifier_bits": max(training_bits),
        "population_training_arm_verifier_comparisons":
            sum(training_bits),
        "population_optimizer_updates": sum(
            int(arm["training_accounting"]["optimizer_updates"])
            for arm in arms),
        "validation_unique_verifier_bits": validation_unique_bits,
        "validation_candidate_verifier_comparisons":
            validation_unique_bits * len(arms),
        "validation_parent_verifier_comparisons":
            parent_validation_bits,
        "total_population_verifier_comparisons": (
            sum(training_bits)
            + validation_unique_bits * len(arms)
            + parent_validation_bits),
        "unique_training_plus_selection_verifier_bits":
            max(training_bits) + validation_unique_bits,
    }
    configuration = {
        "parent": str(args.parent),
        "report": str(args.report),
        "checkpoint_out": str(args.checkpoint_out),
        "work_dir": str(work_dir),
        "seed": args.seed,
        "learning_rates": learning_rates,
        "arms": arm_specs,
        "acquisition_steps": args.acquisition_steps,
        "consolidation_steps": args.consolidation_steps,
        "batch_size": args.batch_size,
        "replay_batch_size": args.replay_batch_size,
        "validation_lifetimes": args.validation_lifetimes,
        "retention_tolerance": args.retention_tolerance,
        "minimum_dot_score": args.minimum_dot_score,
        "device": str(device),
    }
    if not any(bool(arm["selection_eligible"]) for arm in arms):
        report = {
            "schema": "pair-relation-population-selection-v1",
            "claim_boundary": (
                "All clones received identical pixels, opaque attempted "
                "actions, scalar outcomes, and rehearsal. No arm passed the "
                "private held-out mastery-and-retention screen, so no "
                "checkpoint was selected or promoted."),
            "configuration": configuration,
            "parent_validation_accuracy": parent_retention,
            "arms": arms,
            "selection": {
                "checkpoint_saved": False,
                "rejection_reason":
                    "no_arm_passed_mastery_and_retention",
            },
            "accounting": accounting,
            "total_seconds": time.perf_counter() - started,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "checkpoint_saved": False,
            "rejection_reason": "no_arm_passed_mastery_and_retention",
            "total_population_verifier_comparisons":
                accounting["total_population_verifier_comparisons"],
        }, sort_keys=True))
        return
    winner = _select_winner(arms)

    selected_payload = torch.load(
        Path(str(winner["checkpoint_path"])),
        map_location="cpu", weights_only=False)
    selected_payload["source_report"] = str(args.report)
    selected_payload["admission_status"] = (
        "population_selected_unpromoted_pending_causal_audit")
    selected_payload["population_selection"] = {
        "seed": args.seed,
        "learning_rate": winner["learning_rate"],
        "consolidation_steps": winner["consolidation_steps"],
        "dot_score": winner["dot_score"],
        "retention_safe": winner["retention_safe"],
    }
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected_payload, args.checkpoint_out)

    report = {
        "schema": "pair-relation-population-selection-v1",
        "claim_boundary": (
            "All clones received identical pixels, opaque attempted actions, "
            "scalar outcomes, and rehearsal. Private validation outcomes "
            "selected one checkpoint but never entered gradient training. "
            "The selected checkpoint is not promoted without an untouched "
            "causal audit."),
        "configuration": configuration,
        "parent_validation_accuracy": parent_retention,
        "arms": arms,
        "selection": {
            "learning_rate": winner["learning_rate"],
            "consolidation_steps": winner["consolidation_steps"],
            "dot_score": winner["dot_score"],
            "retention_safe": winner["retention_safe"],
            "minimum_retention_margin":
                winner["minimum_retention_margin"],
            "checkpoint_saved": True,
        },
        "accounting": accounting,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_learning_rate": winner["learning_rate"],
        "selected_consolidation_steps": winner["consolidation_steps"],
        "validation_dot_score": winner["dot_score"],
        "minimum_retention_margin": winner["minimum_retention_margin"],
        "total_population_verifier_comparisons":
            report["accounting"]["total_population_verifier_comparisons"],
        "seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
