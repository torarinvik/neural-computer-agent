"""Race complement-slot clones and promote only a retention-safe winner.

Each arm receives the same freshly collected target stream (``--data-seed``)
but a different model/optimizer seed. A private held-out audit selects among
arms; its verifier labels never enter training. The selected arm then receives
one larger untouched audit and an optional matched shuffled-outcome control.
The script writes a checkpoint only when the final causal, reset, and
span-nine/span-ten retention gates pass.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, text=True, capture_output=True)


def _audit_summary(
        audit: dict[str, Any], *, minimum_causal_gain: float,
        minimum_complement_accuracy: float,
        maximum_retention_drop_points: float,
        require_reset: bool = True) -> dict[str, Any]:
    parent = audit["parent_audit"]
    candidate = audit["candidate_audit"]
    retention = {
        name: float(candidate[name]["accuracy"] - parent[name]["accuracy"])
        for name in ("span9", "span10")}
    complement = candidate["complement"]
    causal_gain = float(audit["causal_gain_points"])
    complement_accuracy = float(complement["accuracy"])
    reset_accuracy = float(complement["all_memory_reset_accuracy"])
    causal_ok = causal_gain >= minimum_causal_gain
    accuracy_ok = complement_accuracy >= minimum_complement_accuracy
    retention_ok = all(
        value >= -maximum_retention_drop_points / 100.0
        for value in retention.values())
    reset_ok = 0.45 <= reset_accuracy <= 0.55
    return {
        "complement_accuracy": complement_accuracy,
        "causal_gain_points": causal_gain,
        "retention_margin": retention,
        "minimum_retention_margin": min(retention.values()),
        "reset_accuracy": reset_accuracy,
        "causal_ok": causal_ok,
        "accuracy_ok": accuracy_ok,
        "retention_ok": retention_ok,
        "reset_ok": reset_ok,
        "selection_eligible": (
            causal_ok and accuracy_ok and retention_ok
            and (reset_ok or not require_reset)),
    }


def _select_winner(arms: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [arm for arm in arms if arm["private"]["selection_eligible"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda arm: (
            float(arm["private"]["complement_accuracy"]),
            float(arm["private"]["causal_gain_points"]),
            float(arm["private"]["minimum_retention_margin"]),
            -int(arm["seed"])))


def _train_command(
        *, args: argparse.Namespace, seed: int, report: Path,
        checkpoint: Path, shuffle: bool = False) -> list[str]:
    command = [
        sys.executable, "-m",
        "experiments.unified_cognitive_controller.train_sequence_reward_buffer",
    ]
    command.extend([
        "--parent", str(args.parent),
        "--report", str(report),
        "--checkpoint-out", str(checkpoint),
        "--seed", str(seed),
        "--data-seed", str(args.data_seed),
        "--train-lifetimes", str(args.train_lifetimes),
        "--span", str(args.span),
        "--distractors", str(args.distractors),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--binary-complement-loss",
        "--append-skill-slot",
        "--skill-adapter-width", str(args.skill_adapter_width),
        "--skill-adapter-gate-hidden", str(args.skill_adapter_gate_hidden),
        "--replay-residual-penalty", str(args.replay_residual_penalty),
        "--replay-gate-penalty", str(args.replay_gate_penalty),
        "--replay-logit-penalty", str(args.replay_logit_penalty),
        "--position-augmentation",
        "--target-operation", "complement",
        "--test-operation", "complement",
        "--test-episodes", str(args.private_count),
        "--device", str(args.device),
    ])
    if args.replay_buffer_in is not None:
        command.extend(["--replay-buffer-in", str(args.replay_buffer_in)])
    if args.binary_margin:
        command.extend([
            "--binary-margin-loss", "--binary-margin", str(args.binary_margin)])
    if shuffle:
        command.append("--shuffle-outcomes")
    return command


def _audit_command(
        *, args: argparse.Namespace, candidate: Path, report: Path,
        count: int, seed: int, shuffled: Path | None = None) -> list[str]:
    command = [
        sys.executable, "-m",
        "experiments.unified_cognitive_controller.audit_complement_slot",
        "--parent", str(args.parent),
        "--candidate", str(candidate),
        "--report", str(report),
        "--count", str(count),
        "--seed", str(seed),
        "--span", str(args.span),
        "--distractors", str(args.distractors),
        "--device", str(args.device),
    ]
    if shuffled is not None:
        command.extend(["--shuffled-candidate", str(shuffled)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--arm-seeds", default="93783,93784,93785")
    parser.add_argument("--data-seed", type=int, default=93783)
    parser.add_argument("--private-seed", type=int, default=395000)
    parser.add_argument("--full-seed", type=int, default=495000)
    parser.add_argument("--shuffled-seed", type=int)
    parser.add_argument("--train-lifetimes", type=int, default=1024)
    parser.add_argument("--private-count", type=int, default=256)
    parser.add_argument("--full-count", type=int, default=1024)
    parser.add_argument("--span", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--binary-margin", type=float, default=0.25)
    parser.add_argument("--skill-adapter-width", type=int, default=256)
    parser.add_argument("--skill-adapter-gate-hidden", type=int, default=64)
    parser.add_argument("--replay-buffer-in", type=Path)
    parser.add_argument("--replay-residual-penalty", type=float, default=0.01)
    parser.add_argument("--replay-gate-penalty", type=float, default=0.01)
    parser.add_argument("--replay-logit-penalty", type=float, default=0.01)
    parser.add_argument("--minimum-causal-gain", type=float, default=5.0)
    parser.add_argument("--minimum-complement-accuracy", type=float, default=0.60)
    parser.add_argument(
        "--maximum-retention-drop-points", type=float, default=2.0)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.arm_seeds.split(",") if value)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("population needs at least two unique arm seeds")
    if args.train_lifetimes < 2 or args.train_lifetimes % 2:
        raise ValueError("train-lifetimes must be even and at least two")
    if args.private_count < 2 or args.full_count < 2:
        raise ValueError("audit counts must be at least two")
    if args.minimum_complement_accuracy <= 0.5:
        raise ValueError("minimum complement accuracy must exceed chance")
    if args.maximum_retention_drop_points < 0.0:
        raise ValueError("retention drop must be nonnegative")
    work_dir = args.work_dir or args.report.parent / f"{args.report.stem}_arms"
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    arms: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        tag = f"arm{index}_seed{seed}"
        training_report = work_dir / f"{tag}.json"
        checkpoint = work_dir / f"{tag}.pt"
        _run(_train_command(
            args=args, seed=seed, report=training_report,
            checkpoint=checkpoint))
        private_report = work_dir / f"{tag}_private_audit.json"
        _run(_audit_command(
            args=args, candidate=checkpoint, report=private_report,
            count=args.private_count, seed=args.private_seed + index * 10007))
        private_audit = json.loads(private_report.read_text())
        arms.append({
            "seed": seed,
            "training_report": str(training_report),
            "checkpoint": str(checkpoint),
            "private_report": str(private_report),
            "private": _audit_summary(
                private_audit,
                minimum_causal_gain=args.minimum_causal_gain,
                minimum_complement_accuracy=args.minimum_complement_accuracy,
                maximum_retention_drop_points=args.maximum_retention_drop_points,
                require_reset=False),
        })
    winner = _select_winner(arms)
    report: dict[str, Any] = {
        "schema": "complement-population-selection-v1",
        "claim_boundary": (
            "The learner receives only controller-visible latent features, "
            "opaque attempted actions, and scalar attempted-action outcomes. "
            "Private and full audit labels are verifier-side only."),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
        "arms": arms,
        "selection": {
            "winner_seed": None if winner is None else winner["seed"],
            "private_rejection_reason": (
                None if winner is not None
                else "no arm passed private causal/accuracy/retention/reset gates"),
        },
    }
    if winner is None:
        report["promotion"] = {"promoted": False}
        report["elapsed_seconds"] = time.perf_counter() - started
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        return
    winner_checkpoint = Path(str(winner["checkpoint"]))
    shuffled_checkpoint: Path | None = None
    if args.shuffled_seed is not None:
        shuffled_report = work_dir / f"shuffled_seed{args.shuffled_seed}.json"
        shuffled_checkpoint = work_dir / f"shuffled_seed{args.shuffled_seed}.pt"
        _run(_train_command(
            args=args, seed=args.shuffled_seed, report=shuffled_report,
            checkpoint=shuffled_checkpoint, shuffle=True))
    full_report = work_dir / "winner_full_audit.json"
    _run(_audit_command(
        args=args, candidate=winner_checkpoint, report=full_report,
        count=args.full_count, seed=args.full_seed,
        shuffled=shuffled_checkpoint))
    full_audit = json.loads(full_report.read_text())
    full_summary = _audit_summary(
        full_audit,
        minimum_causal_gain=args.minimum_causal_gain,
        minimum_complement_accuracy=args.minimum_complement_accuracy,
        maximum_retention_drop_points=args.maximum_retention_drop_points,
        require_reset=True)
    shuffled_accuracy = None
    if "shuffled_audit" in full_audit:
        shuffled_accuracy = float(
            full_audit["shuffled_audit"]["complement"]["accuracy"])
    shuffled_ok = shuffled_accuracy is None or shuffled_accuracy <= 0.55
    promoted = bool(full_summary["selection_eligible"] and shuffled_ok)
    if promoted:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(winner_checkpoint, args.checkpoint_out)
    report["full_audit"] = str(full_report)
    report["full_summary"] = full_summary
    report["shuffled_complement_accuracy"] = shuffled_accuracy
    report["promotion"] = {
        "promoted": promoted,
        "checkpoint": str(args.checkpoint_out) if promoted else None,
        "shuffled_ok": shuffled_ok,
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
