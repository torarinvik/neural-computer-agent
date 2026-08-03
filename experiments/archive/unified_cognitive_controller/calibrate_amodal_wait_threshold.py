"""Choose a wait threshold from verifier utility on a training split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .amodal_wait_policy import AmodalArrivalPredictor
from .audit_amodal_adaptive_wait import (
    _load_predictor,
    _load_runtime,
    _prepare,
    _rollout,
    _sample_delays,
)
from .environment import generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=993001)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--deadline", type=int, default=2)
    parser.add_argument("--latency-cost", type=float, default=0.03)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30),
    )
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.count < 4 or args.deadline < 1 or args.latency_cost <= 0:
        raise ValueError("invalid calibration configuration")
    if any(not 0.0 < threshold < 1.0 for threshold in args.thresholds):
        raise ValueError("all thresholds must lie in (0, 1)")
    device = torch.device(args.device)
    runtime, _ = _load_runtime(args.controller, args.input_bus, args.audio, device)
    predictor = _load_predictor(args.predictor, device)
    candidates: list[dict[str, object]] = []
    for threshold in args.thresholds:
        adaptive_rows = []
        fixed_rows = []
        for appearance_index, appearance in enumerate(("bars", "diamonds", "dot_pairs")):
            batch = generate_lifetimes(
                args.count,
                6,
                seed=args.seed + appearance_index * 10_000,
                heldout=False,
                task="pair_relation",
                appearance=appearance,
                support_trials=1,
                device=device,
            )
            first, second = split_complementary_views(batch.frames)
            delays, _ = _sample_delays(
                args.count,
                seed=args.seed + appearance_index * 10_000 + 1,
                device=device,
            )
            encoded_a, encoded_b = _prepare(runtime, first, second)
            adaptive_rows.append(
                _rollout(
                    runtime,
                    predictor,
                    encoded_a,
                    encoded_b,
                    batch.correct_actions,
                    delays,
                    mode="adaptive",
                    threshold=threshold,
                    deadline=args.deadline,
                    latency_cost=args.latency_cost,
                )
            )
            fixed_rows.append(
                _rollout(
                    runtime,
                    predictor,
                    encoded_a,
                    encoded_b,
                    batch.correct_actions,
                    delays,
                    mode="fixed2",
                    threshold=threshold,
                    deadline=args.deadline,
                    latency_cost=args.latency_cost,
                )
            )
        candidates.append(
            {
                "threshold": threshold,
                "adaptive_accuracy": sum(row["accuracy"] for row in adaptive_rows) / 3,
                "adaptive_latency": sum(row["mean_latency"] for row in adaptive_rows) / 3,
                "adaptive_utility": sum(row["verified_utility"] for row in adaptive_rows) / 3,
                "fixed2_accuracy": sum(row["accuracy"] for row in fixed_rows) / 3,
                "fixed2_latency": sum(row["mean_latency"] for row in fixed_rows) / 3,
                "fixed2_utility": sum(row["verified_utility"] for row in fixed_rows) / 3,
            }
        )
    eligible = [
        row
        for row in candidates
        if row["adaptive_accuracy"] >= row["fixed2_accuracy"] - 0.02
        and row["adaptive_latency"] < row["fixed2_latency"]
    ]
    selected = max(
        eligible or candidates,
        key=lambda row: (row["adaptive_utility"], -row["adaptive_latency"]),
    )
    report = {
        "schema": "amodal-wait-threshold-calibration-v1",
        "claim": "Threshold selected from verifier utility on a non-held-out timing split.",
        "labels_used": [],
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "audio": str(args.audio),
        "audio_sha256": _sha256(args.audio),
        "predictor": str(args.predictor),
        "predictor_sha256": _sha256(args.predictor),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "deadline": args.deadline,
            "latency_cost": args.latency_cost,
            "thresholds": list(args.thresholds),
            "device": str(device),
        },
        "selected_threshold": selected["threshold"],
        "selected_candidate": selected,
        "candidates": candidates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_threshold": selected["threshold"], "utility": selected["adaptive_utility"]}))


if __name__ == "__main__":
    main()
