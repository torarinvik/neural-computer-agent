#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .games import CAPABILITY_FAMILIES, GAME_NAMES


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def run_one(args, capability: str, seed: int) -> Path:
    latency = f"_latency{args.latency_weight:g}" if args.latency_weight else ""
    vision = f"_vision-{args.vision_streamer}{'-random' if args.randomize_vision else ''}"
    listener_model = f"_{Path(args.llm_model).name}" if args.listener.startswith("llm") else ""
    stem = (f"{args.variant}_{args.listener}{listener_model}_holdout-capability-{capability}_"
            f"seed{seed}{vision}{latency}")
    destination = args.out / f"{stem}.json"
    if destination.exists():
        return destination
    command = [
        sys.executable, "-m", "experiments.sensory_codec.train",
        "--variant", args.variant, "--listener", args.listener,
        "--games", ",".join(GAME_NAMES), "--holdout-capability", capability,
        "--seed", str(seed), "--samples", str(args.samples),
        "--test-samples", str(args.test_samples), "--epochs", str(args.epochs),
        "--listener-epochs", str(args.listener_epochs), "--batch-size", str(args.batch_size),
        "--latency-weight", str(args.latency_weight),
        "--latency-target-ms", str(args.latency_target_ms),
        "--latency-runs", str(args.latency_runs),
        "--rollout-episodes", str(args.rollout_episodes),
        "--rollout-horizon", str(args.rollout_horizon),
        "--llm-model", args.llm_model, "--out", str(args.out),
        "--vision-streamer", args.vision_streamer,
        "--vision-model", args.vision_model,
        "--vision-size", str(args.vision_size),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    if args.randomize_vision:
        command.append("--randomize-vision")
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    (args.out / f"{stem}.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"{stem} failed; see {args.out / f'{stem}.log'}")
    if not destination.exists():
        raise RuntimeError(f"missing result {destination}")
    return destination


def aggregate(paths: list[Path]) -> dict:
    runs = []
    per_game = []
    for path in sorted(paths):
        result = json.loads(path.read_text())
        capability = result["config"]["holdout_capability"]
        trap = result["modality_trap_audit"]
        latency = result["latency_audit"]
        game_rows = []
        for game in result["config"]["test_games"]:
            rollout = result["rollout_audit"][game]
            references = result["rollout_references"][game]
            teacher = references["teacher"]["mean_events"]
            random = references["random"]["mean_events"]
            transfer = ((rollout["mean_events"] - random)
                        / max(1e-9, teacher - random))
            row = {
                "capability": capability,
                "game": game,
                "seed": result["config"]["seed"],
                "action_accuracy": result["test_by_game"][game]["action_accuracy"],
                "danger_accuracy": result["test_by_game"][game]["danger_accuracy"],
                "event_transfer_fraction": transfer,
            }
            per_game.append(row)
            game_rows.append(row)
        runs.append({
            "capability": capability,
            "seed": result["config"]["seed"],
            "action_accuracy": result["test"]["action_accuracy"],
            "event_transfer_fraction": float(np.mean(
                [row["event_transfer_fraction"] for row in game_rows])),
            "corruption_accuracy": trap["corruption_mean_accuracy"],
            "stale_accuracy": trap["stale_mean_accuracy"],
            "missing_entropy_delta": trap["missing_entropy_delta"],
            "mean_latency_ms": latency.get("mean_ms"),
        })

    def summarize(rows: list[dict], group_key: str) -> dict:
        output = {}
        for group in sorted({row[group_key] for row in rows}):
            selected = [row for row in rows if row[group_key] == group]
            metrics = [key for key in selected[0] if key not in
                       (group_key, "capability", "game", "seed")]
            output[group] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in selected])),
                    "std": float(np.std([row[metric] for row in selected])),
                }
                for metric in metrics if selected[0][metric] is not None
            }
        return output

    return {
        "runs": runs,
        "per_game_runs": per_game,
        "by_capability": summarize(runs, "capability"),
        "by_game": summarize(per_game, "game"),
    }


def print_summary(summary: dict) -> None:
    print("capability            action      transfer    corrupt     stale       missing-H")
    for capability, values in summary["by_capability"].items():
        def cell(name: str) -> str:
            value = values[name]
            return f"{value['mean']:.3f}±{value['std']:.3f}"
        print(f"{capability:<21} {cell('action_accuracy')} {cell('event_transfer_fraction')} "
              f"{cell('corruption_accuracy')} {cell('stale_accuracy')} "
              f"{cell('missing_entropy_delta')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run capability-family holdout exams")
    parser.add_argument("--capabilities", type=parse_csv,
                        default=tuple(CAPABILITY_FAMILIES))
    parser.add_argument("--seeds", type=parse_csv, default=("7", "17", "29"))
    parser.add_argument("--variant", choices=("gameplay", "grounded", "compressed"),
                        default="compressed")
    parser.add_argument("--listener", choices=("grounded", "random", "direct", "llm", "llm_random"),
                        default="grounded")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--listener-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latency-weight", type=float, default=0.002)
    parser.add_argument("--latency-target-ms", type=float, default=50.0)
    parser.add_argument("--latency-runs", type=int, default=64)
    parser.add_argument("--rollout-episodes", type=int, default=30)
    parser.add_argument("--rollout-horizon", type=int, default=250)
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    parser.add_argument("--vision-streamer", choices=("tiny", "smol"), default="tiny")
    parser.add_argument("--vision-model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--vision-size", type=int, default=128)
    parser.add_argument("--randomize-vision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sensory_codec_capabilities"))
    args = parser.parse_args()
    unknown = set(args.capabilities) - set(CAPABILITY_FAMILIES)
    if unknown:
        parser.error(f"unknown capabilities: {', '.join(sorted(unknown))}")
    seeds = tuple(int(seed) for seed in args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(capability, seed) for capability in args.capabilities for seed in seeds]
    paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, args, capability, seed): (capability, seed)
                   for capability, seed in jobs}
        for future in concurrent.futures.as_completed(futures):
            capability, seed = futures[future]
            paths.append(future.result())
            print(f"completed capability={capability} seed={seed}", flush=True)
    summary = aggregate(paths)
    destination = args.out / f"capability_matrix_{args.variant}_{args.listener}.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
