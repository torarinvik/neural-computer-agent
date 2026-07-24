#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .games import GAME_NAMES


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def run_one(python: str, out: Path, variant: str, listener: str, holdout: str,
            seed: int, samples: int, test_samples: int, epochs: int,
            listener_epochs: int, batch_size: int, dagger_rounds: int,
            dagger_samples: int, dagger_epochs: int, latency_weight: float,
            latency_target_ms: float, latency_runs: int, llm_model: str,
            vision_streamer: str, vision_model: str, vision_size: int,
            randomize_vision: bool, local_files_only: bool) -> Path:
    dagger = f"_dagger{dagger_rounds}" if dagger_rounds else ""
    latency = f"_latency{latency_weight:g}" if latency_weight else ""
    vision = f"_vision-{vision_streamer}{'-random' if randomize_vision else ''}"
    listener_model = f"_{Path(llm_model).name}" if listener.startswith("llm") else ""
    stem = (f"{variant}_{listener}{listener_model}_holdout-{holdout}_seed{seed}"
            f"{vision}{dagger}{latency}")
    result_path = out / f"{stem}.json"
    if result_path.exists():
        return result_path
    command = [
        python, "-m", "experiments.sensory_codec.train",
        "--variant", variant, "--listener", listener,
        "--games", ",".join(GAME_NAMES), "--holdout-game", holdout,
        "--seed", str(seed), "--samples", str(samples),
        "--test-samples", str(test_samples), "--epochs", str(epochs),
        "--listener-epochs", str(listener_epochs), "--batch-size", str(batch_size),
        "--dagger-rounds", str(dagger_rounds), "--dagger-samples", str(dagger_samples),
        "--dagger-epochs", str(dagger_epochs),
        "--latency-weight", str(latency_weight),
        "--latency-target-ms", str(latency_target_ms),
        "--latency-runs", str(latency_runs),
        "--llm-model", llm_model,
        "--vision-streamer", vision_streamer,
        "--vision-model", vision_model,
        "--vision-size", str(vision_size),
        "--out", str(out),
    ]
    if randomize_vision:
        command.append("--randomize-vision")
    if local_files_only:
        command.append("--local-files-only")
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    (out / f"{stem}.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"{stem} failed; see {out / f'{stem}.log'}")
    if not result_path.exists():
        raise RuntimeError(f"{stem} completed without {result_path}")
    return result_path


def aggregate(paths: list[Path], variant: str, listener: str) -> dict:
    rows = []
    for path in sorted(paths):
        result = json.loads(path.read_text())
        holdout = result["config"]["holdout_game"]
        rollout = result["rollout_audit"][holdout]
        teacher = result["rollout_references"][holdout]["teacher"]
        random = result["rollout_references"][holdout]["random"]
        test = result["test_by_game"][holdout]
        rows.append({
            "holdout": holdout,
            "seed": result["config"]["seed"],
            "action_accuracy": test["action_accuracy"],
            "horizontal_accuracy": test["horizontal_accuracy"],
            "vertical_accuracy": test["vertical_accuracy"],
            "danger_accuracy": test["danger_accuracy"],
            "event_accuracy": test["ate_accuracy"],
            "mean_steps": rollout["mean_steps"],
            "mean_events": rollout["mean_events"],
            "teacher_events": teacher["mean_events"],
            "random_events": random["mean_events"],
            "event_transfer_fraction": (
                (rollout["mean_events"] - random["mean_events"])
                / max(1e-9, teacher["mean_events"] - random["mean_events"])
            ),
        })
    by_game = {}
    metric_names = [key for key in rows[0] if key not in ("holdout", "seed")]
    for game in GAME_NAMES:
        selected = [row for row in rows if row["holdout"] == game]
        if not selected:
            continue
        by_game[game] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "std": float(np.std([row[metric] for row in selected])),
            }
            for metric in metric_names
        }
    return {"variant": variant, "listener": listener, "runs": rows, "by_game": by_game}


def print_summary(summary: dict) -> None:
    print("holdout     action      danger      events     transfer")
    for game, values in summary["by_game"].items():
        action = values["action_accuracy"]
        danger = values["danger_accuracy"]
        events = values["mean_events"]
        transfer = values["event_transfer_fraction"]
        print(f"{game:<10} {action['mean']:.3f}±{action['std']:.3f} "
              f"{danger['mean']:.3f}±{danger['std']:.3f} "
              f"{events['mean']:.2f}±{events['std']:.2f} "
              f"{transfer['mean']:.3f}±{transfer['std']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate the sensory-codec holdout matrix")
    parser.add_argument("--variant", choices=("gameplay", "grounded", "compressed"), default="compressed")
    parser.add_argument("--listener", choices=("grounded", "random", "direct", "llm", "llm_random"), default="grounded")
    parser.add_argument("--holdouts", type=parse_csv, default=GAME_NAMES)
    parser.add_argument("--seeds", type=parse_csv, default=("7", "17", "29"))
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--test-samples", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--listener-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-samples", type=int, default=2000)
    parser.add_argument("--dagger-epochs", type=int, default=4)
    parser.add_argument("--latency-weight", type=float, default=0.002)
    parser.add_argument("--latency-target-ms", type=float, default=50.0)
    parser.add_argument("--latency-runs", type=int, default=64)
    parser.add_argument("--llm-model", default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    parser.add_argument("--vision-streamer", choices=("tiny", "smol"), default="tiny")
    parser.add_argument("--vision-model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--vision-size", type=int, default=128)
    parser.add_argument("--randomize-vision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("/tmp/sensory_codec_matrix"))
    args = parser.parse_args()
    unknown = set(args.holdouts) - set(GAME_NAMES)
    if unknown:
        parser.error(f"unknown holdouts: {', '.join(sorted(unknown))}")
    seeds = tuple(int(seed) for seed in args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(holdout, seed) for holdout in args.holdouts for seed in seeds]
    paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_one, sys.executable, args.out, args.variant, args.listener,
                            holdout, seed, args.samples, args.test_samples, args.epochs,
                            args.listener_epochs, args.batch_size, args.dagger_rounds,
                            args.dagger_samples, args.dagger_epochs, args.latency_weight,
                            args.latency_target_ms, args.latency_runs, args.llm_model,
                            args.vision_streamer, args.vision_model, args.vision_size,
                            args.randomize_vision, args.local_files_only): (holdout, seed)
            for holdout, seed in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            holdout, seed = futures[future]
            paths.append(future.result())
            print(f"completed holdout={holdout} seed={seed}", flush=True)
    summary = aggregate(paths, args.variant, args.listener)
    dagger = f"_dagger{args.dagger_rounds}" if args.dagger_rounds else ""
    latency = f"_latency{args.latency_weight:g}" if args.latency_weight else ""
    vision = f"_vision-{args.vision_streamer}{'-random' if args.randomize_vision else ''}"
    listener_model = (f"_{Path(args.llm_model).name}"
                      if args.listener.startswith("llm") else "")
    destination = args.out / (f"matrix_{args.variant}_{args.listener}{listener_model}"
                              f"{vision}{dagger}{latency}.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
