#!/usr/bin/env python3
"""Run comparable packet-only model evaluations for configured size slots.

The size labels are experiment slots, not claims about a checkpoint's actual
parameter count.  A slot is evaluated only when a model id/path is supplied;
otherwise the output records ``unassigned`` instead of silently fabricating a
baseline.  Every assigned slot uses the same causal episode runner and emits
the same accuracy/latency/timeout schema.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .model_sizes import MODEL_SLOTS
from .run_vlm_episode import run
from .vlm_policy import SmolVLMPolicy
from .evaluation import summarize


def _slot_map(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("model map must be a JSON object mapping slot names to model ids")
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[str(key)] = {"model": value}
        elif isinstance(value, dict) and value.get("model"):
            result[str(key)] = dict(value)
        else:
            raise ValueError(f"model entry for {key!r} must be a string or object with model")
    return result


def evaluate_slot(slot: dict, entry: dict, args: argparse.Namespace) -> dict:
    model_name = entry["model"]
    started = time.perf_counter()
    model = SmolVLMPolicy.from_pretrained(
        model_name, device=args.device, local_files_only=args.local_files_only,
        image_size=args.image_size, max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )
    rows, traces = run(
        model, episodes=args.episodes, premises=args.premises,
        deadline_ms=args.deadline_ms, seed=args.seed,
    )
    result = summarize(rows)
    result.update({
        "slot": slot["name"],
        "nominal_parameters": slot["parameters"],
        "model": model_name,
        "actual_parameters": entry.get("actual_parameters"),
        "modality": entry.get("modality", "vision+text"),
        "parameter_label_matches_slot": (
            entry.get("actual_parameters") is None or
            abs(entry["actual_parameters"] - slot["parameters"]) <= slot["parameters"] * 0.1
        ),
        "hardware": str(model.device),
        "wall_seconds": time.perf_counter() - started,
        "action_traces": traces,
        "status": "evaluated",
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=None,
                        help="JSON object, e.g. {\"350m\": \"HuggingFaceTB/...\"}")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--premises", type=int, default=2)
    parser.add_argument("--deadline-ms", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("model_matrix.json"))
    args = parser.parse_args()
    mapping = _slot_map(args.models)
    results = []
    for slot in MODEL_SLOTS:
        name = slot["name"]
        entry = mapping.get(name)
        if not entry:
            results.append({"slot": name, "nominal_parameters": slot["parameters"],
                            "status": "unassigned"})
            continue
        try:
            results.append(evaluate_slot(slot, entry, args))
        except Exception as exc:  # preserve the matrix even when one slot is unavailable
            results.append({"slot": name, "nominal_parameters": slot["parameters"],
                            "model": entry["model"], "status": "unavailable",
                            "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "schema": "syllogimous.model-matrix.v1",
        "protocol": {
            "packet_only": True,
            "episodes": args.episodes,
            "premises": args.premises,
            "deadline_ms": args.deadline_ms,
            "seed": args.seed,
        },
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output),
                      "evaluated": sum(x["status"] == "evaluated" for x in results),
                      "unassigned": sum(x["status"] == "unassigned" for x in results),
                      "unavailable": sum(x["status"] == "unavailable" for x in results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
