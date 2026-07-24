from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .consolidation import ReplayScore
from .context_selection import ActiveContextSelector
from .train_consolidation import build_stream, load_controller, seed_everything
from .train_context_selector import score_one, sensory_query


@dataclass(frozen=True)
class ContextDecision:
    view: str
    confidence: float
    full: ReplayScore
    selected: ReplayScore
    random: ReplayScore
    empty: ReplayScore
    full_rows: int
    selected_rows: int


@torch.no_grad()
def collect_decisions(selector, model, device, args, *, streams: int, seed: int):
    decisions = []
    for stream in range(streams):
        memory, queries, audits = build_stream(
            model, device, seed + stream * args.contexts, args.contexts,
            args.delay, args.choices, args.threshold)
        valid = memory.valid.nonzero(as_tuple=False).squeeze(1).tolist()
        for view, episodes in (("query", queries), ("audit", audits)):
            for offset, episode in enumerate(episodes):
                sensory = sensory_query(model, episode, device)
                logits, indices = selector(sensory, memory)
                probabilities = torch.softmax(logits, dim=0)
                choice = int(probabilities.argmax())
                confidence = float(probabilities[choice])
                selected = (memory.select([]) if choice == 0 else
                            memory.select([int(indices[choice - 1])]))
                generator = random.Random(seed + stream * 10_007 + offset * 101 +
                                          (1 if view == "audit" else 0))
                random_memory = (memory.select([]) if choice == 0 else
                                 memory.select([generator.choice(valid)]))
                empty = memory.select([])
                decisions.append(ContextDecision(
                    view, confidence, score_one(model, memory, episode, device),
                    score_one(model, selected, episode, device),
                    score_one(model, random_memory, episode, device),
                    score_one(model, empty, episode, device),
                    memory.count, selected.count))
    return decisions


def summarize(decisions, threshold: float, *, split: int | None = None):
    names = ("full", "selected", "adaptive", "matched_random", "empty")
    result = {}
    for view in ("query", "audit"):
        rows = [item for index, item in enumerate(decisions)
                if item.view == view and (split is None or index % 2 == split)]
        for name in names:
            correct = loss = active_rows = 0.0
            for item in rows:
                use_small = item.confidence >= threshold and item.selected_rows > 0
                if name == "full":
                    score, count = item.full, item.full_rows
                elif name == "selected":
                    score, count = item.selected, item.selected_rows
                elif name == "adaptive":
                    score = item.selected if use_small else item.full
                    count = item.selected_rows if use_small else item.full_rows
                elif name == "matched_random":
                    score = item.random if use_small else item.full
                    count = item.selected_rows if use_small else item.full_rows
                else:
                    score, count = item.empty, 0
                correct += score.correct
                loss += score.loss
                active_rows += count
            result[f"{view}_{name}_accuracy"] = correct / max(1, len(rows))
            result[f"{view}_{name}_loss"] = loss / max(1, len(rows))
            result[f"{view}_{name}_rows"] = active_rows / max(1, len(rows))
    result["threshold"] = threshold
    return result


def is_safe(result):
    return all(
        result[f"{view}_adaptive_accuracy"] >= result[f"{view}_full_accuracy"] and
        result[f"{view}_adaptive_loss"] <= result[f"{view}_full_loss"]
        for view in ("query", "audit"))


def calibrate(decisions):
    confidences = sorted({item.confidence for item in decisions
                          if item.selected_rows > 0})
    candidates = confidences + [1.01]
    trials = []
    selected = 1.01
    for threshold in candidates:
        halves = [summarize(decisions, threshold, split=split) for split in (0, 1)]
        combined = summarize(decisions, threshold)
        useful = combined["query_adaptive_rows"] < combined["query_full_rows"]
        safe = useful and all(is_safe(half) for half in halves)
        trials.append({"threshold": threshold, "safe": safe, "useful": useful,
                       "query_rows": combined["query_adaptive_rows"],
                       "audit_rows": combined["audit_adaptive_rows"]})
        if safe:
            selected = threshold
            break
    return selected, trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a variable active-memory context")
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration-streams", type=int, default=128)
    parser.add_argument("--eval-streams", type=int, default=256)
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = load_controller(args.controller, device)
    payload = torch.load(args.selector, map_location=device, weights_only=False)
    hidden = int(payload["arguments"]["hidden"])
    selector = ActiveContextSelector(model.hidden, hidden).to(device)
    selector.load_state_dict(payload["selector"])
    selector.eval()
    started = time.perf_counter()
    calibration_decisions = collect_decisions(
        selector, model, device, args, streams=args.calibration_streams,
        seed=1_200_000 + args.seed * 10_000)
    selected_threshold, trials = calibrate(calibration_decisions)
    calibration = summarize(calibration_decisions, selected_threshold)
    evaluation_decisions = collect_decisions(
        selector, model, device, args, streams=args.eval_streams,
        seed=1_800_000 + args.seed * 10_000)
    evaluation = summarize(evaluation_decisions, selected_threshold)
    report = {"schema": "syllogimous-adaptive-active-context-v1",
              "controller_weights_frozen": True,
              "selector_inputs": "sensory latent and latent memory only",
              "calibration_splits": 2,
              "selected_threshold": selected_threshold,
              "calibration_trials": trials,
              "calibration": calibration, "evaluation": evaluation,
              "elapsed_seconds": time.perf_counter() - started,
              "config": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_threshold": selected_threshold,
                      "evaluation": evaluation}), flush=True)


if __name__ == "__main__":
    main()
