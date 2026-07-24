from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import PublicEpisode, collate_episodes

from .consolidation import ReplayScore, score_sensory_replay
from .context_selection import ActiveContextSelector
from .memory import PersistentMemory
from .train_consolidation import build_stream, load_controller, seed_everything


@torch.no_grad()
def sensory_query(model, episode: PublicEpisode, device: torch.device) -> torch.Tensor:
    batch = collate_episodes([episode])
    return model.retrieval_summary(batch["frames"].to(device), batch["pcm"].to(device),
                                   batch["mask"].to(device))[0]


@torch.no_grad()
def score_one(model, memory: PersistentMemory, episode: PublicEpisode,
              device: torch.device) -> ReplayScore:
    return score_sensory_replay(model, memory, [episode], device, batch_size=1)


@torch.no_grad()
def oracle_active_context(model, memory: PersistentMemory, episode: PublicEpisode,
                          device: torch.device, context_cost: float
                          ) -> tuple[int, PersistentMemory, ReplayScore]:
    """Find the best empty/single-row context from raw sensory behavior only."""
    candidates = [(0, memory.select([]))]
    indices = memory.valid.nonzero(as_tuple=False).squeeze(1)
    candidates.extend((offset + 1, memory.select([int(index)]))
                      for offset, index in enumerate(indices))
    ranked = []
    for label, candidate in candidates:
        score = score_one(model, candidate, episode, device)
        rows = candidate.count
        rank = (score.correct, -score.loss - context_cost * rows)
        ranked.append((rank, label, candidate, score))
    _, label, candidate, score = max(ranked, key=lambda item: item[0])
    return label, candidate, score


@torch.no_grad()
def active_context_utilities(model, memory: PersistentMemory, episode: PublicEpisode,
                             device: torch.device, context_cost: float) -> torch.Tensor:
    """Dense correctness-first utilities for null and every singleton row."""
    candidates = [memory.select([])]
    indices = memory.valid.nonzero(as_tuple=False).squeeze(1)
    candidates.extend(memory.select([int(index)]) for index in indices)
    utilities = []
    for candidate in candidates:
        score = score_one(model, candidate, episode, device)
        utilities.append(10.0 * score.accuracy - score.loss -
                         context_cost * candidate.count)
    return memory.keys.new_tensor(utilities)


def train_stream(selector, optimizer, model, memory, episodes, device, context_cost,
                 target_temperature):
    losses, correct, nulls = [], 0, 0
    for episode in episodes:
        sensory = sensory_query(model, episode, device)
        utilities = active_context_utilities(
            model, memory, episode, device, context_cost)
        targets = torch.softmax(utilities / target_temperature, dim=0)
        target = int(utilities.argmax())
        logits, _ = selector(sensory, memory)
        losses.append(-(targets * torch.log_softmax(logits, dim=0)).sum())
        correct += int(logits.argmax() == target)
        nulls += int(target == 0)
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach()), correct / len(episodes), nulls / len(episodes)


@torch.no_grad()
def evaluate(selector, model, device, args, *, streams: int, seed: int):
    names = ("empty", "random", "learned", "full", "oracle")
    totals = {f"{view}_{name}_correct": 0 for view in ("query", "audit")
              for name in names}
    totals.update({f"{view}_{name}_rows": 0 for view in ("query", "audit")
                   for name in ("random", "learned", "oracle")})
    total = 0
    learned_nulls = 0
    started = time.perf_counter()
    for stream in range(streams):
        memory, queries, audits = build_stream(
            model, device, seed + stream * args.contexts, args.contexts,
            args.delay, args.choices, args.threshold)
        for view, episodes in (("query", queries), ("audit", audits)):
            for offset, episode in enumerate(episodes):
                sensory = sensory_query(model, episode, device)
                learned, selected = selector.select(sensory, memory)
                generator = random.Random(seed + stream * 10_007 + offset * 101 +
                                          (1 if view == "audit" else 0))
                valid = memory.valid.nonzero(as_tuple=False).squeeze(1).tolist()
                random_memory = (memory.select([]) if selected is None else
                                 memory.select([generator.choice(valid)]))
                _, oracle, oracle_score = oracle_active_context(
                    model, memory, episode, device, args.context_cost)
                candidates = {
                    "empty": score_one(model, memory.select([]), episode, device),
                    "random": score_one(model, random_memory, episode, device),
                    "learned": score_one(model, learned, episode, device),
                    "full": score_one(model, memory, episode, device),
                    "oracle": oracle_score,
                }
                for name, score in candidates.items():
                    totals[f"{view}_{name}_correct"] += score.correct
                totals[f"{view}_random_rows"] += random_memory.count
                totals[f"{view}_learned_rows"] += learned.count
                totals[f"{view}_oracle_rows"] += oracle.count
                learned_nulls += int(selected is None)
                total += 1
    # Each stream contributes equally many original and audit queries.
    per_view = total // 2
    result = {}
    for view in ("query", "audit"):
        for name in names:
            result[f"{view}_{name}_accuracy"] = \
                totals[f"{view}_{name}_correct"] / max(1, per_view)
        for name in ("random", "learned", "oracle"):
            result[f"{view}_{name}_rows"] = \
                totals[f"{view}_{name}_rows"] / max(1, per_view)
    result["learned_null_rate"] = learned_nulls / max(1, total)
    result["evaluation_seconds"] = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a latent long-term context selector")
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-streams", type=int, default=64)
    parser.add_argument("--eval-streams", type=int, default=128)
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--context-cost", type=float, default=0.001)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = load_controller(args.controller, device)
    selector = ActiveContextSelector(model.hidden, args.hidden).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.learning_rate)
    history = []
    started = time.perf_counter()
    for index in range(args.train_streams):
        memory, queries, audits = build_stream(
            model, device, index * args.contexts, args.contexts,
            args.delay, args.choices, args.threshold)
        loss, accuracy, null_rate = train_stream(
            selector, optimizer, model, memory, queries + audits, device,
            args.context_cost, args.target_temperature)
        if (index + 1) % max(1, args.train_streams // 8) == 0:
            row = {"stream": index + 1, "loss": loss,
                   "oracle_label_accuracy": accuracy,
                   "oracle_null_rate": null_rate}
            history.append(row)
            print(json.dumps(row), flush=True)
    evaluation = evaluate(selector, model, device, args, streams=args.eval_streams,
                          seed=1_500_000 + args.seed * 10_000)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"selector": selector.state_dict(), "width": model.hidden,
                "arguments": vars(args)}, args.checkpoint)
    report = {"schema": "syllogimous-active-context-v1",
              "controller_weights_frozen": True,
              "selector_inputs": "sensory latent and latent memory only",
              "maximum_active_rows": 1,
              "history": history, "evaluation": evaluation,
              "training_seconds": time.perf_counter() - started,
              "config": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evaluation": evaluation}), flush=True)


if __name__ == "__main__":
    main()
