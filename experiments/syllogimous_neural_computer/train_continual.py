from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import collate_episodes

from .lifetime import generate_sensory_lifetime
from .model import NeuralComputerAgent, parameter_count
from .train_lifetime import straight_through_admission
from .training_memory import DifferentiableBatchMemory


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def context_streams(start_seed: int, batch: int, contexts: int, delay: int,
                    choices: int, *, heldout: bool = False):
    return [[generate_sensory_lifetime(start_seed + row * contexts + context,
                                      associations=1, delay=delay, choices=choices,
                                      heldout=heldout, contextual=True)
             for context in range(contexts)] for row in range(batch)]


def run_streams(model: NeuralComputerAgent, streams, device: torch.device, *,
                threshold: float, write_cost: float = 0.0,
                intervention: str = "intact") -> tuple[torch.Tensor, dict[str, object]]:
    batch = len(streams)
    contexts = len(streams[0])
    memory = DifferentiableBatchMemory(batch, model.hidden, device=device)
    query_loss = torch.zeros((), device=device)
    interface_loss = torch.zeros((), device=device)
    write_penalty = torch.zeros((), device=device)
    writes = 0.0
    first_correct = [0 for _ in range(contexts)]
    query_episodes = []
    event_count = 0
    for context in range(contexts):
        lifetimes = [stream[context] for stream in streams]
        query_start = 1 + lifetimes[0].delay
        query_episodes.append([lifetime.episodes[-1] for lifetime in lifetimes])
        for step in range(len(lifetimes[0].episodes)):
            items = [lifetime.episodes[step] for lifetime in lifetimes]
            batch_data = collate_episodes(items)
            output = model(batch_data["frames"].to(device), batch_data["pcm"].to(device),
                           batch_data["mask"].to(device), memory)
            targets = batch_data["actions"][:, 0].to(device)
            if step >= query_start:
                logits = output.answer_logits[:, -1]
                query_loss = query_loss + nn.functional.cross_entropy(logits, targets)
                first_correct[context] += int((logits.argmax(-1) == targets).sum())
            else:
                interface_loss = interface_loss + nn.functional.cross_entropy(
                    output.observation_logits[:, 0], targets)
            admissions = straight_through_admission(
                output.write_logits, 1.0, stochastic=False, threshold=threshold)
            write_penalty = write_penalty + admissions.mean()
            writes += float(admissions.detach().mean())
            memory = memory.append(output.write_keys, output.write_values,
                                   output.write_strengths, admissions)
            event_count += 1

    memory = memory.counterfactual(intervention)
    replay_correct = [0 for _ in range(contexts)]
    replay_loss = torch.zeros((), device=device)
    # Re-query every context only after all later contexts have been stored.
    for context, items in enumerate(query_episodes):
        batch_data = collate_episodes(items)
        output = model(batch_data["frames"].to(device), batch_data["pcm"].to(device),
                       batch_data["mask"].to(device), memory)
        targets = batch_data["actions"][:, 0].to(device)
        logits = output.answer_logits[:, -1]
        replay_loss = replay_loss + nn.functional.cross_entropy(logits, targets)
        replay_correct[context] += int((logits.argmax(-1) == targets).sum())
    query_loss = query_loss / contexts
    replay_loss = replay_loss / contexts
    interface_loss = interface_loss / max(1, contexts * (1 + streams[0][0].delay))
    write_penalty = write_penalty / event_count
    loss = query_loss + replay_loss + 0.05 * interface_loss + write_cost * write_penalty
    return loss, {
        "first_accuracy": sum(first_correct) / (batch * contexts),
        "retention_accuracy": sum(replay_correct) / (batch * contexts),
        "first_by_context": [value / batch for value in first_correct],
        "retention_by_context": [value / batch for value in replay_correct],
        "writes_per_stream": writes,
        "stored_rows_per_stream": float(memory.count),
        "query_loss": float(query_loss.detach()),
        "replay_loss": float(replay_loss.detach()),
    }


@torch.no_grad()
def evaluate(model, device, *, samples: int, batch_size: int, contexts: int,
             delay: int, choices: int, threshold: float, intervention: str):
    model.eval()
    totals = {"first_accuracy": 0.0, "retention_accuracy": 0.0,
              "writes_per_stream": 0.0, "stored_rows_per_stream": 0.0}
    first_by_context = [0.0] * contexts
    retention_by_context = [0.0] * contexts
    seen = 0
    for offset in range(0, samples, batch_size):
        count = min(batch_size, samples - offset)
        streams = context_streams(1_000_000 + offset * contexts, count, contexts,
                                  delay, choices, heldout=True)
        _, metrics = run_streams(model, streams, device, threshold=threshold,
                                 intervention=intervention)
        for key in totals:
            totals[key] += float(metrics[key]) * count
        for index in range(contexts):
            first_by_context[index] += metrics["first_by_context"][index] * count
            retention_by_context[index] += metrics["retention_by_context"][index] * count
        seen += count
    result = {key: value / seen for key, value in totals.items()}
    result["first_by_context"] = [value / seen for value in first_by_context]
    result["retention_by_context"] = [value / seen for value in retention_by_context]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-lifetime growing-memory benchmark")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-streams", type=int, default=512)
    parser.add_argument("--eval-streams", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-contexts", type=int, default=4)
    parser.add_argument("--eval-contexts", type=int, default=8)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--read-top-k", type=int, default=8)
    parser.add_argument("--write-cost", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
    config = payload["arguments"]
    model = NeuralComputerAgent(config["hidden"], config["workspace_slots"], config["heads"],
                                config["thought_steps"], config["choices"],
                                read_top_k=args.read_top_k).to(device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if set(incompatible.missing_keys) - {"log_read_scale"} or incompatible.unexpected_keys:
        raise ValueError(f"incompatible initial checkpoint: {incompatible}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        totals = {"first_accuracy": 0.0, "retention_accuracy": 0.0,
                  "writes_per_stream": 0.0}
        seen = 0
        for offset in range(0, args.train_streams, args.batch_size):
            count = min(args.batch_size, args.train_streams - offset)
            streams = context_streams(epoch * args.train_streams * args.train_contexts +
                                      offset * args.train_contexts,
                                      count, args.train_contexts, args.delay, args.choices)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = run_streams(model, streams, device, threshold=args.threshold,
                                        write_cost=args.write_cost)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(metrics[key]) * count
            seen += count
        row = {key: value / seen for key, value in totals.items()}
        row["epoch"] = epoch + 1
        history.append(row)
        print(json.dumps(row), flush=True)
    evaluation = evaluate(model, device, samples=args.eval_streams,
                          batch_size=args.batch_size, contexts=args.eval_contexts,
                          delay=args.delay, choices=args.choices, threshold=args.threshold,
                          intervention="intact")
    counterfactuals = {
        mode: evaluate(model, device, samples=args.eval_streams,
                       batch_size=args.batch_size, contexts=args.eval_contexts,
                       delay=args.delay, choices=args.choices, threshold=args.threshold,
                       intervention=mode)
        for mode in ("empty", "shuffled", "garbage")
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arguments": vars(args),
                "base_config": config}, args.checkpoint)
    report = {"schema": "syllogimous-continual-memory-v1",
              "parameters": parameter_count(model), "history": history,
              "evaluation": evaluation, "counterfactuals": counterfactuals,
              "weights_frozen_during_evaluation": True,
              "training_seconds": time.perf_counter() - started,
              "config": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evaluation": evaluation, "counterfactuals": counterfactuals}),
          flush=True)


if __name__ == "__main__":
    main()
