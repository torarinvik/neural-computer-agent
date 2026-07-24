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
from .training_memory import DifferentiableBatchMemory


CONDITIONS = ("no_memory", "random_write", "learned_memory")
ADMISSION_MODES = ("soft", "stochastic_hard", "deterministic_hard")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lifetime_batch(start_seed: int, batch_size: int, associations: int,
                   delay: int, choices: int, heldout: bool = False):
    return [generate_sensory_lifetime(start_seed + index, associations=associations,
                                     delay=delay, choices=choices, heldout=heldout)
            for index in range(batch_size)]


def straight_through_admission(logits: torch.Tensor, temperature: float,
                               *, stochastic: bool,
                               threshold: float = 0.5) -> torch.Tensor:
    """Binary forward decisions with a relaxed backward derivative."""
    if temperature <= 0:
        raise ValueError("admission temperature must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("admission threshold must be strictly between zero and one")
    perturbed = logits
    if stochastic:
        uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
        perturbed = logits + torch.log(uniform) - torch.log1p(-uniform)
    relaxed = torch.sigmoid(perturbed / temperature)
    hard = (relaxed >= threshold).to(relaxed.dtype)
    return hard.detach() - relaxed.detach() + relaxed


def run_lifetimes(model: NeuralComputerAgent, lifetimes, condition: str,
                  device: torch.device, *, training: bool,
                  write_cost: float = 0.0,
                  memory_intervention: str = "intact",
                  admission_mode: str = "soft",
                  admission_temperature: float = 1.0,
                  admission_threshold: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if admission_mode not in ADMISSION_MODES:
        raise ValueError(f"unknown admission mode {admission_mode!r}")
    batch_size = len(lifetimes)
    associations = lifetimes[0].associations
    delay = lifetimes[0].delay
    query_start = associations + delay
    memory = DifferentiableBatchMemory(batch_size, model.hidden, device=device)
    query_loss = torch.zeros((), device=device)
    interface_loss = torch.zeros((), device=device)
    write_penalty = torch.zeros((), device=device)
    correct = queries = 0
    write_strength_total = 0.0
    for step in range(len(lifetimes[0].episodes)):
        if step == query_start:
            memory = memory.counterfactual(memory_intervention)
        batch = collate_episodes([lifetime.episodes[step] for lifetime in lifetimes])
        frames = batch["frames"].to(device)
        pcm = batch["pcm"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["actions"][:, 0].to(device)
        active_memory = (memory if condition != "no_memory" else
                         DifferentiableBatchMemory(batch_size, model.hidden, device=device))
        output = model(frames, pcm, mask, active_memory)
        if step >= query_start:
            logits = output.answer_logits[:, -1]
            query_loss = query_loss + nn.functional.cross_entropy(logits, targets)
            correct += int((logits.argmax(-1) == targets).sum())
            queries += batch_size
        else:
            interface_loss = interface_loss + nn.functional.cross_entropy(
                output.observation_logits[:, 0], targets)
        if condition == "learned_memory":
            priorities = output.write_strengths
            if admission_mode == "soft":
                admissions = torch.ones_like(output.write_strengths)
            else:
                admissions = straight_through_admission(
                    output.write_logits, admission_temperature,
                    stochastic=admission_mode == "stochastic_hard",
                    threshold=admission_threshold)
        elif condition == "random_write":
            priorities = torch.rand_like(output.write_strengths)
            admissions = torch.ones_like(output.write_strengths)
        else:
            priorities = torch.zeros_like(output.write_strengths)
            admissions = torch.zeros_like(output.write_strengths)
        # Accuracy is primary. The cost can be enabled only after the task is
        # demonstrably learned, matching the project's latency-reward policy.
        write_penalty = write_penalty + admissions.mean()
        write_strength_total += float(admissions.detach().mean())
        if condition != "no_memory":
            memory = memory.append(output.write_keys, output.write_values,
                                   priorities, admissions)
    query_loss = query_loss / max(1, associations)
    interface_loss = interface_loss / max(1, query_start)
    write_penalty = write_penalty / len(lifetimes[0].episodes)
    loss = query_loss + 0.05 * interface_loss + write_cost * write_penalty
    return loss, {
        "accuracy": correct / max(1, queries),
        "query_loss": float(query_loss.detach()),
        "interface_loss": float(interface_loss.detach()),
        "mean_write_strength": write_strength_total / len(lifetimes[0].episodes),
        "writes_per_lifetime": write_strength_total,
        "entries_per_lifetime": float(memory.count),
    }


@torch.no_grad()
def evaluate(model: NeuralComputerAgent, condition: str, device: torch.device, *,
             samples: int, batch_size: int, associations: int, delay: int,
             choices: int, seed: int,
             memory_intervention: str = "intact",
             admission_mode: str = "deterministic_hard",
             admission_temperature: float = 1.0,
             admission_threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for offset in range(0, samples, batch_size):
        count = min(batch_size, samples - offset)
        lifetimes = lifetime_batch(seed + offset, count, associations, delay,
                                   choices, heldout=True)
        _, metrics = run_lifetimes(model, lifetimes, condition, device, training=False,
                                   memory_intervention=memory_intervention,
                                   admission_mode=admission_mode,
                                   admission_temperature=admission_temperature,
                                   admission_threshold=admission_threshold)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * count
        batches += count
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the growing-memory lifetime benchmark")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=2000)
    parser.add_argument("--eval-lifetimes", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--associations", type=int, default=2)
    parser.add_argument("--delay", type=int, default=8)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--workspace-slots", type=int, default=12)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--thought-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--write-cost", type=float, default=0.0)
    parser.add_argument("--write-cost-threshold", type=float, default=0.7,
                        help="enable storage cost after training accuracy reaches this level")
    parser.add_argument("--admission-mode", choices=ADMISSION_MODES,
                        default="stochastic_hard")
    parser.add_argument("--admission-temperature", type=float, default=1.0)
    parser.add_argument("--admission-threshold", type=float, default=0.05)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.train_lifetimes < 1 or args.eval_lifetimes < 1 or args.batch_size < 1:
        raise ValueError("sample and batch counts must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = NeuralComputerAgent(args.hidden, args.workspace_slots, args.heads,
                                args.thought_steps, args.choices).to(device)
    if args.initial_checkpoint is not None:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(initial["model"], strict=False)
        if set(incompatible.missing_keys) - {"log_read_scale"} or incompatible.unexpected_keys:
            raise ValueError(f"incompatible initial checkpoint: {incompatible}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    cost_active = False
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        totals: dict[str, float] = {}
        seen = 0
        for offset in range(0, args.train_lifetimes, args.batch_size):
            count = min(args.batch_size, args.train_lifetimes - offset)
            lifetimes = lifetime_batch(epoch * args.train_lifetimes + offset, count,
                                       args.associations, args.delay, args.choices)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = run_lifetimes(model, lifetimes, args.condition, device,
                                          training=True,
                                          write_cost=args.write_cost if cost_active else 0.0,
                                          admission_mode=args.admission_mode,
                                          admission_temperature=args.admission_temperature,
                                          admission_threshold=args.admission_threshold)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
            seen += count
        epoch_metrics = {key: value / seen for key, value in totals.items()}
        epoch_metrics["epoch"] = epoch + 1
        epoch_metrics["write_cost_active"] = cost_active
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        cost_active = cost_active or epoch_metrics["accuracy"] >= args.write_cost_threshold
    evaluation = evaluate(model, args.condition, device, samples=args.eval_lifetimes,
                          batch_size=args.batch_size, associations=args.associations,
                          delay=args.delay, choices=args.choices, seed=1_000_000,
                          admission_mode="deterministic_hard",
                          admission_temperature=args.admission_temperature,
                          admission_threshold=args.admission_threshold)
    counterfactuals = {}
    if args.condition == "learned_memory":
        for intervention in ("empty", "shuffled", "garbage"):
            counterfactuals[intervention] = evaluate(
                model, args.condition, device, samples=args.eval_lifetimes,
                batch_size=args.batch_size, associations=args.associations,
                delay=args.delay, choices=args.choices, seed=1_000_000,
                memory_intervention=intervention,
                admission_mode="deterministic_hard",
                admission_temperature=args.admission_temperature,
                admission_threshold=args.admission_threshold)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arguments": vars(args)}, args.checkpoint)
    report = {
        "schema": "syllogimous-growing-memory-benchmark-v1",
        "condition": args.condition, "parameters": parameter_count(model),
        "history": history, "evaluation": evaluation,
        "memory_counterfactuals": counterfactuals,
        "training_seconds": time.perf_counter() - started,
        "sensory_only": True, "persistent_memory_grows": args.condition != "no_memory",
        "config": {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in vars(args).items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), "evaluation": evaluation,
                      "memory_counterfactuals": counterfactuals}), flush=True)


if __name__ == "__main__":
    main()
