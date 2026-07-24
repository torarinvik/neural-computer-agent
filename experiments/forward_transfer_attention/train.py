from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import PublicEpisode, collate_episodes
from experiments.syllogimous_neural_computer.model import NeuralComputerAgent, parameter_count
from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory

from .environment import AttentionTransferLifetime, SHOTS, generate_attention_lifetime


CONTROLS = ("inherited", "composed_latest", "transactional_latest", "empty",
            "shuffled", "unrelated", "garbage")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lifetime_batch(start_seed: int, size: int, *, heldout: bool = False,
                   query_count: int = 4) -> list[AttentionTransferLifetime]:
    return [generate_attention_lifetime(start_seed + index, heldout=heldout,
                                        query_count=query_count)
            for index in range(size)]


def _forward(model: NeuralComputerAgent, episodes: list[PublicEpisode],
             memory: DifferentiableBatchMemory, device: torch.device,
             event_memory: DifferentiableBatchMemory | None = None):
    batch = collate_episodes(episodes)
    output = model(batch["frames"].to(device), batch["pcm"].to(device),
                   batch["mask"].to(device), memory, event_memory=event_memory)
    targets = batch["actions"][:, 0].to(device)
    return output, targets


def _append(model: NeuralComputerAgent, episodes: list[PublicEpisode],
            memory: DifferentiableBatchMemory, device: torch.device
            ) -> tuple[DifferentiableBatchMemory, torch.Tensor, float]:
    output, targets = _forward(model, episodes, memory, device)
    interface_loss = nn.functional.cross_entropy(output.observation_logits[:, 0], targets)
    admissions = torch.ones_like(output.write_strengths)
    memory = memory.append(output.write_keys, output.write_values,
                           output.write_strengths, admissions)
    return memory, interface_loss, float(output.write_strengths.detach().mean())


def _control(memory: DifferentiableBatchMemory, mode: str) -> DifferentiableBatchMemory:
    if mode in ("inherited", "composed_latest", "transactional_latest"):
        return memory
    if mode == "empty":
        return DifferentiableBatchMemory(memory.batch, memory.width, device=memory.device)
    if mode == "shuffled":
        if memory.batch < 2:
            raise ValueError("shuffled control requires batch size greater than one")
        return DifferentiableBatchMemory(
            memory.batch, memory.width, device=memory.device, keys=memory.keys,
            values=memory.values.roll(1, dims=0), strengths=memory.strengths,
            admissions=memory.admissions)
    if mode == "unrelated":
        if memory.batch < 2:
            raise ValueError("unrelated control requires batch size greater than one")
        return DifferentiableBatchMemory(
            memory.batch, memory.width, device=memory.device,
            keys=memory.keys.roll(1, dims=0), values=memory.values.roll(1, dims=0),
            strengths=memory.strengths.roll(1, dims=0),
            admissions=memory.admissions.roll(1, dims=0))
    if mode == "garbage":
        index = torch.arange(memory.keys.numel(), device=memory.device,
                             dtype=memory.keys.dtype).reshape_as(memory.keys)
        return DifferentiableBatchMemory(
            memory.batch, memory.width, device=memory.device,
            keys=torch.sin(index * 1.618), values=torch.cos((index + 0.5) * 2.414),
            strengths=memory.strengths, admissions=memory.admissions)
    raise ValueError(f"unknown control {mode!r}")


def _latest_row(memory: DifferentiableBatchMemory) -> DifferentiableBatchMemory:
    if memory.count == 0:
        return memory
    return DifferentiableBatchMemory(
        memory.batch, memory.width, device=memory.device,
        keys=memory.keys[:, -1:], values=memory.values[:, -1:],
        strengths=memory.strengths[:, -1:], admissions=memory.admissions[:, -1:])


def _loss_per_lifetime(model, episode_groups, memory, device):
    losses = torch.zeros(memory.batch, device=device)
    for episodes in episode_groups:
        output, targets = _forward(model, episodes, memory, device)
        losses = losses + nn.functional.cross_entropy(
            output.answer_logits[:, -1], targets, reduction="none")
    return losses


def _transactional_latest(model, episode_groups, previous, candidate, device):
    before = _loss_per_lifetime(model, episode_groups, previous, device)
    after = _loss_per_lifetime(model, episode_groups, candidate, device)
    accept = after <= before + 1e-7
    row_mask = accept[:, None, None]
    scalar_mask = accept[:, None]
    selected = DifferentiableBatchMemory(
        previous.batch, previous.width, device=previous.device,
        keys=torch.where(row_mask, candidate.keys, previous.keys),
        values=torch.where(row_mask, candidate.values, previous.values),
        strengths=torch.where(scalar_mask, candidate.strengths, previous.strengths),
        admissions=torch.where(scalar_mask, candidate.admissions, previous.admissions))
    return selected, float(accept.to(torch.float32).mean())


def _score_queries(model: NeuralComputerAgent, episode_groups, memory, device):
    loss = torch.zeros((), device=device)
    correct = total = 0
    for episodes in episode_groups:
        output, targets = _forward(model, episodes, memory, device)
        logits = output.answer_logits[:, -1]
        loss = loss + nn.functional.cross_entropy(logits, targets)
        correct += int((logits.argmax(-1) == targets).sum())
        total += targets.numel()
    return loss / max(1, len(episode_groups)), correct / max(1, total)


def run_batch(model: NeuralComputerAgent, lifetimes: list[AttentionTransferLifetime],
              device: torch.device, *, condition: str = "inherited",
              future_weight: float = 1.0,
              reference_future_loss: float | None = None,
              advantage_margin: float = 0.0,
              advantage_weight: float = 0.0,
              ) -> tuple[torch.Tensor, dict[str, float]]:
    if condition not in CONTROLS:
        raise ValueError(f"unknown condition {condition!r}")
    batch = len(lifetimes)
    memory = DifferentiableBatchMemory(batch, model.hidden, device=device)
    interface_loss = torch.zeros((), device=device)
    mean_strength = 0.0
    for index in range(len(lifetimes[0].studies)):
        memory, step_loss, strength = _append(
            model, [item.studies[index] for item in lifetimes], memory, device)
        interface_loss = interface_loss + step_loss
        mean_strength += strength
    interface_steps = len(lifetimes[0].studies)
    memory = _control(memory, condition)

    old_groups = [[item.old_queries[index] for item in lifetimes]
                  for index in range(len(lifetimes[0].old_queries))]
    audit_groups = [[item.old_audit_queries[index] for item in lifetimes]
                    for index in range(len(lifetimes[0].old_audit_queries))]
    old_loss, _ = _score_queries(model, old_groups, memory, device)
    _, old_accuracy = _score_queries(model, audit_groups, memory, device)

    future_groups = [[item.future_queries[index] for item in lifetimes]
                     for index in range(len(lifetimes[0].future_queries))]
    shot_losses = []
    shot_accuracies = {}
    support_cursor = 0
    transaction_acceptance = 0.0
    for shots in SHOTS:
        while support_cursor < shots:
            previous = memory
            memory, step_loss, strength = _append(
                model, [item.supports[support_cursor] for item in lifetimes], memory, device)
            interface_loss = interface_loss + step_loss
            mean_strength += strength
            interface_steps += 1
            support_cursor += 1
            if condition == "composed_latest":
                memory = _latest_row(memory)
            elif condition == "transactional_latest":
                candidate = _latest_row(memory)
                memory, accepted = _transactional_latest(
                    model, old_groups, previous, candidate, device)
                transaction_acceptance += accepted
        shot_loss, shot_accuracy = _score_queries(model, future_groups, memory, device)
        shot_losses.append(shot_loss)
        shot_accuracies[shots] = shot_accuracy

    retention_loss, retention_accuracy = _score_queries(
        model, audit_groups, memory, device)
    all_shot_loss = torch.stack(shot_losses).mean()
    # The requested learning-reuse signal is improvement after scarce evidence,
    # not merely carrying old behavior into a zero-shot query. Optimize the one-
    # and two-shot learning curve directly; retain all shot levels for auditing.
    future_loss = (shot_losses[1] + shot_losses[2]) / 2
    interface_loss = interface_loss / interface_steps
    # Future few-shot reuse is primary. Old performance is explicitly retained;
    # interface prediction is merely an auxiliary sensory-learning signal.
    loss = (future_weight * future_loss + 0.5 * old_loss + 0.5 * retention_loss +
            0.05 * interface_loss)
    surrogate_advantage = 0.0
    advantage_penalty = torch.zeros((), device=device)
    if reference_future_loss is not None:
        reference = future_loss.new_tensor(reference_future_loss)
        surrogate_advantage = reference - future_loss.detach()
        # The matched control is detached: the agent is rewarded only for making
        # inherited memory better, never for sabotaging the expected baseline.
        advantage_penalty = torch.relu(
            future_loss - reference + future_loss.new_tensor(advantage_margin))
        loss = loss + advantage_weight * advantage_penalty
    metrics = {
        **{f"accuracy_{shots}_shot": shot_accuracies[shots] for shots in SHOTS},
        "few_shot_auc": sum(shot_accuracies.values()) / len(SHOTS),
        "early_transfer_auc": (shot_accuracies[1] + shot_accuracies[2]) / 2,
        "old_accuracy": old_accuracy,
        "retention_accuracy": retention_accuracy,
        "future_loss": float(future_loss.detach()),
        "all_shot_loss": float(all_shot_loss.detach()),
        "retention_loss": float(retention_loss.detach()),
        "interface_loss": float(interface_loss.detach()),
        "stored_rows": float(memory.count),
        "mean_write_strength": mean_strength / interface_steps,
        "transaction_acceptance": transaction_acceptance / max(1, max(SHOTS)),
        "surrogate_transfer_advantage": float(surrogate_advantage),
        "advantage_margin_penalty": float(advantage_penalty.detach()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model: NeuralComputerAgent, device: torch.device, *, samples: int,
             batch_size: int, seed: int, query_count: int) -> dict[str, object]:
    model.eval()
    totals = {condition: {} for condition in CONTROLS}
    seen = 0
    for offset in range(0, samples, batch_size):
        count = min(batch_size, samples - offset)
        if count < 2:
            raise ValueError("evaluation batches must contain at least two lifetimes")
        lifetimes = lifetime_batch(seed + offset, count, heldout=True,
                                   query_count=query_count)
        for condition in CONTROLS:
            _, metrics = run_batch(model, lifetimes, device, condition=condition)
            for key, value in metrics.items():
                totals[condition][key] = totals[condition].get(key, 0.0) + value * count
        seen += count
    results = {condition: {key: value / seen for key, value in metrics.items()}
               for condition, metrics in totals.items()}
    inherited = results["inherited"]
    controls = (results["empty"], results["shuffled"], results["unrelated"],
                results["garbage"])
    results["transfer_advantage"] = {
        "few_shot_auc_vs_best_control": inherited["few_shot_auc"] -
        max(item["few_shot_auc"] for item in controls),
        "early_auc_vs_best_control": inherited["early_transfer_auc"] -
        max(item["early_transfer_auc"] for item in controls),
        "retention_delta": inherited["retention_accuracy"] - inherited["old_accuracy"],
        "latest_compression_auc_delta":
            results["composed_latest"]["few_shot_auc"] - inherited["few_shot_auc"],
        "latest_compression_retention_delta":
            results["composed_latest"]["retention_accuracy"] - inherited["retention_accuracy"],
        "latest_rows_saved": inherited["stored_rows"] -
            results["composed_latest"]["stored_rows"],
        "transactional_auc_delta":
            results["transactional_latest"]["few_shot_auc"] - inherited["few_shot_auc"],
        "transactional_retention_delta":
            results["transactional_latest"]["retention_accuracy"] -
            results["transactional_latest"]["old_accuracy"],
        "transactional_rows_saved": inherited["stored_rows"] -
            results["transactional_latest"]["stored_rows"],
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot attention transfer through memory")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=1024)
    parser.add_argument("--eval-lifetimes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--workspace-slots", type=int, default=12)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--thought-steps", type=int, default=4)
    parser.add_argument("--read-top-k", type=int, default=12)
    parser.add_argument("--primitive-epochs", type=int, default=2,
                        help="warm up old primitive mappings before transfer loss")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--advantage-weight", type=float, default=0.5)
    parser.add_argument("--advantage-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 2 or args.train_lifetimes < 2 or args.eval_lifetimes < 2:
        raise ValueError("batch and sample counts must be at least two")
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = NeuralComputerAgent(args.hidden, args.workspace_slots, args.heads,
                                args.thought_steps, action_count=8,
                                read_top_k=args.read_top_k).to(device)
    if args.initial_checkpoint:
        payload = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        incompatible = model.load_state_dict(payload["model"], strict=False)
        if set(incompatible.missing_keys) - {"log_read_scale"} or incompatible.unexpected_keys:
            raise ValueError(f"incompatible initial checkpoint: {incompatible}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        totals = {}
        seen = 0
        for offset in range(0, args.train_lifetimes, args.batch_size):
            count = min(args.batch_size, args.train_lifetimes - offset)
            if count < 2:
                continue
            lifetimes = lifetime_batch(epoch * args.train_lifetimes + offset, count,
                                       query_count=args.query_count)
            optimizer.zero_grad(set_to_none=True)
            future_weight = 0.0 if epoch < args.primitive_epochs else 1.0
            reference_loss = None
            if future_weight:
                with torch.no_grad():
                    _, reference_metrics = run_batch(
                        model, lifetimes, device, condition="empty")
                reference_loss = reference_metrics["future_loss"]
            loss, metrics = run_batch(
                model, lifetimes, device, future_weight=future_weight,
                reference_future_loss=reference_loss,
                advantage_margin=args.advantage_margin,
                advantage_weight=args.advantage_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * count
            seen += count
        row = {key: value / seen for key, value in totals.items()}
        row["epoch"] = epoch + 1
        row["future_weight"] = 0.0 if epoch < args.primitive_epochs else 1.0
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    evaluation = evaluate(model, device, samples=args.eval_lifetimes,
                          batch_size=args.batch_size, seed=2_000_000 + args.seed * 10_000,
                          query_count=args.query_count)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arguments": vars(args)}, args.checkpoint)
    report = {
        "schema": "forward-transfer-attention-v1",
        "sensory_only": True,
        "controller_weights_frozen_during_evaluation": True,
        "parameters": parameter_count(model),
        "shots": SHOTS,
        "history": history,
        "evaluation": evaluation,
        "training_seconds": time.perf_counter() - started,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evaluation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
