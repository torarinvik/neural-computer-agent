from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_neural_computer.model import NeuralComputerAgent, parameter_count
from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory

from .consolidator import LatentConsolidator
from .environment import (SHOTS, generate_attention_lifetime,
                          generate_shape_attention_lifetime,
                          generate_temporal_attention_lifetime)
from .train import _append, _forward, lifetime_batch, seed_everything


def _groups(lifetimes, field: str):
    items = getattr(lifetimes[0], field)
    return [[getattr(item, field)[index] for item in lifetimes]
            for index in range(len(items))]


def _initial_memory(model, lifetimes, device, *, train_model: bool = False):
    memory = DifferentiableBatchMemory(len(lifetimes), model.hidden, device=device)
    context = torch.enable_grad() if train_model else torch.no_grad()
    with context:
        for index in range(len(lifetimes[0].studies)):
            memory, _, _ = _append(
                model, [item.studies[index] for item in lifetimes], memory, device)
    return memory


def _mixed_lifetime(seed: int, *, heldout: bool = False, query_count: int = 4):
    """Half temporal practice, with equal spatial and shape rehearsal."""
    selector = seed % 4
    generator = (generate_temporal_attention_lifetime if selector < 2 else
                 generate_attention_lifetime if selector == 2 else
                 generate_shape_attention_lifetime)
    return generator(seed, heldout=heldout, query_count=query_count)


def _query_loss(model, groups, memory, device, teacher_memory=None,
                event_memory=None,
                distill_weight: float = 0.5, read_auxiliary_loss=None):
    total = torch.zeros((), device=device)
    correct = teacher_correct = count = 0
    for episodes in groups:
        student, targets = _forward(
            model, episodes, memory, device, event_memory=event_memory)
        logits = student.answer_logits[:, -1]
        loss = nn.functional.cross_entropy(logits, targets)
        if read_auxiliary_loss is not None:
            loss = loss + read_auxiliary_loss(student, targets)
        if teacher_memory is not None and distill_weight:
            with torch.no_grad():
                teacher, _ = _forward(
                    model, episodes, teacher_memory, device,
                    event_memory=event_memory)
            teacher_logits = teacher.answer_logits[:, -1]
            teacher_prob = torch.softmax(teacher_logits, dim=-1)
            loss = loss + distill_weight * nn.functional.kl_div(
                torch.log_softmax(logits, dim=-1), teacher_prob, reduction="batchmean")
            teacher_correct += int((teacher_logits.argmax(-1) == targets).sum())
        else:
            teacher_correct += int((logits.argmax(-1) == targets).sum())
        total = total + loss
        correct += int((logits.argmax(-1) == targets).sum())
        count += targets.numel()
    return total / len(groups), correct / count, teacher_correct / count


def run_compaction_batch(model, consolidator, lifetimes, device, *, train: bool,
                         condition: str = "intact", train_model: bool = False,
                         preserve_raw_write: bool = False,
                         preserve_first_raw_write: bool = False,
                         preserve_study_raw_writes: bool = False,
                         old_loss_weight: float = 2.0,
                         future_loss_weight: float = 1.0,
                         write_auxiliary_loss=None,
                         write_auxiliary_weight: float = 0.0,
                         read_auxiliary_loss=None,
                         read_auxiliary_weight: float = 0.0,
                         write_residual_penalty_weight: float = 0.0):
    if condition not in ("intact", "empty", "shuffled", "garbage"):
        raise ValueError(f"unknown compact-memory condition {condition!r}")
    if preserve_raw_write and preserve_first_raw_write:
        raise ValueError("choose either the latest or first raw-write sidecar")
    full = _initial_memory(model, lifetimes, device, train_model=train_model)
    compact = full
    study_event_memory = full if preserve_study_raw_writes else None
    old_groups = _groups(lifetimes, "old_queries" if train else "old_audit_queries")
    future_groups = _groups(lifetimes, "future_queries")
    losses = []
    auxiliary_losses = []
    residual_penalties = []
    metrics = {}
    cursor = 0
    first_raw_write = None
    for shots in SHOTS:
        while cursor < shots:
            with torch.no_grad():
                full, _, _ = _append(
                    model, [item.supports[cursor] for item in lifetimes], full, device)
            support_output, _ = _forward(
                model, [item.supports[cursor] for item in lifetimes], compact, device)
            if first_raw_write is None:
                first_raw_write = (
                    support_output.write_keys, support_output.write_values,
                    support_output.write_strengths)
            compact = compact.append(
                support_output.write_keys, support_output.write_values,
                support_output.write_strengths,
                torch.ones_like(support_output.write_strengths))
            if write_auxiliary_loss is not None:
                auxiliary_losses.append(write_auxiliary_loss(support_output))
            if write_residual_penalty_weight:
                residual_penalties.append(
                    support_output.event_binding_residual.square().mean())
            compact = consolidator(compact)
            if preserve_raw_write or preserve_first_raw_write:
                # Diagnostic sidecar: retain the proven raw writer row instead
                # of forcing the consolidator to be the sole carrier.
                sidecar = (first_raw_write if preserve_first_raw_write else
                           (support_output.write_keys, support_output.write_values,
                            support_output.write_strengths))
                compact = compact.append(
                    sidecar[0], sidecar[1], sidecar[2],
                    torch.ones_like(support_output.write_strengths))
            cursor += 1
        observed = compact.counterfactual(condition) if shots else compact
        old_loss, old_accuracy, full_old_accuracy = _query_loss(
            model, old_groups, observed, device, full if shots else None,
            event_memory=study_event_memory,
            read_auxiliary_loss=read_auxiliary_loss)
        future_loss, future_accuracy, full_future_accuracy = _query_loss(
            model, future_groups, observed, device, full if shots else None,
            event_memory=study_event_memory,
            read_auxiliary_loss=read_auxiliary_loss)
        if shots:
            losses.append(old_loss_weight * old_loss + future_loss_weight * future_loss)
        metrics[f"old_accuracy_{shots}_shot"] = old_accuracy
        metrics[f"future_accuracy_{shots}_shot"] = future_accuracy
        metrics[f"full_old_accuracy_{shots}_shot"] = full_old_accuracy
        metrics[f"full_future_accuracy_{shots}_shot"] = full_future_accuracy
    metrics["compact_few_shot_auc"] = sum(
        metrics[f"future_accuracy_{shots}_shot"] for shots in SHOTS) / len(SHOTS)
    metrics["compact_early_auc"] = (
        metrics["future_accuracy_1_shot"] + metrics["future_accuracy_2_shot"]) / 2
    metrics["compact_retention"] = metrics["old_accuracy_4_shot"]
    metrics["full_few_shot_auc"] = sum(
        metrics[f"full_future_accuracy_{shots}_shot"] for shots in SHOTS) / len(SHOTS)
    metrics["full_retention"] = metrics["full_old_accuracy_4_shot"]
    metrics["rows_saved"] = float(full.count - 1)
    total_loss = torch.stack(losses).mean()
    if auxiliary_losses:
        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        total_loss = total_loss + write_auxiliary_weight * auxiliary_loss
        metrics["write_auxiliary_loss"] = float(auxiliary_loss.detach())
    if residual_penalties:
        residual_penalty = torch.stack(residual_penalties).mean()
        total_loss = total_loss + write_residual_penalty_weight * residual_penalty
        metrics["write_residual_penalty"] = float(residual_penalty.detach())
    return total_loss, metrics


@torch.no_grad()
def evaluate(model, consolidator, device, *, samples, batch_size, seed, query_count,
             condition="intact", generator=None):
    totals = {}
    seen = 0
    for offset in range(0, samples, batch_size):
        count = min(batch_size, samples - offset)
        if generator is None:
            lifetimes = lifetime_batch(seed + offset, count, heldout=True,
                                       query_count=query_count)
        else:
            lifetimes = [generator(seed + offset + index, heldout=True,
                                   query_count=query_count)
                         for index in range(count)]
        _, metrics = run_compaction_batch(
            model, consolidator, lifetimes, device, train=False, condition=condition)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * count
        seen += count
    return {key: value / seen for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train sensory-behavior latent consolidation")
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-consolidator-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=1536)
    parser.add_argument("--eval-lifetimes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--primitive", choices=("spatial", "shape", "temporal", "mixed"),
                        default="spatial")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.controller_checkpoint, map_location=device, weights_only=False)
    controller_args = payload["arguments"]
    model = NeuralComputerAgent(
        controller_args["hidden"], controller_args["workspace_slots"],
        controller_args["heads"], controller_args["thought_steps"], action_count=8,
        read_top_k=controller_args["read_top_k"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    consolidator = LatentConsolidator(model.hidden, heads=controller_args["heads"]).to(device)
    if args.initial_consolidator_checkpoint:
        initial = torch.load(args.initial_consolidator_checkpoint, map_location=device,
                             weights_only=False)
        consolidator.load_state_dict(initial["consolidator"])
    optimizer = torch.optim.AdamW(consolidator.parameters(), lr=args.learning_rate)
    history = []
    started = time.perf_counter()
    generators = {"spatial": generate_attention_lifetime,
                  "shape": generate_shape_attention_lifetime,
                  "temporal": generate_temporal_attention_lifetime,
                  "mixed": _mixed_lifetime}
    generator = generators[args.primitive]
    for epoch in range(args.epochs):
        consolidator.train()
        totals = {}
        seen = 0
        for offset in range(0, args.train_lifetimes, args.batch_size):
            count = min(args.batch_size, args.train_lifetimes - offset)
            batch_generator = generator
            if args.primitive == "mixed":
                cycle = (generate_temporal_attention_lifetime,
                         generate_temporal_attention_lifetime,
                         generate_attention_lifetime,
                         generate_shape_attention_lifetime)
                batch_generator = cycle[(offset // args.batch_size) % len(cycle)]
            lifetimes = [batch_generator(epoch * args.train_lifetimes + offset + index,
                                         query_count=args.query_count)
                         for index in range(count)]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = run_compaction_batch(
                model, consolidator, lifetimes, device, train=True)
            loss.backward()
            nn.utils.clip_grad_norm_(consolidator.parameters(), 1.0)
            optimizer.step()
            for key, value in {"loss": float(loss.detach()), **metrics}.items():
                totals[key] = totals.get(key, 0.0) + value * count
            seen += count
        row = {key: value / seen for key, value in totals.items()}
        row["epoch"] = epoch + 1
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    consolidator.eval()
    evaluation_generator = (generate_temporal_attention_lifetime
                            if args.primitive == "mixed" else generator)
    evaluation = evaluate(
        model, consolidator, device, samples=args.eval_lifetimes,
        batch_size=args.batch_size, seed=3_000_000 + args.seed * 10_000,
        query_count=args.query_count, generator=evaluation_generator)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"consolidator": consolidator.state_dict(), "arguments": vars(args)},
               args.checkpoint)
    report = {
        "schema": "forward-transfer-consolidator-v1", "sensory_only": True,
        "controller_frozen": True, "consolidator_parameters": parameter_count(consolidator),
        "history": history, "evaluation": evaluation,
        "training_seconds": time.perf_counter() - started,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evaluation": evaluation}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
