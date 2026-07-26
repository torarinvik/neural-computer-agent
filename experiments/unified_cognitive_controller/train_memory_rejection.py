"""Learn a generic no-match threshold from verified query outcomes."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .audit_selective_disk import _query_accuracy, _query_keys, _support
from .environment import NULL_ACTION, generate_lifetimes
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import seed_everything
from .train_selective_memory import _bernoulli_log_probability


def _sparse_grouped_read(
        keys: torch.Tensor, values: torch.Tensor,
        strengths: torch.Tensor, queries: torch.Tensor, *,
        capacity: int, write_threshold: float
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = keys.shape[0] // capacity
    key_group = torch.nn.functional.normalize(
        keys.reshape(groups, capacity, -1), dim=-1)
    query_group = torch.nn.functional.normalize(
        queries.reshape(groups, capacity, -1), dim=-1)
    value_group = values.reshape(groups, capacity, -1)
    strength_group = strengths.reshape(groups, capacity)
    valid = strength_group >= write_threshold
    cosine_similarity = torch.einsum(
        "gcw,gkw->gck", query_group, key_group)
    similarity = (
        cosine_similarity
        + strength_group.clamp_min(1e-6).log().unsqueeze(1))
    similarity = similarity.masked_fill(
        ~valid.unsqueeze(1), -1e9)
    scores, selected = similarity.max(-1)
    match_confidence = torch.gather(
        cosine_similarity, 2, selected.unsqueeze(-1)).squeeze(-1)
    read = torch.gather(
        value_group, 1,
        selected.unsqueeze(-1).expand(
            -1, -1, value_group.shape[-1]))
    empty = ~valid.any(-1)
    read[empty] = 0
    match_confidence[empty] = -1e9
    return (
        read.reshape_as(values), match_confidence.reshape(-1),
        valid.reshape(-1))


@torch.no_grad()
def _batch(
        model: UnifiedCognitiveController, *, count: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float):
    batch = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    keys, values, strengths = _support(
        model, batch, device=device)
    queries = _query_keys(model, batch, device=device)
    read, scores, stored = _sparse_grouped_read(
        keys, values, strengths, queries, capacity=capacity,
        write_threshold=write_threshold)
    return batch, read, scores, stored


@torch.no_grad()
def _evaluate(
        model: UnifiedCognitiveController, threshold: float, *,
        count: int, capacity: int, seed: int, device: torch.device,
        write_threshold: float) -> dict[str, object]:
    batch, read, scores, stored = _batch(
        model, count=count, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold)
    accepted = scores >= threshold
    gated = torch.where(
        accepted.unsqueeze(-1), read, torch.zeros_like(read))
    accuracy = _query_accuracy(
        model, batch, gated, device=device)
    no_memory = _query_accuracy(
        model, batch, torch.zeros_like(read), device=device)
    ungated = _query_accuracy(
        model, batch, read, device=device)
    report = {
        "threshold": threshold,
        "accuracy": accuracy,
        "no_memory_accuracy": no_memory,
        "ungated_accuracy": ungated,
        "read_accept_rate": float(accepted.float().mean()),
        "stored_context_accept_rate": float(
            accepted[stored].float().mean()),
        "absent_context_false_accept_rate": float(
            accepted[~stored].float().mean()),
    }
    report["gate"] = {
        "accuracy_at_least_85": accuracy >= 0.85,
        "improves_ungated_by_15": accuracy >= ungated + 0.15,
        "memory_is_causal": no_memory <= accuracy - 0.15,
        "absent_false_accept_at_most_25":
            report["absent_context_false_accept_rate"] <= 0.25,
    }
    report["gate"]["accepted"] = all(report["gate"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=5701)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-contexts", type=int, default=4096)
    parser.add_argument("--bank-capacity", type=int, default=8)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--initial-threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--decision-temperature", type=float, default=20.0)
    parser.add_argument("--read-cost", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size % args.bank_capacity:
        raise ValueError("batch size must divide into complete memory banks")
    if args.test_contexts % args.bank_capacity:
        raise ValueError(
            "test contexts must divide into complete memory banks")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    threshold = nn.Parameter(torch.tensor(
        args.initial_threshold, device=device))
    optimizer = torch.optim.Adam([threshold], lr=args.learning_rate)
    baseline = 0.0
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch, read, scores, _ = _batch(
            model, count=args.batch_size,
            capacity=args.bank_capacity,
            seed=args.seed * 1_000_000 + step,
            device=device, write_threshold=args.write_threshold)
        probability = torch.sigmoid(
            (scores - threshold) * args.decision_temperature)
        accepted = torch.bernoulli(probability)
        with torch.no_grad():
            memory = read * accepted.unsqueeze(-1)
            # Preserve per-example credit rather than a batch scalar.
            fresh = model.initial_state(
                args.batch_size, device=device)
            null_action = torch.full(
                (args.batch_size,), NULL_ACTION, dtype=torch.long,
                device=device)
            zeros = torch.zeros(args.batch_size, device=device)
            output, _ = model.step(
                batch.frames[:, 2], fresh, null_action, zeros, zeros,
                retrieved_memory=memory)
            outcomes = (
                output.logits.argmax(-1)
                == batch.correct_actions[:, 2]).to(torch.float32)
        reward = outcomes - args.read_cost * accepted
        baseline = (
            0.95 * baseline
            + 0.05 * float(reward.detach().mean()))
        loss = -(
            (reward.detach() - baseline)
            * _bernoulli_log_probability(
                probability, accepted)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "threshold": float(threshold.detach()),
                "read_rate": float(accepted.detach().mean()),
                "accuracy": float(outcomes.mean()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    evaluation = _evaluate(
        model, float(threshold.detach()),
        count=args.test_contexts, capacity=args.bank_capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)
    admitted = evaluation["gate"]["accepted"]
    report = {
        "schema": "unified-controller-memory-rejection-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "semantic_or_match_labels_used_for_training": False,
        "training_signal":
            "verified_query_success_minus_generic_read_cost",
        "model_weights_changed": False,
        "learned_memory_read_threshold": float(
            threshold.detach()),
        "unique_contexts_seen": args.steps * args.batch_size,
        "history": history,
        "evaluation": evaluation,
        "all_admission_gates_passed": admitted,
        "total_seconds": time.perf_counter() - started,
    }
    if admitted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        output_payload = dict(payload)
        output_payload["memory_read_threshold"] = float(
            threshold.detach())
        output_payload["source_report"] = str(args.report)
        torch.save(output_payload, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint_saved": report["checkpoint_saved"],
        "evaluation": evaluation,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
