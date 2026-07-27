"""Reward-train an adaptive read/no-read head inside the unified controller."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .audit_selective_disk import _query_keys, _support
from .environment import NULL_ACTION, generate_lifetimes
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import evaluate, seed_everything
from .train_selective_memory import _bernoulli_log_probability


def _grouped_read_features(
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
    cosine = torch.einsum(
        "gcw,gkw->gck", query_group, key_group)
    ranked = (
        cosine
        + strength_group.clamp_min(1e-6).log().unsqueeze(1))
    ranked = ranked.masked_fill(~valid.unsqueeze(1), -1e9)
    scores, selected = ranked.topk(2, dim=-1)
    top = selected[:, :, 0]
    read = torch.gather(
        value_group, 1,
        top.unsqueeze(-1).expand(
            -1, -1, value_group.shape[-1]))
    confidence = torch.gather(
        cosine, 2, top.unsqueeze(-1)).squeeze(-1)
    margin = scores[:, :, 0] - scores[:, :, 1]
    valid_count = valid.sum(-1, keepdim=True).expand(-1, capacity)
    margin = torch.where(
        valid_count == 1, torch.ones_like(margin), margin)
    selected_usage = torch.gather(
        strength_group, 1, top)
    occupancy = (
        valid.to(keys.dtype).sum(-1, keepdim=True) / capacity
    ).expand(-1, capacity).clone()
    empty = ~valid.any(-1)
    read[empty] = 0
    confidence[empty] = 0
    margin[empty] = 0
    selected_usage[empty] = 0
    occupancy[empty] = 0
    features = torch.stack((
        confidence, margin, selected_usage, occupancy), dim=-1)
    return (
        read.reshape_as(values),
        features.reshape(keys.shape[0], 4),
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
    read, features, stored = _grouped_read_features(
        keys, values, strengths, queries, capacity=capacity,
        write_threshold=write_threshold)
    return batch, read, features, stored


@torch.no_grad()
def _outcomes(
        model: UnifiedCognitiveController, batch,
        memory: torch.Tensor, *, device: torch.device,
        query_trial: int = 2) -> torch.Tensor:
    count = batch.batch_size
    state = model.initial_state(count, device=device)
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    output, _ = model.step(
        batch.frames[:, query_trial], state, null_action, zeros, zeros,
        retrieved_memory=memory)
    return (
        output.logits.argmax(-1)
        == batch.correct_actions[:, query_trial]).to(torch.float32)


@torch.no_grad()
def _evaluate(
        model: UnifiedCognitiveController, *, count: int, capacity: int,
        seed: int, device: torch.device,
        write_threshold: float) -> dict[str, object]:
    model.eval()
    batch, read, features, stored = _batch(
        model, count=count, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold)
    probability = model.memory_read_probability(features)
    accepted = probability >= 0.5
    gated = torch.where(
        accepted.unsqueeze(-1), read, torch.zeros_like(read))
    accuracy = float(_outcomes(
        model, batch, gated, device=device).mean())
    no_memory = float(_outcomes(
        model, batch, torch.zeros_like(read), device=device).mean())
    ungated = float(_outcomes(
        model, batch, read, device=device).mean())
    report = {
        "accuracy": accuracy,
        "no_memory_accuracy": no_memory,
        "ungated_accuracy": ungated,
        "read_accept_rate": float(accepted.float().mean()),
        "stored_context_accept_rate": float(
            accepted[stored].float().mean()),
        "absent_context_false_accept_rate": float(
            accepted[~stored].float().mean()),
        "probability_mean": float(probability.mean()),
    }
    report["gate"] = {
        "accuracy_at_least_88": accuracy >= 0.88,
        "improves_ungated_by_20": accuracy >= ungated + 0.20,
        "memory_is_causal": no_memory <= accuracy - 0.15,
        "stored_accept_at_least_80":
            report["stored_context_accept_rate"] >= 0.80,
        "absent_false_accept_at_most_20":
            report["absent_context_false_accept_rate"] <= 0.20,
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
    parser.add_argument("--seed", type=int, default=5901)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-contexts", type=int, default=4096)
    parser.add_argument("--bank-capacity", type=int, default=8)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--read-cost", type=float, default=0.02)
    parser.add_argument("--gate-hidden", type=int, default=0)
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
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_read"] = True
    configuration["adaptive_memory_read_hidden"] = args.gate_hidden
    model = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = model.load_state_dict(
        payload["state_dict"], strict=False)
    if (
            not missing
            or not all(name.startswith("memory_read_gate.") for name in missing)
            or unexpected):
        raise ValueError(
            f"unexpected checkpoint mismatch: {missing=}, {unexpected=}")
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.memory_read_gate is not None
    for parameter in model.memory_read_gate.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.memory_read_gate.parameters(), lr=args.learning_rate)

    baseline = 0.0
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        batch, read, features, _ = _batch(
            model, count=args.batch_size,
            capacity=args.bank_capacity,
            seed=args.seed * 1_000_000 + step,
            device=device, write_threshold=args.write_threshold)
        probability = model.memory_read_probability(features)
        accepted = torch.bernoulli(probability)
        with torch.no_grad():
            memory = read * accepted.unsqueeze(-1)
            outcomes = _outcomes(
                model, batch, memory, device=device)
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
        torch.nn.utils.clip_grad_norm_(
            model.memory_read_gate.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "read_rate": float(accepted.detach().mean()),
                "accuracy": float(outcomes.mean()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    adaptive = _evaluate(
        model, count=args.test_contexts,
        capacity=args.bank_capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)
    binary = evaluate(
        model, count=2048, trials=6, seed=args.seed + 91_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=2048, trials=6, seed=args.seed + 92_000_000,
        device=device, task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    only_gate_changed = all(
        name.startswith("memory_read_gate.") for name in changed)
    admitted = (
        adaptive["gate"]["accepted"]
        and binary["gate"]["accepted"]
        and four_rule["gate"]["accepted"]
        and only_gate_changed)
    report = {
        "schema": "unified-controller-adaptive-memory-read-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "memory_read_features": [
            "cosine_match", "top_two_rank_margin",
            "selected_row_strength", "bank_occupancy"],
        "semantic_or_match_labels_used_for_training": False,
        "training_signal":
            "verified_query_success_minus_generic_read_cost",
        "unique_contexts_seen": args.steps * args.batch_size,
        "accounting": {
            "unique_verifier_bits": args.steps * args.batch_size,
            "unique_logical_lifetimes": args.steps * args.batch_size,
            "optimizer_updates": args.steps,
            "replayed_examples": 0,
            "training_seconds": (
                history[-1]["elapsed_seconds"] if history else 0.0),
        },
        "history": history,
        "adaptive_evaluation": adaptive,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "only_memory_read_gate_changed": only_gate_changed,
        "all_admission_gates_passed": admitted,
        "total_seconds": time.perf_counter() - started,
    }
    if admitted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint_saved": report["checkpoint_saved"],
        "adaptive_evaluation": adaptive,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
