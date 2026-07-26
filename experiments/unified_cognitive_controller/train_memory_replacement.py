"""Reward-train the first bounded-memory replacement atom."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from .audit_selective_disk import _query_keys, _support
from .environment import CognitiveLifetimeBatch, generate_lifetimes
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import evaluate, seed_everything


def _select_batch(
        batch: CognitiveLifetimeBatch,
        indices: torch.Tensor) -> CognitiveLifetimeBatch:
    return replace(
        batch,
        frames=batch.frames[indices],
        correct_actions=batch.correct_actions[indices],
        stimulus_identities=batch.stimulus_identities[indices],
        rule_bits=batch.rule_bits[indices],
        seeds=batch.seeds[indices],
    )


def _join_batches(
        batches: list[CognitiveLifetimeBatch]) -> CognitiveLifetimeBatch:
    return CognitiveLifetimeBatch(
        frames=torch.cat([batch.frames for batch in batches]),
        correct_actions=torch.cat([
            batch.correct_actions for batch in batches]),
        stimulus_identities=torch.cat([
            batch.stimulus_identities for batch in batches]),
        rule_bits=torch.cat([batch.rule_bits for batch in batches]),
        seeds=torch.cat([batch.seeds for batch in batches]),
    )


@torch.no_grad()
def _written_contexts(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, write_threshold: float
        ) -> tuple[
            CognitiveLifetimeBatch, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, int]:
    batches: list[CognitiveLifetimeBatch] = []
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    strengths: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    remaining = count
    generated = 0
    round_index = 0
    while remaining:
        request = max(2, remaining * 2)
        request += request % 2
        data_seed = seed + round_index * 1_000_003
        batch = _add_context_signatures(
            generate_lifetimes(
                request, 3, seed=data_seed, heldout=True,
                task="binary_mapping", support_trials=1,
                device=device),
            seed=data_seed + 10_000_000)
        support_keys, support_values, support_strengths = _support(
            model, batch, device=device)
        query_keys = _query_keys(model, batch, device=device)
        eligible = (
            support_strengths >= write_threshold
        ).nonzero(as_tuple=False).squeeze(1)[:remaining]
        if eligible.numel():
            batches.append(_select_batch(batch, eligible))
            keys.append(support_keys[eligible])
            values.append(support_values[eligible])
            strengths.append(support_strengths[eligible])
            queries.append(query_keys[eligible])
            remaining -= int(eligible.numel())
        generated += request
        round_index += 1
        if round_index > 8:
            raise RuntimeError(
                "could not collect enough controller-admitted contexts")
    return (
        _join_batches(batches),
        torch.cat(keys),
        torch.cat(values),
        torch.cat(strengths),
        torch.cat(queries),
        generated,
    )


def _gather_rows(
        rows: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        rows, 1,
        indices.unsqueeze(-1).expand(-1, -1, rows.shape[-1]))


@torch.no_grad()
def replacement_batch(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float
        ) -> dict[str, object]:
    context_count = banks * (capacity + 1)
    batch, keys, values, strengths, query_keys, generated = (
        _written_contexts(
            model, count=context_count, seed=seed, device=device,
            write_threshold=write_threshold))
    width = keys.shape[-1]
    key_group = keys.reshape(banks, capacity + 1, width)
    value_group = values.reshape(banks, capacity + 1, width)
    strength_group = strengths.reshape(banks, capacity + 1)
    query_group = query_keys.reshape(banks, capacity + 1, width)

    generator = torch.Generator().manual_seed(seed + 30_000_000)
    permutation = torch.stack([
        torch.randperm(capacity, generator=generator)
        for _ in range(banks)
    ]).to(device)
    bank_keys = _gather_rows(key_group[:, :capacity], permutation)
    bank_values = _gather_rows(value_group[:, :capacity], permutation)
    bank_strengths = torch.gather(
        strength_group[:, :capacity], 1, permutation)
    logical_age = torch.arange(
        1, capacity + 1, device=device,
        dtype=keys.dtype).expand(banks, -1)
    bank_ages = torch.gather(logical_age, 1, permutation)
    candidate_key = key_group[:, capacity]
    candidate_value = value_group[:, capacity]
    candidate_strength = strength_group[:, capacity]

    candidate_similarity = torch.einsum(
        "bw,bcw->bc",
        torch.nn.functional.normalize(candidate_key, dim=-1),
        torch.nn.functional.normalize(bank_keys, dim=-1))
    row_features = torch.stack((
        bank_ages / capacity,
        bank_strengths,
        candidate_similarity,
        candidate_strength.unsqueeze(1).expand(-1, capacity),
        torch.zeros_like(bank_ages),
    ), dim=-1)
    skip_features = torch.zeros(
        banks, 1, 5, device=device, dtype=keys.dtype)
    skip_features[:, 0, 3] = candidate_strength
    skip_features[:, 0, 4] = 1.0
    option_features = torch.cat((skip_features, row_features), dim=1)
    if model.adaptive_memory_replace_features > 5:
        extra = torch.zeros(
            *option_features.shape[:-1],
            model.adaptive_memory_replace_features - 5,
            device=device, dtype=keys.dtype)
        option_features = torch.cat((option_features, extra), dim=-1)

    base = (
        torch.arange(banks, device=device).unsqueeze(1)
        * (capacity + 1))
    future_indices = (
        base + torch.arange(
            1, capacity + 1, device=device).unsqueeze(0)
    ).reshape(-1)
    future_batch = _select_batch(batch, future_indices)
    future_queries = query_group[:, 1:]
    target_action = bank_ages.argmin(-1) + 1
    return {
        "bank_keys": bank_keys,
        "bank_values": bank_values,
        "bank_strengths": bank_strengths,
        "bank_ages": bank_ages,
        "candidate_key": candidate_key,
        "candidate_value": candidate_value,
        "candidate_strength": candidate_strength,
        "option_features": option_features,
        "future_batch": future_batch,
        "future_queries": future_queries,
        "target_action": target_action,
        "generated_contexts": generated,
    }


def _apply_replacement(
        data: dict[str, object], actions: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = data["bank_keys"].clone()
    values = data["bank_values"].clone()
    strengths = data["bank_strengths"].clone()
    replacing = actions > 0
    if bool(replacing.any()):
        banks = replacing.nonzero(as_tuple=False).squeeze(1)
        slots = actions[banks] - 1
        keys[banks, slots] = data["candidate_key"][banks]
        values[banks, slots] = data["candidate_value"][banks]
        strengths[banks, slots] = data["candidate_strength"][banks]
    return keys, values, strengths


@torch.no_grad()
def _hard_bank_read(
        model: UnifiedCognitiveController,
        keys: torch.Tensor, values: torch.Tensor,
        strengths: torch.Tensor, queries: torch.Tensor
        ) -> torch.Tensor:
    cosine = torch.einsum(
        "bqw,bkw->bqk",
        torch.nn.functional.normalize(queries, dim=-1),
        torch.nn.functional.normalize(keys, dim=-1))
    ranked = cosine + strengths.clamp_min(1e-6).log().unsqueeze(1)
    scores, selected = ranked.topk(2, dim=-1)
    top = selected[:, :, 0]
    read = torch.gather(
        values, 1,
        top.unsqueeze(-1).expand(-1, -1, values.shape[-1]))
    confidence = torch.gather(
        cosine, 2, top.unsqueeze(-1)).squeeze(-1)
    margin = scores[:, :, 0] - scores[:, :, 1]
    selected_strength = torch.gather(strengths, 1, top)
    occupancy = torch.ones_like(confidence)
    features = torch.stack((
        confidence, margin, selected_strength, occupancy), dim=-1)
    probability = model.memory_read_probability(
        features.reshape(-1, 4)).reshape_as(confidence)
    accepted = probability >= 0.5
    return torch.where(
        accepted.unsqueeze(-1), read, torch.zeros_like(read))


@torch.no_grad()
def _bank_reward(
        model: UnifiedCognitiveController, data: dict[str, object],
        actions: torch.Tensor, *, device: torch.device
        ) -> torch.Tensor:
    keys, values, strengths = _apply_replacement(data, actions)
    memory = _hard_bank_read(
        model, keys, values, strengths, data["future_queries"])
    from .train_adaptive_memory_read import _outcomes
    outcomes = _outcomes(
        model, data["future_batch"],
        memory.reshape(-1, memory.shape[-1]), device=device)
    return outcomes.reshape(actions.shape[0], -1).mean(-1)


@torch.no_grad()
def evaluate_replacement(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device,
        write_threshold: float) -> dict[str, object]:
    model.eval()
    data = replacement_batch(
        model, banks=banks, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold)
    scores = model.memory_replacement_scores(data["option_features"])
    learned = scores.argmax(-1)
    generator = torch.Generator(device=device).manual_seed(
        seed + 40_000_000)
    random = torch.randint(
        0, capacity + 1, (banks,), generator=generator, device=device)
    skip = torch.zeros(banks, dtype=torch.long, device=device)
    fixed = torch.ones(banks, dtype=torch.long, device=device)
    oracle = data["target_action"]
    shuffled_features = data["option_features"].clone()
    shuffled_features[:, 1:, 0] = (
        shuffled_features[:, 1:, 0].roll(1, dims=1))
    shuffled = model.memory_replacement_scores(
        shuffled_features).argmax(-1)
    learned_accuracy = float(_bank_reward(
        model, data, learned, device=device).mean())
    random_accuracy = float(_bank_reward(
        model, data, random, device=device).mean())
    fixed_accuracy = float(_bank_reward(
        model, data, fixed, device=device).mean())
    skip_accuracy = float(_bank_reward(
        model, data, skip, device=device).mean())
    oracle_accuracy = float(_bank_reward(
        model, data, oracle, device=device).mean())
    shuffled_accuracy = float(_bank_reward(
        model, data, shuffled, device=device).mean())
    target_rate = float((learned == oracle).float().mean())
    report = {
        "accuracy": learned_accuracy,
        "target_eviction_rate": target_rate,
        "random_accuracy": random_accuracy,
        "fixed_slot_accuracy": fixed_accuracy,
        "skip_accuracy": skip_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "shuffled_age_feature_accuracy": shuffled_accuracy,
        "replace_rate": float((learned > 0).float().mean()),
        "generated_contexts": data["generated_contexts"],
    }
    def captures_oracle_gap(control_accuracy: float) -> bool:
        available = max(0.0, oracle_accuracy - control_accuracy)
        captured = learned_accuracy - control_accuracy
        return captured + 1e-6 >= 0.75 * available

    report["gate"] = {
        "accuracy_at_least_85": learned_accuracy >= 0.85,
        "within_3_points_of_oracle":
            learned_accuracy >= oracle_accuracy - 0.03,
        "target_eviction_at_least_80": target_rate >= 0.80,
        "captures_75_percent_of_oracle_random_gap":
            captures_oracle_gap(random_accuracy),
        "captures_75_percent_of_oracle_fixed_gap":
            captures_oracle_gap(fixed_accuracy),
        "captures_75_percent_of_oracle_skip_gap":
            captures_oracle_gap(skip_accuracy),
        "age_signal_is_causal":
            shuffled_accuracy <= learned_accuracy - 0.10,
        "oracle_is_solved": oracle_accuracy >= 0.85,
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
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-banks", type=int, default=128)
    parser.add_argument("--test-banks", type=int, default=1024)
    parser.add_argument("--bank-capacity", type=int, default=4)
    parser.add_argument("--rehearsal-capacity", type=int)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--replacement-cost", type=float, default=0.01)
    parser.add_argument("--gate-hidden", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.bank_capacity < 2:
        raise ValueError("bank capacity must be at least two")
    if (
            args.rehearsal_capacity is not None
            and args.rehearsal_capacity < 2):
        raise ValueError("rehearsal capacity must be at least two")

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"])
    configuration["adaptive_memory_replace"] = True
    configuration["adaptive_memory_replace_hidden"] = args.gate_hidden
    model = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = model.load_state_dict(
        payload["state_dict"], strict=False)
    if (
            not all(
                name.startswith("memory_replacement_gate.")
                for name in missing)
            or unexpected):
        raise ValueError(
            f"unexpected checkpoint mismatch: {missing=}, {unexpected=}")
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.memory_replacement_gate is not None
    for parameter in model.memory_replacement_gate.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.memory_replacement_gate.parameters(),
        lr=args.learning_rate)

    baseline = 0.0
    history = []
    generated_contexts = 0
    query_bits = 0
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        training_capacity = (
            args.rehearsal_capacity
            if (
                args.rehearsal_capacity is not None
                and step % 2 == 0)
            else args.bank_capacity)
        data = replacement_batch(
            model, banks=args.batch_banks,
            capacity=training_capacity,
            seed=args.seed * 1_000_000 + step,
            device=device, write_threshold=args.write_threshold)
        generated_contexts += int(data["generated_contexts"])
        query_bits += args.batch_banks * training_capacity
        logits = model.memory_replacement_scores(
            data["option_features"])
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        with torch.no_grad():
            accuracy = _bank_reward(
                model, data, actions, device=device)
        reward = (
            accuracy
            - args.replacement_cost * (actions > 0).to(accuracy.dtype))
        baseline = (
            0.95 * baseline
            + 0.05 * float(reward.detach().mean()))
        loss = -(
            (reward.detach() - baseline)
            * distribution.log_prob(actions)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.memory_replacement_gate.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "future_accuracy": float(accuracy.mean()),
                "target_eviction_rate": float(
                    (actions == data["target_action"]).float().mean()),
                "replace_rate": float((actions > 0).float().mean()),
                "training_capacity": training_capacity,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    replacement_report = evaluate_replacement(
        model, banks=args.test_banks, capacity=args.bank_capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)
    rehearsal_report = (
        evaluate_replacement(
            model, banks=args.test_banks,
            capacity=args.rehearsal_capacity,
            seed=args.seed + 90_500_000, device=device,
            write_threshold=args.write_threshold)
        if args.rehearsal_capacity is not None else None)
    binary = evaluate(
        model, count=2048, trials=6, seed=args.seed + 91_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=2048, trials=6, seed=args.seed + 92_000_000,
        device=device, task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    only_replacement_changed = all(
        name.startswith("memory_replacement_gate.")
        for name in changed)
    admitted = (
        replacement_report["gate"]["accepted"]
        and (
            rehearsal_report is None
            or rehearsal_report["gate"]["accepted"])
        and binary["gate"]["accepted"]
        and four_rule["gate"]["accepted"]
        and only_replacement_changed)
    training_seconds = (
        history[-1]["elapsed_seconds"] if history else 0.0)
    report = {
        "schema": "unified-controller-memory-replacement-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "learner_visible": [
            "controller_created_key_value_latents",
            "row_age", "row_write_strength",
            "candidate_row_similarity", "skip_option",
            "later_scalar_verified_outcomes",
        ],
        "semantic_or_utility_labels_used_for_training": False,
        "training_signal":
            "future_verified_success_minus_generic_replacement_cost",
        "accounting": {
            "generated_support_contexts": generated_contexts,
            "future_query_verifier_bits": query_bits,
            "unique_verifier_bits": generated_contexts + query_bits,
            "unique_logical_lifetimes": generated_contexts,
            "optimizer_updates": args.steps,
            "replayed_examples": 0,
            "training_seconds": training_seconds,
        },
        "history": history,
        "replacement_evaluation": replacement_report,
        "rehearsal_replacement_evaluation": rehearsal_report,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "only_memory_replacement_gate_changed":
            only_replacement_changed,
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
        "replacement_evaluation": replacement_report,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
