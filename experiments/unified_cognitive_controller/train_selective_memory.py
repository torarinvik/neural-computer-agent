"""Reward-train the first memory-admission atom: write once, skip a repeat."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from .environment import NULL_ACTION, generate_lifetimes
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import evaluate, seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bernoulli_log_probability(
        probability: torch.Tensor, decision: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    return (
        decision * probability.log()
        + (1.0 - decision) * (1.0 - probability).log())


def selective_rollout(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, sample_writes: bool,
        hide_repeat_read: bool = False,
        occupied_repeat_curriculum: bool = False,
        first_write_mode: str = "learned",
        repeat_write_mode: str = "learned",
        corrupt_first_memory: bool = False
        ) -> dict[str, torch.Tensor]:
    batch = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    ones = torch.ones(count, device=device)

    def support(recalled: torch.Tensor | None = None):
        state = model.initial_state(count, device=device)
        output0, state = model.step(
            batch.frames[:, 0], state, null_action, zeros, zeros,
            retrieved_memory=recalled)
        action = output0.logits.argmax(-1)
        outcome = (
            action == batch.correct_actions[:, 0]).to(torch.float32)
        output1, _ = model.step(
            batch.frames[:, 1], state, action, outcome, ones)
        return output1

    def decide(
            probability: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "all":
            return torch.ones_like(probability)
        if mode == "none":
            return torch.zeros_like(probability)
        if sample_writes:
            decision = torch.bernoulli(probability)
        else:
            decision = (probability >= 0.5).to(probability.dtype)
        if mode == "shuffled":
            return decision.roll(1)
        if mode != "learned":
            raise ValueError("unsupported write mode")
        return decision

    def query(memory: torch.Tensor) -> torch.Tensor:
        fresh = model.initial_state(count, device=device)
        output, _ = model.step(
            batch.frames[:, 2], fresh, null_action, zeros, zeros,
            retrieved_memory=memory)
        action = output.logits.argmax(-1)
        return (
            action == batch.correct_actions[:, 2]).to(torch.float32)

    first = support()
    first_write = decide(
        first.memory_write_strength, first_write_mode)
    first_value = (
        first.memory_value.flip(-1)
        if corrupt_first_memory else first.memory_value)
    memory1 = first_value * first_write.unsqueeze(-1)
    query1_outcome = query(memory1)

    repeat_existing = (
        first.memory_value if occupied_repeat_curriculum else memory1)
    repeat_read = (
        torch.zeros_like(repeat_existing)
        if hide_repeat_read else repeat_existing)
    repeat = support(repeat_read)
    repeat_write = decide(
        repeat.memory_write_strength, repeat_write_mode)
    memory2 = torch.where(
        repeat_write.bool().unsqueeze(-1),
        repeat.memory_value, repeat_existing)
    query2_outcome = query(memory2)
    return {
        "first_probability": first.memory_write_strength,
        "repeat_probability": repeat.memory_write_strength,
        "first_write": first_write,
        "repeat_write": repeat_write,
        "query1_outcome": query1_outcome,
        "query2_outcome": query2_outcome,
    }


@torch.no_grad()
def evaluate_selective(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device) -> dict[str, object]:
    model.eval()
    normal = selective_rollout(
        model, count=count, seed=seed, device=device,
        sample_writes=False)
    hidden = selective_rollout(
        model, count=count, seed=seed, device=device,
        sample_writes=False, hide_repeat_read=True)
    no_writes = selective_rollout(
        model, count=count, seed=seed, device=device,
        sample_writes=False, first_write_mode="none",
        repeat_write_mode="none")
    shuffled = selective_rollout(
        model, count=count, seed=seed, device=device,
        sample_writes=False, first_write_mode="shuffled")
    corrupted = selective_rollout(
        model, count=count, seed=seed, device=device,
        sample_writes=False, corrupt_first_memory=True)
    first_rate = float(normal["first_write"].mean())
    repeat_rate = float(normal["repeat_write"].mean())
    hidden_repeat_rate = float(hidden["repeat_write"].mean())
    query1 = float(normal["query1_outcome"].mean())
    query2 = float(normal["query2_outcome"].mean())
    report = {
        "first_write_rate": first_rate,
        "repeat_write_rate": repeat_rate,
        "hidden_read_repeat_write_rate": hidden_repeat_rate,
        "query_after_first_accuracy": query1,
        "query_after_repeat_accuracy": query2,
        "no_write_query_accuracy": float(
            no_writes["query1_outcome"].mean()),
        "shuffled_admission_query_accuracy": float(
            shuffled["query1_outcome"].mean()),
        "corrupted_value_query_accuracy": float(
            corrupted["query1_outcome"].mean()),
        "writes_per_context": first_rate + repeat_rate,
        "first_probability_mean": float(
            normal["first_probability"].mean()),
        "repeat_probability_mean": float(
            normal["repeat_probability"].mean()),
    }
    report["gate"] = {
        "first_write_at_most_80": first_rate <= 0.80,
        "repeat_write_at_most_20": repeat_rate <= 0.20,
        "first_query_at_least_95": query1 >= 0.95,
        "repeat_query_at_least_95": query2 >= 0.95,
        "at_most_1_2_writes": first_rate + repeat_rate <= 1.20,
        "recall_causally_controls_skip":
            hidden_repeat_rate >= repeat_rate + 0.30,
        "no_writes_hurt": (
            float(no_writes["query1_outcome"].mean())
            <= query1 - 0.20),
        "shuffled_admission_hurts": (
            float(shuffled["query1_outcome"].mean())
            <= query1 - 0.10),
        "corrupted_values_hurt": (
            float(corrupted["query1_outcome"].mean())
            <= query1 - 0.15),
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
    parser.add_argument("--seed", type=int, default=5401)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-contexts", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--write-cost", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.test_contexts < 1:
        raise ValueError("steps and context counts must be positive")
    if not 0.0 < args.write_cost < 0.5:
        raise ValueError("write cost must be between zero and 0.5")

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.memory_write.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.memory_write.parameters(), lr=args.learning_rate,
        weight_decay=1e-5)

    baseline1 = 0.0
    baseline2 = 0.0
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        result = selective_rollout(
            model, count=args.batch_size,
            seed=args.seed * 1_000_000 + step,
            device=device, sample_writes=True,
            occupied_repeat_curriculum=True)
        reward1 = (
            result["query1_outcome"]
            - args.write_cost * result["first_write"])
        reward2 = (
            result["query2_outcome"]
            - args.write_cost * result["repeat_write"])
        baseline1 = (
            0.95 * baseline1
            + 0.05 * float(reward1.detach().mean()))
        baseline2 = (
            0.95 * baseline2
            + 0.05 * float(reward2.detach().mean()))
        loss = -(
            (reward1.detach() - baseline1)
            * _bernoulli_log_probability(
                result["first_probability"], result["first_write"])
            + (reward2.detach() - baseline2)
            * _bernoulli_log_probability(
                result["repeat_probability"], result["repeat_write"])
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.memory_write.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "first_write_rate": float(
                    result["first_write"].detach().mean()),
                "repeat_write_rate": float(
                    result["repeat_write"].detach().mean()),
                "query1_accuracy": float(
                    result["query1_outcome"].detach().mean()),
                "query2_accuracy": float(
                    result["query2_outcome"].detach().mean()),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    selective = evaluate_selective(
        model, count=args.test_contexts, seed=args.seed + 90_000_000,
        device=device)
    binary = evaluate(
        model, count=2048, trials=6, seed=args.seed + 91_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=2048, trials=6, seed=args.seed + 92_000_000,
        device=device, task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    only_write_gate_changed = all(
        name.startswith("memory_write.") for name in changed)
    admitted = (
        selective["gate"]["accepted"]
        and binary["gate"]["accepted"]
        and four_rule["gate"]["accepted"]
        and only_write_gate_changed)
    report = {
        "schema": "unified-controller-selective-memory-atom-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "learner_visible": [
            "rendered_rgb_frame", "own_previous_opaque_action",
            "scalar_verified_outcome", "own_latent_active_state",
            "content_addressed_external_latent_read",
        ],
        "semantic_labels_used_for_training": False,
        "novel_or_repeat_labels_used_for_training": False,
        "occupied_repeat_curriculum": True,
        "write_reward": "later_verified_success_minus_generic_write_cost",
        "unique_contexts_seen": args.steps * args.batch_size,
        "history": history,
        "selective_evaluation": selective,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "only_write_gate_changed": only_write_gate_changed,
        "all_admission_gates_passed": admitted,
        "total_seconds": time.perf_counter() - started,
    }
    if admitted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": payload["model_configuration"],
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
        "selective_evaluation": selective,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
