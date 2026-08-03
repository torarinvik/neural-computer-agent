"""Audit learned admission through actual disk save/reload cycles."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _support(
        model: UnifiedCognitiveController, batch, *,
        device: torch.device, retrieved: torch.Tensor | None = None,
        support_trials: int = 1,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one or more feedback-bearing supports into a memory value.

    A support action's reward arrives alongside the following frame. Hence a
    lifetime with ``n`` supports needs one feedback frame and a separate query
    frame after them. Defaults preserve the historical one-support protocol.
    """
    if support_trials < 1 or batch.trials < support_trials + 2:
        raise ValueError("batch needs support trials plus feedback and query frames")
    count = batch.batch_size
    state = model.initial_state(count, device=device)
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    action, outcome, feedback = null_action, zeros, zeros
    output0 = None
    for trial in range(support_trials):
        output, state = model.step(
            batch.frames[:, trial], state, action, outcome, feedback,
            retrieved_memory=retrieved if trial == 0 else None)
        if output0 is None:
            output0 = output
        action = output.logits.argmax(-1)
        outcome = (action == batch.correct_actions[:, trial]).to(torch.float32)
        feedback = torch.ones_like(outcome)
    output_after_feedback, _ = model.step(
        batch.frames[:, support_trials], state, action, outcome, feedback)
    assert output0 is not None
    return (
        output0.memory_key,
        output_after_feedback.memory_value,
        output_after_feedback.memory_write_strength)


@torch.no_grad()
def _query_keys(
        model: UnifiedCognitiveController, batch, *,
        device: torch.device, query_trial: int = 2) -> torch.Tensor:
    count = batch.batch_size
    state = model.initial_state(count, device=device)
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    output, _ = model.step(
        batch.frames[:, query_trial], state, null_action, zeros, zeros)
    return output.memory_key


@torch.no_grad()
def _query_accuracy(
        model: UnifiedCognitiveController, batch, memory: torch.Tensor, *,
        device: torch.device) -> float:
    count = batch.batch_size
    state = model.initial_state(count, device=device)
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    output, _ = model.step(
        batch.frames[:, 2], state, null_action, zeros, zeros,
        retrieved_memory=memory)
    return float(
        (output.logits.argmax(-1) == batch.correct_actions[:, 2])
        .to(torch.float32).mean())


def _disk_bytes(directory: Path) -> int:
    return sum(
        path.stat().st_size for path in directory.glob("*.pt"))


@torch.no_grad()
def _gated_retrieve(
        model: UnifiedCognitiveController, memory: DiskLatentMemory,
        queries: torch.Tensor, *,
        read_threshold: float | None,
        usage_prior_scale: torch.Tensor | float | None = None) -> torch.Tensor:
    if usage_prior_scale is None:
        usage_prior_scale = model.effective_memory_usage_prior_scale()
    if model.memory_read_gate is not None:
        read, features = memory.retrieve_with_features(
            queries, usage_prior_scale=usage_prior_scale)
        accepted = model.memory_read_probability(features) >= 0.5
        return torch.where(
            accepted.unsqueeze(-1), read, torch.zeros_like(read))
    read, confidence = memory.retrieve(
        queries, top_k=1, confidence_mode="cosine",
        usage_prior_scale=usage_prior_scale)
    if read_threshold is not None:
        read = torch.where(
            (confidence >= read_threshold).unsqueeze(-1),
            read, torch.zeros_like(read))
    return read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--contexts", type=int, default=512)
    parser.add_argument("--bank-capacity", type=int, default=8)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--read-threshold", type=float)
    parser.add_argument(
        "--usage-prior-scale", type=float,
        help="override the controller/default write-strength retrieval prior")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.contexts % args.bank_capacity:
        raise ValueError("contexts must divide into complete memory banks")
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    read_threshold = (
        args.read_threshold
        if args.read_threshold is not None
        else payload.get("memory_read_threshold"))
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    usage_prior_scale = (
        args.usage_prior_scale
        if args.usage_prior_scale is not None
        else float(
            model.effective_memory_usage_prior_scale().detach()))
    batch = _add_context_signatures(
        generate_lifetimes(
            args.contexts, 3, seed=args.seed, heldout=True,
            task="binary_mapping", support_trials=1, device=device),
        seed=args.seed + 10_000_000)
    first_key, first_value, first_strength = _support(
        model, batch, device=device)
    query_key = _query_keys(model, batch, device=device)
    decision = (first_strength >= args.write_threshold).to(
        first_value.dtype)
    tensor_sparse_memory = first_value * decision.unsqueeze(-1)
    tensor_sparse_accuracy = _query_accuracy(
        model, batch, tensor_sparse_memory, device=device)
    no_memory_accuracy = _query_accuracy(
        model, batch, torch.zeros_like(first_value), device=device)

    bank_count = args.contexts // args.bank_capacity
    first_reads = []
    repeat_reads = []
    final_reads = []
    corrupted_reads = []
    first_rows = 0
    final_rows = 0
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        memories = []
        for bank in range(bank_count):
            start = bank * args.bank_capacity
            stop = start + args.bank_capacity
            memory = DiskLatentMemory(
                model.width, capacity=args.bank_capacity, device=device)
            memory.commit(
                first_key[start:stop], first_value[start:stop],
                first_strength[start:stop],
                threshold=args.write_threshold)
            path = directory / f"bank-{bank:04d}.pt"
            memory.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            read = _gated_retrieve(
                model, restored, query_key[start:stop],
                read_threshold=read_threshold,
                usage_prior_scale=usage_prior_scale)
            first_reads.append(read)
            first_rows += restored.count
            memories.append(restored)
        first_disk_bytes = _disk_bytes(directory)
        first_disk_accuracy = _query_accuracy(
            model, batch, torch.cat(first_reads), device=device)

        # Preserve keys and admission statistics but rotate values within each
        # bank. Successful queries must depend on the correct stored content.
        corrupt_directory = directory / "corrupt"
        corrupt_directory.mkdir()
        for bank in range(bank_count):
            start = bank * args.bank_capacity
            stop = start + args.bank_capacity
            corrupt = DiskLatentMemory(
                model.width, capacity=args.bank_capacity, device=device)
            corrupt.commit(
                first_key[start:stop],
                first_value[start:stop].roll(1, dims=0),
                first_strength[start:stop],
                threshold=args.write_threshold)
            path = corrupt_directory / f"bank-{bank:04d}.pt"
            corrupt.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            corrupted_reads.append(_gated_retrieve(
                model, restored, query_key[start:stop],
                read_threshold=read_threshold,
                usage_prior_scale=usage_prior_scale))
        corrupted_disk_accuracy = _query_accuracy(
            model, batch, torch.cat(corrupted_reads), device=device)

        # A repeat must first recall from disk before deciding whether another
        # physical row deserves admission.
        for bank, memory in enumerate(memories):
            start = bank * args.bank_capacity
            stop = start + args.bank_capacity
            read = _gated_retrieve(
                model, memory, first_key[start:stop],
                read_threshold=read_threshold,
                usage_prior_scale=usage_prior_scale)
            repeat_reads.append(read)
        _, repeat_value, repeat_strength = _support(
            model, batch, device=device,
            retrieved=torch.cat(repeat_reads))
        for bank, memory in enumerate(memories):
            start = bank * args.bank_capacity
            stop = start + args.bank_capacity
            memory.commit(
                first_key[start:stop], repeat_value[start:stop],
                repeat_strength[start:stop],
                threshold=args.write_threshold)
            path = directory / f"bank-{bank:04d}.pt"
            memory.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            read = _gated_retrieve(
                model, restored, query_key[start:stop],
                read_threshold=read_threshold,
                usage_prior_scale=usage_prior_scale)
            final_reads.append(read)
            final_rows += restored.count
        final_disk_bytes = _disk_bytes(directory)
        repeat_disk_accuracy = _query_accuracy(
            model, batch, torch.cat(final_reads), device=device)

    duplicate_rows = final_rows - first_rows
    duplicate_rate = duplicate_rows / args.contexts
    report = {
        "schema": "unified-controller-selective-disk-audit-v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "seed": args.seed,
        "contexts": args.contexts,
        "bank_capacity": args.bank_capacity,
        "banks": bank_count,
        "write_threshold": args.write_threshold,
        "read_threshold": read_threshold,
        "usage_prior_scale": usage_prior_scale,
        "read_policy": (
            "adaptive_controller_gate"
            if model.memory_read_gate is not None
            else (
                "scalar_cosine_threshold"
                if read_threshold is not None else "ungated")),
        "weights_changed": False,
        "semantic_labels_used": False,
        "first_rows": first_rows,
        "final_rows": final_rows,
        "duplicate_rows_added": duplicate_rows,
        "duplicate_rows_per_context": duplicate_rate,
        "first_disk_bytes": first_disk_bytes,
        "final_disk_bytes": final_disk_bytes,
        "no_memory_accuracy": no_memory_accuracy,
        "tensor_sparse_accuracy": tensor_sparse_accuracy,
        "first_disk_reload_accuracy": first_disk_accuracy,
        "corrupted_value_disk_accuracy": corrupted_disk_accuracy,
        "repeat_disk_reload_accuracy": repeat_disk_accuracy,
    }
    report["gate"] = {
        "tensor_policy_at_least_95":
            tensor_sparse_accuracy >= 0.95,
        "first_disk_reload_at_least_85":
            first_disk_accuracy >= 0.85,
        "repeat_disk_reload_at_least_85":
            repeat_disk_accuracy >= 0.85,
        "duplicate_rows_at_most_20_percent":
            duplicate_rate <= 0.20,
        "memory_is_causal":
            no_memory_accuracy <= first_disk_accuracy - 0.15,
        "correct_memory_content_is_causal":
            corrupted_disk_accuracy <= first_disk_accuracy - 0.15,
    }
    report["gate"]["accepted"] = all(report["gate"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
