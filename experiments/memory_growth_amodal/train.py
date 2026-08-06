"""Qualify variable-capacity append-only memory through the frozen runtime."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    AppendOnlyContentAddressedMemory,
    ControllerFeedback,
    PersistentAppendOnlyContentAddressedMemory,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime(
    *,
    seed: int,
    width: int,
    persistent_path: Path | None = None,
) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=width,
        workspace_slots=2,
        intention_width=5,
        feedback_width=3,
        event_window_capacity=1,
        stable_memory_address=True,
    )
    if persistent_path is None:
        memory = AppendOnlyContentAddressedMemory(
            width=width,
            write_threshold=0.5,
            write_match_threshold=0.999,
            read_match_threshold=0.9,
        )
    else:
        memory = PersistentAppendOnlyContentAddressedMemory(
            width=width,
            path=persistent_path,
            write_threshold=0.5,
            write_match_threshold=0.999,
            read_match_threshold=0.9,
        )
    return AmodalControllerRuntime(controller, memory=memory)


def _feedback(batch: int) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, 3),
        reward=torch.zeros(batch),
        propensity=torch.ones(batch),
        has_feedback=torch.zeros(batch),
    )


def _event(tokens: torch.Tensor) -> AmodalEventCollection:
    return AmodalEventCollection.from_events([AmodalEvent(tokens)])


@torch.no_grad()
def _write_batch(
    runtime: AmodalControllerRuntime,
    tokens: torch.Tensor,
) -> torch.Tensor:
    batch = tokens.shape[0]
    output, _ = runtime.step_events(
        _event(tokens),
        runtime.initial_state(batch, device="cpu"),
        _feedback(batch),
        memory_write_override=torch.ones(batch),
        memory_write_gradient=False,
    )
    if output.controller.memory_write_receipt is None:
        raise RuntimeError("append-only growth write did not return a receipt")
    if not bool(output.controller.memory_write_receipt.committed.all()):
        raise RuntimeError("append-only growth write did not commit every record")
    return output.controller.memory_value.detach().clone()


@torch.no_grad()
def _query_batch(
    runtime: AmodalControllerRuntime,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = tokens.shape[0]
    output, _ = runtime.step_events(
        _event(tokens),
        runtime.initial_state(batch, device="cpu"),
        _feedback(batch),
        memory_write_override=torch.zeros(batch),
        memory_write_gradient=False,
    )
    if output.controller.memory_read is None:
        raise RuntimeError("append-only growth query did not return a memory read")
    return output.controller.memory_read.hit.detach(), output.controller.memory_read.value.detach()


@torch.no_grad()
def _write_dataset(
    runtime: AmodalControllerRuntime,
    tokens: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], batch_size):
        values.append(_write_batch(runtime, tokens[start : start + batch_size]))
    return torch.cat(values, dim=0)


@torch.no_grad()
def _query_dataset(
    runtime: AmodalControllerRuntime,
    tokens: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hits: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for start in range(0, tokens.shape[0], batch_size):
        batch_hits, batch_values = _query_batch(
            runtime, tokens[start : start + batch_size]
        )
        hits.append(batch_hits)
        values.append(batch_values)
    return torch.cat(hits), torch.cat(values)


@torch.no_grad()
def _persistent_audit(
    *,
    seed: int,
    tokens: torch.Tensor,
    expected_values: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="neural-computer-append-only-") as directory:
        path = Path(directory) / "append-only-memory.pt"
        writer = build_runtime(seed=seed, width=tokens.shape[1], persistent_path=path)
        _write_dataset(writer, tokens, batch_size=batch_size)
        intact_payload = torch.load(path, weights_only=False)

        reader = build_runtime(seed=seed, width=tokens.shape[1], persistent_path=path)
        hits, values = _query_dataset(reader, tokens, batch_size=batch_size)
        reload_recall = float(
            (hits & ((values - expected_values).abs().amax(dim=1) <= 1e-5)).float().mean()
        )

        corrupted_payload = dict(intact_payload)
        corrupted_state = dict(intact_payload["state_dict"])
        corrupted_state["values"] = corrupted_state["values"].clone()
        corrupted_state["values"][0, 0] += 0.25
        corrupted_payload["state_dict"] = corrupted_state
        torch.save(corrupted_payload, path)
        corruption_rejected = False
        try:
            PersistentAppendOnlyContentAddressedMemory(
                width=tokens.shape[1],
                path=path,
                write_threshold=0.5,
                write_match_threshold=0.999,
                read_match_threshold=0.9,
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error)

        torch.save(intact_payload, path)
        recovered = build_runtime(seed=seed, width=tokens.shape[1], persistent_path=path)
        recovered_hits, recovered_values = _query_dataset(
            recovered, tokens, batch_size=batch_size
        )
        recovery_recall = float(
            (
                recovered_hits
                & ((recovered_values - expected_values).abs().amax(dim=1) <= 1e-5)
            )
            .float()
            .mean()
        )
    return {
        "reload_recall": reload_recall,
        "corruption_rejected": corruption_rejected,
        "recovery_recall": recovery_recall,
    }


@torch.no_grad()
def run_condition(
    *,
    seed: int,
    record_count: int,
    batch_size: int,
    width: int,
) -> dict[str, Any]:
    seed_everything(seed + record_count)
    tokens = torch.randn(record_count, width)
    runtime = build_runtime(seed=seed, width=width)
    started = time.perf_counter()
    expected_values = _write_dataset(runtime, tokens, batch_size=batch_size)
    write_seconds = time.perf_counter() - started
    if runtime.memory is None or not hasattr(runtime.memory, "record_count"):
        raise RuntimeError("growth runtime does not expose variable-capacity memory")
    record_count_committed = int(runtime.memory.record_count)

    query_started = time.perf_counter()
    permutation = torch.randperm(record_count)
    query_hits, query_values = _query_dataset(
        runtime, tokens[permutation], batch_size=batch_size
    )
    expected_permuted = expected_values[permutation]
    exact = (query_values - expected_permuted).abs().amax(dim=1) <= 1e-5
    permuted_recall = float((query_hits & exact).float().mean())

    fresh_tokens = torch.randn(record_count, width)
    fresh_hits, _ = _query_dataset(runtime, fresh_tokens, batch_size=batch_size)
    query_seconds = time.perf_counter() - query_started
    runtime.memory.clear()
    cleared_hits, _ = _query_dataset(runtime, tokens, batch_size=batch_size)
    persistent = _persistent_audit(
        seed=seed,
        tokens=tokens,
        expected_values=expected_values,
        batch_size=batch_size,
    )
    return {
        "record_count_requested": record_count,
        "record_count_committed": record_count_committed,
        "permuted_recall": permuted_recall,
        "fresh_token_hit_rate": float(fresh_hits.float().mean()),
        "clear_memory_hit_rate": float(cleared_hits.float().mean()),
        "write_seconds": write_seconds,
        "query_latency_ms": 1000.0 * query_seconds / max(record_count * 2, 1),
        "persistent": persistent,
    }


def run_experiment(
    *,
    seed: int,
    record_counts: list[int],
    batch_size: int,
    width: int,
) -> dict[str, Any]:
    conditions = {
        str(record_count): run_condition(
            seed=seed,
            record_count=record_count,
            batch_size=batch_size,
            width=width,
        )
        for record_count in record_counts
    }
    promoted = all(
        condition["record_count_committed"] == int(record_count)
        and condition["permuted_recall"] >= 0.99
        and condition["fresh_token_hit_rate"] <= 0.02
        and condition["clear_memory_hit_rate"] == 0.0
        and condition["persistent"]["reload_recall"] >= 0.99
        and condition["persistent"]["recovery_recall"] >= 0.99
        and condition["persistent"]["corruption_rejected"]
        for record_count, condition in ((int(key), value) for key, value in conditions.items())
    )
    return {
        "schema": "neural-computer.append-only-growth-report.v1",
        "experiment": "append-only-memory-growth-v1",
        "seed": seed,
        "width": width,
        "batch_size": batch_size,
        "record_counts": record_counts,
        "conditions": conditions,
        "accounting": {
            "unique_logical_lifetimes": sum(record_counts),
            "unique_verifier_bits": sum(record_counts),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "controller_frozen": True,
            "raw_modalities_seen_by_controller": False,
        },
        "promoted": promoted,
        "claim_boundary": (
            "Variable-capacity append-only memory through a frozen canonical "
            "controller; no learned compression or general continual-learning claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-counts", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if any(count < 1 for count in args.record_counts):
        raise SystemExit("record counts must be positive")
    if args.batch_size < 1 or args.width < 1:
        raise SystemExit("batch size and width must be positive")
    report = run_experiment(
        seed=args.seed,
        record_counts=args.record_counts,
        batch_size=args.batch_size,
        width=args.width,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
