"""Qualify the external temporal-history memory contract.

This is a storage and ABI diagnostic, not a learned-capability promotion. It
checks that learned event tensors can grow beyond a fixed hot window, remain
bound to independent scopes, survive a checksummed payload round trip, and
represent missing history explicitly. No verifier target, rule family, or
correct action is supplied to the memory component.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import ExternalTemporalHistoryMemory

TEMPORAL_MEMORY_CONTRACT_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-memory-contract.v1"
)


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.width, args.records, args.query_count) < 1:
        raise ValueError("temporal memory dimensions must be positive")
    started = perf_counter()
    generator = torch.Generator().manual_seed(args.seed)
    memory = ExternalTemporalHistoryMemory(args.width, scope_capacity=2)
    history: list[torch.Tensor] = []
    for _ in range(args.records):
        values = torch.randn(2, args.width, generator=generator)
        history.append(values)
        memory.append(values, scope=torch.tensor([0, 1], dtype=torch.long))

    requested = torch.tensor(
        [
            [0, 1, min(7, args.records - 1), args.records - 1],
            [0, 1, min(7, args.records - 1), args.records - 1],
        ],
        dtype=torch.long,
    )
    if args.query_count != requested.shape[1]:
        requested = requested[:, : args.query_count]
    read = memory.read_relative(
        requested,
        scope=torch.tensor([0, 1], dtype=torch.long),
    )
    expected = torch.stack(
        [
            torch.stack(
                [history[args.records - 1 - int(offset)][scope_index] for offset in row]
            )
            for scope_index, row in enumerate(requested)
        ],
        dim=0,
    )
    max_difference = float((read.values - expected).abs().max())

    restored = ExternalTemporalHistoryMemory.from_payload(memory.payload())
    restored_read = restored.read_relative(
        requested,
        scope=torch.tensor([0, 1], dtype=torch.long),
    )
    restored_difference = float((restored_read.values - read.values).abs().max())

    corrupted = memory.payload()
    state = corrupted["state"]
    if not isinstance(state, dict):
        raise TypeError("temporal memory payload state is not mutable")
    state["values"] = state["values"].clone()
    state["values"][0, 0] += 1.0
    try:
        ExternalTemporalHistoryMemory.from_payload(corrupted)
    except ValueError as error:
        checksum_rejected = "checksum" in str(error)
    else:
        checksum_rejected = False

    restored.clear(torch.tensor([0], dtype=torch.long))
    scope0_after_clear = restored.read_relative(
        torch.tensor([[0]], dtype=torch.long),
        scope=torch.tensor([0], dtype=torch.long),
    )
    scope1_after_clear = restored.read_relative(
        torch.tensor([[0]], dtype=torch.long),
        scope=torch.tensor([1], dtype=torch.long),
    )
    gates = {
        "grows_beyond_fixed_capacity": memory.record_count == args.records * 2,
        "distant_relative_reads_exact": max_difference == 0.0,
        "missing_history_is_explicit": not bool(
            memory.read_relative(
                torch.tensor([[args.records]], dtype=torch.long),
                scope=torch.tensor([0], dtype=torch.long),
            ).present.any()
        ),
        "payload_reload_exact": restored_difference == 0.0,
        "scope_clear_isolated": (
            not bool(scope0_after_clear.present.any())
            and bool(scope1_after_clear.present.all())
        ),
        "corrupted_payload_rejected": checksum_rejected,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": TEMPORAL_MEMORY_CONTRACT_SCHEMA,
        "claim_boundary": (
            "External append-only temporal memory ABI qualification only; "
            "no learned addressing, learned capability gain, or general "
            "continual-learning claim."
        ),
        "architecture": {
            "memory": "external_temporal_history_memory_v1",
            "storage": "append_only_scoped_records",
            "addressing": "opaque_relative_offsets",
            "missing_history": "explicit_present_mask",
            "controller_parameters_touched": False,
        },
        "seed": args.seed,
        "width": args.width,
        "records_per_scope": args.records,
        "record_count": memory.record_count,
        "max_read_difference": max_difference,
        "reload_difference": restored_difference,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": 0,
            "unique_logical_lifetimes": 0,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_memory_contract" if all(gates.values()) else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
