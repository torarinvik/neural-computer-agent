"""Probe causal outcome attribution with unequal memory admission strengths.

The same content query is intentionally redirected by a write-strength prior
when outcomes are re-resolved after the read.  A physical receipt fixes the
causal row at read time.  This is an infrastructure diagnostic: row identities
are private to the verifier and no semantic label is supplied to a learner.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from .memory import DiskLatentMemory


def _new_memory() -> tuple[DiskLatentMemory, torch.Tensor]:
    memory = DiskLatentMemory(width=4, capacity=2)
    # The stable row is an exact content match but has low admission strength.
    # The decoy is slightly less similar and has a strong prior, so ordinary
    # strength-weighted re-resolution selects the wrong physical row.
    keys = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.10, 0.0, 0.0],
    ])
    values = torch.eye(4)[:2]
    memory.commit(
        keys, values, torch.tensor([0.05, 1.0]), threshold=0.0)
    return memory, keys[:1]


def _apply_history(policy: str, events: int) -> DiskLatentMemory:
    if policy not in {"ordinary", "receipt", "shuffled_receipt"}:
        raise ValueError("unknown attribution policy")
    memory, query = _new_memory()
    outcomes = torch.ones(events)
    if policy == "ordinary":
        # This is the old interface: it resolves the row again using the
        # strength prior and therefore credits the high-strength decoy.
        memory.store.record_outcomes(
            query.repeat(events, 1), outcomes, update_volatility=True,
            success_protection_rate=0.2, failure_thaw_rate=0.0,
            stale_thaw_rate=0.0, usage_prior_scale=1.0)
    else:
        for _ in range(events):
            _, _, receipt = memory.retrieve_with_receipt(
                query, top_k=1, confidence_mode="cosine",
                usage_prior_scale=0.0)
            if policy == "shuffled_receipt":
                receipt = receipt.new_tensor([1])
            memory.record_outcomes_from_receipts(
                receipt, torch.ones(1), update_volatility=True,
                success_protection_rate=0.2, failure_thaw_rate=0.0,
                stale_thaw_rate=0.0)
    return memory


def run_trial(events: int = 8) -> dict[str, object]:
    ordinary = _apply_history("ordinary", events)
    receipt = _apply_history("receipt", events)
    shuffled = _apply_history("shuffled_receipt", events)

    # Persist the receipt-corrected bank before making the replacement choice.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "receipt-bank.pt"
        receipt.save(path)
        restored = DiskLatentMemory.load(path)

    ordinary_choice = int(ordinary.store.volatility.argmax())
    receipt_choice = int(restored.store.volatility.argmax())
    shuffled_choice = int(shuffled.store.volatility.argmax())
    return {
        "events": events,
        "unequal_admission_strengths": True,
        "ordinary_volatility": ordinary.store.volatility.tolist(),
        "receipt_volatility": restored.store.volatility.tolist(),
        "shuffled_receipt_volatility": shuffled.store.volatility.tolist(),
        "ordinary_selected_row": ordinary_choice,
        "receipt_selected_row": receipt_choice,
        "shuffled_receipt_selected_row": shuffled_choice,
        "ordinary_evicted_stable": ordinary_choice == 0,
        "receipt_evicted_stable": receipt_choice == 0,
        "shuffled_receipt_evicted_stable": shuffled_choice == 0,
        "disk_round_trip_exact": torch.equal(
            restored.store.volatility,
            receipt.store.volatility),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.events < 1:
        raise ValueError("events must be positive")
    result = {
        "schema": "causal-memory-receipt-probe-v1",
        "learner_visible": [
            "latent key", "latent value", "retrieval confidence",
            "physical read receipt"],
        "semantic_or_task_labels_visible_to_learner": False,
        "trial": run_trial(args.events),
    }
    trial = result["trial"]
    result["gates"] = {
        "ordinary_re_resolve_evicts_stable":
            trial["ordinary_evicted_stable"],
        "receipt_protects_stable":
            not trial["receipt_evicted_stable"],
        "shuffled_receipt_breaks_protection":
            trial["shuffled_receipt_evicted_stable"],
        "disk_round_trip_exact": trial["disk_round_trip_exact"],
    }
    result["gates"]["accepted"] = all(result["gates"].values())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
