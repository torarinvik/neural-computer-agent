"""Audit raw-event shortcutting in a persisted external capability pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.external_capability_composition_amodal.train import (
    _composition_accuracy,
    _pipeline,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    ACTION_WIDTH,
    DECODER_HIDDEN,
    INTENTION_WIDTH,
    _new_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import (
    ExternalCapabilityPipeline,
    OpaqueProtocolDecoder,
    PersistentOpaqueStateStore,
)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("configuration"), dict
    ):
        raise TypeError(f"invalid opaque state manifest: {path}")
    store = PersistentOpaqueStateStore(
        path,
        configuration=payload["configuration"],
    )
    return store.load()


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.parent_updates, args.batch_size, args.audit_count) < 1:
        raise ValueError("parent updates, batch size, and audit count must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")

    parent = _runtime(seed=args.seed, growth=False)
    _, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=max(args.parent_updates, 1),
    )
    pipeline_state = _load_state(args.pipeline_state)
    decoder_state = _load_state(args.decoder_state)
    normal = _pipeline(
        tuple(_new_capability(args.seed + 20 + index)[0] for index in range(2))
    )
    head_only = ExternalCapabilityPipeline(
        tuple(_new_capability(args.seed + 20 + index)[0] for index in range(2)),
        hide_downstream_events=True,
    )
    normal.load_state_dict(pipeline_state, strict=True)
    head_only.load_state_dict(pipeline_state, strict=True)
    decoder = OpaqueProtocolDecoder(
        INTENTION_WIDTH,
        ACTION_WIDTH,
        hidden=DECODER_HIDDEN,
    )
    decoder.load_state_dict(decoder_state, strict=True)
    normal.eval()
    head_only.eval()
    decoder.eval()
    normal_accuracy = _composition_accuracy(
        parent,
        normal,
        decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    head_only_accuracy = _composition_accuracy(
        parent,
        head_only,
        decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    report: dict[str, object] = {
        "schema": "neural-computer.external-capability-event-visibility-audit.v1",
        "claim_boundary": (
            "A persisted composition is evaluated with all programs seeing the "
            "learned event and again with only the first program seeing it. "
            "The latter is a shortcut diagnostic, not a new training result."
        ),
        "seed": args.seed,
        "pipeline_state": str(args.pipeline_state),
        "decoder_state": str(args.decoder_state),
        "normal_accuracy": normal_accuracy,
        "head_only_accuracy": head_only_accuracy,
        "head_only_drop": normal_accuracy - head_only_accuracy,
        "accounting": {
            "unique_verifier_bits": args.parent_updates * args.batch_size * 2,
            "diagnostic_verifier_bits": args.audit_count * 4 * 2,
            "unique_logical_lifetimes": args.parent_updates * args.batch_size,
            "optimizer_updates": args.parent_updates,
            "replayed_examples": 0,
        },
        "parent_mastered": bool(parent_progress)
        and parent_progress[-1]["heldout_accuracy"] >= 0.75,
        "head_only_composition_preserved": head_only_accuracy >= 0.75,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-state", type=Path, required=True)
    parser.add_argument("--decoder-state", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69317)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "normal_accuracy": report["normal_accuracy"],
                "head_only_accuracy": report["head_only_accuracy"],
                "head_only_drop": report["head_only_drop"],
                "parent_mastered": report["parent_mastered"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
