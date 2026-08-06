"""Train an external consumer under opaque intermediate-only visibility."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch

from experiments.external_capability_composition_amodal.train import (
    _composition_accuracy,
    _stable_bits,
    _train_composition,
    _zero_program,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    ACTION_WIDTH,
    DECODER_HIDDEN,
    EVENT_WIDTH,
    INTENTION_WIDTH,
    _capability_accuracy,
    _new_capability,
    _train_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import (
    ExternalCapabilityPipeline,
    OpaqueProtocolDecoder,
    PersistentOpaqueStateStore,
)


def _digest_core(runtime) -> str:
    digest = hashlib.sha256()
    for name, value in runtime.controller.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _head_only_pipeline(head, consumer) -> ExternalCapabilityPipeline:
    return ExternalCapabilityPipeline(
        (head, consumer),
        hide_downstream_events=True,
    )


def _decoder() -> OpaqueProtocolDecoder:
    return OpaqueProtocolDecoder(
        INTENTION_WIDTH,
        ACTION_WIDTH,
        hidden=DECODER_HIDDEN,
    )


def _persist_and_reload(
    pipeline: ExternalCapabilityPipeline,
    decoder: OpaqueProtocolDecoder,
    directory: Path,
) -> tuple[ExternalCapabilityPipeline, OpaqueProtocolDecoder, bool, bool]:
    if directory.exists():
        shutil.rmtree(directory)
    pipeline_store = PersistentOpaqueStateStore(
        directory / "pipeline.pt",
        configuration=pipeline.configuration(),
    )
    decoder_store = PersistentOpaqueStateStore(
        directory / "decoder.pt",
        configuration={
            "component": "intermediate-consumer-decoder",
            "schema": "neural-computer.opaque-protocol-decoder.v1",
            "intention_width": INTENTION_WIDTH,
            "action_width": ACTION_WIDTH,
            "hidden": DECODER_HIDDEN,
        },
    )
    pipeline_digest = pipeline_store.save_module(pipeline)
    decoder_digest = decoder_store.save_module(decoder)
    reloaded_pipeline = _head_only_pipeline(
        _new_capability(91)[0],
        _new_capability(92)[0],
    )
    reloaded_decoder = _decoder()
    reloaded_pipeline_digest = pipeline_store.load_module(reloaded_pipeline)
    reloaded_decoder_digest = decoder_store.load_module(reloaded_decoder)
    pipeline_path = directory / "pipeline.pt"
    intact = pipeline_path.read_bytes()
    payload = torch.load(pipeline_path, weights_only=False)
    state = dict(payload["state_dict"])
    name = next(iter(state))
    changed = state[name].clone()
    changed.reshape(-1)[0] += 1.0
    state[name] = changed
    payload["state_dict"] = state
    torch.save(payload, pipeline_path)
    corruption_rejected = False
    try:
        pipeline_store.load()
    except ValueError as error:
        corruption_rejected = "checksum mismatch" in str(error)
    pipeline_path.write_bytes(intact)
    return (
        reloaded_pipeline,
        reloaded_decoder,
        pipeline_digest == reloaded_pipeline_digest
        and decoder_digest == reloaded_decoder_digest,
        corruption_rejected,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if (
        min(
            args.parent_updates,
            args.head_updates,
            args.consumer_updates,
            args.batch_size,
            args.audit_count,
            args.eval_every,
        )
        < 1
    ):
        raise ValueError("all update and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
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
        eval_every=args.eval_every,
    )
    parent.eval()
    core_before = _digest_core(parent)

    head, head_decoder = _new_capability(args.seed + 1)
    _head_history, head_progress = _train_capability(
        parent,
        head,
        head_decoder,
        operation="complement",
        span=4,
        updates=args.head_updates,
        batch_size=args.batch_size,
        seed=args.seed + 1_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
    )
    head.eval()
    head_decoder.eval()

    consumer = _new_capability(args.seed + 2)[0]
    pipeline = _head_only_pipeline(head, consumer)
    for parameter in pipeline.programs[0].parameters():
        parameter.requires_grad_(False)
    consumer_decoder = _decoder()
    torch.manual_seed(args.seed + 90_001)
    consumer_history, consumer_progress = _train_composition(
        parent,
        pipeline,
        consumer_decoder,
        updates=args.consumer_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=True,
    )

    fresh_pipeline = _head_only_pipeline(
        _new_capability(args.seed + 10)[0],
        _new_capability(args.seed + 11)[0],
    )
    fresh_decoder = _decoder()
    torch.manual_seed(args.seed + 90_002)
    fresh_history, fresh_progress = _train_composition(
        parent,
        fresh_pipeline,
        fresh_decoder,
        updates=args.consumer_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=True,
    )

    blank_pipeline = ExternalCapabilityPipeline(
        event_width=EVENT_WIDTH,
        action_width=ACTION_WIDTH,
        intention_width=INTENTION_WIDTH,
    )
    blank_decoder = _decoder()
    torch.manual_seed(args.seed + 90_003)
    blank_history, blank_progress = _train_composition(
        parent,
        blank_pipeline,
        blank_decoder,
        updates=args.consumer_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=False,
    )

    shuffled_pipeline = copy.deepcopy(pipeline)
    shuffled_decoder = _decoder()
    torch.manual_seed(args.seed + 90_004)
    _, shuffled_progress = _train_composition(
        parent,
        shuffled_pipeline,
        shuffled_decoder,
        updates=args.consumer_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=False,
        shuffle_outcomes=True,
    )

    target_accuracy = _composition_accuracy(
        parent,
        pipeline,
        consumer_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    fresh_accuracy = _composition_accuracy(
        parent,
        fresh_pipeline,
        fresh_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    blank_accuracy = _composition_accuracy(
        parent,
        blank_pipeline,
        blank_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    shuffled_accuracy = _composition_accuracy(
        parent,
        shuffled_pipeline,
        shuffled_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    zero_head_accuracy = _composition_accuracy(
        parent,
        _zero_program(pipeline, 0),
        consumer_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    zero_consumer_accuracy = _composition_accuracy(
        parent,
        _zero_program(pipeline, 1),
        consumer_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    reloaded_pipeline, reloaded_decoder, reload_exact, corruption_rejected = (
        _persist_and_reload(
            pipeline, consumer_decoder, args.report_out.parent / "state"
        )
    )
    reloaded_accuracy = _composition_accuracy(
        parent,
        reloaded_pipeline,
        reloaded_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    core_after = _digest_core(parent)
    bits_per_update = args.batch_size * 4
    head_stable = _stable_bits(
        head_progress,
        threshold=args.mastery_threshold,
        bits_per_update=args.batch_size * 4,
    )
    consumer_stable = _stable_bits(
        consumer_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    fresh_stable = _stable_bits(
        fresh_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    report: dict[str, object] = {
        "schema": "neural-computer.external-capability-intermediate-consumer-report.v1",
        "claim_boundary": (
            "A head external program is acquired first, then a downstream "
            "consumer is trained while downstream raw events are hidden. "
            "This tests an opaque intermediate execution contract, not "
            "arbitrary program induction."
        ),
        "seed": args.seed,
        "head_stable_bits_to_threshold": head_stable,
        "head_accuracy": _capability_accuracy(
            parent,
            head,
            head_decoder,
            operation="complement",
            span=4,
            count=args.audit_count,
            seed=args.seed + 30_001,
        ),
        "consumer": {
            "stable_bits_to_threshold": consumer_stable,
            "target_accuracy": target_accuracy,
            "history": consumer_history,
            "progress": consumer_progress,
        },
        "fresh_pipeline": {
            "stable_bits_to_threshold": fresh_stable,
            "target_accuracy": fresh_accuracy,
            "history": fresh_history,
            "progress": fresh_progress,
        },
        "blank_pipeline": {
            "target_accuracy": blank_accuracy,
            "history": blank_history,
            "progress": blank_progress,
        },
        "reward_shuffled": {
            "target_accuracy": shuffled_accuracy,
            "progress": shuffled_progress,
        },
        "ablations": {
            "zero_head_accuracy": zero_head_accuracy,
            "zero_consumer_accuracy": zero_consumer_accuracy,
        },
        "reload": {
            "exact": reload_exact,
            "accuracy": reloaded_accuracy,
            "corruption_rejected": corruption_rejected,
        },
        "frozen_core": {
            "before": core_before,
            "after": core_after,
            "unchanged": core_before == core_after,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.head_updates * args.batch_size * 6
                + args.consumer_updates * bits_per_update * 4
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.head_updates * args.batch_size * 2
                + args.consumer_updates * args.batch_size * 4
            ),
            "optimizer_updates": (
                args.parent_updates + args.head_updates + args.consumer_updates * 4
            ),
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_mastered": bool(parent_progress)
            and parent_progress[-1]["heldout_accuracy"] >= args.mastery_threshold,
            "head_mastered": head_stable is not None,
            "consumer_mastered": consumer_stable is not None
            and target_accuracy >= args.mastery_threshold,
            "consumer_beats_blank": target_accuracy >= blank_accuracy + 0.05,
            "fresh_audited": fresh_accuracy >= 0.50,
            "reward_shuffled_no_gain": shuffled_accuracy <= blank_accuracy + 0.05,
            "head_is_causal": zero_head_accuracy < target_accuracy - 0.05,
            "consumer_is_causal": zero_consumer_accuracy < target_accuracy - 0.05,
            "positive_transfer": fresh_stable is not None
            and consumer_stable is not None
            and fresh_stable > consumer_stable,
            "reload_exact": reload_exact,
            "reload_behavior_preserved": reloaded_accuracy >= target_accuracy - 0.05,
            "corruption_rejected": corruption_rejected,
            "frozen_core": core_before == core_after,
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--head-updates", type=int, default=256)
    parser.add_argument("--consumer-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "consumer_target_accuracy": report["consumer"]["target_accuracy"],
                "fresh_target_accuracy": report["fresh_pipeline"]["target_accuracy"],
                "blank_target_accuracy": report["blank_pipeline"]["target_accuracy"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
