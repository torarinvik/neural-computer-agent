"""Measure verifier-gated transfer from one external capability to another.

The frozen controller and source artifact remain unchanged while an inherited
external stack and a fresh stack receive the same new runtime program from
fresh outcomes.  A stable-prefix selector chooses a unique winner; only that
candidate may be admitted beside the protected source artifact.  This keeps
the transfer claim separate from routing and makes a failed transfer safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    SPAN,
    THRESHOLD,
    _generated_key,
    _load_stack_artifact,
    _parse_program_specs,
    _stack_artifact,
)
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    _capability_accuracy,
    _new_capability,
    _stable_bits,
    _train_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from neural_computer import (
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    RetentionPolicyConfig,
    select_capability_candidate,
)
from .train_pipeline import _new_stack


def _digest_artifact(artifact: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(artifact.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _candidate_scores(
    progress: list[dict[str, float | int]],
) -> tuple[float, ...]:
    return tuple(float(row["heldout_accuracy"]) for row in progress)


def _source_retention_probe(
    parent,
    source_artifact: dict[str, torch.Tensor],
    source_id: int,
    grammar,
    *,
    count: int,
    probes: int,
    seed: int,
) -> list[float]:
    stack, decoder = _load_stack_artifact(source_artifact)
    return [
        _capability_accuracy(
            parent,
            stack,
            decoder,
            operation="generated_composition",
            span=SPAN,
            count=count,
            seed=seed + probe * 101,
            generated_composition_ids=(source_id,),
            generated_compositions=grammar,
        )
        for probe in range(probes)
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.source_updates,
        args.candidate_updates,
        args.batch_size,
        args.audit_count,
        args.retention_probes,
        args.eval_every,
    ) < 1:
        raise ValueError("all update and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")

    grammar = _parse_program_specs(args.program_spec)
    if len(grammar) < 2:
        raise ValueError("transfer requires at least source and target programs")
    if args.source_id == args.target_id:
        raise ValueError("source and target programs must differ")
    if not 0 <= args.source_id < len(grammar) or not 0 <= args.target_id < len(grammar):
        raise ValueError("source or target program is out of range")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    _parent_history, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        credit_mode="sampled",
    )
    parent.eval()
    parent_digest_before = _digest_core(parent, ())

    source_stack = _new_stack(args.seed + 1_000, program_count=2, stack="routed")
    source_decoder = _new_capability(args.seed + 2_000)[1]
    source_history, source_progress = _train_capability(
        parent,
        source_stack,
        source_decoder,
        operation="generated_composition",
        span=SPAN,
        updates=args.source_updates,
        batch_size=args.batch_size,
        seed=args.seed + 20_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        generated_composition_ids=(args.source_id,),
        generated_compositions=grammar,
    )
    source_stack.eval()
    source_decoder.eval()
    source_artifact = _stack_artifact(source_stack, source_decoder)
    source_digest = _digest_artifact(source_artifact)
    source_stable = _stable_bits(
        source_progress,
        threshold=THRESHOLD,
        bits_per_update=args.batch_size * SPAN,
    )
    source_behavior = _capability_accuracy(
        parent,
        source_stack,
        source_decoder,
        operation="generated_composition",
        span=SPAN,
        count=args.audit_count,
        seed=args.seed + 40_000,
        generated_composition_ids=(args.source_id,),
        generated_compositions=grammar,
    )

    inherited_stack, inherited_decoder = _load_stack_artifact(source_artifact)
    fresh_stack = _new_stack(args.seed + 3_000, program_count=2, stack="routed")
    fresh_decoder = _new_capability(args.seed + 4_000)[1]
    inherited_seed = args.seed + 50_001
    fresh_seed = args.seed + 50_001
    torch.manual_seed(inherited_seed)
    inherited_history, inherited_progress = _train_capability(
        parent,
        inherited_stack,
        inherited_decoder,
        operation="generated_composition",
        span=SPAN,
        updates=args.candidate_updates,
        batch_size=args.batch_size,
        seed=inherited_seed,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        generated_composition_ids=(args.target_id,),
        generated_compositions=grammar,
    )
    torch.manual_seed(fresh_seed)
    fresh_history, fresh_progress = _train_capability(
        parent,
        fresh_stack,
        fresh_decoder,
        operation="generated_composition",
        span=SPAN,
        updates=args.candidate_updates,
        batch_size=args.batch_size,
        seed=fresh_seed,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        generated_composition_ids=(args.target_id,),
        generated_compositions=grammar,
    )
    inherited_stack.eval()
    inherited_decoder.eval()
    fresh_stack.eval()
    fresh_decoder.eval()

    bits_per_update = args.batch_size * SPAN
    inherited_stable = _stable_bits(
        inherited_progress,
        threshold=THRESHOLD,
        bits_per_update=bits_per_update,
    )
    fresh_stable = _stable_bits(
        fresh_progress,
        threshold=THRESHOLD,
        bits_per_update=bits_per_update,
    )
    selection = select_capability_candidate(
        (_candidate_scores(inherited_progress), _candidate_scores(fresh_progress)),
        threshold=THRESHOLD,
        bits_per_observation=args.eval_every * bits_per_update,
    )
    inherited_behavior = _capability_accuracy(
        parent,
        inherited_stack,
        inherited_decoder,
        operation="generated_composition",
        span=SPAN,
        count=args.audit_count,
        seed=args.seed + 60_000,
        generated_composition_ids=(args.target_id,),
        generated_compositions=grammar,
    )
    fresh_behavior = _capability_accuracy(
        parent,
        fresh_stack,
        fresh_decoder,
        operation="generated_composition",
        span=SPAN,
        count=args.audit_count,
        seed=args.seed + 60_000,
        generated_composition_ids=(args.target_id,),
        generated_compositions=grammar,
    )

    source_path = args.report_out.parent / "source_bank"
    if source_path.exists():
        shutil.rmtree(source_path)
    bank = ExecutableArtifactMemory(
        source_path,
        width=48,
        capacity=1,
        write_match_threshold=0.99999,
    )
    bank.retention.config = RetentionPolicyConfig(
        mastery_threshold=THRESHOLD,
        min_mastery_observations=args.retention_probes,
    )
    lifecycle = ExternalCapabilityLifecycle(bank)
    source_key = _generated_key(
        parent,
        args.source_id,
        count=args.audit_count,
        seed=args.seed + 70_000,
        generated_compositions=grammar,
    )
    source_receipt = lifecycle.admit(source_key, source_artifact)
    if not source_receipt.accepted or source_receipt.index is None:
        raise RuntimeError(f"source admission failed: {source_receipt.reason}")
    source_outcomes = _source_retention_probe(
        parent,
        source_artifact,
        args.source_id,
        grammar,
        count=args.audit_count,
        probes=args.retention_probes,
        seed=args.seed + 80_000,
    )
    for outcome in source_outcomes:
        bank.observe_retention(source_key, outcome)
    bank.save()
    source_protected = bool(lifecycle.protection_mask()[source_receipt.index])
    reloaded_source = ExecutableArtifactMemory.load(source_path)
    _, reloaded_source_artifact = reloaded_source.promote_index(source_receipt.index)
    source_reload_digest = _digest_artifact(reloaded_source_artifact)
    reloaded_source_stack, reloaded_source_decoder = _load_stack_artifact(
        reloaded_source_artifact
    )
    source_reload_behavior = _capability_accuracy(
        parent,
        reloaded_source_stack,
        reloaded_source_decoder,
        operation="generated_composition",
        span=SPAN,
        count=args.audit_count,
        seed=args.seed + 85_000,
        generated_composition_ids=(args.source_id,),
        generated_compositions=grammar,
    )
    source_artifact_path = source_path / (bank.paths[source_receipt.index] or "")
    intact_source_bytes = source_artifact_path.read_bytes()
    source_artifact_path.write_bytes(intact_source_bytes + b"corruption")
    source_corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(source_path)
    except ValueError as error:
        source_corruption_rejected = "hash mismatch" in str(error)
    source_artifact_path.write_bytes(intact_source_bytes)

    candidate_artifacts = (
        _stack_artifact(inherited_stack, inherited_decoder),
        _stack_artifact(fresh_stack, fresh_decoder),
    )
    selected_artifact = (
        None
        if selection.selected_index is None
        else candidate_artifacts[selection.selected_index]
    )
    selected_behavior = (
        None
        if selection.selected_index is None
        else (inherited_behavior, fresh_behavior)[selection.selected_index]
    )
    target_key = _generated_key(
        parent,
        args.target_id,
        count=args.audit_count,
        seed=args.seed + 90_000,
        generated_compositions=grammar,
    )
    admission = None
    if (
        selection.accepted
        and selected_artifact is not None
        and selected_behavior is not None
        and selected_behavior >= THRESHOLD
        and source_protected
    ):
        plan = lifecycle.plan_admission(target_key, selected_artifact)
        admission = lifecycle.admit(
            target_key,
            selected_artifact,
            plan=plan,
            grow_destination=args.report_out.parent / "grown_bank",
        )

    parent_digest_after = _digest_core(parent, ())
    source_after = _source_retention_probe(
        parent,
        source_artifact,
        args.source_id,
        grammar,
        count=args.audit_count,
        probes=args.retention_probes,
        seed=args.seed + 100_000,
    )
    inherited_faster = (
        inherited_stable is not None
        and fresh_stable is not None
        and inherited_stable < fresh_stable
    )
    report = {
        "schema": "neural-computer.generated-composition-transfer-report.v1",
        "claim_boundary": (
            "An inherited external artifact and a fresh external candidate "
            "learned the same runtime-supplied target from fresh outcomes. "
            "A stable-prefix selector gated candidate admission beside a "
            "protected source artifact; this is not general continual learning."
        ),
        "seed": args.seed,
        "source_id": args.source_id,
        "target_id": args.target_id,
        "programs": [list(program) for program in grammar],
        "source": {
            "stable_bits_to_threshold": source_stable,
            "behavior": source_behavior,
            "retention_before": source_outcomes,
            "retention_after": source_after,
            "artifact_digest": source_digest,
            "reload_digest": source_reload_digest,
            "reload_behavior": source_reload_behavior,
            "corruption_rejected": source_corruption_rejected,
            "history": source_history,
            "progress": source_progress,
        },
        "inherited_candidate": {
            "stable_bits_to_threshold": inherited_stable,
            "behavior": inherited_behavior,
            "history": inherited_history,
            "progress": inherited_progress,
        },
        "fresh_candidate": {
            "stable_bits_to_threshold": fresh_stable,
            "behavior": fresh_behavior,
            "history": fresh_history,
            "progress": fresh_progress,
        },
        "candidate_selection": {
            "accepted": selection.accepted,
            "selected_index": selection.selected_index,
            "stable_bits_to_threshold": selection.stable_bits_to_threshold,
            "reason": selection.reason,
            "inherited_faster": inherited_faster,
        },
        "admission": None
        if admission is None
        else {
            "accepted": admission.accepted,
            "action": admission.action,
            "source_capacity": admission.source_capacity,
            "destination_capacity": admission.destination_capacity,
            "reason": admission.reason,
        },
        "frozen_core": {
            "digest_before": parent_digest_before,
            "digest_after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.source_updates * args.batch_size * (SPAN + 2)
                + 2 * args.candidate_updates * args.batch_size * (SPAN + 2)
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.source_updates * args.batch_size * 2
                + 2 * args.candidate_updates * args.batch_size * 2
            ),
            "optimizer_updates": (
                args.parent_updates + args.source_updates + 2 * args.candidate_updates
            ),
            "replayed_examples": 0,
        },
        "gates": {
            "parent_stable": _stable_bits(
                parent_progress,
                threshold=THRESHOLD,
                bits_per_update=args.batch_size * 2,
            )
            is not None,
            "source_stable": source_stable is not None,
            "source_mastered": source_behavior >= THRESHOLD,
            "source_protected": source_protected,
            "source_retained": min(source_after) >= THRESHOLD,
            "source_reload_preserved": source_reload_digest == source_digest
            and source_reload_behavior >= THRESHOLD,
            "source_corruption_rejected": source_corruption_rejected,
            "candidate_stable": selection.accepted,
            "candidate_mastered": selected_behavior is not None
            and selected_behavior >= THRESHOLD,
            "positive_transfer": inherited_faster,
            "admitted_only_after_selection": admission is not None
            and admission.accepted,
            "core_unchanged": parent_digest_before == parent_digest_after,
            "no_replayed_examples": True,
        },
        "wall_seconds": perf_counter() - started,
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--source-updates", type=int, default=256)
    parser.add_argument("--candidate-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--source-id", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument(
        "--program-spec",
        action="append",
        default=None,
        help="verifier-private runtime program as comma-separated primitives",
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "candidate_selection": report["candidate_selection"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
