"""Audit behavior-preserving consolidation of two canonical growth artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    FrameEventEncoder,
    _accuracy,
    _artifact,
    _copy_parent_weights,
    _freeze_except,
    _runtime,
    _train,
)
from experiments.working_memory_continuous.canonical_no_replay_artifact_bank import (
    _context_keys,
)
from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalOutputBus,
    ArtifactConsolidationReceipt,
    CapabilityRetentionLedger,
    ExecutableArtifactMemory,
    OpaqueProtocolDecoder,
    RetentionPolicyConfig,
    compose_growth_artifacts,
    decompress_growth_artifact,
    load_growth_artifact,
    select_growth_artifact_view,
)


def _direct_growth_runtime(*, seed: int, width: int = 64) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=32,
        workspace_slots=4,
        intention_width=16,
        feedback_width=2,
        event_window_capacity=16,
        reliability_hidden=16,
        growth_register_widths=(width, width),
    )
    return AmodalControllerRuntime(
        controller,
        encoders={"vision": FrameEventEncoder(32)},
        output_bus=AmodalOutputBus(
            {"action": OpaqueProtocolDecoder(16, 2, hidden=16)}
        ),
    )


def _digest_core(runtime) -> str:
    digest = hashlib.sha256()
    for name, value in runtime.controller.state_dict().items():
        if name.startswith("growth_slots."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train_parent_and_artifacts(
    *,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[Any, dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    parent = _runtime(seed=seed, growth=False)
    _train(
        parent,
        operation="forward",
        updates=updates,
        batch_size=batch_size,
        span=2,
        seed=seed + 100,
        lr=learning_rate,
    )
    parent.eval()
    artifacts: dict[str, dict[str, torch.Tensor]] = {}
    route_keys: dict[str, torch.Tensor] = {}
    for index, span in enumerate((3, 4), start=1):
        acquired = _direct_growth_runtime(seed=seed + index, width=64)
        _copy_parent_weights(parent, acquired)
        _freeze_except(acquired, ("growth_slots.0.",))
        _train(
            acquired,
            operation="forward",
            updates=updates,
            batch_size=batch_size,
            span=span,
            seed=seed + 200 * index,
            lr=learning_rate,
        )
        artifacts[str(span)] = _artifact(acquired, "growth_slots.0.")
        route_batch = generate_sequence_memory_batch(
            64,
            span=span,
            distractors=1,
            seed=seed + 50_000 + index,
            operation="forward",
            heldout=True,
        )
        route_keys[str(span)] = F.normalize(
            _context_keys(
                parent,
                route_batch,
                occupancy_scale=8.0,
            ).mean(dim=0),
            dim=0,
        )
    return parent, artifacts, route_keys


def _load_composed(
    parent,
    artifact: dict[str, torch.Tensor],
    *,
    seed: int,
    view: str,
    allow_dtype_cast: bool = False,
    decompress_artifact: bool = False,
) -> Any:
    runtime = _direct_growth_runtime(seed=seed, width=64)
    _copy_parent_weights(parent, runtime)
    if decompress_artifact:
        artifact = decompress_growth_artifact(artifact)
    selected = select_growth_artifact_view(
        artifact,
        source_prefix=f"growth_slots.{view}.",
    )
    receipt = load_growth_artifact(
        runtime.controller,
        selected,
        growth_prefixes=("growth_slots.0.",),
        allow_dtype_cast=allow_dtype_cast,
    )
    if not receipt.core_unchanged:
        raise RuntimeError("consolidated artifact changed the frozen core")
    runtime.eval()
    return runtime


def _load_single(parent, artifact: dict[str, torch.Tensor], *, seed: int) -> Any:
    runtime = _direct_growth_runtime(seed=seed, width=64)
    _copy_parent_weights(parent, runtime)
    load_growth_artifact(
        runtime.controller,
        artifact,
        growth_prefixes=("growth_slots.0.",),
    )
    runtime.eval()
    return runtime


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(args.updates, args.batch_size, args.audit_count, args.retention_probes) < 1:
        raise ValueError(
            "updates, batch size, audit count, and retention probes must be positive"
        )
    parent, artifacts, route_keys = _train_parent_and_artifacts(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    parent_digest = _digest_core(parent)
    single = {
        2: _accuracy(
            parent,
            operation="forward",
            count=args.audit_count,
            span=2,
            seed=args.seed + 60_002,
        )
    }
    single.update(
        {
            span: _accuracy(
                _load_single(parent, artifacts[str(span)], seed=args.seed + 100 + span),
                operation="forward",
                count=args.audit_count,
                span=span,
                seed=args.seed + 60_000 + span,
            )
            for span in (3, 4)
        }
    )
    source_path = args.report_out.parent / "source_bank"
    if source_path.exists():
        shutil.rmtree(source_path)
    source = ExecutableArtifactMemory(
        source_path,
        width=48,
        capacity=2,
        retention_ledger=CapabilityRetentionLedger(
            48,
            config=RetentionPolicyConfig(
                min_mastery_observations=args.retention_probes
            ),
        ),
    )
    source.put(route_keys["3"], artifacts["3"])
    source.put(route_keys["4"], artifacts["4"])

    composed = compose_growth_artifacts(
        (artifacts["3"], artifacts["4"]),
        prefix_maps=(
            {"growth_slots.0.": "growth_slots.0."},
            {"growth_slots.0.": "growth_slots.1."},
        ),
    )
    consolidated_path = args.report_out.parent / "consolidated_bank"
    if consolidated_path.exists():
        shutil.rmtree(consolidated_path)
    consolidated_behavior: dict[str, float] = {}
    core_digest: str | None = None
    candidate_promotion_latency_ms: float | None = None
    retention_probe_scores: list[float] = []

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        nonlocal candidate_promotion_latency_ms, consolidated_behavior, core_digest
        promotion_started = perf_counter()
        three_handle, from_three = candidate.promote(route_keys["3"])
        four_handle, from_four = candidate.promote(route_keys["4"])
        candidate_promotion_latency_ms = 1000.0 * (perf_counter() - promotion_started)
        if set(from_three) != set(composed) or set(from_four) != set(composed):
            return False
        if (three_handle.view, four_handle.view) != ("0", "1"):
            return False
        three_runtime = _load_composed(
            parent,
            from_three,
            seed=args.seed + 500,
            view=three_handle.view,
        )
        four_runtime = _load_composed(
            parent,
            from_four,
            seed=args.seed + 502,
            view=four_handle.view,
        )
        consolidated_behavior = {
            "2": single[2],
            "3": _accuracy(
                three_runtime,
                operation="forward",
                count=args.audit_count,
                span=3,
                seed=args.seed + 70_003,
            ),
            "4": _accuracy(
                four_runtime,
                operation="forward",
                count=args.audit_count,
                span=4,
                seed=args.seed + 70_004,
            ),
        }
        probe_scores = []
        for probe in range(args.retention_probes):
            three_probe = _accuracy(
                three_runtime,
                operation="forward",
                count=args.audit_count,
                span=3,
                seed=args.seed + 80_000 + probe * 2,
            )
            four_probe = _accuracy(
                four_runtime,
                operation="forward",
                count=args.audit_count,
                span=4,
                seed=args.seed + 80_001 + probe * 2,
            )
            probe_scores.append(min(three_probe, four_probe))
        if not retention_probe_scores:
            retention_probe_scores.extend(probe_scores)
        core_digest = _digest_core(three_runtime)
        return (
            consolidated_behavior["2"] >= single[2] - args.retention_tolerance
            and consolidated_behavior["3"] >= single[3] - args.behavior_tolerance
            and consolidated_behavior["4"] >= single[4] - args.behavior_tolerance
        )

    preflight_path = args.report_out.parent / "preflight_bank"
    if preflight_path.exists():
        shutil.rmtree(preflight_path)
    preflight_candidate, preflight_receipt = source.consolidate_verified(
        (0, 1),
        F.normalize(route_keys["3"] + route_keys["4"], dim=0),
        composed,
        preflight_path,
        replacement_aliases=(route_keys["3"], route_keys["4"]),
        replacement_alias_views=("0", "1"),
        verifier=verifier,
    )
    if not preflight_receipt.accepted or preflight_candidate is None:
        raise RuntimeError(f"retention preflight was rejected: {preflight_receipt}")
    if len(retention_probe_scores) != args.retention_probes:
        raise RuntimeError("retention preflight produced the wrong probe count")
    shutil.rmtree(preflight_path)
    for key in route_keys.values():
        for _ in range(args.retention_probes):
            source.observe_retention(key, 1.0)
    replacement_key = F.normalize(route_keys["3"] + route_keys["4"], dim=0)

    candidate, receipt = source.consolidate_verified(
        (0, 1),
        replacement_key,
        composed,
        consolidated_path,
        replacement_aliases=(route_keys["3"], route_keys["4"]),
        replacement_alias_views=("0", "1"),
        verifier=verifier,
        candidate_outcomes=retention_probe_scores,
        retained_scores=[],
        candidate_threshold=0.8,
        retention_floor=0.8,
        min_candidate_observations=args.retention_probes,
    )
    if not isinstance(receipt, ArtifactConsolidationReceipt):
        raise TypeError("consolidation did not return a receipt")
    if not receipt.accepted or candidate is None:
        raise RuntimeError(f"behavioral consolidation was rejected: {receipt}")
    reloaded = ExecutableArtifactMemory.load(consolidated_path)
    reloaded_behaviors: dict[str, float] = {}
    reload_promotion_started = perf_counter()
    reloaded_three_handle, reloaded_three_artifact = reloaded.promote(route_keys["3"])
    reloaded_four_handle, reloaded_four_artifact = reloaded.promote(route_keys["4"])
    reload_promotion_latency_ms = 1000.0 * (
        perf_counter() - reload_promotion_started
    )
    if (reloaded_three_handle.view, reloaded_four_handle.view) != ("0", "1"):
        raise RuntimeError(
            "reloaded aliases selected the wrong executable views: "
            f"{reloaded_three_handle.view!r}, {reloaded_four_handle.view!r}"
        )
    reloaded_three_runtime = _load_composed(
        parent,
        reloaded_three_artifact,
        seed=args.seed + 501,
        view=reloaded_three_handle.view,
    )
    reloaded_four_runtime = _load_composed(
        parent,
        reloaded_four_artifact,
        seed=args.seed + 503,
        view=reloaded_four_handle.view,
    )
    reloaded_artifact_exact = all(
        name in reloaded_four_artifact
        and torch.equal(reloaded_four_artifact[name], value)
        for name, value in composed.items()
    ) and len(reloaded_four_artifact) == len(composed)
    reloaded_behaviors = {
        "2": single[2],
        "3": _accuracy(
            reloaded_three_runtime,
            operation="forward",
            count=args.audit_count,
            span=3,
            seed=args.seed + 70_003,
        ),
        "4": _accuracy(
            reloaded_four_runtime,
            operation="forward",
            count=args.audit_count,
            span=4,
            seed=args.seed + 70_004,
        ),
    }

    wrong_path = args.report_out.parent / "rejected_bank"
    if wrong_path.exists():
        shutil.rmtree(wrong_path)
    wrong_candidate, wrong_receipt = source.consolidate_verified(
        (0, 1),
        F.normalize(route_keys["3"] + route_keys["4"], dim=0),
        artifacts["3"],
        wrong_path,
        replacement_aliases=(route_keys["3"], route_keys["4"]),
        replacement_alias_views=("0", "1"),
        verifier=lambda _: False,
        candidate_outcomes=retention_probe_scores,
        retained_scores=[],
        candidate_threshold=0.8,
        retention_floor=0.8,
        min_candidate_observations=args.retention_probes,
    )
    artifact_path = consolidated_path / candidate.paths[0]
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(consolidated_path)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)
    restored = ExecutableArtifactMemory.load(consolidated_path)

    report = {
        "schema": "neural-computer.artifact-consolidation-report.v1",
        "claim_boundary": (
            "Two independently learned canonical growth artifacts can be "
            "composed into one behavior-verified artifact row with multiple "
            "opaque aliases; this is logical compaction, not byte compression "
            "or general continual learning."
        ),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "retention_probe_count": args.retention_probes,
        "single_behavior": single,
        "consolidated_behavior": consolidated_behavior,
        "reloaded_behavior": reloaded_behaviors,
        "source_rows": len(source.occupied),
        "consolidated_rows": len(restored.occupied),
        "aliases_per_row": [len(aliases) for aliases in restored.alias_keys],
        "alias_views": restored.alias_views,
        "reloaded_alias_views": [
            reloaded_three_handle.view,
            reloaded_four_handle.view,
        ],
        "reloaded_artifact_exact": reloaded_artifact_exact,
        "retention": {
            "source_capabilities_protected": all(
                source.retention.is_protected(key) for key in route_keys.values()
            ),
            "probe_count": len(retention_probe_scores),
            "probe_scores": retention_probe_scores,
            "replacement_protected": candidate.retention.is_protected(
                replacement_key
            ),
            "replacement_protected_after_reload": reloaded.retention.is_protected(
                replacement_key
            ),
        },
        "receipt": {
            "accepted": receipt.accepted,
            "source_indices": receipt.source_indices,
            "rows_before": receipt.rows_before,
            "rows_after": receipt.rows_after,
            "rows_saved": receipt.rows_saved,
        },
        "rejected_control": {
            "candidate_created": wrong_candidate is not None,
            "accepted": wrong_receipt.accepted,
        },
        "parent_core_digest": parent_digest,
        "consolidated_core_digest": core_digest,
        "core_unchanged": parent_digest == core_digest,
        "corruption_rejected": corruption_rejected,
        "accounting": {
            "unique_logical_lifetimes": args.updates * args.batch_size * 3,
            "unique_verifier_bits": args.updates * args.batch_size * (2 + 3 + 4),
            "optimizer_updates": args.updates * 3,
            "replayed_examples": 0,
            "consolidation_optimizer_updates": 0,
            "retention_observations": args.retention_probes * 3,
            "retention_probe_verifier_bits": (
                args.retention_probes * args.audit_count * 7 * 2
            ),
            "wall_seconds": perf_counter() - started,
        },
        "latency_ms": {
            "candidate_alias_pair_promotion": candidate_promotion_latency_ms,
            "reloaded_alias_pair_promotion": reload_promotion_latency_ms,
        },
        "gates": {
            "behavior_preserved": receipt.accepted,
            "aliases_route_to_one_row": all(
                handle.index == 0
                for handle in (restored.promote(route_keys["3"])[0], restored.promote(route_keys["4"])[0])
            ),
            "rows_saved": receipt.rows_saved == 1,
            "rejected_candidate_not_adopted": wrong_candidate is None
            and not wrong_receipt.accepted,
            "reloaded_behavior_preserved": all(
                reloaded_behaviors[str(span)]
                >= single[span] - (args.retention_tolerance if span == 2 else args.behavior_tolerance)
                for span in (2, 3, 4)
            ),
            "reloaded_artifact_exact": reloaded_artifact_exact,
            "retention_safe_consolidation": (
                all(
                    source.retention.is_protected(key)
                    for key in route_keys.values()
                )
                and candidate.retention.is_protected(replacement_key)
                and reloaded.retention.is_protected(replacement_key)
            ),
            "frozen_core": parent_digest == core_digest,
            "corruption_rejected": corruption_rejected,
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--retention-tolerance", type=float, default=0.05)
    parser.add_argument("--behavior-tolerance", type=float, default=0.05)
    parser.add_argument("--retention-probes", type=int, default=8)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "single_behavior": report["single_behavior"],
                "consolidated_behavior": report["consolidated_behavior"],
                "receipt": report["receipt"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
