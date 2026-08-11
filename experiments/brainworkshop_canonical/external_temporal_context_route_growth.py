"""Learn context-conditioned routing into external temporal capability files.

This experiment composes two already versioned external mechanisms:

* an append-only opaque context route table, updated only from scalar episode
  outcomes; and
* an external temporal capability file whose own offset policy learns which
  relative history record is useful.

The controller, event encoder, and keypress boundary stay frozen after
construction.  A source file is mastered first, a second file is then
acquired without replay, and the route table must discover which file to use
from a learned event key.  No task name, rule ID, n-back depth, target bit, or
correct unattempted action crosses the learner boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import PersistentOpaqueContextRouteEvidence

from .cross_family_rule_growth import RULES
from .external_temporal_offset_growth import (
    EVENT_WIDTH,
    MASTERY_THRESHOLD,
    ExternalTemporalCapabilityFile,
    TemporalOffsetGrowthSystem,
    _build,
    _evaluate,
    _train_file,
)

TEMPORAL_CONTEXT_ROUTE_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-context-route-growth.v1"
)
OLD_FAMILY = "nback4"
NEW_FAMILY = "nback5"
OLD_CUE = 11
NEW_CUE = 12
UNKNOWN_CUE = 10
ROUTE_SELECTION_THRESHOLD = 0.99


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _context_key(system: TemporalOffsetGrowthSystem, cue_symbol: int) -> torch.Tensor:
    """Return the learned event tensor for a rendered context cue."""

    encoder = system.agent.runtime.encoders["stimulus"]
    return encoder(torch.tensor([cue_symbol], dtype=torch.long))[0].detach()


def _stable(rows: list[dict[str, object]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= MASTERY_THRESHOLD


def _route_episode_once(
    system: TemporalOffsetGrowthSystem,
    files: tuple[ExternalTemporalCapabilityFile, ...],
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    batch_size: int,
    steps: int,
    seed: int,
    exploration: bool,
    forced_slot: int | None = None,
) -> tuple[int, float, int]:
    """Single-call route probe with a scalar terminal outcome."""

    if len(files) != evidence.slot_count:
        raise ValueError("temporal route files and evidence slots must match")
    context = _context_key(system, cue_symbol)
    if forced_slot is None:
        selected_slot = int(evidence.preferred_order(context)[0])
        if exploration and len(files) > 1:
            generator = torch.Generator().manual_seed(seed + 91_337)
            if bool(torch.rand((), generator=generator) < 0.5):
                selected_slot = len(files) - 1
    else:
        if not 0 <= forced_slot < len(files):
            raise ValueError("forced temporal route slot is outside the file bank")
        selected_slot = forced_slot
    row = _evaluate(
        system,
        files[selected_slot],
        family=family,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        lifetimes=1,
        cue_symbol=cue_symbol,
    )[0]
    return selected_slot, float(row["accuracy"]), int(row["unique_verifier_bits"])


def _train_route(
    system: TemporalOffsetGrowthSystem,
    files: tuple[ExternalTemporalCapabilityFile, ...],
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    updates: int,
    batch_size: int,
    steps: int,
    seed: int,
    shuffled_outcomes: bool = False,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    delayed_outcome: float | None = None
    context = _context_key(system, cue_symbol)
    for update in range(1, updates + 1):
        selected_slot, outcome, bits = _route_episode_once(
            system,
            files,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            batch_size=batch_size,
            steps=steps,
            seed=seed + update,
            exploration=True,
        )
        observed = 0.5 if delayed_outcome is None else delayed_outcome
        if shuffled_outcomes:
            evidence.observe(context, selected_slot, observed)
        else:
            evidence.observe(context, selected_slot, outcome)
        delayed_outcome = outcome
        history.append(
            {
                "update": update,
                "selected_slot": selected_slot,
                "accuracy": outcome,
                "observed_outcome": observed if shuffled_outcomes else outcome,
                "unique_verifier_bits": bits,
                "replayed_examples": 0,
            }
        )
    return history


def _evaluate_route(
    system: TemporalOffsetGrowthSystem,
    files: tuple[ExternalTemporalCapabilityFile, ...],
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    family: str,
    cue_symbol: int,
    expected_slot: int,
    batch_size: int,
    steps: int,
    seed: int,
    lifetimes: int,
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    context = _context_key(system, cue_symbol)
    for lifetime in range(lifetimes):
        selected_slot, accuracy, bits = _route_episode_once(
            system,
            files,
            evidence,
            family=family,
            cue_symbol=cue_symbol,
            batch_size=batch_size,
            steps=steps,
            seed=seed + lifetime,
            exploration=False,
        )
        rows.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": accuracy,
                "selected_slot": selected_slot,
                "selected_expected": int(selected_slot == expected_slot),
                "unique_verifier_bits": bits,
                "replayed_examples": 0,
            }
        )
    return {
        "accuracy": sum(float(row["accuracy"]) for row in rows) / max(len(rows), 1),
        "selected_slot_fraction": sum(
            int(row["selected_slot"] == expected_slot) for row in rows
        )
        / max(len(rows), 1),
        "context_known": evidence.has_context(context),
        "lifetimes": rows,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.file_updates,
        args.route_updates,
        args.route_calibration_lifetimes,
        args.batch_size,
        args.steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("temporal context-route budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("temporal context-route optimization parameters are invalid")
    if args.steps <= RULES[NEW_FAMILY].warmup:
        raise ValueError("steps must include n-back-5 target trials")

    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    old_file = ExternalTemporalCapabilityFile()
    new_file = ExternalTemporalCapabilityFile()

    old_history = _train_file(
        system,
        old_file,
        family=OLD_FAMILY,
        updates=args.file_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        cue_symbol=OLD_CUE,
    )
    old_before = _evaluate(
        system,
        old_file,
        family=OLD_FAMILY,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 20_000,
        lifetimes=args.retention_lifetimes,
        cue_symbol=OLD_CUE,
    )
    old_file_digest_before_growth = old_file.digest()
    for parameter in old_file.parameters():
        parameter.requires_grad_(False)

    new_history = _train_file(
        system,
        new_file,
        family=NEW_FAMILY,
        updates=args.file_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 30_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        cue_symbol=NEW_CUE,
    )
    new_before = _evaluate(
        system,
        new_file,
        family=NEW_FAMILY,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 40_000,
        lifetimes=args.retention_lifetimes,
        cue_symbol=NEW_CUE,
    )
    for parameter in new_file.parameters():
        parameter.requires_grad_(False)

    files = (old_file, new_file)
    evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    evidence.append_slot()
    source_route_history: list[dict[str, float | int]] = []
    source_context = _context_key(system, OLD_CUE)
    for lifetime in range(args.route_calibration_lifetimes):
        selected_slot, accuracy, bits = _route_episode_once(
            system,
            files[:1],
            evidence,
            family=OLD_FAMILY,
            cue_symbol=OLD_CUE,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 50_000 + lifetime,
            exploration=False,
        )
        evidence.observe(source_context, selected_slot, accuracy)
        source_route_history.append(
            {
                "selected_slot": selected_slot,
                "accuracy": accuracy,
                "unique_verifier_bits": bits,
                "replayed_examples": 0,
            }
        )
    evidence.append_slot()

    target_route_history = _train_route(
        system,
        files,
        evidence,
        family=NEW_FAMILY,
        cue_symbol=NEW_CUE,
        updates=args.route_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 60_000,
    )
    routed_old = _evaluate_route(
        system,
        files,
        evidence,
        family=OLD_FAMILY,
        cue_symbol=OLD_CUE,
        expected_slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 70_000,
        lifetimes=args.retention_lifetimes,
    )
    routed_new = _evaluate_route(
        system,
        files,
        evidence,
        family=NEW_FAMILY,
        cue_symbol=NEW_CUE,
        expected_slot=1,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 80_000,
        lifetimes=args.retention_lifetimes,
    )
    unknown = _evaluate_route(
        system,
        files,
        evidence,
        family=NEW_FAMILY,
        cue_symbol=UNKNOWN_CUE,
        expected_slot=0,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 90_000,
        lifetimes=args.retention_lifetimes,
    )
    wrong_file = _evaluate(
        system,
        old_file,
        family=NEW_FAMILY,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 100_000,
        lifetimes=args.retention_lifetimes,
        cue_symbol=NEW_CUE,
    )
    wrong_offset = _evaluate(
        system,
        new_file,
        family=NEW_FAMILY,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 110_000,
        lifetimes=args.retention_lifetimes,
        cue_symbol=NEW_CUE,
        forced_offset=1,
    )
    missing_history = _evaluate(
        system,
        new_file,
        family=NEW_FAMILY,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 115_000,
        lifetimes=args.retention_lifetimes,
        cue_symbol=NEW_CUE,
        reset_memory_each_step=True,
    )

    shuffled_evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    shuffled_evidence.append_slot()
    shuffled_evidence.append_slot()
    shuffled_route_history = _train_route(
        system,
        files,
        shuffled_evidence,
        family=NEW_FAMILY,
        cue_symbol=NEW_CUE,
        updates=args.route_updates,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 120_000,
        shuffled_outcomes=True,
    )
    shuffled = _evaluate_route(
        system,
        files,
        shuffled_evidence,
        family=NEW_FAMILY,
        cue_symbol=NEW_CUE,
        expected_slot=1,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 130_000,
        lifetimes=args.retention_lifetimes,
    )

    restored = PersistentOpaqueContextRouteEvidence.from_payload(evidence.payload())
    restored_new = _evaluate_route(
        system,
        files,
        restored,
        family=NEW_FAMILY,
        cue_symbol=NEW_CUE,
        expected_slot=1,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 80_000,
        lifetimes=args.retention_lifetimes,
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "old_file_mastered_before_growth": _stable(old_before),
        "new_file_mastered_before_routing": _stable(new_before),
        "routed_old_mastered": float(routed_old["accuracy"]) >= MASTERY_THRESHOLD,
        "routed_new_mastered": float(routed_new["accuracy"]) >= MASTERY_THRESHOLD,
        "old_route_selects_old_file": float(routed_old["selected_slot_fraction"])
        >= ROUTE_SELECTION_THRESHOLD,
        "new_route_selects_new_file": float(routed_new["selected_slot_fraction"])
        >= ROUTE_SELECTION_THRESHOLD,
        "unknown_context_falls_back_to_oldest": float(
            unknown["selected_slot_fraction"]
        )
        >= ROUTE_SELECTION_THRESHOLD,
        "unknown_context_does_not_claim_new_mastery": float(unknown["accuracy"])
        < MASTERY_THRESHOLD,
        "wrong_file_rejects_mastery": not _stable(wrong_file),
        "wrong_offset_rejects_mastery": not _stable(wrong_offset),
        "missing_history_rejects_mastery": not _stable(missing_history),
        "shuffled_route_feedback_rejects_new_selection": float(
            shuffled["selected_slot_fraction"]
        )
        < ROUTE_SELECTION_THRESHOLD,
        "route_reload_exact": routed_new == restored_new,
        "old_file_unchanged_after_growth": old_file_digest_before_growth
        == old_file.digest(),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    primary_bits = args.batch_size * args.file_updates * (
        (args.steps - RULES[OLD_FAMILY].warmup)
        + (args.steps - RULES[NEW_FAMILY].warmup)
    )
    route_bits = args.batch_size * (
        args.route_calibration_lifetimes * (args.steps - RULES[OLD_FAMILY].warmup)
        + args.route_updates * (args.steps - RULES[NEW_FAMILY].warmup)
    )
    control_route_bits = args.batch_size * args.route_updates * (
        args.steps - RULES[NEW_FAMILY].warmup
    )
    report = {
        "schema": TEMPORAL_CONTEXT_ROUTE_SCHEMA,
        "claim_boundary": (
            "Outcome-only context-conditioned routing into isolated external "
            "temporal capability files, with each file learning its own opaque "
            "relative offset; not unrestricted addressing, compression, "
            "arbitrary new computation, or general continual learning."
        ),
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "event_key": "learned_rendered_event_tensor",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "temporal_memory": "external_temporal_history_memory_v1",
            "capability_file": "external_temporal_capability_file_v1",
            "source_family": OLD_FAMILY,
            "target_family": NEW_FAMILY,
            "source_cue": OLD_CUE,
            "target_cue": NEW_CUE,
            "unknown_cue": UNKNOWN_CUE,
            "route_feedback": "terminal_scalar_episode_accuracy_only",
        },
        "seed": args.seed,
        "old_history_tail": old_history[-5:],
        "new_history_tail": new_history[-5:],
        "source_route_history_tail": source_route_history[-5:],
        "target_route_history_tail": target_route_history[-5:],
        "shuffled_route_history_tail": shuffled_route_history[-5:],
        "evaluation": {
            "old_before": old_before,
            "new_before": new_before,
            "routed_old": routed_old,
            "routed_new": routed_new,
            "unknown_context": unknown,
            "wrong_file": wrong_file,
            "wrong_offset": wrong_offset,
            "missing_history": missing_history,
            "shuffled_route_feedback": shuffled,
            "route_payload": evidence.payload(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": primary_bits + route_bits,
            "control_verifier_bits": control_route_bits,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"])
                for result in (
                    old_before,
                    new_before,
                    wrong_file,
                    wrong_offset,
                    missing_history,
                )
                for row in result
            )
            + sum(
                int(row["unique_verifier_bits"])
                for result in (routed_old, routed_new, unknown, shuffled)
                for row in result["lifetimes"]
            ),
            "unique_logical_lifetimes": args.batch_size
            * (
                2 * args.file_updates
                + args.route_calibration_lifetimes
                + args.route_updates
            ),
            "optimizer_updates": args.file_updates * 2,
            "route_memory_updates": args.route_calibration_lifetimes
            + args.route_updates,
            "control_route_memory_updates": args.route_updates,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": primary_bits + route_bits
            if all(gates.values())
            else None,
        },
        "status": "promoted_temporal_context_route_growth"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--file-updates", type=int, default=512)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--route-calibration-lifetimes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
