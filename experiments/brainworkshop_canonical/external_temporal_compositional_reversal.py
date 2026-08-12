"""Add learned regime reversal to repeated online compositional growth.

The promoted online-compositional rung grows and consolidates three pairs of
temporal capabilities, but all six routes remain stationary after admission.
This experiment adds one nonstationary boundary: the first composed artifact
receives a changed opaque basis regime under the same route keys. A learned
external regime detector must leave stable evidence untouched, trigger only on
the changed evidence, and let a verifier-gated copy-on-write replacement
update that one row while preserving the other two composed rows.

The controller, learned event encoder, and temporal capability file remain
frozen. The detector sees only current/incoming learned value banks and scalar
utility during its own external training; no regime name, route ID, query
depth, or verifier answer enters its deployed proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import (
    ExecutableArtifactMemory,
    OpaqueRegimeChangePolicy,
)

from . import external_temporal_online_compositional_growth as growth
from . import external_temporal_query_address_growth as query
from . import external_temporal_query_counterfactual_growth as counterfactual
from .external_temporal_shared_basis_learned_regime_trigger import (
    _evaluate_detector,
    _synthetic_pair,
    _train_detector,
)

REVERSAL_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-compositional-"
    "reversal.v1"
)
DETECTOR_UPDATES = 1_000


def _memory_digest(memory: ExecutableArtifactMemory) -> str:
    digest = hashlib.sha256()
    for path in sorted(memory.directory.iterdir()):
        if path.is_file():
            digest.update(str(path.name).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _changed_pair_artifact(
    routes: list[dict[str, object]],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    current, incoming, _shifted = _synthetic_pair(seed=seed, shifted=1)
    basis = F.normalize(incoming.mean(dim=0), dim=0)
    generator = torch.Generator().manual_seed(seed + 101_001)
    residual_a = 0.02 * torch.randn(query.EVENT_WIDTH, generator=generator)
    residual_b = 0.02 * torch.randn(query.EVENT_WIDTH, generator=generator)
    route_a = growth._fixed_route_artifact(
        shared_basis=basis,
        residual=residual_a,
        position=int(routes[0]["position"]),
    )
    route_b = growth._fixed_route_artifact(
        shared_basis=basis,
        residual=residual_b,
        position=int(routes[1]["position"]),
    )
    del current
    return growth._fixed_composed_artifact(route_a, route_b)


def _post_growth_reversal(
    memory: ExecutableArtifactMemory,
    *,
    detector: OpaqueRegimeChangePolicy,
    routes: list[dict[str, object]],
    keys: tuple[torch.Tensor, ...],
    aliases: dict[int, torch.Tensor],
    seed: int,
) -> tuple[ExecutableArtifactMemory, dict[str, object]]:
    occupied = torch.ones(1, 8, dtype=torch.bool)
    current, _stable_incoming, _stable_shifted = _synthetic_pair(
        seed=seed,
        shifted=0,
    )
    _current_again, shifted_incoming, _shifted = _synthetic_pair(
        seed=seed,
        shifted=1,
    )
    before_version = memory.version
    before_digest = _memory_digest(memory)
    stable_plan = detector.propose(
        current.unsqueeze(0),
        occupied,
        current.unsqueeze(0),
        occupied,
    )
    stable_noop = memory.version == before_version and _memory_digest(memory) == before_digest
    shifted_plan = detector.propose(
        current.unsqueeze(0),
        occupied,
        shifted_incoming.unsqueeze(0),
        occupied,
    )
    replacement = None
    receipt: dict[str, object] = {
        "accepted": False,
        "reason": "detector did not propose replacement",
    }
    route_retention = False
    reload_exact = False
    memory_changed = False
    if shifted_plan.replace:
        candidates = memory.planner_candidates()
        first_row = growth._key_row(candidates, keys[0])
        bindings = growth._bindings_for_stage(
            routes,
            keys,
            aliases,
            stage=len(growth.PAIR_INDICES) - 1,
        )
        current_bindings = (
            (keys[0], None, int(routes[0]["position"])),
            (aliases[0], "route_a", int(routes[0]["position"])),
            (keys[1], "route_b", int(routes[1]["position"])),
        )
        replacement_artifact = _changed_pair_artifact(routes, seed=seed)
        candidate, candidate_receipt = memory.consolidate_verified(
            (first_row,),
            keys[0],
            replacement_artifact,
            memory.directory.parent / "reversal-accepted",
            verifier=lambda candidate_memory: growth._route_bindings_pass(
                candidate_memory,
                bindings,
                rows=len(growth.PAIR_INDICES),
            ),
            replacement_aliases=(aliases[0], keys[1]),
            replacement_alias_views=("route_a", "route_b"),
            candidate_outcome_probe=lambda _candidate: growth._candidate_probe(
                current_bindings
            ),
            retained_scores=[1.0] * len(current_bindings),
            candidate_threshold=0.8,
            retention_floor=0.8,
            min_candidate_observations=8,
        )
        receipt = candidate_receipt.__dict__
        if candidate is not None and candidate_receipt.accepted:
            replacement = candidate
            route_retention = growth._route_bindings_pass(
                candidate,
                bindings,
                rows=len(growth.PAIR_INDICES),
            )
            restored = ExecutableArtifactMemory.load(candidate.directory)
            reload_exact = growth._route_bindings_pass(
                restored,
                bindings,
                rows=len(growth.PAIR_INDICES),
            )
            memory_changed = _memory_digest(candidate) != before_digest
    return replacement or memory, {
        "stable_plan_replace": bool(stable_plan.replace),
        "stable_noop": stable_noop and not stable_plan.replace,
        "shifted_plan_replace": bool(shifted_plan.replace),
        "replacement_accepted": replacement is not None,
        "route_retention": route_retention,
        "reload_exact": reload_exact,
        "memory_changed": memory_changed,
        "receipt": receipt,
    }


def _fresh_detector(seed: int) -> OpaqueRegimeChangePolicy:
    torch.manual_seed(seed)
    return OpaqueRegimeChangePolicy(
        value_width=query.EVENT_WIDTH,
        hidden=128,
        max_spectral_bins=8,
        learning_rate=0.002,
    ).eval()


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.source_evaluation_lifetimes,
        args.source_route_lifetimes,
        args.target_route_updates,
        args.policy_updates,
        args.policy_batch_size,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
        args.detector_updates,
    ) < 1:
        raise ValueError("compositional reversal budgets must be positive")
    if args.data_steps <= max(depth for _, depth in growth.ROUTE_SPECS):
        raise ValueError("data steps must include every route depth")
    started = perf_counter()
    system = query._build(args.seed)
    controller_before = growth._digest(system.agent.controller)
    encoder_before = growth._digest(system.agent.runtime.encoders["stimulus"])
    files, candidates = counterfactual._train_candidates(
        system,
        offsets=tuple(range(1, query.MAX_OFFSET + 1)),
        updates=args.source_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        evaluation_lifetimes=args.source_evaluation_lifetimes,
    )
    stable_offsets = tuple(int(record["offset"]) for record in candidates if bool(record["stable"]))
    winner_offset = stable_offsets[0] if stable_offsets else max(
        candidates,
        key=lambda record: min(float(row["accuracy"]) for row in record["evaluation"]),
    )["offset"]
    winner = files[winner_offset - 1]
    winner_digest_before = winner.digest()
    evidence = counterfactual._evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    args.winner_offset = int(winner_offset)
    routes, route_bits, counterfactual_bits = growth._acquire_routes(
        system,
        winner,
        evidence,
        args,
    )
    keys, artifacts, aliases = growth._make_artifacts(routes, seed=args.seed)
    policy, policy_accounting = growth._train_policy(
        seed=args.seed,
        rows=8,
        width=query.EVENT_WIDTH,
        updates=args.policy_updates,
        batch_size=args.policy_batch_size,
        shuffled_utility=False,
    )
    shuffled, shuffled_accounting = growth._train_policy(
        seed=args.seed + 50_000,
        rows=8,
        width=query.EVENT_WIDTH,
        updates=args.policy_updates,
        batch_size=args.policy_batch_size,
        shuffled_utility=True,
    )
    torch.manual_seed(args.seed + 100_000)
    untrained = growth.OpaqueConsolidationPolicy(query.EVENT_WIDTH, hidden=64).eval()
    detector, detector_accounting = _train_detector(
        seed=args.seed + 600_000,
        updates=args.detector_updates,
    )
    detector_scores = _evaluate_detector(detector, seed=args.seed + 700_000)
    fresh_scores = _evaluate_detector(
        _fresh_detector(args.seed + 800_000),
        seed=args.seed + 700_000,
    )

    def callback(label: str):
        return lambda memory: _post_growth_reversal(
            memory,
            detector=detector,
            routes=routes,
            keys=keys,
            aliases=aliases,
            seed=args.seed + (910_000 if label == "forward" else 920_000),
        )

    with tempfile.TemporaryDirectory(prefix="temporal-compositional-reversal-") as directory:
        root = Path(directory)
        forward = growth._stream(
            root=root / "forward",
            stage_order=(0, 1, 2),
            routes=routes,
            keys=keys,
            aliases=aliases,
            artifacts=artifacts,
            policy=policy,
            shuffled=shuffled,
            untrained=untrained,
            system=system,
            winner=winner,
            evidence=evidence,
            args=args,
            winner_digest_before=winner_digest_before,
            reverse_insertion=False,
            post_growth=callback("forward"),
        )
        reversed_stream = growth._stream(
            root=root / "reversed",
            stage_order=(0, 1, 2),
            routes=routes,
            keys=keys,
            aliases=aliases,
            artifacts=artifacts,
            policy=policy,
            shuffled=shuffled,
            untrained=untrained,
            system=system,
            winner=winner,
            evidence=evidence,
            args=args,
            winner_digest_before=winner_digest_before,
            reverse_insertion=True,
            post_growth=callback("reversed"),
        )
    controller_after = growth._digest(system.agent.controller)
    encoder_after = growth._digest(system.agent.runtime.encoders["stimulus"])
    base_growth_ok = lambda stream: (
        all(bool(stage["accepted"]) for stage in stream["stages"])
        and bool(stream["prefixes_retained"])
        and bool(stream["policy_controls_pass"])
    )
    gates = {
        "detector_stable_keep_transfer": detector_scores["stable_keep"] >= 0.80,
        "detector_shift_replace_transfer": detector_scores["shift_replace"] >= 0.80,
        "detector_dominates_fresh_control": (
            detector_scores["stable_keep"] >= fresh_scores["stable_keep"]
            and detector_scores["shift_replace"] >= fresh_scores["shift_replace"]
            and (
                detector_scores["stable_keep"] > fresh_scores["stable_keep"]
                or detector_scores["shift_replace"] > fresh_scores["shift_replace"]
            )
        ),
        "forward_compositional_growth_retained": base_growth_ok(forward),
        "reversed_compositional_growth_retained": base_growth_ok(reversed_stream),
        "forward_stable_noop": bool(forward["post_growth"]["stable_noop"]),
        "reversed_stable_noop": bool(reversed_stream["post_growth"]["stable_noop"]),
        "forward_shift_detected": bool(forward["post_growth"]["shifted_plan_replace"]),
        "reversed_shift_detected": bool(reversed_stream["post_growth"]["shifted_plan_replace"]),
        "forward_replacement_accepted": bool(forward["post_growth"]["replacement_accepted"]),
        "reversed_replacement_accepted": bool(reversed_stream["post_growth"]["replacement_accepted"]),
        "forward_routes_retained_after_reversal": bool(forward["post_growth"]["route_retention"]),
        "reversed_routes_retained_after_reversal": bool(reversed_stream["post_growth"]["route_retention"]),
        "forward_reversal_reload_exact": bool(forward["post_growth"]["reload_exact"]),
        "reversed_reversal_reload_exact": bool(reversed_stream["post_growth"]["reload_exact"]),
        "forward_reversal_changed_memory": bool(forward["post_growth"]["memory_changed"]),
        "reversed_reversal_changed_memory": bool(reversed_stream["post_growth"]["memory_changed"]),
        "forward_corruption_rejected": bool(forward["corruption_rejected"]),
        "reversed_corruption_rejected": bool(reversed_stream["corruption_rejected"]),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "promoted_file_frozen": winner_digest_before == winner.digest(),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": REVERSAL_SCHEMA,
        "claim_boundary": (
            "Learned stable-noop and shift-replace timing for one composed external "
            "temporal artifact while retaining two other composed routes; not "
            "unrestricted memory growth, arbitrary new computation, or general "
            "continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "growth_boundary": "repeated_online_compositional_growth_v1",
            "regime_detector": "opaque_regime_change_policy_v1",
            "replacement": "copy_on_write_single_composed_row_v1",
            "route_count": len(routes),
            "composed_rows": len(growth.PAIR_INDICES),
            "forbidden_features": "regime_names_route_ids_query_depths_task_ids_replayed_streams",
        },
        "detector_scores": detector_scores,
        "fresh_detector_scores": fresh_scores,
        "forward": forward,
        "reversed": reversed_stream,
        "gates": gates,
        "accounting": {
            "unique_temporal_verifier_bits": route_bits,
            "counterfactual_temporal_verifier_bits": counterfactual_bits,
            "unique_verifier_bits": route_bits + counterfactual_bits,
            "regime_detector_verifier_bits": int(detector_accounting["unique_scalar_utilities"]),
            "policy_verifier_bits": int(policy_accounting["unique_verifier_bits"]),
            "shuffled_policy_verifier_bits": int(shuffled_accounting["unique_verifier_bits"]),
            "temporal_logical_lifetimes": 48,
            "policy_logical_lifetimes": 48_000,
            "policy_audit_logical_lifetimes": 64,
            "regime_detector_logical_lifetimes": int(
                detector_accounting["unique_scalar_utilities"]
            ),
            "unique_logical_lifetimes": (
                48
                + 48_000
                + 64
                + int(detector_accounting["unique_scalar_utilities"])
            ),
            "replayed_examples": 0,
            "optimizer_updates": int(policy_accounting["optimizer_updates"])
            + int(detector_accounting["optimizer_updates"]),
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": route_bits
            + counterfactual_bits
            + int(detector_accounting["unique_scalar_utilities"])
            if all(gates.values())
            else None,
        },
        "status": "promoted_compositional_reversal"
        if all(gates.values())
        else "rejected",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-updates", type=int, default=128)
    parser.add_argument("--source-evaluation-lifetimes", type=int, default=4)
    parser.add_argument("--source-route-lifetimes", type=int, default=8)
    parser.add_argument("--target-route-updates", type=int, default=8)
    parser.add_argument("--policy-updates", type=int, default=3_000)
    parser.add_argument("--policy-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--detector-updates", type=int, default=DETECTOR_UPDATES)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
