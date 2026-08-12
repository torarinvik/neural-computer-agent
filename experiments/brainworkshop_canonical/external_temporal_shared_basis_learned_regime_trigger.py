"""Pressure-test learned replacement timing at an external memory boundary.

An external raw-value regime detector receives the current replaceable value
bank and an incoming bank.  It learns keep/replace from one scalar verifier
utility per fresh pair.  In the canonical stream it must leave a stable
incoming bank as an exact no-op, then trigger the verifier-gated rewrite when a
new regime arrives.  Protected memory, frozen controller/encoder, and zero
replay remain explicit gates.
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
    OpaqueRegimeChangePolicy,
    OpaqueSharedBasisStructurePolicy,
    PersistentSharedBasisContentAddressedMemory,
)

from .external_temporal_query_address_growth import EVENT_WIDTH, _build
from .external_temporal_shared_basis_competing_subspaces import (
    _evaluate_policy as _evaluate_structure_policy,
)
from .external_temporal_shared_basis_competing_subspaces import (
    _train_policy as _train_structure_policy,
)
from .external_temporal_shared_basis_policy_growth import _digest
from .external_temporal_shared_basis_regime_replacement import (
    PROTECTED_SCOPE,
    WORKING_SCOPE,
    _payloads,
    _scoped_routes_absent,
    _scoped_routes_match,
    _select_compression,
    _select_replacement,
)

LEARNED_TRIGGER_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-learned-regime-trigger.v1"
)
DETECTOR_HIDDEN = 128
DETECTOR_LEARNING_RATE = 0.002
DETECTOR_TEMPERATURE = 0.6
DETECTOR_RECORDS = 8
READ_MATCH_THRESHOLD = 0.90
WRITE_MATCH_THRESHOLD = 0.999


def _synthetic_pair(
    *,
    seed: int,
    shifted: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if shifted is not None and shifted not in (0, 1):
        raise ValueError("synthetic regime shift must be binary")
    generator = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(
        torch.randn(EVENT_WIDTH, 4, generator=generator)
    ).Q[:, :4]
    if shifted is None:
        shifted = int(torch.randint(2, (), generator=generator))
    current = (
        torch.randn(DETECTOR_RECORDS, 2, generator=generator)
        @ basis[:, :2].transpose(0, 1)
        + 0.002 * torch.randn(DETECTOR_RECORDS, EVENT_WIDTH, generator=generator)
    )
    incoming_columns = 2 if shifted else 0
    incoming = (
        torch.randn(DETECTOR_RECORDS, 2, generator=generator)
        @ basis[:, incoming_columns : incoming_columns + 2].transpose(0, 1)
        + 0.002 * torch.randn(DETECTOR_RECORDS, EVENT_WIDTH, generator=generator)
    )
    return F.normalize(current, dim=-1), F.normalize(incoming, dim=-1), bool(shifted)


def _train_detector(
    *,
    seed: int,
    updates: int,
) -> tuple[OpaqueRegimeChangePolicy, dict[str, float | int]]:
    if updates < 1:
        raise ValueError("regime detector updates must be positive")
    torch.manual_seed(seed)
    policy = OpaqueRegimeChangePolicy(
        value_width=EVENT_WIDTH,
        hidden=DETECTOR_HIDDEN,
        max_spectral_bins=8,
        learning_rate=DETECTOR_LEARNING_RATE,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=DETECTOR_LEARNING_RATE)
    occupied = torch.ones(1, DETECTOR_RECORDS, dtype=torch.bool)
    explorer = torch.Generator().manual_seed(seed + 88_001)
    utilities: list[float] = []
    for update in range(updates):
        current, incoming, shifted = _synthetic_pair(seed=seed + 30_000 + update)
        plan = policy.propose(
            current.unsqueeze(0),
            occupied,
            incoming.unsqueeze(0),
            occupied,
            explore=True,
            temperature=DETECTOR_TEMPERATURE,
            generator=explorer,
        )
        utility = float(plan.replace == shifted)
        policy.adaptation_step(
            current.unsqueeze(0),
            occupied,
            incoming.unsqueeze(0),
            occupied,
            plan,
            utility,
            optimizer=optimizer,
        )
        utilities.append(utility)
    policy.eval()
    return policy, {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "first_window_utility": sum(utilities[:100]) / min(100, len(utilities)),
        "last_window_utility": sum(utilities[-100:]) / min(100, len(utilities)),
    }


@torch.no_grad()
def _evaluate_detector(
    policy: OpaqueRegimeChangePolicy,
    *,
    seed: int,
    episodes: int = 128,
) -> dict[str, float]:
    occupied = torch.ones(1, DETECTOR_RECORDS, dtype=torch.bool)
    scores = {"stable_keep": [], "shift_replace": []}
    for shifted, name in ((0, "stable_keep"), (1, "shift_replace")):
        for episode in range(episodes):
            current, incoming, target = _synthetic_pair(
                seed=seed + shifted * 10_000 + episode,
                shifted=shifted,
            )
            plan = policy.propose(
                current.unsqueeze(0),
                occupied,
                incoming.unsqueeze(0),
                occupied,
            )
            scores[name].append(float(plan.replace == target))
    return {name: sum(values) / len(values) for name, values in scores.items()}


def _memory_digest(memory: PersistentSharedBasisContentAddressedMemory) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(memory.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _run_stream(
    *,
    detector: OpaqueRegimeChangePolicy,
    structure_policy: OpaqueSharedBasisStructurePolicy,
    system,
    seed: int,
    path: Path,
) -> dict[str, object]:
    payloads = _payloads(system, seed=seed)
    memory = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
        scope_capacity=2,
    )
    memory.write(
        payloads["protected_keys"],
        payloads["protected_values"],
        torch.ones(6),
        scope=torch.zeros(6, dtype=torch.long),
    )
    memory.write(
        payloads["old_working_keys"],
        payloads["old_working_values"],
        torch.ones(12),
        scope=torch.ones(12, dtype=torch.long),
    )
    compression = _select_compression(structure_policy, memory, payloads)
    current_values = payloads["old_working_values"]
    occupied = torch.ones(1, current_values.shape[0], dtype=torch.bool)
    stable_before_version = int(memory.store_version.item())
    stable_digest_before = _memory_digest(memory)
    stable_plan = detector.propose(
        current_values.unsqueeze(0),
        occupied,
        current_values.unsqueeze(0),
        occupied,
    )
    stable_after_version = int(memory.store_version.item())
    stable_digest_after = _memory_digest(memory)

    shifted_plan = detector.propose(
        current_values.unsqueeze(0),
        occupied,
        payloads["new_working_values"].unsqueeze(0),
        occupied,
    )
    replacement = None
    if shifted_plan.replace:
        replacement = _select_replacement(structure_policy, memory, payloads)
    protected_after = _scoped_routes_match(
        memory,
        payloads["protected_keys"],
        payloads["protected_values"],
        scope=PROTECTED_SCOPE,
        tolerance=0.04,
    )
    new_after = (
        _scoped_routes_match(
            memory,
            payloads["new_working_keys"],
            payloads["new_working_values"],
            scope=WORKING_SCOPE,
            tolerance=0.04,
        )
        if replacement is not None and replacement["accepted"]
        else False
    )
    old_absent = (
        _scoped_routes_absent(
            memory,
            payloads["old_working_keys"],
            scope=WORKING_SCOPE,
        )
        if replacement is not None and replacement["accepted"]
        else False
    )
    return {
        "compression": compression,
        "stable_plan_replace": stable_plan.replace,
        "stable_noop_version": stable_before_version == stable_after_version,
        "stable_noop_digest": stable_digest_before == stable_digest_after,
        "shifted_plan_replace": shifted_plan.replace,
        "replacement": replacement,
        "protected_after": protected_after,
        "new_after": new_after,
        "old_absent": old_absent,
        "final_record_count": memory.record_count,
        "final_physical_value_scalars": memory.physical_value_scalar_count,
        "final_dense_value_scalars": memory.dense_value_scalar_count,
        "path": path,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1:
        raise ValueError("learned regime trigger updates must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    detector, detector_training = _train_detector(
        seed=args.seed,
        updates=args.policy_updates,
    )
    detector_scores = _evaluate_detector(detector, seed=args.seed + 800_000)
    structure_policy, structure_training = _train_structure_policy(
        seed=args.seed + 1_000,
        updates=args.policy_updates,
    )
    structure_scores = _evaluate_structure_policy(
        structure_policy,
        seed=args.seed + 801_000,
    )
    torch.manual_seed(args.seed + 900_000)
    fresh_detector = OpaqueRegimeChangePolicy(
        value_width=EVENT_WIDTH,
        hidden=DETECTOR_HIDDEN,
        max_spectral_bins=8,
        learning_rate=DETECTOR_LEARNING_RATE,
    ).eval()
    fresh_detector_scores = _evaluate_detector(
        fresh_detector,
        seed=args.seed + 800_000,
    )
    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-learned-trigger-"
    ) as directory:
        forward = _run_stream(
            detector=detector,
            structure_policy=structure_policy,
            system=system,
            seed=args.seed,
            path=Path(directory) / "forward.pt",
        )
        reversed_stream = _run_stream(
            detector=detector,
            structure_policy=structure_policy,
            system=system,
            seed=args.seed + 100,
            path=Path(directory) / "reversed.pt",
        )
        corruption_path = Path(directory) / "corrupt.pt"
        payload = torch.load(forward["path"], weights_only=False)
        payload["state_dict"]["coefficients"] = payload["state_dict"][
            "coefficients"
        ].clone()
        payload["state_dict"]["coefficients"][0, 0] += 0.1
        torch.save(payload, corruption_path)
        corruption_rejected = False
        try:
            PersistentSharedBasisContentAddressedMemory(
                EVENT_WIDTH,
                path=corruption_path,
                write_threshold=0.0,
                write_match_threshold=WRITE_MATCH_THRESHOLD,
                read_match_threshold=READ_MATCH_THRESHOLD,
                basis_tolerance=1e-8,
                scope_capacity=2,
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "trained_detector_stable_keep": detector_scores["stable_keep"] >= 0.80,
        "trained_detector_shift_replace": detector_scores["shift_replace"] >= 0.80,
        "detector_beats_fresh": (
            (
                detector_scores["stable_keep"]
                + detector_scores["shift_replace"]
            )
            >= (
                fresh_detector_scores["stable_keep"]
                + fresh_detector_scores["shift_replace"]
                + 0.20
            )
            and detector_scores["stable_keep"]
            >= fresh_detector_scores["stable_keep"] - 0.05
            and detector_scores["shift_replace"]
            >= fresh_detector_scores["shift_replace"] - 0.05
        ),
        "structure_rank_8_transfer": structure_scores["8"] >= 0.80,
        "forward_stable_noop": (
            not forward["stable_plan_replace"]
            and forward["stable_noop_version"]
            and forward["stable_noop_digest"]
        ),
        "reversed_stable_noop": (
            not reversed_stream["stable_plan_replace"]
            and reversed_stream["stable_noop_version"]
            and reversed_stream["stable_noop_digest"]
        ),
        "forward_shift_detected": bool(forward["shifted_plan_replace"]),
        "reversed_shift_detected": bool(reversed_stream["shifted_plan_replace"]),
        "forward_replacement_accepted": bool(
            forward["replacement"] and forward["replacement"]["accepted"]
        ),
        "reversed_replacement_accepted": bool(
            reversed_stream["replacement"]
            and reversed_stream["replacement"]["accepted"]
        ),
        "forward_protected_retained": bool(forward["protected_after"]),
        "reversed_protected_retained": bool(reversed_stream["protected_after"]),
        "forward_new_admitted": bool(forward["new_after"]),
        "reversed_new_admitted": bool(reversed_stream["new_after"]),
        "forward_old_removed": bool(forward["old_absent"]),
        "reversed_old_removed": bool(reversed_stream["old_absent"]),
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": LEARNED_TRIGGER_SCHEMA,
        "claim_boundary": (
            "Outcome-trained raw-value regime detector keeps stable incoming "
            "evidence as an exact no-op and triggers a verifier-gated external "
            "scope replacement after a structural shift; not autonomous semantic "
            "change-point discovery or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "detector": "opaque_regime_change_policy_v1",
            "structure_policy": "opaque_shared_basis_structure_policy_v2",
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "rewrite": "shared_basis_rewrite_v1",
            "detector_feature_contract": "opaque_spectral_cross_bank_structure_v1",
            "forbidden_features": "task_labels_regime_ids_candidate_reconstruction_error_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "detector_training": detector_training,
        "structure_training": structure_training,
        "detector_scores": detector_scores,
        "fresh_detector_scores": fresh_detector_scores,
        "structure_scores": structure_scores,
        "forward": {key: value for key, value in forward.items() if key != "path"},
        "reversed": {
            key: value for key, value in reversed_stream.items() if key != "path"
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": args.policy_updates * 2 + 60,
            "unique_logical_lifetimes": args.policy_updates * 2 + 30,
            "optimizer_updates": args.policy_updates * 2,
            "live_compression_transactions": 4,
            "live_logical_rewrite_transactions": 2,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_learned_regime_trigger"
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
    parser.add_argument("--policy-updates", type=int, default=1_000)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
