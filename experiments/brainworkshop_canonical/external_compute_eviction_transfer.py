"""Test learned eviction transfer on a held-out external compute family.

The promoted six-file eviction rung calibrates and serves one family cohort.
This audit freezes that cohort, introduces a held-out n-back-2 artifact, and
compares the inherited memory-side policy with a matched fresh policy. Both
policies receive the same fresh verifier probes and may update only external
policy state; the controller, event encoder, and executable artifacts remain
frozen.

This is a transfer measurement, not an assumption that the inherited policy
must win. A failed transfer is valuable evidence about the remaining
continual-learning bottleneck.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import (
    EpisodicBindingArtifactIndex,
    ExternalCapabilityEvictionPolicy,
    GatedResidualCapabilityEvictionPolicyBank,
    IntentEvent,
    PermutationInvariantCapabilityEvictionPolicy,
    VerifierGatedCapabilityEvictionPolicyBank,
    paired_counterfactual_ranking_loss,
)

from .external_compute_artifact_cache_pressure import (
    _active_digests,
    _append_compute_file,
    _discard_newest_compute_file,
    _FileSnapshot,
    _restore_snapshot,
    _route_key,
    _snapshot,
    _stable,
    _train_file,
)
from .external_compute_growth import (
    ACTION_COUNT,
    ENCODER_SYMBOL_COUNT,
    EVENT_WIDTH,
    ComputeGrowthSystem,
    _build,
    _digest,
    _evaluate,
)
from .external_compute_learned_eviction_scale import (
    ACTIVE_CACHE_SLOTS,
    MASTERY_THRESHOLD,
    MATCHING_THRESHOLD,
    POLICY_CANDIDATE_WIDTH,
    POLICY_CONTEXT_WIDTH,
    POLICY_HIDDEN,
    SCHEDULE,
    SOURCE_PROTECTION_OBSERVATIONS,
    _adapt_policy,
    _eligible_slots,
    _policy_scores,
    _PolicyProbe,
    _probe_active,
    _select_victim,
)
from .external_compute_route_bank import _family_steps

SCHEMA = "neural-computer.brainworkshop-external-compute-eviction-transfer.v1"
TRANSFER_FAMILY = "nback2"
TRANSFER_CUE = 9
UTILITY_GAP_GATE = 0.15
STABILITY_WINDOW = 3
BEHAVIORAL_ARTIFACT_SIGNATURE_SCHEMA = (
    "neural-computer.external-compute-behavioral-artifact-signature.v1"
)
BEHAVIORAL_ARTIFACT_SIGNATURE_V2_SCHEMA = (
    "neural-computer.external-compute-behavioral-artifact-signature.v2"
)
BEHAVIOR_PROBE_SEQUENCES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 0),
    (1, 7),
)
BEHAVIORAL_SIGNATURE_PROJECTION_SEED = 20260813
BEHAVIOR_PROBE_SEQUENCES_V2 = (
    (0, 1, 2, 3),
    (2, 4, 6, 8),
    (4, 7, 10, 0),
    (6, 9, 12, 2),
    (8, 11, 1, 4),
    (10, 0, 3, 6),
    (12, 2, 5, 8),
    (1, 5, 9, 7),
)
BEHAVIORAL_SIGNATURE_V2_PROJECTION_SEED = 20260814


def _make_policy(kind: str):
    if kind == "pointwise":
        policy_class = ExternalCapabilityEvictionPolicy
    elif kind == "set_relative":
        policy_class = PermutationInvariantCapabilityEvictionPolicy
    else:
        raise ValueError(f"unknown eviction policy kind: {kind}")
    return policy_class(
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )


@torch.no_grad()
def _behavioral_artifact_feature_bank(
    system: ComputeGrowthSystem,
    snapshots: Mapping[str, _FileSnapshot],
) -> dict[str, torch.Tensor]:
    """Describe artifacts by a fixed standardized-event execution trace.

    The policy receives this signature instead of sampled parameter
    coordinates.  The probe is deliberately fixed and opaque: it supplies
    only learned event tensors, controller intentions, and neutral feedback
    context to the shared interpreter, then records the artifact's register,
    intention, and decoder traces.  No verifier outcomes, family IDs, or
    correct actions enter the signature.
    """

    if not snapshots:
        raise ValueError("behavioral signature bank cannot be empty")
    probe_symbols = torch.tensor(BEHAVIOR_PROBE_SEQUENCES, dtype=torch.long)
    batch_size = probe_symbols.shape[0]
    if probe_symbols.ndim != 2 or probe_symbols.shape[1] < 1:
        raise ValueError("behavioral probe sequences must be non-empty")

    controller_state = system.agent.initial_state(batch_size, device="cpu")
    feedback = system.agent.initial_feedback(batch_size, device="cpu")
    events: list[torch.Tensor] = []
    intentions: list[IntentEvent] = []
    for timestep in range(probe_symbols.shape[1]):
        collection = system.agent.runtime.encode_streams(
            {"stimulus": probe_symbols[:, timestep]}
        )
        controller_output, controller_state = system.agent.runtime.step_events(
            collection,
            controller_state,
            feedback,
        )
        events.append(collection.payload[:, 0].detach())
        intentions.append(controller_output.intention)

    original = _snapshot(system, 0)
    trace_by_handle: dict[str, torch.Tensor] = {}
    projection: torch.Tensor | None = None
    try:
        for handle, snapshot in snapshots.items():
            _restore_snapshot(system, 0, snapshot)
            register_state = system.machine.initial_state(batch_size, device="cpu")
            action = torch.zeros(batch_size, ACTION_COUNT)
            outcome = torch.zeros(batch_size)
            trace: list[torch.Tensor] = []
            for event, intention in zip(events, intentions, strict=True):
                executed, register_state = system.machine.read_execute_register(
                    event=event,
                    action=action,
                    outcome=outcome,
                    intention=intention,
                    state=register_state,
                    instructions=(system.instructions[0],),
                    basis_slots=(0,),
                )
                readout_intention = system.readouts[0](executed)
                logits = system.decoders[0](IntentEvent(readout_intention))
                trace.append(
                    torch.cat((executed, readout_intention, logits), dim=-1)
                )
            flat = torch.cat(trace, dim=-1).reshape(-1).to(torch.float32)
            if projection is None:
                generator = torch.Generator(device="cpu").manual_seed(
                    BEHAVIORAL_SIGNATURE_PROJECTION_SEED
                )
                projection = torch.randn(
                    flat.numel(),
                    POLICY_CANDIDATE_WIDTH,
                    generator=generator,
                )
            elif flat.numel() != projection.shape[0]:
                raise RuntimeError("behavioral signature trace width changed")
            trace_by_handle[handle] = F.normalize(flat @ projection, dim=0).cpu()
    finally:
        _restore_snapshot(system, 0, original)
    return trace_by_handle


def _controller_probe_segments(
    system: ComputeGrowthSystem,
    probe_symbols: torch.Tensor,
) -> tuple[
    tuple[tuple[torch.Tensor, IntentEvent], ...],
    tuple[tuple[torch.Tensor, IntentEvent], ...],
]:
    """Produce continuous and reset-segment standardized controller traces."""

    if probe_symbols.ndim != 2 or probe_symbols.shape[1] < 2:
        raise ValueError("v2 behavioral probes need at least two timesteps")
    batch_size = probe_symbols.shape[0]
    split = probe_symbols.shape[1] // 2
    if split < 1 or split == probe_symbols.shape[1]:
        raise ValueError("v2 behavioral probes need a non-empty split")

    def collect(
        symbols: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, IntentEvent], ...]:
        state = system.agent.initial_state(batch_size, device="cpu")
        feedback = system.agent.initial_feedback(batch_size, device="cpu")
        result: list[tuple[torch.Tensor, IntentEvent]] = []
        for timestep in range(symbols.shape[1]):
            collection = system.agent.runtime.encode_streams(
                {"stimulus": symbols[:, timestep]}
            )
            controller_output, state = system.agent.runtime.step_events(
                collection,
                state,
                feedback,
            )
            result.append((collection.payload[:, 0].detach(), controller_output.intention))
        return tuple(result)

    continuous = collect(probe_symbols)
    reset_segments = collect(probe_symbols[:, :split]) + collect(
        probe_symbols[:, split:]
    )
    return continuous, reset_segments


@torch.no_grad()
def _behavioral_artifact_feature_bank_v2(
    system: ComputeGrowthSystem,
    snapshots: Mapping[str, _FileSnapshot],
) -> dict[str, torch.Tensor]:
    """Describe files using continuous and reset-segment behavior traces.

    v1 used two-step traces.  v2 intentionally adds a temporal intervention:
    the same fixed learned event sequences are executed once continuously and
    once as two reset segments.  The policy still sees only the resulting
    fixed-width trace; reset boundaries, symbols, verifier outcomes, family
    names, and correct actions are not inputs to the policy.
    """

    if not snapshots:
        raise ValueError("behavioral signature bank cannot be empty")
    probe_symbols = torch.tensor(BEHAVIOR_PROBE_SEQUENCES_V2, dtype=torch.long)
    continuous, reset_segments = _controller_probe_segments(system, probe_symbols)
    original = _snapshot(system, 0)
    trace_by_handle: dict[str, torch.Tensor] = {}
    projection: torch.Tensor | None = None
    try:
        for handle, snapshot in snapshots.items():
            _restore_snapshot(system, 0, snapshot)
            traces: list[torch.Tensor] = []
            for mode_index, mode in enumerate((continuous, reset_segments)):
                register_state = system.machine.initial_state(
                    probe_symbols.shape[0], device="cpu"
                )
                reset_at = (
                    probe_symbols.shape[1] // 2 if mode_index == 1 else None
                )
                action = torch.zeros(probe_symbols.shape[0], ACTION_COUNT)
                outcome = torch.zeros(probe_symbols.shape[0])
                mode_trace: list[torch.Tensor] = []
                for step, (event, intention) in enumerate(mode):
                    executed, register_state = system.machine.read_execute_register(
                        event=event,
                        action=action,
                        outcome=outcome,
                        intention=intention,
                        state=register_state,
                        instructions=(system.instructions[0],),
                        basis_slots=(0,),
                    )
                    readout_intention = system.readouts[0](executed)
                    logits = system.decoders[0](IntentEvent(readout_intention))
                    mode_trace.append(
                        torch.cat((executed, readout_intention, logits), dim=-1)
                    )
                    if reset_at is not None and step + 1 == reset_at:
                        register_state = system.machine.initial_state(
                            probe_symbols.shape[0], device="cpu"
                        )
                traces.append(torch.cat(mode_trace, dim=-1))
            flat = torch.cat(traces, dim=-1).reshape(-1).to(torch.float32)
            if projection is None:
                generator = torch.Generator(device="cpu").manual_seed(
                    BEHAVIORAL_SIGNATURE_V2_PROJECTION_SEED
                )
                projection = torch.randn(
                    flat.numel(),
                    POLICY_CANDIDATE_WIDTH,
                    generator=generator,
                )
            elif flat.numel() != projection.shape[0]:
                raise RuntimeError("v2 behavioral signature trace width changed")
            trace_by_handle[handle] = F.normalize(flat @ projection, dim=0).cpu()
    finally:
        _restore_snapshot(system, 0, original)
    return trace_by_handle


def _candidate_signature_config(candidate_signature: str) -> dict[str, object]:
    if candidate_signature == "behavioral":
        return {
            "schema": BEHAVIORAL_ARTIFACT_SIGNATURE_SCHEMA,
            "source": "frozen_standardized_event_and_intention_probe_v1",
            "probe_sequences": len(BEHAVIOR_PROBE_SEQUENCES),
            "sequence_length": len(BEHAVIOR_PROBE_SEQUENCES[0]),
            "trace": "register_readout_intention_decoder_logits_v1",
            "projection_width": POLICY_CANDIDATE_WIDTH,
            "raw_weight_coordinates": False,
            "probe_symbol_count": ENCODER_SYMBOL_COUNT,
        }
    if candidate_signature == "behavioral_v2":
        return {
            "schema": BEHAVIORAL_ARTIFACT_SIGNATURE_V2_SCHEMA,
            "source": "frozen_standardized_event_and_intention_probe_v2",
            "probe_sequences": len(BEHAVIOR_PROBE_SEQUENCES_V2),
            "sequence_length": len(BEHAVIOR_PROBE_SEQUENCES_V2[0]),
            "trace": "continuous_and_reset_register_readout_intention_decoder_logits_v2",
            "projection_width": POLICY_CANDIDATE_WIDTH,
            "raw_weight_coordinates": False,
            "probe_symbol_count": ENCODER_SYMBOL_COUNT,
            "reset_segments": 2,
        }
    return {
        "schema": "raw-parameter-coordinate-descriptor",
        "source": "sampled_artifact_state_coordinates",
        "projection_width": POLICY_CANDIDATE_WIDTH,
        "raw_weight_coordinates": True,
    }


def _permute_probe(
    probe: _PolicyProbe,
    permutation: torch.Tensor,
) -> tuple[_PolicyProbe, tuple[int, ...]]:
    """Reorder candidate rows while keeping verifier outcomes attached."""

    if permutation.ndim != 1 or permutation.shape[0] != ACTIVE_CACHE_SLOTS:
        raise ValueError("candidate permutation has the wrong shape")
    order = tuple(int(index) for index in permutation.tolist())
    if set(order) != set(range(ACTIVE_CACHE_SLOTS)):
        raise ValueError("candidate permutation is not a bijection")
    return (
        _PolicyProbe(
            context=probe.context,
            features=probe.features.index_select(0, permutation),
            outcomes={
                new_index: probe.outcomes[old_index]
                for new_index, old_index in enumerate(order)
            },
            unique_verifier_bits=probe.unique_verifier_bits,
        ),
        order,
    )


def _acquire_source_cohort(
    args: argparse.Namespace,
) -> tuple[
    ComputeGrowthSystem,
    EpisodicBindingArtifactIndex,
    dict[str, _FileSnapshot],
    list[list[dict[str, float | int]]],
    int,
    int,
    int,
]:
    """Acquire the six-file source cohort and protect its first file."""

    system = _build(args.seed, slot_count=ACTIVE_CACHE_SLOTS)
    index = EpisodicBindingArtifactIndex.create(
        EVENT_WIDTH,
        EVENT_WIDTH,
        active_slots=ACTIVE_CACHE_SLOTS,
        matching_threshold=MATCHING_THRESHOLD,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=SOURCE_PROTECTION_OBSERVATIONS,
        reversal_threshold=0.5,
        reversal_patience=4,
    )
    snapshots: dict[str, _FileSnapshot] = {}
    direct: list[list[dict[str, float | int]]] = []
    verifier_bits = 0
    optimizer_updates = 0
    logical_lifetimes = 0
    source_binding_id: int | None = None

    for route_id, (family, cue_symbol) in enumerate(SCHEDULE):
        scratch = route_id >= ACTIVE_CACHE_SLOTS
        slot = (
            _append_compute_file(system, seed=args.seed + 80_000 + route_id)
            if scratch
            else route_id
        )
        history, fresh = _train_file(
            system,
            slot=slot,
            family=family,
            cue_symbol=cue_symbol,
            updates=args.file_updates,
            batch_size=args.batch_size,
            seed=args.seed + 10_000 * (route_id + 1),
            learning_rate=args.learning_rate,
            first_file=route_id == 0,
        )
        optimizer_updates += len(history)
        logical_lifetimes += args.batch_size * (len(history) + len(fresh))
        verifier_bits += sum(int(row["unique_verifier_bits"]) for row in history)
        verifier_bits += sum(int(row["unique_verifier_bits"]) for row in fresh)
        direct.append(fresh)
        if not _stable(fresh):
            if scratch:
                _discard_newest_compute_file(system)
            raise RuntimeError(f"source family {family} failed fresh mastery")

        artifact = _snapshot(system, slot)
        snapshots[artifact.digest] = artifact
        key = _route_key(system, cue_symbol)
        binding_id = index.register(key, key, artifact.digest)
        if scratch:
            _discard_newest_compute_file(system)
        else:
            index.activate(binding_id, slot)
        index.archive.observe(
            binding_id,
            min(float(row["accuracy"]) for row in fresh),
            step=route_id + 1,
        )

        if route_id == 0:
            source_binding_id = binding_id
            for observation in range(SOURCE_PROTECTION_OBSERVATIONS):
                rows = _evaluate(
                    system,
                    family=family,
                    slot=0,
                    cue_symbol=cue_symbol,
                    lifetimes=1,
                    batch_size=args.batch_size,
                    steps=_family_steps(family),
                    seed=args.seed + 300_000 + observation,
                )
                verifier_bits += sum(
                    int(row["unique_verifier_bits"]) for row in rows
                )
                logical_lifetimes += args.batch_size * len(rows)
                index.archive.observe(
                    binding_id,
                    min(float(row["accuracy"]) for row in rows),
                    step=100 + observation,
                )

    if source_binding_id is None:
        raise RuntimeError("source protection binding was not created")
    return (
        system,
        index,
        snapshots,
        direct,
        verifier_bits,
        optimizer_updates,
        logical_lifetimes,
    )


def _acquire_transfer_file(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: dict[str, _FileSnapshot],
    *,
    args: argparse.Namespace,
) -> tuple[int, str, list[dict[str, float | int]], int, int]:
    """Train the held-out file in scratch capacity and archive it cold."""

    slot = _append_compute_file(system, seed=args.seed + 90_000)
    history, fresh = _train_file(
        system,
        slot=slot,
        family=TRANSFER_FAMILY,
        cue_symbol=TRANSFER_CUE,
        updates=args.file_updates,
        batch_size=args.batch_size,
        seed=args.seed + 700_000,
        learning_rate=args.learning_rate,
        first_file=False,
    )
    bits = sum(int(row["unique_verifier_bits"]) for row in history)
    bits += sum(int(row["unique_verifier_bits"]) for row in fresh)
    logical_lifetimes = args.batch_size * (len(history) + len(fresh))
    if not _stable(fresh):
        _discard_newest_compute_file(system)
        raise RuntimeError("held-out family failed fresh mastery")
    artifact = _snapshot(system, slot)
    snapshots[artifact.digest] = artifact
    key = _route_key(system, TRANSFER_CUE)
    binding_id = index.register(key, key, artifact.digest)
    _discard_newest_compute_file(system)
    index.archive.observe(
        binding_id,
        min(float(row["accuracy"]) for row in fresh),
        step=10_000,
    )
    return binding_id, artifact.digest, fresh, bits, logical_lifetimes


def _install_related_source(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    *,
    args: argparse.Namespace,
) -> tuple[bool, int, int, dict[str, object]]:
    """Place the known n-back-3 source in the hot cache through verification."""

    lookup = index.lookup(_route_key(system, 4))
    if lookup.binding_id is None or lookup.artifact_handle is None:
        raise RuntimeError("related n-back-3 source is missing")
    if lookup.active_slot is not None:
        return True, 0, 0, {"already_active_slot": lookup.active_slot}
    destination = 2
    displaced = index.archive.active_binding(destination)
    displaced_handle = (
        index.artifact_handle(displaced) if displaced is not None else None
    )
    _restore_snapshot(system, destination, snapshots[lookup.artifact_handle])
    rows: list[dict[str, float | int]] = []

    def verify(_candidate: EpisodicBindingArtifactIndex) -> bool:
        rows.extend(
            _evaluate(
                system,
                family="nback3",
                slot=destination,
                cue_symbol=4,
                lifetimes=1,
                batch_size=args.batch_size,
                steps=_family_steps("nback3"),
                seed=args.seed + 850_000,
            )
        )
        return _stable(rows)

    receipt = index.reactivate_verified(lookup.binding_id, destination, verify)
    if not receipt.accepted and displaced_handle is not None:
        _restore_snapshot(system, destination, snapshots[displaced_handle])
    return (
        receipt.accepted,
        sum(int(row["unique_verifier_bits"]) for row in rows),
        args.batch_size * len(rows),
        {
            "binding_id": lookup.binding_id,
            "destination_slot": destination,
            "receipt": receipt.__dict__,
            "retention_probe": rows,
        },
    )


def _policy_update(
    policy,
    optimizer: torch.optim.Optimizer,
    probe,
    *,
    eligible: tuple[int, ...],
) -> dict[str, float | int]:
    """Update one policy using only a shared probe's paired scalar outcomes."""

    if len(eligible) < 2:
        raise RuntimeError("transfer policy needs two eligible residents")
    pair = (eligible[0], eligible[1])
    utility = torch.tensor(
        [[1.0 - probe.outcomes[pair[0]], 1.0 - probe.outcomes[pair[1]]]],
        dtype=torch.float32,
    )
    scores = _policy_scores(policy, probe.context, probe.features)
    training_scores = scores
    if isinstance(policy, VerifierGatedCapabilityEvictionPolicyBank):
        training_scores = policy.probationary_training_scores(
            probe.context.unsqueeze(0),
            probe.features.unsqueeze(0),
        ).squeeze(0)
    masked = scores.clone()
    for slot in range(ACTIVE_CACHE_SLOTS):
        if slot not in eligible:
            masked[slot] = -torch.inf
    chosen = int(masked.argmax())
    oracle = max(eligible, key=lambda slot: 1.0 - probe.outcomes[slot])
    loss, advantage = paired_counterfactual_ranking_loss(
        training_scores.unsqueeze(0),
        torch.tensor([pair], dtype=torch.long),
        utility,
    )
    gap = abs(float(advantage.item()))
    updated = gap >= UTILITY_GAP_GATE
    if updated and loss.requires_grad:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [
            parameter for parameter in policy.parameters() if parameter.grad is not None
        ]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
        else:
            updated = False
    elif updated:
        updated = False
    return {
        "chosen_slot": chosen,
        "oracle_slot": oracle,
        "selection_correct": int(chosen == oracle),
        "policy_updated": int(updated),
        "utility_gap": float(advantage.item()),
        "loss": float(loss.detach()),
        "unique_verifier_bits": probe.unique_verifier_bits,
        "replayed_examples": 0,
    }


def _transfer_curve(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    inherited: ExternalCapabilityEvictionPolicy,
    fresh: ExternalCapabilityEvictionPolicy,
    inherited_optimizer: torch.optim.Optimizer,
    fresh_optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    permute_candidates: bool,
    feature_bank: Mapping[str, torch.Tensor] | None = None,
) -> tuple[list[dict[str, object]], int, int, int]:
    """Give both policies the same fresh held-out-family probes."""

    eligible = _eligible_slots(index)
    rows: list[dict[str, object]] = []
    bits = 0
    policy_updates = 0
    logical_lifetimes = 0
    for update in range(args.transfer_updates):
        raw_probe = _probe_active(
            system,
            index,
            snapshots,
            family=TRANSFER_FAMILY,
            cue_symbol=TRANSFER_CUE,
            batch_size=args.batch_size,
            seed=args.seed + 1_000_000 + update * 10_007,
            retention_lifetimes=1,
            feature_bank=feature_bank,
        )
        if permute_candidates:
            permutation = torch.randperm(
                ACTIVE_CACHE_SLOTS,
                generator=torch.Generator().manual_seed(
                    args.seed + 3_000_000 + update
                ),
            )
            probe, candidate_order = _permute_probe(raw_probe, permutation)
        else:
            probe = raw_probe
            candidate_order = tuple(range(ACTIVE_CACHE_SLOTS))
        policy_eligible = tuple(
            position
            for position, physical_slot in enumerate(candidate_order)
            if physical_slot in eligible
        )
        bits += probe.unique_verifier_bits
        logical_lifetimes += args.batch_size * ACTIVE_CACHE_SLOTS
        inherited_row = _policy_update(
            inherited, inherited_optimizer, probe, eligible=policy_eligible
        )
        safety_row: dict[str, float | int | bool] | None = None
        if isinstance(inherited, VerifierGatedCapabilityEvictionPolicyBank):
            safety_row = inherited.observe_verifier_probe(
                probe.context.unsqueeze(0),
                probe.features.unsqueeze(0),
                torch.tensor(
                    [
                        1.0 - probe.outcomes[slot]
                        for slot in range(ACTIVE_CACHE_SLOTS)
                    ],
                    dtype=torch.float32,
                ),
                0,
                eligible=policy_eligible,
            )
            if (
                bool(safety_row["safe"])
                and int(safety_row["probe_count"])
                >= inherited.minimum_probe_observations
                and not bool(inherited.slot_trusted[0])
            ):
                inherited.promote_slot(0)
        fresh_row = _policy_update(
            fresh,
            fresh_optimizer,
            probe,
            eligible=policy_eligible,
        )
        for row in (inherited_row, fresh_row):
            row["chosen_slot"] = candidate_order[int(row["chosen_slot"])]
            row["oracle_slot"] = candidate_order[int(row["oracle_slot"])]
        if safety_row is not None:
            safety_row["base_index"] = candidate_order[int(safety_row["base_index"])]
            safety_row["combined_index"] = candidate_order[
                int(safety_row["combined_index"])
            ]
        policy_updates += int(inherited_row["policy_updated"])
        policy_updates += int(fresh_row["policy_updated"])
        rows.append(
            {
                "update": update + 1,
                "candidate_order": candidate_order,
                "inherited": inherited_row,
                "fresh": fresh_row,
                **({"safety_gate": safety_row} if safety_row is not None else {}),
            }
        )
    for row in rows:
        prefix = rows[: int(row["update"])]
        row["inherited_cumulative_accuracy"] = sum(
            int(item["inherited"]["selection_correct"]) for item in prefix
        ) / len(prefix)
        row["fresh_cumulative_accuracy"] = sum(
            int(item["fresh"]["selection_correct"]) for item in prefix
        ) / len(prefix)
    return rows, bits, policy_updates, logical_lifetimes


def _stable_window(values: list[int]) -> bool:
    return any(
        sum(values[index : index + STABILITY_WINDOW])
        >= 0.60 * STABILITY_WINDOW
        for index in range(len(values) - STABILITY_WINDOW + 1)
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.file_updates,
        args.batch_size,
        args.policy_calibration_rounds,
        args.policy_updates_per_round,
        args.transfer_updates,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("transfer budgets must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated transfer harness requires batch size 32")
    policy_kind = getattr(args, "policy_kind", "pointwise")
    residual_gain = float(getattr(args, "residual_gain", 32.0))
    safety_gate = bool(getattr(args, "safety_gate", False))
    permute_candidates = bool(getattr(args, "permute_candidates", False))
    probationary_fallback = getattr(args, "probationary_fallback", "base")
    if probationary_fallback not in {"base", "neutral"}:
        raise ValueError(
            f"unknown probationary fallback: {probationary_fallback}"
        )
    candidate_signature = getattr(args, "candidate_signature", "raw")
    if candidate_signature not in {"raw", "behavioral", "behavioral_v2"}:
        raise ValueError(f"unknown candidate signature: {candidate_signature}")

    try:
        (
            system,
            index,
            snapshots,
            direct,
            verifier_bits,
            optimizer_updates,
            logical_lifetimes,
        ) = _acquire_source_cohort(args)
    except RuntimeError as error:
        report = {
            "schema": SCHEMA,
            "seed": args.seed,
            "claim_boundary": (
                "Rejected before source-cohort qualification; no transfer "
                "claim is made."
            ),
            "error": str(error),
            "gates": {
                "source_cohort_mastered": False,
                "frozen_controller": False,
                "frozen_event_encoder": False,
                "zero_replayed_examples": True,
            },
            "accounting": {
                "replayed_examples": 0,
                "unique_logical_lifetimes": 0,
                "stable_bits_to_threshold": None,
            },
            "status": "rejected",
        }
        if args.report_out is not None:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(json.dumps(report, indent=2) + "\n")
        return report
    (
        transfer_id,
        transfer_digest,
        transfer_direct,
        transfer_bits,
        transfer_lifetimes,
    ) = (
        _acquire_transfer_file(system, index, snapshots, args=args)
    )
    verifier_bits += transfer_bits
    logical_lifetimes += transfer_lifetimes
    optimizer_updates += args.file_updates
    (
        related_installed,
        related_bits,
        related_lifetimes,
        related_setup,
    ) = _install_related_source(
        system,
        index,
        snapshots,
        args=args,
    )
    verifier_bits += related_bits
    logical_lifetimes += related_lifetimes
    if candidate_signature == "behavioral":
        behavioral_features = _behavioral_artifact_feature_bank(system, snapshots)
    elif candidate_signature == "behavioral_v2":
        behavioral_features = _behavioral_artifact_feature_bank_v2(system, snapshots)
    else:
        behavioral_features = None
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])

    base_policy = _make_policy(policy_kind)
    base_optimizer = torch.optim.Adam(
        base_policy.parameters(), lr=args.policy_learning_rate
    )
    source_calibration: list[dict[str, object]] = []
    policy_bits = 0
    policy_updates = 0
    for round_index in range(args.policy_calibration_rounds):
        family, cue_symbol = SCHEDULE[round_index % len(SCHEDULE)]
        history, probe = _adapt_policy(
            base_policy,
            base_optimizer,
            system,
            index,
            snapshots,
            family=family,
            cue_symbol=cue_symbol,
            batch_size=args.batch_size,
            updates=args.policy_updates_per_round,
            seed=args.seed + 500_000 + round_index * 10_000,
            retention_lifetimes=1,
            minimum_utility_gap=UTILITY_GAP_GATE,
            feature_bank=behavioral_features,
        )
        policy_bits += sum(int(row["unique_verifier_bits"]) for row in history)
        policy_updates += len(history)
        logical_lifetimes += args.batch_size * len(history) * ACTIVE_CACHE_SLOTS
        if probe is not None:
            chosen, oracle, correct, scores = _select_victim(
                base_policy, index, snapshots, probe
            )
            source_calibration.append(
                {
                    "round": round_index + 1,
                    "family": family,
                    "chosen_slot": chosen,
                    "oracle_slot": oracle,
                    "selection_correct": correct,
                    "scores": scores.tolist(),
                }
            )

    residual_policy_class = (
        VerifierGatedCapabilityEvictionPolicyBank
        if safety_gate
        else GatedResidualCapabilityEvictionPolicyBank
    )
    residual_policy = residual_policy_class(
        base_policy,
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        max_slots=1,
        route_threshold=0.75,
        residual_gain=residual_gain,
        **(
            {
                "minimum_probe_observations": 4,
                "noninferiority_margin": 0.0,
                "probationary_fallback": probationary_fallback,
            }
            if safety_gate
            else {}
        ),
    )
    residual_slot = residual_policy.add_slot(_route_key(system, TRANSFER_CUE))
    residual_policy.activate_slot(residual_slot)
    residual_optimizer = torch.optim.Adam(
        residual_policy.trainable_parameters(residual_slot),
        lr=args.policy_learning_rate,
    )
    fresh = _make_policy(policy_kind)
    fresh_optimizer = torch.optim.Adam(
        fresh.parameters(), lr=args.policy_learning_rate
    )
    active_before = _active_digests(system, index, snapshots)
    transfer_curve, curve_bits, curve_updates, curve_lifetimes = _transfer_curve(
        system,
        index,
        snapshots,
        residual_policy,
        fresh,
        residual_optimizer,
        fresh_optimizer,
        args=args,
        permute_candidates=permute_candidates,
        feature_bank=behavioral_features,
    )
    policy_bits += curve_bits
    policy_updates += curve_updates
    logical_lifetimes += curve_lifetimes

    probe = _probe_active(
        system,
        index,
        snapshots,
        family=TRANSFER_FAMILY,
        cue_symbol=TRANSFER_CUE,
        batch_size=args.batch_size,
        seed=args.seed + 2_000_000,
        retention_lifetimes=1,
        feature_bank=behavioral_features,
    )
    chosen, oracle, selected_correctly, scores = _select_victim(
        residual_policy, index, snapshots, probe
    )
    logical_lifetimes += args.batch_size * ACTIVE_CACHE_SLOTS
    if chosen is None:
        raise RuntimeError("inherited transfer policy selected no victim")
    displaced = index.archive.active_binding(chosen)
    displaced_snapshot = (
        snapshots[index.artifact_handle(displaced)] if displaced is not None else None
    )
    _restore_snapshot(system, chosen, snapshots[transfer_digest])
    retention_rows: list[dict[str, float | int]] = []

    def verify(_candidate: EpisodicBindingArtifactIndex) -> bool:
        retention_rows.extend(
            _evaluate(
                system,
                family=TRANSFER_FAMILY,
                slot=chosen,
                cue_symbol=TRANSFER_CUE,
                lifetimes=args.retention_lifetimes,
                batch_size=args.batch_size,
                steps=_family_steps(TRANSFER_FAMILY),
                seed=args.seed + 2_100_000,
            )
        )
        return _stable(retention_rows)

    activation = index.reactivate_verified(transfer_id, chosen, verify)
    if not activation.accepted and displaced_snapshot is not None:
        _restore_snapshot(system, chosen, displaced_snapshot)
    post_activation = (
        _evaluate(
            system,
            family=TRANSFER_FAMILY,
            slot=chosen,
            cue_symbol=TRANSFER_CUE,
            lifetimes=args.retention_lifetimes,
            batch_size=args.batch_size,
            steps=_family_steps(TRANSFER_FAMILY),
            seed=args.seed + 2_200_000,
        )
        if activation.accepted
        else []
    )
    verifier_bits += sum(int(row["unique_verifier_bits"]) for row in retention_rows)
    verifier_bits += sum(int(row["unique_verifier_bits"]) for row in post_activation)
    logical_lifetimes += args.batch_size * (
        len(retention_rows) + len(post_activation)
    )

    inherited_correct = [
        int(row["inherited"]["selection_correct"]) for row in transfer_curve
    ]
    fresh_correct = [
        int(row["fresh"]["selection_correct"]) for row in transfer_curve
    ]
    inherited_early = sum(inherited_correct[:4])
    fresh_early = sum(fresh_correct[:4])
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_id = index.lookup(_route_key(system, SCHEDULE[0][1])).binding_id
    gates = {
        "source_cohort_mastered": len(direct) == len(SCHEDULE)
        and all(_stable(rows) for rows in direct),
        "related_source_installed": related_installed,
        "held_out_transfer_mastered": _stable(transfer_direct),
        "source_calibration_present": bool(source_calibration),
        "inherited_beats_fresh_early": inherited_early > fresh_early,
        "inherited_reaches_stable_window": _stable_window(inherited_correct),
        "fresh_baseline_measured": len(fresh_correct) == args.transfer_updates,
        "inherited_reactivation_accepted": activation.accepted,
        "held_out_retention_mastery": _stable(post_activation),
        "active_cache_capacity_bounded": len(index.active_binding_ids)
        == ACTIVE_CACHE_SLOTS,
        "protected_source_retained": source_id is not None
        and index.archive.is_protected(source_id),
        "active_files_match_snapshots_before": active_before,
        "active_files_match_snapshots_after": _active_digests(
            system, index, snapshots
        ),
        "selected_transfer_victim": selected_correctly == 1,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
        "inherited_safety_gate_no_harm": (
            all(
                bool(row["safety_gate"]["safe"])
                for row in transfer_curve
                if "safety_gate" in row
            )
            if safety_gate
            else True
        ),
    }
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Matched fresh-policy transfer measurement for outcome-trained "
            "external eviction on a held-out n-back-2 family; not unrestricted "
            "memory growth, semantic compression, arbitrary program induction, "
            "or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "source_schedule": [
                {"family": family, "cue": cue} for family, cue in SCHEDULE
            ],
            "held_out_family": TRANSFER_FAMILY,
            "held_out_cue": TRANSFER_CUE,
            "active_cache_slots": ACTIVE_CACHE_SLOTS,
            "policy": "external_capability_eviction_policy_v1",
            "policy_kind": policy_kind,
            "transfer_protocol": "shared_fresh_verifier_outcomes_v1",
            "transfer_policy": "frozen_base_plus_opaque_context_residual_v1",
            "residual_gain": residual_gain,
            "safety_gate": safety_gate,
            "candidate_order_permuted": permute_candidates,
            "probationary_fallback": probationary_fallback,
            "candidate_signature": candidate_signature,
            "utility_gap_gate": UTILITY_GAP_GATE,
            "fresh_baseline": "same architecture and updates, zero inherited state",
            "candidate_signature_config": _candidate_signature_config(
                candidate_signature
            ),
        },
        "direct_source": direct,
        "direct_transfer": transfer_direct,
        "related_source_setup": related_setup,
        "source_calibration": source_calibration,
        "residual": {
            "slot_count": residual_policy.slot_count,
            "active_slots": int(residual_policy.slot_active.sum().item()),
            "frozen_base": True,
            "trusted_slots": int(
                residual_policy.trusted_slot_count
                if isinstance(
                    residual_policy,
                    VerifierGatedCapabilityEvictionPolicyBank,
                )
                else 0
            ),
        },
        "transfer_curve": transfer_curve,
        "reactivation": {
            "transfer_binding_id": transfer_id,
            "selected_slot": chosen,
            "oracle_slot": oracle,
            "selected_correctly": selected_correctly,
            "scores": scores.tolist(),
            "activation": activation.__dict__,
            "retention_probe": retention_rows,
            "post_activation": post_activation,
        },
        "archive": {
            "record_count": index.record_count,
            "active_binding_ids": list(index.active_binding_ids),
            "transfer_artifact_digest": transfer_digest,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": verifier_bits + policy_bits,
            "policy_verifier_bits": policy_bits,
            "optimizer_updates": optimizer_updates,
            "policy_updates": policy_updates,
            "replayed_examples": 0,
            "unique_logical_lifetimes": logical_lifetimes,
            "inherited_early_correct": inherited_early,
            "fresh_early_correct": fresh_early,
            "inherited_transfer_accuracy": sum(inherited_correct)
            / len(inherited_correct),
            "fresh_transfer_accuracy": sum(fresh_correct) / len(fresh_correct),
            "stable_bits_to_threshold": (
                verifier_bits + policy_bits if all(gates.values()) else None
            ),
            "retention_threshold": MASTERY_THRESHOLD,
            "transfer_ratio_against_fresh_learner": inherited_early
            / max(1, fresh_early),
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_external_compute_eviction_transfer"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--file-updates", type=int, default=256)
    parser.add_argument("--policy-calibration-rounds", type=int, default=48)
    parser.add_argument("--policy-updates-per-round", type=int, default=8)
    parser.add_argument("--transfer-updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--policy-learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--policy-kind",
        choices=("pointwise", "set_relative"),
        default="pointwise",
    )
    parser.add_argument("--residual-gain", type=float, default=32.0)
    parser.add_argument("--safety-gate", action="store_true")
    parser.add_argument("--permute-candidates", action="store_true")
    parser.add_argument(
        "--probationary-fallback",
        choices=("base", "neutral"),
        default="base",
    )
    parser.add_argument(
        "--candidate-signature",
        choices=("raw", "behavioral", "behavioral_v2"),
        default="raw",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
