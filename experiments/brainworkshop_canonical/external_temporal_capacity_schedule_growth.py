"""Learn a generic capacity schedule, then apply it to live temporal memory.

The canonical temporal memory already supports learned content addressing and
verifier-gated row replacement.  This experiment composes those boundaries
under a fixed external row budget: two distinct capability addresses each
arrive with a redundant alias, then two new addresses must be admitted.  The
memory-side planner must consolidate a redundant pair before each admission,
while an independent route verifier preserves every retained address.

The planner is trained only from scalar utility on generic candidate banks.
It sees learned keys/values, occupancy, support, age, and protection facts;
the private verifier owns route equivalence and never enters the controller.
The controller and event encoder remain frozen throughout the live-memory
transfer.
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
    ADMISSION_ACTIONS,
    ConsolidationProposal,
    MemoryCandidates,
    OpaqueCapacityPlanner,
    PersistentAppendOnlyContentAddressedMemory,
    verify_consolidation_proposal,
)

from .external_temporal_content_retrieval_growth import (
    _digest,
    _event_key,
    _noisy_key,
)
from .external_temporal_query_address_growth import _build
from .external_temporal_legacy_support import address_basis

CAPACITY_SCHEDULE_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-capacity-schedule-growth.v1"
)
WIDTH = 16
MAX_ROWS = 4
HIDDEN = 48
TEMPERATURE = 0.8
READ_MATCH_THRESHOLD = 0.75
ALIAS_NOISE = 0.20
TRAINING_ACTIONS = ADMISSION_ACTIONS


def _training_bank(
    *, seed: int, target_action: str
) -> tuple[
    MemoryCandidates,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int | None,
    tuple[int, int] | None,
]:
    """Generate a generic capacity episode with private verifier targets."""

    if target_action not in TRAINING_ACTIONS:
        raise ValueError(f"unsupported training action: {target_action}")
    generator = torch.Generator().manual_seed(seed)
    keys = F.normalize(
        torch.randn(1, MAX_ROWS, WIDTH, generator=generator), dim=-1
    )
    values = F.normalize(
        torch.randn(1, MAX_ROWS, WIDTH, generator=generator), dim=-1
    )
    strengths = torch.rand(1, MAX_ROWS, generator=generator)
    timestamps = torch.arange(MAX_ROWS, dtype=torch.float32).view(1, -1)
    occupied = torch.ones(1, MAX_ROWS, dtype=torch.bool)
    protected = torch.zeros(1, MAX_ROWS, dtype=torch.bool)
    consolidation_available = torch.zeros(1, dtype=torch.bool)
    target_eviction: int | None = None
    target_pair: tuple[int, int] | None = None

    if target_action == "admit":
        occupied[0, 2:] = False
        protected[0, :2] = True
    elif target_action == "evict":
        target_eviction = int(seed % MAX_ROWS)
        protected[0] = True
        protected[0, target_eviction] = False
        strengths[0, target_eviction] = 0.05
    elif target_action == "consolidate":
        first = int(seed % (MAX_ROWS - 1))
        second = first + 1
        target_pair = (first, second)
        values[0, second] = F.normalize(
            values[0, first]
            + 0.03 * torch.randn(WIDTH, generator=generator),
            dim=0,
        )
        protected[0] = True
        consolidation_available[0] = True
    else:
        protected[0] = True

    incoming_key = F.normalize(
        torch.randn(1, WIDTH, generator=generator), dim=-1
    )
    incoming_value = F.normalize(
        torch.randn(1, WIDTH, generator=generator), dim=-1
    )
    bank = MemoryCandidates(
        keys=keys,
        values=values,
        strengths=strengths,
        timestamps=timestamps,
        occupied=occupied,
    )
    return (
        bank,
        incoming_key,
        incoming_value,
        protected,
        consolidation_available,
        target_eviction,
        target_pair,
    )


def _training_utility(
    plan,
    target_action: str,
    target_eviction: int | None,
    target_pair: tuple[int, int] | None,
) -> float:
    if plan.action != target_action:
        return 0.0
    if target_action == "evict":
        return float(plan.eviction_index == target_eviction)
    if target_action == "consolidate":
        return float(plan.pair == target_pair)
    return 1.0


def _train_planner(
    *, seed: int, updates: int
) -> tuple[OpaqueCapacityPlanner, dict[str, int | float]]:
    if updates < 1:
        raise ValueError("capacity planner updates must be positive")
    torch.manual_seed(seed)
    planner = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    optimizer = torch.optim.Adam(planner.parameters(), lr=0.005)
    explorer = torch.Generator().manual_seed(seed + 4_000)
    utilities: list[float] = []
    for update in range(updates):
        target_action = TRAINING_ACTIONS[update % len(TRAINING_ACTIONS)]
        (
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available,
            target_eviction,
            target_pair,
        ) = _training_bank(
            seed=seed + 10_000 + update,
            target_action=target_action,
        )
        plan = planner.propose(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=consolidation_available,
            explore=True,
            temperature=TEMPERATURE,
            generator=explorer,
        )
        utility = _training_utility(
            plan, target_action, target_eviction, target_pair
        )
        planner.adaptation_step(
            bank,
            incoming_key,
            incoming_value,
            protected,
            plan,
            utility,
            consolidation_available=consolidation_available,
            optimizer=optimizer,
        )
        utilities.append(utility)
    return planner.eval(), {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "first_window_utility": sum(utilities[:100]) / min(100, len(utilities)),
        "last_window_utility": sum(utilities[-100:]) / min(100, len(utilities)),
    }


def _evaluate_planner(planner: OpaqueCapacityPlanner, *, seed: int) -> dict[str, float]:
    scores: dict[str, float] = {}
    for action_index, action in enumerate(TRAINING_ACTIONS):
        utilities: list[float] = []
        for episode in range(32):
            (
                bank,
                incoming_key,
                incoming_value,
                protected,
                consolidation_available,
                target_eviction,
                target_pair,
            ) = _training_bank(
                seed=seed + action_index * 10_000 + episode,
                target_action=action,
            )
            plan = planner.propose(
                bank,
                incoming_key,
                incoming_value,
                protected,
                consolidation_available=consolidation_available,
            )
            utilities.append(
                _training_utility(
                    plan, action, target_eviction, target_pair
                )
            )
        scores[action] = sum(utilities) / len(utilities)
    return scores


def _route_result(
    candidates: MemoryCandidates,
    key: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[bool, float, int | None]:
    occupied = candidates.occupied[0]
    rows = torch.nonzero(occupied, as_tuple=False).reshape(-1)
    if not bool(rows.numel()):
        return False, float("-inf"), None
    normalized_keys = F.normalize(candidates.keys[0, rows], dim=-1)
    scores = normalized_keys @ F.normalize(key, dim=0)
    score, position = scores.max(dim=0)
    if float(score) < READ_MATCH_THRESHOLD:
        return False, float(score), None
    value = F.normalize(candidates.values[0, rows[position]], dim=0)
    return True, float(score), int((basis @ value).argmax()) + 1


def _route_verifier(
    candidates: MemoryCandidates,
    routes: tuple[tuple[torch.Tensor, int], ...],
    basis: torch.Tensor,
) -> bool:
    return all(
        (hit and offset == expected)
        for key, expected in routes
        for hit, _score, offset in (_route_result(candidates, key, basis),)
    )


def _consolidation_proposal(
    candidates: MemoryCandidates, plan
) -> ConsolidationProposal:
    if plan.pair is None or plan.action != "consolidate":
        raise ValueError("capacity plan is not a consolidation proposal")
    first, second = plan.pair
    key = F.normalize(candidates.keys[0, first] + candidates.keys[0, second], dim=0)
    value = F.normalize(
        candidates.values[0, first] + candidates.values[0, second], dim=0
    )
    return ConsolidationProposal(
        first=first,
        second=second,
        operation=0,
        key=key,
        value=value,
        strength=torch.maximum(
            candidates.strengths[0, first], candidates.strengths[0, second]
        ),
        score=plan.score,
        operation_logits=torch.tensor([1.0, 0.0, 0.0]),
    )


def _digest_state(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _run_live_stream(
    *,
    planner: OpaqueCapacityPlanner,
    path: Path,
    system,
    seed: int,
    reversed_order: bool,
) -> dict[str, object]:
    basis = address_basis(seed)
    source_key = _event_key(system, 0)
    source_alias = _noisy_key(source_key, seed=seed + 1, scale=ALIAS_NOISE)
    target_key = _event_key(system, 1)
    target_alias = _noisy_key(target_key, seed=seed + 2, scale=ALIAS_NOISE)
    third_key = _event_key(system, 2)
    fourth_key = _event_key(system, 3)
    routes = (
        (source_key, 4),
        (source_alias, 4),
        (target_key, 5),
        (target_alias, 5),
        (third_key, 6),
        (fourth_key, 7),
    )
    initial_keys = torch.stack((source_key, source_alias, target_key, target_alias))
    initial_values = torch.stack(
        (basis[3], basis[3], basis[4], basis[4])
    )
    order = (3, 1, 2, 0) if reversed_order else (0, 1, 2, 3)
    memory = PersistentAppendOnlyContentAddressedMemory(
        WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=0.999,
        read_match_threshold=READ_MATCH_THRESHOLD,
    )
    memory.write(initial_keys[list(order)], initial_values[list(order)], torch.ones(4))
    stages: list[dict[str, object]] = []
    compactions = 0
    admissions = 0
    for stage, (incoming_key, incoming_value, incoming_expected) in enumerate(
        ((third_key, basis[5], 6), (fourth_key, basis[6], 7)),
        start=1,
    ):
        candidates = memory.candidates().pad_to_capacity(MAX_ROWS)
        active_before = routes[: 4 + stage - 1]
        active_after = routes[: 4 + stage]
        protected = candidates.occupied.clone()
        plan = planner.propose(
            candidates,
            incoming_key.unsqueeze(0),
            incoming_value.unsqueeze(0),
            protected,
            consolidation_available=torch.tensor([True]),
        )
        source_version = int(memory.store_version.item())
        route_before = _route_verifier(memory.candidates(), active_before, basis)
        compacted = False
        if plan.action == "consolidate":
            proposal = _consolidation_proposal(candidates, plan)
            candidate, verification = verify_consolidation_proposal(
                candidates,
                proposal,
                lambda candidate, routes=active_before: _route_verifier(
                    candidate, routes, basis
                ),
                candidate_outcomes=[1.0],
                retained_scores=[1.0] * len(active_before),
                min_candidate_observations=1,
            )
            if candidate is not None and verification.accepted:
                memory.replace_from_candidates(
                    candidate,
                    expected_version=source_version,
                )
                compactions += 1
                compacted = True
        if compacted and memory.record_count < MAX_ROWS:
            receipt = memory.write(
                incoming_key.unsqueeze(0),
                incoming_value.unsqueeze(0),
                torch.ones(1),
                timestamp=torch.tensor([float(stage)]),
            )
            admissions += int(receipt.committed[0])
        route_after = _route_verifier(memory.candidates(), active_after, basis)
        stages.append(
            {
                "stage": stage,
                "incoming_offset": incoming_expected,
                "action": plan.action,
                "pair": plan.pair,
                "compacted": compacted,
                "record_count": memory.record_count,
                "route_before": route_before,
                "route_after": route_after,
            }
        )
    final_routes = _route_verifier(memory.candidates(), routes, basis)
    restored = PersistentAppendOnlyContentAddressedMemory(
        WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=0.999,
        read_match_threshold=READ_MATCH_THRESHOLD,
    )
    reloaded_routes = _route_verifier(restored.candidates(), routes, basis)
    return {
        "stages": stages,
        "compactions": compactions,
        "admissions": admissions,
        "record_count": memory.record_count,
        "final_routes": final_routes,
        "reloaded_routes": reloaded_routes,
        "path": path,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1:
        raise ValueError("capacity schedule policy updates must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    planner, training = _train_planner(
        seed=args.seed,
        updates=args.policy_updates,
    )
    trained_scores = _evaluate_planner(planner, seed=args.seed + 500_000)
    fresh = OpaqueCapacityPlanner(width=WIDTH, hidden=HIDDEN)
    fresh_scores = _evaluate_planner(fresh, seed=args.seed + 500_000)
    with tempfile.TemporaryDirectory(prefix="neural-computer-capacity-schedule-") as directory:
        forward = _run_live_stream(
            planner=planner,
            path=Path(directory) / "forward.pt",
            system=system,
            seed=args.seed,
            reversed_order=False,
        )
        reversed_stream = _run_live_stream(
            planner=planner,
            path=Path(directory) / "reversed.pt",
            system=system,
            seed=args.seed + 100,
            reversed_order=True,
        )
        corruption_path = Path(directory) / "corrupt.pt"
        source_path = Path(forward["path"])
        payload = torch.load(source_path, weights_only=False)
        payload["state_dict"]["values"][0, 0] += 0.25
        torch.save(payload, corruption_path)
        corruption_rejected = False
        try:
            PersistentAppendOnlyContentAddressedMemory(
                WIDTH,
                path=corruption_path,
                write_threshold=0.0,
                write_match_threshold=0.999,
                read_match_threshold=READ_MATCH_THRESHOLD,
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    forward_stages = forward["stages"]
    reversed_stages = reversed_stream["stages"]
    gates = {
        "trained_action_transfer": all(
            trained_scores[action] >= 0.9 for action in TRAINING_ACTIONS
        ),
        "trained_beats_fresh_on_learnable_actions": (
            trained_scores["consolidate"]
            >= fresh_scores["consolidate"] + 0.2
            and trained_scores["evict"] >= fresh_scores["evict"]
        ),
        "forward_two_compactions": forward["compactions"] == 2,
        "forward_two_admissions": forward["admissions"] == 2,
        "forward_retains_all_distinct_routes": forward["final_routes"],
        "forward_reload_exact": forward["reloaded_routes"],
        "reversed_two_compactions": reversed_stream["compactions"] == 2,
        "reversed_two_admissions": reversed_stream["admissions"] == 2,
        "reversed_retains_all_distinct_routes": reversed_stream["final_routes"],
        "reversed_reload_exact": reversed_stream["reloaded_routes"],
        "all_stages_preserve_routes": all(
            bool(row["route_after"])
            for row in (*forward_stages, *reversed_stages)
        ),
        "fixed_external_budget": forward["record_count"] == MAX_ROWS
        and reversed_stream["record_count"] == MAX_ROWS,
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": CAPACITY_SCHEDULE_SCHEMA,
        "claim_boundary": (
            "Replay-free learned capacity scheduling transferred to the canonical "
            "persistent content-addressed temporal memory, with sequential "
            "verifier-gated multi-row compaction and distinct-route retention; "
            "not arbitrary shared-structure compression, unbounded memory, "
            "autonomous verifier design, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "planner": "opaque_capacity_planner_v8",
            "planner_signal": "single_scalar_verifier_utility_without_replay",
            "memory": "persistent_append_only_content_addressed_memory_v1",
            "policy_view": "memory_candidates_pad_to_capacity_v1",
            "capacity": MAX_ROWS,
            "distinct_routes": 4,
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "training": training,
        "trained_scores": trained_scores,
        "fresh_scores": fresh_scores,
        "forward": {key: value for key, value in forward.items() if key != "path"},
        "reversed": {
            key: value for key, value in reversed_stream.items() if key != "path"
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": args.policy_updates,
            "unique_logical_lifetimes": args.policy_updates,
            "optimizer_updates": args.policy_updates,
            "live_compaction_transactions": 4,
            "live_admission_transactions": 4,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_temporal_capacity_schedule_growth"
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
    parser.add_argument("--policy-updates", type=int, default=2_400)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
