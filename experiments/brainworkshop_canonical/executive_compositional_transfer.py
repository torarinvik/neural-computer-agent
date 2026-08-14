"""Matched transfer audit for inherited external-executive composition.

A verified source program contributes a generic temporal-equality skeleton.
For a held-out verifier, the warm learner searches only the smallest failed
binding (positive relative displacement).  The matched fresh learner searches
the Cartesian product of the same delay bindings and four generic relation
operators.  Candidate priority is a deterministic hash of the opaque artifact
digest and seed; neither search sees the hidden n-back rule or correct action.

This is a structural program-transfer audit, not controller-weight transfer.
The event encoder, executive operators, and intention decoder are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from neural_computer.agent_brain_bank import ExternalAgentBrainBank
from neural_computer.executive import (
    EXECUTIVE_PROGRAM_SCHEMA,
    ExecutiveInstruction,
    ExternalAmodalExecutive,
    ExternalExecutiveOperator,
    ExternalExecutiveOperatorRegistry,
    ExternalExecutiveProgram,
    TypedWorkspaceValue,
)
from neural_computer.executive_bank import (
    build_temporal_equality_executive_artifact,
)
from neural_computer.executive_memory import ExternalValueDelayOperator
from neural_computer.interface import AmodalEvent, AmodalEventCollection
from neural_computer.program import (
    ExternalProgramAdmissionReceipt,
    ExternalProgramArtifact,
    evaluate_external_program_admission,
)

from .environment import BrainWorkshopEventEncoder, NBackVerifier

EXECUTIVE_COMPOSITION_TRANSFER_SCHEMA = (
    "neural-computer.brainworkshop-executive-composition-transfer.v1"
)
EXECUTIVE_COMPOSITION_SCHEMA = "neural-computer.external-executive-composition.v1"
EXECUTIVE_INTENTION_SCHEMA = "neural-computer.amodal-intention.v1"
RELATION_COUNT = 4


class _EventValue(ExternalExecutiveOperator):
    def __init__(self, width: int) -> None:
        self.width = width
        super().__init__(1, ("events",), "value", interface_version="experiment.event-value.v1")

    def execute(self, arguments: tuple[TypedWorkspaceValue, ...]) -> TypedWorkspaceValue:
        events = arguments[0].payload
        assert isinstance(events, AmodalEventCollection)
        batch = events.payload.shape[0]
        value = torch.zeros(batch, self.width, device=events.payload.device)
        if events.payload.shape[1]:
            value = events.payload[:, 0]
        return TypedWorkspaceValue.from_tensor(
            "value",
            value,
            present=events.present.any(dim=1),
            confidence=arguments[0].confidence,
        )


class _RelationEvidence(ExternalExecutiveOperator):
    def __init__(self, handle: int, relation_index: int) -> None:
        if not 0 <= relation_index < RELATION_COUNT:
            raise ValueError("relation index is outside the generic library")
        self.relation_index = relation_index
        super().__init__(
            handle,
            ("value", "value"),
            "evidence",
            interface_version=f"experiment.binary-relation.{relation_index}.v1",
        )

    def configuration(self) -> dict[str, object]:
        return {**super().configuration(), "opaque_relation_index": self.relation_index}

    def execute(self, arguments: tuple[TypedWorkspaceValue, ...]) -> TypedWorkspaceValue:
        left, right = arguments
        assert isinstance(left.payload, torch.Tensor)
        assert isinstance(right.payload, torch.Tensor)
        equal = torch.isclose(left.payload, right.payload).all(dim=1)
        if self.relation_index == 0:
            positive = equal
        elif self.relation_index == 1:
            positive = ~equal
        elif self.relation_index == 2:
            positive = torch.ones_like(equal)
        else:
            positive = torch.zeros_like(equal)
        score = torch.where(
            positive,
            torch.ones_like(positive, dtype=left.payload.dtype),
            -torch.ones_like(positive, dtype=left.payload.dtype),
        ).unsqueeze(1)
        return TypedWorkspaceValue.from_tensor(
            "evidence",
            score,
            present=left.present & right.present,
            confidence=torch.minimum(left.confidence, right.confidence),
        )


class _EvidenceIntention(ExternalExecutiveOperator):
    def __init__(self) -> None:
        super().__init__(2, ("evidence",), "intention", interface_version="experiment.binary-intention.v1")
        self.weights = torch.tensor([[-1.0], [1.0]])

    def execute(self, arguments: tuple[TypedWorkspaceValue, ...]) -> TypedWorkspaceValue:
        evidence = arguments[0]
        assert isinstance(evidence.payload, torch.Tensor)
        return TypedWorkspaceValue.from_tensor(
            "intention",
            evidence.payload @ self.weights.T.to(evidence.payload),
            present=evidence.present,
            confidence=evidence.confidence,
        )


@dataclass(frozen=True)
class _Candidate:
    relation_index: int
    delay: int
    artifact: ExternalProgramArtifact


@dataclass(frozen=True)
class _SearchResult:
    admitted: _Candidate | None
    receipt: ExternalProgramAdmissionReceipt | None
    unique_verifier_bits: int
    unique_lifetimes: int
    attempted_digests: tuple[str, ...]
    accuracies: tuple[float, ...]
    tick_latencies_seconds: tuple[float, ...]


def _program(delay_handle: int, relation_handle: int) -> ExternalExecutiveProgram:
    return ExternalExecutiveProgram(
        5,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction("call", destination=1, operator_handle=1, arguments=(0,)),
            ExecutiveInstruction("call", destination=2, operator_handle=delay_handle, arguments=(1,)),
            ExecutiveInstruction("call", destination=3, operator_handle=relation_handle, arguments=(1, 2)),
            ExecutiveInstruction("branch", source=3, true_target=5, false_target=5, unknown_target=7),
            ExecutiveInstruction("call", destination=4, operator_handle=2, arguments=(3,)),
            ExecutiveInstruction("emit", source=4, next_target=0),
            ExecutiveInstruction("wait", next_target=0),
            ExecutiveInstruction("halt"),
        ),
    ).validate()


def _candidate(relation_index: int, delay: int) -> _Candidate:
    artifact = ExternalProgramArtifact(
        codes=torch.tensor([[float(relation_index), float(delay)]]),
        interpreter_schema=EXECUTIVE_PROGRAM_SCHEMA,
        execution_schema=EXECUTIVE_COMPOSITION_SCHEMA,
        output_schema=EXECUTIVE_INTENTION_SCHEMA,
    )
    return _Candidate(relation_index, delay, artifact)


def _priority(candidate: _Candidate, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{candidate.artifact.digest()}".encode()).hexdigest()


def _evaluate_candidate(
    candidate: _Candidate,
    *,
    target_n_back: int,
    batch_size: int,
    steps: int,
    seed: int,
    event_width: int,
    encoder_state: dict[str, torch.Tensor],
    reward_destroyed: bool = False,
    action_destroyed: bool = False,
    history_destroyed: bool = False,
    executive_override: ExternalAmodalExecutive | None = None,
) -> tuple[float, int, tuple[float, ...]]:
    delay_handle = 100 + candidate.delay
    relation_handle = 200 + candidate.relation_index
    if executive_override is None:
        operators = ExternalExecutiveOperatorRegistry(
            (
                _EventValue(event_width),
                _EvidenceIntention(),
                ExternalValueDelayOperator(
                    delay_handle, width=event_width, delay=candidate.delay
                ),
                _RelationEvidence(relation_handle, candidate.relation_index),
            )
        )
        executive = ExternalAmodalExecutive(
            _program(delay_handle, relation_handle), operators, intention_width=2
        )
    else:
        executive = executive_override
    encoder = BrainWorkshopEventEncoder(4, event_width)
    encoder.load_state_dict(encoder_state)
    encoder.requires_grad_(False)
    verifier = NBackVerifier(
        batch_size=batch_size,
        n_back=target_n_back,
        steps=steps,
        symbol_count=4,
        seed=seed,
    )
    verifier.reset()
    state = executive.initial_state(batch_size, device="cpu")
    rewards: list[torch.Tensor] = []
    tick_latencies: list[float] = []
    for _ in range(steps):
        if history_destroyed:
            state = executive.initial_state(batch_size, device="cpu")
        with torch.no_grad():
            payload = encoder(verifier.observation())
        events = AmodalEventCollection.from_events(
            (AmodalEvent(payload=payload, confidence=torch.ones(batch_size)),)
        )
        tick_started = perf_counter()
        output, state = executive.tick(events, state)
        tick_latencies.append(perf_counter() - tick_started)
        if output.intention is None:
            action = torch.zeros(batch_size, dtype=torch.long)
        else:
            action = output.intention.payload.argmax(dim=1)
        if action_destroyed:
            generator = torch.Generator().manual_seed(seed + verifier.position * 997)
            action = torch.randint(
                0, 2, action.shape, generator=generator, dtype=torch.long
            )
        scored = verifier.score(action)
        if bool(scored.eligible.any()):
            rewards.append(scored.reward[scored.eligible])
    observed = torch.cat(rewards)
    if reward_destroyed:
        generator = torch.Generator().manual_seed(seed + 91_337)
        observed = torch.randint(
            0, 2, observed.shape, generator=generator, dtype=torch.long
        ).to(torch.float32)
    return float(observed.mean().item()), int(observed.numel()), tuple(tick_latencies)


def _search(
    candidates: tuple[_Candidate, ...],
    *,
    search_seed: int,
    target_n_back: int,
    batch_size: int,
    steps: int,
    event_width: int,
    encoder_state: dict[str, torch.Tensor],
    reward_destroyed: bool = False,
    action_destroyed: bool = False,
    history_destroyed: bool = False,
) -> _SearchResult:
    ordered = tuple(sorted(candidates, key=lambda candidate: _priority(candidate, search_seed)))
    bits = 0
    lifetimes = 0
    attempted: list[str] = []
    accuracies: list[float] = []
    tick_latencies: list[float] = []
    for candidate_index, candidate in enumerate(ordered):
        outcomes: list[float] = []
        accuracy, count, latencies = _evaluate_candidate(
            candidate,
            target_n_back=target_n_back,
            batch_size=batch_size,
            steps=steps,
            seed=search_seed * 10_000 + candidate_index * 2,
            event_width=event_width,
            encoder_state=encoder_state,
            reward_destroyed=reward_destroyed,
            action_destroyed=action_destroyed,
            history_destroyed=history_destroyed,
        )
        bits += count
        tick_latencies.extend(latencies)
        lifetimes += batch_size
        attempted.append(candidate.artifact.digest())
        accuracies.append(accuracy)
        outcomes.append(accuracy)
        if accuracy < 0.9:
            continue
        confirmation, count, latencies = _evaluate_candidate(
            candidate,
            target_n_back=target_n_back,
            batch_size=batch_size,
            steps=steps,
            seed=search_seed * 10_000 + candidate_index * 2 + 1,
            event_width=event_width,
            encoder_state=encoder_state,
            reward_destroyed=reward_destroyed,
            action_destroyed=action_destroyed,
            history_destroyed=history_destroyed,
        )
        bits += count
        tick_latencies.extend(latencies)
        lifetimes += batch_size
        accuracies.append(confirmation)
        outcomes.append(confirmation)
        receipt = evaluate_external_program_admission(
            candidate.artifact,
            outcomes,
            threshold=0.9,
            min_observations=2,
            min_stable_observations=2,
        )
        if receipt.accepted:
            return _SearchResult(
                candidate,
                receipt,
                bits,
                lifetimes,
                tuple(attempted),
                tuple(accuracies),
                tuple(tick_latencies),
            )
    return _SearchResult(
        None,
        None,
        bits,
        lifetimes,
        tuple(attempted),
        tuple(accuracies),
        tuple(tick_latencies),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.seeds, args.batch_size, args.steps, args.event_width) < 1:
        raise ValueError("transfer audit budgets must be positive")
    if args.target_n_back < 1 or args.steps <= args.target_n_back:
        raise ValueError("transfer audit needs target-bearing steps")
    started = perf_counter()
    with torch.random.fork_rng():
        torch.manual_seed(args.seed + 404)
        encoder = BrainWorkshopEventEncoder(4, args.event_width)
    encoder_state = {name: value.detach().clone() for name, value in encoder.state_dict().items()}

    warm_candidates = tuple(_candidate(0, delay) for delay in range(1, 5))
    fresh_candidates = tuple(
        _candidate(relation, delay)
        for relation in range(RELATION_COUNT)
        for delay in range(1, 5)
    )
    irrelevant_candidates = tuple(_candidate(1, delay) for delay in range(1, 5))
    source_artifact = _candidate(0, 1)
    source_outcomes = []
    source_bits = 0
    all_tick_latencies: list[float] = []
    for source_offset in range(2):
        accuracy, bits, latencies = _evaluate_candidate(
            source_artifact,
            target_n_back=1,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 900_000 + source_offset,
            event_width=args.event_width,
            encoder_state=encoder_state,
        )
        source_outcomes.append(accuracy)
        source_bits += bits
        all_tick_latencies.extend(latencies)
    source_receipt = evaluate_external_program_admission(
        source_artifact.artifact,
        source_outcomes,
        threshold=0.9,
        min_observations=2,
        min_stable_observations=2,
    )
    rows: list[dict[str, object]] = []
    admitted_target: _Candidate | None = None
    admitted_target_outcomes: tuple[float, ...] = ()
    for offset in range(args.seeds):
        seed = args.seed + offset
        warm = _search(
            warm_candidates,
            search_seed=seed,
            target_n_back=args.target_n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            event_width=args.event_width,
            encoder_state=encoder_state,
        )
        fresh = _search(
            fresh_candidates,
            search_seed=seed,
            target_n_back=args.target_n_back,
            batch_size=args.batch_size,
            steps=args.steps,
            event_width=args.event_width,
            encoder_state=encoder_state,
        )
        all_tick_latencies.extend(warm.tick_latencies_seconds)
        all_tick_latencies.extend(fresh.tick_latencies_seconds)
        if warm.admitted is not None:
            if admitted_target is None:
                admitted_target = warm.admitted
                admitted_target_outcomes = warm.accuracies[-2:]
            elif warm.admitted.artifact.digest() != admitted_target.artifact.digest():
                raise RuntimeError("warm searches admitted different target artifacts")
        rows.append(
            {
                "seed": seed,
                "warm_bits_to_admission": warm.unique_verifier_bits,
                "fresh_bits_to_admission": fresh.unique_verifier_bits,
                "warm_lifetimes_to_admission": warm.unique_lifetimes,
                "fresh_lifetimes_to_admission": fresh.unique_lifetimes,
                "warm_admitted": warm.admitted is not None,
                "fresh_admitted": fresh.admitted is not None,
                "same_admitted_artifact": (
                    warm.admitted is not None
                    and fresh.admitted is not None
                    and warm.admitted.artifact.digest() == fresh.admitted.artifact.digest()
                ),
            }
        )

    control_seed = args.seed + 100_000
    irrelevant = _search(
        irrelevant_candidates,
        search_seed=control_seed,
        target_n_back=args.target_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        event_width=args.event_width,
        encoder_state=encoder_state,
    )
    shuffled = _search(
        warm_candidates,
        search_seed=control_seed,
        target_n_back=args.target_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        event_width=args.event_width,
        encoder_state=encoder_state,
        reward_destroyed=True,
    )
    action_shuffled = _search(
        warm_candidates,
        search_seed=control_seed,
        target_n_back=args.target_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        event_width=args.event_width,
        encoder_state=encoder_state,
        action_destroyed=True,
    )
    missing_history = _search(
        warm_candidates,
        search_seed=control_seed,
        target_n_back=args.target_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        event_width=args.event_width,
        encoder_state=encoder_state,
        history_destroyed=True,
    )
    for control in (irrelevant, shuffled, action_shuffled, missing_history):
        all_tick_latencies.extend(control.tick_latencies_seconds)
    source_retention = []
    source_retention_bits = 0
    for retention_offset in range(2):
        accuracy, bits, latencies = _evaluate_candidate(
            source_artifact,
            target_n_back=1,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed + 950_000 + retention_offset,
            event_width=args.event_width,
            encoder_state=encoder_state,
        )
        source_retention.append(accuracy)
        source_retention_bits += bits
        all_tick_latencies.extend(latencies)
    if admitted_target is None:
        raise RuntimeError("no target artifact was available for durable admission")
    controller_digest_builder = hashlib.sha256()
    for name, value in sorted(encoder_state.items()):
        tensor = value.detach().cpu().contiguous()
        controller_digest_builder.update(name.encode())
        controller_digest_builder.update(tensor.numpy().tobytes())
    controller_digest = controller_digest_builder.hexdigest()
    durable_bank = ExternalAgentBrainBank(
        controller_digest=controller_digest, capacity=4
    )
    durable_source = build_temporal_equality_executive_artifact(
        event_width=args.event_width, delay=1
    )
    durable_target = build_temporal_equality_executive_artifact(
        event_width=args.event_width, delay=admitted_target.delay
    )
    durable_source_receipt = durable_bank.admit_executive(
        durable_source,
        source_outcomes,
        threshold=0.9,
        min_observations=2,
        min_stable_observations=2,
    )
    durable_target_receipt = durable_bank.admit_executive(
        durable_target,
        list(admitted_target_outcomes),
        threshold=0.9,
        min_observations=2,
        min_stable_observations=2,
    )
    bank_out = getattr(args, "bank_out", None)
    if bank_out is not None:
        durable_bank.save_bank(bank_out)
        reloaded_bank = ExternalAgentBrainBank.load_bank(bank_out)
    else:
        reloaded_bank = ExternalAgentBrainBank.from_payload(
            durable_bank.payload()
        )
    reloaded_source_score, source_reload_bits, source_reload_latencies = _evaluate_candidate(
        source_artifact,
        target_n_back=1,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 970_000,
        event_width=args.event_width,
        encoder_state=encoder_state,
        executive_override=reloaded_bank.executable(
            durable_source_receipt.slot or 0,
            controller_digest=controller_digest,
        ),
    )
    reloaded_target_score, target_reload_bits, target_reload_latencies = _evaluate_candidate(
        admitted_target,
        target_n_back=args.target_n_back,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 980_000,
        event_width=args.event_width,
        encoder_state=encoder_state,
        executive_override=reloaded_bank.executable(
            durable_target_receipt.slot or 0,
            controller_digest=controller_digest,
        ),
    )
    all_tick_latencies.extend(source_reload_latencies)
    all_tick_latencies.extend(target_reload_latencies)
    warm_bits = sum(int(row["warm_bits_to_admission"]) for row in rows)
    fresh_bits = sum(int(row["fresh_bits_to_admission"]) for row in rows)
    gates = {
        "source_artifact_verified": source_receipt.accepted,
        "source_retained_after_target_search": min(source_retention) >= 0.9,
        "durable_source_and_target_admitted": (
            durable_source_receipt.accepted and durable_target_receipt.accepted
        ),
        "bank_reload_digest_exact": reloaded_bank.digest() == durable_bank.digest(),
        "bank_reload_source_mastery": reloaded_source_score >= 0.9,
        "bank_reload_target_mastery": reloaded_target_score >= 0.9,
        "all_warm_admitted": all(bool(row["warm_admitted"]) for row in rows),
        "all_fresh_admitted": all(bool(row["fresh_admitted"]) for row in rows),
        "same_solution": all(bool(row["same_admitted_artifact"]) for row in rows),
        "warm_strictly_faster_every_seed": all(
            int(row["warm_bits_to_admission"]) < int(row["fresh_bits_to_admission"])
            for row in rows
        ),
        "irrelevant_bank_not_admitted": irrelevant.admitted is None,
        "destroyed_reward_not_admitted": shuffled.admitted is None,
        "shuffled_action_not_admitted": action_shuffled.admitted is None,
        "missing_history_not_admitted": missing_history.admitted is None,
        "zero_controller_updates": True,
        "zero_replay": True,
    }
    report = {
        "schema": EXECUTIVE_COMPOSITION_TRANSFER_SCHEMA,
        "claim_boundary": (
            "A verified generic temporal-equality skeleton narrows autonomous held-out "
            "program search to one failed relative-delay binding and reduces unique "
            "verifier bits versus a matched empty-bank Cartesian search."
        ),
        "source_artifact": {
            "relation_fragment": "opaque:0",
            "delay_binding": 1,
            "digest": source_artifact.artifact.digest(),
            "verification_scores": source_outcomes,
            "verification_unique_bits": source_bits,
            "admitted": source_receipt.accepted,
            "retention_scores": source_retention,
            "retention_unique_bits": source_retention_bits,
        },
        "target_rule_private_to_verifier": args.target_n_back,
        "durable_bank": {
            "path": None if bank_out is None else str(bank_out),
            "program_count": reloaded_bank.program_count,
            "source_slot": durable_source_receipt.slot,
            "target_slot": durable_target_receipt.slot,
            "source_artifact_digest": durable_source.digest(),
            "target_artifact_digest": durable_target.digest(),
            "bank_digest": reloaded_bank.digest(),
            "source_reload_score": reloaded_source_score,
            "target_reload_score": reloaded_target_score,
        },
        "rows": rows,
        "aggregate": {
            "warm_unique_verifier_bits": warm_bits,
            "fresh_unique_verifier_bits": fresh_bits,
            "transfer_ratio_fresh_over_warm": fresh_bits / warm_bits,
            "warm_candidate_count": len(warm_candidates),
            "fresh_candidate_count": len(fresh_candidates),
        },
        "controls": {
            "irrelevant_bank_admitted": irrelevant.admitted is not None,
            "destroyed_reward_admitted": shuffled.admitted is not None,
            "shuffled_action_admitted": action_shuffled.admitted is not None,
            "missing_history_admitted": missing_history.admitted is not None,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": {
                "source_verification": source_bits,
                "warm_target_search": warm_bits,
                "fresh_target_search": fresh_bits,
                "source_retention": source_retention_bits,
                "bank_reload": source_reload_bits + target_reload_bits,
                "controls": sum(
                    control.unique_verifier_bits
                    for control in (irrelevant, shuffled, action_shuffled, missing_history)
                ),
            },
            "unique_logical_lifetimes": sum(
                int(row["warm_lifetimes_to_admission"])
                + int(row["fresh_lifetimes_to_admission"])
                for row in rows
            )
            + args.batch_size * 4
            + args.batch_size * 2
            + sum(
                control.unique_lifetimes
                for control in (irrelevant, shuffled, action_shuffled, missing_history)
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "stable_bits_to_threshold": {
                "warm_target_search": warm_bits,
                "fresh_target_search": fresh_bits,
            },
            "tick_latency_ms": {
                "p50": float(
                    torch.tensor(all_tick_latencies, dtype=torch.float64)
                    .quantile(0.50)
                    .item()
                    * 1000.0
                ),
                "p99": float(
                    torch.tensor(all_tick_latencies, dtype=torch.float64)
                    .quantile(0.99)
                    .item()
                    * 1000.0
                ),
            },
            "wall_seconds": perf_counter() - started,
        },
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--bank-out", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--target-n-back", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--event-width", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
