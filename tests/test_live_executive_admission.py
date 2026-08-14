from __future__ import annotations

import torch

from experiments.brainworkshop_canonical import (
    BrainWorkshopEventEncoder,
    BrainWorkshopLiveDevice,
    NBackVerifier,
)
from neural_computer import (
    CognitiveTickRuntime,
    ExecutiveInstruction,
    ExternalAgentBrainBank,
    ExternalExecutiveCandidateLiveMachine,
    ExternalExecutiveLiveCredit,
    ExternalExecutiveOperatorSpec,
    ExternalExecutiveProgram,
    ExternalExecutiveProgramArtifact,
    KeypressDecoder,
    build_temporal_equality_executive_artifact,
)


def _decoder() -> KeypressDecoder:
    decoder = KeypressDecoder(2, 2)
    with torch.no_grad():
        decoder.network.weight.copy_(torch.eye(2))
        decoder.network.bias.zero_()
    return decoder


def _finite_event_prelude(event_width: int = 8) -> ExternalExecutiveProgramArtifact:
    """A finite bank skill that emits one warm-up intention, then hands off."""

    specs = (
        ExternalExecutiveOperatorSpec(1, "singleton_event_value", width=event_width),
        ExternalExecutiveOperatorSpec(2, "value_equality_evidence"),
        ExternalExecutiveOperatorSpec(3, "evidence_binary_intention"),
    )
    program = ExternalExecutiveProgram(
        4,
        (
            ExecutiveInstruction("receive", destination=0),
            ExecutiveInstruction("call", destination=1, operator_handle=1, arguments=(0,)),
            ExecutiveInstruction("call", destination=2, operator_handle=2, arguments=(1, 1)),
            ExecutiveInstruction("call", destination=3, operator_handle=3, arguments=(2,)),
            ExecutiveInstruction("emit", source=3),
            ExecutiveInstruction("halt"),
        ),
    )
    return ExternalExecutiveProgramArtifact(program, specs, 2).validate()


def _run_episode(
    machine: ExternalExecutiveCandidateLiveMachine,
    encoder: BrainWorkshopEventEncoder,
    *,
    n_back: int,
    seed: int,
    steps: int = 10,
    require_perfect: bool = True,
) -> tuple[float, list[float]]:
    machine.reset()
    device = BrainWorkshopLiveDevice(
        NBackVerifier(
            batch_size=1,
            n_back=n_back,
            steps=steps,
            symbol_count=4,
            seed=seed,
        ),
        encoder,
    )
    runtime = CognitiveTickRuntime(device, machine, {"keypress": device})
    results = []
    now = 0.0
    while not device.done or runtime.pending_receipts:
        results.append(runtime.tick(now))
        now += 0.001
        if len(results) > steps + n_back + 8:
            raise AssertionError("candidate admission episode failed to drain")
    outcome = machine.finish_episode()
    assert outcome is not None
    resolved = [item for result in results for item in result.resolved_outcomes]
    eligible = [item for item in resolved if bool(item.event.present.item())]
    assert all(
        isinstance(item.proposal.credit_state, ExternalExecutiveLiveCredit)
        for item in eligible
    )
    rewards = [float(item.event.reward.item()) for item in eligible]
    assert rewards
    if require_perfect:
        assert all(reward == 1.0 for reward in rewards)
    return outcome, rewards


def test_live_candidate_is_admitted_only_after_stable_lifetimes_and_reload(tmp_path) -> None:
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    artifact = build_temporal_equality_executive_artifact(event_width=8, delay=2)
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    before = bank.digest()
    machine = ExternalExecutiveCandidateLiveMachine(
        artifact,
        bank,
        _decoder(),
        batch_size=1,
        output_key="keypress",
        sample=False,
        threshold=0.8,
        min_observations=3,
        min_stable_observations=2,
    )

    outcomes = []
    for seed in (71, 72, 73):
        outcome, _rewards = _run_episode(machine, encoder, n_back=2, seed=seed)
        outcomes.append(outcome)
        if len(outcomes) < 3:
            assert machine.admission_receipt is not None
            assert not machine.admission_receipt.accepted
            assert bank.digest() == before

    assert outcomes == [1.0, 1.0, 1.0]
    assert machine.admitted
    assert machine.admission_receipt is not None
    assert machine.admission_receipt.stable_bits_to_threshold == 1
    assert machine.unique_verifier_bits == 24
    assert machine.unique_logical_lifetimes == 3
    assert machine.replayed_examples == 0
    assert bank.program_count == 1
    assert bank.digest() == machine.bank_digest_after != before

    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    restored = ExternalAgentBrainBank.load_bank(path)
    assert restored.digest() == bank.digest()
    assert restored.artifact("executive_program", 0).digest() == artifact.digest()


def test_live_candidate_rejection_does_not_mutate_bank() -> None:
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    artifact = build_temporal_equality_executive_artifact(event_width=8, delay=1)
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    before = bank.digest()
    machine = ExternalExecutiveCandidateLiveMachine(
        artifact,
        bank,
        _decoder(),
        batch_size=1,
        output_key="keypress",
        sample=False,
        threshold=0.8,
        min_observations=3,
        min_stable_observations=2,
    )

    outcomes = [
        _run_episode(machine, encoder, n_back=2, seed=seed, require_perfect=False)[0]
        for seed in (81, 82, 83)
    ]

    assert tuple(outcomes) == machine.lifetime_outcomes
    assert any(outcome < 0.8 for outcome in outcomes)
    assert machine.admission_receipt is not None
    assert not machine.admission_receipt.accepted
    assert machine.admission_receipt.reason == (
        "candidate did not clear a stable verifier prefix"
    )
    assert bank.program_count == 0
    assert bank.digest() == before
    assert machine.bank_digest_after == before


def test_live_bank_derived_composition_records_parent_provenance(tmp_path) -> None:
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    prelude = _finite_event_prelude()
    temporal = build_temporal_equality_executive_artifact(event_width=8, delay=2)
    assert bank.admit_executive(prelude, [1.0]).accepted
    assert bank.admit_executive(temporal, [1.0]).accepted
    before = bank.digest()
    machine = ExternalExecutiveCandidateLiveMachine.from_parent_slots(
        bank,
        (0, 1),
        _decoder(),
        batch_size=1,
        output_key="keypress",
        sample=False,
        share_compatible_operators=True,
        final_emit_only=True,
        threshold=0.8,
        min_observations=3,
        min_stable_observations=2,
    )
    assert machine.parent_slots == (0, 1)
    assert machine.share_compatible_operators
    assert machine.final_emit_only
    assert machine.candidate_digest == bank.composed_executive_artifact(
        (0, 1), share_compatible_operators=True, final_emit_only=True
    ).digest()
    assert machine.bank_digest_before == before

    outcomes = [
        _run_episode(
            machine,
            encoder,
            n_back=2,
            seed=seed,
            require_perfect=False,
        )[0]
        for seed in (73, 75, 76)
    ]

    assert outcomes == [1.0, 1.0, 1.0]
    assert machine.admitted
    assert machine.admission_receipt is not None
    assert machine.admission_receipt.slot == 2
    assert bank.program_count == 3
    assert len(bank.composition_provenance) == 1
    provenance = bank.composition_provenance[0]
    assert provenance["parent_slots"] == [0, 1]
    assert provenance["child_digest"] == machine.candidate_digest
    assert provenance["admission"] == machine.admission_receipt.payload()
    assert provenance["share_compatible_operators"] is True
    assert provenance["final_emit_only"] is True
    path = tmp_path / "AgentBrain.bank"
    bank.save_bank(path)
    restored = ExternalAgentBrainBank.load_bank(path)
    assert restored.digest() == bank.digest()
    assert restored.composition_provenance == bank.composition_provenance
    assert restored.artifact("executive_program", 2).digest() == machine.candidate_digest
