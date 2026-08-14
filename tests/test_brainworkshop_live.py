from __future__ import annotations

import copy

import pytest
import torch

from experiments.brainworkshop_canonical import (
    BrainWorkshopEventEncoder,
    BrainWorkshopLiveDevice,
    NBackVerifier,
    OnlineTemporalCapabilityMachine,
    run_live_lifetime,
)
from neural_computer import (
    CognitiveTickRuntime,
    ExternalAgentBrainBank,
    ExternalExecutiveLiveCredit,
    ExternalExecutiveLiveMachine,
    KeypressDecoder,
    build_temporal_equality_executive_artifact,
)


def _components(seed: int = 17) -> tuple[
    BrainWorkshopEventEncoder, OnlineTemporalCapabilityMachine
]:
    torch.manual_seed(seed)
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    machine = OnlineTemporalCapabilityMachine(
        8,
        max_history=4,
        intention_width=6,
        hidden=12,
        learning_rate=3e-3,
        sample=False,
    )
    return encoder, machine


def test_live_brainworkshop_updates_before_the_next_action() -> None:
    encoder, machine = _components()
    device = BrainWorkshopLiveDevice(
        NBackVerifier(batch_size=1, n_back=1, steps=4, seed=23),
        encoder,
    )
    runtime = CognitiveTickRuntime(device, machine, {"keypress": device})

    first = runtime.tick(0.0)
    assert first.emitted_receipts[0].model_version == 0
    second = runtime.tick(0.1)
    assert second.outcome_bit_count == 0
    assert second.emitted_receipts[0].model_version == 0
    third = runtime.tick(0.2)

    assert third.outcome_bit_count == 1
    assert machine.optimizer_updates == 1
    assert third.emitted_receipts[0].model_version == 1


def test_live_brainworkshop_consumes_each_experience_once() -> None:
    encoder, machine = _components()
    report = run_live_lifetime(
        machine,
        encoder,
        n_back=2,
        steps=7,
        seed=29,
        tick_seconds=0.01,
        learn=True,
        sample=True,
    )

    assert report.actions.shape == (1, 7)
    assert report.input_events == 7
    assert report.unique_verifier_bits == 5
    assert report.optimizer_updates == 5
    assert report.replayed_examples == 0
    assert report.ticks == 8
    assert report.outcome_present.tolist() == [
        [False, False, True, True, True, True, True]
    ]


def test_accelerated_and_real_clock_ticks_have_identical_causal_semantics() -> None:
    encoder, fast_machine = _components()
    slow_machine = copy.deepcopy(fast_machine)
    slow_encoder = copy.deepcopy(encoder)

    fast = run_live_lifetime(
        fast_machine,
        encoder,
        n_back=2,
        steps=8,
        seed=31,
        tick_seconds=0.001,
        learn=False,
        sample=False,
    )
    slow = run_live_lifetime(
        slow_machine,
        slow_encoder,
        n_back=2,
        steps=8,
        seed=31,
        tick_seconds=1.0,
        learn=False,
        sample=False,
    )

    assert torch.equal(fast.actions, slow.actions)
    assert torch.equal(fast.rewards, slow.rewards)
    assert torch.equal(fast.outcome_present, slow.outcome_present)
    assert fast.unique_verifier_bits == slow.unique_verifier_bits
    assert fast.optimizer_updates == slow.optimizer_updates == 0


@pytest.mark.parametrize("n_back", [1, 2])
def test_admitted_external_skill_runs_live_brainworkshop_with_frozen_controller(
    n_back: int,
    tmp_path,
) -> None:
    event_width = 8
    artifact = build_temporal_equality_executive_artifact(
        event_width=event_width,
        delay=n_back,
    )
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    admission = bank.admit_executive(artifact, [1.0, 1.0, 1.0])
    assert admission.accepted
    bank_path = tmp_path / f"BrainWorkshop{n_back}.bank"
    bank.save_bank(bank_path)
    restored_bank = ExternalAgentBrainBank.load_bank(bank_path)
    persisted_artifact = restored_bank.artifact(
        "executive_program", admission.slot or 0
    )

    decoder = KeypressDecoder(2, 2)
    with torch.no_grad():
        decoder.network.weight.copy_(torch.eye(2))
        decoder.network.bias.zero_()
    machine = ExternalExecutiveLiveMachine.from_artifact(
        persisted_artifact,
        decoder,
        batch_size=1,
        output_key="keypress",
        sample=False,
    )
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=event_width)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    device = BrainWorkshopLiveDevice(
        NBackVerifier(batch_size=1, n_back=n_back, steps=10, seed=71 + n_back),
        encoder,
    )
    runtime = CognitiveTickRuntime(device, machine, {"keypress": device})

    results = []
    now = 0.0
    while not device.done or runtime.pending_receipts:
        results.append(runtime.tick(now))
        now += 0.01
        if len(results) > 10 + n_back + 6:
            raise AssertionError("live external executive failed to drain")

    resolved = [item for result in results for item in result.resolved_outcomes]
    assert len(resolved) == 10
    assert sum(int(item.event.present.item()) for item in resolved) == 10 - n_back
    assert all(float(item.event.reward.item()) == 1.0 for item in resolved if bool(item.event.present.item()))
    assert all(
        isinstance(item.proposal.credit_state, ExternalExecutiveLiveCredit)
        for item in resolved
    )
    assert all(
        item.proposal.credit_state.program_digest == persisted_artifact.digest()
        for item in resolved
    )
    assert machine.executive_ticks == len(results)
    assert machine.executive_ticks > 10
