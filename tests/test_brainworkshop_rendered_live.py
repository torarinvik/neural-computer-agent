from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical import (
    FrozenControllerProgramMachine,
    PretrainedControllerProgramMachine,
    RecursiveTemporalProgramMachine,
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    RenderedBrainWorkshopVerifier,
    SourcePreservingTemporalMachine,
    render_audio,
    render_position,
    run_rendered_live_lifetime,
)
from experiments.brainworkshop_canonical.controller_pretraining import (
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
)
from neural_computer import (
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    AmodalEvent,
    AmodalEventCollection,
    ExternalProgramArtifact,
    compose_recursive_temporal_program,
    one_hot_temporal_address_artifact,
    recursive_temporal_primitive,
)


def _components(
    streams: tuple[str, ...] = ("vision", "audio"),
    *,
    seed: int = 17,
) -> tuple[
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    SourcePreservingTemporalMachine,
]:
    torch.manual_seed(seed)
    config = RenderedBrainWorkshopConfig(n_back=1, steps=8, streams=streams)
    encoders = RenderedBrainWorkshopEncoders(16, source_key_width=4)
    for parameter in encoders.parameters():
        parameter.requires_grad_(False)
    machine = SourcePreservingTemporalMachine(
        16,
        source_key_width=4,
        max_history=4,
        max_sources=2,
        action_count=config.action_count,
        intention_width=16,
        hidden=24,
        sample=False,
    )
    return config, encoders, machine


def test_rendered_observation_exposes_only_raw_device_streams() -> None:
    verifier = RenderedBrainWorkshopVerifier(
        RenderedBrainWorkshopConfig(n_back=2, steps=6),
        seed=17,
    )
    observation = verifier.observation()

    assert [field.name for field in fields(observation)] == ["vision", "audio"]
    assert observation.vision is not None
    assert observation.vision.shape == (3, 36, 36)
    assert observation.audio is not None
    assert observation.audio.shape == (1, 256)
    assert not hasattr(observation, "target")
    assert not hasattr(observation, "n_back")


def test_renderers_are_deterministic_and_symbol_distinct() -> None:
    first_frame = render_position(0, size=36)
    assert torch.equal(first_frame, render_position(0, size=36))
    assert not torch.equal(first_frame, render_position(1, size=36))
    first_audio = render_audio(0, samples=256, sample_rate=8_000)
    assert torch.equal(first_audio, render_audio(0, samples=256, sample_rate=8_000))
    assert not torch.equal(first_audio, render_audio(1, samples=256, sample_rate=8_000))


def test_rendered_encoder_emits_separate_balanced_source_bound_events() -> None:
    config, encoders, _machine = _components()
    verifier = RenderedBrainWorkshopVerifier(config, seed=19)
    with torch.no_grad():
        events = encoders.encode(verifier.observation(), now=1.25)

    assert events.payload.shape == (1, 2, 16)
    assert events.source_key is not None
    assert events.source_key.shape == (1, 2, 4)
    assert not torch.equal(events.source_key[:, 0], events.source_key[:, 1])
    norms = events.payload.norm(dim=-1)
    assert float(norms.max() / norms.min()) < 1.1
    assert events.timestamp is not None
    assert events.timestamp.tolist() == [[1.25, 1.25]]


def test_source_preserving_live_execution_is_event_order_invariant() -> None:
    config, encoders, machine = _components()
    reversed_machine = copy.deepcopy(machine)
    reversed_encoders = copy.deepcopy(encoders)

    ordinary = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=23,
        learn=False,
        sample=False,
    )
    reversed_order = run_rendered_live_lifetime(
        reversed_machine,
        reversed_encoders,
        config,
        seed=23,
        learn=False,
        sample=False,
        reverse_event_order=True,
    )

    assert torch.equal(ordinary.actions, reversed_order.actions)
    assert torch.equal(ordinary.rewards, reversed_order.rewards)
    assert ordinary.input_events == reversed_order.input_events == config.steps * 2


def test_rendered_live_training_updates_once_per_present_scalar_outcome() -> None:
    config, encoders, machine = _components(("vision",))
    report = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=29,
        learn=True,
        sample=True,
    )

    assert report.actions.shape == (1, config.steps)
    assert report.unique_verifier_bits == config.steps - config.n_back
    assert report.optimizer_updates == report.unique_verifier_bits
    assert report.replayed_examples == 0
    assert report.input_events == config.steps
    assert report.ticks == config.steps + 1


def test_external_program_training_freezes_controller_and_updates_only_file() -> None:
    torch.manual_seed(37)
    config = RenderedBrainWorkshopConfig(
        n_back=1, steps=8, streams=("vision",)
    )
    encoders = RenderedBrainWorkshopEncoders(16, source_key_width=4)
    machine = FrozenControllerProgramMachine(
        16,
        source_key_width=4,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=16,
        hidden=24,
        sample=True,
    )
    controller_before = machine.controller_digest()
    program_before = machine.program_digest()

    report = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=37,
        learn=True,
        sample=True,
    )

    assert report.optimizer_updates == 0
    assert report.program_file_updates == report.unique_verifier_bits
    assert machine.controller_digest() == controller_before
    assert machine.program_digest() != program_before


def test_rendered_human_feedback_omits_neutral_true_negative() -> None:
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=3,
        streams=("vision",),
        symbol_count=2,
        neutral_true_negative_absent=True,
    )
    verifier = RenderedBrainWorkshopVerifier(config, seed=17)
    verifier.observation()
    warmup = verifier.score(torch.tensor([0]))
    assert not bool(warmup.eligible.item())

    verifier.observation()
    expected_match = bool(verifier._matches["vision"][0])
    action = torch.tensor([1 if expected_match else 0])
    result = verifier.score(action)
    assert bool(result.eligible.item()) is expected_match
    assert float(result.reward.item()) == float(expected_match)


def test_pretrained_controller_updates_only_temporal_address_file() -> None:
    torch.manual_seed(41)
    source = SourcePreservingTemporalMachine(
        16,
        source_key_width=4,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=16,
        hidden=24,
    )
    controller_state = {
        name: value.detach().clone()
        for name, value in source.named_parameters()
        if name != "relative_address_logits"
    }
    machine = PretrainedControllerProgramMachine(
        16,
        source_key_width=4,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=16,
        hidden=24,
        controller_state=controller_state,
        program_prior=torch.zeros(4),
    )
    encoders = RenderedBrainWorkshopEncoders(16, source_key_width=4)
    controller_before = machine.controller_digest()
    program_before = machine.program_digest()

    report = run_rendered_live_lifetime(
        machine,
        encoders,
        RenderedBrainWorkshopConfig(
            n_back=1, steps=16, streams=("vision",), symbol_count=2
        ),
        seed=41,
    )

    assert report.optimizer_updates == 0
    assert report.program_file_updates == report.unique_verifier_bits
    assert machine.controller_digest() == controller_before
    assert machine.program_digest() != program_before
    assert all(
        not parameter.requires_grad
        for name, parameter in machine.named_parameters()
        if name != "relative_address_logits"
    )
    payload = machine.external_program_payload()
    assert "state" not in payload
    assert set(payload) == {
        "learning_target",
        "controller_digest",
        "program_file_updates",
        "program_digest",
        "relative_address_logits",
        "optimizer_state",
    }


def test_recursive_previous_composition_resolves_two_back_without_new_weights() -> None:
    torch.manual_seed(42)
    source = SourcePreservingTemporalMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
    )
    controller_state = {
        name: value.detach().clone()
        for name, value in source.named_parameters()
        if name != "relative_address_logits"
    }
    machine = RecursiveTemporalProgramMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
        sample=False,
        controller_state=controller_state,
        program_prior=torch.zeros(4),
    )
    legacy = ExternalProgramArtifact(
        codes=torch.tensor([[12.0, -12.0, -12.0, -12.0]]),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )
    machine.load_legacy_primitive_artifact(
        legacy, controller_digest=machine.legacy_controller_digest()
    )
    primitive = recursive_temporal_primitive(legacy)
    composed = compose_recursive_temporal_program(primitive, 2)
    machine.load_recursive_program_artifact(
        composed, controller_digest=machine.controller_digest()
    )
    controller_before = machine.controller_digest()
    program_before = machine.program_digest()

    selected = []
    for index in range(4):
        event = AmodalEvent(
            payload=torch.full((1, 8), float(index)),
            source_key=torch.ones(1, 3),
        )
        collection = AmodalEventCollection.from_events((event,), width=8)
        proposal = machine.tick(collection, (), now=float(index), elapsed=1.0)[0]
        selected.append(int(proposal.credit_state.sources[0].address_index.item()))

    assert selected[2:] == [1, 1]
    assert machine.composition_depth == 2
    assert machine.controller_digest() == controller_before
    assert machine.program_digest() == program_before


def test_recursive_previous_composition_resolves_three_back_without_new_weights() -> None:
    torch.manual_seed(42)
    source = SourcePreservingTemporalMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
    )
    machine = RecursiveTemporalProgramMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
        sample=False,
        controller_state={
            name: value.detach().clone()
            for name, value in source.named_parameters()
            if name != "relative_address_logits"
        },
        program_prior=torch.zeros(4),
    )
    legacy = ExternalProgramArtifact(
        codes=torch.tensor([[12.0, -12.0, -12.0, -12.0]]),
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )
    primitive = recursive_temporal_primitive(legacy)
    composed = compose_recursive_temporal_program(primitive, 3)
    machine.load_recursive_program_artifact(
        composed, controller_digest=machine.controller_digest()
    )
    controller_before = machine.controller_digest()

    selected = []
    for index in range(5):
        event = AmodalEvent(
            payload=torch.full((1, 8), float(index)),
            source_key=torch.ones(1, 3),
        )
        collection = AmodalEventCollection.from_events((event,), width=8)
        proposal = machine.tick(collection, (), now=float(index), elapsed=1.0)[0]
        selected.append(int(proposal.credit_state.sources[0].address_index.item()))

    assert composed.program_length == 3
    assert selected[3:] == [2, 2]
    assert machine.composition_depth == 3
    assert machine.controller_digest() == controller_before


def test_bound_program_ignores_extra_bus_sources() -> None:
    torch.manual_seed(44)
    source = SourcePreservingTemporalMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
        sample=False,
    )
    machine = RecursiveTemporalProgramMachine(
        8,
        source_key_width=3,
        max_history=4,
        max_sources=1,
        action_count=2,
        intention_width=8,
        hidden=8,
        sample=False,
        controller_state={
            name: value.detach().clone()
            for name, value in source.named_parameters()
            if name != "relative_address_logits"
        },
        program_prior=torch.zeros(4),
    )
    play_key = torch.ones(1, 3)
    header_key = torch.tensor([[0.2, -0.4, 0.9]])
    machine.bind_executable_sources((play_key,))
    play_only = []
    both = []
    for index in range(3):
        play = AmodalEvent(
            payload=torch.full((1, 8), float(index)),
            source_key=play_key,
        )
        header = AmodalEvent(
            payload=torch.full((1, 8), 50.0 + index),
            source_key=header_key,
        )
        play_only.append(
            machine.tick(
                AmodalEventCollection.from_events((play,), width=8),
                (),
                now=float(index),
                elapsed=1.0,
            )[0].action.clone()
        )
    machine.reset_history()
    for index in range(3):
        play = AmodalEvent(
            payload=torch.full((1, 8), float(index)),
            source_key=play_key,
        )
        header = AmodalEvent(
            payload=torch.full((1, 8), 50.0 + index),
            source_key=header_key,
        )
        both.append(
            machine.tick(
                AmodalEventCollection.from_events((play, header), width=8),
                (),
                now=float(index),
                elapsed=1.0,
            )[0].action.clone()
        )
    assert all(torch.equal(left, right) for left, right in zip(play_only, both, strict=True))
    assert len(machine._histories) == 1


def test_packed_dual_nback_uses_frozen_binary_decoder() -> None:
    payload = load_temporal_controller_artifact(
        Path("artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt")
    )
    machine = build_recursive_temporal_program_machine(
        payload, sample=False, max_sources=2, pack_source_actions=True
    )
    primitive = recursive_temporal_primitive(
        one_hot_temporal_address_artifact(0, machine.max_history)
    )
    machine.load_recursive_program_artifact(
        primitive, controller_digest=machine.controller_digest()
    )
    encoders = RenderedBrainWorkshopEncoders(
        machine.event_width, source_key_width=machine.source_key_width
    )
    for parameter in encoders.parameters():
        parameter.requires_grad_(False)

    one = run_rendered_live_lifetime(
        machine,
        encoders,
        RenderedBrainWorkshopConfig(n_back=1, steps=24, streams=("vision", "audio")),
        seed=61,
        learn=False,
        sample=False,
    )
    machine.load_recursive_program_artifact(
        compose_recursive_temporal_program(primitive, 2),
        controller_digest=machine.controller_digest(),
    )
    two = run_rendered_live_lifetime(
        machine,
        encoders,
        RenderedBrainWorkshopConfig(n_back=2, steps=24, streams=("vision", "audio")),
        seed=62,
        learn=False,
        sample=False,
    )

    assert machine.action_count == 4
    assert machine.decoder.key_count == 2
    assert one.eligible_accuracy >= 0.8
    assert two.eligible_accuracy >= 0.8
    assert one.optimizer_updates == 0
    assert two.optimizer_updates == 0


def test_negative_feedback_updates_a_saturated_policy_without_clamp_dead_zone() -> None:
    torch.manual_seed(43)
    machine = SourcePreservingTemporalMachine(
        16,
        source_key_width=4,
        max_history=2,
        max_sources=1,
        action_count=2,
        intention_width=16,
        hidden=24,
        sample=False,
    )
    final = machine.decoder.network[-1]
    with torch.no_grad():
        final.weight.zero_()
        final.bias.copy_(torch.tensor([-100.0, 100.0]))
    before = final.bias.detach().clone()
    encoders = RenderedBrainWorkshopEncoders(16, source_key_width=4)

    report = run_rendered_live_lifetime(
        machine,
        encoders,
        RenderedBrainWorkshopConfig(
            n_back=1, steps=6, streams=("vision",), symbol_count=2
        ),
        seed=43,
        sample=False,
    )

    assert 0.0 in report.rewards.flatten().tolist()
    assert bool(torch.isfinite(final.bias).all())
    assert not torch.equal(final.bias, before)


def test_machine_waits_before_emitting_with_actual_later_receipt_tick() -> None:
    config, encoders, machine = _components(("vision",), seed=47)
    machine.action_delay_seconds = 0.2
    observation = RenderedBrainWorkshopVerifier(config, seed=47).observation()
    events = encoders.encode(observation, now=1.0)
    quiet = type(events).empty(1, machine.event_width)

    assert machine.tick(events, (), now=1.0, elapsed=0.0) == ()
    assert machine.tick(quiet, (), now=1.1, elapsed=0.1) == ()
    emitted = machine.tick(quiet, (), now=1.2, elapsed=0.1)

    assert len(emitted) == 1
    assert emitted[0].credit_state is not None
    assert machine.tick(quiet, (), now=1.3, elapsed=0.1) == ()


def test_rendered_live_rejects_dropping_every_input_stream() -> None:
    config, encoders, machine = _components()
    with pytest.raises(ValueError, match="at least one rendered stream"):
        run_rendered_live_lifetime(
            machine,
            encoders,
            config,
            seed=31,
            drop_streams=("vision", "audio"),
        )
