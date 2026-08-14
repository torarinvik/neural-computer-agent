import torch

from experiments.brainworkshop_canonical import (
    BrainWorkshopEventEncoder,
    BrainWorkshopLiveDevice,
    NBackVerifier,
)
from neural_computer import (
    CognitiveTickRuntime,
    ExternalAgentBrainBank,
    ExternalExecutiveRouteCredit,
    ExternalExecutiveRouterLiveMachine,
    ExternalExecutiveSkillRouter,
    KeypressDecoder,
    build_temporal_equality_executive_artifact,
)


class _FirstLearnedEventContext:
    context_width = 8

    def encode(self, events) -> torch.Tensor:
        if events.payload.shape[1] == 0:
            raise ValueError("route context needs a visible learned event")
        return events.payload[0, 0]


def _decoder() -> KeypressDecoder:
    decoder = KeypressDecoder(2, 2)
    with torch.no_grad():
        decoder.network.weight.copy_(torch.eye(2))
        decoder.network.bias.zero_()
    return decoder


def test_live_router_selects_reloaded_skill_and_routes_delayed_outcomes() -> None:
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=8, delay=1),
        [1.0],
    )
    bank.admit_executive(
        build_temporal_equality_executive_artifact(event_width=8, delay=2),
        [1.0],
    )
    encoder = BrainWorkshopEventEncoder(symbol_count=6, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    context_a = encoder(torch.tensor([4]))[0].detach()
    context_b = encoder(torch.tensor([5]))[0].detach()
    router = ExternalExecutiveSkillRouter(
        bank,
        context_width=8,
        min_mastery_observations=2,
    )
    for _ in range(2):
        bank.observe_executive_route(context_a, 0, 1.0)
        bank.observe_executive_route(context_b, 1, 1.0)

    machine = ExternalExecutiveRouterLiveMachine(
        router,
        _decoder(),
        _FirstLearnedEventContext(),
        batch_size=1,
        output_key="keypress",
        sample=False,
    )

    def run_episode(
        n_back: int,
        cue_symbol: int,
        expected_slot: int,
        seed: int,
    ) -> tuple[int, int]:
        machine.reset()
        device = BrainWorkshopLiveDevice(
            NBackVerifier(
                batch_size=1,
                n_back=n_back,
                steps=10,
                seed=seed,
                cue_symbol=cue_symbol,
            ),
            encoder,
        )
        runtime = CognitiveTickRuntime(device, machine, {"keypress": device})
        results = []
        now = 0.0
        while not device.done or runtime.pending_receipts:
            results.append(runtime.tick(now))
            now += 0.01
            if len(results) > 10 + n_back + 6:
                raise AssertionError("live executive router failed to drain")
        resolved = [item for result in results for item in result.resolved_outcomes]
        eligible = [item for item in resolved if bool(item.event.present.item())]
        assert machine.selected_slot == expected_slot
        assert all(float(item.event.reward.item()) == 1.0 for item in eligible)
        assert all(
            isinstance(item.proposal.credit_state, ExternalExecutiveRouteCredit)
            and item.proposal.credit_state.selection.slot == expected_slot
            for item in resolved
        )
        return len(eligible), len(results)

    first_bits, first_ticks = run_episode(1, 4, 0, 801)
    second_bits, second_ticks = run_episode(2, 5, 1, 802)
    assert machine.finish_episode() == 1.0

    assert (first_bits, second_bits) == (9, 8)
    assert first_ticks > 10 and second_ticks > 10
    assert machine.route_updates == 2
    assert router.unique_outcome_bits == 2
