"""Route live Neural Workshop episodes across frozen external executive skills.

The mode cue is rendered into the same learned event tensor that reaches the
executive.  The selector sees only that opaque event, bank slots, and scalar
episode outcomes.  It never receives ``n_back`` or a correct action.  This is a
bounded route-learning diagnostic, not a claim of open-ended program
induction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    CognitiveTickRuntime,
    ExternalAgentBrainBank,
    ExternalExecutiveRouterLiveMachine,
    ExternalExecutiveSkillRouter,
    KeypressDecoder,
    build_temporal_equality_executive_artifact,
)

from .environment import BrainWorkshopEventEncoder, NBackVerifier
from .live_session import BrainWorkshopLiveDevice

ROUTE_LIVE_SCHEMA = "neural-computer.brainworkshop-live-executive-route.v1"


class FirstLearnedEventContext:
    """Use the first amodal event as an opaque, replaceable route context."""

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


def _run_episode(
    machine: ExternalExecutiveRouterLiveMachine,
    encoder: BrainWorkshopEventEncoder,
    *,
    n_back: int,
    seed: int,
    steps: int,
) -> dict[str, object]:
    machine.reset()
    cue_symbol = 3 + n_back
    device = BrainWorkshopLiveDevice(
        NBackVerifier(
            batch_size=1,
            n_back=n_back,
            steps=steps,
            symbol_count=4,
            cue_symbol=cue_symbol,
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
            raise RuntimeError("live executive route episode failed to drain")
    route_outcome = machine.finish_episode()
    resolved = [item for result in results for item in result.resolved_outcomes]
    eligible = [item for item in resolved if bool(item.event.present.item())]
    rewards = [float(item.event.reward.item()) for item in eligible]
    latencies = [result.total_seconds for result in results]
    return {
        "n_back": n_back,
        "cue_symbol_private_to_verifier": cue_symbol,
        "selected_slot": machine.selected_slot,
        "episode_outcome": route_outcome,
        "eligible_verifier_bits": len(eligible),
        "eligible_accuracy": sum(rewards) / len(rewards),
        "ticks": len(results),
        "route_updates": machine.route_updates,
        "latency_p50_seconds": float(torch.tensor(latencies).quantile(0.5).item()),
        "latency_p99_seconds": float(torch.tensor(latencies).quantile(0.99).item()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-per-route", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--min-mastery-observations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes_per_route < 1 or args.steps < 3:
        raise ValueError("route benchmark budgets must be positive")
    if args.min_mastery_observations < 1:
        raise ValueError("route mastery observations must be positive")

    started = perf_counter()
    encoder = BrainWorkshopEventEncoder(symbol_count=6, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    for delay in (1, 2):
        bank.admit_executive(
            build_temporal_equality_executive_artifact(event_width=8, delay=delay),
            [1.0],
        )
    router = ExternalExecutiveSkillRouter(
        bank,
        context_width=8,
        min_mastery_observations=args.min_mastery_observations,
    )
    machine = ExternalExecutiveRouterLiveMachine(
        router,
        _decoder(),
        FirstLearnedEventContext(),
        batch_size=1,
        output_key="keypress",
        sample=False,
        exploration=0.0,
        sample_route=False,
        route_feedback_mode="episode_mean",
    )

    training: list[dict[str, object]] = []
    for index in range(args.episodes_per_route * 2):
        training.append(
            _run_episode(
                machine,
                encoder,
                n_back=1 if index % 2 == 0 else 2,
                seed=args.seed + index,
                steps=args.steps,
            )
        )

    machine.exploration = 0.0
    evaluation = [
        _run_episode(
            machine,
            encoder,
            n_back=1 if index % 2 == 0 else 2,
            seed=args.seed + 1000 + index,
            steps=args.steps,
        )
        for index in range(6)
    ]
    report = {
        "schema": ROUTE_LIVE_SCHEMA,
        "status": (
            "promoted_bounded_live_opaque_route"
            if all(
                row["selected_slot"] == (0 if row["n_back"] == 1 else 1)
                for row in evaluation
            )
            and all(float(row["eligible_accuracy"]) >= 0.8 for row in evaluation)
            else "unpromoted_bounded_live_opaque_route"
        ),
        "configuration": {
            "skills": "temporal_equality_delay_1_and_delay_2",
            "context": "first_visible_mode_cue_through_frozen_event_encoder",
            "route_feedback_mode": "episode_mean",
            "controller_updates": 0,
            "executive_program_updates": 0,
            "replayed_examples": 0,
            "episodes_per_route": args.episodes_per_route,
            "steps": args.steps,
        },
        "training": training,
        "evaluation": evaluation,
        "preferred_orders": {
            "cue_1": list(router.evidence.preferred_order(encoder(torch.tensor([4]))[0])),
            "cue_2": list(router.evidence.preferred_order(encoder(torch.tensor([5]))[0])),
        },
        "accounting": {
            "unique_verifier_bits_training": sum(
                int(row["eligible_verifier_bits"]) for row in training
            ),
            "unique_verifier_bits_evaluation": sum(
                int(row["eligible_verifier_bits"]) for row in evaluation
            ),
            "unique_logical_lifetimes": len(training) + len(evaluation),
            "route_updates": machine.route_updates,
            "wall_seconds": perf_counter() - started,
            "controller_frozen": True,
            "decoder_frozen": True,
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out is None:
        print(text, end="")
    else:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
