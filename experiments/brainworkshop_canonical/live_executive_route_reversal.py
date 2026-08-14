"""Audit nonstationary live routing, retention, and AgentBrain.bank reload.

One visible cue first routes to a verified delay-1 executive. The verifier then
changes the private n-back rule behind that same cue. Lifetime-aggregate route
feedback must demote the old route, discover the already-admitted delay-2
skill, and preserve both immutable programs across a checksummed bank reload.
The controller and decoder remain frozen throughout.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    CognitiveTickRuntime,
    ExternalAgentBrainBank,
    ExternalExecutiveLiveMachine,
    ExternalExecutiveRouterLiveMachine,
    ExternalExecutiveSkillRouter,
    KeypressDecoder,
    build_temporal_equality_executive_artifact,
)

from .environment import BrainWorkshopEventEncoder, NBackVerifier
from .live_executive_route import FirstLearnedEventContext
from .live_session import BrainWorkshopLiveDevice

ROUTE_REVERSAL_SCHEMA = "neural-computer.brainworkshop-live-executive-route-reversal.v1"


def _decoder() -> KeypressDecoder:
    decoder = KeypressDecoder(2, 2)
    with torch.no_grad():
        decoder.network.weight.copy_(torch.eye(2))
        decoder.network.bias.zero_()
    return decoder


def _run_episode(
    machine: ExternalExecutiveRouterLiveMachine | ExternalExecutiveLiveMachine,
    encoder: BrainWorkshopEventEncoder,
    *,
    n_back: int,
    cue_symbol: int,
    seed: int,
    steps: int,
) -> dict[str, object]:
    machine.reset()
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
            raise RuntimeError("live route reversal episode failed to drain")
    finish = getattr(machine, "finish_episode", None)
    route_outcome = None if finish is None else finish()
    resolved = [item for result in results for item in result.resolved_outcomes]
    eligible = [item for item in resolved if bool(item.event.present.item())]
    rewards = [float(item.event.reward.item()) for item in eligible]
    selected_slot = getattr(machine, "selected_slot", None)
    return {
        "n_back": n_back,
        "cue_symbol_private_to_verifier": cue_symbol,
        "selected_slot": selected_slot,
        "episode_outcome": (
            sum(rewards) / len(rewards) if route_outcome is None else route_outcome
        ),
        "eligible_verifier_bits": len(eligible),
        "eligible_accuracy": sum(rewards) / len(rewards),
        "ticks": len(results),
        "route_updates": getattr(machine, "route_updates", 0),
        "latency_p50_seconds": float(
            torch.tensor([result.total_seconds for result in results])
            .quantile(0.5)
            .item()
        ),
        "latency_p99_seconds": float(
            torch.tensor([result.total_seconds for result in results])
            .quantile(0.99)
            .item()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-lifetimes", type=int, default=3)
    parser.add_argument("--reversal-lifetimes", type=int, default=5)
    parser.add_argument("--retention-lifetimes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(
        args.calibration_lifetimes,
        args.reversal_lifetimes,
        args.retention_lifetimes,
        args.steps,
    ) < 1:
        raise ValueError("route reversal budgets must be positive")
    started = perf_counter()
    encoder = BrainWorkshopEventEncoder(symbol_count=6, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder_state = {
        name: value.detach().clone() for name, value in encoder.state_dict().items()
    }
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    for delay in (1, 2):
        bank.admit_executive(
            build_temporal_equality_executive_artifact(event_width=8, delay=delay),
            [1.0],
        )
    context = encoder(torch.tensor([4]))[0].detach()
    router = ExternalExecutiveSkillRouter(
        bank,
        context_width=8,
        mastery_threshold=0.8,
        min_mastery_observations=2,
        reversal_patience=2,
    )
    for _ in range(2):
        bank.observe_executive_route(context, 0, 1.0)
    decoder = _decoder()
    machine = ExternalExecutiveRouterLiveMachine(
        router,
        decoder,
        FirstLearnedEventContext(),
        batch_size=1,
        output_key="keypress",
        sample=False,
        exploration=0.2,
        sample_route=False,
        route_feedback_mode="episode_mean",
    )

    calibration = [
        _run_episode(
            machine,
            encoder,
            n_back=1,
            cue_symbol=4,
            seed=args.seed + index,
            steps=args.steps,
        )
        for index in range(args.calibration_lifetimes)
    ]
    bank_before_reload = bank.digest()
    with tempfile.TemporaryDirectory(prefix="neural-computer-route-") as directory:
        bank_path = Path(directory) / "AgentBrain.bank"
        bank.save_bank(bank_path)
        reloaded = ExternalAgentBrainBank.load_bank(bank_path)
        reload_digest_matches = reloaded.digest() == bank_before_reload

        reloaded_encoder = BrainWorkshopEventEncoder(symbol_count=6, event_width=8)
        reloaded_encoder.load_state_dict(encoder_state, strict=True)
        reloaded_router = ExternalExecutiveSkillRouter(
            reloaded,
            context_width=8,
            mastery_threshold=0.8,
            min_mastery_observations=2,
            reversal_patience=2,
        )
        reloaded_machine = ExternalExecutiveRouterLiveMachine(
            reloaded_router,
            _decoder(),
            FirstLearnedEventContext(),
            batch_size=1,
            output_key="keypress",
            sample=False,
            exploration=0.2,
            sample_route=False,
            route_feedback_mode="episode_mean",
        )
        reversed_lifetimes = [
            _run_episode(
                reloaded_machine,
                reloaded_encoder,
                n_back=2,
                cue_symbol=4,
                seed=args.seed + 100 + index,
                steps=args.steps,
            )
            for index in range(args.reversal_lifetimes)
        ]
        route_after_reversal = list(
            reloaded_router.evidence.preferred_order(
                reloaded_encoder(torch.tensor([4]))[0]
            )
        )

        retention_decoder = _decoder()
        retention_machine = ExternalExecutiveLiveMachine.from_artifact(
            reloaded.artifact("executive_program", 0),
            retention_decoder,
            batch_size=1,
            output_key="keypress",
            sample=False,
        )
        retention = [
            _run_episode(
                retention_machine,
                reloaded_encoder,
                n_back=1,
                cue_symbol=4,
                seed=args.seed + 200 + index,
                steps=args.steps,
            )
            for index in range(args.retention_lifetimes)
        ]

        shuffled = _run_episode(
            reloaded_machine,
            reloaded_encoder,
            n_back=2,
            cue_symbol=5,
            seed=args.seed + 300,
            steps=args.steps,
        )
        bank_after_reversal = reloaded.digest()
        reloaded.save_bank(bank_path)
        restored_again = ExternalAgentBrainBank.load_bank(bank_path)
        second_reload_digest_matches = restored_again.digest() == bank_after_reversal

    reversal_slots = [row["selected_slot"] for row in reversed_lifetimes]
    reversal_scores = [float(row["eligible_accuracy"]) for row in reversed_lifetimes]
    retention_scores = [float(row["eligible_accuracy"]) for row in retention]
    reversal_passed = (
        reversal_slots[-1] == 1
        and reversal_scores[-1] >= 0.8
        and route_after_reversal[0] == 1
    )
    report = {
        "schema": ROUTE_REVERSAL_SCHEMA,
        "status": (
            "promoted_live_nonstationary_route_reversal"
            if reload_digest_matches
            and second_reload_digest_matches
            and reversal_passed
            and min(retention_scores, default=0.0) >= 0.8
            and float(shuffled["eligible_accuracy"]) < 0.8
            else "unpromoted_live_nonstationary_route_reversal"
        ),
        "configuration": {
            "skills": "temporal_equality_delay_1_and_delay_2",
            "calibration": "same_cue_delay_1_route_slot_0",
            "reversal": "same_cue_private_rule_switch_to_delay_2",
            "route_feedback_mode": "episode_mean",
            "mastery_threshold": 0.8,
            "reversal_threshold_default": 0.8,
            "reversal_patience": 2,
            "controller_updates": 0,
            "decoder_updates": 0,
            "executive_program_updates": 0,
            "replayed_examples": 0,
        },
        "calibration": calibration,
        "reversal": reversed_lifetimes,
        "retention_forced_slot_0": retention,
        "cue_shuffled_control": shuffled,
        "preferred_order_after_reversal": route_after_reversal,
        "bank_reload_exact": reload_digest_matches,
        "bank_reload_after_reversal_exact": second_reload_digest_matches,
        "controller_frozen": True,
        "decoder_frozen": True,
        "accounting": {
            "unique_verifier_bits_calibration": sum(
                int(row["eligible_verifier_bits"]) for row in calibration
            ),
            "unique_verifier_bits_reversal": sum(
                int(row["eligible_verifier_bits"]) for row in reversed_lifetimes
            ),
            "unique_verifier_bits_retention": sum(
                int(row["eligible_verifier_bits"]) for row in retention
            ),
            "unique_verifier_bits_cue_shuffled_control": int(
                shuffled["eligible_verifier_bits"]
            ),
            "unique_logical_lifetimes": (
                len(calibration)
                + len(reversed_lifetimes)
                + len(retention)
                + 1
            ),
            "route_updates_calibration": machine.route_updates,
            "route_updates_after_reload": reloaded_machine.route_updates,
            "route_updates_total": (
                machine.route_updates + reloaded_machine.route_updates
            ),
            "wall_seconds": perf_counter() - started,
            "replayed_examples": 0,
        },
        "claim_boundary": "bounded same-cue live route reversal with immutable skill retention and checksummed bank reload; not autonomous program induction or physical desktop deployment",
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
