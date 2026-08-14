"""Audit held-out learned-event route generalization after bank reload.

The route ledger is calibrated on two exact cue embeddings. Held-out context
keys are small perturbations of those learned event tensors; they are never
pre-admitted. The live router must use the nearest protected opaque context for
its behavior probabilities, while each observed variant remains an independent
route row rather than overwriting the source evidence.
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
    ExternalExecutiveRouterLiveMachine,
    ExternalExecutiveSkillRouter,
    KeypressDecoder,
    build_temporal_equality_executive_artifact,
)

from .environment import BrainWorkshopEventEncoder, NBackVerifier
from .live_executive_route import (
    FirstLearnedEventContext,
    _run_episode,
)
from .live_session import BrainWorkshopLiveDevice

ROUTE_GENERALIZATION_SCHEMA = (
    "neural-computer.brainworkshop-live-executive-route-generalization.v1"
)


class VariantLearnedEventContext:
    """Apply an opaque representation perturbation to the first learned event."""

    context_width = 8

    def __init__(self, scale: float = 0.0) -> None:
        self.scale = float(scale)

    def context_for(self, event: torch.Tensor) -> torch.Tensor:
        return event + self.scale * torch.roll(event, shifts=1, dims=0)

    def encode(self, events) -> torch.Tensor:
        if events.payload.shape[1] == 0:
            raise ValueError("route context needs a learned event")
        return self.context_for(events.payload[0, 0])


def _decoder() -> KeypressDecoder:
    decoder = KeypressDecoder(2, 2)
    with torch.no_grad():
        decoder.network.weight.copy_(torch.eye(2))
        decoder.network.bias.zero_()
    return decoder


def _run_variant_episode(
    machine: ExternalExecutiveRouterLiveMachine,
    encoder: BrainWorkshopEventEncoder,
    context_encoder: VariantLearnedEventContext,
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
            raise RuntimeError("variant route episode failed to drain")
    route_outcome = machine.finish_episode()
    resolved = [item for result in results for item in result.resolved_outcomes]
    eligible = [item for item in resolved if bool(item.event.present.item())]
    rewards = [float(item.event.reward.item()) for item in eligible]
    return {
        "n_back": n_back,
        "cue_symbol_private_to_verifier": cue_symbol,
        "selected_slot": machine.selected_slot,
        "episode_outcome": route_outcome,
        "eligible_verifier_bits": len(eligible),
        "eligible_accuracy": sum(rewards) / len(rewards),
        "ticks": len(results),
        "route_updates": machine.route_updates,
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
    parser.add_argument("--training-lifetimes", type=int, default=16)
    parser.add_argument("--variant-lifetimes", type=int, default=6)
    parser.add_argument("--variant-start", type=float, default=0.08)
    parser.add_argument("--variant-step", type=float, default=0.02)
    parser.add_argument("--generalization-tolerance", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.training_lifetimes < 2 or args.variant_lifetimes < 1 or args.steps < 3:
        raise ValueError("route generalization budgets are too small")
    if args.variant_step <= 0.0 or args.generalization_tolerance <= 0.0:
        raise ValueError("route generalization scales must be positive")
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
        generalization_tolerance=args.generalization_tolerance,
        min_mastery_observations=2,
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
    training = [
        _run_episode(
            machine,
            encoder,
            n_back=1 if index % 2 == 0 else 2,
            seed=args.seed + index,
            steps=args.steps,
        )
        for index in range(args.training_lifetimes)
    ]
    base_contexts = {
        cue: encoder(torch.tensor([cue]))[0].detach()
        for cue in (4, 5)
    }
    base_context_count = router.evidence.context_count
    bank_before_reload = bank.digest()

    with tempfile.TemporaryDirectory(prefix="neural-computer-route-generalization-") as directory:
        bank_path = Path(directory) / "AgentBrain.bank"
        bank.save_bank(bank_path)
        reloaded = ExternalAgentBrainBank.load_bank(bank_path)
        reload_exact = reloaded.digest() == bank_before_reload
        reloaded_router = ExternalExecutiveSkillRouter(
            reloaded,
            context_width=8,
            generalization_tolerance=args.generalization_tolerance,
            min_mastery_observations=2,
        )
        variant_context = VariantLearnedEventContext()
        variant_machine = ExternalExecutiveRouterLiveMachine(
            reloaded_router,
            _decoder(),
            variant_context,
            batch_size=1,
            output_key="keypress",
            sample=False,
            exploration=0.0,
            sample_route=False,
            route_feedback_mode="episode_mean",
        )
        variants: list[dict[str, object]] = []
        for index in range(args.variant_lifetimes):
            n_back = 1 if index % 2 == 0 else 2
            cue_symbol = 4 if n_back == 1 else 5
            variant_context.scale = args.variant_start + args.variant_step * index
            variant_key = variant_context.context_for(base_contexts[cue_symbol])
            known_before = reloaded_router.evidence.has_context(variant_key)
            row = _run_variant_episode(
                variant_machine,
                encoder,
                variant_context,
                n_back=n_back,
                cue_symbol=cue_symbol,
                seed=args.seed + 1000 + index,
                steps=args.steps,
            )
            variants.append(
                {
                    **row,
                    "variant_scale": variant_context.scale,
                    "exact_variant_context_known_before": known_before,
                    "context_distance_to_base": float(
                        torch.linalg.vector_norm(
                            torch.nn.functional.normalize(variant_key, dim=0)
                            - torch.nn.functional.normalize(
                                base_contexts[cue_symbol], dim=0
                            )
                        ).item()
                    ),
                }
            )
        variant_context_count = reloaded_router.evidence.context_count
        reloaded.save_bank(bank_path)
        restored = ExternalAgentBrainBank.load_bank(bank_path)
        reload_after_variant_exact = restored.digest() == reloaded.digest()

    selected = [int(row["selected_slot"]) for row in variants]
    expected = [0 if int(row["n_back"]) == 1 else 1 for row in variants]
    scores = [float(row["eligible_accuracy"]) for row in variants]
    report = {
        "schema": ROUTE_GENERALIZATION_SCHEMA,
        "status": (
            "promoted_live_held_out_context_generalization"
            if reload_exact
            and reload_after_variant_exact
            and selected == expected
            and min(scores, default=0.0) >= 0.8
            and all(not bool(row["exact_variant_context_known_before"]) for row in variants)
            else "unpromoted_live_held_out_context_generalization"
        ),
        "configuration": {
            "skills": "temporal_equality_delay_1_and_delay_2",
            "training_contexts": 2,
            "held_out_context_variants": args.variant_lifetimes,
            "generalization_tolerance": args.generalization_tolerance,
            "matching_tolerance": 1e-4,
            "route_feedback_mode": "episode_mean",
            "controller_updates": 0,
            "decoder_updates": 0,
            "executive_program_updates": 0,
            "replayed_examples": 0,
        },
        "training": training,
        "held_out_variants": variants,
        "context_count_before_variants": base_context_count,
        "context_count_after_variants": variant_context_count,
        "bank_reload_exact": reload_exact,
        "bank_reload_after_variant_exact": reload_after_variant_exact,
        "preferred_orders": {
            "base_cue_1": list(
                reloaded_router.evidence.preferred_order(base_contexts[4])
            ),
            "base_cue_2": list(
                reloaded_router.evidence.preferred_order(base_contexts[5])
            ),
        },
        "controller_frozen": True,
        "decoder_frozen": True,
        "accounting": {
            "unique_verifier_bits_training": sum(
                int(row["eligible_verifier_bits"]) for row in training
            ),
            "unique_verifier_bits_held_out_variants": sum(
                int(row["eligible_verifier_bits"]) for row in variants
            ),
            "unique_logical_lifetimes": len(training) + len(variants),
            "route_updates": machine.route_updates + variant_machine.route_updates,
            "wall_seconds": perf_counter() - started,
            "replayed_examples": 0,
        },
        "claim_boundary": "bounded held-out learned-event context generalization with exact bank reload; variant rows remain independently stored and no controller or program weights update",
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
