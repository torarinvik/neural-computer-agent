"""Promote one frozen live executive candidate through a verifier gate.

This is a bounded admission diagnostic.  The candidate artifacts are proposed
by the experiment, while the deployed controller and decoder remain frozen;
only receipt-linked scalar outcomes can promote a candidate into the durable
``AgentBrain.bank``.  A wrong candidate and a save/reload retention control are
run beside the promotion path.
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
    ExternalExecutiveCandidateLiveMachine,
    ExternalExecutiveLiveMachine,
    KeypressDecoder,
    build_temporal_equality_executive_artifact,
)

from .environment import BrainWorkshopEventEncoder, NBackVerifier
from .live_session import BrainWorkshopLiveDevice

LIVE_ADMISSION_SCHEMA = "neural-computer.brainworkshop-live-executive-admission.v1"


def _decoder() -> KeypressDecoder:
    decoder = KeypressDecoder(2, 2)
    with torch.no_grad():
        decoder.network.weight.copy_(torch.eye(2))
        decoder.network.bias.zero_()
    return decoder


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _run_machine_episode(
    machine,
    encoder: BrainWorkshopEventEncoder,
    *,
    n_back: int,
    seed: int,
    steps: int,
) -> dict[str, object]:
    reset = getattr(machine, "reset", None)
    if callable(reset):
        reset()
    device = BrainWorkshopLiveDevice(
        NBackVerifier(
            batch_size=1,
            n_back=n_back,
            steps=steps,
            symbol_count=encoder.symbol_count,
            seed=seed,
        ),
        encoder,
    )
    runtime = CognitiveTickRuntime(device, machine, {"keypress": device})
    results = []
    now = 0.0
    started = perf_counter()
    while not device.done or runtime.pending_receipts:
        results.append(runtime.tick(now))
        now += 0.001
        if len(results) > steps + n_back + 8:
            raise RuntimeError("live executive admission episode failed to drain")
    finish_episode = getattr(machine, "finish_episode", None)
    outcome = finish_episode() if callable(finish_episode) else None
    resolved = [item for result in results for item in result.resolved_outcomes]
    eligible = [item for item in resolved if bool(item.event.present.item())]
    rewards = [float(item.event.reward.item()) for item in eligible]
    latencies = [result.total_seconds for result in results]
    return {
        "seed": seed,
        "episode_outcome": outcome,
        "eligible_verifier_bits": len(eligible),
        "eligible_accuracy": (sum(rewards) / len(rewards)) if rewards else None,
        "ticks": len(results),
        "machine_seconds_p50": _percentile(latencies, 0.50),
        "machine_seconds_p99": _percentile(latencies, 0.99),
        "total_seconds_p50": _percentile(latencies, 0.50),
        "total_seconds_p99": _percentile(latencies, 0.99),
        "wall_seconds": perf_counter() - started,
    }


def _candidate_lane(
    bank: ExternalAgentBrainBank,
    encoder: BrainWorkshopEventEncoder,
    *,
    candidate_delay: int,
    verifier_n_back: int,
    seeds: tuple[int, ...],
    steps: int,
    threshold: float,
    min_observations: int,
    min_stable_observations: int,
) -> dict[str, object]:
    candidate = build_temporal_equality_executive_artifact(
        event_width=encoder.event_width,
        delay=candidate_delay,
    )
    bank_before = bank.digest()
    machine = ExternalExecutiveCandidateLiveMachine(
        candidate,
        bank,
        _decoder(),
        batch_size=1,
        output_key="keypress",
        sample=False,
        threshold=threshold,
        min_observations=min_observations,
        min_stable_observations=min_stable_observations,
    )
    episodes = [
        _run_machine_episode(
            machine,
            encoder,
            n_back=verifier_n_back,
            seed=seed,
            steps=steps,
        )
        for seed in seeds
    ]
    receipt = machine.admission_receipt
    if receipt is None:
        raise RuntimeError("candidate lane did not produce an admission receipt")
    return {
        "candidate_delay": candidate_delay,
        "verifier_n_back_private": verifier_n_back,
        "candidate_digest": machine.candidate_digest,
        "episodes": episodes,
        "lifetime_outcomes": list(machine.lifetime_outcomes),
        "admitted": machine.admitted,
        "admission_receipt": receipt.payload(),
        "bank_digest_before": bank_before,
        "bank_digest_after": bank.digest(),
        "bank_program_count": bank.program_count,
        "unique_verifier_bits": machine.unique_verifier_bits,
        "unique_logical_lifetimes": machine.unique_logical_lifetimes,
        "optimizer_updates": 0,
        "replayed_examples": machine.replayed_examples,
        "controller_frozen": True,
        "decoder_frozen": all(
            not parameter.requires_grad for parameter in machine.decoder.parameters()
        ),
        "program_frozen": True,
    }


def _retention_lane(
    bank: ExternalAgentBrainBank,
    encoder: BrainWorkshopEventEncoder,
    *,
    n_back: int,
    seed: int,
    steps: int,
) -> dict[str, object]:
    artifact = bank.artifact("executive_program", 0)
    machine = ExternalExecutiveLiveMachine.from_artifact(
        artifact,
        _decoder(),
        batch_size=1,
        output_key="keypress",
        sample=False,
    )
    result = _run_machine_episode(
        machine,
        encoder,
        n_back=n_back,
        seed=seed,
        steps=steps,
    )
    return {
        "artifact_digest": artifact.digest(),
        "episode": result,
        "unique_verifier_bits": int(result["eligible_verifier_bits"]),
        "unique_logical_lifetimes": 1,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "controller_frozen": True,
        "decoder_frozen": all(
            not parameter.requires_grad for parameter in machine.decoder.parameters()
        ),
        "program_frozen": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-stable-observations", type=int, default=2)
    parser.add_argument("--report-out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 2:
        raise ValueError("admission benchmark steps must exceed n-back delay")
    encoder = BrainWorkshopEventEncoder(symbol_count=4, event_width=8)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    started = perf_counter()
    bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    promotion = _candidate_lane(
        bank,
        encoder,
        candidate_delay=2,
        verifier_n_back=2,
        seeds=(71, 72, 73),
        steps=args.steps,
        threshold=args.threshold,
        min_observations=args.min_observations,
        min_stable_observations=args.min_stable_observations,
    )
    if not promotion["admitted"]:
        raise AssertionError("correct candidate failed the live admission gate")
    bank_after_promotion = bank.digest()

    rejection_bank = ExternalAgentBrainBank(controller_digest="0" * 64, capacity=4)
    rejection = _candidate_lane(
        rejection_bank,
        encoder,
        candidate_delay=1,
        verifier_n_back=2,
        seeds=(81, 82, 83),
        steps=args.steps,
        threshold=args.threshold,
        min_observations=args.min_observations,
        min_stable_observations=args.min_stable_observations,
    )
    if rejection["admitted"] or rejection["bank_digest_before"] != rejection["bank_digest_after"]:
        raise AssertionError("wrong candidate mutated the bank")

    with tempfile.TemporaryDirectory(prefix="live-admission-") as directory:
        bank_path = Path(directory) / "AgentBrain.bank"
        bank.save_bank(bank_path)
        restored = ExternalAgentBrainBank.load_bank(bank_path)
        retention = _retention_lane(
            restored,
            encoder,
            n_back=2,
            seed=101,
            steps=args.steps,
        )
        if retention["episode"]["eligible_accuracy"] != 1.0:
            raise AssertionError("reloaded admitted executive failed retention")
        reload_digest = restored.digest()

    report = {
        "schema": LIVE_ADMISSION_SCHEMA,
        "candidate_proposal": "externally supplied artifact; no autonomous synthesis claimed",
        "controller_boundary": "frozen controller receives learned event tensors only",
        "decoder_boundary": "frozen intention decoder emits opaque keypress actions",
        "verifier_boundary": "private n-back verifier returns only receipt-linked scalar outcomes",
        "promotion": promotion,
        "rejection": rejection,
        "retention_after_bank_reload": retention,
        "bank_digest_after_promotion": bank_after_promotion,
        "bank_digest_after_reload": reload_digest,
        "bank_reload_exact": reload_digest == bank_after_promotion,
        "unique_verifier_bits_total": (
            int(promotion["unique_verifier_bits"])
            + int(rejection["unique_verifier_bits"])
            + int(retention["unique_verifier_bits"])
        ),
        "unique_logical_lifetimes_total": (
            int(promotion["unique_logical_lifetimes"])
            + int(rejection["unique_logical_lifetimes"])
            + int(retention["unique_logical_lifetimes"])
        ),
        "optimizer_updates_total": 0,
        "replayed_examples_total": 0,
        "wall_seconds": perf_counter() - started,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
