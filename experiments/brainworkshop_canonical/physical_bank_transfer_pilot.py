"""Retrieve and execute an admitted temporal program in a fresh GUI session."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from neural_computer import (
    ExternalTemporalProgramBank,
    LiveInputInstruction,
    TemporalProgramOutcomeObserver,
)

from .controller_pretraining import (
    build_pretrained_controller_program_machine,
    load_temporal_controller_artifact,
)
from .physical_live import (
    PhysicalBrainWorkshopConfig,
    build_physical_brainworkshop_runtime,
    compile_macos_capture_helper,
    compile_macos_keypress_helper,
)
from .physical_program_bank import learned_event_context


class _SensoryWarmupMachine:
    """Consume learned events without emitting actions before bank retrieval."""

    def tick(self, events, outcomes, *, now: float, elapsed: float):
        del events, outcomes, now, elapsed
        return ()


def run_physical_bank_transfer(
    bank: ExternalTemporalProgramBank,
    controller_payload: dict[str, object],
    config: PhysicalBrainWorkshopConfig,
    *,
    seconds: float,
    warmup_events: int = 3,
    seed: int = 17,
) -> dict[str, object]:
    """Warm up from public events, retrieve once, then execute read-only."""

    if seconds <= 0.0 or warmup_events < 1:
        raise ValueError("physical bank transfer settings are invalid")
    machine = build_pretrained_controller_program_machine(
        controller_payload,
        learning_rate=0.3,
        sample=False,
    )
    if not machine.accepts_controller_digest(bank.controller_digest):
        raise ValueError("temporal program bank targets another controller")
    machine.learning_enabled = False
    route_observer = TemporalProgramOutcomeObserver(bank)
    controller_before = machine.controller_digest()
    program_updates_before = machine.program_file_updates
    runtime, window, capture = build_physical_brainworkshop_runtime(
        machine, config, seed=seed
    )
    physical_input = runtime.input_device
    runtime.input_device = LiveInputInstruction({"public-interface": physical_input})
    runtime.outcome_observers = (route_observer,)
    runtime.machine = _SensoryWarmupMachine()
    selection = None
    route_known_before = False
    results = []
    selected_at_event: int | None = None
    bank_before = bank.digest()
    try:
        # Establish a ready-screen baseline before the ordinary public start.
        results.append(runtime.tick(time.monotonic()))
        window.press((" ",))
        started = time.monotonic()
        period = 1.0 / config.tick_hz
        next_tick = started
        while True:
            now = time.monotonic()
            if now - started >= seconds:
                break
            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.monotonic()
            result = runtime.tick(now)
            results.append(result)
            frontend = physical_input.frontend
            payloads = getattr(frontend, "emitted_payloads", ())
            if selection is None and len(payloads) >= warmup_events:
                context = learned_event_context(
                    payloads[:warmup_events], width=machine.event_width
                )
                route_known_before = bank.router.has_context(context)
                selection = bank.select(context, sample=False)
                machine.load_admitted_program_artifact(
                    selection.artifact,
                    controller_digest=bank.controller_digest,
                )
                route_observer.bind(selection)
                runtime.machine = machine
                selected_at_event = len(payloads)
            next_tick += period
            next_tick = max(next_tick, now)
    finally:
        close_capture = getattr(capture, "close", None)
        if close_capture is not None:
            close_capture()
        close_output = getattr(physical_input.output, "close", None)
        if close_output is not None:
            close_output()
    if selection is None:
        raise RuntimeError("physical warm-up ended before program retrieval")
    rewards = [
        float(resolved.event.reward.item())
        for result in results
        for resolved in result.resolved_outcomes
        if bool(resolved.event.present.item())
    ]
    if route_observer.unique_outcome_bits != len(rewards):
        raise RuntimeError("physical reward INPUT accounting does not match the route")
    machine.assert_controller_frozen()
    if machine.controller_digest() != controller_before:
        raise RuntimeError("physical bank transfer changed the controller")
    if machine.program_file_updates != program_updates_before:
        raise RuntimeError("physical bank transfer changed the admitted program")
    artifact_after = machine.admitted_program_artifact().digest()
    if artifact_after != selection.artifact.digest():
        raise RuntimeError("physical bank transfer changed executable instructions")
    route_known_after = bank.router.has_context(selection.context)
    elapsed_seconds = time.monotonic() - started
    tick_seconds = sorted(result.total_seconds for result in results)

    def percentile(fraction: float) -> float | None:
        if not tick_seconds:
            return None
        index = min(len(tick_seconds) - 1, int(fraction * len(tick_seconds)))
        return tick_seconds[index]

    return {
        "schema": "neural-computer.brainworkshop-physical-bank-transfer.v1",
        "warmup_events": warmup_events,
        "selected_at_event": selected_at_event,
        "selection_slot": selection.slot,
        "selection_propensity": selection.propensity,
        "route_known_before": route_known_before,
        "route_known_after": route_known_after,
        "input_events": sum(result.input_event_count for result in results),
        "emitted_actions": sum(len(result.emitted_receipts) for result in results),
        "unique_public_outcomes": len(rewards),
        "positive_public_outcomes": sum(reward >= 0.5 for reward in rewards),
        "rewards": rewards,
        "accuracy": sum(rewards) / len(rewards) if rewards else None,
        "controller_optimizer_updates": 0,
        "program_optimizer_updates": 0,
        "router_observations": route_observer.unique_outcome_bits,
        "replayed_examples": 0,
        "elapsed_seconds": elapsed_seconds,
        "tick_seconds_p50": percentile(0.50),
        "tick_seconds_p99": percentile(0.99),
        "controller_digest": controller_before,
        "program_artifact_digest": artifact_after,
        "bank_digest_before": bank_before,
        "bank_digest_after": bank.digest(),
        "deadline_misses": sum(result.deadline_missed for result in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--seconds", type=float, default=64.0)
    parser.add_argument("--warmup-events", type=int, default=3)
    parser.add_argument("--tick-hz", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--capture-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-capture"),
    )
    parser.add_argument(
        "--keypress-helper",
        type=Path,
        default=Path("/tmp/neural-computer-macos-keypress"),
    )
    parser.add_argument("--report-out", type=Path)
    arguments = parser.parse_args()

    bank = ExternalTemporalProgramBank.load_bank(arguments.bank)
    config = PhysicalBrainWorkshopConfig(
        tick_hz=arguments.tick_hz,
        capture_backend="native",
        capture_helper=compile_macos_capture_helper(arguments.capture_helper),
        keypress_helper=compile_macos_keypress_helper(arguments.keypress_helper),
    )
    report = run_physical_bank_transfer(
        bank,
        load_temporal_controller_artifact(arguments.controller_artifact),
        config,
        seconds=arguments.seconds,
        warmup_events=arguments.warmup_events,
        seed=arguments.seed,
    )
    bank.save_bank(arguments.bank)
    if arguments.report_out is not None:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    print(report)


if __name__ == "__main__":
    main()
