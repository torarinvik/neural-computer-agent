"""Optional read-only bank curriculum against Neural Workshop's public window."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from neural_computer import ExternalTemporalProgramBank

from .controller_pretraining import load_temporal_controller_artifact
from .physical_bank_transfer_pilot import run_physical_bank_transfer
from .physical_live import (
    PhysicalBrainWorkshopConfig,
    compile_macos_capture_helper,
    compile_macos_keypress_helper,
)

PHYSICAL_BANK_CURRICULUM_SCHEMA = (
    "neural-computer.brainworkshop-physical-bank-curriculum.v1"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_bits_to_threshold(
    rewards: list[float],
    *,
    threshold: float,
    min_stable_observations: int,
) -> int | None:
    if len(rewards) < min_stable_observations:
        return None
    cumulative = []
    total = 0.0
    for index, reward in enumerate(rewards, start=1):
        total += reward
        cumulative.append(total / index)
    final_candidate = len(rewards) - min_stable_observations
    for index in range(final_candidate + 1):
        if min(cumulative[index:]) >= threshold:
            return index + 1
    return None


def summarize_curriculum(
    reports: list[dict[str, Any]],
    *,
    requested_sessions: int,
    threshold: float,
    min_stable_observations: int,
    source_bank_sha256: str,
    final_bank: ExternalTemporalProgramBank,
    wall_seconds: float,
) -> dict[str, Any]:
    rewards = [float(value) for report in reports for value in report["rewards"]]
    latencies_p50 = [
        float(report["tick_seconds_p50"])
        for report in reports
        if report["tick_seconds_p50"] is not None
    ]
    latencies_p99 = [
        float(report["tick_seconds_p99"])
        for report in reports
        if report["tick_seconds_p99"] is not None
    ]
    stable_bits = _stable_bits_to_threshold(
        rewards,
        threshold=threshold,
        min_stable_observations=min_stable_observations,
    )
    completed = len(reports)
    controller_digests = {str(report["controller_digest"]) for report in reports}
    program_digests = {
        str(report["program_artifact_digest"]) for report in reports
    }
    immutable = len(controller_digests) <= 1 and len(program_digests) <= 1
    return {
        "schema": PHYSICAL_BANK_CURRICULUM_SCHEMA,
        "requested_sessions": requested_sessions,
        "completed_sessions": completed,
        "unique_logical_lifetimes": completed,
        "unique_verifier_bits": len(rewards),
        "positive_verifier_bits": sum(value >= 0.5 for value in rewards),
        "accuracy": sum(rewards) / len(rewards) if rewards else None,
        "stable_bits_to_threshold": stable_bits,
        "mastery_threshold": threshold,
        "minimum_stable_observations": min_stable_observations,
        "retention_gate_passed": (
            completed == requested_sessions and stable_bits is not None and immutable
        ),
        "input_events": sum(int(report["input_events"]) for report in reports),
        "emitted_actions": sum(
            int(report["emitted_actions"]) for report in reports
        ),
        "controller_optimizer_updates": 0,
        "program_optimizer_updates": 0,
        "router_observations": sum(
            int(report["router_observations"]) for report in reports
        ),
        "replayed_examples": 0,
        "wall_seconds": wall_seconds,
        "configured_live_seconds": sum(
            float(report["elapsed_seconds"]) for report in reports
        ),
        "tick_seconds_p50_max": max(latencies_p50, default=None),
        "tick_seconds_p99_max": max(latencies_p99, default=None),
        "deadline_misses": sum(
            int(report["deadline_misses"]) for report in reports
        ),
        "controller_digest": next(iter(controller_digests), final_bank.controller_digest),
        "program_artifact_digests": sorted(program_digests),
        "controller_and_program_immutable": immutable,
        "program_count": final_bank.program_count,
        "source_bank_sha256": source_bank_sha256,
        "final_bank_sha256": None,
        "final_bank_digest": final_bank.digest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--seconds-per-session", type=float, default=64.0)
    parser.add_argument("--warmup-events", type=int, default=3)
    parser.add_argument("--tick-hz", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-stable-observations", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
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
    arguments = parser.parse_args()
    if (
        arguments.sessions < 1
        or arguments.seconds_per_session <= 0.0
        or arguments.min_stable_observations < 1
        or not 0.0 <= arguments.threshold <= 1.0
    ):
        raise ValueError("physical bank curriculum settings are invalid")

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    campaign_path = arguments.output_dir / "campaign.json"
    existing_paths = sorted(arguments.output_dir.glob("session-*.json"))
    if (existing_paths or campaign_path.exists()) and not arguments.resume:
        raise FileExistsError("curriculum output exists; pass --resume to continue")
    reports = [json.loads(path.read_text()) for path in existing_paths]
    if len(reports) >= arguments.sessions:
        raise ValueError("curriculum already has the requested number of sessions")

    source_bank_sha256 = _sha256_file(arguments.bank)
    bank = ExternalTemporalProgramBank.load_bank(arguments.bank)
    controller_payload = load_temporal_controller_artifact(
        arguments.controller_artifact
    )
    config = PhysicalBrainWorkshopConfig(
        tick_hz=arguments.tick_hz,
        capture_backend="native",
        capture_helper=compile_macos_capture_helper(arguments.capture_helper),
        keypress_helper=compile_macos_keypress_helper(arguments.keypress_helper),
    )
    started = time.monotonic()
    for session_index in range(len(reports) + 1, arguments.sessions + 1):
        report = run_physical_bank_transfer(
            bank,
            controller_payload,
            config,
            seconds=arguments.seconds_per_session,
            warmup_events=arguments.warmup_events,
            seed=arguments.seed + session_index - 1,
        )
        report["session"] = session_index
        reports.append(report)
        bank.save_bank(arguments.bank)
        session_path = arguments.output_dir / f"session-{session_index:03d}.json"
        session_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        summary = summarize_curriculum(
            reports,
            requested_sessions=arguments.sessions,
            threshold=arguments.threshold,
            min_stable_observations=arguments.min_stable_observations,
            source_bank_sha256=source_bank_sha256,
            final_bank=bank,
            wall_seconds=time.monotonic() - started,
        )
        summary["final_bank_sha256"] = _sha256_file(arguments.bank)
        campaign_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(
            {
                "session": session_index,
                "accuracy": report["accuracy"],
                "cumulative_accuracy": summary["accuracy"],
                "unique_verifier_bits": summary["unique_verifier_bits"],
                "stable_bits_to_threshold": summary["stable_bits_to_threshold"],
            },
            flush=True,
        )

    print(json.loads(campaign_path.read_text()))


if __name__ == "__main__":
    main()
