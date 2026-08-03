"""Audit timestamp buffering for delayed and out-of-order audio events."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .amodal_interface import AmodalEvent
from .amodal_runtime import (
    AmodalControllerRuntime,
    AmodalEventWindowBuffer,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train_amodal_audio_alignment import (
    PairRelationAudioEncoder,
    render_pair_relation_audio,
)
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime(
    controller_path: Path,
    bus_path: Path,
    audio_path: Path,
    device: torch.device,
) -> tuple[AmodalControllerRuntime, int]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    audio_payload = torch.load(audio_path, map_location=device, weights_only=False)
    if audio_payload.get("schema") != "amodal-audio-aligned-frontend-v1":
        raise ValueError("unsupported audio frontend artifact schema")
    if audio_payload.get("controller_sha256") != _sha256(controller_path):
        raise ValueError("audio artifact/controller mismatch")
    if audio_payload.get("input_bus_sha256") != _sha256(bus_path):
        raise ValueError("audio artifact/input-bus mismatch")
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    samples = int(audio_payload["audio_samples"])
    audio = PairRelationAudioEncoder(
        extracted.controller.width, samples=samples
    ).to(device)
    audio.load_state_dict(audio_payload["state_dict"])
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={
            "stream_a": copy.deepcopy(extracted.encoder),
            "stream_b": audio,
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device).eval()
    return runtime, samples


def _event(
    runtime: AmodalControllerRuntime,
    name: str,
    raw: torch.Tensor,
    timestamp: int,
) -> AmodalEvent:
    payload = runtime.encoders[name](raw)
    return AmodalEvent(
        payload=payload,
        timestamp=torch.full(
            (raw.shape[0],), float(timestamp), device=raw.device
        ),
    )


@torch.no_grad()
def _rollout(
    runtime: AmodalControllerRuntime,
    first: torch.Tensor,
    audio: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: str,
) -> dict[str, float | int]:
    count, trials = first.shape[:2]
    buffer = AmodalEventWindowBuffer(("stream_a", "stream_b"))
    state = runtime.initial_state(count, device=first.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=first.device)
    reward = torch.zeros(count, device=first.device)
    actions = torch.full(
        (count, trials), -1, dtype=torch.long, device=first.device
    )
    processed = 0
    max_arrival_delay = 0

    def release(windows) -> None:
        nonlocal state, action, reward, processed, max_arrival_delay
        for window in windows:
            timestamp = int(round(window.timestamp))
            feedback = torch.full_like(reward, float(processed > 0))
            output, state = runtime.step_events(
                window.collection, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == labels[:, timestamp]).float()
            actions[:, timestamp] = action
            processed += 1
            if mode == "delayed":
                max_arrival_delay = max(max_arrival_delay, 1)

    if mode == "synchronous":
        for trial in range(trials):
            release(
                buffer.push(
                    {
                        "stream_a": _event(runtime, "stream_a", first[:, trial], trial),
                        "stream_b": _event(runtime, "stream_b", audio[:, trial], trial),
                    }
                )
            )
    elif mode == "out_of_order":
        for trial in range(trials):
            release(
                buffer.push(
                    {
                        "stream_b": _event(runtime, "stream_b", audio[:, trial], trial),
                        "stream_a": _event(runtime, "stream_a", first[:, trial], trial),
                    }
                )
            )
    elif mode == "delayed":
        release(
            buffer.push({"stream_a": _event(runtime, "stream_a", first[:, 0], 0)})
        )
        for trial in range(1, trials):
            release(
                buffer.push(
                    {
                        "stream_a": _event(
                            runtime, "stream_a", first[:, trial], trial
                        ),
                        "stream_b": _event(
                            runtime, "stream_b", audio[:, trial - 1], trial - 1
                        ),
                    }
                )
            )
        release(
            buffer.push(
                {"stream_b": _event(runtime, "stream_b", audio[:, -1], trials - 1)}
            )
        )
    elif mode == "unbuffered_delayed":
        for trial in range(trials):
            streams = {"stream_a": first[:, trial]}
            if trial > 0:
                streams["stream_b"] = audio[:, trial - 1]
            feedback = torch.full_like(reward, float(trial > 0))
            output, state = runtime.step_streams(
                streams, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == labels[:, trial]).float()
            actions[:, trial] = action
            processed += 1
    else:
        raise ValueError(f"unknown rollout mode {mode}")

    if processed != trials:
        raise AssertionError("buffer did not release every timestamp")
    accuracy = float(
        (actions[:, 1:] == labels[:, 1:]).float().mean()
    )
    return {
        "accuracy": accuracy,
        "processed_windows": processed,
        "pending_windows": len(buffer.pending_timestamps),
        "max_arrival_delay_steps": max_arrival_delay,
    }


@torch.no_grad()
def _rollout_redundant_missing(
    runtime: AmodalControllerRuntime,
    frames: torch.Tensor,
    audio: torch.Tensor,
    labels: torch.Tensor,
    *,
    missing_trial: int,
) -> dict[str, float | int]:
    """Release one missing stream after a generic one-step timeout.

    The full visual stream is intentionally redundant, so this audit measures
    whether the transport policy is graceful and timely rather than asking an
    impossible single-view task to recover complementary evidence.
    """
    count, trials = frames.shape[:2]
    buffer = AmodalEventWindowBuffer(
        ("stream_a", "stream_b"), max_wait=1.0
    )
    state = runtime.initial_state(count, device=frames.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=frames.device)
    reward = torch.zeros(count, device=frames.device)
    actions = torch.full(
        (count, trials), -1, dtype=torch.long, device=frames.device
    )
    processed = 0
    partial_windows = 0
    for trial in range(trials):
        arrivals = {
            "stream_a": _event(runtime, "stream_a", frames[:, trial], trial)
        }
        if trial != missing_trial:
            arrivals["stream_b"] = _event(
                runtime, "stream_b", audio[:, trial], trial
            )
        for window in buffer.push(arrivals):
            timestamp = int(round(window.timestamp))
            feedback = torch.full_like(reward, float(processed > 0))
            output, state = runtime.step_events(
                window.collection, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == labels[:, timestamp]).float()
            actions[:, timestamp] = action
            processed += 1
            partial_windows += int(not window.complete)
    if processed != trials:
        raise AssertionError("timeout buffer did not release every timestamp")
    return {
        "accuracy": float(
            (actions[:, 1:] == labels[:, 1:]).float().mean()
        ),
        "processed_windows": processed,
        "partial_windows": partial_windows,
        "pending_windows": len(buffer.pending_timestamps),
        "timeout_steps": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=991001)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.count < 64 or args.count % 2:
        raise ValueError("count must be even and at least 64")
    device = torch.device(args.device)
    runtime, samples = _load_runtime(
        args.controller, args.input_bus, args.audio, device
    )
    appearances = ("bars", "diamonds", "dot_pairs")
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for appearance_index, appearance in enumerate(appearances):
        batch = generate_lifetimes(
            args.count,
            6,
            seed=args.seed + appearance_index * 10_000,
            heldout=True,
            task="pair_relation",
            appearance=appearance,
            support_trials=1,
            device=device,
        )
        first, second = split_complementary_views(batch.frames)
        audio = render_pair_relation_audio(
            second.flatten(0, 1), samples=samples
        ).reshape(args.count, batch.trials, 1, samples)
        results[appearance] = {
            mode: _rollout(
                runtime,
                first,
                audio,
                batch.correct_actions,
                mode=mode,
            )
            for mode in (
                "synchronous",
                "out_of_order",
                "delayed",
                "unbuffered_delayed",
            )
        }
        full_audio = render_pair_relation_audio(
            batch.frames.flatten(0, 1), samples=samples
        ).reshape(args.count, batch.trials, 1, samples)
        results[appearance]["redundant_missing_timeout"] = (
            _rollout_redundant_missing(
                runtime,
                batch.frames,
                full_audio,
                batch.correct_actions,
                missing_trial=2,
            )
        )
    buffered = [
        results[appearance]["delayed"]["accuracy"] for appearance in appearances
    ]
    out_of_order = [
        results[appearance]["out_of_order"]["accuracy"]
        for appearance in appearances
    ]
    unbuffered = [
        results[appearance]["unbuffered_delayed"]["accuracy"]
        for appearance in appearances
    ]
    missing_redundant = [
        results[appearance]["redundant_missing_timeout"]["accuracy"]
        for appearance in appearances
    ]
    passed = bool(
        min(buffered) >= 0.85
        and min(out_of_order) >= 0.85
        and max(unbuffered) <= 0.65
        and min(missing_redundant) >= 0.85
        and all(
            results[appearance]["delayed"]["pending_windows"] == 0
            and results[appearance]["redundant_missing_timeout"]["partial_windows"]
            == 1
            and results[appearance]["redundant_missing_timeout"]["pending_windows"]
            == 0
            for appearance in appearances
        )
    )
    report = {
        "schema": "amodal-audio-timestamp-buffer-audit-v1",
        "claim": (
            "A generic timestamp window buffer restores causal composition "
            "under one-step audio delay without changing controller weights."
        ),
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "audio": str(args.audio),
        "audio_sha256": _sha256(args.audio),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "device": str(device),
            "audio_samples": samples,
            "buffer_stream_handles": ["stream_a", "stream_b"],
            "delay_steps": 1,
        },
        "appearances": results,
        "gate_summary": {
            "buffered_delayed_min": min(buffered),
            "out_of_order_min": min(out_of_order),
            "unbuffered_delayed_max": max(unbuffered),
            "redundant_missing_timeout_min": min(missing_redundant),
        },
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["gate_summary"], sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
