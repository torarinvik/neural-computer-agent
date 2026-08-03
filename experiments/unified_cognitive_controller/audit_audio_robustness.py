"""Stress the saved audio neural-IR frontend with noise, dropout, and delay."""

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


@torch.no_grad()
def _evaluate_condition(
    runtime: AmodalControllerRuntime,
    first: torch.Tensor,
    audio: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    confidence: float = 1.0,
) -> dict[str, float]:
    count = first.shape[0]
    state = runtime.initial_state(count, device=first.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=first.device)
    reward = torch.zeros(count, device=first.device)
    actions = []
    for trial in range(first.shape[1]):
        feedback = torch.full_like(reward, float(trial == 1))
        if audio is None:
            streams = {"stream_a": first[:, trial]}
        else:
            if confidence == 1.0:
                audio_source: torch.Tensor | AmodalEvent = audio[:, trial]
            else:
                audio_source = AmodalEvent(
                    payload=runtime.encoders["stream_b"](audio[:, trial]),
                    confidence=torch.full(
                        (count,), confidence, device=first.device
                    ),
                )
            streams = {"stream_a": first[:, trial], "stream_b": audio_source}
        output, state = runtime.step_streams(
            streams, state, action, reward * feedback, feedback
        )
        action = output.decoded["action"].argmax(dim=-1)
        reward = (action == labels[:, trial]).float()
        actions.append(action)
    predictions = torch.stack(actions, dim=1)[:, 1:]
    targets = labels[:, 1:]
    return {"accuracy": float((predictions == targets).float().mean())}


def _audio_conditions(
    clean: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    rms = clean.square().mean().sqrt()
    conditions: dict[str, torch.Tensor | None] = {"clean": clean}
    for level in (0.10, 0.25, 0.50, 1.00):
        noise = torch.randn(
            clean.shape, generator=generator, device=clean.device
        )
        conditions[f"gaussian_{level:.2f}"] = clean + noise * rms * level
    for fraction in (0.10, 0.25, 0.50):
        mask = torch.rand(
            clean.shape, generator=generator, device=clean.device
        ) < fraction
        conditions[f"sample_dropout_{fraction:.2f}"] = clean.masked_fill(mask, 0.0)
    burst_mask = torch.zeros_like(clean, dtype=torch.bool)
    burst_length = clean.shape[-1] // 4
    starts = torch.randint(
        0,
        clean.shape[-1] - burst_length + 1,
        (clean.shape[0], clean.shape[1]),
        generator=generator,
        device=clean.device,
    )
    for batch_index in range(clean.shape[0]):
        for trial in range(clean.shape[1]):
            burst_mask[
                batch_index,
                trial,
                :, starts[batch_index, trial] : starts[batch_index, trial] + burst_length,
            ] = True
    conditions["burst_dropout_0.25"] = clean.masked_fill(burst_mask, 0.0)
    conditions["one_trial_delay"] = clean.roll(1, dims=1)
    conditions["missing_audio"] = None
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=990001)
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
    torch.manual_seed(args.seed)
    runtime, samples = _load_runtime(
        args.controller, args.input_bus, args.audio, device
    )
    appearances = ("bars", "diamonds", "dot_pairs")
    report_appearances = {}
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
        conditions = _audio_conditions(audio, seed=args.seed + appearance_index)
        rows = {
            name: _evaluate_condition(
                runtime,
                first,
                value,
                batch.correct_actions,
            )
            for name, value in conditions.items()
        }
        rows["low_confidence"] = _evaluate_condition(
            runtime,
            first,
            audio,
            batch.correct_actions,
            confidence=0.05,
        )
        report_appearances[appearance] = rows
    clean = [report_appearances[a]["clean"]["accuracy"] for a in appearances]
    gaussian_50 = [
        report_appearances[a]["gaussian_0.50"]["accuracy"] for a in appearances
    ]
    burst = [
        report_appearances[a]["burst_dropout_0.25"]["accuracy"]
        for a in appearances
    ]
    passed = bool(
        min(clean) >= 0.85
        and min(gaussian_50) >= 0.75
        and min(burst) >= 0.70
    )
    report = {
        "schema": "amodal-audio-robustness-audit-v1",
        "claim": (
            "A saved audio neural-IR frontend retains causal cross-stream use "
            "under waveform corruption."
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
            "conditions": list(next(iter(report_appearances.values())).keys()),
        },
        "appearances": report_appearances,
        "gate_summary": {
            "clean_min": min(clean),
            "gaussian_0.50_min": min(gaussian_50),
            "burst_dropout_0.25_min": min(burst),
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
