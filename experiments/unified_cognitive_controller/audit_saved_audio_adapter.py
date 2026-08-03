"""Replay a saved audio frontend through the frozen amodal controller."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .amodal_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .train_amodal_audio_alignment import (
    PairRelationAudioEncoder,
    evaluate_audio_alignment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_from_saved_audio(
    controller_path: Path,
    bus_path: Path,
    adapter_path: Path,
    device: torch.device,
) -> tuple[AmodalControllerRuntime, int]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    artifact = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "amodal-audio-aligned-frontend-v1":
        raise ValueError("unsupported audio frontend artifact schema")
    if artifact.get("controller_sha256") != _sha256(controller_path):
        raise ValueError("audio frontend was trained against another controller")
    if artifact.get("input_bus_sha256") != _sha256(bus_path):
        raise ValueError("audio frontend was trained against another input bus")
    samples = int(artifact["audio_samples"])
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    audio = PairRelationAudioEncoder(
        extracted.controller.width, samples=samples
    ).to(device)
    audio.load_state_dict(
        {
            name: value.to(device=device)
            for name, value in artifact["state_dict"].items()
        }
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=989001)
    parser.add_argument("--eval-count", type=int, default=2_048)
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
    if args.eval_count < 128:
        raise ValueError("eval-count must provide a meaningful audit")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    runtime, samples = _runtime_from_saved_audio(
        args.controller, args.input_bus, args.adapter, device
    )
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    appearances = ("bars", "diamonds", "dot_pairs")
    results = {
        appearance: evaluate_audio_alignment(
            runtime,
            count=args.eval_count,
            seed=args.seed + index * 10_000,
            device=device,
            appearance=appearance,
            audio_samples=samples,
        )
        for index, appearance in enumerate(appearances)
    }
    controller_unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in controller_before.items()
    )
    passed = controller_unchanged and all(
        row["fused_accuracy"] >= threshold
        and row["stream_a_accuracy"] <= 0.65
        and row["stream_b_accuracy"] <= 0.65
        and row["shuffled_partner_accuracy"] <= 0.60
        and row["contradictory_partner_accuracy"] <= 0.25
        and row["contradictory_prediction_flip_rate"] >= flip_threshold
        and row["full_n1_accuracy"] >= 0.95
        for row, threshold, flip_threshold in zip(
            results.values(), (0.85, 0.80, 0.85), (0.70, 0.65, 0.70)
        )
    )
    report = {
        "schema": "amodal-audio-adapter-replay-v1",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "adapter": str(args.adapter),
        "adapter_sha256": _sha256(args.adapter),
        "configuration": {
            "seed": args.seed,
            "eval_count": args.eval_count,
            "device": str(device),
            "audio_samples": samples,
        },
        "results": results,
        "controller_parameters_unchanged": controller_unchanged,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "results": results}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
