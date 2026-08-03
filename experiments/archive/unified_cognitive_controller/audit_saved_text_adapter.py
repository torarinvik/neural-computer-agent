"""Replay a saved discrete text frontend through the frozen runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .legacy_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .train_amodal_text_alignment import (
    PairRelationTextEncoder,
    evaluate_text_alignment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_from_saved_text(
    controller_path: Path,
    bus_path: Path,
    adapter_path: Path,
    device: torch.device,
) -> AmodalControllerRuntime:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    artifact = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "amodal-text-aligned-frontend-v1":
        raise ValueError("unsupported text frontend artifact schema")
    if artifact.get("controller_sha256") != _sha256(controller_path):
        raise ValueError("text frontend was trained against another controller")
    if artifact.get("input_bus_sha256") != _sha256(bus_path):
        raise ValueError("text frontend was trained against another input bus")
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    text_encoder = PairRelationTextEncoder(
        extracted.controller.width,
        grid=int(artifact["text_grid"]),
        levels=int(artifact["text_levels"]),
        embedding_width=int(artifact["embedding_width"]),
    ).to(device)
    text_encoder.load_state_dict(
        {
            name: value.to(device=device)
            for name, value in artifact["state_dict"].items()
        }
    )
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={
            "stream_a": copy.deepcopy(extracted.encoder),
            "stream_b": text_encoder,
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device).eval()
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=997901)
    parser.add_argument("--eval-count", type=int, default=512)
    parser.add_argument(
        "--device",
        default=(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.eval_count < 128:
        raise ValueError("eval-count must provide a meaningful audit")
    device = torch.device(args.device)
    runtime = _runtime_from_saved_text(
        args.controller, args.input_bus, args.adapter, device
    )
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    results = {
        appearance: evaluate_text_alignment(
            runtime, count=args.eval_count, seed=args.seed + index * 10_000,
            device=device, appearance=appearance,
        )
        for index, appearance in enumerate(("bars", "diamonds", "dot_pairs"))
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
            results.values(), (0.90, 0.85, 0.90), (0.80, 0.70, 0.80)
        )
    )
    report = {
        "schema": "amodal-text-frontend-replay-v1",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "adapter": str(args.adapter),
        "adapter_sha256": _sha256(args.adapter),
        "configuration": {
            "seed": args.seed, "eval_count": args.eval_count,
            "device": str(device),
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
