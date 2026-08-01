"""Audit simultaneous independent decoders on one frozen intention stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .amodal_interface import IntentEvent
from .amodal_runtime import (
    AmodalOutputBus,
    OpaqueProtocolDecoder,
    runtime_from_legacy_payload,
)
from .environment import generate_lifetimes
from .train_procedural_shape_span import (
    generate_procedural_shape_batch,
    nuisance_from_level,
)
from .train_protocol_decoder_fanout import (
    PROTOCOL_CODES,
    collect_frozen_intentions,
    collect_frozen_span_intentions,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _score(
    bus: AmodalOutputBus,
    runtime,
    intentions: torch.Tensor,
    correct_actions: torch.Tensor,
) -> dict[str, float | int | bool]:
    event = IntentEvent(intentions)
    simultaneous = bus(event)
    primary_reference = runtime.decode(event)
    correct_commands = PROTOCOL_CODES.to(correct_actions.device)[correct_actions]
    shuffled = bus.decoders["protocol"](
        IntentEvent(intentions.roll(max(1, intentions.shape[0] // 3), 0))
    ).argmax(dim=-1)
    zeroed = bus.decoders["protocol"](IntentEvent(torch.zeros_like(intentions))).argmax(
        dim=-1
    )
    return {
        "examples": int(intentions.shape[0]),
        "primary_exact": torch.equal(simultaneous["primary"], primary_reference),
        "primary_accuracy": float(
            (simultaneous["primary"].argmax(dim=-1) == correct_actions).float().mean()
        ),
        "protocol_accuracy": float(
            (simultaneous["protocol"].argmax(dim=-1) == correct_commands).float().mean()
        ),
        "shuffled_intention_accuracy": float(
            (shuffled == correct_commands).float().mean()
        ),
        "zero_intention_accuracy": float((zeroed == correct_commands).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--protocol-decoder", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=141_001)
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
    if args.count < 64 or args.count % 16:
        raise ValueError("count must be at least 64 and divisible by 16")

    device = torch.device(args.device)
    controller_payload = torch.load(
        args.controller, map_location=device, weights_only=False
    )
    decoder_payload = torch.load(
        args.protocol_decoder, map_location=device, weights_only=False
    )
    runtime = runtime_from_legacy_payload(controller_payload, device=device).eval()
    protocol = OpaqueProtocolDecoder(
        int(decoder_payload["intention_width"]),
        int(decoder_payload["commands"]),
    ).to(device)
    protocol.load_state_dict(decoder_payload["state_dict"])
    protocol.eval()
    bus = AmodalOutputBus({"primary": runtime.decoder, "protocol": protocol}).eval()
    runtime_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.state_dict().items()
    }
    protocol_before = {
        name: value.detach().cpu().clone()
        for name, value in protocol.state_dict().items()
    }

    results = {}
    task_specs = (
        ("binary", "binary_mapping", 1, "bars"),
        ("four_rule", "four_rule", 2, "bars"),
        ("relation_bars", "pair_relation", 1, "bars"),
        ("relation_diamonds", "pair_relation", 1, "diamonds"),
    )
    for offset, (name, task, feedback_trials, appearance) in enumerate(task_specs):
        batch = generate_lifetimes(
            args.count,
            6,
            seed=args.seed + offset * 1_000,
            heldout=True,
            task=task,
            support_trials=feedback_trials,
            appearance=appearance,
            device=device,
        )
        intentions, correct_actions, _ = collect_frozen_intentions(
            runtime, batch, feedback_trials=feedback_trials
        )
        results[name] = _score(bus, runtime, intentions, correct_actions)

    span_batch = generate_procedural_shape_batch(
        args.count,
        span=2,
        vocabulary=2,
        seed=args.seed + 4_000,
        nuisance=nuisance_from_level(0.0),
        heldout=True,
        objective="recognition",
        query_count=2,
        device=device,
    )
    intentions, correct_actions, _ = collect_frozen_span_intentions(runtime, span_batch)
    results["span2"] = _score(bus, runtime, intentions, correct_actions)

    runtime_after = runtime.state_dict()
    protocol_after = protocol.state_dict()
    unchanged = all(
        torch.equal(value, runtime_after[name].detach().cpu())
        for name, value in runtime_before.items()
    ) and all(
        torch.equal(value, protocol_after[name].detach().cpu())
        for name, value in protocol_before.items()
    )
    variable_cardinality = {
        "zero": list(AmodalOutputBus()(IntentEvent(intentions)).keys()),
        "one": list(
            AmodalOutputBus({"protocol": protocol})(IntentEvent(intentions)).keys()
        ),
        "two": list(bus(IntentEvent(intentions)).keys()),
    }
    passed = bool(
        unchanged
        and variable_cardinality
        == {"zero": [], "one": ["protocol"], "two": ["primary", "protocol"]}
        and all(
            row["primary_exact"]
            and row["primary_accuracy"] >= 0.90
            and row["protocol_accuracy"] >= 0.90
            and row["shuffled_intention_accuracy"] <= 0.60
            and row["zero_intention_accuracy"] <= 0.60
            for row in results.values()
        )
    )
    report = {
        "schema": "amodal-output-fanout-audit-v1",
        "claim": (
            "One frozen base intention was consumed simultaneously by the inherited "
            "action decoder and an independently calibrated opaque protocol decoder."
        ),
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "protocol_decoder": str(args.protocol_decoder),
        "protocol_decoder_sha256": _sha256(args.protocol_decoder),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "device": str(device),
        },
        "variable_cardinality": variable_cardinality,
        "all_parameters_unchanged": unchanged,
        "results": results,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "tasks": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
