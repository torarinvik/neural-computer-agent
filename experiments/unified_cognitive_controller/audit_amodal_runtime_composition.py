"""Audit complementary N=2 composition through the complete N-to-M runtime."""

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
from .environment import NULL_ACTION, generate_lifetimes
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
    device: torch.device,
) -> tuple[object, AmodalControllerRuntime]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    legacy = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    bus.eval()
    # The two frontends are independently registered copies of the external
    # encoder. They share no controller branch and receive no stream-name
    # feature; the names only select which adapter lowers raw input.
    runtime = AmodalControllerRuntime(
        legacy.controller,
        encoders={
            "stream_a": copy.deepcopy(legacy.encoder),
            "stream_b": copy.deepcopy(legacy.encoder),
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": legacy.decoder}),
    ).to(device).eval()
    return legacy, runtime


@torch.no_grad()
def _evaluate(
    legacy,
    runtime: AmodalControllerRuntime,
    *,
    count: int,
    seed: int,
    device: torch.device,
    appearance: str,
) -> dict[str, float | bool]:
    normal = generate_lifetimes(
        count,
        6,
        seed=seed,
        heldout=True,
        task="pair_relation",
        appearance=appearance,
        support_trials=1,
        device=device,
    )
    contradictory = generate_lifetimes(
        count,
        6,
        seed=seed,
        heldout=True,
        task="pair_relation",
        appearance=appearance,
        support_trials=1,
        reverse_contexts=True,
        device=device,
    )
    first, second = split_complementary_views(normal.frames)
    _, contradictory_second = split_complementary_views(contradictory.frames)

    def run(
        streams: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, bool, float]:
        legacy_state = legacy.initial_state(count, device=device)
        runtime_state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        logits = []
        exact = True
        maximum_difference = 0.0
        for trial in range(normal.trials):
            has_feedback = torch.full_like(reward, float(trial == 1))
            legacy_events = [legacy.encode(stream[:, trial]) for stream in streams]
            legacy_core, legacy_state = legacy.step_intention_event(
                runtime.input_bus(legacy_events),
                legacy_state,
                action,
                reward * has_feedback,
                has_feedback,
            )
            legacy_logits = legacy.decode(legacy_core.intent_event)
            named_streams = {
                name: stream[:, trial]
                for name, stream in zip(("stream_a", "stream_b"), streams)
            }
            runtime_output, runtime_state = runtime.step_streams(
                named_streams,
                runtime_state,
                action,
                reward * has_feedback,
                has_feedback,
            )
            difference = float(
                (legacy_logits - runtime_output.decoded["action"])
                .abs()
                .max()
            )
            maximum_difference = max(maximum_difference, difference)
            exact = exact and torch.equal(legacy_logits, runtime_output.decoded["action"])
            current_logits = runtime_output.decoded["action"]
            action = current_logits.argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
            logits.append(current_logits)
        return torch.stack(actions, dim=1), torch.stack(logits, dim=1), exact, maximum_difference

    first_actions, _, first_exact, first_delta = run((first,))
    second_actions, _, second_exact, second_delta = run((second,))
    fused_actions, fused_logits, fused_exact, fused_delta = run((first, second))
    permuted_actions, permuted_logits, permuted_exact, permuted_delta = run((second, first))
    shuffled_actions, _, shuffled_exact, shuffled_delta = run((first, second.roll(1, 0)))
    contradictory_actions, _, contradictory_exact, contradictory_delta = run(
        (first, contradictory_second)
    )
    full_actions, full_logits, full_exact, full_delta = run((normal.frames,))
    duplicate_actions, duplicate_logits, duplicate_exact, duplicate_delta = run(
        (normal.frames, normal.frames)
    )
    query = slice(1, None)

    def accuracy(actions: torch.Tensor) -> float:
        return float(
            (actions[:, query] == normal.correct_actions[:, query]).float().mean()
        )

    return {
        "stream_a_accuracy": accuracy(first_actions),
        "stream_b_accuracy": accuracy(second_actions),
        "fused_accuracy": accuracy(fused_actions),
        "permuted_stream_order_accuracy": accuracy(permuted_actions),
        "permuted_stream_order_actions_exact": torch.equal(
            fused_actions, permuted_actions
        ),
        "shuffled_partner_accuracy": accuracy(shuffled_actions),
        "contradictory_partner_accuracy": accuracy(contradictory_actions),
        "contradictory_prediction_flip_rate": float(
            (fused_actions[:, query] != contradictory_actions[:, query])
            .float()
            .mean()
        ),
        "full_n1_accuracy": accuracy(full_actions),
        "duplicate_n2_accuracy": accuracy(duplicate_actions),
        "duplicate_actions_exact": torch.equal(full_actions, duplicate_actions),
        "duplicate_max_logit_difference": float(
            (full_logits - duplicate_logits).abs().max()
        ),
        "permuted_stream_order_max_logit_difference": float(
            (permuted_logits - fused_logits).abs().max()
        ),
        "legacy_wrapper_exact": all(
            (
                first_exact,
                second_exact,
                fused_exact,
                permuted_exact,
                shuffled_exact,
                contradictory_exact,
                full_exact,
                duplicate_exact,
            )
        ),
        "legacy_wrapper_max_logit_difference": max(
            first_delta,
            second_delta,
            fused_delta,
            permuted_delta,
            shuffled_delta,
            contradictory_delta,
            full_delta,
            duplicate_delta,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=250_001)
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
    if args.count < 64 or args.count % 4:
        raise ValueError("count must be at least 64 and divisible by four")
    device = torch.device(args.device)
    legacy, runtime = _load_runtime(args.controller, args.input_bus, device)
    before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    results = {
        appearance: _evaluate(
            legacy,
            runtime,
            count=args.count,
            seed=args.seed + offset * 10_000,
            device=device,
            appearance=appearance,
        )
        for offset, appearance in enumerate(("bars", "diamonds", "dot_pairs"))
    }
    unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in before.items()
    )
    thresholds = {"bars": 0.90, "diamonds": 0.85, "dot_pairs": 0.90}
    flip_thresholds = {"bars": 0.80, "diamonds": 0.70, "dot_pairs": 0.80}
    gates = {
        appearance: bool(
            row["fused_accuracy"] >= thresholds[appearance]
            and row["stream_a_accuracy"] <= 0.65
            and row["stream_b_accuracy"] <= 0.65
            and row["shuffled_partner_accuracy"] <= 0.60
            and row["contradictory_partner_accuracy"] <= 0.25
            and row["contradictory_prediction_flip_rate"] >= flip_thresholds[appearance]
            and row["full_n1_accuracy"] >= 0.95
            and row["duplicate_actions_exact"]
            and row["permuted_stream_order_actions_exact"]
            and row["legacy_wrapper_exact"]
            and row["legacy_wrapper_max_logit_difference"] == 0.0
        )
        for appearance, row in results.items()
    }
    report = {
        "schema": "amodal-runtime-wrapper-composition-audit-v1",
        "claim": (
            "The complete N-to-M runtime wrapper preserves the prior causal "
            "complementary N=2 result while keeping one frozen controller and "
            "two independently registered encoder frontends."
        ),
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "device": str(device),
            "encoder_count": 2,
            "decoder_count": 1,
        },
        "appearance_gates": gates,
        "controller_parameters_unchanged": unchanged,
        "results": results,
        "passed": bool(unchanged and all(gates.values())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "appearance_gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
