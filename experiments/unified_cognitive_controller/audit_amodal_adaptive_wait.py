"""Audit a learned, payload-blind wait/proceed policy on delayed streams."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .amodal_interface import AmodalEvent
from .amodal_runtime import AmodalControllerRuntime, AmodalEventWindowBuffer
from .amodal_wait_policy import AmodalArrivalPredictor, arrival_features
from .audit_audio_timing_buffer import _load_runtime
from .environment import NULL_ACTION, generate_lifetimes
from .train_amodal_audio_alignment import render_pair_relation_audio
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_predictor(path: Path, device: torch.device) -> AmodalArrivalPredictor:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "amodal-arrival-predictor-v1":
        raise ValueError("unsupported arrival predictor artifact schema")
    predictor = AmodalArrivalPredictor(int(payload["hidden"])).to(device)
    predictor.load_state_dict(payload["state_dict"])
    return predictor.eval()


def _sample_delays(
    count: int, *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate held-out transport traces with latent reliable/sparse regimes."""
    generator = torch.Generator(device=device).manual_seed(seed)
    reliable = torch.rand(count, generator=generator, device=device) >= 0.5
    random = torch.rand(count, 6, generator=generator, device=device)
    delays = torch.full((count, 6), -1, dtype=torch.long, device=device)
    reliable_cutoffs = (0.70, 0.95, 0.99)
    sparse_cutoffs = (0.20, 0.35, 0.45)
    for index, cutoff in enumerate(reliable_cutoffs):
        delays = torch.where(
            reliable[:, None] & (random < cutoff) & (delays < 0),
            torch.full_like(delays, index),
            delays,
        )
    for index, cutoff in enumerate(sparse_cutoffs):
        delays = torch.where(
            (~reliable[:, None]) & (random < cutoff) & (delays < 0),
            torch.full_like(delays, index),
            delays,
        )
    return delays, reliable


def _event(
    runtime: AmodalControllerRuntime,
    name: str,
    payload: torch.Tensor,
    timestamp: int,
) -> AmodalEvent:
    payload = payload.reshape(1, -1)
    return AmodalEvent(
        payload=payload,
        timestamp=torch.tensor([float(timestamp)], device=payload.device),
    )


@torch.no_grad()
def _rollout(
    runtime: AmodalControllerRuntime,
    predictor: AmodalArrivalPredictor,
    encoded_a: torch.Tensor,
    encoded_b: torch.Tensor,
    labels: torch.Tensor,
    delays: torch.Tensor,
    *,
    mode: str,
    threshold: float,
    deadline: int,
    latency_cost: float,
) -> dict[str, float | int]:
    count, trials, width = encoded_a.shape
    if encoded_b.shape != encoded_a.shape or labels.shape[:2] != (count, trials):
        raise ValueError("encoded streams and labels have incompatible shapes")
    if mode not in {
        "immediate",
        "fixed1",
        "fixed2",
        "adaptive",
        "adaptive_no_history",
        "adaptive_inverted_history",
    }:
        raise ValueError(f"unknown wait mode {mode}")
    actions = torch.full((count, trials), -1, dtype=torch.long, device=labels.device)
    latencies = torch.zeros(count, trials, device=labels.device)
    partial_windows = 0
    dropped_late = 0
    complete_windows = 0
    for episode in range(count):
        buffer = AmodalEventWindowBuffer(("stream_a", "stream_b"))
        state = runtime.initial_state(1, device=labels.device)
        previous_action = torch.full(
            (1,), NULL_ACTION, dtype=torch.long, device=labels.device
        )
        previous_reward = torch.zeros(1, device=labels.device)
        processed = 0
        history: list[bool] = []
        for trial in range(trials):
            base = trial * 10
            delay = int(delays[episode, trial])
            released = False

            def consume(window, release_clock: int) -> None:
                nonlocal state, previous_action, previous_reward, processed
                nonlocal partial_windows, complete_windows, released
                window_trial = int(round(window.timestamp / 10.0))
                if window_trial != trial:
                    raise AssertionError("window released at the wrong trial")
                feedback = torch.full(
                    (1,), float(processed > 0), device=labels.device
                )
                output, state = runtime.step_events(
                    window.collection,
                    state,
                    previous_action,
                    previous_reward * feedback,
                    feedback,
                )
                previous_action = output.decoded["action"].argmax(dim=-1)
                previous_reward = (
                    previous_action == labels[episode, trial].reshape(1)
                ).float()
                actions[episode, trial] = previous_action
                latencies[episode, trial] = float(release_clock - base)
                processed += 1
                partial_windows += int(not window.complete)
                complete_windows += int(window.complete)
                released = True

            arrivals = {
                "stream_a": _event(
                    runtime, "stream_a", encoded_a[episode, trial], base
                )
            }
            if delay == 0:
                arrivals["stream_b"] = _event(
                    runtime, "stream_b", encoded_b[episode, trial], base
                )
            ready = buffer.push(arrivals)
            for window in ready:
                consume(window, base)
            for offset in range(0, deadline + 1):
                if released:
                    break
                if delay > 0 and delay == offset:
                    ready = buffer.push(
                        {
                            "stream_b": _event(
                                runtime,
                                "stream_b",
                                encoded_b[episode, trial],
                                base,
                            )
                        }
                    )
                    for window in ready:
                        consume(window, base + offset)
                    if released:
                        break
                status = buffer.pending_status(current_timestamp=base + offset)
                if not status:
                    raise AssertionError("pending window disappeared")
                current = status[0]
                if mode == "immediate":
                    should_release = True
                elif mode == "fixed1":
                    should_release = current.age >= 1.0
                elif mode == "fixed2":
                    should_release = current.age >= 2.0
                else:
                    policy_history = history
                    if mode == "adaptive_no_history":
                        policy_history = []
                    elif mode == "adaptive_inverted_history":
                        policy_history = [not value for value in history]
                    feature = arrival_features(
                        current,
                        policy_history,
                        deadline=float(deadline),
                    ).to(labels.device)
                    probability = float(predictor(feature.unsqueeze(0))[0])
                    should_release = probability < threshold
                if should_release or offset >= deadline:
                    consume(buffer.release_pending(current.timestamp), base + offset)
                    if delay >= 0 and delay > offset:
                        dropped_late += 1
            if not released:
                raise AssertionError("wait policy failed to release a window")
            history.append(delay >= 0)
            del history[:-4]
        if processed != trials or buffer.pending_timestamps:
            raise AssertionError("wait policy left pending or missing windows")
    query = slice(1, None)
    accuracy = float((actions[:, query] == labels[:, query]).float().mean())
    mean_latency = float(latencies[:, query].mean())
    utility = accuracy - latency_cost * mean_latency
    return {
        "accuracy": accuracy,
        "mean_latency": mean_latency,
        "verified_utility": utility,
        "partial_windows": partial_windows,
        "complete_windows": complete_windows,
        "dropped_late": dropped_late,
    }


def _prepare(
    runtime: AmodalControllerRuntime,
    frames: torch.Tensor,
    audio_source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count, trials = frames.shape[:2]
    encoded_a = runtime.encoders["stream_a"](
        frames.reshape(-1, *frames.shape[2:])
    ).reshape(count, trials, -1)
    audio = render_pair_relation_audio(
        audio_source.reshape(-1, *audio_source.shape[2:]), samples=2048
    )
    encoded_b = runtime.encoders["stream_b"](audio).reshape(count, trials, -1)
    return encoded_a, encoded_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=992101)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--deadline", type=int, default=2)
    parser.add_argument("--latency-cost", type=float, default=0.03)
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
    if args.count < 4 or args.threshold <= 0 or args.threshold >= 1:
        raise ValueError("count and threshold are invalid")
    if args.deadline < 1 or args.latency_cost <= 0:
        raise ValueError("deadline and latency cost are invalid")
    device = torch.device(args.device)
    runtime, _ = _load_runtime(args.controller, args.input_bus, args.audio, device)
    predictor = _load_predictor(args.predictor, device)
    results: dict[str, dict[str, dict[str, float | int]]] = {}
    for appearance_index, appearance in enumerate(("bars", "diamonds", "dot_pairs")):
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
        delays, reliable = _sample_delays(
            args.count,
            seed=args.seed + appearance_index * 10_000 + 1,
            device=device,
        )
        mode_results: dict[str, dict[str, float | int]] = {}
        for task_mode, frames, audio_source in (
            ("complementary", first, second),
            ("redundant", batch.frames, batch.frames),
        ):
            encoded_a, encoded_b = _prepare(runtime, frames, audio_source)
            for wait_mode in (
                "immediate",
                "fixed1",
                "fixed2",
                "adaptive",
                "adaptive_no_history",
                "adaptive_inverted_history",
            ):
                mode_results[f"{task_mode}_{wait_mode}"] = _rollout(
                    runtime,
                    predictor,
                    encoded_a,
                    encoded_b,
                    batch.correct_actions,
                    delays,
                    mode=wait_mode,
                    threshold=args.threshold,
                    deadline=args.deadline,
                    latency_cost=args.latency_cost,
                )
        mode_results["transport_reliable_fraction"] = {
            "value": float(reliable.float().mean())
        }
        results[appearance] = mode_results
    adaptive = [
        results[appearance]["complementary_adaptive"]
        for appearance in results
    ]
    adaptive_no_history = [
        results[appearance]["complementary_adaptive_no_history"]
        for appearance in results
    ]
    fixed2 = [results[appearance]["complementary_fixed2"] for appearance in results]
    passed = bool(
        sum(row["accuracy"] for row in adaptive) / len(adaptive)
        >= sum(row["accuracy"] for row in fixed2) / len(fixed2) - 0.02
        and sum(row["verified_utility"] for row in adaptive) / len(adaptive)
        >= sum(row["verified_utility"] for row in fixed2) / len(fixed2) - 0.01
        and sum(row["mean_latency"] for row in adaptive) / len(adaptive)
        < sum(row["mean_latency"] for row in fixed2) / len(fixed2)
        and sum(row["verified_utility"] for row in adaptive) / len(adaptive)
        >= sum(row["verified_utility"] for row in adaptive_no_history)
        / len(adaptive_no_history) + 0.005
        and all(
            results[appearance]["redundant_adaptive"]["accuracy"] >= 0.85
            for appearance in results
        )
    )
    report = {
        "schema": "amodal-adaptive-wait-audit-v1",
        "claim": (
            "A payload-blind arrival predictor chooses wait/proceed on held-out "
            "delays with verified utility near a fixed two-step timeout."
        ),
        "labels_used": [],
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "audio": str(args.audio),
        "audio_sha256": _sha256(args.audio),
        "predictor": str(args.predictor),
        "predictor_sha256": _sha256(args.predictor),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "threshold": args.threshold,
            "deadline": args.deadline,
            "latency_cost": args.latency_cost,
            "device": str(device),
        },
        "appearances": results,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "appearance_count": len(results)}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
