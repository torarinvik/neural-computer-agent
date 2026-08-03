"""Train generic N=2 event integration from attempted actions and outcomes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .legacy_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import (
    ACTIONS,
    NULL_ACTION,
    CognitiveLifetimeBatch,
    generate_lifetimes,
)
from .train import attempted_success_loss


def split_complementary_views(
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a rendered scene along a neutral spatial boundary.

    This is environment construction, not learner input metadata. Each frontend
    receives ordinary pixels and the controller receives only encoded events.
    """
    size = frames.shape[-1]
    coordinates = torch.arange(size, device=frames.device)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    first_mask = (xx >= yy).reshape(1, 1, 1, size, size)
    background = frames[..., :1, :1]
    return (
        torch.where(first_mask, frames, background),
        torch.where(~first_mask, frames, background),
    )


@torch.no_grad()
def evaluate_bus(
    runtime,
    bus: AmodalInputBus,
    *,
    count: int,
    seed: int,
    device: torch.device,
    appearance: str = "bars",
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

    def run(streams: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        logits = []
        for trial in range(normal.trials):
            has_feedback = torch.full_like(reward, float(trial == 1))
            events = [runtime.encode(stream[:, trial]) for stream in streams]
            core, state = runtime.step_intention_event(
                bus(events),
                state,
                action,
                reward * has_feedback,
                has_feedback,
            )
            current_logits = runtime.decode(core.intent_event)
            action = current_logits.argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
            logits.append(current_logits)
        return torch.stack(actions, dim=1), torch.stack(logits, dim=1)

    first_actions, _ = run((first,))
    second_actions, _ = run((second,))
    fused_actions, fused_logits = run((first, second))
    permuted_actions, permuted_logits = run((second, first))
    shuffled_actions, _ = run((first, second.roll(1, 0)))
    contradictory_actions, _ = run((first, contradictory_second))
    full_actions, full_logits = run((normal.frames,))
    duplicate_actions, duplicate_logits = run((normal.frames, normal.frames))
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
            (fused_actions[:, query] != contradictory_actions[:, query]).float().mean()
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
    }


def training_loss(
    runtime,
    bus: AmodalInputBus,
    batch: CognitiveLifetimeBatch,
    *,
    exploration: float,
    shuffle_outcomes: bool,
    second_erase: float = 0.0,
) -> tuple[torch.Tensor, float]:
    first, second = split_complementary_views(batch.frames)
    if not 0.0 <= second_erase <= 1.0:
        raise ValueError("second stream erasure must be within [0, 1]")
    if second_erase:
        background = second[..., :1, :1]
        erase = (
            torch.rand(
                second.shape[0],
                second.shape[1],
                1,
                second.shape[-2],
                second.shape[-1],
                device=second.device,
            )
            < second_erase
        )
        second = torch.where(erase, background, second)
    state = runtime.initial_state(batch.batch_size, device=batch.frames.device)
    action = torch.full(
        (batch.batch_size,),
        NULL_ACTION,
        dtype=torch.long,
        device=batch.frames.device,
    )
    reward = torch.zeros(batch.batch_size, device=batch.frames.device)
    losses = []
    successes = []
    for trial in range(batch.trials):
        has_feedback = torch.full_like(reward, float(trial == 1))
        core, state = runtime.step_intention_event(
            bus([runtime.encode(first[:, trial]), runtime.encode(second[:, trial])]),
            state,
            action,
            reward * has_feedback,
            has_feedback,
        )
        logits = runtime.decode(core.intent_event)
        probabilities = torch.softmax(logits, dim=-1)
        behavior = probabilities * (1.0 - exploration) + exploration / ACTIONS
        action = torch.multinomial(behavior, 1).squeeze(1)
        reward = (action == batch.correct_actions[:, trial]).to(logits.dtype)
        learner_reward = (
            reward[torch.randperm(reward.numel(), device=reward.device)]
            if shuffle_outcomes
            else reward
        )
        losses.append(attempted_success_loss(logits, action, learner_reward))
        successes.append(float(reward.mean()))
    return torch.stack(losses).mean(), sum(successes) / len(successes)


def _stable_crossing(curve: list[dict[str, float]], threshold: float) -> int | None:
    for index, row in enumerate(curve):
        suffix = curve[index:]
        if (
            row["verifier_bits"] > 0
            and len(suffix) >= 2
            and all(later["fused_accuracy"] >= threshold for later in suffix)
        ):
            return int(row["verifier_bits"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bus-out", type=Path)
    parser.add_argument("--bus-in", type=Path)
    parser.add_argument("--seed", type=int, default=144_001)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument("--training-erase", type=float, default=0.0)
    parser.add_argument(
        "--training-appearances",
        nargs="+",
        choices=("bars", "diamonds", "dot_pairs"),
        default=["bars"],
        help="Appearances rotate by update but are never exposed to the learner.",
    )
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
    if args.updates < 1 or args.batch_size < 4 or args.batch_size % 4:
        raise ValueError("updates and batch size are invalid")

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(payload, device=device).eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.state_dict().items()
    }
    torch.manual_seed(args.seed)
    bus = AmodalInputBus(runtime.controller.width, args.hidden).to(device)
    if args.bus_in is not None:
        bus_payload = torch.load(args.bus_in, map_location=device, weights_only=False)
        if (
            int(bus_payload["event_width"]) != runtime.controller.width
            or int(bus_payload["residual_hidden"]) != args.hidden
        ):
            raise ValueError("input bus checkpoint dimensions do not match")
        bus.load_state_dict(bus_payload["state_dict"])
    optimizer = torch.optim.Adam(bus.parameters(), lr=args.learning_rate)
    start = time.perf_counter()
    initial = evaluate_bus(
        runtime, bus, count=args.eval_count, seed=args.seed + 50_000, device=device
    )
    curve = [
        {
            "update": 0,
            "verifier_bits": 0,
            "loss": None,
            "attempted_accuracy": None,
            "fused_accuracy": initial["fused_accuracy"],
        }
    ]
    verifier_bits = 0
    for update in range(1, args.updates + 1):
        training_appearance = args.training_appearances[
            (update - 1) % len(args.training_appearances)
        ]
        batch = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + update,
            task="pair_relation",
            appearance=training_appearance,
            support_trials=1,
            device=device,
        )
        loss, attempted_accuracy = training_loss(
            runtime,
            bus,
            batch,
            exploration=args.exploration,
            shuffle_outcomes=args.shuffle_outcomes,
            second_erase=args.training_erase,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        verifier_bits += args.batch_size * batch.trials
        evaluation = evaluate_bus(
            runtime,
            bus,
            count=args.eval_count,
            seed=args.seed + 50_000,
            device=device,
        )
        curve.append(
            {
                "update": update,
                "training_appearance": training_appearance,
                "verifier_bits": verifier_bits,
                "loss": float(loss.detach()),
                "attempted_accuracy": attempted_accuracy,
                "fused_accuracy": evaluation["fused_accuracy"],
            }
        )
    final = evaluate_bus(
        runtime,
        bus,
        count=args.eval_count * 2,
        seed=args.seed + 60_000,
        device=device,
    )
    stable = _stable_crossing(curve, args.threshold)
    runtime_after = runtime.state_dict()
    unchanged = all(
        torch.equal(value, runtime_after[name].detach().cpu())
        for name, value in frozen_before.items()
    )
    learning_gain = curve[-1]["fused_accuracy"] - curve[0]["fused_accuracy"]
    invariants_pass = bool(
        final["fused_accuracy"] >= args.threshold
        and final["stream_a_accuracy"] <= 0.65
        and final["stream_b_accuracy"] <= 0.65
        and final["shuffled_partner_accuracy"] <= 0.65
        and final["contradictory_partner_accuracy"] <= 0.65
        and final["contradictory_prediction_flip_rate"] >= 0.80
        and final["permuted_stream_order_actions_exact"]
        and final["permuted_stream_order_max_logit_difference"] == 0.0
        and final["duplicate_actions_exact"]
        and final["duplicate_max_logit_difference"] == 0.0
        and unchanged
    )
    passed = bool(
        invariants_pass
        and (
            (args.bus_in is not None) or (stable is not None and learning_gain >= 0.10)
        )
    )
    report = {
        "schema": "complementary-amodal-input-bus-bandit-v1",
        "learner_visible": [
            "two encoded event payloads",
            "own attempted opaque action",
            "scalar success of that attempted action",
        ],
        "forbidden": [
            "correct action",
            "unattempted outcomes",
            "task identity",
            "object identity or relation labels",
        ],
        "configuration": {
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "report": str(args.report),
            "bus_in": str(args.bus_in) if args.bus_in is not None else None,
            "bus_out": str(args.bus_out) if args.bus_out is not None else None,
            "device": str(device),
        },
        "curve": curve,
        "stable_bits_to_threshold": stable,
        "learning_gain": learning_gain,
        "initial": initial,
        "final": final,
        "controller_and_adapters_unchanged": unchanged,
        "passed": passed,
        "wall_seconds": time.perf_counter() - start,
    }
    if passed and args.bus_out is not None:
        args.bus_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "amodal-input-bus-v1",
                "event_width": runtime.controller.width,
                "residual_hidden": args.hidden,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in bus.state_dict().items()
                },
                "training": {
                    "verifier_bits": verifier_bits,
                    "stable_bits_to_threshold": stable,
                    "controller_checkpoint": str(args.checkpoint),
                },
            },
            args.bus_out,
        )
        report["bus_saved"] = True
    else:
        report["bus_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": passed,
                "initial": curve[0]["fused_accuracy"],
                "final": curve[-1]["fused_accuracy"],
                "stable_bits_to_threshold": stable,
                "wall_seconds": report["wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
