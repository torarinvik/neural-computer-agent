"""Train the generic input bus to ignore an opaque third-stream distractor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .amodal_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import ACTIONS, NULL_ACTION, generate_lifetimes
from .train import attempted_success_loss
from .train_complementary_input_bus import split_complementary_views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--baseline-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bus-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=165_001)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--freeze-residual", action="store_true")
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
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
    if args.rehearsal_weight < 0:
        raise ValueError("rehearsal weight must be nonnegative")
    device = torch.device(args.device)
    runtime_payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(runtime_payload, device=device).eval()
    bus_payload = torch.load(args.baseline_bus, map_location=device, weights_only=False)
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    if args.freeze_residual and bus.residual is not None:
        for parameter in bus.residual.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(bus.parameters(), lr=args.learning_rate)
    start = time.perf_counter()
    curve = []
    for update in range(1, args.updates + 1):
        batch = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + update,
            task="pair_relation",
            appearance="bars",
            support_trials=1,
            device=device,
        )
        distractor = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + 100_000 + update,
            task="pair_relation",
            appearance="bars",
            support_trials=1,
            device=device,
        )
        first, second = split_complementary_views(batch.frames)
        def run(streams, batch_to_run):
            loss_rows = []
            successes = []
            state = runtime.initial_state(batch_to_run.batch_size, device=device)
            action = torch.full(
                (batch_to_run.batch_size,), NULL_ACTION, dtype=torch.long, device=device
            )
            reward = torch.zeros(batch_to_run.batch_size, device=device)
            for trial in range(batch_to_run.trials):
                feedback = torch.full_like(reward, float(trial == 1))
                events = [runtime.encode(stream[:, trial]) for stream in streams]
                core, state = runtime.step_intention_event(
                    bus(events), state, action, reward * feedback, feedback
                )
                logits = runtime.decode(core.intent_event)
                probabilities = torch.softmax(logits, dim=-1)
                behavior = probabilities * (1.0 - args.exploration) + args.exploration / ACTIONS
                action = torch.multinomial(behavior, 1).squeeze(1)
                reward = (action == batch_to_run.correct_actions[:, trial]).to(logits.dtype)
                loss_rows.append(attempted_success_loss(logits, action, reward))
                successes.append(float(reward.mean()))
            return torch.stack(loss_rows).mean(), sum(successes) / len(successes)

        n3_loss, n3_success = run((first, second, distractor.frames), batch)
        n2_loss, n2_success = run((first, second), batch)
        loss = n3_loss + args.rehearsal_weight * n2_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append(
            {
                "update": update,
                "loss": float(loss.detach()),
                "attempted_accuracy": n3_success,
                "rehearsal_accuracy": n2_success,
                "verifier_bits": args.batch_size * batch.trials,
            }
        )
    args.bus_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "amodal-input-bus-v1",
            "event_width": runtime.controller.width,
            "residual_hidden": bus.residual[0].out_features
            if bus.residual is not None
            else 0,
            "state_dict": {
                name: value.detach().cpu() for name, value in bus.state_dict().items()
            },
            "training": {
                "method": "scalar-outcome-n3-distractor",
                "controller_checkpoint": str(args.controller),
                "baseline_bus": str(args.baseline_bus),
                "updates": args.updates,
                "batch_size": args.batch_size,
                "rehearsal_weight": args.rehearsal_weight,
                "freeze_residual": args.freeze_residual,
            },
        },
        args.bus_out,
    )
    report = {
        "schema": "amodal-input-n3-distractor-training-v1",
        "learner_visible": [
            "three encoded event payloads",
            "own attempted opaque action",
            "scalar success of that attempted action",
        ],
        "forbidden": ["correct action", "distractor identity", "task labels"],
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "freeze_residual": args.freeze_residual,
            "rehearsal_weight": args.rehearsal_weight,
            "device": str(device),
        },
        "curve": curve,
        "wall_seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final_attempted_accuracy": curve[-1]["attempted_accuracy"]}))


if __name__ == "__main__":
    main()
