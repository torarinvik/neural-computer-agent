"""Calibrate a second output backend from attempted commands and scalar reward."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .legacy_interface import IntentEvent
from .legacy_runtime import OpaqueProtocolDecoder, runtime_from_legacy_payload
from .environment import (
    ACTIONS,
    NULL_ACTION,
    CognitiveLifetimeBatch,
    generate_lifetimes,
)
from .train import attempted_success_loss
from .train_procedural_shape_span import (
    ProceduralShapeBatch,
    generate_procedural_shape_batch,
    nuisance_from_level,
)

PROTOCOL_CODES = torch.tensor([1, 0], dtype=torch.long)


@torch.no_grad()
def collect_frozen_intentions(
    runtime,
    batch: CognitiveLifetimeBatch,
    *,
    feedback_trials: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return opaque intentions plus verifier-private scoring facts.

    The primary decoder drives the already-audited recurrent trajectory. The
    candidate decoder never sees the returned correct actions; its training
    caller uses them only to score the candidate's attempted protocol command.
    """
    state = runtime.initial_state(
        batch.batch_size, device=batch.frames.device, dtype=batch.frames.dtype
    )
    previous_action = torch.full(
        (batch.batch_size,),
        NULL_ACTION,
        device=batch.frames.device,
        dtype=torch.long,
    )
    previous_reward = torch.zeros(batch.batch_size, device=batch.frames.device)
    intentions = []
    correct_actions = []
    primary_actions = []
    for trial in range(batch.trials):
        has_feedback = torch.full_like(
            previous_reward, float(0 < trial <= feedback_trials)
        )
        core_output, state = runtime.step_intention(
            batch.frames[:, trial],
            state,
            previous_action,
            previous_reward * has_feedback,
            has_feedback,
        )
        primary_logits = runtime.decode(core_output.intent_event)
        primary_action = primary_logits.argmax(dim=-1)
        reward = (primary_action == batch.correct_actions[:, trial]).to(
            primary_logits.dtype
        )
        if trial >= feedback_trials:
            intentions.append(core_output.intent_event.payload)
            correct_actions.append(batch.correct_actions[:, trial])
            primary_actions.append(primary_action)
        previous_action = primary_action
        previous_reward = reward
    return (
        torch.cat(intentions).detach(),
        torch.cat(correct_actions),
        torch.cat(primary_actions),
    )


@torch.no_grad()
def collect_frozen_span_intentions(
    runtime, batch: ProceduralShapeBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect query intentions while the frozen primary path drives history."""
    device = batch.presentation_frames.device
    state = runtime.initial_state(
        batch.batch_size,
        device=device,
        dtype=batch.presentation_frames.dtype,
    )
    null_action = torch.full(
        (batch.batch_size,), NULL_ACTION, device=device, dtype=torch.long
    )
    zeros = torch.zeros(batch.batch_size, device=device)
    for index in range(batch.span):
        _, state = runtime.step_intention(
            batch.presentation_frames[:, index],
            state,
            null_action,
            zeros,
            zeros,
        )

    previous_action = null_action
    previous_reward = zeros
    intentions = []
    primary_actions = []
    for index in range(batch.query_frames.shape[1]):
        has_feedback = torch.full_like(previous_reward, float(index > 0))
        core_output, state = runtime.step_intention(
            batch.query_frames[:, index],
            state,
            previous_action,
            previous_reward * has_feedback,
            has_feedback,
        )
        primary_action = runtime.decode(core_output.intent_event).argmax(dim=-1)
        previous_reward = (primary_action == batch.correct_actions[:, index]).to(
            previous_reward.dtype
        )
        previous_action = primary_action
        intentions.append(core_output.intent_event.payload)
        primary_actions.append(primary_action)
    return (
        torch.cat(intentions).detach(),
        batch.correct_actions.T.reshape(-1),
        torch.stack(primary_actions).reshape(-1),
    )


def collect_training_batch(
    runtime,
    *,
    task: str,
    count: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one task stream without exposing its identity to the decoder."""
    if task == "span2":
        batch = generate_procedural_shape_batch(
            count,
            span=2,
            vocabulary=2,
            seed=seed,
            nuisance=nuisance_from_level(0.0),
            objective="recognition",
            query_count=2,
            device=device,
        )
        return collect_frozen_span_intentions(runtime, batch)
    feedback_trials = 2 if task == "four_rule" else 1
    batch = generate_lifetimes(
        count,
        6,
        seed=seed,
        task=task,
        support_trials=feedback_trials,
        device=device,
    )
    return collect_frozen_intentions(runtime, batch, feedback_trials=feedback_trials)


def protocol_attempt_loss(
    decoder: OpaqueProtocolDecoder,
    intentions: torch.Tensor,
    verifier_correct_actions: torch.Tensor,
    *,
    exploration: float,
    shuffle_outcomes: bool = False,
) -> tuple[torch.Tensor, float]:
    """Bandit update: only attempted command and its scalar outcome enter loss."""
    logits = decoder(IntentEvent(intentions))
    probabilities = torch.softmax(logits, dim=-1)
    behavior = probabilities * (1.0 - exploration) + exploration / ACTIONS
    attempted = torch.multinomial(behavior, 1).squeeze(1)
    # This line is the verifier boundary. Correct commands never enter the
    # decoder or loss; only whether its one attempted command succeeded does.
    correct_commands = PROTOCOL_CODES.to(verifier_correct_actions.device)[
        verifier_correct_actions
    ]
    outcomes = (attempted == correct_commands).to(logits.dtype)
    attempted_accuracy = float(outcomes.mean())
    learner_outcomes = (
        outcomes[torch.randperm(outcomes.numel(), device=outcomes.device)]
        if shuffle_outcomes
        else outcomes
    )
    return (
        attempted_success_loss(logits, attempted, learner_outcomes),
        attempted_accuracy,
    )


@torch.no_grad()
def evaluate_decoder(
    runtime,
    decoder: OpaqueProtocolDecoder,
    *,
    count: int,
    seed: int,
    task: str,
    feedback_trials: int,
    appearance: str = "bars",
    device: torch.device,
) -> dict[str, float]:
    batch = generate_lifetimes(
        count,
        6,
        seed=seed,
        heldout=True,
        task=task,
        appearance=appearance,
        support_trials=feedback_trials,
        device=device,
    )
    intentions, correct_actions, primary_actions = collect_frozen_intentions(
        runtime, batch, feedback_trials=feedback_trials
    )
    correct_commands = PROTOCOL_CODES.to(device)[correct_actions]
    aligned = decoder(IntentEvent(intentions)).argmax(dim=-1)
    shuffled = decoder(IntentEvent(intentions.roll(1, 0))).argmax(dim=-1)
    zeroed = decoder(IntentEvent(torch.zeros_like(intentions))).argmax(dim=-1)
    return {
        "aligned_accuracy": float((aligned == correct_commands).float().mean()),
        "shuffled_accuracy": float((shuffled == correct_commands).float().mean()),
        "zero_intention_accuracy": float((zeroed == correct_commands).float().mean()),
        "primary_accuracy": float((primary_actions == correct_actions).float().mean()),
        "examples": int(intentions.shape[0]),
    }


def _stable_crossing(
    curve: list[dict[str, float]], threshold: float, *, confirmations: int = 2
) -> int | None:
    for index, row in enumerate(curve):
        suffix = curve[index:]
        if (
            row["verifier_bits"] > 0
            and len(suffix) >= confirmations
            and all(later["accuracy"] >= threshold for later in suffix)
        ):
            return int(row["verifier_bits"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decoder-out", type=Path)
    parser.add_argument("--seed", type=int, default=131_001)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument(
        "--training-tasks",
        nargs="+",
        choices=("binary_mapping", "four_rule", "pair_relation", "span2"),
        default=["binary_mapping"],
        help="Task streams are rotated by update; the decoder never receives the name.",
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[0.001, 0.003, 0.01]
    )
    parser.add_argument("--threshold", type=float, default=0.85)
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
    if not 0.0 <= args.exploration <= 1.0:
        raise ValueError("exploration must be between zero and one")

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(payload, device=device).eval()
    if runtime.compatibility_suffix_active:
        raise ValueError("checkpoint still has an active compatibility suffix")
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.state_dict().items()
    }

    start = time.perf_counter()
    arms = []
    for arm_index, learning_rate in enumerate(args.learning_rates):
        torch.manual_seed(args.seed + arm_index)
        decoder = OpaqueProtocolDecoder(runtime.controller.intention_width).to(device)
        # A neutral decoder prevents a lucky random orientation from being
        # mistaken for learning the protocol direction.
        with torch.no_grad():
            decoder.network.weight.zero_()
            decoder.network.bias.zero_()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        zero_shot = evaluate_decoder(
            runtime,
            decoder,
            count=args.eval_count,
            seed=args.seed + 50_000,
            task="binary_mapping",
            feedback_trials=1,
            device=device,
        )
        curve = [
            {
                "update": 0,
                "verifier_bits": 0,
                "loss": None,
                "attempted_accuracy": None,
                "accuracy": zero_shot["aligned_accuracy"],
            }
        ]
        verifier_bits = 0
        for update in range(1, args.updates + 1):
            training_task = args.training_tasks[(update - 1) % len(args.training_tasks)]
            intentions, correct_actions, _ = collect_training_batch(
                runtime,
                task=training_task,
                count=args.batch_size,
                seed=args.seed + update,
                device=device,
            )
            loss, attempted_accuracy = protocol_attempt_loss(
                decoder,
                intentions,
                correct_actions,
                exploration=args.exploration,
                shuffle_outcomes=args.shuffle_outcomes,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            verifier_bits += int(intentions.shape[0])
            evaluation = evaluate_decoder(
                runtime,
                decoder,
                count=args.eval_count,
                seed=args.seed + 50_000,
                task="binary_mapping",
                feedback_trials=1,
                device=device,
            )
            curve.append(
                {
                    "update": update,
                    "training_task": training_task,
                    "verifier_bits": verifier_bits,
                    "loss": float(loss.detach()),
                    "attempted_accuracy": attempted_accuracy,
                    "accuracy": evaluation["aligned_accuracy"],
                }
            )
        final_controls = {
            "binary": evaluate_decoder(
                runtime,
                decoder,
                count=args.eval_count * 2,
                seed=args.seed + 60_000,
                task="binary_mapping",
                feedback_trials=1,
                device=device,
            ),
            "four_rule": evaluate_decoder(
                runtime,
                decoder,
                count=args.eval_count * 2,
                seed=args.seed + 61_000,
                task="four_rule",
                feedback_trials=2,
                device=device,
            ),
            "relation_bars": evaluate_decoder(
                runtime,
                decoder,
                count=args.eval_count * 2,
                seed=args.seed + 62_000,
                task="pair_relation",
                feedback_trials=1,
                appearance="bars",
                device=device,
            ),
            "relation_diamonds": evaluate_decoder(
                runtime,
                decoder,
                count=args.eval_count * 2,
                seed=args.seed + 63_000,
                task="pair_relation",
                feedback_trials=1,
                appearance="diamonds",
                device=device,
            ),
        }
        stable = _stable_crossing(curve, args.threshold)
        learning_gain = curve[-1]["accuracy"] - curve[0]["accuracy"]
        passed = bool(
            stable is not None
            and learning_gain >= 0.10
            and all(
                row["aligned_accuracy"] >= args.threshold
                and row["shuffled_accuracy"] <= 0.60
                and row["zero_intention_accuracy"] <= 0.60
                and row["primary_accuracy"] >= 0.85
                for row in final_controls.values()
            )
        )
        arms.append(
            {
                "learning_rate": learning_rate,
                "curve": curve,
                "stable_bits_to_threshold": stable,
                "zero_shot_accuracy": curve[0]["accuracy"],
                "learning_gain": learning_gain,
                "controls": final_controls,
                "passed": passed,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in decoder.state_dict().items()
                },
            }
        )

    eligible = [arm for arm in arms if arm["passed"]]
    selected = min(
        eligible,
        key=lambda arm: (
            arm["stable_bits_to_threshold"],
            -arm["controls"]["binary"]["aligned_accuracy"],
        ),
        default=None,
    )
    frozen_after = runtime.state_dict()
    controller_unchanged = all(
        torch.equal(value, frozen_after[name].detach().cpu())
        for name, value in frozen_before.items()
    )
    report_arms = [
        {key: value for key, value in arm.items() if key != "state_dict"}
        for arm in arms
    ]
    report = {
        "schema": "amodal-output-fanout-bandit-v1",
        "checkpoint": str(args.checkpoint),
        "learner_visible": [
            "base intention payload",
            "attempted opaque protocol command",
            "scalar success of that attempted command",
        ],
        "forbidden": [
            "correct protocol command",
            "unattempted command outcomes",
            "primary decoder logits",
            "task identity as decoder input",
        ],
        "configuration": {
            **vars(args),
            "checkpoint": str(args.checkpoint),
            "report": str(args.report),
            "decoder_out": (
                str(args.decoder_out) if args.decoder_out is not None else None
            ),
            "device": str(device),
        },
        "arms": report_arms,
        "selected_learning_rate": (
            selected["learning_rate"] if selected is not None else None
        ),
        "controller_and_primary_decoder_unchanged": controller_unchanged,
        "passed": bool(selected is not None and controller_unchanged),
        "wall_seconds": time.perf_counter() - start,
    }
    if selected is not None and args.decoder_out is not None:
        args.decoder_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "opaque-protocol-decoder-v1",
                "intention_width": runtime.controller.intention_width,
                "commands": ACTIONS,
                "protocol_codes": PROTOCOL_CODES,
                "state_dict": selected["state_dict"],
                "training": {
                    "learning_rate": selected["learning_rate"],
                    "stable_bits_to_threshold": selected["stable_bits_to_threshold"],
                    "controller_checkpoint": str(args.checkpoint),
                },
            },
            args.decoder_out,
        )
        report["decoder_saved"] = True
    else:
        report["decoder_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "selected_learning_rate": report["selected_learning_rate"],
                "stable_bits_to_threshold": (
                    selected["stable_bits_to_threshold"]
                    if selected is not None
                    else None
                ),
                "wall_seconds": report["wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
