"""Measure forward transfer into isolated frozen-core external growth state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.artifact_consolidation_amodal.train import _direct_growth_runtime
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    FrameEventEncoder,
    _accuracy,
    _copy_parent_weights,
    _feedback,
    _freeze_except,
    _rollout,
    _runtime,
)
from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalOutputBus,
    OpaqueProtocolDecoder,
    paired_counterfactual_ranking_loss,
)


def _stable_bits(
    progress: list[dict[str, float | int]],
    *,
    threshold: float,
    bits_per_update: int,
) -> int | None:
    for index, row in enumerate(progress):
        if all(
            float(later["heldout_accuracy"]) >= threshold
            for later in progress[index:]
        ):
            return int(row["update"]) * bits_per_update
    return None


def _paired_rollout(runtime, batch, *, shuffle_outcomes: bool) -> dict[str, torch.Tensor]:
    """Train action preference from paired attempted outcomes only."""
    device = batch.input_frames.device
    state = runtime.initial_state(batch.batch_size, device=device)
    zeros = torch.zeros(batch.batch_size, device=device)
    previous_action = torch.zeros(batch.batch_size, 2, device=device)
    previous_reward = zeros
    previous_propensity = torch.ones(batch.batch_size, device=device)
    previous_has_feedback = zeros

    def tick(frame, current_state, feedback):
        return runtime.step_streams({"vision": frame}, current_state, feedback)

    quiet = _feedback(previous_action, previous_reward, previous_propensity, zeros)
    for frame in batch.input_frames.transpose(0, 1):
        _, state = tick(frame, state, quiet)
    for frame in batch.distractor_frames.transpose(0, 1):
        _, state = tick(frame, state, quiet)

    attempted = torch.tensor([[0, 1]], dtype=torch.long, device=device).expand(
        batch.batch_size, -1
    )
    losses: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    for index in range(batch.span):
        feedback = _feedback(
            previous_action,
            previous_reward,
            previous_propensity,
            previous_has_feedback,
        )
        output, state = tick(batch.query_frames[:, index], state, feedback)
        logits = output.decoded["action"]
        utilities = (
            attempted == batch.correct_actions[:, index : index + 1]
        ).float()
        if shuffle_outcomes:
            utilities = utilities.roll(1, dims=0)
        loss, _ = paired_counterfactual_ranking_loss(
            logits,
            attempted,
            utilities,
        )
        probabilities = torch.softmax(logits, dim=-1)
        behavior = probabilities * 0.9 + 0.05
        action_index = torch.multinomial(behavior, 1).squeeze(1)
        reward = (action_index == batch.correct_actions[:, index]).float()
        delivered = reward.roll(1) if shuffle_outcomes else reward
        losses.append(loss)
        rewards.append(reward)
        previous_action = torch.nn.functional.one_hot(action_index, 2).float()
        previous_reward = delivered
        previous_propensity = probabilities.gather(
            1, action_index.unsqueeze(1)
        ).squeeze(1).detach()
        previous_has_feedback = torch.ones_like(previous_reward)
    return {
        "loss": torch.stack(losses).mean(),
        "rewards": torch.stack(rewards, dim=1),
    }


def _train_with_progress(
    runtime,
    *,
    operation: str,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    learning_rate: float,
    audit_count: int,
    eval_every: int,
    shuffle_outcomes: bool = False,
    auxiliary_operation: str | None = None,
    auxiliary_span: int | None = None,
    credit_mode: str = "sampled",
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    trainable = [parameter for parameter in runtime.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("learner has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    progress: list[dict[str, float | int]] = []
    runtime.train()
    for update in range(1, updates + 1):
        batch = generate_sequence_memory_batch(
            batch_size,
            span=span,
            distractors=1,
            seed=seed + update * 10_007,
            operation=operation,
        )
        if credit_mode == "sampled":
            result = _rollout(
                runtime,
                batch,
                train=True,
                shuffle_outcomes=shuffle_outcomes,
            )
        elif credit_mode == "paired_counterfactual":
            result = _paired_rollout(
                runtime,
                batch,
                shuffle_outcomes=shuffle_outcomes,
            )
        else:
            raise ValueError(f"unknown credit mode: {credit_mode!r}")
        loss = result["loss"]
        verifier_bits = batch_size * span
        if auxiliary_operation is not None:
            if auxiliary_span is None:
                raise ValueError("auxiliary span is required with auxiliary operation")
            auxiliary_batch = generate_sequence_memory_batch(
                batch_size,
                span=auxiliary_span,
                distractors=1,
                seed=seed + 5_000_003 + update * 20_021,
                operation=auxiliary_operation,
            )
            auxiliary_result = (
                _rollout(
                    runtime,
                    auxiliary_batch,
                    train=True,
                    shuffle_outcomes=shuffle_outcomes,
                )
                if credit_mode == "sampled"
                else _paired_rollout(
                    runtime,
                    auxiliary_batch,
                    shuffle_outcomes=shuffle_outcomes,
                )
            )
            loss = loss + auxiliary_result["loss"]
            verifier_bits += batch_size * auxiliary_span
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "unique_logical_lifetimes": update * batch_size,
                "unique_verifier_bits": update * verifier_bits,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(loss.detach()),
            }
        )
        if update == updates or (eval_every > 0 and update % eval_every == 0):
            runtime.eval()
            progress.append(
                {
                    "update": update,
                    "unique_verifier_bits": update * batch_size * span,
                    "heldout_accuracy": _accuracy(
                        runtime,
                        operation=operation,
                        count=audit_count,
                        span=span,
                        seed=seed + 1_000_000 + update,
                    ),
                }
            )
            runtime.train()
    runtime.eval()
    return history, progress


def _state_digest(runtime, excluded_prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, value in runtime.controller.state_dict().items():
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dynamic_growth_runtime(*, seed: int, width: int) -> AmodalControllerRuntime:
    """Build an external slot with its own recurrent temporal state."""
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=32,
        workspace_slots=4,
        intention_width=16,
        feedback_width=2,
        event_window_capacity=16,
        reliability_hidden=16,
        growth_register_widths=(width, width),
        growth_recurrent_from=0,
        growth_gated=True,
        growth_from_intention=True,
        growth_gate_from_context=True,
    )
    return AmodalControllerRuntime(
        controller,
        encoders={"vision": FrameEventEncoder(32)},
        output_bus=AmodalOutputBus(
            {"action": OpaqueProtocolDecoder(16, 2, hidden=16)}
        ),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.transfer_updates,
        args.batch_size,
        args.audit_count,
        args.eval_every,
    ) < 1:
        raise ValueError("updates, batch size, and eval interval must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    if args.rehearse_span < 0:
        raise ValueError("rehearse span must be non-negative")
    if args.credit_mode not in {"sampled", "paired_counterfactual"}:
        raise ValueError("unknown credit mode")
    if args.parent_extra_updates < 0:
        raise ValueError("parent extra updates must be non-negative")
    rehearsal_operation = "forward" if args.rehearse_span else None

    parent = _runtime(seed=args.seed, growth=False)
    torch.manual_seed(args.seed + 900_001)
    parent_history, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        credit_mode=args.credit_mode,
    )
    parent_extra_history: list[dict[str, float | int]] = []
    parent_extra_progress: list[dict[str, float | int]] = []
    if args.parent_extra_updates:
        parent_extra_history, parent_extra_progress = _train_with_progress(
            parent,
            operation=args.parent_extra_operation,
            updates=args.parent_extra_updates,
            batch_size=args.batch_size,
            span=2,
            seed=args.seed + 200,
            learning_rate=args.learning_rate,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            auxiliary_operation="forward",
            auxiliary_span=2,
            credit_mode=args.credit_mode,
        )
    parent.eval()
    parent_accuracy = _accuracy(
        parent,
        operation="forward",
        count=args.audit_count,
        span=2,
        seed=args.seed + 2_000_001,
    )
    parent_extra_accuracy = (
        _accuracy(
            parent,
            operation=args.parent_extra_operation,
            count=args.audit_count,
            span=2,
            seed=args.seed + 2_000_002,
        )
        if args.parent_extra_updates
        else None
    )

    # Both arms start with the exact same random growth architecture. Only the
    # transferred arm receives the mastered parent state. The fresh arm is
    # intentionally allowed to train its entire model.
    growth_runtime = (
        _dynamic_growth_runtime
        if args.dynamic_growth
        else _direct_growth_runtime
    )
    transferred = growth_runtime(seed=args.seed + 1, width=args.growth_width)
    fresh = growth_runtime(seed=args.seed + 1, width=args.growth_width)
    _copy_parent_weights(parent, transferred)
    _freeze_except(transferred, ("growth_slots.0.",))
    transferred_digest_before = _state_digest(transferred, ("growth_slots.0.",))

    torch.manual_seed(args.seed + 900_002)
    transferred_history, transferred_progress = _train_with_progress(
        transferred,
        operation=args.target_operation,
        updates=args.transfer_updates,
        batch_size=args.batch_size,
        span=args.target_span,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        auxiliary_operation=rehearsal_operation,
        auxiliary_span=args.rehearse_span or None,
        credit_mode=args.credit_mode,
    )
    transferred_digest_after = _state_digest(transferred, ("growth_slots.0.",))

    torch.manual_seed(args.seed + 900_002)
    fresh_history, fresh_progress = _train_with_progress(
        fresh,
        operation=args.target_operation,
        updates=args.transfer_updates,
        batch_size=args.batch_size,
        span=args.target_span,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        auxiliary_operation=rehearsal_operation,
        auxiliary_span=args.rehearse_span or None,
        credit_mode=args.credit_mode,
    )

    shuffled = growth_runtime(seed=args.seed + 1, width=args.growth_width)
    _copy_parent_weights(parent, shuffled)
    _freeze_except(shuffled, ("growth_slots.0.",))
    torch.manual_seed(args.seed + 900_003)
    _, shuffled_progress = _train_with_progress(
        shuffled,
        operation=args.target_operation,
        updates=args.transfer_updates,
        batch_size=args.batch_size,
        span=args.target_span,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        shuffle_outcomes=True,
        auxiliary_operation=rehearsal_operation,
        auxiliary_span=args.rehearse_span or None,
        credit_mode=args.credit_mode,
    )

    bits_per_update = args.batch_size * (
        args.target_span + args.rehearse_span
    )

    transferred_stable_bits = _stable_bits(
        transferred_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    fresh_stable_bits = _stable_bits(
        fresh_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    parent_stable_bits = _stable_bits(
        parent_progress,
        threshold=args.parent_mastery_threshold,
        bits_per_update=args.batch_size * 2,
    )
    parent_extra_stable_bits = (
        _stable_bits(
            parent_extra_progress,
            threshold=args.parent_mastery_threshold,
            bits_per_update=args.batch_size * 4,
        )
        if args.parent_extra_updates
        else None
    )
    transfer_ratio = (
        float(fresh_stable_bits) / float(transferred_stable_bits)
        if transferred_stable_bits and fresh_stable_bits
        else None
    )
    retention_accuracy = _accuracy(
        transferred,
        operation="forward",
        count=args.audit_count,
        span=2,
        seed=args.seed + 2_000_001,
    )
    parent_extra_retention_accuracy = (
        _accuracy(
            transferred,
            operation=args.parent_extra_operation,
            count=args.audit_count,
            span=2,
            seed=args.seed + 2_000_002,
        )
        if args.parent_extra_updates
        else None
    )
    target_accuracy = _accuracy(
        transferred,
        operation=args.target_operation,
        count=args.audit_count,
        span=args.target_span,
        seed=args.seed + 2_000_002,
    )
    parent_target_accuracy = _accuracy(
        parent,
        operation=args.target_operation,
        count=args.audit_count,
        span=args.target_span,
        seed=args.seed + 2_000_003,
    )
    fresh_target_accuracy = _accuracy(
        fresh,
        operation=args.target_operation,
        count=args.audit_count,
        span=args.target_span,
        seed=args.seed + 2_000_002,
    )
    shuffled_target_accuracy = _accuracy(
        shuffled,
        operation=args.target_operation,
        count=args.audit_count,
        span=args.target_span,
        seed=args.seed + 2_000_002,
    )
    transfer_status = (
        "qualified"
        if (
            parent_stable_bits is not None
            and transferred_stable_bits is not None
            and fresh_stable_bits is not None
            and (
                not args.parent_extra_updates
                or parent_extra_stable_bits is not None
            )
            and transfer_ratio is not None
            and transfer_ratio > 1.0
        )
        else "unqualified"
    )
    report = {
        "schema": "neural-computer.frozen-core-transfer-report.v1",
        "claim_boundary": (
            "A previously mastered controller is reused as a frozen core while "
            "one external growth slot learns a new procedure from fresh rendered "
            "events, opaque actions, and scalar outcomes. The fresh baseline has "
            "the same architecture and data but all parameters trainable. This "
            "does not establish general continual learning or unrestricted growth."
        ),
        "seed": args.seed,
        "target_operation": args.target_operation,
        "target_span": args.target_span,
        "growth_width": args.growth_width,
        "dynamic_growth": args.dynamic_growth,
        "growth_gated": args.dynamic_growth,
        "credit_mode": args.credit_mode,
        "parent_updates": args.parent_updates,
        "parent_extra_operation": (
            args.parent_extra_operation if args.parent_extra_updates else None
        ),
        "parent_extra_updates": args.parent_extra_updates,
        "transfer_updates": args.transfer_updates,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "eval_every": args.eval_every,
        "mastery_threshold": args.mastery_threshold,
        "parent_mastery_threshold": args.parent_mastery_threshold,
        "learning_rate": args.learning_rate,
        "online_rehearsal": (
            None
            if args.rehearse_span == 0
            else {"operation": "forward", "span": args.rehearse_span}
        ),
        "replayed_examples": 0,
        "parent": {
            "stable_bits_to_threshold": parent_stable_bits,
            "heldout_accuracy": parent_accuracy,
            "target_accuracy_before_growth": parent_target_accuracy,
            "extra_stable_bits_to_threshold": parent_extra_stable_bits,
            "extra_heldout_accuracy": parent_extra_accuracy,
            "history": parent_history,
            "progress": parent_progress,
            "extra_history": parent_extra_history,
            "extra_progress": parent_extra_progress,
        },
        "transferred": {
            "stable_bits_to_threshold": transferred_stable_bits,
            "target_accuracy": target_accuracy,
            "retention_accuracy": retention_accuracy,
            "retention_delta": retention_accuracy - parent_accuracy,
            "extra_retention_accuracy": parent_extra_retention_accuracy,
            "extra_retention_delta": (
                None
                if parent_extra_retention_accuracy is None
                else parent_extra_retention_accuracy - (parent_extra_accuracy or 0.0)
            ),
            "trainable_parameter_count": sum(p.numel() for p in transferred.parameters() if p.requires_grad),
            "history": transferred_history,
            "progress": transferred_progress,
        },
        "fresh": {
            "stable_bits_to_threshold": fresh_stable_bits,
            "target_accuracy": fresh_target_accuracy,
            "trainable_parameter_count": sum(p.numel() for p in fresh.parameters() if p.requires_grad),
            "history": fresh_history,
            "progress": fresh_progress,
        },
        "reward_shuffled": {
            "target_accuracy": shuffled_target_accuracy,
            "target_gain_over_parent": (
                shuffled_target_accuracy - parent_target_accuracy
            ),
            "target_gap_below_transferred": target_accuracy
            - shuffled_target_accuracy,
            "progress": shuffled_progress,
        },
        "transfer_ratio_fresh_over_transferred": transfer_ratio,
        "transfer_status": transfer_status,
        "frozen_core_digest_before": transferred_digest_before,
        "frozen_core_digest_after": transferred_digest_after,
        "core_unchanged": transferred_digest_before == transferred_digest_after,
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.parent_extra_updates * args.batch_size * 4
                + args.transfer_updates * bits_per_update * 3
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.parent_extra_updates * args.batch_size * 2
                + args.transfer_updates
                * args.batch_size
                * (1 + bool(args.rehearse_span))
                * 3
            ),
            "optimizer_updates": (
                args.parent_updates + args.transfer_updates * 3
            ),
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_stable": parent_stable_bits is not None,
            "parent_extra_stable": (
                not args.parent_extra_updates
                or parent_extra_stable_bits is not None
            ),
            "transferred_stable": transferred_stable_bits is not None,
            "fresh_stable": fresh_stable_bits is not None,
            "positive_transfer": transfer_ratio is not None and transfer_ratio > 1.0,
            "target_mastered": target_accuracy >= args.mastery_threshold,
            "fresh_target_audited": fresh_target_accuracy >= 0.50,
            "parent_retained": retention_accuracy >= parent_accuracy - 0.02,
            "parent_extra_retained": (
                not args.parent_extra_updates
                or (
                    parent_extra_retention_accuracy is not None
                    and parent_extra_accuracy is not None
                    and parent_extra_retention_accuracy
                    >= parent_extra_accuracy - 0.02
                )
            ),
            "core_unchanged": transferred_digest_before == transferred_digest_after,
            "reward_shuffled_no_target_gain": (
                shuffled_target_accuracy <= parent_target_accuracy + 0.05
            ),
            "transferred_beats_shuffled": (
                target_accuracy >= shuffled_target_accuracy + 0.05
            ),
            "no_replayed_examples": True,
            "online_rehearsal_is_fresh": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=256)
    parser.add_argument("--parent-extra-updates", type=int, default=0)
    parser.add_argument("--parent-extra-operation", default="reverse")
    parser.add_argument("--transfer-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--growth-width", type=int, default=64)
    parser.add_argument(
        "--static-growth",
        action="store_false",
        dest="dynamic_growth",
        default=True,
        help="use the legacy per-step growth residual instead of recurrent external state",
    )
    parser.add_argument("--target-span", type=int, default=4)
    parser.add_argument("--target-operation", default="reverse")
    parser.add_argument(
        "--rehearse-span",
        type=int,
        default=0,
        help="fresh parent-task episodes interleaved with target acquisition; zero disables",
    )
    parser.add_argument(
        "--credit-mode",
        choices=("sampled", "paired_counterfactual"),
        default="sampled",
    )
    parser.add_argument("--mastery-threshold", type=float, default=0.75)
    parser.add_argument("--parent-mastery-threshold", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "transfer_status": report["transfer_status"],
                "transfer_ratio_fresh_over_transferred": report[
                    "transfer_ratio_fresh_over_transferred"
                ],
                "transferred_target_accuracy": report["transferred"]["target_accuracy"],
                "fresh_target_accuracy": report["fresh"]["target_accuracy"],
                "reward_shuffled_target_accuracy": report["reward_shuffled"][
                    "target_accuracy"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
