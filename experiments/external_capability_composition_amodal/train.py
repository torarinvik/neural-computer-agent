"""Pressure-test behavioral composition of independently learned programs.

Two external programs first learn separate primitives. Their recurrent state
and intention adapters are then frozen and serialized into a variable-length
pipeline. Only a fresh output decoder learns a novel composed target. The
controller, frontend, and parent output path remain frozen throughout the
composition phase.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.frozen_core_transfer_amodal.train import (
    _stable_bits,
    _train_with_progress,
)
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    ACTION_WIDTH,
    DECODER_HIDDEN,
    EVENT_WIDTH,
    INTENTION_WIDTH,
    _accuracy,
    _capability_accuracy,
    _new_capability,
    _train_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _feedback,
    _runtime,
)
from neural_computer import (
    ExternalCapabilityPipeline,
    ExternalCapabilityProgram,
    OpaqueProtocolDecoder,
    PersistentOpaqueStateStore,
    select_capability_candidate,
)

PRIMITIVES = (("complement", 4), ("reverse", 4))
COMPOSITE_OPERATION = "complement_reverse"


def _pipeline(
    programs: tuple[ExternalCapabilityProgram, ...],
) -> ExternalCapabilityPipeline:
    return ExternalCapabilityPipeline(programs)


def _empty_pipeline() -> ExternalCapabilityPipeline:
    return ExternalCapabilityPipeline(
        event_width=EVENT_WIDTH,
        action_width=ACTION_WIDTH,
        intention_width=INTENTION_WIDTH,
    )


def _zero_program(
    pipeline: ExternalCapabilityPipeline,
    index: int,
) -> ExternalCapabilityPipeline:
    if not 0 <= index < len(pipeline.programs):
        raise IndexError("pipeline program index is out of range")
    ablated = copy.deepcopy(pipeline)
    for parameter in ablated.programs[index].parameters():
        parameter.data.zero_()
    for parameter in ablated.parameters():
        parameter.requires_grad_(False)
    ablated.eval()
    return ablated


def _pipeline_artifact(
    pipeline: ExternalCapabilityPipeline,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in pipeline.state_dict().items()
    }


def _core_digest(runtime) -> str:
    digest = hashlib.sha256()
    for name, value in runtime.controller.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rollout_pipeline(
    parent,
    pipeline: ExternalCapabilityPipeline,
    decoder: OpaqueProtocolDecoder,
    batch,
    *,
    train: bool,
    shuffle_outcomes: bool = False,
    train_pipeline: bool = False,
) -> dict[str, torch.Tensor]:
    device = batch.input_frames.device
    state = parent.initial_state(batch.batch_size, device=device)
    pipeline_state = pipeline.initial_state(batch.batch_size, device=device)
    zeros = torch.zeros(batch.batch_size, device=device)
    previous_action = torch.zeros(batch.batch_size, ACTION_WIDTH, device=device)
    previous_reward = zeros
    previous_propensity = torch.ones(batch.batch_size, device=device)
    previous_has_feedback = zeros
    present = torch.ones(batch.batch_size, dtype=torch.bool, device=device)
    encoder = parent.encoders["vision"]
    quiet = _feedback(
        previous_action,
        previous_reward,
        previous_propensity,
        previous_has_feedback,
    )

    def tick(frame: torch.Tensor, feedback):
        nonlocal state, pipeline_state
        with torch.no_grad():
            event = encoder(frame)
            output, state = parent.step_streams({"vision": frame}, state, feedback)
        if train_pipeline:
            adapted, pipeline_state = pipeline.step(
                event=event,
                action=previous_action,
                outcome=previous_reward,
                intention=output.intention,
                state=pipeline_state,
                present=present,
            )
        else:
            with torch.no_grad():
                adapted, pipeline_state = pipeline.step(
                    event=event,
                    action=previous_action,
                    outcome=previous_reward,
                    intention=output.intention,
                    state=pipeline_state,
                    present=present,
                )
        return decoder(adapted)

    for frame in batch.input_frames.transpose(0, 1):
        if train_pipeline:
            tick(frame, quiet)
        else:
            with torch.no_grad():
                tick(frame, quiet)
    for frame in batch.distractor_frames.transpose(0, 1):
        if train_pipeline:
            tick(frame, quiet)
        else:
            with torch.no_grad():
                tick(frame, quiet)

    losses: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    for frame, correct in zip(
        batch.query_frames.transpose(0, 1),
        batch.correct_actions.transpose(0, 1),
        strict=True,
    ):
        feedback = _feedback(
            previous_action,
            previous_reward,
            previous_propensity,
            previous_has_feedback,
        )
        logits = tick(frame, feedback)
        probabilities = torch.softmax(logits, dim=-1)
        if train:
            action = torch.multinomial(probabilities * 0.9 + 0.05, 1).squeeze(1)
        else:
            action = logits.argmax(dim=-1)
        reward = (action == correct).to(logits.dtype)
        delivered = reward.roll(1) if shuffle_outcomes else reward
        selected = logits.gather(1, action.unsqueeze(1)).squeeze(1)
        losses.append(F.binary_cross_entropy_with_logits(selected, delivered))
        rewards.append(reward)
        previous_action = F.one_hot(action, ACTION_WIDTH).to(logits.dtype)
        previous_reward = delivered
        previous_propensity = (
            probabilities.gather(
                1,
                action.unsqueeze(1),
            )
            .squeeze(1)
            .detach()
        )
        previous_has_feedback = torch.ones_like(previous_reward)
    return {"loss": torch.stack(losses).mean(), "rewards": torch.stack(rewards, dim=1)}


def _train_composition(
    parent,
    pipeline: ExternalCapabilityPipeline,
    decoder: OpaqueProtocolDecoder,
    *,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    audit_count: int,
    eval_every: int,
    learning_rate: float,
    train_pipeline: bool,
    shuffle_outcomes: bool = False,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    trainable = list(decoder.parameters())
    if train_pipeline:
        trainable.extend(pipeline.parameters())
    if not trainable:
        raise RuntimeError("composition learner has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    progress: list[dict[str, float | int]] = []
    pipeline.train(train_pipeline)
    decoder.train()
    for update in range(1, updates + 1):
        batch = generate_sequence_memory_batch(
            batch_size,
            span=span,
            distractors=1,
            seed=seed + update * 10_007,
            operation=COMPOSITE_OPERATION,
        )
        result = _rollout_pipeline(
            parent,
            pipeline,
            decoder,
            batch,
            train=True,
            shuffle_outcomes=shuffle_outcomes,
            train_pipeline=train_pipeline,
        )
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "unique_logical_lifetimes": update * batch_size,
                "unique_verifier_bits": update * batch_size * span,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
            }
        )
        if update == updates or (eval_every > 0 and update % eval_every == 0):
            pipeline.eval()
            decoder.eval()
            heldout = generate_sequence_memory_batch(
                audit_count,
                span=span,
                distractors=1,
                seed=seed + 1_000_000 + update,
                operation=COMPOSITE_OPERATION,
            )
            progress.append(
                {
                    "update": update,
                    "unique_verifier_bits": update * batch_size * span,
                    "heldout_accuracy": float(
                        _rollout_pipeline(
                            parent,
                            pipeline,
                            decoder,
                            heldout,
                            train=False,
                        )["rewards"].mean()
                    ),
                }
            )
            pipeline.train(train_pipeline)
            decoder.train()
    pipeline.eval()
    decoder.eval()
    return history, progress


@torch.no_grad()
def _composition_accuracy(
    parent,
    pipeline: ExternalCapabilityPipeline,
    decoder: OpaqueProtocolDecoder,
    *,
    count: int,
    seed: int,
) -> float:
    batch = generate_sequence_memory_batch(
        count,
        span=4,
        distractors=1,
        seed=seed,
        operation=COMPOSITE_OPERATION,
    )
    return float(
        _rollout_pipeline(
            parent,
            pipeline,
            decoder,
            batch,
            train=False,
        )["rewards"].mean()
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if (
        min(
            args.parent_updates,
            args.primitive_updates,
            args.composition_updates,
            args.batch_size,
            args.audit_count,
            args.eval_every,
        )
        < 1
    ):
        raise ValueError("all update and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    _, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
    )
    parent.eval()
    parent_digest_before = _core_digest(parent)
    parent_accuracy = _accuracy(
        parent,
        operation="forward",
        count=args.audit_count,
        span=2,
        seed=args.seed + 20_000,
    )

    primitive_programs: list[ExternalCapabilityProgram] = []
    primitive_decoders: list[OpaqueProtocolDecoder] = []
    primitive_reports: dict[str, object] = {}
    for index, (operation, span) in enumerate(PRIMITIVES):
        program, decoder = _new_capability(args.seed + index + 1)
        history, progress = _train_capability(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            updates=args.primitive_updates,
            batch_size=args.batch_size,
            seed=args.seed + 1_000 * (index + 1),
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
        )
        primitive_programs.append(program)
        primitive_decoders.append(decoder)
        primitive_reports[operation] = {
            "stable_bits_to_threshold": _stable_bits(
                progress,
                threshold=args.mastery_threshold,
                bits_per_update=args.batch_size * span,
            ),
            "accuracy": _capability_accuracy(
                parent,
                program,
                decoder,
                operation=operation,
                span=span,
                count=args.audit_count,
                seed=args.seed + 30_000 + index,
            ),
            "history": history,
            "progress": progress,
        }

    composed_pipeline = _pipeline(tuple(primitive_programs))
    for parameter in composed_pipeline.parameters():
        parameter.requires_grad_(False)
    composed_decoder = OpaqueProtocolDecoder(
        INTENTION_WIDTH,
        ACTION_WIDTH,
        hidden=DECODER_HIDDEN,
    )
    initial_decoder_state = copy.deepcopy(composed_decoder.state_dict())
    torch.manual_seed(args.seed + 90_001)
    composed_history, composed_progress = _train_composition(
        parent,
        composed_pipeline,
        composed_decoder,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=False,
    )

    blank_pipeline = _empty_pipeline()
    blank_decoder = OpaqueProtocolDecoder(
        INTENTION_WIDTH,
        ACTION_WIDTH,
        hidden=DECODER_HIDDEN,
    )
    blank_decoder.load_state_dict(initial_decoder_state)
    torch.manual_seed(args.seed + 90_002)
    blank_history, blank_progress = _train_composition(
        parent,
        blank_pipeline,
        blank_decoder,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=False,
    )

    fresh_pipeline = _pipeline(
        tuple(
            _new_capability(args.seed + 10 + index)[0]
            for index in range(len(PRIMITIVES))
        )
    )
    fresh_decoder = OpaqueProtocolDecoder(
        INTENTION_WIDTH,
        ACTION_WIDTH,
        hidden=DECODER_HIDDEN,
    )
    torch.manual_seed(args.seed + 90_003)
    fresh_history, fresh_progress = _train_composition(
        parent,
        fresh_pipeline,
        fresh_decoder,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=True,
    )

    shuffled_pipeline = copy.deepcopy(composed_pipeline)
    shuffled_decoder = OpaqueProtocolDecoder(
        INTENTION_WIDTH,
        ACTION_WIDTH,
        hidden=DECODER_HIDDEN,
    )
    shuffled_decoder.load_state_dict(initial_decoder_state)
    torch.manual_seed(args.seed + 90_004)
    _, shuffled_progress = _train_composition(
        parent,
        shuffled_pipeline,
        shuffled_decoder,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=4,
        seed=args.seed + 40_000,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        train_pipeline=False,
        shuffle_outcomes=True,
    )

    target_accuracy = _composition_accuracy(
        parent,
        composed_pipeline,
        composed_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    blank_accuracy = _composition_accuracy(
        parent,
        blank_pipeline,
        blank_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    fresh_accuracy = _composition_accuracy(
        parent,
        fresh_pipeline,
        fresh_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    shuffled_accuracy = _composition_accuracy(
        parent,
        shuffled_pipeline,
        shuffled_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    zero_first_accuracy = _composition_accuracy(
        parent,
        _zero_program(composed_pipeline, 0),
        composed_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    zero_second_accuracy = _composition_accuracy(
        parent,
        _zero_program(composed_pipeline, 1),
        composed_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )

    persistence_dir = args.report_out.parent / "composition_state"
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)
    pipeline_store = PersistentOpaqueStateStore(
        persistence_dir / "pipeline.pt",
        configuration=composed_pipeline.configuration(),
    )
    decoder_store = PersistentOpaqueStateStore(
        persistence_dir / "decoder.pt",
        configuration={
            "component": "composition-decoder",
            "schema": "neural-computer.opaque-protocol-decoder.v1",
            "intention_width": INTENTION_WIDTH,
            "action_width": ACTION_WIDTH,
            "hidden": DECODER_HIDDEN,
        },
    )
    pipeline_digest = pipeline_store.save_module(composed_pipeline)
    decoder_digest = decoder_store.save_module(composed_decoder)
    reloaded_pipeline = _pipeline(
        tuple(
            _new_capability(args.seed + 20 + index)[0]
            for index in range(len(PRIMITIVES))
        )
    )
    reloaded_decoder = OpaqueProtocolDecoder(
        INTENTION_WIDTH,
        ACTION_WIDTH,
        hidden=DECODER_HIDDEN,
    )
    reloaded_pipeline_digest = pipeline_store.load_module(reloaded_pipeline)
    reloaded_decoder_digest = decoder_store.load_module(reloaded_decoder)
    reloaded_accuracy = _composition_accuracy(
        parent,
        reloaded_pipeline,
        reloaded_decoder,
        count=args.audit_count,
        seed=args.seed + 50_001,
    )
    pipeline_path = persistence_dir / "pipeline.pt"
    intact_pipeline = pipeline_path.read_bytes()
    corrupted_payload = torch.load(pipeline_path, weights_only=False)
    corrupted_state = dict(corrupted_payload["state_dict"])
    first_name = next(iter(corrupted_state))
    corrupted_value = corrupted_state[first_name].clone()
    corrupted_value.reshape(-1)[0] += 1.0
    corrupted_state[first_name] = corrupted_value
    corrupted_payload["state_dict"] = corrupted_state
    torch.save(corrupted_payload, pipeline_path)
    corruption_rejected = False
    try:
        pipeline_store.load()
    except ValueError as error:
        corruption_rejected = "checksum mismatch" in str(error)
    pipeline_path.write_bytes(intact_pipeline)

    parent_digest_after = _core_digest(parent)
    bits_per_update = args.batch_size * 4
    composed_stable = _stable_bits(
        composed_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    blank_stable = _stable_bits(
        blank_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    fresh_stable = _stable_bits(
        fresh_progress,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update,
    )
    transfer_ratio = (
        float(fresh_stable) / float(composed_stable)
        if fresh_stable and composed_stable
        else None
    )
    candidate_selection = select_capability_candidate(
        (
            tuple(float(row["heldout_accuracy"]) for row in composed_progress),
            tuple(float(row["heldout_accuracy"]) for row in fresh_progress),
        ),
        threshold=args.mastery_threshold,
        bits_per_observation=args.eval_every * bits_per_update,
    )
    report: dict[str, object] = {
        "schema": "neural-computer.external-capability-composition-report.v1",
        "claim_boundary": (
            "Two independently learned external primitive programs are frozen "
            "and serially composed before a fresh decoder learns a novel "
            "composite target. This tests reusable representation composition; "
            "it is not arbitrary program induction or general continual learning."
        ),
        "seed": args.seed,
        "primitives": [{"operation": op, "span": span} for op, span in PRIMITIVES],
        "composite_operation": COMPOSITE_OPERATION,
        "parent_accuracy": parent_accuracy,
        "primitive_reports": primitive_reports,
        "composition": {
            "stable_bits_to_threshold": composed_stable,
            "target_accuracy": target_accuracy,
            "history": composed_history,
            "progress": composed_progress,
        },
        "blank_control": {
            "stable_bits_to_threshold": blank_stable,
            "target_accuracy": blank_accuracy,
            "history": blank_history,
            "progress": blank_progress,
        },
        "fresh_pipeline": {
            "stable_bits_to_threshold": fresh_stable,
            "target_accuracy": fresh_accuracy,
            "history": fresh_history,
            "progress": fresh_progress,
        },
        "reward_shuffled": {
            "target_accuracy": shuffled_accuracy,
            "progress": shuffled_progress,
        },
        "ablations": {
            "zero_first_program_accuracy": zero_first_accuracy,
            "zero_second_program_accuracy": zero_second_accuracy,
        },
        "transfer_ratio_fresh_over_composed": transfer_ratio,
        "candidate_selection": {
            "accepted": candidate_selection.accepted,
            "selected_index": candidate_selection.selected_index,
            "stable_bits_to_threshold": candidate_selection.stable_bits_to_threshold,
            "reason": candidate_selection.reason,
        },
        "persistence": {
            "pipeline_digest": pipeline_digest,
            "reloaded_pipeline_digest": reloaded_pipeline_digest,
            "decoder_digest": decoder_digest,
            "reloaded_decoder_digest": reloaded_decoder_digest,
            "reload_exact": pipeline_digest == reloaded_pipeline_digest
            and decoder_digest == reloaded_decoder_digest,
            "reloaded_accuracy": reloaded_accuracy,
            "corruption_rejected": corruption_rejected,
        },
        "frozen_core": {
            "digest_before": parent_digest_before,
            "digest_after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.primitive_updates
                * args.batch_size
                * sum(span + 2 for _, span in PRIMITIVES)
                + args.composition_updates * bits_per_update * 4
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.primitive_updates * args.batch_size * len(PRIMITIVES) * 2
                + args.composition_updates * args.batch_size * 4
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.primitive_updates * len(PRIMITIVES)
                + args.composition_updates * 4
            ),
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_mastered": bool(parent_progress)
            and parent_progress[-1]["heldout_accuracy"] >= args.mastery_threshold,
            "primitives_mastered": all(
                item["stable_bits_to_threshold"] is not None
                for item in primitive_reports.values()
            ),
            "composition_mastered": composed_stable is not None
            and target_accuracy >= args.mastery_threshold,
            "composed_beats_blank": target_accuracy >= blank_accuracy + 0.05,
            "fresh_pipeline_audited": fresh_accuracy >= 0.50,
            "reward_shuffled_no_gain": shuffled_accuracy <= blank_accuracy + 0.05,
            "first_program_is_causal": zero_first_accuracy < target_accuracy - 0.05,
            "second_program_is_causal": zero_second_accuracy < target_accuracy - 0.05,
            "positive_transfer": transfer_ratio is not None and transfer_ratio > 1.0,
            "candidate_selector_accepted": candidate_selection.accepted,
            "reload_exact": pipeline_digest == reloaded_pipeline_digest
            and decoder_digest == reloaded_decoder_digest,
            "reload_behavior_preserved": reloaded_accuracy >= target_accuracy - 0.05,
            "corruption_rejected": corruption_rejected,
            "frozen_core": parent_digest_before == parent_digest_after,
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--primitive-updates", type=int, default=256)
    parser.add_argument("--composition-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "composition_target_accuracy": report["composition"]["target_accuracy"],
                "blank_target_accuracy": report["blank_control"]["target_accuracy"],
                "fresh_target_accuracy": report["fresh_pipeline"]["target_accuracy"],
                "reward_shuffled_target_accuracy": report["reward_shuffled"][
                    "target_accuracy"
                ],
                "transfer_ratio_fresh_over_composed": report[
                    "transfer_ratio_fresh_over_composed"
                ],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
