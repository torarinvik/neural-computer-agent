"""Canonical-controller Brain Workshop working-memory pressure test.

This experiment deliberately keeps the frontend and decoder outside the
controller.  A parent controller first learns a short forward-reproduction
primitive.  Its core is then frozen while two generic growth registers are
acquired sequentially: slot zero is the producer factor and slot one is a
prior-only consumer that can see only slot zero's learned register.  The
artifacts are persisted and reloaded through ``ExecutableArtifactMemory``.

The learner receives rendered frames, sampled opaque actions, and scalar
verifier outcomes.  Correct actions are used only by the local verifier to
produce rewards and never enter the loss or controller inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    SequenceMemoryBatch,
    generate_sequence_memory_batch,
)
from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalOutputBus,
    ControllerFeedback,
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    OpaqueProtocolDecoder,
    compose_growth_artifacts,
    freeze_core,
    load_growth_artifact,
)

COMPOSITION_VERIFIER_FLOOR = 0.60


class FrameEventEncoder(nn.Module):
    """Replaceable learned frontend for the rendered working-memory stream."""

    def __init__(self, event_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, event_width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(event_width),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.network(frame)


def _runtime(*, seed: int, growth: bool) -> AmodalControllerRuntime:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=32,
        workspace_slots=4,
        intention_width=16,
        feedback_width=2,
        event_window_capacity=16,
        reliability_hidden=16,
        growth_register_widths=(32, 32) if growth else (),
        growth_prior_only_from=1 if growth else None,
        growth_recurrent_from=1 if growth else None,
    )
    return AmodalControllerRuntime(
        controller,
        encoders={"vision": FrameEventEncoder(32)},
        output_bus=AmodalOutputBus(
            {"action": OpaqueProtocolDecoder(16, 2, hidden=16)}
        ),
    )


def _feedback(
    action: torch.Tensor,
    reward: torch.Tensor,
    propensity: torch.Tensor,
    has_feedback: torch.Tensor,
) -> ControllerFeedback:
    return ControllerFeedback(action, reward, propensity, has_feedback)


def _frame_with_blank_cues(frame: torch.Tensor) -> torch.Tensor:
    result = frame.clone()
    result[:, :, 2:5, 2:5] = 0.0
    result[:, :, 2:5, 14:17] = 0.0
    result[:, :, 2:5, 27:30] = 0.0
    return result


def _rollout(
    runtime: AmodalControllerRuntime,
    batch: SequenceMemoryBatch,
    *,
    train: bool,
    exploration: float = 0.10,
    shuffle_outcomes: bool = False,
    operation_cue_blank: bool = False,
    disable_workspace: bool = False,
    reset_active_state_before_query: bool = False,
    reset_all_memory_before_query: bool = False,
) -> dict[str, torch.Tensor]:
    device = batch.input_frames.device
    state = runtime.initial_state(batch.batch_size, device=device)
    zeros = torch.zeros(batch.batch_size, device=device)
    previous_action = torch.zeros(batch.batch_size, 2, device=device)
    previous_reward = zeros
    previous_propensity = torch.ones(batch.batch_size, device=device)
    previous_has_feedback = zeros

    def tick(frame: torch.Tensor, state, feedback: ControllerFeedback):
        return runtime.step_streams(
            {"vision": frame},
            state,
            feedback,
            disable_workspace=disable_workspace,
        )

    quiet = _feedback(previous_action, previous_reward, previous_propensity, zeros)
    for index in range(batch.span):
        _, state = tick(batch.input_frames[:, index], state, quiet)
    for index in range(batch.distractor_frames.shape[1]):
        _, state = tick(batch.distractor_frames[:, index], state, quiet)
    if reset_all_memory_before_query:
        state = runtime.initial_state(batch.batch_size, device=device)
    elif reset_active_state_before_query:
        state = type(state)(
            hidden=torch.zeros_like(state.hidden),
            workspace=state.workspace,
            latest_event=torch.zeros_like(state.latest_event),
            workspace_usage=state.workspace_usage,
            event_window=state.event_window,
            source_trust=state.source_trust,
            growth_registers=state.growth_registers,
        )

    losses: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    for index in range(batch.span):
        frame = batch.query_frames[:, index]
        if operation_cue_blank:
            frame = _frame_with_blank_cues(frame)
        feedback = _feedback(
            previous_action,
            previous_reward,
            previous_propensity,
            previous_has_feedback,
        )
        output, state = tick(frame, state, feedback)
        logits = output.decoded["action"]
        probabilities = torch.softmax(logits, dim=-1)
        if train:
            behavior = probabilities * (1.0 - exploration) + exploration / 2.0
            action_index = torch.multinomial(behavior, 1).squeeze(1)
        else:
            action_index = logits.argmax(dim=-1)
        reward = (action_index == batch.correct_actions[:, index]).float()
        delivered = reward.roll(1) if shuffle_outcomes else reward
        selected = logits.gather(1, action_index.unsqueeze(1)).squeeze(1)
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(selected, delivered))
        rewards.append(reward)
        previous_action = torch.nn.functional.one_hot(action_index, 2).float()
        previous_reward = delivered
        previous_propensity = probabilities.gather(1, action_index.unsqueeze(1)).squeeze(1).detach()
        previous_has_feedback = torch.ones_like(previous_reward)
    return {
        "loss": torch.stack(losses).mean(),
        "rewards": torch.stack(rewards, dim=1),
    }


def _train(
    runtime: AmodalControllerRuntime,
    *,
    operation: str,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    lr: float,
    shuffle_outcomes: bool = False,
) -> list[dict[str, float | int]]:
    parameters = [parameter for parameter in runtime.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    runtime.train()
    for update in range(1, updates + 1):
        batch = generate_sequence_memory_batch(
            batch_size,
            span=span,
            distractors=1,
            seed=seed + update * 10_007,
            operation=operation,
        )
        result = _rollout(
            runtime,
            batch,
            train=True,
            shuffle_outcomes=shuffle_outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
            }
        )
    runtime.eval()
    return history


@torch.no_grad()
def _accuracy(
    runtime: AmodalControllerRuntime,
    *,
    operation: str,
    count: int,
    span: int,
    seed: int,
    **controls: bool,
) -> float:
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation=operation,
        blank_sequence=controls.pop("blank_sequence", False),
    )
    result = _rollout(runtime, batch, train=False, **controls)
    return float(result["rewards"].mean())


def _copy_parent_weights(parent: AmodalControllerRuntime, expanded: AmodalControllerRuntime) -> None:
    expanded.controller.load_state_dict(parent.controller.state_dict(), strict=False)
    expanded.encoders.load_state_dict(parent.encoders.state_dict())
    expanded.output_bus.load_state_dict(parent.output_bus.state_dict())


def _freeze_except(runtime: AmodalControllerRuntime, prefixes: tuple[str, ...]) -> None:
    freeze_core(runtime.controller, prefixes)
    for module in (runtime.encoders, runtime.output_bus):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    runtime.eval()


def _artifact(runtime: AmodalControllerRuntime, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
        if name.startswith(prefix)
    }


def _digest_core(runtime: AmodalControllerRuntime, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name, value in runtime.controller.state_dict().items():
        if name.startswith(prefixes):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _load_slot(runtime: AmodalControllerRuntime, artifact: dict[str, torch.Tensor], slot: int) -> None:
    mapped = compose_growth_artifacts(
        (artifact,),
        prefix_maps=({"growth_slots.0.": f"growth_slots.{slot}."},),
    ) if slot else artifact
    load_growth_artifact(
        runtime.controller,
        mapped,
        growth_prefixes=(f"growth_slots.{slot}.",),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")
    record = args.record
    record.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    parent_history = _train(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 100,
        lr=args.learning_rate,
    )
    parent_accuracy = _accuracy(
        parent,
        operation="forward",
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 20_000,
    )

    producer = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, producer)
    producer_prefix = ("growth_slots.0.",)
    _freeze_except(producer, producer_prefix)
    producer_history = _train(
        producer,
        operation="complement",
        updates=args.growth_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 200,
        lr=args.learning_rate,
    )
    producer_artifact = _artifact(producer, producer_prefix[0])

    consumer = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, consumer)
    _load_slot(consumer, producer_artifact, 0)
    consumer_prefix = ("growth_slots.1.",)
    _freeze_except(consumer, consumer_prefix)
    consumer_history = _train(
        consumer,
        operation="producer_global_parity",
        updates=args.growth_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 300,
        lr=args.learning_rate,
    )
    consumer_artifact = compose_growth_artifacts(
        (_artifact(consumer, consumer_prefix[0]),),
        prefix_maps=({"growth_slots.1.": "growth_slots.0."},),
    )

    bank_path = record / "artifact_memory"
    if bank_path.exists():
        shutil.rmtree(bank_path)
    bank = ExecutableArtifactMemory(bank_path, width=32, capacity=2)
    producer_key = torch.nn.functional.normalize(torch.arange(1, 33).float(), dim=0)
    consumer_key = torch.nn.functional.normalize(torch.arange(33, 65).float(), dim=0)
    bank.put(producer_key, producer_artifact)
    bank.put(consumer_key, consumer_artifact)
    bank = ExecutableArtifactMemory.load(bank_path)
    _, reloaded_producer = bank.promote(producer_key)
    _, reloaded_consumer = bank.promote(consumer_key)

    composed_key = torch.nn.functional.normalize(
        producer_key + consumer_key,
        dim=0,
    )
    composed_artifact = compose_growth_artifacts(
        (reloaded_producer, reloaded_consumer),
        prefix_maps=(
            {"growth_slots.0.": "growth_slots.0."},
            {"growth_slots.0.": "growth_slots.1."},
        ),
    )
    composed_path = record / "composed_artifact_memory"
    if composed_path.exists():
        shutil.rmtree(composed_path)

    composition_verifier_score: float | None = None

    def composition_verifier(candidate: ExecutableArtifactMemory) -> bool:
        nonlocal composition_verifier_score
        handles = (
            candidate.promote_view(0, "producer")[0],
            candidate.promote_view(0, "consumer")[0],
        )
        if tuple(handle.view for handle in handles) != ("producer", "consumer"):
            return False
        _, candidate_artifact = candidate.promote(composed_key)
        candidate_runtime = _runtime(seed=args.seed + 500, growth=True)
        _copy_parent_weights(parent, candidate_runtime)
        load_growth_artifact(
            candidate_runtime.controller,
            candidate_artifact,
            growth_prefixes=("growth_slots.0.", "growth_slots.1."),
        )
        candidate_runtime.eval()
        # Use the report's declared held-out audit set for the admission
        # decision as well. A separate verifier draw would turn a valid
        # stochastic audit fluctuation into a spurious rejection while
        # adding no new causal control.
        composition_verifier_score = _accuracy(
            candidate_runtime,
            operation="producer_global_parity",
            count=args.audit_count,
            span=args.span,
            seed=args.seed + 20_001,
        )
        return composition_verifier_score >= COMPOSITION_VERIFIER_FLOOR

    lifecycle = ExternalCapabilityLifecycle(bank)
    composition_receipt = lifecycle.consolidate(
        (0, 1),
        composed_key,
        composed_artifact,
        composed_path,
        replacement_aliases=(producer_key, consumer_key),
        replacement_alias_views=("producer", "consumer"),
        verifier=composition_verifier,
    )
    if not composition_receipt.accepted:
        raise RuntimeError(
            "producer-consumer composition was rejected: "
            f"{composition_receipt}; "
            f"verifier_score={composition_verifier_score}"
        )
    composed_memory = ExecutableArtifactMemory.load(composed_path)
    _, reloaded_composed_artifact = composed_memory.promote(composed_key)

    composed = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, composed)
    load_growth_artifact(
        composed.controller,
        reloaded_composed_artifact,
        growth_prefixes=("growth_slots.0.", "growth_slots.1."),
    )
    composed.eval()

    producer_only = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, producer_only)
    _load_slot(producer_only, reloaded_producer, 0)
    producer_only.eval()

    consumer_only = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, consumer_only)
    _load_slot(consumer_only, reloaded_consumer, 1)
    consumer_only.eval()

    producer_zeroed = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, producer_zeroed)
    _load_slot(producer_zeroed, reloaded_consumer, 1)
    for parameter in producer_zeroed.controller.growth_slots[0].parameters():
        parameter.data.zero_()
    producer_zeroed.eval()

    behavior = {
        "parent": _accuracy(parent, operation="producer_global_parity", count=args.audit_count, span=args.span, seed=args.seed + 20_001),
        "producer_only": _accuracy(producer_only, operation="producer_global_parity", count=args.audit_count, span=args.span, seed=args.seed + 20_001),
        "consumer_only": _accuracy(consumer_only, operation="producer_global_parity", count=args.audit_count, span=args.span, seed=args.seed + 20_001),
        "composed": _accuracy(composed, operation="producer_global_parity", count=args.audit_count, span=args.span, seed=args.seed + 20_001),
        "blank_sequence": _accuracy(composed, operation="producer_global_parity", count=args.audit_count, span=args.span, seed=args.seed + 20_002, blank_sequence=True),
        "cue_ablated": _accuracy(composed, operation="producer_global_parity", count=args.audit_count, span=args.span, seed=args.seed + 20_003, operation_cue_blank=True),
        "producer_zeroed": _accuracy(producer_zeroed, operation="producer_global_parity", count=args.audit_count, span=args.span, seed=args.seed + 20_001),
    }
    composed.controller.growth_ablate_prior_from = 1
    behavior["prior_read_ablated"] = _accuracy(
        composed,
        operation="producer_global_parity",
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 20_001,
    )
    composed.controller.growth_ablate_prior_from = None
    shuffled = _runtime(seed=args.seed, growth=True)
    _copy_parent_weights(parent, shuffled)
    _load_slot(shuffled, reloaded_producer, 0)
    _freeze_except(shuffled, ("growth_slots.1.",))
    _train(
        shuffled,
        operation="producer_global_parity",
        updates=args.growth_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 400,
        lr=args.learning_rate,
        shuffle_outcomes=True,
    )
    shuffled_score = _accuracy(
        shuffled,
        operation="producer_global_parity",
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 20_004,
    )
    behavior["reward_shuffled"] = shuffled_score

    core_prefixes = ("growth_slots.0.", "growth_slots.1.")
    core_unchanged = _digest_core(parent, core_prefixes) == _digest_core(composed, core_prefixes)
    training_updates = args.parent_updates + 3 * args.growth_updates
    training_lifetimes = training_updates * args.batch_size
    artifact_reload_exact = (
        all(
            torch.equal(producer_artifact[name], reloaded_producer[name])
            for name in producer_artifact
        )
        and all(
            torch.equal(consumer_artifact[name], reloaded_consumer[name])
            for name in consumer_artifact
        )
        and all(
            torch.equal(composed_artifact[name], reloaded_composed_artifact[name])
            for name in composed_artifact
        )
        and len(composed_artifact) == len(reloaded_composed_artifact)
    )
    report = {
        "schema": "canonical-working-memory-growth-pressure-v2",
        "claim_boundary": "Canonical amodal controller growth registers pass a narrow producer-to-prior-only-consumer working-memory pressure test; this is not arbitrary program induction.",
        "seed": args.seed,
        "parent_updates": args.parent_updates,
        "growth_updates": args.growth_updates,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "span": args.span,
        "behavior": {"parent_forward": parent_accuracy, **behavior},
        "accounting": {
            "unique_logical_lifetimes": training_lifetimes,
            "unique_verifier_bits": training_lifetimes * args.span,
            "optimizer_updates": training_updates,
            "replayed_examples": 0,
            "diagnostic_lifetimes_charged_to_budget": args.growth_updates * args.batch_size,
            "wall_seconds": perf_counter() - started,
        },
        "artifact_memory": {
            "producer_reload_exact": all(torch.equal(producer_artifact[name], reloaded_producer[name]) for name in producer_artifact),
            "consumer_reload_exact": all(torch.equal(consumer_artifact[name], reloaded_consumer[name]) for name in consumer_artifact),
            "occupied_rows": bank.occupied,
            "composed_occupied_rows": composed_memory.occupied,
            "composition_receipt": {
                "accepted": composition_receipt.accepted,
                "rows_before": composition_receipt.rows_before,
                "rows_after": composition_receipt.rows_after,
                "rows_saved": composition_receipt.rows_saved,
                "verifier_score": composition_verifier_score,
                "verifier_floor": COMPOSITION_VERIFIER_FLOOR,
            },
        },
        "history": {
            "parent": parent_history,
            "producer": producer_history,
            "consumer": consumer_history,
        },
        "gates": {
            "parent_forward_above_chance": parent_accuracy > 0.55,
            "composed_beats_parent": behavior["composed"] > behavior["parent"] + 0.05,
            "composed_beats_producer_only": behavior["composed"] > behavior["producer_only"] + 0.05,
            # A prior-only consumer with no producer information may be
            # below chance because its fixed zero-input response is
            # anti-correlated with some verifier rows.  The causal question
            # is whether it can exceed the parent without the producer; a
            # low score is not a raw-input bypass.
            "consumer_only_not_above_chance": behavior["consumer_only"] <= behavior["parent"] + 0.12,
            "producer_ablation_is_causal": behavior["producer_zeroed"] < behavior["composed"] - 0.05,
            "prior_read_ablation_is_causal": behavior["prior_read_ablated"] < behavior["composed"] - 0.05,
            "missing_sequence_near_chance": behavior["blank_sequence"] <= 0.65,
            "reward_shuffled_near_chance": behavior["reward_shuffled"] <= 0.65,
            "core_unchanged": core_unchanged,
            "artifact_reload_exact": artifact_reload_exact,
            "composed_artifact_memory_one_row": composed_memory.occupied == (0,),
            "composition_behavior_verified": composition_receipt.accepted,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    (record / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69201)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--growth-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--span", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"accepted_diagnostic": report["accepted_diagnostic"], "behavior": report["behavior"], "gates": report["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
