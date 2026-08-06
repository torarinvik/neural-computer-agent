"""Measure the transition from short-term retention to working memory.

The learner sees only RGB events, its own opaque binary actions, and scalar
success.  A lifetime presents a short sequence, optional irrelevant sensory
events, and then asks for either the original or reversed sequence.  The
verifier-private sequence and operation are used only to score attempted
actions.

Forward reproduction is the short-term-memory control.  Conditional reversal
requires the same retained content plus an operation over its temporal order.
All tensors remain on the selected device, so the controller's recurrent state
and differentiable workspace are literal RAM/VRAM-resident fast memory.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from .environment import ACTIONS, IMAGE_SIZE, NULL_ACTION
from .legacy_model import ControllerState, UnifiedCognitiveController
from .train import attempted_success_loss, seed_everything


def _attempted_reinforce_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor, *, baseline: float = 0.5) -> torch.Tensor:
    """Policy-gradient loss using only the sampled action and scalar reward."""
    log_probability = torch.log_softmax(logits, dim=-1).gather(
        1, attempted.unsqueeze(1)).squeeze(1)
    advantage = outcomes.detach() - baseline
    return -(advantage * log_probability).mean()


def _successful_action_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor) -> torch.Tensor:
    """Imitate only actions the verifier explicitly rewarded."""
    successful = outcomes > 0.5
    if not bool(successful.any()):
        return logits.sum() * 0.0
    return torch.nn.functional.cross_entropy(
        logits[successful], attempted[successful])


@dataclass(frozen=True)
class SequenceMemoryBatch:
    """Public sensory stream and verifier-private deterministic answers."""

    input_frames: torch.Tensor
    distractor_frames: torch.Tensor
    query_frames: torch.Tensor
    correct_actions: torch.Tensor
    sequence: torch.Tensor
    operation_bits: torch.Tensor
    seeds: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_frames.shape[0])

    @property
    def span(self) -> int:
        return int(self.input_frames.shape[1])


_GENERATED_PRIMITIVE_COLUMNS = {
    "forward": 2,
    "reverse": 27,
    "complement": 14,
    "rotate": 9,
}
_GENERATED_COMPOSITIONS = (
    ("reverse", "complement"),
    ("complement", "reverse"),
    ("rotate", "complement"),
    ("complement", "rotate"),
    ("reverse", "rotate"),
    ("rotate", "reverse"),
)


def _apply_generated_primitive(
    sequence: torch.Tensor,
    primitive: str,
) -> torch.Tensor:
    """Apply one verifier-private primitive to binary sequence rows."""

    if primitive == "forward":
        return sequence
    if primitive == "reverse":
        return sequence.flip(1)
    if primitive == "complement":
        return 1 - sequence
    if primitive == "rotate":
        return sequence.roll(shifts=-1, dims=1)
    raise ValueError(f"unknown generated primitive: {primitive}")


def _apply_generated_compositions(
    sequence: torch.Tensor,
    composition_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply one sampled two-primitive program independently to each row."""

    return torch.stack(
        tuple(
            _apply_generated_primitive(
                _apply_generated_primitive(
                    sequence[row : row + 1],
                    _GENERATED_COMPOSITIONS[int(composition_ids[row])][0],
                ),
                _GENERATED_COMPOSITIONS[int(composition_ids[row])][1],
            )[0]
            for row in range(sequence.shape[0])
        )
    )


def _balanced_binary_sequences(
        count: int, span: int, generator: torch.Generator) -> torch.Tensor:
    """Cover every binary sequence evenly before deterministic shuffling."""
    patterns = 2 ** span
    ids = torch.arange(count) % patterns
    sequence = torch.stack(
        [((ids >> shift) & 1) for shift in reversed(range(span))], dim=1)
    return sequence[torch.randperm(count, generator=generator)]


def _identity_masks(
        positions: torch.Tensor, identities: torch.Tensor) -> torch.Tensor:
    """Render two abstract identities at nuisance-controlled positions."""
    count, span = identities.shape
    masks = torch.zeros(count, span, IMAGE_SIZE, IMAGE_SIZE)
    for row in range(count):
        for column in range(span):
            y, x = positions[row, column].tolist()
            if int(identities[row, column]) == 0:
                masks[row, column, y - 5:y + 6, x - 2:x + 3] = 1.0
            else:
                masks[row, column, y - 2:y + 3, x - 5:x + 6] = 1.0
    return masks


def generate_sequence_memory_batch(
        count: int, *, span: int, distractors: int, seed: int,
        operation: str = "mixed", heldout: bool = False,
        generated_composition_ids: tuple[int, ...] | None = None,
        position_shift: bool = False,
        position_blend: float = 0.0,
        position_augmentation: bool = False,
        reverse_operations: bool = False,
        reverse_sequence: bool = False,
        blank_sequence: bool = False,
        sequence_override: torch.Tensor | None = None,
        operation_bits_override: torch.Tensor | None = None,
        device: torch.device | str = "cpu") -> SequenceMemoryBatch:
    """Render a deterministic sequence task without learner-visible labels."""
    if count < 2 or count % 2:
        raise ValueError("count must be an even integer of at least two")
    if span < 1:
        raise ValueError("span must be positive")
    if distractors < 0:
        raise ValueError("distractors must not be negative")
    if generated_composition_ids is not None:
        if not generated_composition_ids:
            raise ValueError("generated composition pool must not be empty")
        if any(
            composition_id < 0
            or composition_id >= len(_GENERATED_COMPOSITIONS)
            for composition_id in generated_composition_ids
        ):
            raise ValueError("generated composition ID is out of range")
    if operation not in (
        "forward", "reverse", "mixed", "complement", "complement_reverse",
        "complement_rotate", "adjacent_xor", "prefix_parity", "global_parity",
        "rotate", "undo_complement", "producer_global_parity",
        "generated_composition",
    ):
        raise ValueError(
            "operation must be forward, reverse, mixed, complement, "
            "complement_reverse, complement_rotate, adjacent_xor, "
            "prefix_parity, global_parity, rotate, undo_complement, or "
            "producer_global_parity, or generated_composition"
        )
    if not 0.0 <= position_blend <= 1.0:
        raise ValueError("position blend must be within [0, 1]")
    if position_shift:
        position_blend = 1.0

    generator = torch.Generator().manual_seed(seed)
    sequence = _balanced_binary_sequences(count, span, generator)
    if operation == "generated_composition":
        composition_pool = tuple(
            range(len(_GENERATED_COMPOSITIONS))
            if generated_composition_ids is None
            else generated_composition_ids
        )
        pool = torch.tensor(composition_pool, dtype=torch.long)
        composition_ids = torch.randint(
            len(composition_pool), (count,), generator=generator
        )
        composition_ids = pool[composition_ids]
        operation_bits = composition_ids.remainder(2)
    elif operation in (
        "forward", "complement", "complement_reverse", "complement_rotate",
        "adjacent_xor", "prefix_parity", "global_parity", "rotate",
        "undo_complement", "producer_global_parity",
    ):
        operation_bits = torch.zeros(count, dtype=torch.long)
    elif operation == "reverse":
        operation_bits = torch.ones(count, dtype=torch.long)
    else:
        operation_bits = torch.arange(count, dtype=torch.long) % 2
        operation_bits = operation_bits[
            torch.randperm(count, generator=generator)]
    if reverse_operations:
        operation_bits = 1 - operation_bits
    if sequence_override is not None:
        if tuple(sequence_override.shape) != (count, span):
            raise ValueError("sequence override shape does not match batch")
        sequence = sequence_override.detach().to("cpu").long().clone()
        if bool(((sequence != 0) & (sequence != 1)).any()):
            raise ValueError("sequence override must contain binary values")
    if operation_bits_override is not None:
        if tuple(operation_bits_override.shape) != (count,):
            raise ValueError(
                "operation-bits override shape does not match batch")
        operation_bits = (
            operation_bits_override.detach().to("cpu").long().clone())
        if bool(((operation_bits != 0) & (operation_bits != 1)).any()):
            raise ValueError(
                "operation-bits override must contain binary values")

    # Train and held-out positions are disjoint. Colour remains nuisance:
    # identity is shape, not a stable RGB code.
    base_positions = torch.tensor(
        ((10, 10), (22, 22), (10, 22), (22, 10)),
        dtype=torch.float32)
    shifted_positions = torch.tensor(
        ((7, 7), (25, 25), (7, 25), (25, 7)),
        dtype=torch.float32)
    if position_augmentation:
        position_generator = torch.Generator().manual_seed(
            seed ^ 0x5A17)
        row_blends = (torch.arange(count) % 5).float() / 4.0
        row_blends = row_blends[
            torch.randperm(count, generator=position_generator)]
    else:
        row_blends = torch.full((count,), position_blend)
    position_banks = torch.lerp(
        base_positions.unsqueeze(0), shifted_positions.unsqueeze(0),
        row_blends[:, None, None]).round().long()
    position_ids = torch.randint(
        0, len(base_positions), (count, span), generator=generator)
    positions = position_banks[
        torch.arange(count).unsqueeze(1), position_ids]
    masks = _identity_masks(positions, sequence)
    colors = 0.35 + 0.60 * torch.rand(
        count, span, 3, generator=generator)
    backgrounds = 0.015 + 0.025 * torch.rand(
        count, 1, 3, 1, 1, generator=generator)
    input_frames = backgrounds.expand(
        -1, span, -1, IMAGE_SIZE, IMAGE_SIZE).clone()
    input_frames += (
        masks.unsqueeze(2) * colors.unsqueeze(-1).unsqueeze(-1))
    if blank_sequence:
        input_frames.copy_(backgrounds.expand_as(input_frames))

    distractor_frames = backgrounds.expand(
        -1, distractors, -1, IMAGE_SIZE, IMAGE_SIZE).clone()
    if distractors:
        # An irrelevant X changes colour and location independently of the
        # sequence and operation. It is a valid sensory event, not blank time.
        for row in range(count):
            for step in range(distractors):
                center = int(torch.randint(
                    9, 24, (), generator=generator))
                for offset in range(-4, 5):
                    distractor_frames[
                        row, step, :, center + offset, center + offset] = 0.75
                    distractor_frames[
                        row, step, :, center + offset, center - offset] = 0.75

    query_frames = backgrounds.expand(
        -1, span, -1, IMAGE_SIZE, IMAGE_SIZE).clone()
    for row in range(count):
        if operation == "complement":
            # A third generic operation cue. It carries no operation name
            # or answer; it only makes this adjacent primitive observable
            # without conflating it with the inherited forward/reverse
            # protocol.
            operation_column = 14
        elif operation == "undo_complement":
            # Keep the producer cue visible so an acquired complement factor
            # can open. The second generic cue marks the consumer phase; a
            # prior-only consumer receives it only through the producer's
            # learned register, never as a raw event feature.
            operation_column = 14
        elif operation == "producer_global_parity":
            # The complement producer cue remains visible while the global
            # aggregation cue defines the consumer's verifier objective.
            operation_column = 14
        elif operation == "complement_reverse":
            # A fifth generic operation cue selects reverse-then-complement.
            # It carries no operation name or answer.
            operation_column = 19
        elif operation == "rotate":
            # A fourth generic operation cue selects the next-item rotation.
            # It is a visual event, not a semantic operation label.
            operation_column = 9
        elif operation == "complement_rotate":
            # A sixth generic operation cue selects rotate-then-complement.
            # It carries no operation name or answer.
            operation_column = 24
        elif operation == "adjacent_xor":
            # A seventh generic operation cue selects adjacent comparison.
            # It carries no operation name or answer.
            operation_column = 5
        elif operation == "prefix_parity":
            # A cumulative binding cue; it is a visual event, not a label.
            operation_column = 0
        elif operation == "global_parity":
            # A global aggregation cue; it carries no answer or task label.
            operation_column = 28
        elif operation == "generated_composition":
            # The sampled grammar emits two generic primitive cues. The
            # verifier-private composition ID never enters the learner.
            for primitive in _GENERATED_COMPOSITIONS[int(composition_ids[row])]:
                operation_column = _GENERATED_PRIMITIVE_COLUMNS[primitive]
                query_frames[
                    row, :, :, 2:5, operation_column:operation_column + 3
                ] = 0.95
            operation_column = None
        else:
            operation_column = 2 if int(operation_bits[row]) == 0 else 27
        if operation_column is not None:
            query_frames[
                row, :, :, 2:5, operation_column:operation_column + 3
            ] = 0.95
        if operation == "undo_complement":
            query_frames[row, :, :, 2:5, 6:9] = 0.95
        if operation == "producer_global_parity":
            query_frames[row, :, :, 2:5, 28:31] = 0.95
        for query_index in range(span):
            # A unary ordinal cue: no written number or semantic position ID.
            for mark in range(query_index + 1):
                start = 3 + mark * 4
                query_frames[
                    row, query_index, :, 27:30, start:start + 2] = 0.85

    source_index = torch.arange(span).unsqueeze(0).expand(count, -1)
    reverse_index = torch.arange(span - 1, -1, -1).unsqueeze(0).expand(
        count, -1)
    rotate_index = (source_index + 1) % span
    selected_index = (
        rotate_index
        if operation in ("rotate", "complement_rotate", "adjacent_xor")
        else reverse_index
        if operation == "complement_reverse"
        else torch.where(
            operation_bits.unsqueeze(1).bool(), reverse_index, source_index
        )
    )
    if operation == "generated_composition":
        selected_sequence = _apply_generated_compositions(
            sequence, composition_ids
        )
    else:
        selected_sequence = torch.gather(sequence, 1, selected_index)
    if operation == "adjacent_xor":
        correct = (sequence != selected_sequence).long()
    elif operation == "prefix_parity":
        correct = torch.cumsum(sequence, dim=1).remainder(2)
    elif operation == "global_parity":
        correct = sequence.sum(dim=1, keepdim=True).remainder(2).expand_as(sequence)
    elif operation in (
        "complement", "complement_reverse", "complement_rotate"
    ):
        correct = 1 - selected_sequence
    elif operation == "undo_complement":
        correct = selected_sequence
    elif operation == "producer_global_parity":
        correct = sequence.sum(dim=1, keepdim=True).remainder(2).expand_as(sequence)
    elif operation == "generated_composition":
        correct = selected_sequence
    else:
        correct = selected_sequence

    if reverse_sequence:
        sequence = sequence.flip(1)
        input_frames = input_frames.flip(1)
        if operation == "generated_composition":
            selected_sequence = _apply_generated_compositions(
                sequence, composition_ids
            )
        else:
            selected_sequence = torch.gather(sequence, 1, selected_index)
        if operation == "adjacent_xor":
            correct = (sequence != selected_sequence).long()
        elif operation == "prefix_parity":
            correct = torch.cumsum(sequence, dim=1).remainder(2)
        elif operation == "global_parity":
            correct = sequence.sum(dim=1, keepdim=True).remainder(2).expand_as(
                sequence
            )
        elif operation in (
            "complement", "complement_reverse", "complement_rotate"
        ):
            correct = 1 - selected_sequence
        elif operation == "undo_complement":
            correct = selected_sequence
        elif operation == "producer_global_parity":
            correct = sequence.sum(dim=1, keepdim=True).remainder(2).expand_as(
                sequence)
        elif operation == "generated_composition":
            correct = selected_sequence
        else:
            correct = selected_sequence

    return SequenceMemoryBatch(
        input_frames.to(device), distractor_frames.to(device),
        query_frames.to(device), correct.to(device), sequence.to(device),
        operation_bits.to(device),
        torch.arange(seed, seed + count, dtype=torch.long, device=device))


def _reset_active_state_keep_workspace(
        state: ControllerState) -> ControllerState:
    """Ablate the recurrent carrier while preserving physical workspace."""
    return ControllerState(
        torch.zeros_like(state.hidden), state.workspace,
        torch.zeros_like(state.latest_event), state.workspace_usage,
        state.workspace_volatility, state.event_age)


def rollout_sequence_memory(
        model: UnifiedCognitiveController, batch: SequenceMemoryBatch, *,
        sample_actions: bool, exploration: float = 0.10,
        disable_workspace: bool = False,
        reset_active_state_before_query: bool = False,
        reset_all_memory_before_query: bool = False,
        operation_cue_blank: bool = False,
        shuffle_outcomes: bool = False,
        loss_output: str = "all",
        loss_mode: str = "bce",
        return_slot_activity: bool = False,
        ) -> dict[str, torch.Tensor]:
    """Run one real-time episode and return attempted-action evidence."""
    device = batch.input_frames.device
    if loss_output not in ("all", "first", "last"):
        raise ValueError("loss output must be all, first, or last")
    if loss_mode not in ("bce", "reinforce", "success_only"):
        raise ValueError("loss mode must be bce, reinforce, or success_only")
    state = model.initial_state(batch.batch_size, device=device)
    null = torch.full(
        (batch.batch_size,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(batch.batch_size, device=device)
    slot_openings: list[torch.Tensor] = []
    slot_residual_norms: list[torch.Tensor] = []

    def record_activity(output) -> None:
        if return_slot_activity and output.skill_adapter_openings is not None:
            slot_openings.append(output.skill_adapter_openings)
            slot_residual_norms.append(output.skill_adapter_residual_norms)

    for frame_index in range(batch.span):
        output, state = model.step(
            batch.input_frames[:, frame_index], state, null, zeros, zeros,
            disable_workspace=disable_workspace)
        record_activity(output)
    for frame_index in range(batch.distractor_frames.shape[1]):
        output, state = model.step(
            batch.distractor_frames[:, frame_index], state, null, zeros, zeros,
            disable_workspace=disable_workspace)
        record_activity(output)
    if reset_all_memory_before_query:
        state = model.initial_state(batch.batch_size, device=device)
    elif reset_active_state_before_query:
        state = _reset_active_state_keep_workspace(state)

    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []

    previous_action = null
    previous_reward = zeros
    for query_index in range(batch.span):
        frame = batch.query_frames[:, query_index]
        if operation_cue_blank:
            frame = frame.clone()
            # Clear every public operation-cue slot, including the center
            # cue used by the adjacent complement primitive. The ordinal
            # query marks at the bottom remain intact.
            frame[:, :, 2:5, 2:5] = 0.0
            frame[:, :, 2:5, 14:17] = 0.0
            frame[:, :, 2:5, 27:30] = 0.0
        feedback = torch.full_like(
            previous_reward, float(query_index > 0))
        output, state = model.step(
            frame, state, previous_action,
            previous_reward * feedback, feedback,
            disable_workspace=disable_workspace)
        record_activity(output)
        probabilities = torch.softmax(output.logits, dim=-1)
        if sample_actions:
            behavior = (
                probabilities * (1.0 - exploration)
                + exploration / ACTIONS)
            action = torch.multinomial(behavior, 1).squeeze(1)
        else:
            action = output.logits.argmax(dim=-1)
        reward = (
            action == batch.correct_actions[:, query_index]).float()
        delivered_outcome = (
            reward.roll(1) if shuffle_outcomes else reward)
        losses.append(
            attempted_success_loss(output.logits, action, delivered_outcome)
            if loss_mode == "bce" else (
                _attempted_reinforce_loss(
                    output.logits, action, delivered_outcome)
                if loss_mode == "reinforce" else _successful_action_loss(
                    output.logits, action, delivered_outcome)))
        actions.append(action)
        rewards.append(reward)
        logits.append(output.logits)
        previous_action = action
        previous_reward = delivered_outcome
    selected_losses = (
        losses if loss_output == "all"
        else [losses[0] if loss_output == "first" else losses[-1]])
    result = {
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "logits": torch.stack(logits, dim=1),
        "loss": torch.stack(selected_losses).mean(),
        "final_workspace": state.workspace,
        "final_hidden": state.hidden,
    }
    if return_slot_activity and slot_openings:
        result["skill_adapter_openings"] = torch.stack(slot_openings, dim=1)
        result["skill_adapter_residual_norms"] = torch.stack(
            slot_residual_norms, dim=1)
    return result


@torch.no_grad()
def evaluate_sequence_memory(
        model: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, operation: str,
        device: torch.device) -> dict[str, float | list[float]]:
    """Run normal and causal audits on lifetime-disjoint rendered episodes."""
    model.eval()
    normal_batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation=operation, heldout=True, device=device)
    reversed_operation_batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation=operation, heldout=True, reverse_operations=True,
        device=device)
    reversed_sequence_batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation=operation, heldout=True, reverse_sequence=True,
        device=device)
    blank_batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation=operation, heldout=True, blank_sequence=True,
        device=device)
    position_batches = {
        blend: generate_sequence_memory_batch(
            count, span=span, distractors=distractors, seed=seed,
            operation=operation, heldout=True, position_blend=blend,
            device=device)
        for blend in (0.25, 0.50, 0.75, 1.0)}
    zero_distractor_batch = (
        generate_sequence_memory_batch(
            count, span=span, distractors=0, seed=seed,
            operation=operation, heldout=True, device=device)
        if distractors else None)

    def run(
            batch: SequenceMemoryBatch, **kwargs: bool,
            ) -> dict[str, torch.Tensor]:
        return rollout_sequence_memory(
            model, batch, sample_actions=False, **kwargs)

    normal = run(normal_batch)
    reverse_operation = run(reversed_operation_batch)
    reverse_sequence = run(reversed_sequence_batch)
    blank = run(blank_batch)
    cue_blank = run(normal_batch, operation_cue_blank=True)
    workspace_disabled = run(normal_batch, disable_workspace=True)
    active_reset = run(
        normal_batch, reset_active_state_before_query=True)
    all_reset = run(normal_batch, reset_all_memory_before_query=True)
    position_results = {
        blend: run(batch) for blend, batch in position_batches.items()}
    zero_distractor = (
        run(zero_distractor_batch)
        if zero_distractor_batch is not None else normal)
    nonpalindrome = (
        normal_batch.sequence != normal_batch.sequence.flip(1)).any(dim=1)
    operation_flip = (
        (
            normal["actions"][nonpalindrome]
            != reverse_operation["actions"][nonpalindrome]).float().mean()
        if bool(nonpalindrome.any()) else torch.tensor(float("nan")))
    sequence_flip = (
        (
            normal["actions"][nonpalindrome]
            != reverse_sequence["actions"][nonpalindrome]).float().mean()
        if bool(nonpalindrome.any()) else torch.tensor(float("nan")))

    def accuracy(result: dict[str, torch.Tensor]) -> float:
        return float(result["rewards"].mean())

    forward_rows = normal_batch.operation_bits == 0
    reverse_rows = normal_batch.operation_bits == 1
    return {
        "accuracy": accuracy(normal),
        "forward_accuracy": float(normal["rewards"][forward_rows].mean()),
        "reverse_accuracy": float(normal["rewards"][reverse_rows].mean()),
        "accuracy_by_output": [
            float(value) for value in normal["rewards"].mean(0)],
        "reverse_operation_accuracy": accuracy(reverse_operation),
        "reverse_operation_prediction_flip_rate_nonpalindrome": float(
            operation_flip),
        "reverse_sequence_accuracy": accuracy(reverse_sequence),
        "reverse_sequence_prediction_flip_rate_nonpalindrome": float(
            sequence_flip),
        "blank_sequence_accuracy": accuracy(blank),
        "blank_operation_cue_accuracy": accuracy(cue_blank),
        "workspace_disabled_accuracy": accuracy(workspace_disabled),
        "active_state_reset_accuracy": accuracy(active_reset),
        "all_memory_reset_accuracy": accuracy(all_reset),
        "zero_distractor_accuracy": accuracy(zero_distractor),
        "heldout_position_accuracy": accuracy(position_results[1.0]),
        "position_accuracy_by_blend": {
            str(blend): accuracy(result)
            for blend, result in position_results.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=26001)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--epochs-per-batch", type=int, default=1,
        help=(
            "optimizer passes over each uniquely generated episode packet; "
            "increasing this spends private compute without consuming "
            "additional verifier bits"))
    parser.add_argument(
        "--rerender-each-epoch", action="store_true",
        help=(
            "when reusing a packet, preserve its logical sequences and "
            "operation bits but re-render colors, positions, and distractors "
            "for each extra pass"))
    parser.add_argument(
        "--train-only-action-adapter", action="store_true",
        help=(
            "freeze the inherited controller and train only the optional "
            "generic action adapter; requires action_adapter_width > 0"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-episodes", type=int, default=2048)
    parser.add_argument("--span", type=int, default=2)
    parser.add_argument("--distractors", type=int, default=1)
    parser.add_argument(
        "--position-blend", type=float, default=0.0,
        help="fixed train-position interpolation from base (0) to shifted (1)")
    parser.add_argument(
        "--position-curriculum", default="",
        help=(
            "comma-separated increasing position blends. Training divides "
            "updates into stages and rehearses every prior level."))
    parser.add_argument(
        "--position-augmentation", action="store_true",
        help=(
            "balance base, intermediate, and shifted positions inside every "
            "training batch; no position value is exposed to the learner"))
    parser.add_argument(
        "--rehearse-zero-distractor", action="store_true",
        help=(
            "alternate target-distractor episodes with the already mastered "
            "zero-distractor stream to prevent retention loss"))
    parser.add_argument(
        "--rehearse-span2", action="store_true",
        help=(
            "alternate span-2 episodes with the target span to protect the "
            "previously mastered two-item primitive"))
    parser.add_argument(
        "--rehearse-spans", default="",
        help=(
            "comma-separated mastered spans to cycle with the target span; "
            "the target span is sampled first, and repeated values weight "
            "rehearsal frequency"))
    parser.add_argument(
        "--operation", choices=(
            "forward", "reverse", "mixed", "rotate", "undo_complement",
            "producer_global_parity"),
        default="mixed")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=4)
    parser.add_argument("--intention-width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument(
        "--eval-every", type=int, default=0,
        help="measure a lifetime-disjoint learning curve every N updates")
    parser.add_argument("--curve-test-episodes", type=int, default=1024)
    parser.add_argument("--mastery-threshold", type=float, default=0.90)
    parser.add_argument(
        "--shuffle-outcomes", action="store_true",
        help="negative control: deliver another lifetime's scalar outcome")
    parser.add_argument(
        "--loss-output", choices=("all", "first", "last"), default="all",
        help=(
            "temporal curriculum: optimize all outputs or one endpoint while "
            "still learning only from that attempted action's scalar outcome"))
    parser.add_argument(
        "--loss-mode", choices=("bce", "reinforce", "success_only"),
        default="bce",
        help="bandit objective; reinforce uses only sampled outcomes")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.steps < 1 or args.epochs_per_batch < 1 or args.batch_size < 2:
        raise ValueError(
            "steps, epochs-per-batch, and batch-size must be positive")
    if args.batch_size % 2:
        raise ValueError("batch-size must be divisible by two")

    seed_everything(args.seed)
    device = torch.device(args.device)
    curriculum = (
        tuple(float(value) for value in args.position_curriculum.split(","))
        if args.position_curriculum else ())
    rehearsal_spans = (
        tuple(int(value) for value in args.rehearse_spans.split(","))
        if args.rehearse_spans else ())
    if any(value < 1 for value in rehearsal_spans):
        raise ValueError("rehearsal spans must be positive")
    if args.rehearse_span2 and rehearsal_spans:
        raise ValueError(
            "use --rehearse-span2 or --rehearse-spans, not both")
    if any(
            not 0.0 < value <= 1.0 for value in curriculum
            ) or any(
                later <= earlier
                for earlier, later in itertools.pairwise(curriculum)):
        raise ValueError(
            "position curriculum must be strictly increasing within (0, 1]")
    configuration: dict[str, object] = {
        "width": args.width,
        "workspace_slots": args.workspace_slots,
        "intention_width": args.intention_width,
    }
    payload = None
    if args.checkpoint_in is not None:
        payload = torch.load(
            args.checkpoint_in, map_location=device, weights_only=False)
        configuration = dict(payload["model_configuration"])
    model = UnifiedCognitiveController(**configuration).to(device)
    if payload is not None:
        load_result = model.load_state_dict(
            payload["state_dict"], strict=False)
        allowed_missing = set()
        if configuration.get("workspace_slot_addressing", False):
            allowed_missing.update({
                "workspace_read_address_scale",
                "workspace_write_address_scale",
                "workspace_write_content_address_scale",
            })
        if configuration.get("action_adapter_width", 0):
            allowed_missing.update(
                name for name in model.state_dict()
                if name.startswith((
                    "action_adapter.", "action_adapter_gate."))
                and name not in payload["state_dict"])
        if configuration.get("relation_adapter_width", 0):
            allowed_missing.update(
                name for name in model.state_dict()
                if name.startswith((
                    "relation_adapter.", "relation_adapter_gate."))
                and name not in payload["state_dict"])
        if set(load_result.missing_keys) != allowed_missing:
            raise RuntimeError(
                "checkpoint/configuration mismatch: "
                f"missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}")
        if load_result.unexpected_keys:
            raise RuntimeError(
                "checkpoint/configuration mismatch: "
                f"unexpected={load_result.unexpected_keys}")
    if args.train_only_action_adapter:
        if model.action_adapter is None:
            raise ValueError(
                "train-only-action-adapter requires action_adapter_width")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith(("action_adapter.", "action_adapter_gate.")))
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters remain")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    position_counts: dict[str, int] = {}
    distractor_counts: dict[str, int] = {}
    span_counts: dict[str, int] = {}
    seen_verifier_bits = 0
    started = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        if rehearsal_spans:
            span_cycle = (args.span,) + rehearsal_spans
            train_span = span_cycle[(step - 1) % len(span_cycle)]
        else:
            train_span = (
                2 if args.rehearse_span2 and step % 2 == 1 else args.span)
        span_key = str(train_span)
        span_counts[span_key] = span_counts.get(span_key, 0) + 1
        if curriculum:
            stage_length = max(1, (args.steps + len(curriculum) - 1)
                               // len(curriculum))
            stage = min((step - 1) // stage_length, len(curriculum) - 1)
            available_blends = (0.0,) + curriculum[:stage + 1]
            train_position_blend = available_blends[
                (step - 1) % len(available_blends)]
        else:
            train_position_blend = args.position_blend
        blend_key = str(train_position_blend)
        position_counts[blend_key] = position_counts.get(blend_key, 0) + 1
        train_distractors = (
            0 if (
                args.rehearse_zero_distractor
                and args.distractors
                and step % 2 == 1)
            else args.distractors)
        distractor_key = str(train_distractors)
        distractor_counts[distractor_key] = (
            distractor_counts.get(distractor_key, 0) + 1)
        batch = generate_sequence_memory_batch(
            args.batch_size, span=train_span, distractors=train_distractors,
            seed=args.seed + step * args.batch_size,
            operation=args.operation,
            position_blend=train_position_blend,
            position_augmentation=args.position_augmentation,
            device=device)
        result = None
        for epoch in range(args.epochs_per_batch):
            epoch_batch = batch
            if epoch and args.rerender_each_epoch:
                epoch_batch = generate_sequence_memory_batch(
                    args.batch_size, span=train_span,
                    distractors=train_distractors,
                    seed=(
                        args.seed + step * args.batch_size
                        + epoch * 1_000_003),
                    operation=args.operation,
                    position_blend=train_position_blend,
                    position_augmentation=args.position_augmentation,
                    sequence_override=batch.sequence,
                    operation_bits_override=batch.operation_bits,
                    device=device)
            result = rollout_sequence_memory(
                model, epoch_batch, sample_actions=True,
                exploration=args.exploration,
                shuffle_outcomes=args.shuffle_outcomes,
                loss_output=args.loss_output,
                loss_mode=args.loss_mode)
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
        assert result is not None
        seen_verifier_bits += args.batch_size * train_span
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "update": step,
                "unique_episodes": step * args.batch_size,
                "unique_verifier_bits": seen_verifier_bits,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
                "train_position_blend": train_position_blend,
                "train_distractors": train_distractors,
            }
            if args.eval_every and (
                    step % args.eval_every == 0 or step == args.steps):
                curve_audit = evaluate_sequence_memory(
                    model, count=args.curve_test_episodes, span=args.span,
                    distractors=args.distractors,
                    seed=args.seed + 20_000_000 + step,
                    operation=args.operation, device=device)
                row["heldout_accuracy"] = float(curve_audit["accuracy"])
                row["heldout_reverse_accuracy"] = float(
                    curve_audit["reverse_accuracy"])
                row["heldout_operation_flip_rate"] = float(
                    curve_audit[
                        "reverse_operation_prediction_flip_rate_nonpalindrome"])
            history.append(row)
            print(json.dumps(row), flush=True)

    audit = evaluate_sequence_memory(
        model, count=args.test_episodes, span=args.span,
        distractors=args.distractors, seed=args.seed + 10_000_000,
        operation=args.operation, device=device)
    measured_prefixes = [
        row for row in history if "heldout_accuracy" in row]
    stable_bits_to_threshold = None
    for index, row in enumerate(measured_prefixes):
        if all(
                float(later["heldout_accuracy"]) >= args.mastery_threshold
                for later in measured_prefixes[index:]):
            stable_bits_to_threshold = int(row["unique_verifier_bits"])
            break
    report = {
        "schema": "sequence-working-memory-experiment-v1",
        "learner_visible_information": (
            "RGB streams, own opaque actions, scalar attempted-action outcome"),
        "operation": args.operation,
        "span": args.span,
        "distractors": args.distractors,
        "fixed_position_blend": args.position_blend,
        "position_curriculum": curriculum,
        "position_augmentation": args.position_augmentation,
        "position_update_counts": position_counts,
        "rehearse_zero_distractor": args.rehearse_zero_distractor,
        "rehearse_span2": args.rehearse_span2,
        "rehearse_spans": rehearsal_spans,
        "span_update_counts": span_counts,
        "distractor_update_counts": distractor_counts,
        "epochs_per_batch": args.epochs_per_batch,
        "rerender_each_epoch": args.rerender_each_epoch,
        "optimizer_updates": args.steps * args.epochs_per_batch,
        "unique_logical_episodes": args.steps * args.batch_size,
        "unique_verifier_bits": seen_verifier_bits,
        "optimizer_lifetime_exposures": (
            args.steps * args.batch_size * args.epochs_per_batch),
        "replayed_examples": 0,
        "outcomes_shuffled": args.shuffle_outcomes,
        "loss_output": args.loss_output,
        "loss_mode": args.loss_mode,
        "mastery_threshold": args.mastery_threshold,
        "stable_bits_to_threshold": stable_bits_to_threshold,
        "wall_seconds": perf_counter() - started,
        "model_configuration": configuration,
        "history": history,
        "audit": audit,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "sequence_working_memory_report": report,
        }, args.checkpoint_out)


if __name__ == "__main__":
    main()
