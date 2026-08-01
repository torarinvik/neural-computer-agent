"""Progressive, pixel-only procedural-shape short-term-memory benchmark.

A lifetime presents ``span`` abstract shapes one at a time.  Each query shows
an independently re-rendered candidate plus a unary ordinal cue.  The opaque
binary action is rewarded iff it says whether the candidate has the identity
that appeared at that ordinal.  Identity is never exposed to the learner.

The task deliberately separates content load (span/vocabulary) from nuisance
variation (position, scale, rotation, colour, background and deformation).
This permits a gradual curriculum and honest sample-efficiency comparisons.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from .environment import ACTIONS, IMAGE_SIZE, NULL_ACTION
from .model import ControllerState, UnifiedCognitiveController
from .train import attempted_success_loss, seed_everything


@dataclass(frozen=True)
class ShapeNuisance:
    """Independent label-preserving visual variation."""

    position_px: float = 1.0
    size_fraction: float = 0.01
    rotation_degrees: float = 1.0
    color_delta: float = 0.02
    background_delta: float = 0.01
    deformation: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.position_px <= 6.0:
            raise ValueError("position_px must be within [0, 6]")
        if not 0.0 <= self.size_fraction <= 0.35:
            raise ValueError("size_fraction must be within [0, .35]")
        if not 0.0 <= self.rotation_degrees <= 180.0:
            raise ValueError("rotation_degrees must be within [0, 180]")
        if not 0.0 <= self.color_delta <= 0.35:
            raise ValueError("color_delta must be within [0, .35]")
        if not 0.0 <= self.background_delta <= 0.20:
            raise ValueError("background_delta must be within [0, .20]")
        if not 0.0 <= self.deformation <= 0.25:
            raise ValueError("deformation must be within [0, .25]")


@dataclass(frozen=True)
class ProceduralShapeBatch:
    presentation_frames: torch.Tensor
    query_frames: torch.Tensor
    correct_actions: torch.Tensor
    sequence_identities: torch.Tensor
    candidate_identities: torch.Tensor
    query_ordinals: torch.Tensor
    query_cue_ordinals: torch.Tensor
    query_operations: torch.Tensor
    new_slot_independent: torch.Tensor
    logical_lifetime_ids: torch.Tensor
    objective: str

    @property
    def batch_size(self) -> int:
        return int(self.presentation_frames.shape[0])

    @property
    def span(self) -> int:
        return int(self.presentation_frames.shape[1])


def nuisance_from_level(level: float) -> ShapeNuisance:
    """Map one curriculum coordinate to bounded, independently reported axes."""
    if not 0.0 <= level <= 1.0:
        raise ValueError("randomness level must be within [0, 1]")
    # The nonzero intercept is intentional: even rung zero cannot be solved by
    # exact pixel matching.
    return ShapeNuisance(
        position_px=1.0 + 5.0 * level,
        size_fraction=0.01 + 0.29 * level,
        rotation_degrees=1.0 + 179.0 * level,
        color_delta=0.02 + 0.28 * level,
        background_delta=0.01 + 0.14 * level,
        deformation=0.20 * level,
    )


def nuisance_with_overrides(
        base: ShapeNuisance, *, position_px: float | None = None,
        size_fraction: float | None = None,
        rotation_degrees: float | None = None,
        color_delta: float | None = None,
        background_delta: float | None = None,
        deformation: float | None = None) -> ShapeNuisance:
    """Replace selected axes while leaving all other axes unchanged."""
    values = asdict(base)
    overrides = {
        "position_px": position_px,
        "size_fraction": size_fraction,
        "rotation_degrees": rotation_degrees,
        "color_delta": color_delta,
        "background_delta": background_delta,
        "deformation": deformation,
    }
    values.update({
        name: value for name, value in overrides.items()
        if value is not None})
    result = ShapeNuisance(**values)
    result.validate()
    return result


def _balanced_logical_content(
        count: int, span: int, vocabulary: int,
        generator: torch.Generator,
        permutations: int,
        allow_partial_balance: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balance identities and answers exactly; query orders independently."""
    answer_patterns = 2 ** span
    sequence_patterns = vocabulary ** span
    logical_patterns = sequence_patterns * answer_patterns
    if allow_partial_balance and count % logical_patterns:
        full_repeats, remainder = divmod(count, logical_patterns)
        design_ids = torch.cat((
            torch.arange(logical_patterns).repeat(full_repeats),
            torch.randperm(logical_patterns, generator=generator)[:remainder],
        ))
    else:
        design_ids = torch.arange(count) % logical_patterns
    ids = design_ids % logical_patterns
    if count % (logical_patterns * permutations) == 0:
        # Preserve exact query-order balance for the existing full designs.
        permutation_ids = (
            torch.arange(count) // logical_patterns) % permutations
    else:
        # Longer spans make the factorial query-order cross prohibitively
        # large. Identity and answer balance stay exact; independent random
        # query orders prevent an order/content correlation in a microbatch.
        permutation_ids = torch.randint(
            permutations, (count,), generator=generator)
    sequence_ids = ids // answer_patterns
    answer_ids = ids % answer_patterns
    columns = [
        (sequence_ids // (vocabulary ** shift)) % vocabulary
        for shift in reversed(range(span))]
    answer_columns = [
        (answer_ids >> shift) & 1
        for shift in reversed(range(span))]
    order = torch.randperm(count, generator=generator)
    return (
        torch.stack(columns, dim=1)[order],
        torch.stack(answer_columns, dim=1)[order],
        permutation_ids[order])


def _render_shapes(
        identities: torch.Tensor, *, seed: int, nuisance: ShapeNuisance,
        heldout: bool, ordinal_cues: bool = False) -> torch.Tensor:
    """Render analytic radial shapes; all nuisance draws are independent."""
    count, steps = identities.shape
    generator = torch.Generator().manual_seed(seed)
    axis = torch.arange(IMAGE_SIZE, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    frames = torch.empty(count, steps, 3, IMAGE_SIZE, IMAGE_SIZE)
    base_palette = torch.tensor((
        (0.86, 0.47, 0.22), (0.20, 0.70, 0.88),
        (0.64, 0.83, 0.25), (0.78, 0.34, 0.78)))

    for row in range(count):
        for step in range(steps):
            identity = int(identities[row, step])
            dx = (torch.rand((), generator=generator) * 2 - 1)
            dy = (torch.rand((), generator=generator) * 2 - 1)
            dx = float(dx) * nuisance.position_px
            dy = float(dy) * nuisance.position_px
            scale_noise = float(
                torch.rand((), generator=generator) * 2 - 1)
            radius = 7.2 * (1.0 + nuisance.size_fraction * scale_noise)
            rotation = math.radians(nuisance.rotation_degrees) * float(
                torch.rand((), generator=generator) * 2 - 1)
            phase = float(torch.rand((), generator=generator)) * math.tau
            deformation = nuisance.deformation * float(
                torch.rand((), generator=generator) * 2 - 1)
            centered_x = xx - (15.5 + dx)
            centered_y = yy - (14.5 + dy)
            polar_radius = torch.sqrt(centered_x.square() + centered_y.square())
            theta = torch.atan2(centered_y, centered_x) - rotation

            # Identity determines harmonic count, not colour or position.
            harmonic = 3 + 2 * (identity % 4)
            boundary = radius * (
                0.78 + 0.20 * torch.cos(harmonic * theta)
                + deformation * torch.cos((harmonic + 1) * theta + phase))
            mask = polar_radius <= boundary.clamp_min(radius * 0.42)

            background_base = 0.045 if not heldout else 0.055
            background = background_base + nuisance.background_delta * float(
                torch.rand((), generator=generator) * 2 - 1)
            frame = torch.full((3, IMAGE_SIZE, IMAGE_SIZE), background)
            palette_index = int(torch.randint(
                0, len(base_palette), (), generator=generator))
            colour = base_palette[palette_index].clone()
            colour += nuisance.color_delta * (
                torch.rand(3, generator=generator) * 2 - 1)
            colour = colour.clamp(0.15, 0.98)
            frame[:, mask] = colour[:, None]
            if ordinal_cues:
                # Unary, modality-native cue. It carries only the queried
                # ordinal; it never carries identity or the correct action.
                for mark in range(step + 1):
                    left = 2 + 3 * mark
                    if left + 1 < IMAGE_SIZE:
                        frame[:, 28:30, left:left + 2] = 0.96
            frames[row, step] = frame
    return frames


def _add_query_cues(
        frames: torch.Tensor, cue_ordinals: torch.Tensor,
        operations: torch.Tensor, *, show_operation_cue: bool) -> torch.Tensor:
    """Overlay learner-visible ordinal and abstract operation cues."""
    frames = frames.clone()
    count, steps = cue_ordinals.shape
    for row in range(count):
        for step in range(steps):
            for mark in range(int(cue_ordinals[row, step]) + 1):
                left = 2 + 3 * mark
                if left + 1 < IMAGE_SIZE:
                    frames[row, step, :, 28:30, left:left + 2] = 0.96
            if show_operation_cue:
                # Three arbitrary, modality-native glyphs. Their meaning is not
                # exposed: only scalar verifier outcomes can ground them.
                operation = int(operations[row, step])
                if operation == 0:
                    frames[row, step, 0, 2:8, 2:8] = 0.92
                    frames[row, step, 1:, 2:8, 2:8] = 0.28
                elif operation == 1:
                    frames[row, step, 2, 2:8, 24:30] = 0.92
                    frames[row, step, :2, 2:8, 24:30] = 0.28
                elif operation == 2:
                    frames[row, step, 1, 20:26, 24:30] = 0.92
                    frames[row, step, (0, 2), 20:26, 24:30] = 0.28
                else:
                    raise ValueError(f"unsupported query operation {operation}")
    return frames


def generate_procedural_shape_batch(
        count: int, *, span: int, vocabulary: int, seed: int,
        nuisance: ShapeNuisance, heldout: bool = False,
        objective: str = "recognition",
        query_count: int | None = None,
        new_slot_difficulty: float = 1.0,
        query_frontier_difficulty: float = 1.0,
        query_history_difficulty: float = 1.0,
        third_query_history_stage: int = 2,
        previous_query_stage: int = -1,
        previous_query_scope_difficulty: float = 1.0,
        previous_query_position: int = -1,
        previous_query_anchor_focus: int = -1,
        next_query_stage: int = -1,
        next_query_position: int = -1,
        next_query_anchor_focus: int = -1,
        next_query_target_aligned: bool = False,
        blank_presentation: bool = False,
        reverse_presentation: bool = False,
        flip_candidates: bool = False,
        flip_query_operations: bool = False,
        allow_partial_balance: bool = False,
        device: torch.device | str = "cpu") -> ProceduralShapeBatch:
    """Generate deterministic balanced episodes and private verifier answers."""
    nuisance.validate()
    if objective not in ("visible_identity", "recognition"):
        raise ValueError("objective must be visible_identity or recognition")
    permutations = tuple(itertools.permutations(range(span)))
    logical_patterns = (vocabulary * 2) ** span
    if count < 1 or (
            not allow_partial_balance
            and (count < logical_patterns or count % logical_patterns)):
        raise ValueError(
            f"count must be a positive multiple of {logical_patterns}"
            " unless partial balance is enabled")
    if span < 1:
        raise ValueError("span must be positive")
    if query_count is None:
        query_count = span
    if not 1 <= query_count <= span:
        raise ValueError("query count must be within [1, span]")
    if not 0.0 <= new_slot_difficulty <= 1.0:
        raise ValueError("new slot difficulty must be within [0, 1]")
    if not 0.0 <= query_frontier_difficulty <= 1.0:
        raise ValueError("query frontier difficulty must be within [0, 1]")
    if not 0.0 <= query_history_difficulty <= 1.0:
        raise ValueError("query history difficulty must be within [0, 1]")
    if third_query_history_stage not in (0, 1, 2):
        raise ValueError("third query history stage must be 0, 1, or 2")
    if previous_query_stage not in (-1, 0, 1, 2):
        raise ValueError("previous query stage must be -1, 0, 1, or 2")
    if previous_query_stage >= 0 and span < 2:
        raise ValueError("previous query curriculum requires span at least 2")
    if previous_query_stage == 2 and span != 3:
        raise ValueError("previous query stage 2 currently requires span 3")
    if not 0.0 <= previous_query_scope_difficulty <= 1.0:
        raise ValueError(
            "previous query scope difficulty must be within [0, 1]")
    if previous_query_position not in (-1, 0, 1, 2):
        raise ValueError("previous query position must be -1, 0, 1, or 2")
    if (
            previous_query_position >= 0
            and (
                previous_query_stage < 1
                or previous_query_position >= query_count
                or previous_query_scope_difficulty < 1.0)):
        raise ValueError(
            "forced previous query position requires an active previous "
            "operation, full scope, and a position within the queried prefix")
    if previous_query_anchor_focus not in (-1, 0, 1):
        raise ValueError("previous query anchor focus must be -1, 0, or 1")
    if (
            previous_query_anchor_focus >= 0
            and (
                span != 3
                or previous_query_stage < previous_query_anchor_focus + 1
                or previous_query_scope_difficulty < 1.0
                or (
                    query_count == 1 and previous_query_position >= 0)
                or (
                    query_count > 1 and previous_query_position < 0))):
        raise ValueError(
            "anchor focus requires span three, an enabled anchor, full scope, "
            "and either one natural query or a forced multi-query position")
    if previous_query_stage >= 0 and objective != "recognition":
        raise ValueError(
            "previous query curriculum requires the recognition objective")
    if next_query_stage not in (-1, 1, 2, 3):
        raise ValueError("next query stage must be -1, 1, 2, or 3")
    if next_query_stage >= 1 and span < 2:
        raise ValueError("next query curriculum requires span at least 2")
    if next_query_stage == 2 and span < 3:
        raise ValueError("next query stage 2 requires span at least 3")
    if next_query_stage == 3 and span != 3:
        raise ValueError(
            "next query stage 3's mixed-operation curriculum requires span 3")
    if next_query_stage in (1, 2) and previous_query_stage >= 1:
        raise ValueError(
            "next-only stages must rehearse previous-item behavior in a "
            "separate stream")
    if next_query_position not in (-1, 0, 1, 2):
        raise ValueError("next query position must be -1, 0, 1, or 2")
    if (
            next_query_position >= 0
            and (
                next_query_stage < 1
                or next_query_position >= query_count)):
        raise ValueError(
            "forced next query position requires an active next operation "
            "and a position within the queried prefix")
    if not -1 <= next_query_anchor_focus < span - 1:
        raise ValueError(
            "next query anchor focus must be -1 or a valid nonfinal ordinal")
    if (
            next_query_anchor_focus >= 0
            and (
                next_query_stage < (1 if next_query_anchor_focus == 0 else 2)
                or (
                    query_count == 1 and next_query_position >= 0)
                or (
                    query_count > 1 and next_query_position < 0))):
        raise ValueError(
            "next anchor focus requires an enabled stage and either one "
            "natural query or a forced multi-query position")
    if (
            span != 3
            and next_query_stage == 2
            and next_query_anchor_focus < 0):
        raise ValueError(
            "span above three requires an explicit next anchor at stage 2")
    if (
            next_query_target_aligned
            and (
                query_count != 1
                or next_query_anchor_focus < 0
                or next_query_position >= 0)):
        raise ValueError(
            "target-aligned next bridge requires one natural focused query")
    if next_query_stage >= 0 and objective != "recognition":
        raise ValueError(
            "next query curriculum requires the recognition objective")
    if not 2 <= vocabulary <= 4:
        raise ValueError("vocabulary must be within [2, 4]")
    generator = torch.Generator().manual_seed(seed)
    sequence, match, permutation_ids = _balanced_logical_content(
        count, span, vocabulary, generator, len(permutations),
        allow_partial_balance=allow_partial_balance)
    independent_mask = torch.ones(count, dtype=torch.bool)
    if span >= 3 and new_slot_difficulty < 1.0:
        # A gradual capacity bridge: only a deterministic fraction of rows
        # receive an independently sampled final item. Other rows make the
        # final identity redundant with item one.
        independent_count = round(count * new_slot_difficulty)
        independent_rows = torch.randperm(
            count, generator=generator)[:independent_count]
        independent_mask = torch.zeros(count, dtype=torch.bool)
        independent_mask[independent_rows] = True
        sequence[~independent_mask, -1] = sequence[~independent_mask, 0]
    offsets = 1 + torch.randint(
        0, vocabulary - 1, (count, span), generator=generator)
    candidates = torch.where(
        match.bool(), sequence, (sequence + offsets) % vocabulary)

    if reverse_presentation:
        sequence = sequence.flip(1)
    if flip_candidates:
        # For vocabulary two this is a strict answer counterfactual. With a
        # larger vocabulary it remains a valid rerender, but need not flip all
        # answers, so the audit reports the verifier's actual changed rows.
        candidates = (candidates + 1) % vocabulary
    presentation = _render_shapes(
        sequence, seed=seed ^ 0x13579BDF, nuisance=nuisance,
        heldout=heldout)
    query_frames_by_target = _render_shapes(
        candidates, seed=seed ^ 0x2468ACE0, nuisance=nuisance,
        heldout=heldout)
    # Query order is independently permuted per lifetime.  Therefore neither
    # elapsed query time nor previous feedback reveals the requested ordinal;
    # the visual cue must be read.
    query_ordinals = torch.empty(count, span, dtype=torch.long)
    for row in range(count):
        # Cross query permutations with all content/answer patterns. Query time
        # is consequently independent of identity, ordinal, and correct action.
        query_ordinals[row] = torch.tensor(
            permutations[int(permutation_ids[row])])
    if span == 3 and query_count >= 2 and query_frontier_difficulty < 1.0:
        frontier_rows = query_ordinals[:, 1] == span - 1
        frontier_indices = torch.where(frontier_rows)[0]
        keep_count = round(
            len(frontier_indices) * query_frontier_difficulty)
        keep = frontier_indices[
            torch.randperm(
                len(frontier_indices), generator=generator)[:keep_count]]
        demote = frontier_rows.clone()
        demote[keep] = False
        moved = query_ordinals[demote, 1].clone()
        query_ordinals[demote, 1] = query_ordinals[demote, 2]
        query_ordinals[demote, 2] = moved
    if span == 3 and query_count >= 2 and query_history_difficulty < 1.0:
        # Keep the hard third-ordinal query frequent while varying only what
        # precedes it. At zero, those rows repeat the third lookup; as the
        # scalar rises, a deterministic fraction retain the original different
        # first lookup. This avoids starving the learner of frontier evidence.
        frontier_rows = query_ordinals[:, 1] == span - 1
        frontier_indices = torch.where(frontier_rows)[0]
        cross_count = round(
            len(frontier_indices) * query_history_difficulty)
        cross = frontier_indices[
            torch.randperm(
                len(frontier_indices), generator=generator)[:cross_count]]
        repeat = frontier_rows.clone()
        repeat[cross] = False
        query_ordinals[repeat, 0] = span - 1
    if span == 3 and query_count >= 3:
        if third_query_history_stage == 0:
            query_ordinals[:, 2] = query_ordinals[:, 1]
        elif third_query_history_stage == 1:
            query_ordinals[:, 2] = query_ordinals[:, 0]
    if (
            span == 3 and query_count == 1 and previous_query_stage >= 1
            and previous_query_scope_difficulty < 1.0):
        # The first relative-operation bridge should not spend a third of its
        # experience on an unrelated direct lookup. Preserve a deterministic
        # fraction of those rows and remap the rest evenly across the direct
        # and previous interpretations of the first valid anchor. Cross each
        # remap with both verifier outcomes so target choice cannot leak it.
        third_rows = query_ordinals[:, 0] == 2
        for first_outcome in range(2):
            for second_outcome in range(2):
                eligible = torch.where(
                    third_rows
                    & (match[:, 0] == first_outcome)
                    & (match[:, 1] == second_outcome))[0]
                shuffled = eligible[
                    torch.randperm(len(eligible), generator=generator)]
                keep_count = round(
                    len(shuffled) * previous_query_scope_difficulty)
                remap = shuffled[keep_count:]
                midpoint = len(remap) // 2
                query_ordinals[remap[:midpoint], 0] = 0
                query_ordinals[remap[midpoint:], 0] = 1
    if previous_query_anchor_focus >= 0 and query_count == 1:
        anchor = previous_query_anchor_focus
        direct_target = anchor + 1
        outside = (
            (query_ordinals[:, 0] != anchor)
            & (query_ordinals[:, 0] != direct_target))
        for previous_outcome in range(2):
            for direct_outcome in range(2):
                eligible = torch.where(
                    outside
                    & (match[:, anchor] == previous_outcome)
                    & (match[:, direct_target] == direct_outcome))[0]
                shuffled = eligible[
                    torch.randperm(len(eligible), generator=generator)]
                midpoint = len(shuffled) // 2
                query_ordinals[shuffled[:midpoint], 0] = anchor
                query_ordinals[shuffled[midpoint:], 0] = direct_target
    if previous_query_position >= 0:
        # Swap the first ordinal into one requested query position. This keeps
        # every identity, answer pattern, and remaining query permutation
        # unchanged while making history depth a single controlled variable.
        forced_ordinal = (
            previous_query_anchor_focus
            if previous_query_anchor_focus >= 0 else 0)
        for row in range(count):
            source = int(
                torch.where(
                    query_ordinals[row] == forced_ordinal)[0][0])
            target = previous_query_position
            moved = int(query_ordinals[row, target])
            query_ordinals[row, target] = forced_ordinal
            query_ordinals[row, source] = moved
    if next_query_target_aligned:
        query_ordinals[:, 0] = next_query_anchor_focus + 1
    elif next_query_anchor_focus >= 0 and query_count == 1:
        anchor = next_query_anchor_focus
        direct_target = anchor
        next_target = anchor + 1
        outside = (
            (query_ordinals[:, 0] != direct_target)
            & (query_ordinals[:, 0] != next_target))
        for direct_outcome in range(2):
            for next_outcome in range(2):
                eligible = torch.where(
                    outside
                    & (match[:, direct_target] == direct_outcome)
                    & (match[:, next_target] == next_outcome))[0]
                shuffled = eligible[
                    torch.randperm(len(eligible), generator=generator)]
                midpoint = len(shuffled) // 2
                query_ordinals[shuffled[:midpoint], 0] = direct_target
                query_ordinals[shuffled[midpoint:], 0] = next_target
    if next_query_position >= 0:
        forced_ordinal = (
            next_query_anchor_focus + 1
            if next_query_anchor_focus >= 0 else 1)
        for row in range(count):
            source = int(
                torch.where(
                    query_ordinals[row] == forced_ordinal)[0][0])
            target = next_query_position
            moved = int(query_ordinals[row, target])
            query_ordinals[row, target] = forced_ordinal
            query_ordinals[row, source] = moved
    query_operations = torch.zeros_like(query_ordinals)
    query_cue_ordinals = query_ordinals.clone()
    if next_query_stage == 3:
        # Balance every valid direct/previous/next interpretation within each
        # target and verifier-outcome class. The operation glyph is therefore
        # independent of the binary answer.
        valid_operations = {
            0: (0, 1),
            1: (0, 1, 2),
            2: (0, 2),
        }
        for target_ordinal, operations in valid_operations.items():
            for outcome in range(2):
                selected = (
                    (query_ordinals == target_ordinal)
                    & (match[:, target_ordinal, None] == outcome))
                flat = torch.where(selected.flatten())[0]
                flat = flat[
                    torch.randperm(len(flat), generator=generator)]
                for index, operation in enumerate(operations):
                    begin = round(index * len(flat) / len(operations))
                    end = round((index + 1) * len(flat) / len(operations))
                    query_operations.flatten()[flat[begin:end]] = operation
        previous = query_operations == 1
        following = query_operations == 2
        query_cue_ordinals[previous] += 1
        query_cue_ordinals[following] -= 1
    elif previous_query_stage >= 1:
        if previous_query_anchor_focus >= 0:
            previous = query_ordinals == previous_query_anchor_focus
        else:
            previous = query_ordinals == 0
        if previous_query_stage == 2 and previous_query_anchor_focus < 0:
            # Admit the second valid anchor without removing direct queries
            # with the same cue. Select half of each verifier-outcome class so
            # the operation glyph cannot leak the answer distribution.
            second_anchor_rows = torch.zeros(count, dtype=torch.bool)
            for outcome in range(2):
                eligible = torch.where(match[:, 1] == outcome)[0]
                selected = eligible[
                    torch.randperm(
                        len(eligible), generator=generator
                    )[:len(eligible) // 2]]
                second_anchor_rows[selected] = True
            previous |= (
                (query_ordinals == 1)
                & second_anchor_rows[:, None])
        query_operations[previous] = 1
        query_cue_ordinals[previous] += 1
    elif next_query_stage >= 1:
        if next_query_target_aligned:
            following = torch.zeros_like(
                query_ordinals, dtype=torch.bool)
            target_ordinal = next_query_anchor_focus + 1
            for outcome in range(2):
                eligible = torch.where(
                    match[:, target_ordinal] == outcome)[0]
                selected = eligible[
                    torch.randperm(
                        len(eligible), generator=generator
                    )[:len(eligible) // 2]]
                following[selected, 0] = True
        elif next_query_anchor_focus >= 0:
            following = query_ordinals == next_query_anchor_focus + 1
        else:
            following = query_ordinals == 1
        if next_query_stage == 2 and next_query_anchor_focus < 0:
            # Keep direct and next interpretations of both shared cues. Select
            # half of every target/outcome class so the glyph cannot leak the
            # answer distribution.
            following = torch.zeros_like(query_ordinals, dtype=torch.bool)
            for target_ordinal in (1, 2):
                selected_rows = torch.zeros(count, dtype=torch.bool)
                for outcome in range(2):
                    eligible = torch.where(
                        match[:, target_ordinal] == outcome)[0]
                    selected = eligible[
                        torch.randperm(
                            len(eligible), generator=generator
                        )[:len(eligible) // 2]]
                    selected_rows[selected] = True
                following |= (
                    (query_ordinals == target_ordinal)
                    & selected_rows[:, None])
        query_operations[following] = 2
        query_cue_ordinals[following] -= 1
    source_ordinals = query_ordinals.clone()
    gather_frames = source_ordinals[:, :, None, None, None].expand_as(
        query_frames_by_target)
    queries = torch.gather(query_frames_by_target, 1, gather_frames)
    candidates = torch.gather(candidates, 1, source_ordinals)
    if flip_query_operations:
        if next_query_stage == 3:
            cue_zero = query_cue_ordinals == 0
            cue_one = query_cue_ordinals == 1
            cue_two = query_cue_ordinals == 2
            query_operations = torch.where(
                cue_zero, 2 - query_operations, query_operations)
            query_operations = torch.where(
                cue_two, 1 - query_operations, query_operations)
            query_operations = torch.where(
                cue_one, (query_operations + 1) % 3, query_operations)
        elif next_query_stage >= 1:
            valid_flip = query_cue_ordinals < span - 1
            query_operations = torch.where(
                valid_flip, 2 - query_operations, query_operations)
        else:
            valid_flip = query_cue_ordinals > 0
            query_operations = torch.where(
                valid_flip, 1 - query_operations, query_operations)
        query_ordinals = torch.where(
            query_operations == 1,
            query_cue_ordinals - 1,
            torch.where(
                query_operations == 2,
                query_cue_ordinals + 1,
                query_cue_ordinals))
    queries = _add_query_cues(
        queries, query_cue_ordinals, query_operations,
        show_operation_cue=(
            previous_query_stage >= 0 or next_query_stage >= 0))
    answers = (
        candidates.clone()
        if objective == "visible_identity"
        else (
            candidates
            == torch.gather(sequence, 1, query_ordinals)).long())
    queries = queries[:, :query_count]
    candidates = candidates[:, :query_count]
    answers = answers[:, :query_count]
    query_ordinals = query_ordinals[:, :query_count]
    query_cue_ordinals = query_cue_ordinals[:, :query_count]
    query_operations = query_operations[:, :query_count]
    if blank_presentation:
        backgrounds = presentation[:, :, :, :1, :1]
        presentation = backgrounds.expand_as(presentation).clone()
    lifetime_ids = torch.arange(
        seed, seed + count, dtype=torch.long, device=device)
    return ProceduralShapeBatch(
        presentation_frames=presentation.to(device),
        query_frames=queries.to(device),
        correct_actions=answers.to(device),
        sequence_identities=sequence.to(device),
        candidate_identities=candidates.to(device),
        query_ordinals=query_ordinals.to(device),
        query_cue_ordinals=query_cue_ordinals.to(device),
        query_operations=query_operations.to(device),
        new_slot_independent=independent_mask.to(device),
        logical_lifetime_ids=lifetime_ids,
        objective=objective)


def _reset_active_keep_workspace(state: ControllerState) -> ControllerState:
    return ControllerState(
        torch.zeros_like(state.hidden), state.workspace,
        torch.zeros_like(state.latest_event))


def binary_outcome_complete_targets(
        attempted_actions: torch.Tensor,
        outcomes: torch.Tensor) -> torch.Tensor:
    """Recover the unique binary answer from own action and scalar outcome."""
    if ACTIONS != 2:
        raise ValueError("outcome completion is exact only for two actions")
    return torch.where(
        outcomes > 0.5, attempted_actions, 1 - attempted_actions)


def project_gradient_against_reference(
        named_parameters: list[tuple[str, torch.nn.Parameter]],
        reference_gradient: dict[str, torch.Tensor],
        strength: float) -> tuple[bool, float | None, float | None]:
    """Remove a target-gradient component that opposes verified rehearsal."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("projection strength must be within [0, 1]")
    target_norm_sq = torch.zeros(
        (), device=named_parameters[0][1].device)
    reference_norm_sq = torch.zeros_like(target_norm_sq)
    dot = torch.zeros_like(target_norm_sq)
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        reference = reference_gradient[name]
        dot += torch.sum(parameter.grad * reference)
        target_norm_sq += torch.sum(parameter.grad.square())
        reference_norm_sq += torch.sum(reference.square())
    if float(target_norm_sq) == 0.0 or float(reference_norm_sq) == 0.0:
        return False, None, None
    cosine = float(dot / torch.sqrt(target_norm_sq * reference_norm_sq))
    applied = float(dot) < 0.0 and strength > 0.0
    if applied:
        coefficient = strength * dot / reference_norm_sq
        with torch.no_grad():
            for name, parameter in named_parameters:
                if parameter.grad is not None:
                    parameter.grad.sub_(
                        coefficient * reference_gradient[name])
    post_dot = torch.zeros_like(dot)
    for name, parameter in named_parameters:
        if parameter.grad is not None:
            post_dot += torch.sum(
                parameter.grad * reference_gradient[name])
    return applied, cosine, float(post_dot)


def project_parameter_update_against_reference(
        named_parameters: list[tuple[str, torch.nn.Parameter]],
        parameters_before: dict[str, torch.Tensor],
        reference_gradient: dict[str, torch.Tensor],
        strength: float) -> tuple[bool, float | None, float | None]:
    """Remove an actual optimizer update that would raise rehearsal loss."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("projection strength must be within [0, 1]")
    update_norm_sq = torch.zeros(
        (), device=named_parameters[0][1].device)
    reference_norm_sq = torch.zeros_like(update_norm_sq)
    dot = torch.zeros_like(update_norm_sq)
    for name, parameter in named_parameters:
        update = parameter.detach() - parameters_before[name]
        reference = reference_gradient[name]
        dot += torch.sum(update * reference)
        update_norm_sq += torch.sum(update.square())
        reference_norm_sq += torch.sum(reference.square())
    if float(update_norm_sq) == 0.0 or float(reference_norm_sq) == 0.0:
        return False, None, None
    cosine = float(dot / torch.sqrt(update_norm_sq * reference_norm_sq))
    applied = float(dot) > 0.0 and strength > 0.0
    if applied:
        coefficient = strength * dot / reference_norm_sq
        with torch.no_grad():
            for name, parameter in named_parameters:
                parameter.sub_(coefficient * reference_gradient[name])
    post_dot = torch.zeros_like(dot)
    for name, parameter in named_parameters:
        update = parameter.detach() - parameters_before[name]
        post_dot += torch.sum(update * reference_gradient[name])
    return applied, cosine, float(post_dot)


def rollout_procedural_shape_span(
        model: UnifiedCognitiveController, batch: ProceduralShapeBatch, *,
        sample_actions: bool, exploration: float = 0.10,
        query_thought_steps: int = 0,
        new_slot_novelty_weight: float = 1.0,
        query_history_novelty_weight: float = 1.0,
        previous_conflict_novelty_weight: float = 1.0,
        next_conflict_novelty_weight: float = 1.0,
        next_nonconflict_novelty_weight: float = 1.0,
        hard_example_focal_gamma: float = 0.0,
        complete_binary_outcomes: bool = False,
        disable_workspace: bool = False,
        reset_active_before_query: bool = False,
        reset_all_before_query: bool = False,
        blank_ordinal_cues: bool = False,
        blank_operation_cues: bool = False,
        shuffle_outcomes: bool = False) -> dict[str, torch.Tensor]:
    device = batch.presentation_frames.device
    if query_thought_steps < 0:
        raise ValueError("query thought steps cannot be negative")
    state = model.initial_state(batch.batch_size, device=device)
    null = torch.full(
        (batch.batch_size,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(batch.batch_size, device=device)
    if batch.objective == "recognition":
        for index in range(batch.span):
            _, state = model.step(
                batch.presentation_frames[:, index], state, null, zeros, zeros,
                disable_workspace=disable_workspace)
    if reset_all_before_query:
        state = model.initial_state(batch.batch_size, device=device)
    elif reset_active_before_query:
        state = _reset_active_keep_workspace(state)

    actions, rewards, losses, logits = [], [], [], []
    previous_action, previous_reward = null, zeros
    query_count = int(batch.query_frames.shape[1])
    for index in range(query_count):
        frame = batch.query_frames[:, index]
        if blank_ordinal_cues:
            frame = frame.clone()
            frame[:, :, 28:30, :] = frame[:, :, :1, :1]
        if blank_operation_cues:
            frame = frame.clone()
            background = frame[:, :, :1, :1]
            frame[:, :, 2:8, 2:8] = background
            frame[:, :, 2:8, 24:30] = background
            frame[:, :, 20:26, 24:30] = background
        has_feedback = torch.full_like(previous_reward, float(index > 0))
        # A generic extra read/write cycle lets an answer that was bound into
        # workspace by the query be read on the following internal step. It
        # sees only the same pixel frame and past scalar feedback; no verifier
        # state, answer, or semantic feature is exposed.
        for _ in range(query_thought_steps):
            _, state = model.step(
                frame, state, previous_action,
                previous_reward * has_feedback, has_feedback,
                disable_workspace=disable_workspace)
        output, state = model.step(
            frame, state, previous_action,
            previous_reward * has_feedback, has_feedback,
            disable_workspace=disable_workspace)
        probabilities = torch.softmax(output.logits, dim=-1)
        if sample_actions:
            behavior = (
                probabilities * (1.0 - exploration) + exploration / ACTIONS)
            action = torch.multinomial(behavior, 1).squeeze(1)
        else:
            action = output.logits.argmax(dim=-1)
        reward = (action == batch.correct_actions[:, index]).float()
        delivered = reward.roll(1) if shuffle_outcomes else reward
        if complete_binary_outcomes:
            per_attempt_loss = torch.nn.functional.cross_entropy(
                output.logits,
                binary_outcome_complete_targets(action, delivered),
                reduction="none")
        elif new_slot_novelty_weight != 1.0:
            selected_logit = output.logits.gather(
                1, action.unsqueeze(1)).squeeze(1)
            per_attempt_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                selected_logit, delivered, reduction="none")
        else:
            losses.append(attempted_success_loss(
                output.logits, action, delivered))
            per_attempt_loss = None
        if per_attempt_loss is not None:
            frontier = (
                batch.new_slot_independent
                & (batch.query_ordinals[:, index] == batch.span - 1)
                if batch.span >= 3
                else torch.zeros_like(
                    batch.new_slot_independent, dtype=torch.bool))
            weights = torch.where(
                frontier,
                torch.full_like(
                    per_attempt_loss, new_slot_novelty_weight),
                torch.ones_like(per_attempt_loss))
            crossed_history = (
                (batch.query_ordinals[:, index] == batch.span - 1)
                & (batch.query_ordinals[:, 0] != batch.span - 1)
                if batch.span >= 3 and index > 0
                else torch.zeros_like(frontier))
            weights = weights * torch.where(
                crossed_history,
                torch.full_like(
                    per_attempt_loss, query_history_novelty_weight),
                torch.ones_like(per_attempt_loss))
            previous_conflict = (
                (batch.query_operations[:, index] == 1)
                & (
                    torch.gather(
                        batch.sequence_identities, 1,
                        batch.query_ordinals[
                            :, index:index + 1]).squeeze(1)
                    != torch.gather(
                        batch.sequence_identities, 1,
                        batch.query_cue_ordinals[
                            :, index:index + 1]).squeeze(1)))
            weights = weights * torch.where(
                previous_conflict,
                torch.full_like(
                    per_attempt_loss, previous_conflict_novelty_weight),
                torch.ones_like(per_attempt_loss))
            next_conflict = (
                (batch.query_operations[:, index] == 2)
                & (
                    torch.gather(
                        batch.sequence_identities, 1,
                        batch.query_ordinals[
                            :, index:index + 1]).squeeze(1)
                    != torch.gather(
                        batch.sequence_identities, 1,
                        batch.query_cue_ordinals[
                            :, index:index + 1]).squeeze(1)))
            weights = weights * torch.where(
                next_conflict,
                torch.full_like(
                    per_attempt_loss, next_conflict_novelty_weight),
                torch.ones_like(per_attempt_loss))
            next_nonconflict = (
                (batch.query_operations[:, index] == 2)
                & ~next_conflict)
            weights = weights * torch.where(
                next_nonconflict,
                torch.full_like(
                    per_attempt_loss, next_nonconflict_novelty_weight),
                torch.ones_like(per_attempt_loss))
            if hard_example_focal_gamma > 0.0:
                # Generic verifier-driven curriculum: examples the current
                # controller already solves contribute less, without exposing
                # semantic subgroup metadata to the learner.
                focal = (
                    1.0 - torch.exp(-per_attempt_loss.detach())
                ).pow(hard_example_focal_gamma)
                weights = weights * (
                    focal / focal.mean().clamp_min(1e-12))
            losses.append((per_attempt_loss * weights).sum() / weights.sum())
        actions.append(action)
        rewards.append(reward)
        logits.append(output.logits)
        previous_action, previous_reward = action, delivered
    return {
        "actions": torch.stack(actions, dim=1),
        "rewards": torch.stack(rewards, dim=1),
        "logits": torch.stack(logits, dim=1),
        "loss": torch.stack(losses).mean(),
        "final_hidden": state.hidden,
        "final_workspace": state.workspace,
    }


@torch.no_grad()
def evaluate_procedural_shape_span(
        model: UnifiedCognitiveController, *, count: int, span: int,
        vocabulary: int, seed: int, nuisance: ShapeNuisance,
        device: torch.device,
        objective: str = "recognition",
        query_count: int | None = None,
        new_slot_difficulty: float = 1.0,
        query_frontier_difficulty: float = 1.0,
        query_history_difficulty: float = 1.0,
        third_query_history_stage: int = 2,
        previous_query_stage: int = -1,
        previous_query_scope_difficulty: float = 1.0,
        previous_query_position: int = -1,
        previous_query_anchor_focus: int = -1,
        next_query_stage: int = -1,
        next_query_position: int = -1,
        next_query_anchor_focus: int = -1,
        next_query_target_aligned: bool = False,
        query_thought_steps: int = 0) -> dict[str, object]:
    model.eval()
    kwargs = dict(
        count=count, span=span, vocabulary=vocabulary, seed=seed,
        nuisance=nuisance, heldout=True, objective=objective,
        query_count=query_count, new_slot_difficulty=new_slot_difficulty,
        query_frontier_difficulty=query_frontier_difficulty,
        query_history_difficulty=query_history_difficulty,
        third_query_history_stage=third_query_history_stage,
        previous_query_stage=previous_query_stage,
        previous_query_scope_difficulty=previous_query_scope_difficulty,
        previous_query_position=previous_query_position,
        previous_query_anchor_focus=previous_query_anchor_focus,
        next_query_stage=next_query_stage,
        next_query_position=next_query_position,
        next_query_anchor_focus=next_query_anchor_focus,
        next_query_target_aligned=next_query_target_aligned,
        device=device)
    normal_batch = generate_procedural_shape_batch(**kwargs)
    blank_batch = generate_procedural_shape_batch(
        **kwargs, blank_presentation=True)
    reverse_batch = generate_procedural_shape_batch(
        **kwargs, reverse_presentation=True)
    flipped_batch = generate_procedural_shape_batch(
        **kwargs, flip_candidates=True)
    operation_flipped_batch = generate_procedural_shape_batch(
        **kwargs, flip_query_operations=True)

    def run(batch: ProceduralShapeBatch, **controls: bool):
        return rollout_procedural_shape_span(
            model, batch, sample_actions=False,
            query_thought_steps=query_thought_steps, **controls)

    normal = run(normal_batch)
    blank = run(blank_batch)
    reverse = run(reverse_batch)
    flipped = run(flipped_batch)
    operation_flipped = run(operation_flipped_batch)
    cue_blank = run(normal_batch, blank_ordinal_cues=True)
    operation_blank = run(normal_batch, blank_operation_cues=True)
    workspace_off = run(normal_batch, disable_workspace=True)
    active_reset = run(normal_batch, reset_active_before_query=True)
    all_reset = run(normal_batch, reset_all_before_query=True)

    def accuracy(result: dict[str, torch.Tensor]) -> float:
        return float(result["rewards"].mean())

    ordinal_accuracies = []
    for ordinal in range(span):
        selected = normal_batch.query_ordinals == ordinal
        ordinal_accuracies.append(
            float(normal["rewards"][selected].mean())
            if bool(selected.any()) else None)
    query_position_and_ordinal_accuracies = []
    for query_position in range(normal_batch.query_ordinals.shape[1]):
        row = []
        for ordinal in range(span):
            selected = (
                normal_batch.query_ordinals[:, query_position] == ordinal)
            row.append(
                float(normal["rewards"][:, query_position][selected].mean())
                if bool(selected.any()) else None)
        query_position_and_ordinal_accuracies.append(row)
    new_slot_selected = (
        normal_batch.new_slot_independent[:, None]
        & (normal_batch.query_ordinals == span - 1)
        if span >= 3
        else torch.zeros_like(
            normal_batch.query_ordinals, dtype=torch.bool))
    new_slot_conflict = (
        new_slot_selected
        & (
            normal_batch.sequence_identities[:, -1:]
            != normal_batch.sequence_identities[:, :1]))
    crossed_history = (
        (normal_batch.query_ordinals[:, 1] == span - 1)
        & (normal_batch.query_ordinals[:, 0] != span - 1)
        if span >= 3 and normal_batch.query_ordinals.shape[1] >= 2
        else torch.zeros(
            normal_batch.batch_size, dtype=torch.bool, device=device))
    repeated_history = (
        (normal_batch.query_ordinals[:, 1] == span - 1)
        & (normal_batch.query_ordinals[:, 0] == span - 1)
        if span >= 3 and normal_batch.query_ordinals.shape[1] >= 2
        else torch.zeros(
            normal_batch.batch_size, dtype=torch.bool, device=device))
    operation_accuracies = []
    operation_count = 3 if next_query_stage >= 1 else 2
    for operation in range(operation_count):
        selected = normal_batch.query_operations == operation
        operation_accuracies.append(
            float(normal["rewards"][selected].mean())
            if bool(selected.any()) else None)
    operation_cue_accuracies = []
    for operation in range(operation_count):
        row = []
        for cue_ordinal in range(span):
            selected = (
                (normal_batch.query_operations == operation)
                & (normal_batch.query_cue_ordinals == cue_ordinal))
            row.append(
                float(normal["rewards"][selected].mean())
                if bool(selected.any()) else None)
        operation_cue_accuracies.append(row)
    previous_conflict = (
        (normal_batch.query_operations == 1)
        & (
            torch.gather(
                normal_batch.sequence_identities, 1,
                normal_batch.query_ordinals)
            != torch.gather(
                normal_batch.sequence_identities, 1,
                normal_batch.query_cue_ordinals)))
    next_conflict = (
        (normal_batch.query_operations == 2)
        & (
            torch.gather(
                normal_batch.sequence_identities, 1,
                normal_batch.query_ordinals)
            != torch.gather(
                normal_batch.sequence_identities, 1,
                normal_batch.query_cue_ordinals)))
    next_selected = normal_batch.query_operations == 2
    next_nonconflict = next_selected & ~next_conflict
    next_accuracy_by_conflict_and_action = []
    for conflict in (False, True):
        row = []
        for action in range(ACTIONS):
            selected = (
                next_selected
                & (next_conflict == conflict)
                & (normal_batch.correct_actions == action))
            row.append(
                float(normal["rewards"][selected].mean())
                if bool(selected.any()) else None)
        next_accuracy_by_conflict_and_action.append(row)
    relative_conflict = previous_conflict | next_conflict
    reverse_changed = (
        normal_batch.correct_actions != reverse_batch.correct_actions)
    candidate_changed = (
        normal_batch.correct_actions != flipped_batch.correct_actions)
    operation_changed = (
        normal_batch.correct_actions
        != operation_flipped_batch.correct_actions)
    return {
        "accuracy": accuracy(normal),
        "accuracy_by_ordinal": [
            float(value) for value in normal["rewards"].mean(0)],
        "accuracy_by_presented_ordinal": ordinal_accuracies,
        "accuracy_by_query_position_and_presented_ordinal": (
            query_position_and_ordinal_accuracies),
        "independent_new_slot_queries": int(new_slot_selected.sum()),
        "independent_new_slot_accuracy": (
            float(normal["rewards"][new_slot_selected].mean())
            if bool(new_slot_selected.any()) else None),
        "conflicting_new_slot_queries": int(new_slot_conflict.sum()),
        "conflicting_new_slot_accuracy": (
            float(normal["rewards"][new_slot_conflict].mean())
            if bool(new_slot_conflict.any()) else None),
        "crossed_history_frontier_queries": int(crossed_history.sum()),
        "crossed_history_frontier_accuracy": (
            float(normal["rewards"][:, 1][crossed_history].mean())
            if bool(crossed_history.any()) else None),
        "repeated_history_frontier_queries": int(repeated_history.sum()),
        "repeated_history_frontier_accuracy": (
            float(normal["rewards"][:, 1][repeated_history].mean())
            if bool(repeated_history.any()) else None),
        "accuracy_by_operation": operation_accuracies,
        "accuracy_by_operation_and_cue_ordinal": operation_cue_accuracies,
        "previous_conflict_queries": int(previous_conflict.sum()),
        "previous_conflict_accuracy": (
            float(normal["rewards"][previous_conflict].mean())
            if bool(previous_conflict.any()) else None),
        "next_conflict_queries": int(next_conflict.sum()),
        "next_conflict_accuracy": (
            float(normal["rewards"][next_conflict].mean())
            if bool(next_conflict.any()) else None),
        "next_nonconflict_queries": int(next_nonconflict.sum()),
        "next_nonconflict_accuracy": (
            float(normal["rewards"][next_nonconflict].mean())
            if bool(next_nonconflict.any()) else None),
        "next_accuracy_by_conflict_and_action": (
            next_accuracy_by_conflict_and_action),
        "relative_conflict_queries": int(relative_conflict.sum()),
        "relative_conflict_accuracy": (
            float(normal["rewards"][relative_conflict].mean())
            if bool(relative_conflict.any()) else None),
        "blank_presentation_accuracy": accuracy(blank),
        "blank_ordinal_cue_accuracy": accuracy(cue_blank),
        "blank_operation_cue_accuracy": accuracy(operation_blank),
        "workspace_disabled_accuracy": accuracy(workspace_off),
        "active_state_reset_accuracy": accuracy(active_reset),
        "all_memory_reset_accuracy": accuracy(all_reset),
        "reverse_presentation_accuracy": accuracy(reverse),
        "reverse_changed_fraction": float(reverse_changed.float().mean()),
        "reverse_prediction_flip_rate_on_changed": (
            float((
                normal["actions"][reverse_changed]
                != reverse["actions"][reverse_changed]).float().mean())
            if bool(reverse_changed.any()) else None),
        "candidate_flip_accuracy": accuracy(flipped),
        "candidate_changed_fraction": float(candidate_changed.float().mean()),
        "candidate_prediction_flip_rate_on_changed": (
            float((
                normal["actions"][candidate_changed]
                != flipped["actions"][candidate_changed]).float().mean())
            if bool(candidate_changed.any()) else None),
        "operation_flip_accuracy": accuracy(operation_flipped),
        "operation_changed_fraction": float(operation_changed.float().mean()),
        "operation_prediction_flip_rate_on_changed": (
            float((
                normal["actions"][operation_changed]
                != operation_flipped["actions"][operation_changed]
            ).float().mean())
            if bool(operation_changed.any()) else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=27001)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-episodes", type=int, default=2048)
    parser.add_argument("--span", type=int, default=2)
    parser.add_argument(
        "--query-count", type=int, default=0,
        help="queries per lifetime; zero means query every stored item")
    parser.add_argument(
        "--query-thought-steps", type=int, default=0,
        help=(
            "extra generic internal read/write cycles on each query before "
            "acting; each cycle adds latency but no new information"))
    parser.add_argument(
        "--new-slot-difficulty", type=float, default=1.0,
        help=(
            "for span >=3, fraction of independently sampled final identities; "
            "all evidence remains fully visible so capacity is the only axis"))
    parser.add_argument(
        "--query-frontier-difficulty", type=float, default=1.0,
        help=(
            "fraction of balanced third-ordinal second queries admitted; "
            "zero restricts query two to mastered earlier ordinals"))
    parser.add_argument(
        "--query-history-difficulty", type=float, default=1.0,
        help=(
            "fraction of third-ordinal second queries preceded by a different "
            "lookup; zero repeats the same lookup and one is fully crossed"))
    parser.add_argument(
        "--third-query-history-stage", type=int, choices=(0, 1, 2), default=2,
        help=(
            "third-query curriculum: 0 repeats query two, 1 repeats query one "
            "after a delay, and 2 queries the remaining item"))
    parser.add_argument(
        "--previous-query-stage", type=int, choices=(-1, 0, 1, 2), default=-1,
        help=(
            "relative-query curriculum: -1 is legacy direct lookup without an "
            "operation glyph, 0 adds the direct glyph, 1 crosses direct and "
            "previous lookup at the first valid anchor, and 2 enables both "
            "valid anchors for span three"))
    parser.add_argument(
        "--previous-query-scope-difficulty", type=float, default=1.0,
        help=(
            "for a one-query span-three bridge, fraction of unrelated direct "
            "third-position queries retained; zero focuses experience on the "
            "first direct-versus-previous anchor pair"))
    parser.add_argument(
        "--previous-query-position", type=int, choices=(-1, 0, 1, 2),
        default=-1,
        help=(
            "force the first-anchor previous query into one query position; "
            "-1 preserves the naturally crossed query order"))
    parser.add_argument(
        "--previous-query-anchor-focus", type=int, choices=(-1, 0, 1),
        default=-1,
        help=(
            "for a one-query span-three bridge, balance one previous target "
            "against the direct target sharing its cue; -1 uses full scope"))
    parser.add_argument(
        "--next-query-stage", type=int, choices=(-1, 1, 2, 3), default=-1,
        help=(
            "next-item curriculum: -1 disables it, 1 enables the first valid "
            "anchor, 2 enables both span-three anchors, and 3 balances direct, "
            "previous, and next operations"))
    parser.add_argument(
        "--next-query-position", type=int, choices=(-1, 0, 1, 2), default=-1,
        help=(
            "force the focused next-item target into one query position; -1 "
            "preserves natural query order"))
    parser.add_argument(
        "--next-query-anchor-focus", type=int, default=-1,
        help=(
            "balance one next target against the direct target sharing its "
            "visual cue; -1 uses the stage's natural scope; otherwise use a "
            "nonfinal ordinal"))
    parser.add_argument(
        "--next-query-target-aligned", action="store_true",
        help=(
            "focused bridge that balances direct and next access to the same "
            "stored target before introducing same-cue disambiguation"))
    parser.add_argument(
        "--new-slot-novelty-weight", type=float, default=1.0,
        help=(
            "loss weight for verifier outcomes on independent new-slot "
            "queries; values above one prioritize rare frontier experience"))
    parser.add_argument(
        "--query-history-novelty-weight", type=float, default=1.0,
        help=(
            "loss weight for rare third-ordinal second queries that follow a "
            "different lookup"))
    parser.add_argument(
        "--previous-conflict-novelty-weight", type=float, default=1.0,
        help=(
            "verifier-side loss weight for previous-item queries whose target "
            "identity differs from the directly cued identity"))
    parser.add_argument(
        "--next-conflict-novelty-weight", type=float, default=1.0,
        help=(
            "verifier-side loss weight for next-item queries whose target "
            "identity differs from the directly cued identity"))
    parser.add_argument(
        "--next-nonconflict-novelty-weight", type=float, default=1.0,
        help=(
            "verifier-side curriculum weight for next-item examples whose "
            "target identity equals the directly cued identity"))
    parser.add_argument(
        "--hard-example-focal-gamma", type=float, default=0.0,
        help=(
            "task-agnostic verifier-driven emphasis on currently difficult "
            "examples; zero preserves ordinary outcome loss"))
    parser.add_argument(
        "--complete-binary-outcomes", action="store_true",
        help=(
            "use the exact correct binary action implied by own attempted "
            "action and scalar success/failure"))
    parser.add_argument("--vocabulary", type=int, default=2)
    parser.add_argument(
        "--objective", choices=("visible_identity", "recognition"),
        default="recognition")
    parser.add_argument("--randomness", type=float, default=0.0)
    parser.add_argument(
        "--allow-partial-balance", action="store_true",
        help=(
            "sample a balanced subset when a span's full logical design is "
            "too large for one microbatch; exact balance remains the default"))
    parser.add_argument("--position-px", type=float)
    parser.add_argument("--size-fraction", type=float)
    parser.add_argument("--rotation-degrees", type=float)
    parser.add_argument("--color-delta", type=float)
    parser.add_argument("--background-delta", type=float)
    parser.add_argument("--deformation", type=float)
    parser.add_argument(
        "--rehearse-floor", action="store_true",
        help=(
            "alternate the target nuisance with the mastered randomness-zero "
            "floor; target experience is accounted separately"))
    parser.add_argument(
        "--rehearsal-randomness", default="",
        help=(
            "comma-separated mastered scalar rungs to interleave with the "
            "target; each is audited and accounted separately"))
    parser.add_argument(
        "--rehearsal-spans", default="",
        help=(
            "comma-separated spans aligned with rehearsal-randomness; omit "
            "to rehearse the target span"))
    parser.add_argument(
        "--rehearsal-query-counts", default="",
        help=(
            "comma-separated query counts aligned with rehearsal streams; "
            "omit to query every item in each rehearsal span"))
    parser.add_argument(
        "--rehearsal-new-slot-difficulties", default="",
        help=(
            "comma-separated new-slot difficulties aligned with rehearsal "
            "streams; defaults to fully active"))
    parser.add_argument(
        "--rehearsal-previous-query-stages", default="",
        help=(
            "comma-separated previous-query stages aligned with rehearsal "
            "streams; defaults to legacy direct lookup"))
    parser.add_argument(
        "--rehearsal-next-query-stages", default="",
        help=(
            "comma-separated next-query stages aligned with rehearsal streams; "
            "defaults to disabled"))
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=4)
    parser.add_argument(
        "--addressed-workspace", action="store_true",
        help=(
            "give generic RAM slots distinct learnable addresses; when "
            "upgrading a checkpoint only the two address scales are new"))
    parser.add_argument(
        "--relation-adapter-width", type=int, default=0,
        help=(
            "append a zero-output generic state-query relation adapter when "
            "upgrading a checkpoint"))
    parser.add_argument(
        "--relation-adapter-layer-norm", action="store_true",
        help="normalize the generic state-query vector before the adapter")
    parser.add_argument(
        "--relation-adapter-gated", action="store_true",
        help="learn when the generic relation residual should affect intention")
    parser.add_argument(
        "--train-relation-adapter-only", action="store_true",
        help="freeze inherited parameters and train only the relation adapter")
    parser.add_argument(
        "--action-adapter-width", type=int, default=0,
        help=(
            "append a zero-output generic state-query action residual when "
            "upgrading a checkpoint"))
    parser.add_argument(
        "--action-adapter-gated", action="store_true",
        help="learn when the generic action residual should affect behavior")
    parser.add_argument(
        "--train-action-adapter-only", action="store_true",
        help="freeze inherited parameters and train only the action adapter")
    parser.add_argument("--intention-width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--usage-protection-strength", type=float, default=0.0,
        help=(
            "scale target gradients away from parameters used by recent "
            "experience; zero disables learned selective plasticity"))
    parser.add_argument(
        "--usage-importance-decay", type=float, default=0.95,
        help=(
            "EMA decay for per-parameter usage importance; lower importance "
            "makes an infrequently used parameter more volatile"))
    parser.add_argument(
        "--rehearsal-gradient-projection", type=float, default=0.0,
        help=(
            "project this fraction of a target gradient component that "
            "conflicts with the current cycle's aggregate rehearsal gradient"))
    parser.add_argument(
        "--gradient-projection-mode",
        choices=("aggregate", "per_stream"), default="aggregate",
        help=(
            "protect either the average rehearsal direction or every "
            "rehearsal stream separately"))
    parser.add_argument(
        "--worst-stream-residual-projection", type=float, default=0.0,
        help=(
            "after aggregate projection, softly remove this fraction of the "
            "single worst remaining rehearsal-stream conflict"))
    parser.add_argument(
        "--project-optimizer-update", action="store_true",
        help=(
            "apply the same protection to AdamW's actual parameter update, "
            "after momentum and adaptive scaling"))
    parser.add_argument(
        "--reference-only-rehearsal", action="store_true",
        help=(
            "use rehearsal outcomes to define protected directions without "
            "taking separate old-skill optimizer steps"))
    parser.add_argument(
        "--functional-retention-tolerance", type=float, default=-1.0,
        help=(
            "maximum deterministic rehearsal-accuracy drop allowed after a "
            "target update; a nonnegative value rolls back harmful updates"))
    parser.add_argument(
        "--functional-retention-validation-batch-size", type=int, default=0,
        help=(
            "fresh verifier-generated lifetimes per rehearsal stream used by "
            "a functional retention check; counted as validation experience"))
    parser.add_argument(
        "--functional-retention-stream-indices", default="",
        help=(
            "comma-separated rehearsal-stream indices to validate; empty "
            "checks every stream"))
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--curve-test-episodes", type=int, default=512)
    parser.add_argument("--mastery-threshold", type=float, default=0.90)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument(
        "--shuffle-target-outcomes", action="store_true",
        help=(
            "shuffle only target-task scalar outcomes while keeping "
            "retention rehearsal truthful"))
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    floor_nuisance = nuisance_from_level(0.0)
    nuisance = nuisance_with_overrides(
        nuisance_from_level(args.randomness),
        position_px=args.position_px,
        size_fraction=args.size_fraction,
        rotation_degrees=args.rotation_degrees,
        color_delta=args.color_delta,
        background_delta=args.background_delta,
        deformation=args.deformation)
    rehearsal_levels = [
        float(value) for value in args.rehearsal_randomness.split(",")
        if value]
    if args.rehearse_floor and 0.0 not in rehearsal_levels:
        rehearsal_levels.insert(0, 0.0)
    if any(
            not 0.0 <= level <= 1.0
            for level in rehearsal_levels):
        raise ValueError(
            "rehearsal randomness must be within [0, 1]")
    rehearsal_nuisances = [
        nuisance_from_level(level) for level in rehearsal_levels]
    rehearsal_spans = [
        int(value) for value in args.rehearsal_spans.split(",") if value]
    if rehearsal_spans and len(rehearsal_spans) != len(rehearsal_levels):
        raise ValueError(
            "rehearsal spans must align one-to-one with randomness levels")
    if not rehearsal_spans:
        rehearsal_spans = [args.span] * len(rehearsal_levels)
    if any(span < 1 for span in rehearsal_spans):
        raise ValueError("rehearsal spans must be positive")
    target_query_count = args.query_count or args.span
    if not 1 <= target_query_count <= args.span:
        raise ValueError("query count must be within target span")
    if args.query_thought_steps < 0:
        raise ValueError("query thought steps cannot be negative")
    if args.new_slot_novelty_weight < 1.0:
        raise ValueError("new-slot novelty weight must be at least one")
    if args.query_history_novelty_weight < 1.0:
        raise ValueError("query-history novelty weight must be at least one")
    if args.previous_conflict_novelty_weight < 1.0:
        raise ValueError(
            "previous-conflict novelty weight must be at least one")
    if args.next_conflict_novelty_weight < 1.0:
        raise ValueError(
            "next-conflict novelty weight must be at least one")
    if args.next_nonconflict_novelty_weight < 1.0:
        raise ValueError(
            "next-nonconflict novelty weight must be at least one")
    if args.hard_example_focal_gamma < 0.0:
        raise ValueError("hard-example focal gamma cannot be negative")
    if args.usage_protection_strength < 0.0:
        raise ValueError("usage-protection strength cannot be negative")
    if args.functional_retention_tolerance < -1.0:
        raise ValueError(
            "functional retention tolerance must be -1 (disabled) or nonnegative")
    if (
            args.functional_retention_tolerance >= 0.0
            and args.functional_retention_validation_batch_size < 1):
        raise ValueError(
            "functional retention requires a positive fresh validation batch size")
    if not 0.0 <= args.usage_importance_decay < 1.0:
        raise ValueError("usage-importance decay must be within [0, 1)")
    if not 0.0 <= args.rehearsal_gradient_projection <= 1.0:
        raise ValueError(
            "rehearsal-gradient projection must be within [0, 1]")
    if not 0.0 <= args.worst_stream_residual_projection <= 1.0:
        raise ValueError(
            "worst-stream residual projection must be within [0, 1]")
    if (
            args.worst_stream_residual_projection > 0.0
            and (
                args.rehearsal_gradient_projection == 0.0
                or args.gradient_projection_mode != "aggregate")):
        raise ValueError(
            "worst-stream residual projection requires aggregate rehearsal "
            "gradient projection")
    if (
            args.project_optimizer_update
            and (
                args.rehearsal_gradient_projection == 0.0
                or args.gradient_projection_mode != "aggregate")):
        raise ValueError(
            "optimizer-update projection requires aggregate rehearsal "
            "gradient projection")
    if (
            args.reference_only_rehearsal
            and args.rehearsal_gradient_projection == 0.0):
        raise ValueError(
            "reference-only rehearsal requires rehearsal gradient projection")
    if (
            args.usage_protection_strength > 0.0
            and args.rehearsal_gradient_projection > 0.0):
        raise ValueError(
            "test scalar usage protection and direction-aware projection "
            "separately")
    rehearsal_query_counts = [
        int(value) for value in args.rehearsal_query_counts.split(",")
        if value]
    if (
            rehearsal_query_counts
            and len(rehearsal_query_counts) != len(rehearsal_levels)):
        raise ValueError(
            "rehearsal query counts must align with rehearsal streams")
    if not rehearsal_query_counts:
        rehearsal_query_counts = [
            target_query_count if span == args.span else span
            for span in rehearsal_spans]
    if any(
            not 1 <= count <= span for count, span in zip(
                rehearsal_query_counts, rehearsal_spans)):
        raise ValueError(
            "each rehearsal query count must be within its span")
    rehearsal_slot_difficulties = [
        float(value)
        for value in args.rehearsal_new_slot_difficulties.split(",")
        if value]
    if (
            rehearsal_slot_difficulties
            and len(rehearsal_slot_difficulties) != len(rehearsal_levels)):
        raise ValueError(
            "rehearsal slot difficulties must align with streams")
    if not rehearsal_slot_difficulties:
        rehearsal_slot_difficulties = [1.0] * len(rehearsal_levels)
    if any(
            not 0.0 <= value <= 1.0
            for value in rehearsal_slot_difficulties):
        raise ValueError(
            "rehearsal slot difficulties must be within [0, 1]")
    rehearsal_previous_stages = [
        int(value)
        for value in args.rehearsal_previous_query_stages.split(",")
        if value]
    if (
            rehearsal_previous_stages
            and len(rehearsal_previous_stages) != len(rehearsal_levels)):
        raise ValueError(
            "rehearsal previous-query stages must align with streams")
    if not rehearsal_previous_stages:
        rehearsal_previous_stages = [-1] * len(rehearsal_levels)
    if any(value not in (-1, 0, 1, 2) for value in rehearsal_previous_stages):
        raise ValueError(
            "rehearsal previous-query stages must be -1, 0, 1, or 2")
    rehearsal_next_stages = [
        int(value)
        for value in args.rehearsal_next_query_stages.split(",")
        if value]
    if (
            rehearsal_next_stages
            and len(rehearsal_next_stages) != len(rehearsal_levels)):
        raise ValueError(
            "rehearsal next-query stages must align with streams")
    if not rehearsal_next_stages:
        rehearsal_next_stages = [-1] * len(rehearsal_levels)
    if any(value not in (-1, 1, 2, 3) for value in rehearsal_next_stages):
        raise ValueError(
            "rehearsal next-query stages must be -1, 1, 2, or 3")
    functional_retention_stream_indices = [
        int(value)
        for value in args.functional_retention_stream_indices.split(",")
        if value]
    if not functional_retention_stream_indices:
        functional_retention_stream_indices = list(range(len(rehearsal_levels)))
    if (
            len(set(functional_retention_stream_indices))
            != len(functional_retention_stream_indices)
            or any(
                index < 0 or index >= len(rehearsal_levels)
                for index in functional_retention_stream_indices)):
        raise ValueError(
            "functional-retention stream indices must be unique valid "
            "rehearsal-stream indices")
    if (
            (
                args.usage_protection_strength > 0.0
                or args.rehearsal_gradient_projection > 0.0)
            and not rehearsal_levels):
        raise ValueError(
            "usage protection and gradient projection require rehearsal "
            "experience")
    for span in {args.span, *rehearsal_spans}:
        logical_patterns = (args.vocabulary * 2) ** span
        for name, count in (
                ("batch size", args.batch_size),
                ("test episodes", args.test_episodes),
                ("curve test episodes", args.curve_test_episodes)):
            if count < 1 or (
                    not args.allow_partial_balance
                    and (count < logical_patterns or count % logical_patterns)):
                raise ValueError(
                    f"{name} must be a positive multiple of "
                    f"{logical_patterns} for span {span} unless partial "
                    "balance is enabled")
    configuration: dict[str, object] = {
        "width": args.width, "workspace_slots": args.workspace_slots,
        "intention_width": args.intention_width,
        "workspace_slot_addressing": args.addressed_workspace,
        "relation_adapter_width": args.relation_adapter_width,
        "relation_adapter_layer_norm": args.relation_adapter_layer_norm,
        "relation_adapter_gated": args.relation_adapter_gated,
        "action_adapter_width": args.action_adapter_width,
        "action_adapter_gated": args.action_adapter_gated}
    payload = None
    if args.checkpoint_in is not None:
        payload = torch.load(
            args.checkpoint_in, map_location=device, weights_only=False)
        configuration = dict(payload["model_configuration"])
        if args.addressed_workspace:
            configuration["workspace_slot_addressing"] = True
        if args.relation_adapter_width:
            inherited_width = int(
                configuration.get("relation_adapter_width", 0))
            if inherited_width not in (0, args.relation_adapter_width):
                raise ValueError(
                    "cannot resize an existing relation adapter")
            configuration["relation_adapter_width"] = (
                args.relation_adapter_width)
            configuration["relation_adapter_layer_norm"] = (
                args.relation_adapter_layer_norm)
            configuration["relation_adapter_gated"] = (
                args.relation_adapter_gated)
        if args.action_adapter_width:
            inherited_width = int(
                configuration.get("action_adapter_width", 0))
            if inherited_width not in (0, args.action_adapter_width):
                raise ValueError(
                    "cannot resize an existing action adapter")
            configuration["action_adapter_width"] = args.action_adapter_width
            configuration["action_adapter_gated"] = (
                args.action_adapter_gated)
    model = UnifiedCognitiveController(**configuration).to(device)
    if payload is not None:
        upgrading_workspace = (
            args.addressed_workspace
            and not payload["model_configuration"].get(
                "workspace_slot_addressing", False))
        upgrading_relation = (
            args.relation_adapter_width > 0
            and not payload["model_configuration"].get(
                "relation_adapter_width", 0))
        upgrading_relation_gate = (
            args.relation_adapter_gated
            and not payload["model_configuration"].get(
                "relation_adapter_gated", False))
        upgrading_action = (
            args.action_adapter_width > 0
            and not payload["model_configuration"].get(
                "action_adapter_width", 0))
        upgrading_action_gate = (
            args.action_adapter_gated
            and not payload["model_configuration"].get(
                "action_adapter_gated", False))
        incompatible = model.load_state_dict(
            payload["state_dict"],
            strict=not (
                upgrading_workspace or upgrading_relation
                or upgrading_relation_gate or upgrading_action
                or upgrading_action_gate))
        allowed_missing = {
            "workspace_read_address_scale",
            "workspace_write_address_scale",
        }
        unexpected_missing = {
            name for name in incompatible.missing_keys
            if name not in allowed_missing
            and not (
                upgrading_relation
                and name.startswith("relation_adapter."))
            and not (
                upgrading_relation_gate
                and name.startswith("relation_adapter_gate."))
            and not (
                upgrading_action
                and name.startswith("action_adapter."))
            and not (
                upgrading_action_gate
                and name.startswith("action_adapter_gate."))}
        if (
                unexpected_missing
                or incompatible.unexpected_keys):
            raise RuntimeError(
                "unexpected checkpoint incompatibility: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}")
    if args.train_relation_adapter_only:
        if model.relation_adapter is None:
            raise ValueError(
                "relation-adapter-only training requires an adapter")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("relation_adapter"))
    if args.train_action_adapter_only:
        if args.train_relation_adapter_only:
            raise ValueError(
                "choose only one adapter-only training mode")
        if model.action_adapter is None:
            raise ValueError(
                "action-adapter-only training requires an adapter")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("action_adapter"))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=args.learning_rate, weight_decay=1e-5)
    usage_importance = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad}
    if payload is not None:
        for name, value in payload.get(
                "parameter_usage_importance", {}).items():
            if name in usage_importance:
                usage_importance[name].copy_(value.to(device))
    rehearsal_gradient = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad}
    rehearsal_stream_gradients = [
        {
            name: torch.zeros_like(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad}
        for _ in rehearsal_levels]
    history: list[dict[str, object]] = []
    target_updates = 0
    functional_validation_unique_bits = 0
    functional_validation_evaluation_bits = 0
    functional_anchor_batches: list[ProceduralShapeBatch] = []
    functional_anchor_baseline: list[list[float]] = []
    rehearsal_update_counts = {
        (
            f"span{span}:queries{query_count}:slot{slot}:"
            f"previous{previous}:next{following}:randomness{level}"
        ): 0
        for span, query_count, slot, previous, following, level in zip(
            rehearsal_spans, rehearsal_query_counts,
            rehearsal_slot_difficulties, rehearsal_previous_stages,
            rehearsal_next_stages, rehearsal_levels)}
    rehearsal_verifier_bits = dict.fromkeys(rehearsal_update_counts, 0)
    weight_changing_updates = 0

    def functional_scores(
            validation_batch: ProceduralShapeBatch) -> list[float]:
        """Return overall and causal-conflict accuracy for a fresh stream."""
        validation = rollout_procedural_shape_span(
            model, validation_batch, sample_actions=False,
            query_thought_steps=args.query_thought_steps)
        scores = [float(validation["rewards"].mean())]
        target_identities = torch.gather(
            validation_batch.sequence_identities, 1,
            validation_batch.query_ordinals)
        cue_identities = torch.gather(
            validation_batch.sequence_identities, 1,
            validation_batch.query_cue_ordinals)
        relative_conflict = (
            (validation_batch.query_operations > 0)
            & (target_identities != cue_identities))
        if bool(relative_conflict.any()):
            scores.append(float(
                validation["rewards"][relative_conflict].mean()))
        return scores

    started = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        schedule_index = (step - 1) % (1 + len(rehearsal_levels))
        rehearsal_first = (
            args.usage_protection_strength > 0.0
            or args.rehearsal_gradient_projection > 0.0)
        if rehearsal_first:
            # Observe usage before taking the plastic target step. Every
            # outcome is still counted; only the within-cycle order changes.
            train_on_target = schedule_index == len(rehearsal_levels)
        else:
            train_on_target = schedule_index == 0
        if train_on_target:
            train_nuisance = nuisance
            train_span = args.span
            train_query_count = target_query_count
            train_distribution = "target"
            target_updates += 1
        else:
            rehearsal_index = (
                schedule_index
                if rehearsal_first
                else schedule_index - 1)
            train_nuisance = rehearsal_nuisances[rehearsal_index]
            train_span = rehearsal_spans[rehearsal_index]
            train_query_count = rehearsal_query_counts[rehearsal_index]
            train_slot_difficulty = rehearsal_slot_difficulties[
                rehearsal_index]
            train_previous_stage = rehearsal_previous_stages[
                rehearsal_index]
            train_next_stage = rehearsal_next_stages[rehearsal_index]
            train_distribution = (
                f"rehearsal:span{train_span}:queries{train_query_count}:"
                f"slot{train_slot_difficulty}:"
                f"previous{train_previous_stage}:"
                f"next{train_next_stage}:"
                f"randomness{rehearsal_levels[rehearsal_index]}")
            key = (
                f"span{train_span}:queries{train_query_count}:"
                f"slot{train_slot_difficulty}:"
                f"previous{train_previous_stage}:"
                f"next{train_next_stage}:"
                f"randomness{rehearsal_levels[rehearsal_index]}")
            rehearsal_update_counts[key] += 1
            rehearsal_verifier_bits[key] += (
                args.batch_size * train_query_count)
        batch = generate_procedural_shape_batch(
            args.batch_size, span=train_span, vocabulary=args.vocabulary,
            seed=args.seed + step * args.batch_size, nuisance=train_nuisance,
            objective=args.objective, query_count=train_query_count,
            new_slot_difficulty=(
                args.new_slot_difficulty
                if train_on_target else train_slot_difficulty),
            query_frontier_difficulty=(
                args.query_frontier_difficulty if train_on_target else 1.0),
            query_history_difficulty=(
                args.query_history_difficulty if train_on_target else 1.0),
            third_query_history_stage=(
                args.third_query_history_stage if train_on_target else 2),
            previous_query_stage=(
                args.previous_query_stage
                if train_on_target else train_previous_stage),
            previous_query_scope_difficulty=(
                args.previous_query_scope_difficulty
                if train_on_target else 1.0),
            previous_query_position=(
                args.previous_query_position if train_on_target else -1),
            previous_query_anchor_focus=(
                args.previous_query_anchor_focus if train_on_target else -1),
            next_query_stage=(
                args.next_query_stage
                if train_on_target else train_next_stage),
            next_query_position=(
                args.next_query_position if train_on_target else -1),
            next_query_anchor_focus=(
                args.next_query_anchor_focus if train_on_target else -1),
            next_query_target_aligned=(
                args.next_query_target_aligned if train_on_target else False),
            allow_partial_balance=args.allow_partial_balance,
            device=device)
        result = rollout_procedural_shape_span(
            model, batch, sample_actions=True,
            exploration=args.exploration,
            query_thought_steps=args.query_thought_steps,
            new_slot_novelty_weight=(
                args.new_slot_novelty_weight if train_on_target else 1.0),
            query_history_novelty_weight=(
                args.query_history_novelty_weight
                if train_on_target else 1.0),
            previous_conflict_novelty_weight=(
                args.previous_conflict_novelty_weight
                if train_on_target else 1.0),
            next_conflict_novelty_weight=(
                args.next_conflict_novelty_weight
                if train_on_target else 1.0),
            next_nonconflict_novelty_weight=(
                args.next_nonconflict_novelty_weight
                if train_on_target else 1.0),
            hard_example_focal_gamma=(
                args.hard_example_focal_gamma if train_on_target else 0.0),
            complete_binary_outcomes=args.complete_binary_outcomes,
            shuffle_outcomes=(
                args.shuffle_outcomes
                or (train_on_target and args.shuffle_target_outcomes)))
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        plasticity_values = []
        projection_applied = False
        projection_count = 0
        reference_cosine = None
        post_projection_dot = None
        stream_cosines: list[float | None] = []
        aggregate_post_stream_cosines: list[float | None] = []
        post_projection_stream_cosines: list[float | None] = []
        residual_projection_applied = False
        residual_projection_stream = None
        optimizer_projection_applied = False
        optimizer_update_cosine = None
        optimizer_update_post_dot = None
        optimizer_update_stream_cosines: list[float | None] = []
        optimizer_update_final_stream_cosines: list[float | None] = []
        optimizer_residual_projection_stream = None
        if train_on_target and args.rehearsal_gradient_projection > 0.0:
            named_parameters = list(model.named_parameters())
            for reference in rehearsal_stream_gradients:
                _, cosine, _ = project_gradient_against_reference(
                    named_parameters, reference, 0.0)
                stream_cosines.append(cosine)
            if args.gradient_projection_mode == "aggregate":
                (
                    projection_applied,
                    reference_cosine,
                    post_projection_dot,
                ) = project_gradient_against_reference(
                    named_parameters, rehearsal_gradient,
                    args.rehearsal_gradient_projection)
                projection_count = int(projection_applied)
            else:
                initial_cosines = []
                projection_count = 0
                post_dots = []
                # Cyclic projection prevents a later stream from silently
                # reintroducing a conflict with an earlier protected stream.
                for projection_pass in range(4):
                    pass_applied = False
                    post_dots = []
                    for reference in rehearsal_stream_gradients:
                        applied, cosine, post_dot = (
                            project_gradient_against_reference(
                                named_parameters, reference,
                                args.rehearsal_gradient_projection))
                        if projection_pass == 0 and cosine is not None:
                            initial_cosines.append(cosine)
                        projection_count += int(applied)
                        pass_applied |= applied
                        if post_dot is not None:
                            post_dots.append(post_dot)
                    if not pass_applied:
                        break
                projection_applied = projection_count > 0
                reference_cosine = (
                    min(initial_cosines) if initial_cosines else None)
                post_projection_dot = (
                    min(post_dots) if post_dots else None)
            for reference in rehearsal_stream_gradients:
                _, cosine, _ = project_gradient_against_reference(
                    named_parameters, reference, 0.0)
                aggregate_post_stream_cosines.append(cosine)
            valid_streams = [
                (cosine, index)
                for index, cosine in enumerate(
                    aggregate_post_stream_cosines)
                if cosine is not None]
            if (
                    args.worst_stream_residual_projection > 0.0
                    and projection_applied
                    and valid_streams):
                worst_cosine, worst_index = min(valid_streams)
                if worst_cosine < 0.0:
                    residual_projection_applied, _, _ = (
                        project_gradient_against_reference(
                            named_parameters,
                            rehearsal_stream_gradients[worst_index],
                            args.worst_stream_residual_projection))
                    residual_projection_stream = worst_index
            for reference in rehearsal_stream_gradients:
                _, cosine, _ = project_gradient_against_reference(
                    named_parameters, reference, 0.0)
                post_projection_stream_cosines.append(cosine)
        if train_on_target and args.usage_protection_strength > 0.0:
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                importance = usage_importance[name]
                normalized = importance / importance.mean().clamp_min(1e-12)
                plasticity = 1.0 / (
                    1.0 + args.usage_protection_strength * normalized)
                parameter.grad.mul_(plasticity)
                plasticity_values.append(float(plasticity.mean()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not train_on_target and args.rehearsal_gradient_projection > 0.0:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        rehearsal_gradient[name].add_(parameter.grad)
                        rehearsal_stream_gradients[
                            rehearsal_index][name].add_(parameter.grad)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                usage_importance[name].mul_(
                    args.usage_importance_decay).addcmul_(
                        parameter.grad, parameter.grad,
                        value=1.0 - args.usage_importance_decay)
        protected_named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad]
        functional_before: list[list[float]] = []
        functional_after: list[list[float]] = []
        functional_update_rejected = False
        functional_model_before = None
        functional_optimizer_before = None
        if (
                train_on_target
                and args.functional_retention_tolerance >= 0.0):
            if not functional_anchor_batches:
                for index in functional_retention_stream_indices:
                    functional_anchor_batches.append(
                        generate_procedural_shape_batch(
                            args.functional_retention_validation_batch_size,
                            span=rehearsal_spans[index],
                            vocabulary=args.vocabulary,
                            seed=args.seed + 40_000_000 + index,
                            nuisance=rehearsal_nuisances[index],
                            objective=args.objective,
                            query_count=rehearsal_query_counts[index],
                            new_slot_difficulty=(
                                rehearsal_slot_difficulties[index]),
                            previous_query_stage=(
                                rehearsal_previous_stages[index]),
                            next_query_stage=rehearsal_next_stages[index],
                            allow_partial_balance=(
                                args.allow_partial_balance),
                            device=device))
                functional_validation_unique_bits = sum(
                    validation.batch_size * validation.query_ordinals.shape[1]
                    for validation in functional_anchor_batches)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                for validation_batch in functional_anchor_batches:
                    functional_before.append(functional_scores(validation_batch))
            model.train(was_training)
            if not functional_anchor_baseline:
                functional_anchor_baseline = [
                    scores.copy() for scores in functional_before]
            functional_validation_evaluation_bits += (
                functional_validation_unique_bits)
            functional_model_before = copy.deepcopy(model.state_dict())
            functional_optimizer_before = copy.deepcopy(optimizer.state_dict())
        parameters_before = (
            {
                name: parameter.detach().clone()
                for name, parameter in protected_named_parameters}
            if train_on_target and args.project_optimizer_update
            else {})
        if train_on_target or not args.reference_only_rehearsal:
            optimizer.step()
            weight_changing_updates += 1
        if train_on_target and args.project_optimizer_update:
            (
                optimizer_projection_applied,
                optimizer_update_cosine,
                optimizer_update_post_dot,
            ) = project_parameter_update_against_reference(
                protected_named_parameters, parameters_before,
                rehearsal_gradient, args.rehearsal_gradient_projection)
            for reference in rehearsal_stream_gradients:
                _, cosine, _ = project_parameter_update_against_reference(
                    protected_named_parameters, parameters_before,
                    reference, 0.0)
                optimizer_update_stream_cosines.append(cosine)
            valid_updates = [
                (cosine, index)
                for index, cosine in enumerate(
                    optimizer_update_stream_cosines)
                if cosine is not None]
            if (
                    args.worst_stream_residual_projection > 0.0
                    and optimizer_projection_applied
                    and valid_updates):
                worst_cosine, worst_index = max(valid_updates)
                if worst_cosine > 0.0:
                    project_parameter_update_against_reference(
                        protected_named_parameters, parameters_before,
                        rehearsal_stream_gradients[worst_index],
                        args.worst_stream_residual_projection)
                    optimizer_residual_projection_stream = worst_index
            for reference in rehearsal_stream_gradients:
                _, cosine, _ = project_parameter_update_against_reference(
                    protected_named_parameters, parameters_before,
                    reference, 0.0)
                optimizer_update_final_stream_cosines.append(cosine)
        if functional_model_before is not None:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                for validation_batch in functional_anchor_batches:
                    functional_after.append(functional_scores(validation_batch))
            model.train(was_training)
            functional_validation_evaluation_bits += (
                functional_validation_unique_bits)
            functional_update_rejected = any(
                after < before - args.functional_retention_tolerance
                for before_stream, after_stream in zip(
                    functional_anchor_baseline, functional_after)
                for before, after in zip(before_stream, after_stream))
            if functional_update_rejected:
                model.load_state_dict(functional_model_before)
                assert functional_optimizer_before is not None
                optimizer.load_state_dict(functional_optimizer_before)
                weight_changing_updates -= 1
        if train_on_target and args.rehearsal_gradient_projection > 0.0:
            for value in rehearsal_gradient.values():
                value.zero_()
            for reference in rehearsal_stream_gradients:
                for value in reference.values():
                    value.zero_()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row: dict[str, object] = {
                "update": step,
                "unique_logical_lifetimes": step * args.batch_size,
                "unique_verifier_bits": (
                    target_updates * args.batch_size * target_query_count
                    + sum(rehearsal_verifier_bits.values())
                    + functional_validation_unique_bits),
                "target_verifier_bits": (
                    target_updates * args.batch_size * target_query_count),
                "functional_validation_unique_bits": (
                    functional_validation_unique_bits),
                "functional_validation_evaluation_bits": (
                    functional_validation_evaluation_bits),
                "floor_rehearsal_bits": (
                    sum(
                        bits for key, bits
                        in rehearsal_verifier_bits.items()
                        if key.startswith(
                            f"span{args.span}:"
                            f"queries{target_query_count}:")
                        and key.endswith("randomness0.0"))),
                "train_distribution": train_distribution,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach())}
            if plasticity_values:
                row["mean_target_plasticity"] = (
                    sum(plasticity_values) / len(plasticity_values))
            if functional_before:
                row["functional_retention_before"] = functional_before
                row["functional_retention_after"] = functional_after
                row["functional_update_rejected"] = functional_update_rejected
            if reference_cosine is not None:
                row["target_rehearsal_gradient_cosine"] = reference_cosine
                row["gradient_projection_applied"] = projection_applied
                row["gradient_projection_count"] = projection_count
                row["post_projection_dot"] = post_projection_dot
                row["target_rehearsal_stream_cosines"] = stream_cosines
                row["aggregate_post_stream_cosines"] = (
                    aggregate_post_stream_cosines)
                row["post_projection_stream_cosines"] = (
                    post_projection_stream_cosines)
                row["residual_projection_applied"] = (
                    residual_projection_applied)
                row["residual_projection_stream"] = (
                    residual_projection_stream)
                if optimizer_update_cosine is not None:
                    row["optimizer_update_rehearsal_cosine"] = (
                        optimizer_update_cosine)
                    row["optimizer_update_projection_applied"] = (
                        optimizer_projection_applied)
                    row["optimizer_update_post_dot"] = (
                        optimizer_update_post_dot)
                    row["optimizer_update_stream_cosines"] = (
                        optimizer_update_stream_cosines)
                    row["optimizer_update_final_stream_cosines"] = (
                        optimizer_update_final_stream_cosines)
                    row["optimizer_residual_projection_stream"] = (
                        optimizer_residual_projection_stream)
            if (
                    step % args.eval_every == 0
                    or step == args.steps):
                curve = evaluate_procedural_shape_span(
                    model, count=args.curve_test_episodes, span=args.span,
                    vocabulary=args.vocabulary,
                    seed=args.seed + 20_000_000 + step,
                    nuisance=nuisance, device=device,
                    objective=args.objective,
                    query_count=target_query_count,
                    new_slot_difficulty=args.new_slot_difficulty,
                    query_frontier_difficulty=args.query_frontier_difficulty,
                    query_history_difficulty=args.query_history_difficulty,
                    third_query_history_stage=args.third_query_history_stage,
                    previous_query_stage=args.previous_query_stage,
                    previous_query_scope_difficulty=(
                        args.previous_query_scope_difficulty),
                    previous_query_position=args.previous_query_position,
                    previous_query_anchor_focus=(
                        args.previous_query_anchor_focus),
                    next_query_stage=args.next_query_stage,
                    next_query_position=args.next_query_position,
                    next_query_anchor_focus=args.next_query_anchor_focus,
                    next_query_target_aligned=args.next_query_target_aligned,
                    query_thought_steps=args.query_thought_steps)
                row["heldout_accuracy"] = curve["accuracy"]
                row["heldout_accuracy_by_presented_ordinal"] = (
                    curve["accuracy_by_presented_ordinal"])
                row[
                    "heldout_accuracy_by_query_position_and_presented_ordinal"
                ] = curve[
                    "accuracy_by_query_position_and_presented_ordinal"]
                row["heldout_independent_new_slot_accuracy"] = (
                    curve["independent_new_slot_accuracy"])
                row["heldout_conflicting_new_slot_accuracy"] = (
                    curve["conflicting_new_slot_accuracy"])
                row["heldout_crossed_history_frontier_accuracy"] = (
                    curve["crossed_history_frontier_accuracy"])
                row["heldout_accuracy_by_operation"] = (
                    curve["accuracy_by_operation"])
                row["heldout_accuracy_by_operation_and_cue_ordinal"] = (
                    curve["accuracy_by_operation_and_cue_ordinal"])
                row["heldout_previous_conflict_accuracy"] = (
                    curve["previous_conflict_accuracy"])
                row["heldout_next_conflict_accuracy"] = (
                    curve["next_conflict_accuracy"])
                row["heldout_relative_conflict_accuracy"] = (
                    curve["relative_conflict_accuracy"])
                row["blank_presentation_accuracy"] = (
                    curve["blank_presentation_accuracy"])
            history.append(row)
            print(json.dumps(row), flush=True)

    audit = evaluate_procedural_shape_span(
        model, count=args.test_episodes, span=args.span,
        vocabulary=args.vocabulary, seed=args.seed + 10_000_000,
        nuisance=nuisance, device=device, objective=args.objective,
        query_count=target_query_count,
        new_slot_difficulty=args.new_slot_difficulty,
        query_frontier_difficulty=args.query_frontier_difficulty,
        query_history_difficulty=args.query_history_difficulty,
        third_query_history_stage=args.third_query_history_stage,
        previous_query_stage=args.previous_query_stage,
        previous_query_scope_difficulty=(
            args.previous_query_scope_difficulty),
        previous_query_position=args.previous_query_position,
        previous_query_anchor_focus=args.previous_query_anchor_focus,
        next_query_stage=args.next_query_stage,
        next_query_position=args.next_query_position,
        next_query_anchor_focus=args.next_query_anchor_focus,
        next_query_target_aligned=args.next_query_target_aligned,
        query_thought_steps=args.query_thought_steps)
    floor_audit = evaluate_procedural_shape_span(
        model, count=args.test_episodes, span=args.span,
        vocabulary=args.vocabulary, seed=args.seed + 11_000_000,
        nuisance=floor_nuisance, device=device, objective=args.objective,
        query_count=target_query_count,
        new_slot_difficulty=args.new_slot_difficulty,
        query_frontier_difficulty=args.query_frontier_difficulty,
        query_history_difficulty=args.query_history_difficulty,
        third_query_history_stage=args.third_query_history_stage,
        previous_query_stage=args.previous_query_stage,
        previous_query_scope_difficulty=(
            args.previous_query_scope_difficulty),
        previous_query_position=args.previous_query_position,
        previous_query_anchor_focus=args.previous_query_anchor_focus,
        next_query_stage=args.next_query_stage,
        next_query_position=args.next_query_position,
        next_query_anchor_focus=args.next_query_anchor_focus,
        next_query_target_aligned=args.next_query_target_aligned,
        query_thought_steps=args.query_thought_steps)
    rehearsal_audits = {
        (
            f"span{span}:queries{query_count}:slot{slot}:"
            f"previous{previous}:next{following}:randomness{level}"
        ):
        evaluate_procedural_shape_span(
            model, count=args.test_episodes, span=span,
            vocabulary=args.vocabulary,
            seed=args.seed + 12_000_000 + index,
            nuisance=rehearsal_nuisances[index], device=device,
            objective=args.objective, query_count=query_count,
            new_slot_difficulty=slot,
            previous_query_stage=previous,
            next_query_stage=following,
            query_thought_steps=args.query_thought_steps)
        for index, (
                span, query_count, slot, previous, following, level
        ) in enumerate(zip(
            rehearsal_spans, rehearsal_query_counts,
            rehearsal_slot_difficulties, rehearsal_previous_stages,
            rehearsal_next_stages, rehearsal_levels))}
    prefixes = [row for row in history if "heldout_accuracy" in row]
    stable_bits = None
    for index, row in enumerate(prefixes):
        if all(
                all(
                    value is None or value >= args.mastery_threshold
                    for query_position in later[
                        "heldout_accuracy_by_query_position_and_presented_ordinal"
                    ]
                    for value in query_position)
                and (
                    later["heldout_independent_new_slot_accuracy"] is None
                    or later["heldout_independent_new_slot_accuracy"]
                    >= args.mastery_threshold)
                and (
                    later["heldout_conflicting_new_slot_accuracy"] is None
                    or later["heldout_conflicting_new_slot_accuracy"]
                    >= args.mastery_threshold)
                and (
                    later["heldout_crossed_history_frontier_accuracy"] is None
                    or later["heldout_crossed_history_frontier_accuracy"]
                    >= args.mastery_threshold)
                and all(
                    value is None or value >= args.mastery_threshold
                    for operation in later[
                        "heldout_accuracy_by_operation_and_cue_ordinal"]
                    for value in operation)
                and (
                    later["heldout_previous_conflict_accuracy"] is None
                    or later["heldout_previous_conflict_accuracy"]
                    >= args.mastery_threshold)
                and (
                    later["heldout_next_conflict_accuracy"] is None
                    or later["heldout_next_conflict_accuracy"]
                    >= args.mastery_threshold)
                and (
                    later["heldout_relative_conflict_accuracy"] is None
                    or later["heldout_relative_conflict_accuracy"]
                    >= args.mastery_threshold)
                for later in prefixes[index:]):
            stable_bits = int(row["target_verifier_bits"])
            break
    report = {
        "schema": "procedural-shape-span-experiment-v1",
        "learner_visible_information": (
            "RGB streams, own opaque actions, scalar attempted-action outcome"),
        "span": args.span,
        "query_count": target_query_count,
        "query_thought_steps": args.query_thought_steps,
        "new_slot_difficulty": args.new_slot_difficulty,
        "query_frontier_difficulty": args.query_frontier_difficulty,
        "query_history_difficulty": args.query_history_difficulty,
        "third_query_history_stage": args.third_query_history_stage,
        "previous_query_stage": args.previous_query_stage,
        "previous_query_scope_difficulty": (
            args.previous_query_scope_difficulty),
        "previous_query_position": args.previous_query_position,
        "previous_query_anchor_focus": args.previous_query_anchor_focus,
        "next_query_stage": args.next_query_stage,
        "next_query_position": args.next_query_position,
        "next_query_anchor_focus": args.next_query_anchor_focus,
        "next_query_target_aligned": args.next_query_target_aligned,
        "new_slot_novelty_weight": args.new_slot_novelty_weight,
        "query_history_novelty_weight": (
            args.query_history_novelty_weight),
        "previous_conflict_novelty_weight": (
            args.previous_conflict_novelty_weight),
        "next_conflict_novelty_weight": args.next_conflict_novelty_weight,
        "next_nonconflict_novelty_weight": (
            args.next_nonconflict_novelty_weight),
        "hard_example_focal_gamma": args.hard_example_focal_gamma,
        "usage_protection_strength": args.usage_protection_strength,
        "usage_importance_decay": args.usage_importance_decay,
        "rehearsal_gradient_projection": (
            args.rehearsal_gradient_projection),
        "gradient_projection_mode": args.gradient_projection_mode,
        "worst_stream_residual_projection": (
            args.worst_stream_residual_projection),
        "project_optimizer_update": args.project_optimizer_update,
        "reference_only_rehearsal": args.reference_only_rehearsal,
        "functional_retention_tolerance": (
            args.functional_retention_tolerance),
        "functional_retention_validation_batch_size": (
            args.functional_retention_validation_batch_size),
        "functional_retention_stream_indices": (
            functional_retention_stream_indices),
        "functional_validation_unique_bits": functional_validation_unique_bits,
        "functional_validation_evaluation_bits": (
            functional_validation_evaluation_bits),
        "binary_outcomes_completed": args.complete_binary_outcomes,
        "train_relation_adapter_only": args.train_relation_adapter_only,
        "train_action_adapter_only": args.train_action_adapter_only,
        "vocabulary": args.vocabulary,
        "objective": args.objective,
        "randomness": args.randomness,
        "nuisance": asdict(nuisance),
        "floor_nuisance": asdict(floor_nuisance),
        "rehearse_floor": args.rehearse_floor,
        "rehearsal_randomness": rehearsal_levels,
        "rehearsal_spans": rehearsal_spans,
        "rehearsal_query_counts": rehearsal_query_counts,
        "rehearsal_new_slot_difficulties": rehearsal_slot_difficulties,
        "rehearsal_previous_query_stages": rehearsal_previous_stages,
        "rehearsal_next_query_stages": rehearsal_next_stages,
        "rehearsal_update_counts": rehearsal_update_counts,
        "rehearsal_verifier_bits": rehearsal_verifier_bits,
        "optimizer_updates": weight_changing_updates,
        "gradient_evaluations": args.steps,
        "unique_logical_lifetimes": args.steps * args.batch_size,
        "unique_verifier_bits": (
            target_updates * args.batch_size * target_query_count
            + sum(rehearsal_verifier_bits.values())),
        "target_verifier_bits": (
            target_updates * args.batch_size * target_query_count),
        "floor_rehearsal_bits": (
            sum(
                bits for key, bits in rehearsal_verifier_bits.items()
                if key.startswith(
                    f"span{args.span}:queries{target_query_count}:")
                and key.endswith("randomness0.0"))),
        "replayed_examples": 0,
        "outcomes_shuffled": args.shuffle_outcomes,
        "target_outcomes_shuffled": args.shuffle_target_outcomes,
        "mastery_threshold": args.mastery_threshold,
        "stable_bits_to_threshold": stable_bits,
        "wall_seconds": perf_counter() - started,
        "model_configuration": configuration,
        "history": history,
        "audit": audit,
        "floor_retention_audit": floor_audit,
        "rehearsal_retention_audits": rehearsal_audits,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "parameter_usage_importance": {
                name: value.detach().cpu()
                for name, value in usage_importance.items()},
            "procedural_shape_span_report": report,
        }, args.checkpoint_out)


if __name__ == "__main__":
    main()
