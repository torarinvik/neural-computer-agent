"""Deterministic hidden-rule lifetimes rendered only as RGB observations.

Each lifetime selects one of the two possible bijections between two visible
stimuli and two opaque actions.  The bijection is private verifier state and
changes independently between lifetimes.  A controller can therefore improve
after its first attempted action only by retaining the visible stimulus, its
own action, and the scalar outcome.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


IMAGE_SIZE = 32
ACTIONS = 2
NULL_ACTION = ACTIONS

_TRAIN_PALETTE = torch.tensor([
    [[0.86, 0.22, 0.20], [0.18, 0.72, 0.94]],
    [[0.80, 0.30, 0.88], [0.22, 0.82, 0.42]],
    [[0.96, 0.58, 0.14], [0.18, 0.78, 0.72]],
    [[0.92, 0.34, 0.58], [0.46, 0.70, 0.16]],
], dtype=torch.float32)

_HELDOUT_PALETTE = torch.tensor([
    [[0.94, 0.86, 0.18], [0.24, 0.38, 0.96]],
    [[0.68, 0.92, 0.24], [0.90, 0.26, 0.70]],
    [[0.28, 0.88, 0.86], [0.96, 0.42, 0.20]],
], dtype=torch.float32)


@dataclass(frozen=True)
class CognitiveLifetimeBatch:
    """Public frames plus verifier-private values used only to score actions."""

    frames: torch.Tensor
    correct_actions: torch.Tensor
    stimulus_identities: torch.Tensor
    rule_bits: torch.Tensor
    seeds: torch.Tensor
    # Context is public sensory structure for contextual tasks, but remains
    # verifier-private metadata in reports/tests.  Older callers that build a
    # batch directly need not supply it.
    context_ids: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.frames.shape[0])

    @property
    def trials(self) -> int:
        return int(self.frames.shape[1])


def _balanced_bits(count: int, generator: torch.Generator) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.long) % 2
    return values[torch.randperm(count, generator=generator)]

def _balanced_classes(
        count: int, classes: int,
        generator: torch.Generator) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.long) % classes
    return values[torch.randperm(count, generator=generator)]


def _bar_mask_bank() -> torch.Tensor:
    """Original rectangular identities at four nuisance positions."""
    centers = (
        (12, 12), (20, 12), (12, 20), (20, 20),
        # Non-overlapping comparison positions.  The first pair is horizontal
        # and the held-out pair vertical, so magnitude can graduate across
        # layout without either object touching the other at maximum dilation.
        (6, 16), (26, 16), (16, 6), (16, 26))
    masks = torch.zeros(2, len(centers), IMAGE_SIZE, IMAGE_SIZE)
    for position, (center_x, center_y) in enumerate(centers):
        masks[0, position, center_y - 7:center_y + 8,
              center_x - 3:center_x + 4] = 1.0
        masks[1, position, center_y - 3:center_y + 4,
              center_x - 7:center_x + 8] = 1.0
        # A small shared center removes the trivial "more lit pixels" cue.
        masks[:, position, center_y - 1:center_y + 2,
              center_x - 1:center_x + 2] = 0.35
    return masks


def _diamond_mask_bank() -> torch.Tensor:
    """Novel contours preserving only the tall-versus-wide relation."""
    centers = (
        (12, 12), (20, 12), (12, 20), (20, 20),
        (6, 16), (26, 16), (16, 6), (16, 26))
    masks = torch.zeros(2, len(centers), IMAGE_SIZE, IMAGE_SIZE)
    coordinates = torch.arange(IMAGE_SIZE)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    for position, (center_x, center_y) in enumerate(centers):
        dx = (xx - center_x).abs()
        dy = (yy - center_y).abs()
        masks[0, position] = (2 * dx + dy <= 8).float()
        masks[1, position] = (dx + 2 * dy <= 8).float()
        masks[:, position, center_y - 1:center_y + 2,
              center_x - 1:center_x + 2] = 0.35
    return masks


def _dot_pair_mask_bank() -> torch.Tensor:
    """Disconnected objects preserving only vertical/horizontal arrangement."""
    centers = (
        (12, 12), (20, 12), (12, 20), (20, 20),
        (6, 16), (26, 16), (16, 6), (16, 26))
    masks = torch.zeros(2, len(centers), IMAGE_SIZE, IMAGE_SIZE)
    coordinates = torch.arange(IMAGE_SIZE)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    for position, (center_x, center_y) in enumerate(centers):
        vertical = (
            ((xx - center_x) ** 2 + (yy - (center_y - 5)) ** 2 <= 5)
            | ((xx - center_x) ** 2 + (yy - (center_y + 5)) ** 2 <= 5))
        horizontal = (
            ((xx - (center_x - 5)) ** 2 + (yy - center_y) ** 2 <= 5)
            | ((xx - (center_x + 5)) ** 2 + (yy - center_y) ** 2 <= 5))
        masks[0, position] = vertical.float()
        masks[1, position] = horizontal.float()
    return masks


_ALL_POSITION_MASK_BANKS = {
    "bars": _bar_mask_bank(),
    "diamonds": _diamond_mask_bank(),
    "dot_pairs": _dot_pair_mask_bank(),
}
# Legacy tasks must retain their original four-position renderer exactly.
# Magnitude alone needs the four extra non-overlapping comparison positions.
_MASK_BANKS = {
    name: bank[:, :4].clone()
    for name, bank in _ALL_POSITION_MASK_BANKS.items()
}
_MAGNITUDE_MASK_BANKS = _ALL_POSITION_MASK_BANKS


def _magnitude_mask_levels(mask_bank: torch.Tensor) -> torch.Tensor:
    """Five overlapping absolute sizes for a genuine comparison task.

    Adjacent levels form each pair.  Either object in isolation is therefore
    ambiguous at the three interior sizes; only the two-object comparison is
    deterministic.  Thresholding removes the decorative opacity code before
    resizing, leaving occupied extent as the only magnitude signal.
    """
    binary = (mask_bank > 0).to(mask_bank.dtype)
    dilated_two = F.max_pool2d(
        binary, kernel_size=5, stride=1, padding=2)
    dilated_one = F.max_pool2d(
        binary, kernel_size=3, stride=1, padding=1)
    eroded_one = 1.0 - F.max_pool2d(
        1.0 - binary, kernel_size=3, stride=1, padding=1)
    eroded_two = 1.0 - F.max_pool2d(
        1.0 - binary, kernel_size=5, stride=1, padding=2)
    return torch.stack((
        dilated_two, dilated_one, binary, eroded_one, eroded_two))


def _magnitude_level_indices(
        interval: torch.Tensor, relation: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Map an adjacent interval and order bit to two absolute size levels."""
    larger_level = interval
    smaller_level = interval + 1
    first_large = relation == 0
    return (
        torch.where(first_large, larger_level, smaller_level),
        torch.where(first_large, smaller_level, larger_level))


def _numerosity_mask_bank() -> torch.Tensor:
    """Five dot counts under eight layouts in two non-overlapping fields.

    The first four layouts are used for acquisition and the last four only for
    held-out evaluation.  Layout permutations are nested within a count so the
    number of disconnected components is the stable fact; absolute dot
    positions are nuisance.
    """
    left_slots = (
        (7, 4), (14, 10), (22, 6), (9, 13), (25, 13),
        (18, 3), (5, 10), (18, 13))
    right_slots = tuple((y, IMAGE_SIZE - 1 - x) for y, x in left_slots)
    permutations = (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (2, 4, 1, 5, 0, 7, 3, 6),
        (5, 0, 3, 6, 2, 1, 7, 4),
        (7, 3, 5, 1, 6, 4, 0, 2),
        (1, 6, 4, 0, 7, 2, 5, 3),
        (3, 7, 0, 4, 1, 6, 2, 5),
        (4, 2, 6, 7, 5, 0, 3, 1),
        (6, 5, 7, 2, 3, 1, 4, 0),
    )
    bank = torch.zeros(
        2, 5, len(permutations), IMAGE_SIZE, IMAGE_SIZE)
    for side, slots in enumerate((left_slots, right_slots)):
        for layout, permutation in enumerate(permutations):
            for count in range(1, 6):
                for slot in permutation[:count]:
                    y, x = slots[slot]
                    bank[side, count - 1, layout,
                         y - 1:y + 2, x - 1:x + 2] = 1.0
    return bank


_NUMEROSITY_MASK_BANK = _numerosity_mask_bank()


def _numerosity_count_indices(
        interval: torch.Tensor, relation: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Map an adjacent count interval and order bit to zero-based counts."""
    smaller = interval
    larger = interval + 1
    first_large = relation == 0
    return (
        torch.where(first_large, larger, smaller),
        torch.where(first_large, smaller, larger))


def _numerosity_mass_scale(
        zero_based_count: torch.Tensor, control: float,
        ) -> torch.Tensor:
    """Blend from opaque dots to exactly count-normalized total opacity."""
    count = zero_based_count.to(torch.float32) + 1.0
    return (1.0 - control) + control / count


# Public cue slot of each requested operation, as (rows, columns). Contextual
# tasks without an entry carry no bar, which is itself the direct-context code.
# Every slot must fall where no stimulus glyph and no corner context marker can
# reach, so announcing the operation never occludes the content it applies to,
# and no two operations may share a slot or they render identically while
# demanding different actions. The XOR and composition slots are fixed by
# already promoted controllers and must not move.
_OPERATION_CUE_SLOTS = {
    "visible_context_xor": ((2, 5), (14, 19)),
    "contextual_composition": ((IMAGE_SIZE - 3, IMAGE_SIZE), (14, 19)),
    "contextual_override": ((0, 3), (5, 10)),
    "contextual_mapping": ((0, 3), (19, 24)),
    # Deliberately a wider bar rather than another five-column one. The frozen
    # encoder ends in a global average pool, so two cues of equal area differing
    # only in position are nearly identical to it, and a slot cannot shut on a
    # skill it cannot tell apart. Separating by area is what that encoder can
    # actually see. Check any new slot with probe_cue_separability first.
    "context_rule_xor": ((IMAGE_SIZE - 3, IMAGE_SIZE), (3, 13)),
    # Both slots were chosen by searching candidates against
    # probe_cue_separability rather than by eye. What that search showed is that
    # the band matters far more than the mass: every bottom-band candidate
    # scored 0.22 to 0.58 against the two cues already there, while top-band
    # ones reached 1.01 to 1.09 regardless of area or intensity. Cue capacity is
    # bounded by the frozen encoder, so new slots have to be measured, not
    # assumed.
    "context_identity_and": ((0, 2), (12, 17)),
    "context_identity_or": ((0, 2), (24, 32), 0.45),
}


def generate_lifetimes(
        count: int, trials: int, *, seed: int, heldout: bool = False,
        reverse_rules: bool = False,
        reverse_stimuli: bool = False,
        reverse_contexts: bool = False,
        task: str = "binary_mapping",
        appearance: str = "bars",
        appearance_blend: float | None = None,
        numerosity_mass_control: float = 0.0,
        numerosity_appearance_blend: float = 1.0,
        position_holdout: bool | None = None,
        support_trials: int = 1,
        device: torch.device | str = "cpu") -> CognitiveLifetimeBatch:
    """Generate a balanced batch with one uniquely correct opaque action.

    One verified outcome contains enough information to infer the complete
    binary bijection. Every query identity is sampled independently so trial
    position cannot reveal whether the stimulus matches the support event.
    """
    if task not in (
        "constant_action", "visible_identity", "pair_relation",
        "pair_magnitude", "visible_pair_magnitude",
        "visible_pair_numerosity",
        "binary_mapping",
        "visible_context", "visible_context_xor", "four_rule", "contextual_mapping",
        "contextual_override", "contextual_composition", "context_rule_xor",
        "context_identity_and", "context_identity_or"):
        raise ValueError(
            "task must be constant_action, visible_identity, pair_relation, "
            "pair_magnitude, visible_pair_magnitude, "
            "visible_pair_numerosity, "
            "binary_mapping, visible_context, visible_context_xor, four_rule, "
            "contextual_mapping, contextual_override, contextual_composition, "
            "context_rule_xor, context_identity_and, or context_identity_or")
    if appearance not in _MASK_BANKS:
        raise ValueError(
            f"appearance must be one of {sorted(_MASK_BANKS)}")
    if (
            appearance_blend is not None
            and not 0.0 <= appearance_blend <= 1.0):
        raise ValueError("appearance blend must be within [0, 1]")
    if not 0.0 <= numerosity_mass_control <= 1.0:
        raise ValueError("numerosity mass control must be within [0, 1]")
    if not 0.0 <= numerosity_appearance_blend <= 1.0:
        raise ValueError(
            "numerosity appearance blend must be within [0, 1]")
    if (
            task != "visible_pair_numerosity"
            and (
                numerosity_mass_control != 0.0
                or numerosity_appearance_blend != 1.0)):
        raise ValueError(
            "numerosity controls are only valid for numerosity")
    if count < 2 or count % 2:
        raise ValueError("count must be positive and divisible by two")
    if task in ("four_rule", "contextual_mapping") and count % 4:
        raise ValueError(
            "four-rule tasks require a count divisible by four")
    if trials < 2:
        raise ValueError("at least two trials are required")
    if support_trials < 1 or support_trials >= trials:
        raise ValueError(
            "support_trials must be between one and trials - 1")
    generator = torch.Generator().manual_seed(seed)
    rule_bits = _balanced_classes(
        count, 4 if task in ("four_rule", "contextual_mapping") else 2,
        generator)
    if reverse_rules:
        rule_bits = (
            rule_bits ^ (3 if task == "contextual_mapping" else 1)
            if task in ("four_rule", "contextual_mapping")
            else 1 - rule_bits)
    first_identity = _balanced_bits(count, generator)
    identities = torch.randint(
        0, 2, (count, trials), generator=generator)
    identities[:, 0] = first_identity
    if task in ("binary_mapping", "four_rule") and support_trials >= 2:
        # Both identities are observed with feedback during the easier binding
        # rungs. This constraint disappears at the final one-support rung.
        identities[:, 1] = 1 - first_identity
    if reverse_stimuli:
        identities = 1 - identities

    palette = _HELDOUT_PALETTE if heldout else _TRAIN_PALETTE
    palette_indices = torch.randint(
        0, len(palette), (count,), generator=generator)
    colors = palette[palette_indices]
    # Reverse color orientation independently so identity cannot mean a fixed
    # global hue. The within-lifetime glyph remains the identity carrier.
    color_flip = _balanced_bits(count, generator)
    colors = torch.stack([
        colors[index, [flip, 1 - flip]]
        for index, flip in enumerate(color_flip.tolist())
    ])

    use_heldout_positions = (
        heldout if position_holdout is None else position_holdout)
    if use_heldout_positions:
        position_choices = torch.tensor([1, 2], dtype=torch.long)
    else:
        position_choices = torch.tensor([0, 3], dtype=torch.long)
    positions = position_choices[torch.randint(
        0, len(position_choices), (count, trials), generator=generator)]
    task_mask_banks = (
        _MAGNITUDE_MASK_BANKS
        if task in ("pair_magnitude", "visible_pair_magnitude")
        else _MASK_BANKS)
    if appearance_blend is None:
        mask_bank = task_mask_banks[appearance]
    else:
        # A task-preserving difficulty continuum. Endpoints are the exact bars
        # and diamond renderers; intermediate pixels expose progressively more
        # contour change without revealing a semantic curriculum stage.
        mask_bank = (
            (1.0 - appearance_blend) * task_mask_banks["bars"]
            + appearance_blend * task_mask_banks["diamonds"])
    masks = mask_bank[identities, positions]
    pair_identities = None
    pair_masks = None
    pair_context_ids = None
    if task in (
            "pair_relation", "pair_magnitude",
            "visible_pair_magnitude", "visible_pair_numerosity"):
        # A second simultaneously visible object creates a genuinely different
        # perceptual primitive: infer whether two identities match.  The
        # verifier-private relation is balanced independently of nuisance
        # position, colour, and the unused hidden-rule bit.  The controller
        # receives no relation label, task ID, or correct unattempted action.
        # Balance separately within every lifetime so a previous reward never
        # reveals the next event's answer. A lifetime-constant relation would
        # let recurrence solve the task from feedback while ignoring vision.
        pair_relation = torch.stack([
            _balanced_bits(trials, generator) for _ in range(count)])
        if task == "pair_relation":
            pair_identities = identities ^ pair_relation
            if reverse_contexts:
                # A valid pixel-level counterfactual: change the second object,
                # which flips same<->different on every event while preserving
                # the first object and all private sampling decisions.
                pair_identities = 1 - pair_identities
            pair_context_ids = pair_identities
        elif task in ("pair_magnitude", "visible_pair_magnitude"):
            # Identity and size order vary independently.  The previous
            # same/different primitive may help parse the two objects, but its
            # answer cannot solve this task.  Size order is balanced separately
            # inside every lifetime, so one reward cannot reveal later trials.
            pair_identities = torch.randint(
                0, 2, (count, trials), generator=generator)
            magnitude_interval = torch.randint(
                0, 4, (count, trials), generator=generator)
            if reverse_contexts:
                pair_relation = 1 - pair_relation
            pair_context_ids = pair_relation
        else:
            # The adjacent primitive preserves the already learned abstract
            # greater/less action relation while replacing continuous extent
            # with the number of disconnected visible components.
            pair_identities = identities.clone()
            numerosity_interval = torch.randint(
                0, 4, (count, trials), generator=generator)
            if reverse_contexts:
                pair_relation = 1 - pair_relation
            pair_context_ids = pair_relation
        if task == "pair_relation":
            first_pair_position = 1 if use_heldout_positions else 0
            second_pair_position = 2 if use_heldout_positions else 3
            positions = torch.full_like(identities, first_pair_position)
            pair_positions = torch.full_like(
                identities, second_pair_position)
            masks = mask_bank[identities, positions]
            pair_masks = mask_bank[pair_identities, pair_positions]
        elif task in ("pair_magnitude", "visible_pair_magnitude"):
            first_pair_position = 6 if use_heldout_positions else 4
            second_pair_position = 7 if use_heldout_positions else 5
            positions = torch.full_like(identities, first_pair_position)
            pair_positions = torch.full_like(
                identities, second_pair_position)
            if appearance_blend is None:
                magnitude_levels = _magnitude_mask_levels(mask_bank)
            else:
                # Resize each binary endpoint before blending. Thresholding a
                # blended mask first turns every nonzero blend into the same
                # union contour, destroying the intended gradual curriculum.
                magnitude_levels = (
                    (1.0 - appearance_blend)
                    * _magnitude_mask_levels(
                        _MAGNITUDE_MASK_BANKS["bars"])
                    + appearance_blend
                    * _magnitude_mask_levels(
                        _MAGNITUDE_MASK_BANKS["diamonds"]))
            first_level, second_level = _magnitude_level_indices(
                magnitude_interval, pair_relation)
            masks = magnitude_levels[first_level, identities, positions]
            pair_masks = magnitude_levels[
                second_level, pair_identities, pair_positions]
        else:
            first_count, second_count = _numerosity_count_indices(
                numerosity_interval, pair_relation)
            layout_start = 4 if use_heldout_positions else 0
            first_layout = (
                layout_start
                + torch.randint(
                    0, 4, (count, trials), generator=generator))
            second_layout = (
                layout_start
                + torch.randint(
                    0, 4, (count, trials), generator=generator))
            dot_masks = _NUMEROSITY_MASK_BANK[
                0, first_count, first_layout]
            pair_dot_masks = _NUMEROSITY_MASK_BANK[
                1, second_count, second_layout]
            magnitude_levels = _magnitude_mask_levels(
                _MAGNITUDE_MASK_BANKS["bars"])
            first_pair_position = 6 if use_heldout_positions else 4
            second_pair_position = 7 if use_heldout_positions else 5
            first_positions = torch.full_like(
                identities, first_pair_position)
            second_positions = torch.full_like(
                identities, second_pair_position)
            # Magnitude level zero is largest while count index four is five
            # dots. Reversing the index makes both endpoints express the same
            # abstract greater-than relation.
            bar_masks = magnitude_levels[
                4 - first_count, identities, first_positions]
            pair_bar_masks = magnitude_levels[
                4 - second_count, pair_identities, second_positions]
            masks = (
                (1.0 - numerosity_appearance_blend) * bar_masks
                + numerosity_appearance_blend * dot_masks)
            pair_masks = (
                (1.0 - numerosity_appearance_blend) * pair_bar_masks
                + numerosity_appearance_blend * pair_dot_masks)
            first_scale = _numerosity_mass_scale(
                first_count, numerosity_mass_control)
            second_scale = _numerosity_mass_scale(
                second_count, numerosity_mass_control)
            masks = masks * first_scale.unsqueeze(-1).unsqueeze(-1)
            pair_masks = (
                pair_masks
                * second_scale.unsqueeze(-1).unsqueeze(-1))

    backgrounds = (
        0.025 + 0.075 * torch.rand(
            count, 1, 3, 1, 1, generator=generator))
    selected_colors = colors[
        torch.arange(count).unsqueeze(1), identities]
    frames = backgrounds.expand(-1, trials, -1, IMAGE_SIZE, IMAGE_SIZE).clone()
    frames = frames * (1.0 - masks.unsqueeze(2))
    frames = frames + selected_colors.unsqueeze(-1).unsqueeze(-1) * masks.unsqueeze(2)
    if pair_masks is not None:
        assert pair_identities is not None
        pair_colors = colors[
            torch.arange(count).unsqueeze(1), pair_identities]
        frames = frames * (1.0 - pair_masks.unsqueeze(2))
        frames = (
            frames
            + pair_colors.unsqueeze(-1).unsqueeze(-1)
            * pair_masks.unsqueeze(2))

    # Public nuisance markers change every trial and are independent of rules.
    nuisance_x = torch.randint(
        3, IMAGE_SIZE - 3, (count, trials), generator=generator)
    nuisance_y = torch.randint(
        3, IMAGE_SIZE - 3, (count, trials), generator=generator)
    nuisance_value = (
        0.10 + 0.12 * torch.rand(count, trials, generator=generator))
    batch_indices = torch.arange(count).unsqueeze(1).expand(-1, trials)
    trial_indices = torch.arange(trials).unsqueeze(0).expand(count, -1)
    frames[
        batch_indices, trial_indices, :, nuisance_y, nuisance_x
    ] = nuisance_value.unsqueeze(-1)

    # A context-conditioned mapping contains two independent hidden binary
    # associations.  Its first two support events deliberately cover both
    # contexts, so each outcome supplies information the other cannot.  The
    # visible context token is a public sensory cue; its identity and the
    # mapping remain unavailable to the learner except through RGB/outcomes.
    context_ids = None
    if task in (
            "visible_context", "visible_context_xor", "contextual_mapping",
            "contextual_override", "contextual_composition", "context_rule_xor",
        "context_identity_and", "context_identity_or"):
        context_ids = torch.randint(0, 2, (count, trials), generator=generator)
        if support_trials >= 2:
            context_ids[:, 0] = 0
            context_ids[:, 1] = 1
        if reverse_contexts:
            context_ids = 1 - context_ids
        context_positions = ((3, 3), (IMAGE_SIZE - 4, IMAGE_SIZE - 4))
        for context, (y, x) in enumerate(context_positions):
            selected = context_ids == context
            frames[selected, :, y - 1:y + 2, x - 1:x + 2] = 0.98
        # Public visual operation cue. Without it, the direct-context task and
        # its successors have identical observations while demanding
        # conflicting actions. Each requested operation owns one cue slot, so
        # every contextual task remains observationally distinguishable; the
        # XOR slot is fixed by already-consolidated controllers.
        cue_slot = _OPERATION_CUE_SLOTS.get(task)
        if cue_slot is not None:
            # A slot is (rows, columns) or (rows, columns, intensity). The
            # frozen encoder pools globally, so what it can actually read is
            # roughly lit area times intensity; intensity is the second axis
            # available once distinct areas run out.
            (first_row, last_row), (first_column, last_column) = cue_slot[:2]
            value = cue_slot[2] if len(cue_slot) > 2 else 0.98
            frames[
                :, :, :, first_row:last_row,
                first_column:last_column] = value

    if task == "constant_action":
        correct = rule_bits.unsqueeze(1).expand(-1, trials).clone()
    elif task == "visible_identity":
        correct = identities.clone()
    elif task == "pair_relation":
        assert pair_identities is not None
        # Opaque action zero happens to mean "same" to the verifier and action
        # one "different"; those meanings are never shown to the learner.
        correct = identities ^ pair_identities
    elif task == "pair_magnitude":
        assert pair_context_ids is not None
        # One support outcome identifies the lifetime-private opaque action
        # orientation.  Subsequent answers still require comparing the two
        # visible objects because their size order changes on every event.
        correct = pair_context_ids ^ rule_bits.unsqueeze(1)
    elif task == "visible_pair_magnitude":
        assert pair_context_ids is not None
        # The easiest atom: the verifier's two opaque actions consistently
        # distinguish which visible object is larger.  No semantic action name
        # is exposed; attempted actions receive only scalar outcomes.
        correct = pair_context_ids.clone()
    elif task == "visible_pair_numerosity":
        assert pair_context_ids is not None
        # The action semantics are deliberately aligned with magnitude:
        # opaque action zero means the first field has more components.
        correct = pair_context_ids.clone()
    elif task == "four_rule":
        expanded_rule = rule_bits.unsqueeze(1).expand(-1, trials)
        correct = torch.where(
            expanded_rule < 2,
            expanded_rule,
            identities ^ (expanded_rule - 2))
    elif task == "contextual_mapping":
        assert context_ids is not None
        context_rule = (rule_bits.unsqueeze(1) >> context_ids) & 1
        correct = identities ^ context_rule
    elif task == "contextual_override":
        assert context_ids is not None
        mapping_action = identities ^ rule_bits.unsqueeze(1)
        # The context-one response is a simpler invariant. Counterfactual
        # rerendering complements it together with the hidden context-zero
        # mapping, preserving the standard reversal audit.
        override_action = 1 if reverse_rules else 0
        correct = torch.where(
            context_ids == 0, mapping_action,
            torch.full_like(mapping_action, override_action))
    elif task == "contextual_composition":
        assert context_ids is not None
        # The minimal aligned composition: reuse the old identity/rule
        # mapping and the acquired visible-context bit on every event.
        correct = (
            identities ^ rule_bits.unsqueeze(1) ^ context_ids)
    elif task == "context_identity_and":
        assert context_ids is not None
        # The hidden rule composed with a conjunction of the two visible bits.
        # Reversing the rule still flips every action, so the whole gate suite
        # applies unchanged; only the composition being learned is different.
        correct = rule_bits.unsqueeze(1) ^ (identities & context_ids)
    elif task == "context_identity_or":
        assert context_ids is not None
        correct = rule_bits.unsqueeze(1) ^ (identities | context_ids)
    elif task == "context_rule_xor":
        assert context_ids is not None
        # The hidden rule composed with the visible context, with the stimulus
        # identity deliberately irrelevant. Neither branch is a constant, so
        # nothing here is memorisable without tracking the rule.
        correct = rule_bits.unsqueeze(1) ^ context_ids
    elif task == "visible_context":
        assert context_ids is not None
        correct = context_ids.clone()
    elif task == "visible_context_xor":
        assert context_ids is not None
        correct = identities ^ context_ids
    else:
        correct = identities ^ rule_bits.unsqueeze(1)
    seeds = torch.arange(seed, seed + count, dtype=torch.long)
    return CognitiveLifetimeBatch(
        frames.to(device), correct.to(device), identities.to(device),
        rule_bits.to(device), seeds.to(device),
        (
            pair_context_ids.to(device)
            if pair_context_ids is not None
            else context_ids.to(device) if context_ids is not None else None))
