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
    masks = torch.zeros(2, 4, IMAGE_SIZE, IMAGE_SIZE)
    centers = ((12, 12), (20, 12), (12, 20), (20, 20))
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
    masks = torch.zeros(2, 4, IMAGE_SIZE, IMAGE_SIZE)
    centers = ((12, 12), (20, 12), (12, 20), (20, 20))
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
    masks = torch.zeros(2, 4, IMAGE_SIZE, IMAGE_SIZE)
    centers = ((12, 12), (20, 12), (12, 20), (20, 20))
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


_MASK_BANKS = {
    "bars": _bar_mask_bank(),
    "diamonds": _diamond_mask_bank(),
    "dot_pairs": _dot_pair_mask_bank(),
}

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
}


def generate_lifetimes(
        count: int, trials: int, *, seed: int, heldout: bool = False,
        reverse_rules: bool = False,
        reverse_stimuli: bool = False,
        reverse_contexts: bool = False,
        task: str = "binary_mapping",
        appearance: str = "bars",
        support_trials: int = 1,
        device: torch.device | str = "cpu") -> CognitiveLifetimeBatch:
    """Generate a balanced batch with one uniquely correct opaque action.

    One verified outcome contains enough information to infer the complete
    binary bijection. Every query identity is sampled independently so trial
    position cannot reveal whether the stimulus matches the support event.
    """
    if task not in (
        "constant_action", "visible_identity", "binary_mapping",
        "visible_context", "visible_context_xor", "four_rule", "contextual_mapping",
        "contextual_override", "contextual_composition", "context_rule_xor"):
        raise ValueError(
            "task must be constant_action, visible_identity, "
            "binary_mapping, visible_context, visible_context_xor, four_rule, "
            "contextual_mapping, contextual_override, contextual_composition, "
            "or context_rule_xor")
    if appearance not in _MASK_BANKS:
        raise ValueError(
            f"appearance must be one of {sorted(_MASK_BANKS)}")
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

    if heldout:
        position_choices = torch.tensor([1, 2], dtype=torch.long)
    else:
        position_choices = torch.tensor([0, 3], dtype=torch.long)
    positions = position_choices[torch.randint(
        0, len(position_choices), (count, trials), generator=generator)]
    masks = _MASK_BANKS[appearance][identities, positions]

    backgrounds = (
        0.025 + 0.075 * torch.rand(
            count, 1, 3, 1, 1, generator=generator))
    selected_colors = colors[
        torch.arange(count).unsqueeze(1), identities]
    frames = backgrounds.expand(-1, trials, -1, IMAGE_SIZE, IMAGE_SIZE).clone()
    frames = frames * (1.0 - masks.unsqueeze(2))
    frames = frames + selected_colors.unsqueeze(-1).unsqueeze(-1) * masks.unsqueeze(2)

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
            "contextual_override", "contextual_composition", "context_rule_xor"):
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
            (first_row, last_row), (first_column, last_column) = cue_slot
            frames[
                :, :, :, first_row:last_row,
                first_column:last_column] = 0.98

    if task == "constant_action":
        correct = rule_bits.unsqueeze(1).expand(-1, trials).clone()
    elif task == "visible_identity":
        correct = identities.clone()
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
        context_ids.to(device) if context_ids is not None else None)
