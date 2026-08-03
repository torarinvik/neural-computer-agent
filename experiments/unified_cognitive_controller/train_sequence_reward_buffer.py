"""Train a generic action adapter from a latent replay buffer.

The buffer contains only the controller-visible recurrent/event latent, the
opaque action that was attempted, and the scalar outcome for that attempt.
Verifier-private correct actions are used only inside the environment while
the buffer is collected.  The inherited controller is frozen; only a
zero-initialized generic action adapter is plastic.  This isolates whether
offline reuse of sparse outcomes improves sample efficiency without changing
the representation or adding semantic labels.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .environment import ACTIONS, NULL_ACTION
from .model import UnifiedCognitiveController
from .train import seed_everything
from .train_sequence_working_memory import (
    evaluate_sequence_memory, generate_sequence_memory_batch)


BUFFER_SCHEMA = "latent-action-outcome-buffer-v1"


def _action_conditioned_policy_loss(
        logits: torch.Tensor, critic_logits: torch.Tensor,
        fresh_mask: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Distill a detached action-success critic into the new policy slot.

    The critic is trained only from attempted opaque actions and scalar
    outcomes.  Detaching its prediction makes this a one-way credit bridge:
    the slot learns to select the critic's currently best action, while the
    critic cannot be rewarded by changing its own target.  Old replay rows
    are excluded by ``fresh_mask`` so this auxiliary cannot rewrite inherited
    behavior.
    """
    if temperature <= 0.0:
        raise ValueError("critic policy temperature must be positive")
    if not bool(fresh_mask.any()):
        return logits.sum() * 0.0
    target = F.softmax(
        critic_logits.detach()[fresh_mask] / temperature, dim=-1)
    return F.kl_div(
        F.log_softmax(logits[fresh_mask], dim=-1), target,
        reduction="batchmean")


def _validate_buffer_tensors(
        tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *, device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and move a disk buffer without accepting hidden labels."""
    features, base_logits, actions, outcomes = tensors
    if features.ndim != 2 or base_logits.ndim != 2:
        raise ValueError("buffer features and logits must be rank-two")
    if actions.ndim != 1 or outcomes.ndim != 1:
        raise ValueError("buffer actions and outcomes must be rank-one")
    count = features.shape[0]
    if any(value.shape[0] != count for value in (
            base_logits, actions, outcomes)):
        raise ValueError("buffer tensors have inconsistent lengths")
    if base_logits.shape[1] != ACTIONS:
        raise ValueError("buffer logits have an unexpected action width")
    if actions.dtype != torch.long:
        raise ValueError("buffer actions must be int64")
    if outcomes.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError("buffer outcomes must be floating point")
    if not bool(torch.isfinite(features).all()) or not bool(
            torch.isfinite(base_logits).all()):
        raise ValueError("buffer contains non-finite latent values")
    if not bool(((actions >= 0) & (actions < ACTIONS)).all()):
        raise ValueError("buffer actions are out of range")
    if not bool(((outcomes == 0) | (outcomes == 1)).all()):
        raise ValueError("buffer outcomes must be binary scalar rewards")
    return tuple(value.to(device) for value in tensors)  # type: ignore[return-value]


def _save_buffer(
        path: Path, tensors: tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor], *,
        parent: Path, stream_specs: list[dict[str, int | str]],
        ) -> None:
    """Persist only controller-visible replay evidence on CPU."""
    features, base_logits, actions, outcomes = tensors
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": BUFFER_SCHEMA,
        "claim_boundary": (
            "Only latent features, opaque attempted actions, and scalar "
            "attempt outcomes are stored; no correct unattempted action or "
            "semantic task label is present."),
        "parent": str(parent),
        "stream_specs": stream_specs,
        "features": features.detach().cpu(),
        "base_logits": base_logits.detach().cpu(),
        "actions": actions.detach().cpu(),
        "outcomes": outcomes.detach().cpu(),
    }, path)


def _load_buffer(
        path: Path, *, device: torch.device,
        ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                   dict[str, object]]:
    """Load a validated buffer and reject schema/label contamination."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != BUFFER_SCHEMA:
        raise ValueError(f"unsupported replay buffer schema in {path}")
    required = ("features", "base_logits", "actions", "outcomes")
    if any(key not in payload for key in required):
        raise ValueError(f"replay buffer is missing required tensors: {path}")
    tensors = _validate_buffer_tensors(
        tuple(payload[key] for key in required), device=device)
    metadata = {
        "path": str(path),
        "parent": payload.get("parent"),
        "stream_specs": payload.get("stream_specs", []),
        "transition_count": int(tensors[0].shape[0]),
    }
    return tensors, metadata


def _skill_slot_logits(
        model: UnifiedCognitiveController, features: torch.Tensor,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor | None, torch.Tensor | None]:
    """Compute one successor-slot residual from cached controller latents.

    Feature order mirrors ``UnifiedCognitiveController.step_event``: hidden
    state, current event, optional pre-step workspace and usage, then optional
    intention.
    Keeping the split explicit prevents a workspace-aware experiment from
    silently training on the wrong slice of an older buffer.
    """
    if not len(model.skill_adapters):
        raise ValueError("skill-slot training requires at least one slot")
    # New rungs append one zero-output slot to an already promoted parent.
    # Always train the final slot; older slots remain frozen and continue to
    # provide the inherited base logits.
    slot_index = len(model.skill_adapters) - 1
    base_width = model.width * 2
    workspace_width = (
        model.workspace_slots * model.width
        if model.skill_adapter_reads_workspace_from is not None else 0)
    usage_width = (
        model.workspace_slots
        if model.skill_adapter_reads_workspace_usage_from is not None else 0)
    intention_width = (
        model.intention_width
        if model.skill_adapter_reads_intention_from is not None else 0)
    event_snapshot_width = (
        model.width
        if getattr(model, "skill_adapter_reads_event_snapshot_from", None)
        is not None else 0)
    age_width = (
        1 if getattr(model, "skill_adapter_reads_event_age_from", None)
        is not None else 0)
    parent_action_width = (
        ACTIONS
        if getattr(model, "skill_adapter_reads_parent_action_from", None)
        is not None else 0)
    parent_entropy_width = (
        1
        if getattr(model, "skill_adapter_reads_parent_entropy_from", None)
        is not None else 0)
    expected_width = (
        base_width + workspace_width + usage_width + intention_width
        + event_snapshot_width + age_width + parent_action_width
        + parent_entropy_width)
    if features.shape[1] != expected_width:
        raise ValueError("cached feature width does not match skill slot")
    slot_features = features[:, :base_width]
    reads = []
    if model.skill_adapter_legacy_read_from is not None:
        if model.relation_adapter is not None:
            reads.append(model.relation_adapter[1](
                model.relation_adapter[0](slot_features)))
        if model.action_adapter is not None:
            reads.append(model.action_adapter[1](
                model.action_adapter[0](slot_features)))
    cursor = base_width
    if workspace_width:
        reads.append(features[:, cursor:cursor + workspace_width])
        cursor += workspace_width
    if usage_width:
        reads.append(features[:, cursor:cursor + usage_width])
        cursor += usage_width
    if intention_width:
        reads.append(features[:, cursor:cursor + intention_width])
        cursor += intention_width
    if event_snapshot_width:
        reads.append(features[:, cursor:cursor + event_snapshot_width])
        cursor += event_snapshot_width
    if age_width:
        reads.append(features[:, cursor:cursor + age_width])
        cursor += age_width
    if parent_action_width:
        reads.append(features[:, cursor:cursor + parent_action_width])
        cursor += parent_action_width
    if parent_entropy_width:
        reads.append(features[:, cursor:cursor + parent_entropy_width])
    if reads:
        projected = model.skill_adapter_read_projections[slot_index](
            torch.cat(reads, dim=-1))
        own_features = torch.cat([slot_features, projected], dim=-1)
    else:
        own_features = slot_features
    score = model.skill_adapter_gates[slot_index](own_features)
    opening = (
        F.leaky_relu(score, model.skill_adapter_gate_leak)
        if model.skill_adapter_gate_mode == "relu"
        else torch.sigmoid(score))
    hidden_read = model.skill_adapters[slot_index][1](
        model.skill_adapters[slot_index][0](own_features))
    residual = model.skill_adapters[slot_index][2](hidden_read) * opening
    if slot_index < len(model.skill_adapter_residual_scales):
        residual = residual * model.skill_adapter_residual_scales[slot_index]
    critic_logits = None
    critic_residual = None
    critic = model.skill_adapter_critics[slot_index]
    if not isinstance(critic, torch.nn.Identity):
        critic_logits = critic(own_features)
        critic_residual = (
            critic_logits - critic_logits.mean(dim=-1, keepdim=True))
        critic_residual = (
            critic_residual * model.skill_adapter_critic_scales[slot_index]
            * opening)
    return (
        model.actuator(residual), residual, opening, score,
        critic_logits, critic_residual)


def _remove_final_slot_from_logits(
        model: UnifiedCognitiveController, features: torch.Tensor,
        logits: torch.Tensor) -> torch.Tensor:
    """Remove an inherited final slot before continuing its training."""
    action_residual, _, _, _, _, critic_residual = _skill_slot_logits(
        model, features)
    result = logits - action_residual
    if critic_residual is not None:
        result = result - critic_residual
    return result


def _replay_refinement_indices(
        all_indices: torch.Tensor, replay_indices: torch.Tensor,
        gate_source_weight: float,
        gate_preserve_fresh_weight: float = 0.0) -> torch.Tensor:
    """Choose the rows that can identify fresh versus persisted provenance.

    With source supervision enabled, both classes must be present.  Keeping
    this as a small pure helper makes the provenance split auditable instead
    of burying a degenerate all-old minibatch inside the training loop.
    """
    return (
        all_indices
        if gate_source_weight or gate_preserve_fresh_weight
        else replay_indices)


def _protected_rehearsal_mask(
        replay_mask: torch.Tensor, *, target_rows: int,
        protect_rehearsal: bool) -> torch.Tensor:
    """Mark current-run rehearsal as protected without changing provenance.

    ``replay_mask`` historically meant persisted rows only.  That made
    retention penalties silently skip freshly collected rehearsal streams.
    This separate mask lets an experiment protect all non-target rehearsal
    rows while preserving the old default and the persisted-buffer provenance
    semantics.
    """
    if replay_mask.ndim != 1:
        raise ValueError("replay mask must be one-dimensional")
    protected = replay_mask.clone()
    if not protect_rehearsal:
        return protected
    if target_rows < 0 or target_rows > protected.numel():
        raise ValueError("target rows must fit within the replay mask")
    # A freshly collected buffer is laid out as target rows followed by
    # rehearsal rows, then any persisted rows. Persisted rows are already
    # marked in replay_mask, so only the non-persisted suffix is added here.
    persisted_start = protected.numel()
    persisted = torch.nonzero(protected, as_tuple=False).flatten()
    if persisted.numel():
        persisted_start = int(persisted.min())
    protected[target_rows:persisted_start] = True
    return protected


def _balanced_provenance_loss(
        score: torch.Tensor, source_target: torch.Tensor) -> torch.Tensor:
    """Use equal total weight for fresh and persisted provenance classes."""
    target = source_target.to(score.dtype)
    positive = target.sum()
    negative = target.numel() - positive
    if not bool(positive) or not bool(negative):
        return F.binary_cross_entropy_with_logits(score, target)
    weights = torch.where(
        target > 0.5,
        negative / positive,
        torch.ones_like(target))
    return (
        F.binary_cross_entropy_with_logits(
            score, target, reduction="none") * weights).mean()


def _outcome_only_query_weights(
        features: torch.Tensor, outcomes: torch.Tensor,
        replay_mask: torch.Tensor, *, age_column: int | None,
        power: float, floor: float) -> torch.Tensor:
    """Prioritize hard stream positions using verifier outcomes only.

    ``event_age`` is a generic stream clock, not a task or answer label.  For
    each fresh age bucket we estimate difficulty as ``1 - mean(outcome)`` and
    normalize the resulting weights to mean one.  Persisted replay rows are
    deliberately left at weight one: old-skill protection should not be
    weakened by a noisy difficulty estimate from the new task.  This is a
    curriculum signal, not privileged supervision; no correct action is read.
    """
    if power < 0.0:
        raise ValueError("query difficulty power must not be negative")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("query difficulty floor must be within [0, 1]")
    weights = torch.ones_like(outcomes)
    if power == 0.0 or not bool((~replay_mask).any()):
        return weights
    if age_column is None:
        raise ValueError(
            "query difficulty requires a generic event-age feature")
    ages = features[:, age_column]
    fresh = ~replay_mask
    fresh_ages = torch.unique(ages[fresh], sorted=True)
    for age in fresh_ages:
        bucket = fresh & (ages == age)
        difficulty = (1.0 - outcomes[bucket].mean()).clamp_min(floor)
        weights[bucket] = difficulty.pow(power)
    mean = weights[fresh].mean().clamp_min(torch.finfo(weights.dtype).eps)
    weights[fresh] = weights[fresh] / mean
    return weights


def _query_curriculum_indices(
        total_rows: int, *, target_lifetimes: int, span: int,
        cutoff: int, device: torch.device) -> torch.Tensor:
    """Keep all replay/rehearsal rows and only a target prefix for warmup."""
    if target_lifetimes < 1 or span < 1:
        raise ValueError("target lifetimes and span must be positive")
    if not 1 <= cutoff <= span:
        raise ValueError("query curriculum cutoff must be within the span")
    target_rows = target_lifetimes * span
    if target_rows > total_rows:
        raise ValueError("target stream is shorter than its declared size")
    target_mask = torch.zeros(total_rows, dtype=torch.bool, device=device)
    target_mask[:target_rows] = True
    query_index = torch.zeros(total_rows, dtype=torch.long, device=device)
    query_index[:target_rows] = torch.arange(
        span, device=device).repeat_interleave(target_lifetimes)
    return torch.arange(total_rows, device=device)[
        (~target_mask) | (query_index < cutoff)]


def _query_window_indices(
        total_rows: int, *, target_lifetimes: int, span: int,
        start: int, end: int, device: torch.device) -> torch.Tensor:
    """Keep replay rows and a selected query window from the fresh target.

    Query positions are a generic stream coordinate, not a semantic task
    label.  A successor slot can therefore be trained only where the parent
    has not already demonstrated the primitive, while all rehearsal and
    persisted-replay rows remain available to protect older skills.
    """
    if target_lifetimes < 1 or span < 1:
        raise ValueError("target lifetimes and span must be positive")
    if not 0 <= start < end <= span:
        raise ValueError("query window must satisfy 0 <= start < end <= span")
    target_rows = target_lifetimes * span
    if target_rows > total_rows:
        raise ValueError("target stream is shorter than its declared size")
    target_mask = torch.zeros(total_rows, dtype=torch.bool, device=device)
    target_mask[:target_rows] = True
    query_index = torch.zeros(total_rows, dtype=torch.long, device=device)
    query_index[:target_rows] = torch.arange(
        span, device=device).repeat_interleave(target_lifetimes)
    selected_target = target_mask & (query_index >= start) & (query_index < end)
    return torch.arange(total_rows, device=device)[~target_mask | selected_target]


def _base_mistake_weights(
        base_logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor, replay_mask: torch.Tensor,
        *, weight: float) -> torch.Tensor:
    """Focus binary complementary updates where the frozen parent is wrong.

    The target is reconstructed only from the attempted opaque action and its
    scalar outcome.  This control is intentionally binary: for two actions a
    failed attempt identifies the alternative, while for a larger action set
    it would invent information and must not be enabled by the trainer.
    Persisted rows stay at weight one so the control cannot silently weaken
    old-skill rehearsal.
    """
    if weight < 1.0:
        raise ValueError("base mistake weight must be at least one")
    weights = torch.ones_like(outcomes)
    if weight == 1.0:
        return weights
    target = torch.where(outcomes > 0.5, attempted, 1 - attempted)
    mistake = base_logits.argmax(dim=1) != target
    weights[mistake & ~replay_mask] = weight
    return weights


def _weighted_attempted_success_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor, replay_mask: torch.Tensor,
        replay_outcome_weight: float,
        positive_outcome_weight: float = 1.0,
        sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Weight old replay outcomes separately from fresh-task outcomes."""
    selected = logits.gather(1, attempted.unsqueeze(1)).squeeze(1)
    per_example = F.binary_cross_entropy_with_logits(
        selected, outcomes, reduction="none")
    # Random action exploration yields roughly one successful attempt for
    # every three failures.  This optional balance changes only how much the
    # same verifier reward contributes; it supplies no correct-action label.
    per_example = per_example * torch.where(
        outcomes > 0.5,
        torch.full_like(per_example, positive_outcome_weight),
        torch.ones_like(per_example))
    weights = torch.where(
        replay_mask,
        torch.full_like(per_example, replay_outcome_weight),
        torch.ones_like(per_example))
    if sample_weight is not None:
        weights = weights * sample_weight
    return (per_example * weights).sum() / weights.sum().clamp_min(1.0)


def _weighted_binary_complement_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor, replay_mask: torch.Tensor,
        replay_outcome_weight: float,
        positive_outcome_weight: float = 1.0,
        sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Diagnostic binary bandit loss using the logically implied alternative.

    With exactly two opaque actions, a failed attempt identifies the other
    action as the successful alternative.  This derives a training target from
    the attempted action and scalar outcome only; it stores or receives no
    correct-action label.  It is intentionally an explicit binary control,
    not a claim about the general N-action architecture.
    """
    if logits.shape[1] != 2:
        raise ValueError("binary complement loss requires exactly two actions")
    target = torch.where(outcomes > 0.5, attempted, 1 - attempted)
    per_example = F.cross_entropy(logits, target, reduction="none")
    per_example = per_example * torch.where(
        outcomes > 0.5,
        torch.full_like(per_example, positive_outcome_weight),
        torch.ones_like(per_example))
    weights = torch.where(
        replay_mask,
        torch.full_like(per_example, replay_outcome_weight),
        torch.ones_like(per_example))
    if sample_weight is not None:
        weights = weights * sample_weight
    return (per_example * weights).sum() / weights.sum().clamp_min(1.0)


def _weighted_binary_margin_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor, replay_mask: torch.Tensor,
        replay_outcome_weight: float, margin: float = 1.0,
        sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Use an outcome-only hinge to correct a frozen binary parent.

    Cross-entropy can saturate when inherited logits have very large margins.
    The hinge keeps a constant gradient while the implied target is still on
    the wrong side of the requested margin.  As with the complement loss, the
    alternative target is valid only for two opaque actions.
    """
    if logits.shape[1] != 2:
        raise ValueError("binary margin loss requires exactly two actions")
    if margin < 0.0:
        raise ValueError("binary margin must not be negative")
    target = torch.where(outcomes > 0.5, attempted, 1 - attempted)
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    other_logit = logits.gather(1, (1 - target).unsqueeze(1)).squeeze(1)
    per_example = F.relu(margin - (target_logit - other_logit))
    weights = torch.where(
        replay_mask,
        torch.full_like(per_example, replay_outcome_weight),
        torch.ones_like(per_example))
    if sample_weight is not None:
        weights = weights * sample_weight
    return (per_example * weights).sum() / weights.sum().clamp_min(1.0)


def _collect_buffer(
        model: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, device: torch.device,
        position_augmentation: bool, include_intention: bool = False,
        include_workspace: bool = False,
        include_workspace_usage: bool = False,
        include_event_snapshot: bool = False,
        include_event_age: bool = False, exploration: float = 1.0,
        include_parent_action: bool = False,
        parent_action_probabilities: bool = False,
        include_parent_entropy: bool = False,
        operation: str = "mixed",
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect latent transitions using uniformly random opaque actions."""
    batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation=operation, position_augmentation=position_augmentation,
        device=device)
    null = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    state = model.initial_state(count, device=device)
    previous_action = null
    previous_reward = zeros
    features: list[torch.Tensor] = []
    base_logits: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(span):
            _, state = model.step(
                batch.input_frames[:, index], state, null, zeros, zeros)
        for index in range(distractors):
            _, state = model.step(
                batch.distractor_frames[:, index], state, null, zeros, zeros)
        for query in range(span):
            has_feedback = torch.full_like(
                previous_reward, float(query > 0))
            frame = batch.query_frames[:, query]
            event = model.vision(frame)
            hidden_before = state.hidden.clone()
            workspace_before = state.workspace.flatten(1).clone()
            usage_before = (
                None if state.workspace_usage is None
                else state.workspace_usage.clone())
            event_snapshot_before = state.latest_event.clone()
            age_before = (
                None if state.event_age is None else state.event_age.clone())
            output, state = model.step(
                frame, state, previous_action,
                previous_reward * has_feedback, has_feedback)
            if not 0.0 <= exploration <= 1.0:
                raise ValueError("buffer exploration must be within [0, 1]")
            probabilities = torch.softmax(output.logits, dim=-1)
            behavior = (
                probabilities * (1.0 - exploration)
                + exploration / ACTIONS)
            action = torch.multinomial(behavior, 1).squeeze(1)
            outcome = (
                action == batch.correct_actions[:, query]).to(torch.float32)
            feature_parts = [hidden_before, event]
            if include_workspace:
                feature_parts.append(workspace_before)
            if include_workspace_usage:
                if usage_before is None:
                    usage_before = torch.zeros(
                        count, model.workspace_slots, device=device)
                feature_parts.append(usage_before)
            if include_intention:
                feature_parts.append(output.intention.detach())
            if include_event_snapshot:
                feature_parts.append(event_snapshot_before)
            if include_event_age:
                if age_before is None:
                    age_before = torch.zeros(count, 1, device=device)
                feature_parts.append(age_before)
            if include_parent_action:
                # For an appended slot, the frozen base output is exactly the
                # inherited action context. Existing-slot continuation removes
                # its inherited final residual before training.
                parent_action = output.logits.detach()
                if parent_action_probabilities:
                    parent_action = torch.softmax(parent_action, dim=-1)
                feature_parts.append(parent_action)
            if include_parent_entropy:
                parent_action = output.logits.detach()
                probabilities = torch.softmax(parent_action, dim=-1)
                entropy = -(
                    probabilities
                    * probabilities.clamp_min(1e-8).log()).sum(
                        dim=-1, keepdim=True)
                feature_parts.append(entropy / math.log(ACTIONS))
            features.append(torch.cat(feature_parts, dim=-1))
            base_logits.append(output.logits.detach())
            actions.append(action)
            outcomes.append(outcome)
            previous_action = action
            previous_reward = outcome
    return (
        torch.cat(features), torch.cat(base_logits), torch.cat(actions),
        torch.cat(outcomes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument(
        "--buffer-in", type=Path,
        help="load this exact buffer as the complete training set")
    parser.add_argument(
        "--replay-buffer-in", type=Path,
        help="append this persisted buffer to freshly collected streams")
    parser.add_argument(
        "--buffer-out", type=Path,
        help="save the collected training streams before outcome shuffling")
    parser.add_argument(
        "--replay-max-transitions", type=int, default=0,
        help=(
            "deterministically subsample a persisted replay buffer before "
            "mixing it with the fresh target stream; zero keeps all"))
    parser.add_argument(
        "--replay-outcome-weight", type=float, default=1.0,
        help="weight old replay outcomes relative to fresh outcomes")
    parser.add_argument(
        "--replay-residual-penalty", type=float, default=0.0,
        help="penalize new residuals on persisted replay transitions")
    parser.add_argument(
        "--replay-gate-penalty", type=float, default=0.0,
        help="penalize successor-slot opening on persisted transitions")
    parser.add_argument(
        "--replay-logit-penalty", type=float, default=0.0,
        help="penalize action-logit changes on persisted transitions")
    parser.add_argument(
        "--protect-rehearsal", action="store_true",
        help=(
            "also apply replay weights and retention penalties to freshly "
            "collected non-target rehearsal rows; default preserves the "
            "historical persisted-only mask"))
    parser.add_argument(
        "--replay-gate-source-weight", type=float, default=0.0,
        help=(
            "train the successor gate to distinguish fresh versus persisted "
            "provenance; this is diagnostic metadata, not a task label"))
    parser.add_argument(
        "--replay-gate-preserve-fresh-weight", type=float, default=0.0,
        help=(
            "during gate-only refinement, regress fresh gate scores to their "
            "post-acquisition values and old scores toward zero"))
    parser.add_argument(
        "--replay-refinement-epochs", type=int, default=0,
        help="after fresh learning, refine only on persisted replay data")
    parser.add_argument(
        "--replay-refine-gate-only", action="store_true",
        help=(
            "during replay refinement, freeze the new residual and train "
            "its gate"))
    parser.add_argument("--seed", type=int, default=30941)
    parser.add_argument(
        "--data-seed", type=int,
        help=(
            "seed freshly collected target/rehearsal streams separately "
            "from model initialization; population arms can share data"))
    parser.add_argument(
        "--buffer-exploration", type=float, default=1.0,
        help=(
            "probability of uniform exploration while collecting replay; "
            "zero follows the frozen parent policy"))
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--rehearse-spans", default="")
    parser.add_argument("--rehearsal-lifetimes", type=int, default=512)
    parser.add_argument("--span", type=int, default=8)
    parser.add_argument(
        "--target-operation",
        choices=("forward", "reverse", "mixed", "complement"),
        default="mixed",
        help=(
            "operation for the new target stream; complement is an adjacent "
            "primitive"))
    parser.add_argument(
        "--test-operation",
        choices=("forward", "reverse", "mixed", "complement"),
        default=None,
        help="held-out operation; defaults to target-operation")
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--positive-outcome-weight", type=float, default=1.0,
        help=(
            "multiply observed successful-attempt loss; this is a reward "
            "balance control, not a correct-action label"))
    parser.add_argument(
        "--query-difficulty-power", type=float, default=0.0,
        help=(
            "outcome-only curriculum: upweight fresh stream positions with "
            "lower observed success; requires the generic event-age read"))
    parser.add_argument(
        "--query-difficulty-floor", type=float, default=0.25,
        help=(
            "minimum per-position difficulty before applying the query "
            "difficulty power"))
    parser.add_argument(
        "--query-curriculum-warmup-epochs", type=int, default=0,
        help=(
            "train a fresh target prefix for this many epochs before the "
            "full query sequence; requires a freshly collected target"))
    parser.add_argument(
        "--query-curriculum-cutoff", type=int, default=0,
        help=(
            "number of initial query positions in the curriculum warmup; "
            "zero disables the staged prefix"))
    parser.add_argument(
        "--query-window-start", type=int, default=0,
        help=(
            "first fresh-target query position trained by the successor "
            "slot; zero with --query-window-end zero disables the window"))
    parser.add_argument(
        "--query-window-end", type=int, default=0,
        help=(
            "exclusive fresh-target query position for a generic training "
            "window; zero means the end of the span"))
    parser.add_argument(
        "--base-mistake-weight", type=float, default=1.0,
        help=(
            "binary diagnostic: upweight fresh rows where the frozen parent "
            "disagrees with the outcome-implied action"))
    parser.add_argument(
        "--binary-complement-loss", action="store_true",
        help=(
            "diagnostic only: for two opaque actions, treat a failed attempt "
            "as evidence for the other action; no unattempted label is stored"))
    parser.add_argument(
        "--binary-complement-critic-loss", action="store_true",
        help=(
            "apply the same binary-only outcome inference to the temporary "
            "per-action critic; diagnostic control"))
    parser.add_argument(
        "--binary-margin-loss", action="store_true",
        help=(
            "diagnostic only: replace binary policy cross-entropy with a "
            "constant-gradient outcome-implied margin loss"))
    parser.add_argument(
        "--binary-margin-critic-loss", action="store_true",
        help="apply the binary margin loss to the temporary critic")
    parser.add_argument(
        "--binary-margin", type=float, default=1.0,
        help="requested target-vs-alternative logit margin")
    parser.add_argument("--action-adapter-width", type=int, default=64)
    parser.add_argument(
        "--skill-adapter-width", type=int, default=0,
        help=(
            "train one new successor skill slot while preserving the "
            "parent action adapter"))
    parser.add_argument(
        "--skill-adapter-residual-scale", type=float, default=1.0,
        help=(
            "fixed positive amplitude for the newly appended slot residual; "
            "zero output still preserves insertion exactly"))
    parser.add_argument(
        "--append-skill-slot", action="store_true",
        help=(
            "append the trained slot to a parent that already has promoted "
            "skill slots; existing slots remain frozen"))
    parser.add_argument(
        "--train-existing-skill-slot", action="store_true",
        help=(
            "continue the parent's final skill slot with protected replay "
            "instead of appending another slot"))
    parser.add_argument(
        "--action-conditioned-critic-width", type=int, default=0,
        help=(
            "add a zero-impact per-action success predictor trained from "
            "attempted-action rewards"))
    parser.add_argument(
        "--action-conditioned-critic-weight", type=float, default=1.0,
        help="weight the critic's attempted-outcome loss")
    parser.add_argument(
        "--action-conditioned-critic-policy-weight", type=float, default=0.0,
        help=(
            "distill the detached critic's per-action success estimate into "
            "the fresh successor-slot policy after warmup"))
    parser.add_argument(
        "--action-conditioned-critic-policy-warmup", type=int, default=4,
        help="critic-only epochs before policy distillation begins")
    parser.add_argument(
        "--action-conditioned-critic-policy-temperature", type=float,
        default=1.0,
        help="temperature for the detached critic policy target")
    parser.add_argument(
        "--skill-adapter-gate-mode", choices=("sigmoid", "relu"),
        default="sigmoid")
    parser.add_argument("--skill-adapter-gate-hidden", type=int, default=0)
    parser.add_argument(
        "--skill-adapter-reads-intention", action="store_true",
        help="give the successor slot the parent's latent intention summary")
    parser.add_argument(
        "--skill-adapter-reads-workspace", action="store_true",
        help=(
            "give the successor slot the pre-query short-term workspace; "
            "this is the learned RAM-routing experiment"))
    parser.add_argument(
        "--skill-adapter-reads-workspace-usage", action="store_true",
        help=(
            "give the successor slot the controller's EMA access frequency "
            "for each RAM slot"))
    parser.add_argument(
        "--skill-adapter-reads-event-age", action="store_true",
        help=(
            "give the successor slot a normalized generic stream clock; "
            "this is not a task or operation label"))
    parser.add_argument(
        "--skill-adapter-reads-parent-action", action="store_true",
        help=(
            "give the successor slot the inherited controller action logits "
            "as a generic context/novelty signal"))
    parser.add_argument(
        "--skill-adapter-reads-parent-entropy", action="store_true",
        help=(
            "give the successor slot one normalized parent-action entropy "
            "scalar as a generic uncertainty/context signal"))
    parser.add_argument(
        "--skill-adapter-parent-action-probabilities", action="store_true",
        help=(
            "normalize the inherited action context to probabilities before "
            "the successor slot reads it"))
    parser.add_argument(
        "--skill-adapter-read-bottleneck", type=int, default=0,
        help=(
            "compress a wide workspace/legacy read before the successor "
            "slot; zero keeps the raw concatenation"))
    parser.add_argument(
        "--skill-adapter-no-legacy-read", action="store_true",
        help=(
            "do not feed the parent adapter hidden state to the successor; "
            "use this as a matched workspace-only routing control"))
    parser.add_argument(
        "--train-parent-action-adapter", action="store_true",
        help=(
            "jointly adapt the inherited action reader with the new slot; "
            "old logits are protected by replay/rehearsal when supplied"))
    parser.add_argument(
        "--train-workspace-address-scales", action="store_true",
        help=(
            "train generic RAM read/write address scales alongside the new "
            "slot; requires a parent with workspace_slot_addressing"))
    parser.add_argument(
        "--train-workspace-write-address-content", action="store_true",
        help=(
            "train the generic address-conditioned write residual so RAM "
            "rows can store distinct content"))
    parser.add_argument(
        "--residual-action-adapter", action="store_true",
        help=(
            "inherit the parent's action adapter in the frozen base and "
            "train a new zero-output adapter as an additive residual"))
    parser.add_argument(
        "--residual-penalty", type=float, default=0.0,
        help="L2 penalty on a learned action residual")
    parser.add_argument("--test-episodes", type=int, default=512)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument("--position-augmentation", action="store_true")
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if (
            args.train_lifetimes < 2
            or args.train_lifetimes % 2
            or args.rehearsal_lifetimes < 2
            or args.rehearsal_lifetimes % 2
            or args.span < 1
            or args.distractors < 0
            or args.epochs < 1
            or args.batch_size < 1
            or args.action_adapter_width < 1
            or args.skill_adapter_width < 0
            or args.action_conditioned_critic_width < 0
            or args.skill_adapter_read_bottleneck < 0
            or args.skill_adapter_gate_hidden < 0
            or args.query_curriculum_warmup_epochs < 0
            or args.query_curriculum_cutoff < 0
            or args.query_window_start < 0
            or args.query_window_end < 0
            or args.test_episodes < 2):
        raise ValueError("invalid replay-buffer dimensions")
    if args.replay_max_transitions < 0:
        raise ValueError("replay-max-transitions must not be negative")
    if not 0.0 <= args.buffer_exploration <= 1.0:
        raise ValueError("buffer exploration must be within [0, 1]")
    if args.replay_refinement_epochs < 0:
        raise ValueError("replay-refinement-epochs must not be negative")
    if args.residual_penalty < 0:
        raise ValueError("residual-penalty must not be negative")
    if args.positive_outcome_weight < 0:
        raise ValueError("positive-outcome-weight must not be negative")
    if args.query_difficulty_power < 0:
        raise ValueError("query-difficulty-power must not be negative")
    if not 0.0 <= args.query_difficulty_floor <= 1.0:
        raise ValueError("query-difficulty-floor must be within [0, 1]")
    if args.query_curriculum_warmup_epochs and not args.query_curriculum_cutoff:
        raise ValueError(
            "query-curriculum-cutoff is required for a warmup curriculum")
    if args.query_curriculum_cutoff > args.span:
        raise ValueError("query-curriculum-cutoff cannot exceed span")
    query_window_active = bool(
        args.query_window_start or args.query_window_end)
    query_window_end = args.query_window_end or args.span
    if query_window_active and not (
            0 <= args.query_window_start < query_window_end <= args.span):
        raise ValueError(
            "query window must satisfy 0 <= start < end <= span")
    if args.skill_adapter_residual_scale <= 0.0:
        raise ValueError("skill-adapter-residual-scale must be positive")
    if args.train_existing_skill_slot and not args.skill_adapter_width:
        raise ValueError(
            "continuing a skill slot requires --skill-adapter-width")
    if args.train_existing_skill_slot and args.append_skill_slot:
        raise ValueError(
            "continuation cannot append a second slot at the same time")
    if args.base_mistake_weight < 1.0:
        raise ValueError("base-mistake-weight must be at least one")
    if args.base_mistake_weight > 1.0 and not args.binary_complement_loss:
        raise ValueError(
            "base-mistake-weight requires the binary complement diagnostic")
    if (args.binary_margin_loss or args.binary_margin_critic_loss) \
            and not args.binary_complement_loss:
        raise ValueError(
            "binary margin diagnostics require the binary complement loss")
    if args.binary_margin < 0.0:
        raise ValueError("binary-margin must not be negative")
    if args.query_curriculum_warmup_epochs and args.buffer_in is not None:
        raise ValueError(
            "query curriculum needs a freshly collected target stream")
    if query_window_active and args.buffer_in is not None:
        raise ValueError(
            "query windows need a freshly collected target stream")
    if args.query_curriculum_warmup_epochs and args.replay_refinement_epochs:
        raise ValueError(
            "query curriculum cannot be combined with staged replay refinement")
    if query_window_active and args.replay_refinement_epochs:
        raise ValueError(
            "query windows cannot be combined with staged replay refinement")
    if args.action_conditioned_critic_weight < 0:
        raise ValueError("action-conditioned-critic-weight must not be negative")
    if args.action_conditioned_critic_policy_weight < 0:
        raise ValueError(
            "action-conditioned-critic-policy-weight must not be negative")
    if args.action_conditioned_critic_policy_warmup < 0:
        raise ValueError(
            "action-conditioned-critic-policy-warmup must not be negative")
    if args.action_conditioned_critic_policy_temperature <= 0.0:
        raise ValueError(
            "action-conditioned-critic-policy-temperature must be positive")
    if (args.replay_outcome_weight < 0
            or args.replay_residual_penalty < 0
            or args.replay_gate_penalty < 0
            or args.replay_logit_penalty < 0
            or args.replay_gate_source_weight < 0
            or args.replay_gate_preserve_fresh_weight < 0):
        raise ValueError("replay weights and penalties must not be negative")
    if args.skill_adapter_width and args.residual_action_adapter:
        raise ValueError(
            "skill-adapter-width and residual-action-adapter are exclusive")
    if args.skill_adapter_reads_intention and not args.skill_adapter_width:
        raise ValueError("intention reads require a skill slot")
    if args.skill_adapter_reads_workspace and not args.skill_adapter_width:
        raise ValueError("workspace reads require a skill slot")
    if (args.skill_adapter_reads_workspace_usage
            and not args.skill_adapter_width):
        raise ValueError("workspace-usage reads require a skill slot")
    if args.skill_adapter_reads_event_age and not args.skill_adapter_width:
        raise ValueError("event-age reads require a skill slot")
    if args.skill_adapter_reads_parent_action and not args.skill_adapter_width:
        raise ValueError("parent-action reads require a skill slot")
    if args.skill_adapter_reads_parent_entropy and not args.skill_adapter_width:
        raise ValueError("parent entropy reads require a skill slot")
    if (args.skill_adapter_parent_action_probabilities
            and not args.skill_adapter_reads_parent_action):
        raise ValueError(
            "parent-action probabilities require parent-action reads")
    if args.append_skill_slot and not args.skill_adapter_width:
        raise ValueError("appending a skill slot requires --skill-adapter-width")
    if (args.action_conditioned_critic_width
            and not args.skill_adapter_width):
        raise ValueError("the critic requires a skill slot")
    if args.train_parent_action_adapter and not args.skill_adapter_width:
        raise ValueError("parent action adaptation requires a skill slot")
    if args.train_workspace_address_scales and not args.skill_adapter_width:
        raise ValueError("workspace address training requires a skill slot")
    if (args.train_workspace_write_address_content
            and not args.skill_adapter_width):
        raise ValueError("address-conditioned writes require a skill slot")
    if args.replay_refine_gate_only and not args.skill_adapter_width:
        raise ValueError("gate-only replay refinement requires a skill slot")
    if args.replay_refinement_epochs and args.replay_buffer_in is None:
        raise ValueError("replay refinement requires --replay-buffer-in")
    seed_everything(args.seed)
    data_seed = args.seed if args.data_seed is None else args.data_seed
    device = torch.device(args.device)
    payload = torch.load(args.parent, map_location=device, weights_only=False)
    base_configuration = dict(payload["model_configuration"])
    # A successor slot inherits these generic reads from the parent. Cache the
    # same pre-step tensors that the slot reads; otherwise append runs can
    # train against a narrower representation than the live model consumes.
    include_intention = (
        args.skill_adapter_reads_intention
        or base_configuration.get("skill_adapter_reads_intention_from")
        is not None)
    include_workspace = (
        args.skill_adapter_reads_workspace
        or base_configuration.get("skill_adapter_reads_workspace_from")
        is not None)
    include_workspace_usage = (
        args.skill_adapter_reads_workspace_usage
        or base_configuration.get("skill_adapter_reads_workspace_usage_from")
        is not None)
    include_event_snapshot = (
        base_configuration.get("skill_adapter_reads_event_snapshot_from")
        is not None)
    include_event_age = (
        args.skill_adapter_reads_event_age
        or base_configuration.get("skill_adapter_reads_event_age_from")
        is not None)
    include_parent_action = (
        args.skill_adapter_reads_parent_action
        or base_configuration.get("skill_adapter_reads_parent_action_from")
        is not None)
    include_parent_entropy = (
        args.skill_adapter_reads_parent_entropy
        or base_configuration.get("skill_adapter_reads_parent_entropy_from")
        is not None)
    parent_action_probabilities = (
        args.skill_adapter_parent_action_probabilities
        or base_configuration.get("skill_adapter_parent_action_probabilities", False))
    base = UnifiedCognitiveController(**base_configuration).to(device)
    compatibility = base.load_state_dict(
        payload["state_dict"], strict=False)
    allowed_missing = set()
    if base_configuration.get("workspace_slot_addressing", False):
        allowed_missing.update({
            "workspace_read_address_scale",
            "workspace_write_address_scale",
            "workspace_write_content_address_scale",
        })
    if not set(compatibility.missing_keys).issubset(allowed_missing):
        raise RuntimeError(
            "parent checkpoint/configuration mismatch: "
            f"missing={compatibility.missing_keys}, "
            f"unexpected={compatibility.unexpected_keys}")
    if compatibility.unexpected_keys:
        raise RuntimeError(
            "parent checkpoint/configuration mismatch: "
            f"unexpected={compatibility.unexpected_keys}")
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    rehearsal_spans = tuple(
        int(value) for value in args.rehearse_spans.split(",") if value)
    if any(value < 1 for value in rehearsal_spans):
        raise ValueError("rehearsal spans must be positive")
    stream_specs: list[dict[str, int | str]] = []
    loaded_buffer_metadata: list[dict[str, object]] = []
    persisted_count = 0
    persisted_parent_matches_target = False
    if args.buffer_in is not None:
        (features, base_logits, actions, outcomes), metadata = _load_buffer(
            args.buffer_in, device=device)
        loaded_buffer_metadata.append(metadata)
        replay_mask = torch.ones(
            features.shape[0], dtype=torch.bool, device=device)
    else:
        target_buffer = _collect_buffer(
            base, count=args.train_lifetimes, span=args.span,
            distractors=args.distractors, seed=data_seed, device=device,
            position_augmentation=args.position_augmentation,
            include_intention=include_intention,
            include_workspace=include_workspace,
            include_workspace_usage=include_workspace_usage,
            include_event_snapshot=include_event_snapshot,
            include_event_age=include_event_age,
            include_parent_action=include_parent_action,
            parent_action_probabilities=parent_action_probabilities,
            include_parent_entropy=include_parent_entropy,
            exploration=args.buffer_exploration,
            operation=args.target_operation)
        buffer_parts = [target_buffer]
        stream_specs.append({
            "kind": "target", "span": args.span,
            "lifetimes": args.train_lifetimes})
        for index, rehearsal_span in enumerate(rehearsal_spans):
            buffer_parts.append(_collect_buffer(
                base, count=args.rehearsal_lifetimes, span=rehearsal_span,
                distractors=args.distractors,
                seed=data_seed + (index + 1) * 1_000_003,
                device=device,
                position_augmentation=args.position_augmentation,
                include_intention=include_intention,
                include_workspace=include_workspace,
                include_workspace_usage=include_workspace_usage,
                include_event_snapshot=include_event_snapshot,
                include_event_age=include_event_age,
                include_parent_action=include_parent_action,
                parent_action_probabilities=parent_action_probabilities,
                include_parent_entropy=include_parent_entropy,
                exploration=args.buffer_exploration,
                operation="mixed"))
            stream_specs.append({
                "kind": "rehearsal", "span": rehearsal_span,
                "lifetimes": args.rehearsal_lifetimes})
        features = torch.cat([value[0] for value in buffer_parts])
        base_logits = torch.cat([value[1] for value in buffer_parts])
        actions = torch.cat([value[2] for value in buffer_parts])
        outcomes = torch.cat([value[3] for value in buffer_parts])
        replay_mask = torch.zeros(
            features.shape[0], dtype=torch.bool, device=device)
        if args.buffer_out is not None:
            _save_buffer(
                args.buffer_out,
                (features, base_logits, actions, outcomes),
                parent=args.parent, stream_specs=stream_specs)
    if args.replay_buffer_in is not None:
        persisted, metadata = _load_buffer(
            args.replay_buffer_in, device=device)
        if persisted[0].shape[1] != features.shape[1]:
            raise ValueError(
                "replay buffer feature width does not match the current "
                "slot interface: "
                f"current={features.shape[1]}, "
                f"replay={persisted[0].shape[1]}; recollect the replay "
                "buffer with the same successor reads")
        loaded_buffer_metadata.append(metadata)
        persisted_count = persisted[0].shape[0]
        persisted_parent = metadata.get("parent")
        if persisted_parent is not None:
            persisted_parent_matches_target = (
                Path(str(persisted_parent)).resolve()
                == Path(args.parent).resolve())
        if args.replay_max_transitions and (
                args.replay_max_transitions < persisted[0].shape[0]):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 3_130_001)
            indices = torch.randperm(
                persisted[0].shape[0], generator=generator)[
                    :args.replay_max_transitions].to(device)
            persisted = tuple(value[indices] for value in persisted)
        persisted_count = persisted[0].shape[0]
        features = torch.cat([features, persisted[0]])
        base_logits = torch.cat([base_logits, persisted[1]])
        actions = torch.cat([actions, persisted[2]])
        outcomes = torch.cat([outcomes, persisted[3]])
        replay_mask = torch.cat([
            replay_mask,
            torch.ones(persisted[0].shape[0], dtype=torch.bool, device=device)])
    if args.train_existing_skill_slot:
        fresh_mask = ~replay_mask
        if not bool(fresh_mask.any()):
            raise ValueError(
                "continuation requires fresh target transitions in addition "
                "to replay")
        with torch.no_grad():
            base_logits[fresh_mask] = _remove_final_slot_from_logits(
                base, features[fresh_mask], base_logits[fresh_mask])
            if include_parent_action:
                features[fresh_mask, -ACTIONS:] = base_logits[fresh_mask]
            if persisted_parent_matches_target and persisted_count:
                persisted_start = features.shape[0] - persisted_count
                base_logits[persisted_start:] = (
                    _remove_final_slot_from_logits(
                        base, features[persisted_start:],
                        base_logits[persisted_start:]))
                if include_parent_action:
                    features[persisted_start:, -ACTIONS:] = (
                        base_logits[persisted_start:])
    rehearsal_lifetime_count = (
        args.rehearsal_lifetimes * len(rehearsal_spans)
        if args.buffer_in is None else 0)
    protected_mask = _protected_rehearsal_mask(
        replay_mask,
        target_rows=(args.train_lifetimes * args.span),
        protect_rehearsal=args.protect_rehearsal)
    if args.shuffle_outcomes:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        permutation = torch.randperm(
            outcomes.shape[0], generator=generator).to(device)
        outcomes = outcomes[permutation]
    query_sample_weights = _outcome_only_query_weights(
        features, outcomes, protected_mask,
        age_column=(features.shape[1] - 1 if include_event_age else None),
        power=args.query_difficulty_power,
        floor=args.query_difficulty_floor)
    query_sample_weights = query_sample_weights * _base_mistake_weights(
        base_logits, actions, outcomes, protected_mask,
        weight=args.base_mistake_weight)
    curriculum_indices = None
    if args.query_curriculum_warmup_epochs:
        curriculum_indices = _query_curriculum_indices(
            features.shape[0], target_lifetimes=args.train_lifetimes,
            span=args.span, cutoff=args.query_curriculum_cutoff,
            device=device)
    window_indices = None
    if query_window_active:
        window_indices = _query_window_indices(
            features.shape[0], target_lifetimes=args.train_lifetimes,
            span=args.span, start=args.query_window_start,
            end=query_window_end, device=device)
    if args.skill_adapter_width:
        existing_slots = tuple(base_configuration.get(
            "skill_adapter_widths", ()))
        if args.train_existing_skill_slot:
            if not existing_slots:
                raise ValueError(
                    "continuation requires a parent with a skill slot")
            if args.skill_adapter_width != existing_slots[-1]:
                raise ValueError(
                    "continuation width must match the parent's final slot")
            configuration = dict(base_configuration)
        elif existing_slots and not args.append_skill_slot:
            raise ValueError(
                "parent already has skill slots; use --append-skill-slot "
                "for a new successor rung")
        elif args.append_skill_slot:
            configuration = dict(
                base_configuration,
                skill_adapter_widths=existing_slots + (
                    args.skill_adapter_width,))
            existing_scales = tuple(base_configuration.get(
                "skill_adapter_residual_scales", ()))
            configuration["skill_adapter_residual_scales"] = (
                existing_scales
                + (1.0,) * (len(existing_slots) - len(existing_scales))
                + (args.skill_adapter_residual_scale,))
            if args.skill_adapter_reads_parent_action:
                configuration[
                    "skill_adapter_reads_parent_action_from"] = len(
                        existing_slots)
            if args.skill_adapter_reads_parent_entropy:
                configuration[
                    "skill_adapter_reads_parent_entropy_from"] = len(
                        existing_slots)
            if args.skill_adapter_gate_hidden:
                # Give only the new slot a nonlinear gate.  The index keeps
                # every inherited gate shape and checkpoint key unchanged.
                configuration.update(
                    skill_adapter_gate_hidden=args.skill_adapter_gate_hidden,
                    skill_adapter_gate_hidden_from=len(existing_slots))
        else:
            configuration = dict(
                base_configuration,
                skill_adapter_widths=(args.skill_adapter_width,),
                skill_adapter_residual_scales=(
                    args.skill_adapter_residual_scale,),
                skill_adapter_gate_mode=args.skill_adapter_gate_mode,
                skill_adapter_gate_hidden=args.skill_adapter_gate_hidden,
                skill_adapter_legacy_read_from=(
                    None if args.skill_adapter_no_legacy_read else 0),
                skill_adapter_reads_intention_from=(
                    0 if args.skill_adapter_reads_intention else None),
                skill_adapter_reads_workspace_from=(
                    0 if args.skill_adapter_reads_workspace else None),
                skill_adapter_reads_workspace_usage_from=(
                    0 if args.skill_adapter_reads_workspace_usage else None),
                skill_adapter_reads_event_age_from=(
                    0 if args.skill_adapter_reads_event_age else None),
                skill_adapter_reads_parent_action_from=(
                    0 if args.skill_adapter_reads_parent_action else None),
                skill_adapter_reads_parent_entropy_from=(
                    0 if args.skill_adapter_reads_parent_entropy else None),
                event_age=args.skill_adapter_reads_event_age,
                skill_adapter_read_bottleneck=args.skill_adapter_read_bottleneck,
                skill_adapter_critic_width=(
                    args.action_conditioned_critic_width
                    if args.action_conditioned_critic_width else 0))
    else:
        configuration = dict(
            base_configuration,
            action_adapter_width=args.action_adapter_width,
            action_adapter_gated=False)
    student = UnifiedCognitiveController(**configuration).to(device)
    load_result = student.load_state_dict(
        payload["state_dict"], strict=False)
    allowed_missing = set(student.state_dict()) - set(payload["state_dict"])
    if set(load_result.missing_keys) != allowed_missing:
        raise RuntimeError(
            "parent/append checkpoint mismatch: "
            f"missing={load_result.missing_keys}, unexpected="
            f"{load_result.unexpected_keys}")
    if load_result.unexpected_keys:
        raise RuntimeError(
            f"unexpected parent keys: {load_result.unexpected_keys}")
    if args.skill_adapter_width:
        expected_slots = (
            len(base_configuration.get("skill_adapter_widths", ())) + 1
            if args.append_skill_slot
            else len(base_configuration.get("skill_adapter_widths", ()))
            if args.train_existing_skill_slot else 1)
        assert len(student.skill_adapters) == expected_slots
    else:
        assert student.action_adapter is not None
    if args.residual_action_adapter:
        if not base_configuration.get("action_adapter_width", 0):
            raise ValueError(
                "residual-action-adapter requires a parent action adapter")
        with torch.no_grad():
            torch.nn.init.zeros_(student.action_adapter[-1].weight)
            torch.nn.init.zeros_(student.action_adapter[-1].bias)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    if args.skill_adapter_width:
        slot_index = len(student.skill_adapters) - 1
        for module in (
                student.skill_adapters[slot_index],
                student.skill_adapter_gates[slot_index],
                student.skill_adapter_read_projections[slot_index],
                student.skill_adapter_critics[slot_index]):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if args.action_conditioned_critic_width:
            student.skill_adapter_critic_scales[slot_index].requires_grad_(True)
        if args.train_parent_action_adapter:
            if student.action_adapter is None:
                raise ValueError(
                    "parent action adaptation requires an action adapter")
            if student.action_adapter_emits_intention:
                raise ValueError(
                    "parent action adaptation currently expects an action "
                    "logit adapter")
            for parameter in student.action_adapter.parameters():
                parameter.requires_grad_(True)
        if args.train_workspace_address_scales:
            if not student.workspace_slot_addressing:
                raise ValueError(
                    "workspace address scales require addressable workspace")
            student.workspace_read_address_scale.requires_grad_(True)
            student.workspace_write_address_scale.requires_grad_(True)
        if args.train_workspace_write_address_content:
            if not student.workspace_slot_addressing:
                raise ValueError(
                    "address-conditioned writes require addressable workspace")
            student.workspace_write_content_address_scale.requires_grad_(True)
    else:
        for parameter in student.action_adapter.parameters():
            parameter.requires_grad_(True)
    protected_replay_logits: torch.Tensor | None = None
    protected_replay_residual: torch.Tensor | None = None
    protected_replay_opening: torch.Tensor | None = None
    if args.train_existing_skill_slot:
        # Absolute penalties are correct for a zero-output appended slot but
        # would erase an already learned slot.  Continuation instead protects
        # the parent's replay behavior with a fixed local baseline.
        with torch.no_grad():
            (
                protected_replay_logits,
                protected_replay_residual,
                protected_replay_opening,
                _, _, _,
            ) = _skill_slot_logits(student, features)
            protected_replay_logits = protected_replay_logits.detach()
            protected_replay_residual = protected_replay_residual.detach()
            protected_replay_opening = protected_replay_opening.detach()
    def train_epochs(
            epochs: int, sample_indices: torch.Tensor, *,
            outcome_weight: float, residual_penalty: float,
            gate_penalty: float, logit_penalty: float,
            gate_targets: torch.Tensor | None = None) -> None:
        trainable_parameters = [
            parameter for parameter in student.parameters()
            if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=args.learning_rate,
            weight_decay=1e-5)
        for epoch_index in range(epochs):
            permutation = sample_indices[torch.randperm(
                sample_indices.shape[0], device=device)]
            for start in range(0, sample_indices.shape[0], args.batch_size):
                indices = permutation[start:start + args.batch_size]
                if args.skill_adapter_width:
                    (
                        logits_residual, residual, opening, gate_score,
                        critic_logits, critic_residual,
                    ) = _skill_slot_logits(student, features[indices])
                else:
                    residual = student.action_adapter(features[indices])
                    logits_residual = residual
                    opening = None
                    gate_score = None
                    critic_logits = None
                    critic_residual = None
                logits = base_logits[indices] + logits_residual
                if critic_residual is not None:
                    logits = logits + critic_residual
                if args.train_parent_action_adapter:
                    base_features = features[indices, :student.width * 2]
                    inherited = base.action_adapter(base_features)
                    current = student.action_adapter(base_features)
                    logits = logits + current - inherited
                batch_replay_mask = protected_mask[indices]
                loss = (
                    _weighted_binary_margin_loss(
                        logits, actions[indices], outcomes[indices],
                        batch_replay_mask, outcome_weight,
                        args.binary_margin, query_sample_weights[indices])
                    if args.binary_margin_loss else
                    _weighted_binary_complement_loss(
                        logits, actions[indices], outcomes[indices],
                        batch_replay_mask, outcome_weight,
                        args.positive_outcome_weight,
                        query_sample_weights[indices])
                    if args.binary_complement_loss else
                    _weighted_attempted_success_loss(
                        logits, actions[indices], outcomes[indices],
                        batch_replay_mask, outcome_weight,
                        args.positive_outcome_weight,
                        query_sample_weights[indices]))
                if critic_logits is not None and args.action_conditioned_critic_weight:
                    critic_loss = (
                        _weighted_binary_margin_loss(
                            critic_logits, actions[indices], outcomes[indices],
                            batch_replay_mask, outcome_weight,
                            args.binary_margin, query_sample_weights[indices])
                        if args.binary_margin_critic_loss else
                        _weighted_binary_complement_loss(
                            critic_logits, actions[indices], outcomes[indices],
                            batch_replay_mask, outcome_weight,
                            args.positive_outcome_weight,
                            query_sample_weights[indices])
                        if args.binary_complement_critic_loss else
                        _weighted_attempted_success_loss(
                            critic_logits, actions[indices], outcomes[indices],
                            batch_replay_mask, outcome_weight,
                            args.positive_outcome_weight,
                            query_sample_weights[indices]))
                    loss = loss + (
                        args.action_conditioned_critic_weight * critic_loss)
                if (
                        critic_logits is not None
                        and args.action_conditioned_critic_policy_weight
                        and epoch_index >= args.action_conditioned_critic_policy_warmup):
                    fresh_mask = ~batch_replay_mask
                    loss = loss + (
                        args.action_conditioned_critic_policy_weight
                        * _action_conditioned_policy_loss(
                            base_logits[indices] + logits_residual,
                            critic_logits, fresh_mask,
                            temperature=(
                                args.action_conditioned_critic_policy_temperature)))
                if residual_penalty and bool(batch_replay_mask.any()):
                    residual_target = (
                        protected_replay_residual[indices]
                        if protected_replay_residual is not None
                        else torch.zeros_like(residual))
                    loss = loss + residual_penalty * (
                        residual[batch_replay_mask]
                        - residual_target[batch_replay_mask]).square().mean()
                if (gate_penalty and opening is not None
                        and bool(batch_replay_mask.any())):
                    opening_target = (
                        protected_replay_opening[indices]
                        if protected_replay_opening is not None
                        else torch.zeros_like(opening))
                    loss = loss + gate_penalty * (
                        opening[batch_replay_mask]
                        - opening_target[batch_replay_mask]).square().mean()
                if logit_penalty and bool(batch_replay_mask.any()):
                    logit_target = (
                        protected_replay_logits[indices]
                        if protected_replay_logits is not None
                        else torch.zeros_like(logits_residual))
                    loss = loss + logit_penalty * (
                        logits_residual[batch_replay_mask]
                        - logit_target[batch_replay_mask]).square().mean()
                if (args.replay_gate_source_weight and gate_score is not None):
                    source_target = (~batch_replay_mask).to(gate_score.dtype)
                    loss = loss + args.replay_gate_source_weight * _balanced_provenance_loss(
                        gate_score.squeeze(1), source_target)
                if (
                        args.replay_gate_preserve_fresh_weight
                        and gate_score is not None
                        and gate_targets is not None):
                    loss = loss + args.replay_gate_preserve_fresh_weight * F.mse_loss(
                        gate_score.squeeze(1), gate_targets[indices])
                if args.residual_penalty:
                    loss = loss + args.residual_penalty * residual.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    all_indices = torch.arange(features.shape[0], device=device)
    fresh_indices = all_indices[~protected_mask]
    replay_indices = all_indices[protected_mask]
    training_indices = all_indices if window_indices is None else window_indices
    gate_targets = None
    if (
            args.replay_refinement_epochs
            and args.replay_gate_preserve_fresh_weight
            and args.replay_refine_gate_only):
        if not args.skill_adapter_width:
            raise ValueError("gate preservation requires a skill adapter")
        with torch.no_grad():
            gate_targets = _skill_slot_logits(student, features)[3].squeeze(1)
            gate_targets = gate_targets.masked_fill(protected_mask, 0.0)
    if args.replay_refinement_epochs:
        if not bool(fresh_indices.numel()) or not bool(replay_indices.numel()):
            raise ValueError("staged replay needs fresh and persisted samples")
        train_epochs(
            args.epochs, fresh_indices,
            outcome_weight=1.0, residual_penalty=0.0, gate_penalty=0.0,
            logit_penalty=0.0)
        if args.replay_refine_gate_only:
            for parameter in student.parameters():
                parameter.requires_grad_(False)
            for parameter in student.skill_adapter_gates[-1].parameters():
                parameter.requires_grad_(True)
        # A provenance classifier needs both sides of the split.  Training
        # it only on ``replay_indices`` makes every target zero (old), so its
        # supposedly fresh-vs-persisted supervision is silently degenerate.
        # Keep the old-only refinement when no source-aware objective is
        # requested; otherwise use the complete mixed set so ``~replay_mask``
        # contains genuine fresh positives.
        refinement_indices = _replay_refinement_indices(
            all_indices, replay_indices, args.replay_gate_source_weight,
            args.replay_gate_preserve_fresh_weight)
        train_epochs(
            args.replay_refinement_epochs, refinement_indices,
            outcome_weight=0.0,
            residual_penalty=args.replay_residual_penalty,
            gate_penalty=args.replay_gate_penalty,
            logit_penalty=args.replay_logit_penalty,
            gate_targets=gate_targets)
    elif curriculum_indices is not None:
        train_epochs(
            args.query_curriculum_warmup_epochs, curriculum_indices,
            outcome_weight=args.replay_outcome_weight,
            residual_penalty=args.replay_residual_penalty,
            gate_penalty=args.replay_gate_penalty,
            logit_penalty=args.replay_logit_penalty)
        train_epochs(
            args.epochs, training_indices,
            outcome_weight=args.replay_outcome_weight,
            residual_penalty=args.replay_residual_penalty,
            gate_penalty=args.replay_gate_penalty,
            logit_penalty=args.replay_logit_penalty)
    else:
        train_epochs(
            args.epochs, training_indices,
            outcome_weight=args.replay_outcome_weight,
            residual_penalty=args.replay_residual_penalty,
            gate_penalty=args.replay_gate_penalty,
            logit_penalty=args.replay_logit_penalty)
    student.eval()
    audit = evaluate_sequence_memory(
        student, count=args.test_episodes, span=args.span,
        distractors=args.distractors, seed=args.seed + 90_000,
        operation=(args.target_operation
                   if args.test_operation is None else args.test_operation),
        device=device)
    report = {
        "schema": "latent-replay-adapter-v2",
        "claim_boundary": (
            "The learner sees only RGB streams, opaque attempted actions, "
            "and scalar attempted-action outcomes. No correct unattempted "
            "action or semantic task label is stored in the buffer."),
        "parent": str(args.parent),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "unique_train_lifetimes": args.train_lifetimes,
        "unique_rehearsal_lifetimes": rehearsal_lifetime_count,
        "total_unique_lifetimes": args.train_lifetimes + rehearsal_lifetime_count,
        "query_transition_count": int(features.shape[0]),
        "rehearse_spans": rehearsal_spans,
        "rehearsal_lifetimes_per_stream": args.rehearsal_lifetimes,
        "buffer_in": str(args.buffer_in) if args.buffer_in else None,
        "replay_buffer_in": (
            str(args.replay_buffer_in) if args.replay_buffer_in else None),
        "persisted_parent_matches_target": (
            persisted_parent_matches_target),
        "buffer_out": str(args.buffer_out) if args.buffer_out else None,
        "replay_max_transitions": args.replay_max_transitions,
        "replay_outcome_weight": args.replay_outcome_weight,
        "replay_residual_penalty": args.replay_residual_penalty,
        "replay_gate_penalty": args.replay_gate_penalty,
        "replay_logit_penalty": args.replay_logit_penalty,
        "protect_rehearsal": args.protect_rehearsal,
        "replay_gate_source_weight": args.replay_gate_source_weight,
        "replay_refinement_epochs": args.replay_refinement_epochs,
        "replay_refine_gate_only": args.replay_refine_gate_only,
        "residual_action_adapter": args.residual_action_adapter,
        "residual_penalty": args.residual_penalty,
        "positive_outcome_weight": args.positive_outcome_weight,
        "query_difficulty_power": args.query_difficulty_power,
        "query_difficulty_floor": args.query_difficulty_floor,
        "query_curriculum_warmup_epochs": (
            args.query_curriculum_warmup_epochs),
        "query_curriculum_cutoff": args.query_curriculum_cutoff,
        "query_window_start": args.query_window_start,
        "query_window_end": (
            query_window_end if query_window_active else 0),
        "base_mistake_weight": args.base_mistake_weight,
        "target_operation": args.target_operation,
        "test_operation": (
            args.target_operation
            if args.test_operation is None else args.test_operation),
        "buffer_exploration": args.buffer_exploration,
        "binary_complement_loss": args.binary_complement_loss,
        "binary_complement_critic_loss": args.binary_complement_critic_loss,
        "binary_margin_loss": args.binary_margin_loss,
        "binary_margin_critic_loss": args.binary_margin_critic_loss,
        "binary_margin": args.binary_margin,
        "action_conditioned_critic_width": (
            args.action_conditioned_critic_width),
        "action_conditioned_critic_weight": (
            args.action_conditioned_critic_weight),
        "action_conditioned_critic_policy_weight": (
            args.action_conditioned_critic_policy_weight),
        "action_conditioned_critic_policy_warmup": (
            args.action_conditioned_critic_policy_warmup),
        "action_conditioned_critic_policy_temperature": (
            args.action_conditioned_critic_policy_temperature),
        "skill_adapter_width": args.skill_adapter_width,
        "train_existing_skill_slot": args.train_existing_skill_slot,
        "skill_adapter_residual_scale": (
            args.skill_adapter_residual_scale),
        "skill_adapter_gate_mode": args.skill_adapter_gate_mode,
        "skill_adapter_reads_intention": args.skill_adapter_reads_intention,
        "skill_adapter_reads_workspace": args.skill_adapter_reads_workspace,
        "skill_adapter_reads_workspace_usage": (
            args.skill_adapter_reads_workspace_usage),
        "skill_adapter_reads_event_age": args.skill_adapter_reads_event_age,
        "skill_adapter_reads_parent_action": (
            args.skill_adapter_reads_parent_action),
        "skill_adapter_reads_parent_entropy": (
            args.skill_adapter_reads_parent_entropy),
        "skill_adapter_read_bottleneck": args.skill_adapter_read_bottleneck,
        "skill_adapter_no_legacy_read": args.skill_adapter_no_legacy_read,
        "train_parent_action_adapter": args.train_parent_action_adapter,
        "train_workspace_address_scales": args.train_workspace_address_scales,
        "train_workspace_write_address_content": (
            args.train_workspace_write_address_content),
        "loaded_buffer_metadata": loaded_buffer_metadata,
        "collected_stream_specs": stream_specs,
        "buffer_outcomes_shuffled": args.shuffle_outcomes,
        "audit": audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "model_configuration": configuration,
        "state_dict": student.state_dict(),
        "source_report": str(args.report),
        "admission_status": "diagnostic_reward_buffer_candidate",
    }, args.checkpoint_out)
    print(json.dumps({
        "accuracy": audit["accuracy"],
        "reverse_flip": audit[
            "reverse_operation_prediction_flip_rate_nonpalindrome"],
        "blank": audit["blank_sequence_accuracy"],
        "reset": audit["all_memory_reset_accuracy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
