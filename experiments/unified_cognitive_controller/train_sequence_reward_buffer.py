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
from pathlib import Path

import torch
import torch.nn.functional as F

from .environment import ACTIONS, NULL_ACTION
from .model import UnifiedCognitiveController
from .train import seed_everything
from .train_sequence_working_memory import (
    evaluate_sequence_memory, generate_sequence_memory_batch)


BUFFER_SCHEMA = "latent-action-outcome-buffer-v1"


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
    if len(model.skill_adapters) != 1:
        raise ValueError("this diagnostic expects exactly one new skill slot")
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
    expected_width = base_width + workspace_width + usage_width + intention_width
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
    if reads:
        projected = model.skill_adapter_read_projections[0](
            torch.cat(reads, dim=-1))
        own_features = torch.cat([slot_features, projected], dim=-1)
    else:
        own_features = slot_features
    score = model.skill_adapter_gates[0](own_features)
    opening = (
        F.leaky_relu(score, model.skill_adapter_gate_leak)
        if model.skill_adapter_gate_mode == "relu"
        else torch.sigmoid(score))
    hidden_read = model.skill_adapters[0][1](
        model.skill_adapters[0][0](own_features))
    residual = model.skill_adapters[0][2](hidden_read) * opening
    critic_logits = None
    critic_residual = None
    critic = model.skill_adapter_critics[0]
    if not isinstance(critic, torch.nn.Identity):
        critic_logits = critic(own_features)
        critic_residual = (
            critic_logits - critic_logits.mean(dim=-1, keepdim=True))
        critic_residual = (
            critic_residual * model.skill_adapter_critic_scales[0]
            * opening)
    return (
        model.actuator(residual), residual, opening, score,
        critic_logits, critic_residual)


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


def _weighted_attempted_success_loss(
        logits: torch.Tensor, attempted: torch.Tensor,
        outcomes: torch.Tensor, replay_mask: torch.Tensor,
        replay_outcome_weight: float,
        positive_outcome_weight: float = 1.0) -> torch.Tensor:
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
    return (per_example * weights).sum() / weights.sum().clamp_min(1.0)


def _collect_buffer(
        model: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, device: torch.device,
        position_augmentation: bool, include_intention: bool = False,
        include_workspace: bool = False,
        include_workspace_usage: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect latent transitions using uniformly random opaque actions."""
    batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation="mixed", position_augmentation=position_augmentation,
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
            output, state = model.step(
                frame, state, previous_action,
                previous_reward * has_feedback, has_feedback)
            action = torch.randint(ACTIONS, (count,), device=device)
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
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--rehearse-spans", default="")
    parser.add_argument("--rehearsal-lifetimes", type=int, default=512)
    parser.add_argument("--span", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--positive-outcome-weight", type=float, default=1.0,
        help=(
            "multiply observed successful-attempt loss; this is a reward "
            "balance control, not a correct-action label"))
    parser.add_argument("--action-adapter-width", type=int, default=64)
    parser.add_argument(
        "--skill-adapter-width", type=int, default=0,
        help=(
            "train one new successor skill slot while preserving the "
            "parent action adapter"))
    parser.add_argument(
        "--action-conditioned-critic-width", type=int, default=0,
        help=(
            "add a zero-impact per-action success predictor trained from "
            "attempted-action rewards"))
    parser.add_argument(
        "--action-conditioned-critic-weight", type=float, default=1.0,
        help="weight the critic's attempted-outcome loss")
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
            or args.test_episodes < 2):
        raise ValueError("invalid replay-buffer dimensions")
    if args.replay_max_transitions < 0:
        raise ValueError("replay-max-transitions must not be negative")
    if args.replay_refinement_epochs < 0:
        raise ValueError("replay-refinement-epochs must not be negative")
    if args.residual_penalty < 0:
        raise ValueError("residual-penalty must not be negative")
    if args.positive_outcome_weight < 0:
        raise ValueError("positive-outcome-weight must not be negative")
    if args.action_conditioned_critic_weight < 0:
        raise ValueError("action-conditioned-critic-weight must not be negative")
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
    device = torch.device(args.device)
    payload = torch.load(args.parent, map_location=device, weights_only=False)
    base_configuration = dict(payload["model_configuration"])
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
    if args.buffer_in is not None:
        (features, base_logits, actions, outcomes), metadata = _load_buffer(
            args.buffer_in, device=device)
        loaded_buffer_metadata.append(metadata)
        replay_mask = torch.ones(
            features.shape[0], dtype=torch.bool, device=device)
    else:
        target_buffer = _collect_buffer(
            base, count=args.train_lifetimes, span=args.span,
            distractors=args.distractors, seed=args.seed, device=device,
            position_augmentation=args.position_augmentation,
            include_intention=args.skill_adapter_reads_intention,
            include_workspace=args.skill_adapter_reads_workspace,
            include_workspace_usage=args.skill_adapter_reads_workspace_usage)
        buffer_parts = [target_buffer]
        stream_specs.append({
            "kind": "target", "span": args.span,
            "lifetimes": args.train_lifetimes})
        for index, rehearsal_span in enumerate(rehearsal_spans):
            buffer_parts.append(_collect_buffer(
                base, count=args.rehearsal_lifetimes, span=rehearsal_span,
                distractors=args.distractors,
                seed=args.seed + (index + 1) * 1_000_003,
                device=device,
                position_augmentation=args.position_augmentation,
                include_intention=args.skill_adapter_reads_intention,
                include_workspace=args.skill_adapter_reads_workspace,
                include_workspace_usage=args.skill_adapter_reads_workspace_usage))
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
        loaded_buffer_metadata.append(metadata)
        if args.replay_max_transitions and (
                args.replay_max_transitions < persisted[0].shape[0]):
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed + 3_130_001)
            indices = torch.randperm(
                persisted[0].shape[0], generator=generator)[
                    :args.replay_max_transitions].to(device)
            persisted = tuple(value[indices] for value in persisted)
        features = torch.cat([features, persisted[0]])
        base_logits = torch.cat([base_logits, persisted[1]])
        actions = torch.cat([actions, persisted[2]])
        outcomes = torch.cat([outcomes, persisted[3]])
        replay_mask = torch.cat([
            replay_mask,
            torch.ones(persisted[0].shape[0], dtype=torch.bool, device=device)])
    rehearsal_lifetime_count = (
        args.rehearsal_lifetimes * len(rehearsal_spans)
        if args.buffer_in is None else 0)
    if args.shuffle_outcomes:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        permutation = torch.randperm(
            outcomes.shape[0], generator=generator).to(device)
        outcomes = outcomes[permutation]
    if args.skill_adapter_width:
        if base_configuration.get("skill_adapter_widths", ()):
            raise ValueError(
                "this diagnostic expects a parent without skill slots")
        configuration = dict(
            base_configuration,
            skill_adapter_widths=(args.skill_adapter_width,),
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
    student.load_state_dict(payload["state_dict"], strict=False)
    if args.skill_adapter_width:
        assert len(student.skill_adapters) == 1
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
        for module in (
                student.skill_adapters[0], student.skill_adapter_gates[0],
                student.skill_adapter_read_projections[0],
                student.skill_adapter_critics[0]):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if args.action_conditioned_critic_width:
            student.skill_adapter_critic_scales[0].requires_grad_(True)
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
        for _ in range(epochs):
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
                batch_replay_mask = replay_mask[indices]
                loss = _weighted_attempted_success_loss(
                    logits, actions[indices], outcomes[indices],
                    batch_replay_mask, outcome_weight,
                    args.positive_outcome_weight)
                if critic_logits is not None and args.action_conditioned_critic_weight:
                    critic_loss = _weighted_attempted_success_loss(
                        critic_logits, actions[indices], outcomes[indices],
                        batch_replay_mask, outcome_weight,
                        args.positive_outcome_weight)
                    loss = loss + (
                        args.action_conditioned_critic_weight * critic_loss)
                if residual_penalty and bool(batch_replay_mask.any()):
                    loss = loss + residual_penalty * residual[
                        batch_replay_mask].square().mean()
                if (gate_penalty and opening is not None
                        and bool(batch_replay_mask.any())):
                    loss = loss + gate_penalty * opening[
                        batch_replay_mask].square().mean()
                if logit_penalty and bool(batch_replay_mask.any()):
                    loss = loss + logit_penalty * logits_residual[
                        batch_replay_mask].square().mean()
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
    fresh_indices = all_indices[~replay_mask]
    replay_indices = all_indices[replay_mask]
    gate_targets = None
    if (
            args.replay_refinement_epochs
            and args.replay_gate_preserve_fresh_weight
            and args.replay_refine_gate_only):
        if not args.skill_adapter_width:
            raise ValueError("gate preservation requires a skill adapter")
        with torch.no_grad():
            gate_targets = _skill_slot_logits(student, features)[3].squeeze(1)
            gate_targets = gate_targets.masked_fill(replay_mask, 0.0)
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
            for parameter in student.skill_adapter_gates[0].parameters():
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
    else:
        train_epochs(
            args.epochs, all_indices,
            outcome_weight=args.replay_outcome_weight,
            residual_penalty=args.replay_residual_penalty,
            gate_penalty=args.replay_gate_penalty,
            logit_penalty=args.replay_logit_penalty)
    student.eval()
    audit = evaluate_sequence_memory(
        student, count=args.test_episodes, span=args.span,
        distractors=args.distractors, seed=args.seed + 90_000,
        operation="mixed", device=device)
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
        "buffer_out": str(args.buffer_out) if args.buffer_out else None,
        "replay_max_transitions": args.replay_max_transitions,
        "replay_outcome_weight": args.replay_outcome_weight,
        "replay_residual_penalty": args.replay_residual_penalty,
        "replay_gate_penalty": args.replay_gate_penalty,
        "replay_logit_penalty": args.replay_logit_penalty,
        "replay_gate_source_weight": args.replay_gate_source_weight,
        "replay_refinement_epochs": args.replay_refinement_epochs,
        "replay_refine_gate_only": args.replay_refine_gate_only,
        "residual_action_adapter": args.residual_action_adapter,
        "residual_penalty": args.residual_penalty,
        "positive_outcome_weight": args.positive_outcome_weight,
        "action_conditioned_critic_width": (
            args.action_conditioned_critic_width),
        "action_conditioned_critic_weight": (
            args.action_conditioned_critic_weight),
        "skill_adapter_width": args.skill_adapter_width,
        "skill_adapter_gate_mode": args.skill_adapter_gate_mode,
        "skill_adapter_reads_intention": args.skill_adapter_reads_intention,
        "skill_adapter_reads_workspace": args.skill_adapter_reads_workspace,
        "skill_adapter_reads_workspace_usage": (
            args.skill_adapter_reads_workspace_usage),
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
