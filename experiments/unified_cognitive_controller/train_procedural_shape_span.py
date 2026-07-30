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


def _balanced_logical_content(
        count: int, span: int, vocabulary: int,
        generator: torch.Generator,
        permutations: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross every identity sequence with every answer pattern evenly."""
    answer_patterns = 2 ** span
    sequence_patterns = vocabulary ** span
    logical_patterns = sequence_patterns * answer_patterns
    design_ids = torch.arange(count) % (logical_patterns * permutations)
    ids = design_ids % logical_patterns
    permutation_ids = design_ids // logical_patterns
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


def generate_procedural_shape_batch(
        count: int, *, span: int, vocabulary: int, seed: int,
        nuisance: ShapeNuisance, heldout: bool = False,
        objective: str = "recognition",
        blank_presentation: bool = False,
        reverse_presentation: bool = False,
        flip_candidates: bool = False,
        device: torch.device | str = "cpu") -> ProceduralShapeBatch:
    """Generate deterministic balanced episodes and private verifier answers."""
    nuisance.validate()
    if objective not in ("visible_identity", "recognition"):
        raise ValueError("objective must be visible_identity or recognition")
    permutations = tuple(itertools.permutations(range(span)))
    design_patterns = (vocabulary * 2) ** span * len(permutations)
    if count < design_patterns or count % design_patterns:
        raise ValueError(
            f"count must be a positive multiple of {design_patterns}")
    if span < 1:
        raise ValueError("span must be positive")
    if not 2 <= vocabulary <= 4:
        raise ValueError("vocabulary must be within [2, 4]")
    generator = torch.Generator().manual_seed(seed)
    sequence, match, permutation_ids = _balanced_logical_content(
        count, span, vocabulary, generator, len(permutations))
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
    answers = (
        candidates.clone()
        if objective == "visible_identity"
        else (candidates == sequence).long())

    presentation = _render_shapes(
        sequence, seed=seed ^ 0x13579BDF, nuisance=nuisance,
        heldout=heldout)
    queries = _render_shapes(
        candidates, seed=seed ^ 0x2468ACE0, nuisance=nuisance,
        heldout=heldout, ordinal_cues=True)
    # Query order is independently permuted per lifetime.  Therefore neither
    # elapsed query time nor previous feedback reveals the requested ordinal;
    # the visual cue must be read.
    query_ordinals = torch.empty(count, span, dtype=torch.long)
    for row in range(count):
        # Cross query permutations with all content/answer patterns. Query time
        # is consequently independent of identity, ordinal, and correct action.
        query_ordinals[row] = torch.tensor(
            permutations[int(permutation_ids[row])])
    gather_frames = query_ordinals[:, :, None, None, None].expand_as(queries)
    queries = torch.gather(queries, 1, gather_frames)
    candidates = torch.gather(candidates, 1, query_ordinals)
    answers = torch.gather(answers, 1, query_ordinals)
    if blank_presentation:
        backgrounds = presentation[:, :, :, :1, :1]
        presentation = backgrounds.expand_as(presentation).clone()
    lifetime_ids = torch.arange(
        seed, seed + count, dtype=torch.long, device=device)
    return ProceduralShapeBatch(
        presentation.to(device), queries.to(device), answers.to(device),
        sequence.to(device), candidates.to(device), query_ordinals.to(device),
        lifetime_ids, objective)


def _reset_active_keep_workspace(state: ControllerState) -> ControllerState:
    return ControllerState(
        torch.zeros_like(state.hidden), state.workspace,
        torch.zeros_like(state.latest_event))


def rollout_procedural_shape_span(
        model: UnifiedCognitiveController, batch: ProceduralShapeBatch, *,
        sample_actions: bool, exploration: float = 0.10,
        disable_workspace: bool = False,
        reset_active_before_query: bool = False,
        reset_all_before_query: bool = False,
        blank_ordinal_cues: bool = False,
        shuffle_outcomes: bool = False) -> dict[str, torch.Tensor]:
    device = batch.presentation_frames.device
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
    for index in range(batch.span):
        frame = batch.query_frames[:, index]
        if blank_ordinal_cues:
            frame = frame.clone()
            frame[:, :, 28:30, :] = frame[:, :, :1, :1]
        has_feedback = torch.full_like(previous_reward, float(index > 0))
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
        losses.append(attempted_success_loss(
            output.logits, action, delivered))
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
        objective: str = "recognition") -> dict[str, object]:
    model.eval()
    kwargs = dict(
        count=count, span=span, vocabulary=vocabulary, seed=seed,
        nuisance=nuisance, heldout=True, objective=objective, device=device)
    normal_batch = generate_procedural_shape_batch(**kwargs)
    blank_batch = generate_procedural_shape_batch(
        **kwargs, blank_presentation=True)
    reverse_batch = generate_procedural_shape_batch(
        **kwargs, reverse_presentation=True)
    flipped_batch = generate_procedural_shape_batch(
        **kwargs, flip_candidates=True)

    def run(batch: ProceduralShapeBatch, **controls: bool):
        return rollout_procedural_shape_span(
            model, batch, sample_actions=False, **controls)

    normal = run(normal_batch)
    blank = run(blank_batch)
    reverse = run(reverse_batch)
    flipped = run(flipped_batch)
    cue_blank = run(normal_batch, blank_ordinal_cues=True)
    workspace_off = run(normal_batch, disable_workspace=True)
    active_reset = run(normal_batch, reset_active_before_query=True)
    all_reset = run(normal_batch, reset_all_before_query=True)

    def accuracy(result: dict[str, torch.Tensor]) -> float:
        return float(result["rewards"].mean())

    reverse_changed = (
        normal_batch.correct_actions != reverse_batch.correct_actions)
    candidate_changed = (
        normal_batch.correct_actions != flipped_batch.correct_actions)
    return {
        "accuracy": accuracy(normal),
        "accuracy_by_ordinal": [
            float(value) for value in normal["rewards"].mean(0)],
        "blank_presentation_accuracy": accuracy(blank),
        "blank_ordinal_cue_accuracy": accuracy(cue_blank),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=27001)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-episodes", type=int, default=2048)
    parser.add_argument("--span", type=int, default=2)
    parser.add_argument("--vocabulary", type=int, default=2)
    parser.add_argument(
        "--objective", choices=("visible_identity", "recognition"),
        default="recognition")
    parser.add_argument("--randomness", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=4)
    parser.add_argument("--intention-width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--curve-test-episodes", type=int, default=512)
    parser.add_argument("--mastery-threshold", type=float, default=0.90)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    nuisance = nuisance_from_level(args.randomness)
    configuration: dict[str, object] = {
        "width": args.width, "workspace_slots": args.workspace_slots,
        "intention_width": args.intention_width}
    payload = None
    if args.checkpoint_in is not None:
        payload = torch.load(
            args.checkpoint_in, map_location=device, weights_only=False)
        configuration = dict(payload["model_configuration"])
    model = UnifiedCognitiveController(**configuration).to(device)
    if payload is not None:
        model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history: list[dict[str, object]] = []
    started = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        batch = generate_procedural_shape_batch(
            args.batch_size, span=args.span, vocabulary=args.vocabulary,
            seed=args.seed + step * args.batch_size, nuisance=nuisance,
            objective=args.objective, device=device)
        result = rollout_procedural_shape_span(
            model, batch, sample_actions=True,
            exploration=args.exploration,
            shuffle_outcomes=args.shuffle_outcomes)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row: dict[str, object] = {
                "update": step,
                "unique_logical_lifetimes": step * args.batch_size,
                "unique_verifier_bits": (
                    step * args.batch_size * args.span),
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach())}
            if (
                    step % args.eval_every == 0
                    or step == args.steps):
                curve = evaluate_procedural_shape_span(
                    model, count=args.curve_test_episodes, span=args.span,
                    vocabulary=args.vocabulary,
                    seed=args.seed + 20_000_000 + step,
                    nuisance=nuisance, device=device,
                    objective=args.objective)
                row["heldout_accuracy"] = curve["accuracy"]
                row["blank_presentation_accuracy"] = (
                    curve["blank_presentation_accuracy"])
            history.append(row)
            print(json.dumps(row), flush=True)

    audit = evaluate_procedural_shape_span(
        model, count=args.test_episodes, span=args.span,
        vocabulary=args.vocabulary, seed=args.seed + 10_000_000,
        nuisance=nuisance, device=device, objective=args.objective)
    prefixes = [row for row in history if "heldout_accuracy" in row]
    stable_bits = None
    for index, row in enumerate(prefixes):
        if all(
                float(later["heldout_accuracy"]) >= args.mastery_threshold
                for later in prefixes[index:]):
            stable_bits = int(row["unique_verifier_bits"])
            break
    report = {
        "schema": "procedural-shape-span-experiment-v1",
        "learner_visible_information": (
            "RGB streams, own opaque actions, scalar attempted-action outcome"),
        "span": args.span,
        "vocabulary": args.vocabulary,
        "objective": args.objective,
        "randomness": args.randomness,
        "nuisance": asdict(nuisance),
        "optimizer_updates": args.steps,
        "unique_logical_lifetimes": args.steps * args.batch_size,
        "unique_verifier_bits": args.steps * args.batch_size * args.span,
        "replayed_examples": 0,
        "outcomes_shuffled": args.shuffle_outcomes,
        "mastery_threshold": args.mastery_threshold,
        "stable_bits_to_threshold": stable_bits,
        "wall_seconds": perf_counter() - started,
        "model_configuration": configuration,
        "history": history,
        "audit": audit,
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
            "procedural_shape_span_report": report,
        }, args.checkpoint_out)


if __name__ == "__main__":
    main()
