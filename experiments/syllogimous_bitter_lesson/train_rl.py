from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.utils.data import DataLoader

from experiments.syllogimous_latent_agent.data import EpisodeDataset, collate_episodes

from .model import BitterLessonAgent, model_config, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reaction_difficulties(choice_counts: tuple[int, ...], distractors: int,
                          delay_frames: int, audio_distractors: int,
                          target_like_distractors: int = 0,
                          temporal_distractors: int = 0):
    from .choice_reaction import ReactionDifficulty
    return tuple(ReactionDifficulty(count, distractors, delay_frames, audio_distractors,
                                    target_like_distractors, temporal_distractors)
                 for count in choice_counts)


def reasoning_dataset(family: str, samples: int, lengths: tuple[int, ...],
                      modulus: int, cyclic_fraction: float, *,
                      start_seed: int, heldout: bool = False,
                      cyclic_lengths: tuple[int, ...] | None = None):
    """Build one reasoning stream, retaining parity replay for richer logic."""
    from .parity_transfer import ParityDataset
    from .cyclic_transfer import CyclicDataset
    if family == "parity":
        return ParityDataset(samples, lengths, start_seed=start_seed, heldout=heldout)
    if family == "cyclic":
        return CyclicDataset(samples, cyclic_lengths or lengths, modulus,
                             start_seed=start_seed, heldout=heldout)
    from .choice_reaction import CognitiveMixtureDataset
    cyclic_samples = round(samples * cyclic_fraction)
    parity_samples = samples - cyclic_samples
    return CognitiveMixtureDataset(
        ParityDataset(parity_samples, lengths,
                      start_seed=start_seed, heldout=heldout),
        CyclicDataset(cyclic_samples, cyclic_lengths or lengths, modulus,
                      start_seed=start_seed + parity_samples, heldout=heldout),
    )


def load_initial_state(model: BitterLessonAgent, payload: dict,
                       allow_action_expansion: bool) -> torch.nn.modules.module._IncompatibleKeys:
    """Load a checkpoint, optionally preserving old motor rows in a wider head."""
    state = dict(payload["model"])
    expanded: dict[str, torch.Tensor] = {}
    for name in ("observation_head.weight", "observation_head.bias",
                 "answer_head.weight", "answer_head.bias"):
        if name in state and state[name].shape != model.state_dict()[name].shape:
            if not allow_action_expansion or state[name].shape[0] > model.state_dict()[name].shape[0]:
                raise ValueError(f"incompatible action head {name}: {state[name].shape}")
            expanded[name] = state.pop(name)
    incompatible = model.load_state_dict(state, strict=False)
    with torch.no_grad():
        current = model.state_dict()
        for name, old in expanded.items():
            current[name][:old.shape[0]].copy_(old)
    return incompatible


def curriculum_sampling_choices(lengths: tuple[int, ...]) -> tuple[int, ...]:
    """Replay mastered lengths while concentrating experience on the newest."""
    if not lengths:
        raise ValueError("curriculum requires at least one length")
    if len(lengths) == 1:
        return lengths
    if len(lengths) == 2:
        return (lengths[0],) * 3 + (lengths[1],) * 7
    if len(lengths) == 3:
        return (lengths[0],) * 3 + (lengths[1],) * 5 + (lengths[2],) * 12
    if len(lengths) == 4:
        return ((lengths[0],) * 2 + (lengths[1],) * 3 +
                (lengths[2],) * 5 + (lengths[3],) * 10)
    if len(lengths) == 5:
        return ((lengths[0],) * 2 + (lengths[1],) * 3 +
                (lengths[2],) * 4 + (lengths[3],) * 5 +
                (lengths[4],) * 6)
    if len(lengths) == 6:
        return ((lengths[0],) * 1 + (lengths[1],) * 2 +
                (lengths[2],) * 3 + (lengths[3],) * 4 +
                (lengths[4],) * 4 + (lengths[5],) * 6)
    if len(lengths) == 7:
        return ((lengths[0],) * 1 + (lengths[1],) * 2 +
                (lengths[2],) * 3 + (lengths[3],) * 3 +
                (lengths[4],) * 3 + (lengths[5],) * 4 +
                (lengths[6],) * 6)
    # Reserve 60% for the frontier and share replay evenly among older skills.
    replay_each = max(1, 40 // (len(lengths) - 1))
    return tuple(item for length in lengths[:-1] for item in (length,) * replay_each) + \
        (lengths[-1],) * 60


def sample_halting(halt_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the first halt decision; the final thought step must halt."""
    batch, steps = halt_logits.shape
    alive = torch.ones(batch, dtype=torch.bool, device=halt_logits.device)
    chosen = torch.full((batch,), steps - 1, dtype=torch.long, device=halt_logits.device)
    log_probability = halt_logits.new_zeros(batch)
    for index in range(steps - 1):
        distribution = Bernoulli(logits=halt_logits[:, index])
        decision = distribution.sample().bool() & alive
        sampled = decision.to(halt_logits.dtype)
        log_probability += torch.where(alive, distribution.log_prob(sampled), 0.0)
        chosen = torch.where(decision, index, chosen)
        # Do not mutate a mask saved by autograd for the preceding where().
        alive = alive & ~decision
    return chosen, log_probability


def policy_loss(output, actions: torch.Tensor, mask: torch.Tensor,
                speed_bonus: float, entropy_weight: float,
                value_weight: float,
                latency_multiplier: float = 1.0,
                fixed_thoughts: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    """Outcome-reward policy gradient; labels are used only by the verifier."""
    batch = actions.shape[0]
    rows = torch.arange(batch, device=actions.device)
    final_indices = mask.sum(1) - 1
    verifier_answer = actions[rows, final_indices]
    if fixed_thoughts:
        thought_steps = torch.full((batch,), output.answer_logits.shape[1] - 1,
                                   dtype=torch.long, device=actions.device)
        halt_log_probability = output.halt_logits.new_zeros(batch)
    else:
        thought_steps, halt_log_probability = sample_halting(output.halt_logits)
    selected_logits = output.answer_logits[rows, thought_steps]
    action_distribution = Categorical(logits=selected_logits)
    sampled_action = action_distribution.sample()
    correct = sampled_action == verifier_answer
    speed = 1.0 - thought_steps.float() / max(1, output.answer_logits.shape[1] - 1)
    active_speed_bonus = speed_bonus * latency_multiplier
    reward = torch.where(correct, 1.0 + active_speed_bonus * speed,
                         torch.full_like(speed, -1.0))
    selected_values = output.values[rows, thought_steps]
    advantage = reward - selected_values.detach()
    answer_log_probability = action_distribution.log_prob(sampled_action)
    actor_loss = -(advantage * (answer_log_probability + halt_log_probability)).mean()
    critic_loss = nn.functional.mse_loss(selected_values, reward)
    entropy = action_distribution.entropy().mean()

    # Dense interface reward teaches only which public UI action advances a
    # non-final card. It carries no proposition semantics or answer information.
    premise_mask = mask.clone()
    premise_mask[rows, final_indices] = False
    premise_logits = output.observation_logits[premise_mask]
    premise_actions = actions[premise_mask]
    interface_loss = nn.functional.cross_entropy(premise_logits, premise_actions)
    loss = actor_loss + value_weight * critic_loss + 0.05 * interface_loss - entropy_weight * entropy
    metrics = {
        "reward": float(reward.mean().detach()),
        "accuracy": float(correct.float().mean()),
        "thought_steps": float((thought_steps.float() + 1).mean()),
        "latency_multiplier": latency_multiplier,
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "entropy": float(entropy.detach()),
    }
    return loss, metrics


def q_learning_loss(output, actions: torch.Tensor, mask: torch.Tensor,
                    epsilon: float = 0.2) -> tuple[torch.Tensor, dict[str, float]]:
    """One-step outcome Q-learning with no semantic or counterfactual labels."""
    batch = actions.shape[0]
    rows = torch.arange(batch, device=actions.device)
    final_indices = mask.sum(1) - 1
    verifier_answer = actions[rows, final_indices]
    q_values = output.answer_logits[:, -1]
    greedy = q_values.argmax(-1)
    random_actions = torch.randint(q_values.shape[-1], (batch,), device=q_values.device)
    explore = torch.rand(batch, device=q_values.device) < epsilon
    selected_actions = torch.where(explore, random_actions, greedy)
    correct = selected_actions == verifier_answer
    rewards = torch.where(correct, torch.ones(batch, device=q_values.device),
                          -torch.ones(batch, device=q_values.device))
    selected_q = q_values[rows, selected_actions]
    q_loss = nn.functional.smooth_l1_loss(selected_q, rewards)

    premise_mask = mask.clone()
    premise_mask[rows, final_indices] = False
    interface_loss = nn.functional.cross_entropy(output.observation_logits[premise_mask],
                                                   actions[premise_mask])
    loss = q_loss + 0.05 * interface_loss
    return loss, {
        "reward": float(rewards.mean()),
        "accuracy": float(correct.float().mean()),
        "thought_steps": float(output.answer_logits.shape[1]),
        "latency_multiplier": 0.0,
        "q_loss": float(q_loss.detach()),
        "exploration": epsilon,
    }


def verifier_loss(output, actions: torch.Tensor,
                  mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Full-information outcome control, with no proposition supervision.

    This is a diagnostic upper bound for credit assignment: it supplies the
    correct public action already held by the reward verifier, but no entities,
    relations, proof steps, or intermediate state.
    """
    batch = actions.shape[0]
    rows = torch.arange(batch, device=actions.device)
    final_indices = mask.sum(1) - 1
    targets = actions[rows, final_indices]
    logits = output.answer_logits[:, -1]
    answer_loss = nn.functional.cross_entropy(logits, targets)
    predictions = logits.argmax(-1)
    premise_mask = mask.clone()
    premise_mask[rows, final_indices] = False
    interface_loss = nn.functional.cross_entropy(output.observation_logits[premise_mask],
                                                   actions[premise_mask])
    return answer_loss + 0.05 * interface_loss, {
        "reward": float(torch.where(predictions == targets, 1.0, -1.0).mean()),
        "accuracy": float((predictions == targets).float().mean()),
        "thought_steps": float(output.answer_logits.shape[1]),
        "latency_multiplier": 0.0,
        "verifier_loss": float(answer_loss.detach()),
    }


def randomized_depth_verifier_loss(output, actions: torch.Tensor,
                                   mask: torch.Tensor,
                                   consistency_weight: float
                                   ) -> tuple[torch.Tensor, dict[str, float]]:
    """Outcome supervision at random compute depths plus cross-depth agreement."""
    batch, depths = output.answer_logits.shape[:2]
    rows = torch.arange(batch, device=actions.device)
    targets = actions[rows, mask.sum(1) - 1]
    chosen_depths = torch.randint(depths, (batch,), device=actions.device)
    chosen_logits = output.answer_logits[rows, chosen_depths]
    answer_loss = nn.functional.cross_entropy(chosen_logits, targets)

    # The deepest prediction is a stop-gradient consistency target. This asks
    # shallow and deep computations to agree without supplying proof states or
    # any task-specific intermediate labels.
    teacher = torch.softmax(output.answer_logits[:, -1].detach(), dim=-1)
    consistency = nn.functional.kl_div(
        torch.log_softmax(chosen_logits, dim=-1), teacher, reduction="batchmean")
    premise_mask = mask.clone()
    premise_mask[rows, mask.sum(1) - 1] = False
    interface_loss = nn.functional.cross_entropy(output.observation_logits[premise_mask],
                                                   actions[premise_mask])
    predictions = chosen_logits.argmax(-1)
    return answer_loss + consistency_weight * consistency + 0.05 * interface_loss, {
        "reward": float(torch.where(predictions == targets, 1.0, -1.0).mean()),
        "accuracy": float((predictions == targets).float().mean()),
        "thought_steps": float((chosen_depths.float() + 1).mean()),
        "latency_multiplier": 0.0,
        "verifier_loss": float(answer_loss.detach()),
        "consistency_loss": float(consistency.detach()),
    }


def direct_sensory_loss(output, actions: torch.Tensor,
                        mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Diagnostic bypass of learned memory, using the final sensory state."""
    rows = torch.arange(actions.shape[0], device=actions.device)
    final_indices = mask.sum(1) - 1
    targets = actions[rows, final_indices]
    logits = output.observation_logits[rows, final_indices]
    loss = nn.functional.cross_entropy(logits, targets)
    predictions = logits.argmax(-1)
    return loss, {
        "reward": float(torch.where(predictions == targets, 1.0, -1.0).mean()),
        "accuracy": float((predictions == targets).float().mean()),
        "thought_steps": 0.0,
        "latency_multiplier": 0.0,
        "direct_sensory_loss": float(loss.detach()),
    }


@torch.no_grad()
def evaluate(model: BitterLessonAgent, loader: DataLoader,
             device: torch.device, speed_bonus: float,
             fixed_thought_steps: bool = False,
             direct_sensory: bool = False) -> dict[str, object]:
    model.eval()
    total = correct = 0
    total_steps = 0.0
    correct_speed = 0.0
    by_premises: dict[int, list[int]] = {}
    correct_by_depth: torch.Tensor | None = None
    started = time.perf_counter()
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        pcm = batch["pcm"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        groups = batch["groups"]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(frames, pcm, mask)
        if direct_sensory:
            chosen_steps = torch.zeros(frames.shape[0], dtype=torch.long, device=device)
        elif fixed_thought_steps:
            chosen_steps = torch.full((frames.shape[0],), output.answer_logits.shape[1] - 1,
                                      dtype=torch.long, device=device)
        else:
            hazards = torch.sigmoid(output.halt_logits)
            hazards[:, -1] = 1.0
            halted = hazards >= 0.5
            chosen_steps = halted.float().argmax(1)
        rows = torch.arange(frames.shape[0], device=device)
        final_indices = mask.sum(1) - 1
        targets = actions[rows, final_indices]
        if direct_sensory:
            predictions = output.observation_logits[rows, final_indices].argmax(-1)
        else:
            predictions = output.answer_logits[rows, chosen_steps].argmax(-1)
            depth_matches = output.answer_logits.argmax(-1) == targets[:, None]
            batch_depth_correct = depth_matches.sum(0).cpu()
            correct_by_depth = (batch_depth_correct if correct_by_depth is None
                                else correct_by_depth + batch_depth_correct)
        matches = predictions == targets
        speed = 1.0 - chosen_steps.float() / max(1, output.answer_logits.shape[1] - 1)
        total += frames.shape[0]
        correct += int(matches.sum())
        total_steps += float((chosen_steps + 1).sum())
        correct_speed += float((speed * matches).sum())
        bucket_ids = [group if group >= 0 else premises
                      for group, premises in zip(groups.tolist(), final_indices.tolist())]
        for premises, matched in zip(bucket_ids, matches.tolist()):
            bucket = by_premises.setdefault(int(premises), [0, 0])
            bucket[0] += int(matched)
            bucket[1] += 1
    accuracy = correct / max(1, total)
    # Accuracy is lexicographically primary. Latency is scored only after the
    # policy has cleared a deliberately conservative above-chance threshold.
    latency_active = accuracy >= 0.55
    mean_reward = (2.0 * accuracy - 1.0)
    if latency_active:
        mean_reward += speed_bonus * correct_speed / max(1, total)
    return {
        "episodes": total,
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "latency_reward_active": latency_active,
        "mean_thought_steps": total_steps / max(1, total),
        "milliseconds_per_episode": (time.perf_counter() - started) * 1000 / max(1, total),
        "accuracy_by_premises": {
            str(length): right / count for length, (right, count) in sorted(by_premises.items())
        },
        "accuracy_by_thought_depth": ({
            str(depth + 1): int(right) / max(1, total)
            for depth, right in enumerate(correct_by_depth)
        } if correct_by_depth is not None else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("1m", "2m", "5m", "20m"), default="1m")
    parser.add_argument("--memory-core",
                        choices=("soft_slots", "residual_slots", "residual_gru",
                                 "event_transformer"), default="soft_slots")
    parser.add_argument("--thought-steps", type=int,
                        help="override the scale preset's recurrent computation budget")
    parser.add_argument("--thought-dynamics", choices=("replace", "gated_residual"),
                        default="replace")
    parser.add_argument("--action-count", type=int, default=5,
                        help="shared motor vocabulary; use 8 for choice reaction tasks")
    parser.add_argument("--random-thought-depth", action="store_true",
                        help="train each example at a uniformly sampled compute depth")
    parser.add_argument("--consistency-weight", type=float, default=0.25)
    parser.add_argument("--adaptive-curriculum", action="store_true",
                        help="unlock the next length only after sustained held-out mastery")
    parser.add_argument("--mastery-threshold", type=float, default=0.95)
    parser.add_argument("--mastery-patience", type=int, default=2)
    parser.add_argument("--curriculum-eval-samples", type=int, default=1000,
                        help="held-out validation examples per active length and epoch")
    parser.add_argument("--curriculum-start-stage", type=int, default=1,
                        help="number of already-unlocked lengths when resuming mastery training")
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--eval-samples", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-premises", default="2,3,4,6,8,12,16")
    parser.add_argument("--eval-premises", default="2,4,8,16,24,32,64")
    parser.add_argument("--speed-bonus", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--learning-signal",
                        choices=("reinforce", "reinforce_fixed", "q_learning", "verifier",
                                 "direct_sensory"),
                        default="reinforce")
    parser.add_argument("--q-epsilon", type=float, default=0.2)
    parser.add_argument("--overfit-fixed", action="store_true",
                        help="evaluate on the exact fixed training episodes")
    parser.add_argument("--randomize-rendering", action="store_true",
                        help="derive public card colors independently for every episode")
    parser.add_argument("--training-tasks",
                        choices=("chain", "mixed_structural", "parity",
                                 "cyclic", "choice_reaction", "mixed_cognitive"),
                        default="chain")
    parser.add_argument("--evaluation-tasks",
                        choices=("chain", "parity", "cyclic", "choice_reaction",
                                 "mixed_cognitive"),
                        default="chain")
    parser.add_argument("--choice-counts", default="2,3,4,5,6,7,8")
    parser.add_argument("--choice-distractors", type=int, default=0)
    parser.add_argument("--choice-delay-frames", type=int, default=0)
    parser.add_argument("--choice-audio-distractors", type=int, default=0)
    parser.add_argument("--choice-target-like-distractors", type=int, default=0)
    parser.add_argument("--choice-temporal-distractors", type=int, default=0)
    parser.add_argument("--reaction-fraction", type=float, default=0.5,
                        help="reaction share of mixed-cognitive training/evaluation")
    parser.add_argument("--reasoning-family", choices=("parity", "cyclic", "mixed"),
                        default="parity",
                        help="reasoning family inside a mixed-cognitive stream")
    parser.add_argument("--cyclic-fraction", type=float, default=0.25,
                        help="cyclic share of reasoning when its family is mixed")
    parser.add_argument("--logic-modulus", type=int, choices=(2, 4, 8), default=2,
                        help="states composed by each cyclic relation")
    parser.add_argument("--cyclic-premises", default="",
                        help="optional cyclic lengths, independent of parity replay lengths")
    parser.add_argument("--branched-configs", default="8:2,16:4,32:4,16:8")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-precision", action="store_true",
                        help="disable accelerator autocast for optimization diagnostics")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="initialize model weights for a matched fine-tuning run")
    args = parser.parse_args()
    if not 0.0 < args.reaction_fraction < 1.0:
        raise ValueError("reaction fraction must be strictly between zero and one")
    if not 0.0 < args.cyclic_fraction < 1.0:
        raise ValueError("cyclic fraction must be strictly between zero and one")
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_lengths = tuple(map(int, args.train_premises.split(",")))
    eval_lengths = tuple(map(int, args.eval_premises.split(",")))
    cyclic_lengths = (tuple(map(int, args.cyclic_premises.split(",")))
                      if args.cyclic_premises else None)
    choice_counts = tuple(map(int, args.choice_counts.split(",")))
    difficulties = reaction_difficulties(choice_counts, args.choice_distractors,
                                         args.choice_delay_frames,
                                         args.choice_audio_distractors,
                                         args.choice_target_like_distractors,
                                         args.choice_temporal_distractors)
    curriculum_units = (choice_counts if args.training_tasks == "choice_reaction"
                        else train_lengths)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "collate_fn": collate_episodes, "pin_memory": device.type == "cuda"}
    if args.evaluation_tasks == "choice_reaction":
        from .choice_reaction import ChoiceReactionDataset
        evaluation_data = ChoiceReactionDataset(args.eval_samples, difficulties,
                                                 start_seed=500_000, heldout=True)
    elif args.evaluation_tasks == "mixed_cognitive":
        from .choice_reaction import ChoiceReactionDataset, CognitiveMixtureDataset
        reaction_samples = round(args.eval_samples * args.reaction_fraction)
        reasoning_samples = args.eval_samples - reaction_samples
        reasoning_data = reasoning_dataset(
            args.reasoning_family, reasoning_samples, eval_lengths,
            args.logic_modulus, args.cyclic_fraction,
            start_seed=500_000, heldout=True, cyclic_lengths=cyclic_lengths)
        evaluation_data = CognitiveMixtureDataset(
            reasoning_data,
            ChoiceReactionDataset(reaction_samples, difficulties,
                                  start_seed=600_000, heldout=True))
    elif args.evaluation_tasks == "cyclic":
        from .cyclic_transfer import CyclicDataset
        evaluation_data = CyclicDataset(args.eval_samples, eval_lengths,
                                        args.logic_modulus,
                                        start_seed=500_000, heldout=True)
    elif args.evaluation_tasks == "parity":
        if args.overfit_fixed:
            raise ValueError("parity evaluation does not support fixed chain overfit mode")
        from .parity_transfer import ParityDataset
        evaluation_data = ParityDataset(args.eval_samples, eval_lengths,
                                        start_seed=500_000, heldout=True)
    elif args.overfit_fixed:
        if train_lengths != eval_lengths:
            raise ValueError("overfit mode requires identical train/eval premise choices")
        evaluation_data = EpisodeDataset(args.train_samples, premise_choices=train_lengths,
                                         entity_count=128,
                                         randomize_rendering=args.randomize_rendering)
    else:
        evaluation_data = EpisodeDataset(args.eval_samples, start_seed=100_000,
                                         premise_choices=eval_lengths, heldout=True,
                                         final=True, entity_count=128,
                                         randomize_rendering=args.randomize_rendering)
    evaluation = DataLoader(evaluation_data, shuffle=False, **loader_options)
    config = model_config(args.scale)
    config["memory_core"] = args.memory_core
    config["thought_dynamics"] = args.thought_dynamics
    config["action_count"] = args.action_count
    if args.thought_steps is not None:
        if args.thought_steps < 1:
            raise ValueError("thought steps must be positive")
        config["max_thought_steps"] = args.thought_steps
    model = BitterLessonAgent(**config).to(device)
    initial_metadata = None
    if args.initial_checkpoint is not None:
        payload = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        initial_metadata = payload.get("metadata")
        initial_config = {}
        initial_action_count = 5
        if initial_metadata is not None:
            initial_config = dict(initial_metadata.get("config", {}))
            requested_config = dict(config)
            # Recurrent thought steps share weights, so changing only their
            # iteration count is a valid matched-compute initialization.
            initial_config.pop("max_thought_steps", None)
            requested_config.pop("max_thought_steps", None)
            # A legacy replace-dynamics checkpoint supplies the sensory and
            # sequence weights; the new residual gate is deliberately fresh.
            initial_config.pop("thought_dynamics", None)
            requested_config.pop("thought_dynamics", None)
            initial_actions = initial_config.pop("action_count", 5)
            initial_action_count = initial_actions
            requested_actions = requested_config.pop("action_count", 5)
            if requested_actions < initial_actions:
                raise ValueError("cannot contract an initialized action vocabulary")
            if initial_config != requested_config:
                raise ValueError("initial checkpoint model configuration does not match")
        incompatible = load_initial_state(
            model, payload,
            allow_action_expansion=(args.action_count > initial_action_count))
        allowed_missing = ({"thought_gate.weight", "thought_gate.bias"}
                           if args.thought_dynamics == "gated_residual" else set())
        if args.action_count > initial_action_count:
            allowed_missing |= {"observation_head.weight", "observation_head.bias",
                                "answer_head.weight", "answer_head.bias"}
        missing = set(incompatible.missing_keys)
        if not missing.issubset(allowed_missing) or incompatible.unexpected_keys:
            raise ValueError(f"incompatible initial checkpoint weights: {incompatible}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    accuracy_ema = 0.5
    if not 1 <= args.curriculum_start_stage <= len(curriculum_units):
        raise ValueError("curriculum start stage must index the configured training lengths")
    curriculum_stage = args.curriculum_start_stage
    mastery_streak = 0
    best_curriculum_score = -1.0
    best_curriculum_epoch = 0
    best_curriculum_stage = 0
    best_state = None
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        stage = (curriculum_stage if args.adaptive_curriculum else
                 max(1, (epoch + 1) * len(curriculum_units) // args.epochs))
        epoch_lengths = curriculum_units[:stage]
        sampling_lengths = (curriculum_sampling_choices(epoch_lengths)
                            if args.adaptive_curriculum else epoch_lengths)
        if args.training_tasks == "choice_reaction":
            from .choice_reaction import ChoiceReactionDataset
            active_difficulties = reaction_difficulties(
                choice_counts[:stage] if args.adaptive_curriculum else choice_counts,
                args.choice_distractors, args.choice_delay_frames,
                args.choice_audio_distractors, args.choice_target_like_distractors,
                args.choice_temporal_distractors)
            training_data = ChoiceReactionDataset(args.train_samples, active_difficulties,
                                                   start_seed=epoch * args.train_samples)
        elif args.training_tasks == "mixed_cognitive":
            from .choice_reaction import ChoiceReactionDataset, CognitiveMixtureDataset
            reaction_samples = round(args.train_samples * args.reaction_fraction)
            reasoning_samples = args.train_samples - reaction_samples
            reasoning_data = reasoning_dataset(
                args.reasoning_family, reasoning_samples, sampling_lengths,
                args.logic_modulus, args.cyclic_fraction,
                start_seed=epoch * args.train_samples,
                cyclic_lengths=cyclic_lengths)
            training_data = CognitiveMixtureDataset(
                reasoning_data,
                ChoiceReactionDataset(reaction_samples, difficulties,
                                      start_seed=epoch * args.train_samples +
                                      reasoning_samples))
        elif args.training_tasks == "parity":
            from .parity_transfer import ParityDataset
            training_data = ParityDataset(args.train_samples, sampling_lengths,
                                          start_seed=epoch * args.train_samples)
        elif args.training_tasks == "cyclic":
            from .cyclic_transfer import CyclicDataset
            training_data = CyclicDataset(args.train_samples, sampling_lengths,
                                          args.logic_modulus,
                                          start_seed=epoch * args.train_samples)
        elif args.training_tasks == "mixed_structural":
            from .structural_transfer import MixedStructuralDataset
            branched_configs = tuple(tuple(map(int, item.split(":")))
                                     for item in args.branched_configs.split(","))
            training_data = MixedStructuralDataset(args.train_samples, epoch_lengths,
                                                    branched_configs)
        else:
            training_data = EpisodeDataset(args.train_samples,
                                           premise_choices=epoch_lengths, entity_count=128,
                                           randomize_rendering=args.randomize_rendering)
        training = DataLoader(training_data, shuffle=True, **loader_options)
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch in training:
            frames = batch["frames"].to(device, non_blocking=True)
            pcm = batch["pcm"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and not args.full_precision):
                output = model(frames, pcm, mask)
                if args.learning_signal == "q_learning":
                    loss, metrics = q_learning_loss(output, actions, mask, args.q_epsilon)
                elif args.learning_signal == "verifier":
                    if args.random_thought_depth:
                        loss, metrics = randomized_depth_verifier_loss(
                            output, actions, mask, args.consistency_weight)
                    else:
                        loss, metrics = verifier_loss(output, actions, mask)
                elif args.learning_signal == "direct_sensory":
                    loss, metrics = direct_sensory_loss(output, actions, mask)
                else:
                    # Do not let chance-level guessing optimize for speed. The
                    # latency objective ramps in only after sustained accuracy.
                    latency_multiplier = max(0.0, min(1.0, (accuracy_ema - 0.55) / 0.25))
                    loss, metrics = policy_loss(output, actions, mask, args.speed_bonus,
                                                args.entropy_weight, args.value_weight,
                                                latency_multiplier,
                                                fixed_thoughts=(args.learning_signal
                                                                == "reinforce_fixed"))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            accuracy_ema = 0.99 * accuracy_ema + 0.01 * metrics["accuracy"]
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        record = {"epoch": epoch + 1, "premise_choices": epoch_lengths,
                  "sampling_choices": sampling_lengths,
                  **{key: value / max(1, batches) for key, value in totals.items()}}
        if args.adaptive_curriculum:
            if args.training_tasks == "parity":
                validation_data = ParityDataset(
                    args.curriculum_eval_samples * len(epoch_lengths), epoch_lengths,
                    start_seed=700_000, heldout=True)
            elif args.training_tasks == "cyclic":
                from .cyclic_transfer import CyclicDataset
                validation_data = CyclicDataset(
                    args.curriculum_eval_samples * len(epoch_lengths), epoch_lengths,
                    args.logic_modulus, start_seed=700_000, heldout=True)
            elif args.training_tasks == "choice_reaction":
                from .choice_reaction import ChoiceReactionDataset
                active_counts = choice_counts[:stage]
                validation_data = ChoiceReactionDataset(
                    args.curriculum_eval_samples * len(active_counts),
                    reaction_difficulties(active_counts, args.choice_distractors,
                                          args.choice_delay_frames,
                                          args.choice_audio_distractors,
                                          args.choice_target_like_distractors,
                                          args.choice_temporal_distractors),
                    start_seed=700_000, heldout=True)
                epoch_lengths = active_counts
            else:
                raise ValueError("adaptive curriculum requires parity, cyclic, or choice reaction")
            validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
            validation_result = evaluate(model, validation_loader, device, 0.0,
                                         fixed_thought_steps=True)
            validation_by_length = validation_result["accuracy_by_premises"]
            curriculum_score = min(validation_by_length[str(length)]
                                   for length in epoch_lengths)
            if (stage > best_curriculum_stage or
                    (stage == best_curriculum_stage and
                     curriculum_score > best_curriculum_score)):
                best_curriculum_stage = stage
                best_curriculum_score = curriculum_score
                best_curriculum_epoch = epoch + 1
                best_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}
            mastered = all(validation_by_length[str(length)] >= args.mastery_threshold
                           for length in epoch_lengths)
            mastery_streak = mastery_streak + 1 if mastered else 0
            record["curriculum_validation"] = validation_by_length
            record["curriculum_score"] = curriculum_score
            record["mastery_streak"] = mastery_streak
            if mastery_streak >= args.mastery_patience:
                if curriculum_stage < len(curriculum_units):
                    curriculum_stage += 1
                    mastery_streak = 0
                    record["unlocked_length"] = curriculum_units[curriculum_stage - 1]
                else:
                    record["curriculum_complete"] = True
        history.append(record)
        print(json.dumps(record), flush=True)
        if record.get("curriculum_complete"):
            break
    if args.adaptive_curriculum and best_state is not None:
        model.load_state_dict(best_state)
    result = evaluate(model, evaluation, device, args.speed_bonus,
                      fixed_thought_steps=args.learning_signal in {
                          "reinforce_fixed", "q_learning", "verifier"
                      },
                      direct_sensory=args.learning_signal == "direct_sensory")
    metadata = {"schema": "syllogimous-bitter-lesson-v1", "scale": args.scale,
                "config": config, "parameters": parameter_count(model),
                "reward": {"correct": 1.0, "incorrect": -1.0,
                           "max_correct_speed_bonus": args.speed_bonus,
                           "latency_ramp_accuracy": [0.55, 0.80]},
                "semantic_auxiliary_labels": False, "seed": args.seed,
                "learning_signal": args.learning_signal,
                "overfit_fixed": args.overfit_fixed,
                "randomize_rendering": args.randomize_rendering,
                "training_tasks": args.training_tasks,
                "evaluation_tasks": args.evaluation_tasks,
                "choice_counts": choice_counts,
                "choice_distractors": args.choice_distractors,
                "choice_delay_frames": args.choice_delay_frames,
                "choice_audio_distractors": args.choice_audio_distractors,
                "choice_target_like_distractors": args.choice_target_like_distractors,
                "choice_temporal_distractors": args.choice_temporal_distractors,
                "reaction_fraction": args.reaction_fraction,
                "reasoning_family": args.reasoning_family,
                "cyclic_fraction": args.cyclic_fraction,
                "logic_modulus": args.logic_modulus,
                "cyclic_premises": list(cyclic_lengths) if cyclic_lengths else None,
                "initial_checkpoint": (str(args.initial_checkpoint)
                                       if args.initial_checkpoint is not None else None),
                "random_thought_depth": args.random_thought_depth,
                "consistency_weight": args.consistency_weight,
                "adaptive_curriculum": args.adaptive_curriculum,
                "mastery_threshold": args.mastery_threshold,
                "mastery_patience": args.mastery_patience,
                "full_precision": args.full_precision,
                "curriculum_start_stage": args.curriculum_start_stage,
                "best_curriculum_epoch": best_curriculum_epoch,
                "best_curriculum_stage": best_curriculum_stage,
                "best_curriculum_score": best_curriculum_score,
                "train_premises": train_lengths, "eval_premises": eval_lengths,
                "history": history, "evaluation": result,
                "training_seconds": time.perf_counter() - training_started}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, args.checkpoint)
    args.report.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint": str(args.checkpoint), "report": str(args.report),
                      "parameters": metadata["parameters"], "evaluation": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
