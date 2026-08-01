"""Tiny reward-only learning probe on the Brain Workshop-style gym.

This is deliberately the first learning rung rather than a claim of mastery:
vision-only n-back-1, two opaque keypress actions, scalar verifier reward, and
an external encoder/controller/output-decoder decomposition.  The verifier
keeps the match target private; it is used only by ``score_action``.

The default run is short enough to act as a gate.  A healthy run should show
live gradients and a changing policy before it is given a longer budget.  The
script also evaluates a history-reset control, which prevents a lucky
per-frame classifier from being mistaken for working memory.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .amodal_interface import AmodalEvent
from .amodal_runtime import AmodalInputBus, OpaqueProtocolDecoder
from .brainworkshop_gym import BrainWorkshopConfig, generate_brainworkshop_episode
from .model import UnifiedCognitiveController


@dataclass
class Rollout:
    log_probs: torch.Tensor
    entropies: torch.Tensor
    rewards: torch.Tensor
    actions: torch.Tensor


class BrainWorkshopPolicy(nn.Module):
    """N encoders -> amodal bus -> one controller -> one decoder."""

    def __init__(self, *, width: int = 32, intention_width: int = 16,
                 workspace_slots: int = 2,
                 controller: UnifiedCognitiveController | None = None) -> None:
        super().__init__()
        from .brainworkshop_gym import BrainWorkshopVisionEncoder

        self.controller = controller or UnifiedCognitiveController(
            width=width, workspace_slots=workspace_slots,
            intention_width=intention_width)
        width = self.controller.width
        intention_width = self.controller.intention_width
        self.encoder = BrainWorkshopVisionEncoder(width)
        self.input_bus = AmodalInputBus(width)
        # The legacy model owns adapters for compatibility.  The probe removes
        # them so only this independent output adapter formats the intention.
        self.controller.vision = None
        self.controller.actuator = None
        self.decoder = OpaqueProtocolDecoder(intention_width, commands=2)

    def initial_state(self, batch: int, device: torch.device):
        return self.controller.initial_state(batch, device=device)


def _device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def _make_batch(config: BrainWorkshopConfig, *, batch_size: int,
                seed: int, device: torch.device):
    episodes = [generate_brainworkshop_episode(
        config, seed=seed + index, device=device) for index in range(batch_size)]
    frames = torch.stack([
        torch.stack([observation.vision for observation in episode.observations])
        for episode in episodes])
    return episodes, frames


def _rollout(policy: BrainWorkshopPolicy, config: BrainWorkshopConfig,
             *, batch_size: int, seed: int, device: torch.device,
             sample: bool, reset_history: bool = False,
             shuffle_time: bool = False) -> Rollout:
    episodes, frames = _make_batch(
        config, batch_size=batch_size, seed=seed, device=device)
    if shuffle_time:
        # This is an adversarial control: preserve the frame multiset but break
        # the temporal relation that n-back requires.
        permutations = torch.stack([
            torch.randperm(config.trials, device=device)
            for _ in range(batch_size)
        ])
        frames = torch.stack([
            frames[index, permutations[index]] for index in range(batch_size)
        ])
    state = policy.initial_state(batch_size, device)
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    has_feedback = torch.zeros(batch_size, device=device)
    log_probs, entropies, rewards, actions = [], [], [], []
    for trial in range(config.trials):
        if reset_history and trial:
            state = policy.initial_state(batch_size, device)
            # Do not mutate tensors already retained in the rollout log.
            # In-place zeroing here would silently erase earlier rewards and
            # actions, making the history-reset control look better or worse
            # for purely bookkeeping reasons.
            previous_action = torch.zeros(
                batch_size, dtype=torch.long, device=device)
            previous_reward = torch.zeros(batch_size, device=device)
            has_feedback = torch.zeros(batch_size, device=device)
        encoded = policy.encoder(frames[:, trial])
        event = policy.input_bus([AmodalEvent(payload=encoded)])
        core, state = policy.controller.step_event(
            event, state, previous_action, previous_reward, has_feedback)
        logits = policy.decoder(core.intent_event)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if sample else logits.argmax(dim=-1)
        reward = torch.tensor([
            episode.score_action(trial, int(action[index]), latency_ms=0.0)
            for index, episode in enumerate(episodes)
        ], device=device)
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        rewards.append(reward)
        actions.append(action)
        previous_action = action
        previous_reward = reward
        has_feedback.fill_(1.0)
    return Rollout(
        torch.stack(log_probs), torch.stack(entropies),
        torch.stack(rewards), torch.stack(actions))


def _supervised_step(policy: BrainWorkshopPolicy, config: BrainWorkshopConfig,
                     *, batch_size: int, seed: int,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Disposable verifier-label ceiling probe.

    Labels are used only to answer the architecture question and are never
    part of the reward-only claim.  Feedback still comes from the sampled
    policy action's scalar verifier reward, so this checks the same recurrent
    interface rather than feeding the answer back as an input.
    """
    episodes, frames = _make_batch(
        config, batch_size=batch_size, seed=seed, device=device)
    state = policy.initial_state(batch_size, device)
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    has_feedback = torch.zeros(batch_size, device=device)
    losses, rewards = [], []
    for trial in range(config.trials):
        encoded = policy.encoder(frames[:, trial])
        event = policy.input_bus([AmodalEvent(payload=encoded)])
        core, state = policy.controller.step_event(
            event, state, previous_action, previous_reward, has_feedback)
        logits = policy.decoder(core.intent_event)
        targets = torch.tensor(
            [episode.verifier_targets()[trial] for episode in episodes],
            dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, targets))
        action = logits.argmax(dim=-1)
        reward = torch.tensor([
            episode.score_action(trial, int(action[index]), latency_ms=0.0)
            for index, episode in enumerate(episodes)
        ], device=device)
        rewards.append(reward)
        previous_action = action
        previous_reward = reward
        has_feedback = torch.ones(batch_size, device=device)
    return torch.stack(losses).mean(), torch.stack(rewards)


@torch.no_grad()
def _evaluate(policy: BrainWorkshopPolicy, config: BrainWorkshopConfig, *,
              count: int, seed: int, device: torch.device,
              reset_history: bool = False, shuffle_time: bool = False) -> dict:
    rollout = _rollout(
        policy, config, batch_size=count, seed=seed, device=device,
        sample=False, reset_history=reset_history, shuffle_time=shuffle_time)
    rewards = rollout.rewards
    return {
        "accuracy": float((rewards > 0).float().mean()),
        "mean_reward": float(rewards.mean()),
        "trial_count": int(rewards.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--checkpoint-in", type=Path,
                        help="optional promoted controller to keep frozen")
    parser.add_argument("--supervised-diagnostic", action="store_true",
                        help="use disposable verifier labels for a ceiling probe")
    parser.add_argument("--seed", type=int, default=44011)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--position-vocab", type=int, default=2,
                        choices=(2, 4, 8))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 4 or args.eval_count < 4:
        raise ValueError("updates, batch size, and eval count must be positive")
    if not 0.0 < args.discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")

    torch.manual_seed(args.seed)
    device = _device(args.device)
    config = BrainWorkshopConfig(
        n_back=1, trials=args.trials, position_vocab=args.position_vocab,
        modalities=("vision",), trial_ms=1_000,
        # Independent seeded match flags prevent a fixed-count clock policy
        # from looking like working memory.  The balanced mode remains the
        # default gym control; this training rung explicitly removes that
        # shortcut while keeping all outcomes verifier-private.
        balanced_matches=False,
    )
    frozen_controller = None
    if args.checkpoint_in:
        payload = torch.load(args.checkpoint_in, map_location=device,
                             weights_only=False)
        frozen_controller = UnifiedCognitiveController(
            **payload["model_configuration"]).to(device)
        frozen_controller.load_state_dict(payload["state_dict"])
        frozen_controller.vision = None
        frozen_controller.actuator = None
    if args.checkpoint_in and args.supervised_diagnostic:
        raise ValueError("the supervised diagnostic is for a fresh policy only")
    policy = BrainWorkshopPolicy(controller=frozen_controller).to(device)
    if frozen_controller is not None:
        for parameter in policy.controller.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in policy.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    started = time.perf_counter()
    history = []
    before = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 10_000,
        device=device)
    for update in range(1, args.updates + 1):
        policy.train()
        if args.supervised_diagnostic:
            loss, diagnostic_rewards = _supervised_step(
                policy, config, batch_size=args.batch_size,
                seed=args.seed + update * 1_000, device=device)
            batch_accuracy = float((diagnostic_rewards > 0).float().mean())
            batch_mean_reward = float(diagnostic_rewards.mean())
            policy_entropy = 0.0
        else:
            rollout = _rollout(
                policy, config, batch_size=args.batch_size,
                seed=args.seed + update * 1_000, device=device, sample=True)
            returns = torch.zeros_like(rollout.rewards)
            running = torch.zeros(args.batch_size, device=device)
            for trial in range(args.trials - 1, -1, -1):
                running = rollout.rewards[trial] + args.discount * running
                returns[trial] = running
            # A per-time baseline uses only verifier rewards from this batch;
            # it is a variance reducer, not an additional task label.
            advantages = returns - returns.mean(dim=1, keepdim=True)
            loss = -(
                advantages.detach() * rollout.log_probs).mean()
            loss = loss - args.entropy_coef * rollout.entropies.mean()
            batch_accuracy = float((rollout.rewards > 0).float().mean())
            batch_mean_reward = float(rollout.rewards.mean())
            policy_entropy = float(rollout.entropies.mean().detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            policy.parameters(), 1.0).detach())
        optimizer.step()
        record = {
            "update": update,
            "loss": float(loss.detach()),
            "batch_accuracy": batch_accuracy,
            "batch_mean_reward": batch_mean_reward,
            "policy_entropy": policy_entropy,
            "gradient_norm": gradient_norm,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    policy.eval()
    after = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 10_000,
        device=device)
    reset = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 20_000,
        device=device, reset_history=True)
    shuffled = _evaluate(
        policy, config, count=args.eval_count, seed=args.seed + 30_000,
        device=device, shuffle_time=True)
    report = {
        "experiment": "brainworkshop_vision_nback1_reward_only",
        "objective": (
            "supervised_diagnostic" if args.supervised_diagnostic
            else "reward_only"),
        "controller_frozen": frozen_controller is not None,
        "device": str(device),
        "config": config.__dict__,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "unique_training_lifetimes": args.updates * args.batch_size,
        "before": before,
        "after": after,
        "history_reset_control": reset,
        "time_shuffle_control": shuffled,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "gate": {
            "gradient_alive": all(
                entry["gradient_norm"] > 1e-8 for entry in history),
            "policy_changed": after["mean_reward"] != before["mean_reward"],
            "reset_control_recorded": True,
            "accepted_for_longer_run": (
                after["accuracy"] > before["accuracy"] + 0.05
                and after["accuracy"] > 0.60
                and reset["accuracy"] < after["accuracy"] - 0.05),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "brainworkshop-policy.v1",
            "model_configuration": {
                "width": 32, "intention_width": 16, "workspace_slots": 2,
            },
            "state_dict": policy.state_dict(),
            "config": config.__dict__,
            "report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "final": report["gate"], "before": before, "after": after,
        "history_reset": reset, "time_shuffle": shuffled,
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
