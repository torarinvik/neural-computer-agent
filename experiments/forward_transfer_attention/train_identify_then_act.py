"""Two-decision identify-then-act curriculum rung.

An opaque probe command causes a visible cursor displacement.  The cursor then
resets and a target appears.  The second opaque command succeeds only if the
agent retained which command caused which effect in this lifetime.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import IMAGE_HEIGHT, IMAGE_WIDTH
from experiments.syllogimous_latent_agent.model import VisionEncoder

from .environment import COLORS
from .train import seed_everything
from .train_action_conditioned_success import selected_success_loss
from .train_actuator_transfer import SuccessSystem
from .train_micro_intercept import _ranked_balanced
from .train_zero_label_predictive_state import (
    _correlation_loss,
    _ema_update,
    _standardized_prediction_loss,
    _variance_loss,
)


PRETRAIN_START = 101_000_000
TRAIN_START = 103_000_000
TEST_START = 105_000_000
NULL_ACTION = 2
PROBE_PIXELS = 22
ACT_PIXELS = 28
PROTOCOLS = ((-1, 1), (1, -1))


class DirectSuccessSystem(nn.Module):
    """Minimal attempted-action success head without a latent bottleneck."""

    def __init__(self, hidden: int, width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


def make_readout(
        kind: str, hidden: int, intention_width: int) -> nn.Module:
    if kind == "bottleneck":
        return SuccessSystem(hidden, intention_width, actions=2)
    if kind == "direct":
        return DirectSuccessSystem(hidden, width=intention_width)
    raise ValueError(kind)


def _render(
        seed: int, frame_index: int, *, cursor_x: int,
        target_direction: int | None, target_color: int,
        cursor_color: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = tuple(int(value) for value in rng.integers(5, 23, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
        outline=(72, 82, 108), width=2)
    for nuisance in range(3):
        x = int(rng.integers(18, IMAGE_WIDTH - 18))
        y = 18 + nuisance * 8
        color = tuple(int(value) for value in rng.integers(28, 68, size=3))
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    if target_direction is not None:
        target_x = IMAGE_WIDTH // 2 + target_direction * ACT_PIXELS
        draw.ellipse(
            (target_x - 8, 20, target_x + 8, 36),
            fill=COLORS[target_color], outline=(245, 245, 250), width=2)
    cursor_y = 71
    draw.rounded_rectangle(
        (cursor_x - 9, cursor_y - 5, cursor_x + 9, cursor_y + 5),
        radius=3, fill=COLORS[cursor_color],
        outline=(245, 245, 250), width=2)
    pulse = 25 + frame_index * 9
    draw.rectangle((7, 85, 11, 89), fill=(pulse, pulse, pulse))
    return np.asarray(image, dtype=np.uint8).copy()


def identify_batch(
        start: int, count: int, *, heldout: bool,
        fixed_protocol: int | None = None,
        fixed_probe_action: int | None = None,
        probe_action_one_fraction: float | None = None,
        fixed_target_direction: int | None = None,
        no_probe_effect: bool = False,
        missing_consequence: bool = False,
        swap_protocol: bool = False,
        reverse_target: bool = False) -> dict[str, torch.Tensor]:
    if count % 8:
        raise ValueError("count must be divisible by eight")
    seeds = list(range(start, start + count))
    protocol_ids = _ranked_balanced(
        seeds, heldout, "identify-protocol", classes=2)
    probe_actions = _ranked_balanced(
        seeds, heldout, "identify-probe", classes=2)
    if probe_action_one_fraction is not None:
        if probe_action_one_fraction not in (0.125, 0.25, 0.5):
            raise ValueError(
                "probe_action_one_fraction must be 0.125, 0.25, or 0.5")
        eighths = int(probe_action_one_fraction * 8)
        probe_buckets = _ranked_balanced(
            seeds, heldout, "identify-probe-curriculum", classes=8)
        probe_actions = [
            int(bucket < eighths) for bucket in probe_buckets]
    if fixed_probe_action is not None:
        if probe_action_one_fraction is not None:
            raise ValueError(
                "fixed_probe_action and probe_action_one_fraction conflict")
        probe_actions = [fixed_probe_action] * count
    target_ids = _ranked_balanced(
        seeds, heldout, "identify-target", classes=2)
    logged_choices = _ranked_balanced(
        seeds, heldout, "identify-choice", classes=2)
    color_ids = _ranked_balanced(
        seeds, heldout, "identify-color", classes=4)
    all_frames, all_actions = [], []
    rewards, correct_actions, private_protocols = [], [], []
    center = IMAGE_WIDTH // 2
    for index, seed in enumerate(seeds):
        protocol_id = (
            fixed_protocol if fixed_protocol is not None
            else protocol_ids[index])
        if swap_protocol:
            protocol_id = 1 - protocol_id
        protocol = PROTOCOLS[protocol_id]
        probe_action = probe_actions[index]
        target_direction = (
            fixed_target_direction
            if fixed_target_direction is not None
            else (-1, 1)[target_ids[index]])
        if reverse_target:
            target_direction = -target_direction
        choice = logged_choices[index]
        probe_effect = 0 if no_probe_effect else protocol[probe_action]
        visible_probe_x = center + probe_effect * PROBE_PIXELS
        if missing_consequence:
            visible_probe_x = center
        correct = protocol.index(target_direction)
        final_x = center + protocol[choice] * ACT_PIXELS
        base = seed * 17 + (9 if heldout else 0)
        target_color = color_ids[index]
        cursor_color = (target_color + 1) % len(COLORS)
        frames = [
            _render(
                base, 0, cursor_x=center, target_direction=None,
                target_color=target_color, cursor_color=cursor_color),
            _render(
                base, 1, cursor_x=visible_probe_x, target_direction=None,
                target_color=target_color, cursor_color=cursor_color),
            _render(
                base, 2, cursor_x=center,
                target_direction=target_direction,
                target_color=target_color, cursor_color=cursor_color),
            _render(
                base, 3, cursor_x=final_x,
                target_direction=target_direction,
                target_color=target_color, cursor_color=cursor_color),
        ]
        all_frames.append(np.stack(frames))
        all_actions.append((probe_action, NULL_ACTION, choice))
        rewards.append(float(choice == correct))
        correct_actions.append(correct)
        private_protocols.append(protocol_id)
    frame_tensor = torch.from_numpy(np.stack(all_frames)).permute(
        0, 1, 4, 2, 3).float().div_(255.0)
    actions = torch.tensor(all_actions, dtype=torch.long)
    previous_actions = torch.stack([
        torch.full((count,), NULL_ACTION, dtype=torch.long),
        actions[:, 0],
        torch.full((count,), NULL_ACTION, dtype=torch.long),
    ], dim=1)
    return {
        "frames": frame_tensor,
        "transition_actions": actions,
        "previous_actions": previous_actions,
        "attempted_actions": actions[:, 2],
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "correct_actions": torch.tensor(correct_actions, dtype=torch.long),
        "probe_actions": actions[:, 0],
        "private_protocol_ids": torch.tensor(
            private_protocols, dtype=torch.long),
    }


class ActionHistoryCore(nn.Module):
    def __init__(self, hidden: int = 64, action_width: int = 12) -> None:
        super().__init__()
        self.hidden = hidden
        self.vision = VisionEncoder(hidden)
        self.action_embedding = nn.Embedding(3, action_width)
        self.recurrent = nn.GRU(
            hidden + action_width, hidden, batch_first=True)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden + action_width),
            nn.Linear(hidden + action_width, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden))

    def states(
            self, frames: torch.Tensor, previous_actions: torch.Tensor, *,
            passive: bool) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        visual = self.vision(frames.flatten(0, 1)).reshape(
            batch, steps, self.hidden)
        action = self.action_embedding(previous_actions)
        if passive:
            action = torch.zeros_like(action)
        return self.recurrent(torch.cat([visual, action], dim=-1))[0]

    def predict(
            self, state: torch.Tensor, action: torch.Tensor, *,
            passive: bool) -> torch.Tensor:
        embedded = self.action_embedding(action)
        if passive:
            embedded = torch.zeros_like(embedded)
        return self.predictor(torch.cat([state, embedded], dim=-1))


def pretrain_core(
        core: ActionHistoryCore, data: dict[str, torch.Tensor], *,
        mode: str, steps: int, batch_size: int, learning_rate: float,
        seed: int, device: torch.device) -> dict[str, object]:
    if mode not in ("action_conditioned", "passive", "shuffled_action"):
        raise ValueError(mode)
    target = copy.deepcopy(core.vision).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        core.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 31)
    frames = data["frames"]
    transitions = data["transition_actions"]
    previous = data["previous_actions"]
    history = []
    core.train()
    for step in range(1, steps + 1):
        indices = torch.randint(
            frames.shape[0], (min(batch_size, frames.shape[0]),),
            generator=generator, device=device)
        batch_frames = frames[indices.cpu()].to(device)
        batch_actions = transitions[indices.cpu()].to(device)
        batch_previous = previous[indices.cpu()].to(device)
        if mode == "shuffled_action":
            batch_actions = batch_actions.roll(1, dims=0)
            batch_previous = batch_previous.clone()
            batch_previous[:, 1] = batch_actions[:, 0]
        states = core.states(
            batch_frames[:, :3], batch_previous,
            passive=(mode == "passive"))
        # Only score controllable transitions.  The reset/target-appearance
        # transition is exogenous and intentionally excluded.
        selected_states = torch.cat([states[:, 0], states[:, 2]], dim=0)
        selected_actions = torch.cat([
            batch_actions[:, 0], batch_actions[:, 2]], dim=0)
        prediction = core.predict(
            selected_states, selected_actions,
            passive=(mode == "passive"))
        with torch.no_grad():
            encoded = target(batch_frames.flatten(0, 1)).reshape(
                batch_frames.shape[0], 4, -1)
            desired = torch.cat([
                encoded[:, 1] - encoded[:, 0],
                encoded[:, 3] - encoded[:, 2],
            ], dim=0)
        predictive = _standardized_prediction_loss(prediction, desired)
        variance = (
            _variance_loss(selected_states) + _variance_loss(prediction))
        correlation = _correlation_loss(selected_states)
        loss = predictive + 2.0 * variance + 0.5 * correlation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(
            core.parameters(), 1.0))
        optimizer.step()
        _ema_update(target, core.vision, 0.99)
        if step in (1, max(1, steps // 2), steps):
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "predictive_loss": float(predictive.detach()),
                "gradient_norm": gradient,
            })
    return {
        "history": history,
        "optimizer_updates": steps,
        "examples_processed": (
            steps * min(batch_size, frames.shape[0]) * 2),
        "unique_lifetimes": frames.shape[0],
        "unique_scored_transitions": frames.shape[0] * 2,
    }


@torch.no_grad()
def decision_features(
        core: ActionHistoryCore, data: dict[str, torch.Tensor], *,
        passive: bool, device: torch.device) -> torch.Tensor:
    core.eval()
    frames = data["frames"][:, :3].to(device)
    previous = data["previous_actions"].to(device)
    state = core.states(frames, previous, passive=passive)[:, -1]
    consequences = [
        core.predict(
            state,
            torch.full(
                (state.shape[0],), action,
                device=device, dtype=torch.long),
            passive=passive)
        for action in range(2)
    ]
    return torch.cat([state, *consequences], dim=-1)


@torch.no_grad()
def predictive_metrics(
        core: ActionHistoryCore, data: dict[str, torch.Tensor], *,
        passive: bool, device: torch.device) -> dict[str, float]:
    frames = data["frames"].to(device)
    previous = data["previous_actions"].to(device)
    actions = data["transition_actions"].to(device)
    states = core.states(frames[:, :3], previous, passive=passive)
    state = states[:, 2]
    attempted = actions[:, 2]
    desired = core.vision(frames[:, 3]) - core.vision(frames[:, 2])
    normal = _standardized_prediction_loss(
        core.predict(state, attempted, passive=passive), desired)
    shuffled = _standardized_prediction_loss(
        core.predict(state, attempted.roll(1), passive=passive), desired)
    return {
        "heldout_standardized_loss": float(normal),
        "heldout_action_shuffled_loss": float(shuffled),
        "action_binding_loss_increase": float(shuffled - normal),
    }


def fit_readout(
        initial: dict[str, torch.Tensor], features: torch.Tensor,
        attempted: torch.Tensor, rewards: torch.Tensor, *,
        readout_kind: str,
        intention_width: int, updates: int, batch_size: int,
        learning_rate: float, seed: int) -> nn.Module:
    model = make_readout(
        readout_kind, features.shape[-1], intention_width).to(features.device)
    model.load_state_dict(initial)
    attempted = attempted.to(features.device)
    rewards = rewards.to(features.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=features.device).manual_seed(seed + 47)
    for _ in range(updates):
        indices = torch.randint(
            features.shape[0],
            (min(batch_size, features.shape[0]),),
            generator=generator, device=features.device)
        loss = selected_success_loss(
            model(features[indices]), attempted[indices], rewards[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate(
        model: nn.Module, features: torch.Tensor,
        correct: torch.Tensor) -> dict[str, object]:
    logits = model(features)
    predictions = logits.argmax(-1).cpu()
    policy = logits.softmax(-1)
    entropy = -(policy * policy.clamp_min(1e-8).log()).sum(-1)
    return {
        "verified_accuracy": float(
            (predictions == correct).float().mean()),
        "predictions": predictions,
        "mean_margin": float(
            logits.topk(2, dim=-1).values.diff(dim=-1).abs().mean()),
        "normalized_policy_entropy": float(
            entropy.mean() / math.log(2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument(
        "--readout-kind", choices=("bottleneck", "direct"),
        default="bottleneck")
    parser.add_argument("--pretrain-lifetimes", type=int, default=128)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--policy-lifetimes", type=int, default=128)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=68)
    parser.add_argument(
        "--incremental-readout", action="store_true",
        help="Warm-start each reward-bit prefix from the prior prefix.")
    parser.add_argument(
        "--curriculum-rung",
        choices=(
            "direct_target", "fixed_probe", "fixed_target",
            "probe_12_5", "probe_25", "random_probe"),
        default="random_probe")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument(
        "--reset-readout-on-transfer", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    behavior_fixed_protocol = (
        0 if args.curriculum_rung == "direct_target" else None)
    behavior_fixed_probe = (
        0 if args.curriculum_rung in ("direct_target", "fixed_probe")
        else None)
    behavior_probe_fraction = {
        "probe_12_5": 0.125,
        "probe_25": 0.25,
    }.get(args.curriculum_rung)
    behavior_fixed_target = (
        -1 if args.curriculum_rung == "fixed_target" else None)
    pretrain = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=behavior_fixed_probe,
        probe_action_one_fraction=behavior_probe_fraction,
        fixed_target_direction=behavior_fixed_target)
    no_effect = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=behavior_fixed_probe,
        probe_action_one_fraction=behavior_probe_fraction,
        fixed_target_direction=behavior_fixed_target,
        no_probe_effect=True)
    fixed_protocol = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False,
        fixed_protocol=0, fixed_probe_action=behavior_fixed_probe,
        probe_action_one_fraction=behavior_probe_fraction,
        fixed_target_direction=behavior_fixed_target)
    train = identify_batch(
        TRAIN_START, args.policy_lifetimes, heldout=False,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=behavior_fixed_probe,
        probe_action_one_fraction=behavior_probe_fraction,
        fixed_target_direction=behavior_fixed_target)
    heldout = identify_batch(
        TEST_START, args.test_lifetimes, heldout=True,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=(
            behavior_fixed_probe
            if behavior_probe_fraction is None else None),
        fixed_target_direction=behavior_fixed_target)
    protocol_swap = identify_batch(
        TEST_START, args.test_lifetimes, heldout=True,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=(
            behavior_fixed_probe
            if behavior_probe_fraction is None else None),
        fixed_target_direction=behavior_fixed_target,
        swap_protocol=True)
    target_reverse = identify_batch(
        TEST_START, args.test_lifetimes, heldout=True,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=(
            behavior_fixed_probe
            if behavior_probe_fraction is None else None),
        fixed_target_direction=behavior_fixed_target,
        reverse_target=True)
    missing = identify_batch(
        TEST_START, args.test_lifetimes, heldout=True,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=(
            behavior_fixed_probe
            if behavior_probe_fraction is None else None),
        fixed_target_direction=behavior_fixed_target,
        missing_consequence=True)
    no_probe = identify_batch(
        TEST_START, args.test_lifetimes, heldout=True,
        fixed_protocol=behavior_fixed_protocol,
        fixed_probe_action=(
            behavior_fixed_probe
            if behavior_probe_fraction is None else None),
        fixed_target_direction=behavior_fixed_target,
        no_probe_effect=True)

    experience = None
    if args.initialize_from is not None:
        experience = torch.load(
            args.initialize_from, map_location=device, weights_only=True)
    initial_core = ActionHistoryCore(args.hidden).to(device)
    core_state = copy.deepcopy(initial_core.state_dict())
    cores, pretraining = {}, {}
    for name, mode, data in (
            ("action_conditioned", "action_conditioned", pretrain),
            ("passive", "passive", pretrain),
            ("shuffled_action", "shuffled_action", pretrain),
            ("no_probe_effect", "action_conditioned", no_effect),
            ("fixed_protocol", "action_conditioned", fixed_protocol)):
        core = ActionHistoryCore(args.hidden).to(device)
        if name == "action_conditioned" and experience is not None:
            core.load_state_dict(experience["core"])
        else:
            core.load_state_dict(core_state)
        pretraining[name] = pretrain_core(
            core, data, mode=mode, steps=args.pretrain_steps,
            batch_size=args.batch_size,
            learning_rate=args.pretrain_learning_rate,
            seed=args.seed, device=device)
        cores[name] = core
    if experience is not None:
        fresh_candidate = ActionHistoryCore(args.hidden).to(device)
        fresh_candidate.load_state_dict(core_state)
        pretraining["action_conditioned_fresh"] = pretrain_core(
            fresh_candidate, pretrain, mode="action_conditioned",
            steps=args.pretrain_steps, batch_size=args.batch_size,
            learning_rate=args.pretrain_learning_rate,
            seed=args.seed, device=device)
        cores["action_conditioned_fresh"] = fresh_candidate
    fresh = ActionHistoryCore(args.hidden).to(device)
    fresh.load_state_dict(core_state)
    cores["fully_fresh"] = fresh
    passive_modes = {
        "action_conditioned": False,
        "passive": True,
        "shuffled_action": False,
        "no_probe_effect": False,
        "fixed_protocol": False,
        "fully_fresh": False,
    }
    if "action_conditioned_fresh" in cores:
        passive_modes["action_conditioned_fresh"] = False
    for name, core in cores.items():
        pretraining.setdefault(name, {})["heldout_predictive_metrics"] = (
            predictive_metrics(
                core, heldout, passive=passive_modes[name],
                device=device))
    train_features = {
        name: decision_features(
            core, train, passive=passive_modes[name], device=device)
        for name, core in cores.items()
    }
    heldout_features = {
        name: decision_features(
            core, heldout, passive=passive_modes[name], device=device)
        for name, core in cores.items()
    }
    action_audits = {
        "protocol_swap": decision_features(
            cores["action_conditioned"], protocol_swap,
            passive=False, device=device),
        "target_reverse": decision_features(
            cores["action_conditioned"], target_reverse,
            passive=False, device=device),
        "missing_consequence": decision_features(
            cores["action_conditioned"], missing,
            passive=False, device=device),
        "no_probe_effect": decision_features(
            cores["action_conditioned"], no_probe,
            passive=False, device=device),
    }
    order = torch.randperm(
        args.policy_lifetimes,
        generator=torch.Generator().manual_seed(args.seed + 59))
    train_features = {
        name: value[order.to(device)] for name, value in train_features.items()}
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]
    seed_everything(args.seed + 61)
    template = make_readout(
        args.readout_kind, args.hidden * 3,
        args.intention_width).to(device)
    initial_readout = copy.deepcopy(template.state_dict())
    experienced_readout = (
        experience["readout"]
        if experience is not None and not args.reset_readout_on_transfer
        else initial_readout)
    prefixes = [
        value for value in (8, 16, 32, 64, 128, 256, 512)
        if value <= args.policy_lifetimes]
    if prefixes[-1] != args.policy_lifetimes:
        prefixes.append(args.policy_lifetimes)
    sources = {
        **{name: name for name in cores},
        "action_shuffled_replay": "action_conditioned",
        "reward_shuffled_replay": "action_conditioned",
    }
    arms, final_models = {}, {}
    for arm, source in sources.items():
        curve = []
        cumulative_updates = 0
        cumulative_examples = 0
        continuing_state = None
        for prefix in prefixes:
            arm_actions = attempted[:prefix].clone()
            arm_rewards = rewards[:prefix].clone()
            permutation = torch.randperm(
                prefix, generator=torch.Generator().manual_seed(
                    args.seed + 71 + prefix))
            if arm == "action_shuffled_replay":
                arm_actions = arm_actions[permutation]
            elif arm == "reward_shuffled_replay":
                arm_rewards = arm_rewards[permutation]
            arm_initial = (
                experienced_readout
                if source == "action_conditioned"
                else initial_readout)
            if args.incremental_readout and continuing_state is not None:
                arm_initial = continuing_state
            model = fit_readout(
                arm_initial, train_features[source][:prefix],
                arm_actions, arm_rewards,
                readout_kind=args.readout_kind,
                intention_width=args.intention_width,
                updates=args.fit_updates, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + 80 + prefix)
            if args.incremental_readout:
                continuing_state = copy.deepcopy(model.state_dict())
            cumulative_updates += args.fit_updates
            cumulative_examples += (
                args.fit_updates * min(args.batch_size, prefix))
            result = evaluate(
                model, heldout_features[source],
                heldout["correct_actions"])
            curve.append({
                "unique_reward_bits": prefix,
                "unique_lifetimes": prefix,
                "optimizer_updates": (
                    cumulative_updates if args.incremental_readout
                    else args.fit_updates),
                "optimizer_updates_this_stage": args.fit_updates,
                "examples_processed": (
                    cumulative_examples if args.incremental_readout
                    else args.fit_updates * min(args.batch_size, prefix)),
                "verified_accuracy": result["verified_accuracy"],
            })
            if prefix == prefixes[-1]:
                final_models[arm] = model
        stable_bits = next((
            curve[index]["unique_reward_bits"]
            for index in range(len(curve))
            if all(
                later["verified_accuracy"] >= 0.75
                for later in curve[index:])
        ), None)
        arms[arm] = {
            "curve": curve,
            "aulc_above_chance": sum(
                max(0.0, point["verified_accuracy"] - 0.5)
                for point in curve) / len(curve),
            "final_accuracy": curve[-1]["verified_accuracy"],
            "bits_to_75": next((
                point["unique_reward_bits"] for point in curve
                if point["verified_accuracy"] >= 0.75), None),
            "stable_bits_to_75": stable_bits,
        }
        print(json.dumps({"arm": arm, **arms[arm]}, sort_keys=True),
              flush=True)

    candidate = final_models["action_conditioned"]
    normal = evaluate(
        candidate, heldout_features["action_conditioned"],
        heldout["correct_actions"])
    swapped = evaluate(
        candidate, action_audits["protocol_swap"],
        protocol_swap["correct_actions"])
    reversed_target = evaluate(
        candidate, action_audits["target_reverse"],
        target_reverse["correct_actions"])
    missing_result = evaluate(
        candidate, action_audits["missing_consequence"],
        missing["correct_actions"])
    no_probe_result = evaluate(
        candidate, action_audits["no_probe_effect"],
        no_probe["correct_actions"])
    audit = {
        "normal_accuracy": normal["verified_accuracy"],
        "normal_entropy": normal["normalized_policy_entropy"],
        "protocol_swap_accuracy": swapped["verified_accuracy"],
        "protocol_swap_prediction_flip": float(
            (normal["predictions"] !=
             swapped["predictions"]).float().mean()),
        "target_reverse_accuracy": reversed_target["verified_accuracy"],
        "target_reverse_prediction_flip": float(
            (normal["predictions"] !=
             reversed_target["predictions"]).float().mean()),
        "missing_consequence_accuracy": (
            missing_result["verified_accuracy"]),
        "missing_consequence_entropy": (
            missing_result["normalized_policy_entropy"]),
        "no_probe_effect_accuracy": no_probe_result["verified_accuracy"],
    }
    if args.curriculum_rung in (
            "fixed_target", "probe_12_5", "probe_25", "random_probe"):
        control_names = [
            "passive", "shuffled_action", "no_probe_effect",
            "fixed_protocol", "fully_fresh"]
        if "action_conditioned_fresh" in arms:
            control_names.append("action_conditioned_fresh")
    elif args.curriculum_rung == "fixed_probe":
        # A fixed probe makes previous-action identity redundant by design.
        # Passive perception is therefore a legitimate solution at this rung.
        control_names = ["no_probe_effect", "fully_fresh"]
        if "action_conditioned_fresh" in arms:
            control_names.append("action_conditioned_fresh")
    else:
        # Direct target selection is a plumbing/mastery rung; the consequence
        # and protocol are deliberately unnecessary.
        control_names = ["fully_fresh"]
    controls = [arms[name] for name in control_names]
    best_control = max(
        float(value["aulc_above_chance"]) for value in controls)
    bits = arms["action_conditioned"]["stable_bits_to_75"]
    fewer = bits is not None and all(
        value["stable_bits_to_75"] is None or
        bits < value["stable_bits_to_75"]
        for value in controls)
    protocol_required = args.curriculum_rung != "direct_target"
    target_reversal_required = args.curriculum_rung != "fixed_target"
    gate = {
        "aulc_advantage": (
            float(arms["action_conditioned"]["aulc_above_chance"]) -
            best_control),
        "fewer_bits_to_75": fewer,
        "causal_audits_pass": bool(
            audit["normal_accuracy"] >= 0.75 and
            (not protocol_required or (
                audit["protocol_swap_accuracy"] >= 0.75 and
                audit["protocol_swap_prediction_flip"] >= 0.75)) and
            (not target_reversal_required or (
                audit["target_reverse_accuracy"] >= 0.75 and
                audit["target_reverse_prediction_flip"] >= 0.75)) and
            (not protocol_required or (
                audit["missing_consequence_accuracy"] <= 0.60 and
                audit["missing_consequence_entropy"] >
                audit["normal_entropy"] and
                audit["no_probe_effect_accuracy"] <= 0.60)) and
            arms["action_shuffled_replay"]["final_accuracy"] <= 0.60 and
            arms["reward_shuffled_replay"]["final_accuracy"] <= 0.60),
    }
    gate["advance_to_second_seed"] = bool(
        gate["aulc_advantage"] >= 0.03 and
        gate["fewer_bits_to_75"] and gate["causal_audits_pass"])
    report = {
        "schema": "identify-then-act-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb", "own_previous_opaque_probe_action",
            "own_attempted_second_action", "scalar_terminal_success"],
        "verifier_private": [
            "actuator_protocol", "effect_directions",
            "target_direction", "correct_second_action"],
        "configuration": vars(args) | {
            "report": str(args.report),
            "initialize_from": (
                str(args.initialize_from)
                if args.initialize_from is not None else None),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
        },
        "pretraining": pretraining,
        "arms": arms,
        "causal_audit": audit,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "identify-then-act-checkpoint-v1",
            "curriculum_rung": args.curriculum_rung,
            "core": {
                key: value.detach().cpu()
                for key, value in
                cores["action_conditioned"].state_dict().items()},
            "readout": {
                key: value.detach().cpu()
                for key, value in
                final_models["action_conditioned"].state_dict().items()},
        }, args.checkpoint_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "causal_audit": audit,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
