"""Six-decision pixels-only closed-loop intercept admission experiment.

The learner receives RGB frames, its own opaque attempted commands, and scalar
outcomes.  Coordinates, velocities, actuator effects, and counterfactual
actions remain verifier-private.
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

from experiments.syllogimous_latent_agent.data import IMAGE_HEIGHT, IMAGE_WIDTH

from .environment import COLORS
from .train import seed_everything
from .train_action_conditioned_success import selected_success_loss
from .train_actuator_transfer import SuccessSystem
from .train_micro_intercept import (
    EFFECTS,
    InterceptPredictiveCore,
    _ranked_balanced,
    protocol_for_seed,
)
from .train_zero_label_predictive_state import (
    _correlation_loss,
    _ema_update,
    _standardized_prediction_loss,
    _variance_loss,
)


PRETRAIN_START = 91_000_000
TRAIN_START = 93_000_000
TEST_START = 95_000_000
STEP_PIXELS = 8
TARGET_STEP_PIXELS = 14
CURSOR_MAX_SPEED = 2


def _bounce(position: int, velocity: int, low: int = 18,
            high: int = IMAGE_WIDTH - 19) -> tuple[int, int]:
    position += velocity * TARGET_STEP_PIXELS
    if position < low:
        position = low + (low - position)
        velocity = -velocity
    elif position > high:
        position = high - (position - high)
        velocity = -velocity
    return position, velocity


def _render(seed: int, frame_index: int, *, target_x: int,
            cursor_x: int, target_color: int, cursor_color: int,
            target_visible: bool) -> np.ndarray:
    """Render from private state; only the returned pixels reach the learner."""
    rng = np.random.default_rng(seed)
    background = tuple(int(value) for value in rng.integers(5, 22, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
        outline=(70, 82, 105), width=2)
    # Static nuisance objects vary by lifetime but not by action or outcome.
    for nuisance in range(3):
        x = int(rng.integers(18, IMAGE_WIDTH - 18))
        y = 44 + nuisance * 5
        color = tuple(int(value) for value in rng.integers(30, 75, size=3))
        draw.rectangle((x - 2, y - 2, x + 2, y + 2), fill=color)
    if target_visible:
        draw.ellipse(
            (target_x - 7, 18, target_x + 7, 32),
            fill=COLORS[target_color], outline=(245, 245, 250), width=2)
    else:
        # Generic occlusion bar reveals that evidence is unavailable without
        # encoding where the hidden target is.
        draw.rounded_rectangle(
            (12, 17, IMAGE_WIDTH - 13, 33), radius=4,
            fill=(28, 31, 42), outline=(62, 68, 86), width=1)
    draw.rounded_rectangle(
        (cursor_x - 9, 68, cursor_x + 9, 78), radius=3,
        fill=COLORS[cursor_color], outline=(245, 245, 250), width=2)
    # A non-semantic pulse makes frame time observable but not the task state.
    pulse = 25 + (frame_index % 3) * 12
    draw.rectangle((7, 85, 11, 89), fill=(pulse, pulse, pulse))
    return np.asarray(image, dtype=np.uint8).copy()


def trajectory_batch(
        start: int, count: int, *, heldout: bool,
        protocol: tuple[int, int, int], horizon: int = 6,
        fixed_stay: bool = False, reverse_motion: bool = False,
        missing_motion: bool = False, no_effect: bool = False,
        ) -> dict[str, torch.Tensor]:
    if count % 6:
        raise ValueError("trajectory count must be divisible by six")
    seeds = list(range(start, start + count))
    direction_ids = _ranked_balanced(
        seeds, heldout, "closed-direction", classes=2)
    target_positions = _ranked_balanced(
        seeds, heldout, "closed-target-position", classes=6)
    cursor_positions = _ranked_balanced(
        seeds, heldout, "closed-cursor-position", classes=3)
    target_colors = _ranked_balanced(
        seeds, heldout, "closed-target-color", classes=3)
    commands_by_step = [
        _ranked_balanced(
            seeds, heldout, f"closed-command-{step}", classes=3)
        for step in range(horizon)
    ]
    stay = protocol.index(0)
    all_frames, all_actions, all_rewards, all_private_states = [], [], [], []
    terminal_success, initial_directions = [], []
    for index, seed in enumerate(seeds):
        target_velocity = (-1, 1)[direction_ids[index]]
        if reverse_motion:
            target_velocity = -target_velocity
        target_x = 45 + target_positions[index] * 14
        cursor_x = 62 + cursor_positions[index] * 18
        cursor_velocity = 0
        target_color = target_colors[index]
        cursor_color = (target_color + 1) % len(COLORS)
        frames = [_render(
            seed * 13 + (7 if heldout else 0), 0,
            target_x=target_x, cursor_x=cursor_x,
            target_color=target_color, cursor_color=cursor_color,
            target_visible=not missing_motion)]
        actions, rewards = [], []
        private_states = []
        for step in range(horizon):
            private_states.append(
                (target_x, target_velocity, cursor_x, cursor_velocity))
            command = stay if fixed_stay else commands_by_step[step][index]
            old_distance = abs(target_x - cursor_x)
            effect = 0 if no_effect else protocol[command]
            cursor_velocity = max(
                -CURSOR_MAX_SPEED,
                min(CURSOR_MAX_SPEED, cursor_velocity + effect))
            cursor_x = max(
                18, min(IMAGE_WIDTH - 19,
                        cursor_x + cursor_velocity * STEP_PIXELS))
            target_x, target_velocity = _bounce(
                target_x, target_velocity)
            new_distance = abs(target_x - cursor_x)
            # This is the sole dense verifier outcome: whether the command the
            # agent actually attempted improved interception.  No unattempted
            # command is evaluated for learning.
            rewards.append(float(new_distance < old_distance))
            actions.append(command)
            visible = (
                (step + 1) <= 1 or (step + 1) == horizon)
            if missing_motion:
                visible = (step + 1) == horizon
            frames.append(_render(
                seed * 13 + (7 if heldout else 0), step + 1,
                target_x=target_x, cursor_x=cursor_x,
                target_color=target_color, cursor_color=cursor_color,
                target_visible=visible))
        all_frames.append(np.stack(frames))
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_private_states.append(private_states)
        terminal_success.append(float(abs(target_x - cursor_x) <= 14))
        initial_directions.append(target_velocity)
    frames_np = np.stack(all_frames)
    frames_tensor = torch.from_numpy(frames_np).permute(
        0, 1, 4, 2, 3).float().div_(255.0)
    return {
        "frames": frames_tensor,
        "actions": torch.tensor(all_actions, dtype=torch.long),
        "rewards": torch.tensor(all_rewards, dtype=torch.float32),
        "terminal_success": torch.tensor(
            terminal_success, dtype=torch.float32),
        # Verifier-private and diagnostic-only.  Deployed training code never
        # reads this tensor.
        "private_states": torch.tensor(
            all_private_states, dtype=torch.long),
        "private_final_directions": torch.tensor(
            initial_directions, dtype=torch.long),
    }


def pretrain_core(
        core: InterceptPredictiveCore, data: dict[str, torch.Tensor], *,
        mode: str, steps: int, batch_size: int, learning_rate: float,
        seed: int, device: torch.device) -> dict[str, object]:
    if mode not in ("action_conditioned", "passive", "shuffled_action"):
        raise ValueError(mode)
    target = copy.deepcopy(core.vision).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        core.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 101)
    history = []
    frames, actions = data["frames"], data["actions"]
    core.train()
    for step in range(1, steps + 1):
        indices = torch.randint(
            frames.shape[0], (min(batch_size, frames.shape[0]),),
            generator=generator, device=device)
        batch = frames[indices.cpu()].to(device)
        attempted = actions[indices.cpu()].to(device)
        states = core.states(batch[:, :-1])
        flat_states = states.flatten(0, 1)
        flat_actions = attempted.flatten()
        if mode == "shuffled_action":
            flat_actions = flat_actions.roll(1)
        prediction = core.predict(
            flat_states, flat_actions, passive=(mode == "passive"))
        with torch.no_grad():
            encoded = target(batch.flatten(0, 1)).reshape(
                batch.shape[0], batch.shape[1], -1)
            desired = (encoded[:, 1:] - encoded[:, :-1]).flatten(0, 1)
        predictive = _standardized_prediction_loss(prediction, desired)
        variance = (
            _variance_loss(flat_states) + _variance_loss(prediction))
        correlation = _correlation_loss(flat_states)
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
        "transition_examples_processed": (
            steps * min(batch_size, frames.shape[0]) *
            actions.shape[1]),
        "unique_trajectories": frames.shape[0],
        "unique_transitions": frames.shape[0] * actions.shape[1],
    }


@torch.no_grad()
def decision_features(
        core: InterceptPredictiveCore, frames: torch.Tensor, *,
        passive: bool, device: torch.device) -> torch.Tensor:
    core.eval()
    states = core.states(frames.to(device))
    flat = states.flatten(0, 1)
    consequences = [
        core.predict(
            flat,
            torch.full(
                (flat.shape[0],), action, device=device, dtype=torch.long),
            passive=passive)
        for action in range(3)
    ]
    return torch.cat([flat, *consequences], dim=-1).reshape(
        frames.shape[0], frames.shape[1], -1)


@torch.no_grad()
def predictive_metrics(
        core: InterceptPredictiveCore, data: dict[str, torch.Tensor], *,
        passive: bool, device: torch.device) -> dict[str, float]:
    frames = data["frames"].to(device)
    actions = data["actions"].to(device)
    states = core.states(frames[:, :-1]).flatten(0, 1)
    actions_flat = actions.flatten()
    desired = (
        core.vision(frames[:, 1:].flatten(0, 1)) -
        core.vision(frames[:, :-1].flatten(0, 1)))
    normal = _standardized_prediction_loss(
        core.predict(states, actions_flat, passive=passive), desired)
    shuffled = _standardized_prediction_loss(
        core.predict(states, actions_flat.roll(1), passive=passive), desired)
    return {
        "heldout_standardized_loss": float(normal),
        "heldout_action_shuffled_loss": float(shuffled),
        "action_binding_loss_increase": float(shuffled - normal),
    }


def fit_readout(
        initial: dict[str, torch.Tensor], features: torch.Tensor,
        actions: torch.Tensor, rewards: torch.Tensor, *,
        intention_width: int, updates: int, batch_size: int,
        learning_rate: float, seed: int) -> SuccessSystem:
    flat_features = features.flatten(0, 1)
    flat_actions = actions.flatten().to(features.device)
    flat_rewards = rewards.flatten().to(features.device)
    model = SuccessSystem(
        flat_features.shape[-1], intention_width, actions=3).to(
            features.device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=features.device).manual_seed(seed + 113)
    model.train()
    for _ in range(updates):
        indices = torch.randint(
            flat_features.shape[0],
            (min(batch_size, flat_features.shape[0]),),
            generator=generator, device=features.device)
        loss = selected_success_loss(
            model(flat_features[indices]),
            flat_actions[indices], flat_rewards[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _initial_private_state(
        start: int, count: int, heldout: bool, *,
        reverse_motion: bool) -> dict[str, list[int]]:
    seeds = list(range(start, start + count))
    directions = _ranked_balanced(
        seeds, heldout, "closed-direction", classes=2)
    target_positions = _ranked_balanced(
        seeds, heldout, "closed-target-position", classes=6)
    cursor_positions = _ranked_balanced(
        seeds, heldout, "closed-cursor-position", classes=3)
    colors = _ranked_balanced(
        seeds, heldout, "closed-target-color", classes=3)
    velocity = [(-1, 1)[value] for value in directions]
    if reverse_motion:
        velocity = [-value for value in velocity]
    return {
        "seeds": seeds,
        "target_x": [45 + value * 14 for value in target_positions],
        "target_v": velocity,
        "cursor_x": [62 + value * 18 for value in cursor_positions],
        "cursor_v": [0] * count,
        "colors": colors,
    }


def _frames_tensor(frame_histories: list[list[np.ndarray]]) -> torch.Tensor:
    array = np.stack([np.stack(history) for history in frame_histories])
    return torch.from_numpy(array).permute(
        0, 1, 4, 2, 3).float().div_(255.0)


@torch.no_grad()
def execute_policy(
        core: InterceptPredictiveCore, model: SuccessSystem, *,
        start: int, count: int, protocol: tuple[int, int, int],
        horizon: int, passive: bool, device: torch.device,
        reverse_motion: bool = False, missing_motion: bool = False,
        no_effect: bool = False,
        return_diagnostic_data: bool = False) -> dict[str, object]:
    private = _initial_private_state(
        start, count, True, reverse_motion=reverse_motion)
    histories = []
    for index in range(count):
        histories.append([_render(
            private["seeds"][index] * 13 + 7, 0,
            target_x=private["target_x"][index],
            cursor_x=private["cursor_x"][index],
            target_color=private["colors"][index],
            cursor_color=(private["colors"][index] + 1) % len(COLORS),
            target_visible=not missing_motion)])
    action_history, rewards, diagnostic_private = [], [], []
    margins, entropies = [], []
    for step in range(horizon):
        if return_diagnostic_data:
            diagnostic_private.append(torch.tensor([
                (
                    private["target_x"][index],
                    private["target_v"][index],
                    private["cursor_x"][index],
                    private["cursor_v"][index],
                )
                for index in range(count)
            ], dtype=torch.long))
        frames = _frames_tensor(histories)
        feature = decision_features(
            core, frames, passive=passive, device=device)[:, -1]
        logits = model(feature)
        action = logits.argmax(-1).cpu()
        policy = logits.softmax(-1)
        margins.append(float(
            logits.topk(2, dim=-1).values.diff(dim=-1).abs().mean()))
        entropies.append(float(
            (-(policy * policy.clamp_min(1e-8).log()).sum(-1) /
             math.log(3.0)).mean()))
        action_history.append(action)
        step_rewards = []
        for index in range(count):
            old_distance = abs(
                private["target_x"][index] - private["cursor_x"][index])
            effect = 0 if no_effect else protocol[int(action[index])]
            private["cursor_v"][index] = max(
                -CURSOR_MAX_SPEED,
                min(CURSOR_MAX_SPEED,
                    private["cursor_v"][index] + effect))
            private["cursor_x"][index] = max(
                18, min(
                    IMAGE_WIDTH - 19,
                    private["cursor_x"][index] +
                    private["cursor_v"][index] * STEP_PIXELS))
            target_x, target_v = _bounce(
                private["target_x"][index],
                private["target_v"][index])
            private["target_x"][index] = target_x
            private["target_v"][index] = target_v
            new_distance = abs(target_x - private["cursor_x"][index])
            step_rewards.append(float(new_distance < old_distance))
            visible = (step + 1) <= 1 or (step + 1) == horizon
            if missing_motion:
                visible = (step + 1) == horizon
            histories[index].append(_render(
                private["seeds"][index] * 13 + 7, step + 1,
                target_x=target_x,
                cursor_x=private["cursor_x"][index],
                target_color=private["colors"][index],
                cursor_color=(
                    private["colors"][index] + 1) % len(COLORS),
                target_visible=visible))
        rewards.append(torch.tensor(step_rewards))
    actions = torch.stack(action_history, dim=1)
    reward_tensor = torch.stack(rewards, dim=1)
    terminal = torch.tensor([
        abs(private["target_x"][index] - private["cursor_x"][index]) <= 14
        for index in range(count)
    ])
    result = {
        "terminal_success": float(terminal.float().mean()),
        "mean_step_reward": float(reward_tensor.mean()),
        "actions": actions,
        "mean_margin": sum(margins) / len(margins),
        "normalized_policy_entropy": sum(entropies) / len(entropies),
    }
    if return_diagnostic_data:
        result["diagnostic_frames"] = _frames_tensor(histories)[:, :-1]
        result["diagnostic_private_states"] = torch.stack(
            diagnostic_private, dim=1)
    return result


def random_policy_baseline(
        start: int, count: int, *, protocol: tuple[int, int, int],
        horizon: int) -> float:
    private = _initial_private_state(
        start, count, True, reverse_motion=False)
    generator = torch.Generator().manual_seed(start + 127)
    actions = torch.randint(
        0, 3, (count, horizon), generator=generator)
    for step in range(horizon):
        for index in range(count):
            private["cursor_v"][index] = max(
                -CURSOR_MAX_SPEED,
                min(CURSOR_MAX_SPEED,
                    private["cursor_v"][index] +
                    protocol[int(actions[index, step])]))
            private["cursor_x"][index] = max(
                18, min(
                    IMAGE_WIDTH - 19,
                    private["cursor_x"][index] +
                    private["cursor_v"][index] * STEP_PIXELS))
            target_x, target_v = _bounce(
                private["target_x"][index],
                private["target_v"][index])
            private["target_x"][index] = target_x
            private["target_v"][index] = target_v
    return sum(
        abs(private["target_x"][index] - private["cursor_x"][index]) <= 14
        for index in range(count)) / count


def _aulc(curve: list[dict[str, object]], baseline: float) -> float:
    return sum(
        max(0.0, float(point["terminal_success"]) - baseline)
        for point in curve) / len(curve)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--pretrain-lifetimes", type=int, default=96)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--policy-lifetimes", type=int, default=90)
    parser.add_argument("--test-lifetimes", type=int, default=96)
    parser.add_argument("--fit-updates", type=int, default=68)
    parser.add_argument(
        "--readout-target",
        choices=("step_improvement", "terminal_return"),
        default="step_improvement")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    protocol = protocol_for_seed(args.seed)
    alternate_protocol = tuple(reversed(protocol))
    if alternate_protocol == protocol:
        alternate_protocol = protocol[1:] + protocol[:1]

    pretrain = trajectory_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False,
        protocol=protocol, horizon=args.horizon)
    fixed = trajectory_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False,
        protocol=protocol, horizon=args.horizon, fixed_stay=True)
    train = trajectory_batch(
        TRAIN_START, args.policy_lifetimes, heldout=False,
        protocol=protocol, horizon=args.horizon)
    heldout = trajectory_batch(
        TEST_START, args.test_lifetimes, heldout=True,
        protocol=protocol, horizon=args.horizon)

    initial_core = InterceptPredictiveCore(args.hidden).to(device)
    core_state = copy.deepcopy(initial_core.state_dict())
    cores, pretraining = {}, {}
    for name, mode, data in (
            ("action_conditioned", "action_conditioned", pretrain),
            ("passive", "passive", pretrain),
            ("shuffled_action", "shuffled_action", pretrain),
            ("fixed_no_action", "action_conditioned", fixed)):
        core = InterceptPredictiveCore(args.hidden).to(device)
        core.load_state_dict(core_state)
        pretraining[name] = pretrain_core(
            core, data, mode=mode, steps=args.pretrain_steps,
            batch_size=args.batch_size,
            learning_rate=args.pretrain_learning_rate,
            seed=args.seed, device=device)
        cores[name] = core
    fresh = InterceptPredictiveCore(args.hidden).to(device)
    fresh.load_state_dict(core_state)
    cores["fully_fresh"] = fresh
    passive_modes = {
        "action_conditioned": False,
        "passive": True,
        "shuffled_action": False,
        "fixed_no_action": False,
        "fully_fresh": False,
    }
    for name, core in cores.items():
        pretraining.setdefault(name, {})["heldout_predictive_metrics"] = (
            predictive_metrics(
                core, heldout, passive=passive_modes[name],
                device=device))

    train_features = {
        name: decision_features(
            core, train["frames"][:, :-1],
            passive=passive_modes[name], device=device)
        for name, core in cores.items()
    }
    if args.readout_target == "terminal_return":
        behavior_rewards = train["terminal_success"].unsqueeze(1).expand(
            -1, args.horizon).clone()
        reward_bits_per_trajectory = 1
    else:
        behavior_rewards = train["rewards"]
        reward_bits_per_trajectory = args.horizon
    seed_everything(args.seed + 201)
    template = SuccessSystem(
        args.hidden * 4, args.intention_width, actions=3).to(device)
    initial_readout = copy.deepcopy(template.state_dict())
    prefixes = [
        value for value in (6, 15, 30, 45, 90)
        if value <= args.policy_lifetimes]
    if prefixes[-1] != args.policy_lifetimes:
        prefixes.append(args.policy_lifetimes)
    baseline = random_policy_baseline(
        TEST_START, args.test_lifetimes, protocol=protocol,
        horizon=args.horizon)
    arms, final_models = {}, {}
    sources = {
        **{name: name for name in cores},
        "action_shuffled_replay": "action_conditioned",
        "reward_shuffled_replay": "action_conditioned",
    }
    for arm, source in sources.items():
        curve = []
        feature = train_features[source]
        actions = train["actions"]
        rewards = behavior_rewards
        for prefix in prefixes:
            prefix_features = feature[:prefix]
            prefix_actions = actions[:prefix].clone()
            prefix_rewards = rewards[:prefix].clone()
            flat_count = prefix * args.horizon
            prefix_permutation = torch.randperm(
                flat_count, generator=torch.Generator().manual_seed(
                    args.seed + 211 + prefix))
            if arm == "action_shuffled_replay":
                prefix_actions = prefix_actions.flatten()[
                    prefix_permutation].reshape(prefix, args.horizon)
            elif arm == "reward_shuffled_replay":
                prefix_rewards = prefix_rewards.flatten()[
                    prefix_permutation].reshape(prefix, args.horizon)
            model = fit_readout(
                initial_readout, prefix_features,
                prefix_actions, prefix_rewards,
                intention_width=args.intention_width,
                updates=args.fit_updates, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + prefix + 300)
            execution = execute_policy(
                cores[source], model, start=TEST_START,
                count=args.test_lifetimes, protocol=protocol,
                horizon=args.horizon, passive=passive_modes[source],
                device=device)
            curve.append({
                "unique_trajectories": prefix,
                "unique_reward_bits": (
                    prefix * reward_bits_per_trajectory),
                "optimizer_updates": args.fit_updates,
                "examples_processed": (
                    args.fit_updates * min(
                        args.batch_size, prefix * args.horizon)),
                "terminal_success": execution["terminal_success"],
                "mean_step_reward": execution["mean_step_reward"],
            })
            if prefix == prefixes[-1]:
                final_models[arm] = model
        arms[arm] = {
            "curve": curve,
            "aulc_above_random": _aulc(curve, baseline),
            "final_terminal_success": curve[-1]["terminal_success"],
            "bits_to_60": next((
                point["unique_reward_bits"] for point in curve
                if point["terminal_success"] >= 0.60), None),
        }
        print(json.dumps({"arm": arm, **arms[arm]}, sort_keys=True),
              flush=True)

    candidate = final_models["action_conditioned"]
    normal = execute_policy(
        cores["action_conditioned"], candidate, start=TEST_START,
        count=args.test_lifetimes, protocol=protocol,
        horizon=args.horizon, passive=False, device=device)
    reverse = execute_policy(
        cores["action_conditioned"], candidate, start=TEST_START,
        count=args.test_lifetimes, protocol=protocol,
        horizon=args.horizon, passive=False, device=device,
        reverse_motion=True)
    remapped = execute_policy(
        cores["action_conditioned"], candidate, start=TEST_START,
        count=args.test_lifetimes, protocol=alternate_protocol,
        horizon=args.horizon, passive=False, device=device)
    missing = execute_policy(
        cores["action_conditioned"], candidate, start=TEST_START,
        count=args.test_lifetimes, protocol=protocol,
        horizon=args.horizon, passive=False, device=device,
        missing_motion=True)
    no_effect = execute_policy(
        cores["action_conditioned"], candidate, start=TEST_START,
        count=args.test_lifetimes, protocol=protocol,
        horizon=args.horizon, passive=False, device=device,
        no_effect=True)
    audit = {
        "normal_terminal_success": normal["terminal_success"],
        "reverse_terminal_success": reverse["terminal_success"],
        "reverse_action_sequence_change": float(
            (normal["actions"] != reverse["actions"]).float().mean()),
        "remapped_protocol_terminal_success": remapped["terminal_success"],
        "missing_motion_terminal_success": missing["terminal_success"],
        "missing_motion_entropy": missing["normalized_policy_entropy"],
        "normal_entropy": normal["normalized_policy_entropy"],
        "no_effect_terminal_success": no_effect["terminal_success"],
    }
    candidate_arm = arms["action_conditioned"]
    controls = [
        arms["passive"], arms["shuffled_action"], arms["fixed_no_action"]]
    best_control = max(float(value["aulc_above_random"])
                       for value in controls)
    bits = candidate_arm["bits_to_60"]
    fewer = bits is not None and all(
        value["bits_to_60"] is None or bits < value["bits_to_60"]
        for value in controls)
    gate = {
        "random_policy_terminal_success": baseline,
        "aulc_advantage": (
            float(candidate_arm["aulc_above_random"]) - best_control),
        "fewer_bits_to_60": fewer,
        "causal_audits_pass": bool(
            audit["normal_terminal_success"] >= 0.60 and
            audit["reverse_terminal_success"] >= 0.55 and
            audit["reverse_action_sequence_change"] >= 0.35 and
            audit["remapped_protocol_terminal_success"] <=
            audit["normal_terminal_success"] - 0.15 and
            audit["no_effect_terminal_success"] <=
            audit["normal_terminal_success"] - 0.15 and
            audit["missing_motion_terminal_success"] <=
            audit["normal_terminal_success"] - 0.15 and
            audit["missing_motion_entropy"] >
            audit["normal_entropy"]),
    }
    gate["advance_to_three_minutes"] = bool(
        gate["aulc_advantage"] >= 0.03 and
        gate["fewer_bits_to_60"] and gate["causal_audits_pass"])
    report = {
        "schema": "closed-loop-micro-intercept-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb", "own_opaque_actions",
            "logging_propensities", "scalar_attempted_action_outcomes"],
        "verifier_private": [
            "positions", "velocities", "actuator_effects",
            "terminal_distance", "counterfactual_actions"],
        "configuration": vars(args) | {"report": str(args.report)},
        "private_protocol_effects": list(protocol),
        "pretraining": pretraining,
        "arms": arms,
        "causal_audit": audit,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "causal_audit": audit,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
