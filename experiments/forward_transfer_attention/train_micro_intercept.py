"""Pixels-only closed-loop micro-intercept predictive-state experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from .train_zero_label_predictive_state import (
    _correlation_loss,
    _ema_update,
    _standardized_prediction_loss,
    _variance_loss,
)


INTERCEPT_PRETRAIN_START = 81_000_000
INTERCEPT_TRAIN_START = 83_000_000
INTERCEPT_TEST_START = 85_000_000
MOVE_PIXELS = 18
EFFECTS = (-1, 0, 1)


def protocol_for_seed(seed: int) -> tuple[int, int, int]:
    order = torch.randperm(
        3, generator=torch.Generator().manual_seed(seed + 17)).tolist()
    return tuple(EFFECTS[index] for index in order)


def _ranked_balanced(seed_values: list[int], heldout: bool,
                     purpose: str, classes: int) -> list[int]:
    if len(seed_values) % classes:
        raise ValueError(f"count must be divisible by {classes}")
    ranked = sorted(
        range(len(seed_values)),
        key=lambda index: hashlib.blake2b(
            f"micro-intercept-v1:{seed_values[index]}:{int(heldout)}:"
            f"{purpose}".encode(), digest_size=16).digest())
    values = [0] * len(seed_values)
    block = len(seed_values) // classes
    for rank, index in enumerate(ranked):
        values[index] = min(classes - 1, rank // block)
    return values


def _render_frame(seed: int, *, target_x: int, cursor_x: int,
                  target_color: int, cursor_color: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    background = tuple(int(v) for v in rng.integers(5, 25, size=3))
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (3, 3, IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4), radius=6,
        outline=(75, 85, 110), width=2)
    draw.ellipse(
        (target_x - 7, 21, target_x + 7, 35),
        fill=COLORS[target_color], outline=(245, 245, 250), width=2)
    draw.rounded_rectangle(
        (cursor_x - 10, 67, cursor_x + 10, 77), radius=3,
        fill=COLORS[cursor_color], outline=(245, 245, 250), width=2)
    return np.asarray(image, dtype=np.uint8).copy()


def intercept_sequences(
        start: int, count: int, *, heldout: bool,
        protocol: tuple[int, int, int],
        fixed_stay: bool = False, reverse_velocity: bool = False,
        mirror: bool = False, omit_first: bool = False,
        omit_second: bool = False,
        ) -> dict[str, torch.Tensor]:
    if omit_first and omit_second:
        raise ValueError("cannot remove both velocity frames")
    seeds = list(range(start, start + count))
    velocity_ids = _ranked_balanced(
        seeds, heldout, "velocity", classes=3)
    logged_commands = _ranked_balanced(
        seeds, heldout, "logged-command", classes=3)
    position_ids = _ranked_balanced(
        seeds, heldout, "position", classes=6)
    target_colors = _ranked_balanced(
        seeds, heldout, "target-color", classes=3)
    frames, pre_frames = [], []
    commands, rewards, correct_commands, velocities = [], [], [], []
    stay_command = protocol.index(0)
    for index, seed in enumerate(seeds):
        velocity = EFFECTS[velocity_ids[index]]
        if reverse_velocity:
            velocity = -velocity
        x1 = 43 + position_ids[index] * 14
        x0 = x1 - velocity * MOVE_PIXELS
        x2 = x1 + velocity * MOVE_PIXELS
        command = stay_command if fixed_stay else logged_commands[index]
        effect = protocol[command]
        correct_velocity = velocity
        if mirror:
            # Mirror the sensory scene while keeping actuator effects in the
            # screen coordinate system.  Thus target motion reverses but the
            # same opaque command still moves the cursor the same screenward
            # direction, and the verifier's correct command must change.
            x0 = IMAGE_WIDTH - 1 - x0
            x1 = IMAGE_WIDTH - 1 - x1
            x2 = IMAGE_WIDTH - 1 - x2
            correct_velocity = -velocity
        cursor2 = x1 + effect * MOVE_PIXELS
        correct = protocol.index(correct_velocity)
        base = seed * 10 + (5 if heldout else 0)
        target_color = target_colors[index]
        cursor_color = (target_color + 1) % len(COLORS)
        first = _render_frame(
            base, target_x=x0, cursor_x=x1,
            target_color=target_color, cursor_color=cursor_color)
        second = _render_frame(
            base + 1, target_x=x1, cursor_x=x1,
            target_color=target_color, cursor_color=cursor_color)
        third = _render_frame(
            base + 2, target_x=x2, cursor_x=cursor2,
            target_color=target_color, cursor_color=cursor_color)
        visible_pre = []
        if not omit_first:
            visible_pre.append(first)
        if not omit_second:
            visible_pre.append(second)
        pre_frames.append(np.stack(visible_pre))
        frames.append(np.stack([first, second, third]))
        commands.append(command)
        rewards.append(float(command == correct))
        correct_commands.append(correct)
        velocities.append(velocity)
    to_tensor = lambda items: torch.from_numpy(
        np.stack(items)).permute(0, 1, 4, 2, 3).float().div_(255.0)
    return {
        "frames": to_tensor(frames),
        "pre_frames": to_tensor(pre_frames),
        "actions": torch.tensor(commands, dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "correct_actions": torch.tensor(correct_commands, dtype=torch.long),
        "velocities": torch.tensor(velocities, dtype=torch.long),
    }


class InterceptPredictiveCore(nn.Module):
    def __init__(self, hidden: int = 64, action_width: int = 12) -> None:
        super().__init__()
        self.hidden = hidden
        self.vision = VisionEncoder(hidden)
        self.recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.action_embedding = nn.Embedding(3, action_width)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden + action_width),
            nn.Linear(hidden + action_width, hidden * 2),
            nn.GELU(), nn.Linear(hidden * 2, hidden))

    def states(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        encoded = self.vision(frames.flatten(0, 1)).reshape(
            batch, steps, self.hidden)
        return self.recurrent(encoded)[0]

    def predict(self, state: torch.Tensor, actions: torch.Tensor, *,
                passive: bool) -> torch.Tensor:
        action = self.action_embedding(actions)
        if passive:
            action = torch.zeros_like(action)
        return self.predictor(torch.cat([state, action], dim=-1))


def pretrain_core(
        core: InterceptPredictiveCore, frames: torch.Tensor,
        actions: torch.Tensor, *, mode: str, steps: int,
        batch_size: int, learning_rate: float, seed: int,
        device: torch.device) -> dict[str, object]:
    if mode not in ("action_conditioned", "passive", "shuffled_action"):
        raise ValueError(f"unknown mode {mode!r}")
    target = copy.deepcopy(core.vision).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        core.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 23)
    history = []
    core.train()
    for step in range(1, steps + 1):
        indices = torch.randint(
            frames.shape[0], (min(batch_size, frames.shape[0]),),
            generator=generator, device=device)
        batch = frames[indices.cpu()].to(device)
        attempted = actions[indices.cpu()].to(device)
        if mode == "shuffled_action":
            attempted = attempted.roll(1)
        states = core.states(batch[:, :2])
        prediction = core.predict(
            states[:, -1], attempted, passive=(mode == "passive"))
        with torch.no_grad():
            encoded = target(batch[:, 1:].flatten(0, 1)).reshape(
                batch.shape[0], 2, -1)
            desired = encoded[:, 1] - encoded[:, 0]
        predictive = _standardized_prediction_loss(prediction, desired)
        variance = _variance_loss(states[:, -1]) + _variance_loss(prediction)
        correlation = _correlation_loss(states[:, -1])
        loss = predictive + 2.0 * variance + 0.5 * correlation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(
            core.parameters(), 1.0))
        optimizer.step()
        _ema_update(target, core.vision, 0.99)
        if step in (1, max(1, steps // 2), steps):
            history.append({
                "step": step, "loss": float(loss.detach()),
                "predictive_loss": float(predictive.detach()),
                "gradient_norm": gradient,
            })
    return {
        "history": history,
        "optimizer_updates": steps,
        "examples_processed": steps * min(batch_size, frames.shape[0]),
        "unique_transition_lifetimes": frames.shape[0],
    }


@torch.no_grad()
def frozen_decision_features(
        core: InterceptPredictiveCore, frames: torch.Tensor,
        batch_size: int, device: torch.device, *,
        passive: bool) -> torch.Tensor:
    """Expose state plus learned consequences of each opaque command.

    These are model-generated latent predictions, not verifier labels.  The
    downstream learner still receives reward for only its attempted command.
    """
    core.eval()
    outputs = []
    for offset in range(0, frames.shape[0], batch_size):
        batch = frames[offset:offset + batch_size].to(device)
        state = core.states(batch)[:, -1]
        consequences = [
            core.predict(
                state,
                torch.full(
                    (state.shape[0],), action,
                    device=device, dtype=torch.long),
                passive=passive)
            for action in range(3)
        ]
        outputs.append(torch.cat([state, *consequences], dim=-1))
    return torch.cat(outputs)


@torch.no_grad()
def predictive_metrics(
        core: InterceptPredictiveCore, frames: torch.Tensor,
        actions: torch.Tensor, *, passive: bool,
        batch_size: int, device: torch.device) -> dict[str, float]:
    core.eval()
    normal, shuffled = [], []
    permutation = torch.randperm(
        actions.shape[0], generator=torch.Generator().manual_seed(914))
    shuffled_actions = actions[permutation]
    for offset in range(0, frames.shape[0], batch_size):
        batch = frames[offset:offset + batch_size].to(device)
        state = core.states(batch[:, :2])[:, -1]
        desired = (
            core.vision(batch[:, 2]) - core.vision(batch[:, 1]))
        normal.append(float(_standardized_prediction_loss(
            core.predict(
                state, actions[offset:offset + batch_size].to(device),
                passive=passive),
            desired)))
        shuffled.append(float(_standardized_prediction_loss(
            core.predict(
                state, shuffled_actions[offset:offset + batch_size].to(device),
                passive=passive),
            desired)))
    return {
        "heldout_standardized_loss": sum(normal) / len(normal),
        "heldout_action_shuffled_loss": sum(shuffled) / len(shuffled),
        "action_binding_loss_increase": (
            sum(shuffled) - sum(normal)) / len(normal),
    }


def uniform_logged_buffer(states: torch.Tensor,
                          correct_actions: torch.Tensor, *,
                          seed: int) -> tuple[torch.Tensor, ...]:
    count = states.shape[0]
    order = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed + 31))
    states = states[order.to(states.device)]
    correct = correct_actions[order]
    attempted = (torch.arange(count) % 3).to(states.device)
    rewards = (attempted.cpu() == correct).to(states.device, states.dtype)
    propensities = torch.full(
        (count,), 1 / 3, device=states.device, dtype=states.dtype)
    return states, correct, attempted, rewards, propensities


def fit_head(initial, states, actions, rewards, args, seed):
    model = SuccessSystem(
        states.shape[-1], args.intention_width, actions=3).to(states.device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=states.device).manual_seed(seed + 41)
    for _ in range(args.fit_updates):
        indices = torch.randint(
            states.shape[0], (min(args.batch_size, states.shape[0]),),
            generator=generator, device=states.device)
        loss = selected_success_loss(
            model(states[indices]), actions[indices], rewards[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


@torch.no_grad()
def evaluate_metrics(model, states, correct) -> dict[str, float]:
    logits = model(states)
    predictions = logits.argmax(-1).cpu()
    probabilities = logits.sigmoid().cpu()
    targets = torch.nn.functional.one_hot(
        correct, num_classes=3).to(probabilities.dtype)
    policy = logits.softmax(-1)
    entropy = -(policy * policy.clamp_min(1e-8).log()).sum(-1)
    return {
        "verified_accuracy": float((predictions == correct).float().mean()),
        "brier": float((probabilities - targets).square().mean()),
        "mean_margin": float(
            logits.topk(2, dim=-1).values.diff(dim=-1).abs().mean()),
        "normalized_policy_entropy": float(
            entropy.mean() / np.log(3.0)),
    }


def evaluate(model, states, correct) -> float:
    return evaluate_metrics(model, states, correct)["verified_accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--policy-lifetimes", type=int, default=510)
    parser.add_argument("--test-lifetimes", type=int, default=384)
    parser.add_argument("--fit-updates", type=int, default=200)
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
    pretrain_data = intercept_sequences(
        INTERCEPT_PRETRAIN_START, args.pretrain_lifetimes,
        heldout=False, protocol=protocol)
    fixed_data = intercept_sequences(
        INTERCEPT_PRETRAIN_START, args.pretrain_lifetimes,
        heldout=False, protocol=protocol, fixed_stay=True)
    train_data = intercept_sequences(
        INTERCEPT_TRAIN_START, args.policy_lifetimes,
        heldout=False, protocol=protocol)
    test_data = intercept_sequences(
        INTERCEPT_TEST_START, args.test_lifetimes,
        heldout=True, protocol=protocol)
    reverse_data = intercept_sequences(
        INTERCEPT_TEST_START, args.test_lifetimes,
        heldout=True, protocol=protocol, reverse_velocity=True)
    mirror_data = intercept_sequences(
        INTERCEPT_TEST_START, args.test_lifetimes,
        heldout=True, protocol=protocol, mirror=True)
    missing_first = intercept_sequences(
        INTERCEPT_TEST_START, args.test_lifetimes,
        heldout=True, protocol=protocol, omit_first=True)
    missing_second = intercept_sequences(
        INTERCEPT_TEST_START, args.test_lifetimes,
        heldout=True, protocol=protocol, omit_second=True)

    seed_everything(args.seed)
    initial = InterceptPredictiveCore(args.hidden).to(device)
    initial_state = copy.deepcopy(initial.state_dict())
    cores, pretraining = {}, {}
    for name, mode, data in (
            ("action_conditioned", "action_conditioned", pretrain_data),
            ("passive", "passive", pretrain_data),
            ("shuffled_action", "shuffled_action", pretrain_data),
            ("fixed_no_action", "action_conditioned", fixed_data)):
        core = InterceptPredictiveCore(args.hidden).to(device)
        core.load_state_dict(initial_state)
        pretraining[name] = pretrain_core(
            core, data["frames"], data["actions"], mode=mode,
            steps=args.pretrain_steps, batch_size=args.batch_size,
            learning_rate=args.pretrain_learning_rate,
            seed=args.seed, device=device)
        cores[name] = core
    fresh = InterceptPredictiveCore(args.hidden).to(device)
    fresh.load_state_dict(initial_state)
    cores["fully_fresh"] = fresh

    passive_modes = {
        "action_conditioned": False,
        "passive": True,
        "shuffled_action": False,
        "fixed_no_action": False,
        "fully_fresh": False,
    }
    train_states = {
        name: frozen_decision_features(
            core, train_data["pre_frames"], args.batch_size, device,
            passive=passive_modes[name])
        for name, core in cores.items()}
    test_states = {
        name: frozen_decision_features(
            core, test_data["pre_frames"], args.batch_size, device,
            passive=passive_modes[name])
        for name, core in cores.items()}
    action_reverse_states = frozen_decision_features(
        cores["action_conditioned"], reverse_data["pre_frames"],
        args.batch_size, device, passive=False)
    action_mirror_states = frozen_decision_features(
        cores["action_conditioned"], mirror_data["pre_frames"],
        args.batch_size, device, passive=False)
    action_missing_first = frozen_decision_features(
        cores["action_conditioned"], missing_first["pre_frames"],
        args.batch_size, device, passive=False)
    action_missing_second = frozen_decision_features(
        cores["action_conditioned"], missing_second["pre_frames"],
        args.batch_size, device, passive=False)

    for name, core in cores.items():
        pretraining.setdefault(name, {})["heldout_predictive_metrics"] = (
            predictive_metrics(
                core, test_data["frames"], test_data["actions"],
                passive=passive_modes[name],
                batch_size=args.batch_size, device=device))

    logged = {
        name: uniform_logged_buffer(
            states, train_data["correct_actions"], seed=args.seed + 300)
        for name, states in train_states.items()}
    reference = tuple(item.cpu() for item in logged[
        "action_conditioned"][1:4])
    for output in logged.values():
        assert all(torch.equal(left, right) for left, right in zip(
            reference, (item.cpu() for item in output[1:4])))
    seed_everything(args.seed + 400)
    template = SuccessSystem(
        args.hidden * 4, args.intention_width, actions=3).to(device)
    initial_head = copy.deepcopy(template.state_dict())
    permutation = torch.randperm(
        args.policy_lifetimes,
        generator=torch.Generator().manual_seed(args.seed + 401))
    arms, final_models = {}, {}
    prefixes = [
        value for value in (30, 90, 180, 270, 510)
        if value <= args.policy_lifetimes]
    if prefixes[-1] != args.policy_lifetimes:
        prefixes.append(args.policy_lifetimes)
    sources = {
        **{name: name for name in cores},
        "action_shuffled_replay": "action_conditioned",
        "reward_shuffled_replay": "action_conditioned",
    }
    for arm, source in sources.items():
        curve = []
        states, _, actions, rewards, _ = logged[source]
        for prefix in prefixes:
            arm_actions = actions[:prefix]
            arm_rewards = rewards[:prefix]
            if arm == "action_shuffled_replay":
                arm_actions = actions[permutation.to(device)][:prefix]
            elif arm == "reward_shuffled_replay":
                arm_rewards = rewards[permutation.to(device)][:prefix]
            model = fit_head(
                initial_head, states[:prefix], arm_actions, arm_rewards,
                args, args.seed + 500 + prefix)
            accuracy = evaluate(
                model, test_states[source],
                test_data["correct_actions"])
            curve.append({
                "unique_reward_bits": prefix,
                "optimizer_updates": args.fit_updates,
                "examples_processed": args.fit_updates * min(
                    args.batch_size, prefix),
                "verified_accuracy": accuracy,
            })
            if prefix == prefixes[-1]:
                final_models[arm] = model
        arms[arm] = {
            "curve": curve,
            "aulc_above_majority": sum(max(
                0.0, point["verified_accuracy"] - 1 / 3)
                for point in curve) / len(curve),
            "final_accuracy": curve[-1]["verified_accuracy"],
            "bits_to_60": next((
                point["unique_reward_bits"] for point in curve
                if point["verified_accuracy"] >= 0.60), None),
        }
        print(json.dumps({"arm": arm, **arms[arm]}, sort_keys=True),
              flush=True)

    candidate = final_models["action_conditioned"]
    normal_predictions = candidate(
        test_states["action_conditioned"]).argmax(-1).cpu()
    reverse_predictions = candidate(action_reverse_states).argmax(-1).cpu()
    mirror_predictions = candidate(action_mirror_states).argmax(-1).cpu()
    moving = test_data["velocities"] != 0
    normal_metrics = evaluate_metrics(
        candidate, test_states["action_conditioned"],
        test_data["correct_actions"])
    missing_first_metrics = evaluate_metrics(
        candidate, action_missing_first,
        missing_first["correct_actions"])
    missing_second_metrics = evaluate_metrics(
        candidate, action_missing_second,
        missing_second["correct_actions"])
    audit = {
        "normal_brier": normal_metrics["brier"],
        "normal_mean_margin": normal_metrics["mean_margin"],
        "normal_normalized_policy_entropy": (
            normal_metrics["normalized_policy_entropy"]),
        "reverse_velocity_accuracy": evaluate(
            candidate, action_reverse_states,
            reverse_data["correct_actions"]),
        "reverse_moving_prediction_flip_rate": float(
            (normal_predictions[moving] !=
             reverse_predictions[moving]).float().mean()),
        "mirror_accuracy": evaluate(
            candidate, action_mirror_states,
            mirror_data["correct_actions"]),
        "mirror_moving_prediction_flip_rate": float(
            (normal_predictions[moving] !=
             mirror_predictions[moving]).float().mean()),
        "missing_first_accuracy": (
            missing_first_metrics["verified_accuracy"]),
        "missing_first_normalized_policy_entropy": (
            missing_first_metrics["normalized_policy_entropy"]),
        "missing_second_accuracy": (
            missing_second_metrics["verified_accuracy"]),
        "missing_second_normalized_policy_entropy": (
            missing_second_metrics["normalized_policy_entropy"]),
        "frozen_cursor_success": float(
            (test_data["velocities"] == 0).float().mean()),
    }
    candidate_arm = arms["action_conditioned"]
    controls = [arms["passive"], arms["shuffled_action"]]
    best_control = max(float(arm["aulc_above_majority"])
                       for arm in controls)
    bits = candidate_arm["bits_to_60"]
    faster = bits is not None and all(
        arm["bits_to_60"] is None or bits < arm["bits_to_60"]
        for arm in controls)
    gate = {
        "aulc_advantage": float(
            candidate_arm["aulc_above_majority"]) - best_control,
        "fewer_bits_to_60": faster,
        "causal_audits_pass": (
            audit["reverse_velocity_accuracy"] >= 0.60 and
            audit["reverse_moving_prediction_flip_rate"] >= 0.60 and
            audit["mirror_accuracy"] >= 0.60 and
            audit["mirror_moving_prediction_flip_rate"] >= 0.60 and
            audit["missing_first_accuracy"] <= 0.40 and
            audit["missing_second_accuracy"] <= 0.40 and
            audit["missing_first_normalized_policy_entropy"] >
            audit["normal_normalized_policy_entropy"] and
            audit["missing_second_normalized_policy_entropy"] >
            audit["normal_normalized_policy_entropy"]),
    }
    gate["advance_to_second_seed"] = bool(
        gate["aulc_advantage"] >= 0.03 and gate["fewer_bits_to_60"] and
        gate["causal_audits_pass"])
    report = {
        "schema": "micro-intercept-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb", "own_opaque_action",
            "logging_propensity", "scalar_terminal_reward"],
        "verifier_private": [
            "position", "velocity", "cursor_position",
            "correct_command", "intercept_distance"],
        "configuration": vars(args) | {"report": str(args.report)},
        "protocol_effects_private": list(protocol),
        "pretraining": pretraining,
        "arms": arms,
        "causal_audit": audit,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "causal_audit": audit,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
