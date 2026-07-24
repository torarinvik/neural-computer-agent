"""Zero-label transfer from a learned intention to a new action protocol.

Both acquisition phases train only from attempted actions and observed scalar
rewards. Private verifier facts are used for evaluation and reward generation,
never as differentiable labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn

from .train import seed_everything
from .train_action_conditioned_success import (
    evaluate_action_head,
    frozen_final_states,
    selected_success_loss,
)
from .train_fixed_reward_replay_sweep import select_policy_input
from .train_zero_label_predictive_state import (
    POLICY_TEST_START,
    POLICY_TRAIN_START,
    PRETRAIN_START,
    TEST_PALETTES,
    TRAIN_PALETTES,
    PredictiveStateAgent,
    policy_sequences,
    predictive_sequences,
    pretrain,
)


class IntentionModule(nn.Module):
    """Small bottleneck learned without assigning semantics to its coordinates."""

    def __init__(self, hidden: int, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
            nn.LayerNorm(width),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)


class SuccessSystem(nn.Module):
    def __init__(self, hidden: int, intention_width: int,
                 actions: int) -> None:
        super().__init__()
        self.intention = IntentionModule(hidden, intention_width)
        self.adapter = nn.Sequential(
            nn.Linear(intention_width, max(16, intention_width * 2)),
            nn.GELU(),
            nn.Linear(max(16, intention_width * 2), actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.intention(states))


class FrozenIntentionAdapter(nn.Module):
    def __init__(self, intention: IntentionModule, intention_width: int,
                 actions: int) -> None:
        super().__init__()
        self.intention = intention
        for parameter in self.intention.parameters():
            parameter.requires_grad_(False)
        self.adapter = nn.Sequential(
            nn.Linear(intention_width, max(16, intention_width * 2)),
            nn.GELU(),
            nn.Linear(max(16, intention_width * 2), actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            intention = self.intention(states)
        return self.adapter(intention)


def protocol_for_seed(seed: int, actions: int = 4) -> torch.Tensor:
    if actions < 2:
        raise ValueError("the protocol needs at least two commands")
    return torch.randperm(
        actions, generator=torch.Generator().manual_seed(seed + 51))[:2]


def correct_protocol_actions(private_rules: torch.Tensor,
                             protocol: torch.Tensor) -> torch.Tensor:
    return protocol.cpu()[private_rules.cpu()]


def opposite_rule_permutation(private_rules: torch.Tensor) -> torch.Tensor:
    """Audit-only pairing that guarantees every latent has the wrong rule."""
    rules = private_rules.cpu()
    zero = torch.where(rules == 0)[0]
    one = torch.where(rules == 1)[0]
    if zero.numel() != one.numel():
        raise ValueError("opposite-rule audit requires balanced private rules")
    permutation = torch.empty_like(rules)
    permutation[zero] = one.roll(1)
    permutation[one] = zero.roll(1)
    if not torch.all(rules[permutation] != rules):
        raise AssertionError("stale-intention pairing did not flip every rule")
    return permutation


def uniform_logged_protocol_buffer(
        states: torch.Tensor, private_rules: torch.Tensor,
        protocol: torch.Tensor, *, actions: int, seed: int
        ) -> tuple[torch.Tensor, ...]:
    """Log commands uniformly without consulting rule or correct command."""
    count = states.shape[0]
    order = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed + 11))
    states = states[order.to(states.device)]
    rules = private_rules[order]
    attempted = (torch.arange(count) % actions).to(states.device)
    correct = correct_protocol_actions(rules, protocol).to(states.device)
    rewards = (attempted == correct).to(states.dtype)
    propensities = torch.full(
        (count,), 1.0 / actions, device=states.device, dtype=states.dtype)
    return states, rules, attempted, rewards, propensities


def _fit_selected_success(
        model: nn.Module, states: torch.Tensor, actions: torch.Tensor,
        rewards: torch.Tensor, *, updates: int, batch_size: int,
        learning_rate: float, seed: int) -> dict[str, float | int]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=states.device).manual_seed(seed + 29)
    last_loss = 0.0
    last_gradient_norm = 0.0
    model.train()
    for _ in range(updates):
        indices = torch.randint(
            states.shape[0],
            (min(batch_size, states.shape[0]),),
            generator=generator, device=states.device)
        loss = selected_success_loss(
            model(states[indices]), actions[indices], rewards[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters()
             if parameter.requires_grad), 1.0))
        optimizer.step()
        last_loss = float(loss.detach())
    return {
        "optimizer_updates": updates,
        "examples_processed": updates * min(batch_size, states.shape[0]),
        "unique_reward_bits": states.shape[0],
        "last_loss": last_loss,
        "last_gradient_norm": last_gradient_norm,
    }


@torch.no_grad()
def _evaluate(model: nn.Module, states: torch.Tensor,
              private_rules: torch.Tensor, protocol: torch.Tensor,
              batch_size: int) -> dict[str, float]:
    model.eval()
    logits = torch.cat([
        model(states[offset:offset + batch_size])
        for offset in range(0, states.shape[0], batch_size)
    ])
    predictions = logits.argmax(-1).cpu()
    targets = correct_protocol_actions(private_rules, protocol)
    return {
        "verified_accuracy": float((predictions == targets).float().mean()),
        "mean_margin": float(
            (logits.topk(2, dim=-1).values[:, 0] -
             logits.topk(2, dim=-1).values[:, 1]).mean()),
    }


def _curve_aulc(curve: list[dict[str, object]], chance: float) -> float:
    return sum(
        max(0.0, float(point["verified_accuracy"]) - chance)
        for point in curve) / len(curve)


def _thresholds(curve: list[dict[str, object]]) -> dict[str, int | None]:
    return {
        str(threshold): next((
            int(point["unique_reward_bits"]) for point in curve
            if float(point["verified_accuracy"]) >= threshold), None)
        for threshold in (0.55, 0.65, 0.75)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--commands", type=int, default=4)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--phase-a-lifetimes", type=int, default=510)
    parser.add_argument("--phase-a-updates", type=int, default=200)
    parser.add_argument("--phase-b-lifetimes", type=int, default=510)
    parser.add_argument("--phase-b-updates", type=int, default=200)
    parser.add_argument("--test-lifetimes", type=int, default=384)
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

    pretrain_frames = predictive_sequences(
        PRETRAIN_START, args.pretrain_lifetimes)
    train_frames, train_rules = policy_sequences(
        POLICY_TRAIN_START, max(args.phase_a_lifetimes,
                                args.phase_b_lifetimes),
        heldout=False, palettes=TRAIN_PALETTES)
    test_frames, test_rules = policy_sequences(
        POLICY_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES)
    reversed_frames, reversed_rules = policy_sequences(
        POLICY_TEST_START, args.test_lifetimes,
        heldout=True, palettes=TEST_PALETTES, reverse_support_only=True)
    train_frames = select_policy_input(train_frames, "support-only")
    test_frames = select_policy_input(test_frames, "support-only")
    reversed_frames = select_policy_input(reversed_frames, "support-only")

    agent = PredictiveStateAgent(args.hidden).to(device)
    _, predictive_accounting = pretrain(
        agent, pretrain_frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=False, objective="standardized",
        variance_weight=2.0, correlation_weight=0.5,
        target_kind="delta", seed=args.seed, device=device)
    train_states = frozen_final_states(
        agent, train_frames, args.batch_size, device)
    test_states = frozen_final_states(
        agent, test_frames, args.batch_size, device)
    reversed_states = frozen_final_states(
        agent, reversed_frames, args.batch_size, device)

    # Phase A: acquire the latent intention with only two-action outcomes.
    phase_a_states, phase_a_rules, phase_a_actions, phase_a_rewards, _ = (
        uniform_logged_protocol_buffer(
            train_states[:args.phase_a_lifetimes],
            train_rules[:args.phase_a_lifetimes], torch.tensor([0, 1]),
            actions=2, seed=args.seed + 100))
    seed_everything(args.seed + 200)
    acquired = SuccessSystem(
        args.hidden, args.intention_width, actions=2).to(device)
    phase_a_accounting = _fit_selected_success(
        acquired, phase_a_states, phase_a_actions, phase_a_rewards,
        updates=args.phase_a_updates, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed + 201)
    phase_a_audit = evaluate_action_head(
        acquired, test_states, test_rules, args.batch_size)

    protocol = protocol_for_seed(args.seed, args.commands)
    phase_b_states, ordered_rules, attempted, rewards, propensities = (
        uniform_logged_protocol_buffer(
            train_states[:args.phase_b_lifetimes],
            train_rules[:args.phase_b_lifetimes], protocol,
            actions=args.commands, seed=args.seed + 300))

    # Identical initialization is used wherever interfaces permit it.
    seed_everything(args.seed + 400)
    adapter_template = FrozenIntentionAdapter(
        copy.deepcopy(acquired.intention), args.intention_width,
        args.commands).to(device)
    adapter_initial = copy.deepcopy(adapter_template.adapter.state_dict())
    fresh_template = SuccessSystem(
        args.hidden, args.intention_width, args.commands).to(device)
    fresh_initial = copy.deepcopy(fresh_template.state_dict())
    command_permutation = torch.randperm(
        attempted.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 401))
    reward_permutation = torch.randperm(
        rewards.shape[0],
        generator=torch.Generator().manual_seed(args.seed + 402))

    prefixes = [
        value for value in (32, 128, 256, 384, 510)
        if value <= args.phase_b_lifetimes]
    if not prefixes or prefixes[-1] != args.phase_b_lifetimes:
        prefixes.append(args.phase_b_lifetimes)

    arms: dict[str, dict[str, object]] = {}
    final_models: dict[str, nn.Module] = {}
    for arm in (
            "experienced", "fresh", "lifetime_shuffled",
            "action_shuffled", "reward_shuffled"):
        curve = []
        for prefix in prefixes:
            seed_everything(args.seed + 500 + prefix)
            if arm == "fresh":
                model: nn.Module = SuccessSystem(
                    args.hidden, args.intention_width,
                    args.commands).to(device)
                model.load_state_dict(fresh_initial)
            else:
                model = FrozenIntentionAdapter(
                    copy.deepcopy(acquired.intention),
                    args.intention_width, args.commands).to(device)
                model.adapter.load_state_dict(adapter_initial)
            states = phase_b_states[:prefix]
            arm_actions = attempted[:prefix]
            arm_rewards = rewards[:prefix]
            if arm == "lifetime_shuffled":
                states = states.roll(1, dims=0)
            elif arm == "action_shuffled":
                arm_actions = attempted[command_permutation.to(device)][:prefix]
            elif arm == "reward_shuffled":
                arm_rewards = rewards[reward_permutation.to(device)][:prefix]
            accounting = _fit_selected_success(
                model, states, arm_actions, arm_rewards,
                updates=args.phase_b_updates, batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + 600 + prefix)
            curve.append(accounting | _evaluate(
                model, test_states, test_rules, protocol, args.batch_size))
            if prefix == prefixes[-1]:
                final_models[arm] = model
        arms[arm] = {
            "curve": curve,
            # Uniform logging has 1/commands reward chance, but deterministic
            # argmax evaluation has a 50% majority floor because only the two
            # protocol-mapped commands can ever be correct.
            "reward_aulc_above_majority": _curve_aulc(curve, 0.5),
            "unique_reward_bits_to_threshold": _thresholds(curve),
            "heldout_accuracy_final": curve[-1]["verified_accuracy"],
        }
        print(json.dumps({"arm": arm, **arms[arm]}, sort_keys=True),
              flush=True)

    candidate = final_models["experienced"]
    normal_logits = candidate(test_states).argmax(-1).cpu()
    reversed_logits = candidate(reversed_states).argmax(-1).cpu()
    reversed_targets = correct_protocol_actions(reversed_rules, protocol)
    rolled_stale_accuracy = _evaluate(
        candidate, test_states.roll(1, dims=0), test_rules,
        protocol, args.batch_size)["verified_accuracy"]
    opposite_permutation = opposite_rule_permutation(test_rules)
    opposite_stale_accuracy = _evaluate(
        candidate, test_states[opposite_permutation.to(device)], test_rules,
        protocol, args.batch_size)["verified_accuracy"]
    swapped_protocol = protocol.flip(0)
    swapped_protocol_accuracy = _evaluate(
        candidate, test_states, test_rules, swapped_protocol,
        args.batch_size)["verified_accuracy"]
    causal_audit = {
        "reversed_relabeled_accuracy": float(
            (reversed_logits == reversed_targets).float().mean()),
        "prediction_flip_rate": float(
            (reversed_logits != normal_logits).float().mean()),
        "rolled_stale_intention_accuracy": rolled_stale_accuracy,
        "rolled_stale_same_rule_rate": float(
            (test_rules.roll(1) == test_rules).float().mean()),
        "opposite_rule_stale_intention_accuracy": opposite_stale_accuracy,
        "swapped_protocol_accuracy_without_recalibration":
            swapped_protocol_accuracy,
    }

    candidate_aulc = float(arms["experienced"][
        "reward_aulc_above_majority"])
    best_control_aulc = max(float(arms[name][
        "reward_aulc_above_majority"]) for name in (
            "fresh", "lifetime_shuffled",
            "action_shuffled", "reward_shuffled"))
    experienced_thresholds = arms["experienced"][
        "unique_reward_bits_to_threshold"]
    fresh_thresholds = arms["fresh"]["unique_reward_bits_to_threshold"]
    faster_threshold = any(
        experienced_thresholds[key] is not None and (
            fresh_thresholds[key] is None or
            int(experienced_thresholds[key]) < int(fresh_thresholds[key]))
        for key in experienced_thresholds)
    gate = {
        "candidate_aulc_advantage": candidate_aulc - best_control_aulc,
        "fewer_reward_bits_than_fresh": faster_threshold,
        "causal_reversal": (
            causal_audit["reversed_relabeled_accuracy"] >= 0.60 and
            causal_audit["prediction_flip_rate"] >= 0.50),
        "stale_intention_degrades": (
            causal_audit["opposite_rule_stale_intention_accuracy"] <= 0.40),
    }
    gate["advance_to_second_seed"] = bool(
        gate["candidate_aulc_advantage"] >= 0.03 and
        float(arms["experienced"]["heldout_accuracy_final"]) >= 0.60 and
        gate["fewer_reward_bits_than_fresh"] and
        gate["causal_reversal"] and gate["stale_intention_degrades"])

    report = {
        "schema": "zero-label-actuator-transfer-v2",
        "claim_boundary": (
            "device/actuator transfer only; not cross-primitive cognitive "
            "transfer"),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "learner_visible": [
            "rendered_rgb_stream", "recurrent_state", "attempted_action",
            "uniform_logging_propensity", "scalar_observed_reward"],
        "verifier_private": [
            "first_last_rule", "correct_protocol_command",
            "object_identity", "palette", "logical_lifetime_metadata"],
        "configuration": vars(args) | {"report": str(args.report)},
        "protocol": protocol.tolist(),
        "predictive_pretraining": predictive_accounting,
        "phase_a": {
            "accounting": phase_a_accounting,
            "heldout_audit": phase_a_audit,
        },
        "phase_b_logging": {
            "command_counts": torch.bincount(
                attempted.cpu(), minlength=args.commands).tolist(),
            "mean_propensity": float(propensities.mean()),
            "reward_rate": float(rewards.mean()),
        },
        "arms": arms,
        "causal_audit": causal_audit,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "causal_audit": causal_audit,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
