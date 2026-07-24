"""Sub-minute zero-semantic-label predictive-state experiment.

The representation learner sees only rendered RGB sequences.  The downstream
policy receives only sampled-action reward from a private verifier; rule and
identity metadata never enter a differentiable loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.syllogimous_latent_agent.model import VisionEncoder

from .environment import generate_temporal_attention_lifetime
from .probe_palette_sample_efficiency import _balanced_specs
from .train import seed_everything


PRETRAIN_START = 61_000_000
PRETRAIN_TEST_START = 63_000_000
POLICY_TRAIN_START = 65_000_000
POLICY_TEST_START = 67_000_000
TRAIN_PALETTES = ((0, 1), (1, 2), (2, 3))
TEST_PALETTES = ((0, 2), (0, 3), (1, 3))


class PredictiveStateAgent(nn.Module):
    """Small visual recurrent core with disposable predictor and policy heads."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.vision = VisionEncoder(hidden)
        self.recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden))
        self.policy = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.value = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def states(self, frames: torch.Tensor) -> torch.Tensor:
        batch, steps = frames.shape[:2]
        encoded = self.vision(frames.flatten(0, 1)).reshape(
            batch, steps, self.hidden)
        return self.recurrent(encoded)[0]

    def act(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        final = self.states(frames)[:, -1]
        return self.policy(final), self.value(final).squeeze(-1)


def _frames(items) -> torch.Tensor:
    arrays = [np.concatenate(item, axis=0) for item in items]
    return torch.from_numpy(np.stack(arrays)).permute(0, 1, 4, 2, 3).float().div_(255.0)


def predictive_sequences(start: int, count: int, palettes=TRAIN_PALETTES,
                         *, heldout: bool = False) -> torch.Tensor:
    sequences = []
    for seed, palette in _balanced_specs(
            start, count, palettes, heldout=heldout):
        lifetime = generate_temporal_attention_lifetime(
            seed, heldout=heldout, feedback_mode="color-object",
            color_ids=palette)
        sequences.append([support.frames for support in lifetime.supports])
    return _frames(sequences)


def policy_sequences(start: int, count: int, *, heldout: bool,
                     palettes, reverse_events: bool = False,
                     reverse_support_only: bool = False,
                     reverse_query_only: bool = False,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    if sum((reverse_events, reverse_support_only, reverse_query_only)) > 1:
        raise ValueError("choose at most one reversal intervention")
    sequences = []
    private_rules = []
    for seed, palette in _balanced_specs(
            start, count, palettes, heldout=heldout):
        lifetime = generate_temporal_attention_lifetime(
            seed, heldout=heldout, feedback_mode="color-object",
            color_ids=palette)
        support = lifetime.supports[0].frames
        query = lifetime.future_queries[0].frames
        if reverse_events or reverse_support_only:
            # Preserve feedback identity while reversing the demonstrated
            # order. The inferred first/last rule must therefore flip.
            support = support[[1, 0, 2]]
        if reverse_events or reverse_query_only:
            query = query[[1, 0]]
        sequences.append([support, query])
        private_rules.append(
            1 - lifetime.rule
            if (reverse_events or reverse_support_only)
            else lifetime.rule)
    return _frames(sequences), torch.tensor(private_rules, dtype=torch.long)


def _normalized_cosine_loss(prediction: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
    prediction = nn.functional.normalize(prediction, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return (2.0 - 2.0 * (prediction * target).sum(-1)).mean()


def _standardized_prediction_loss(prediction: torch.Tensor,
                                  target: torch.Tensor) -> torch.Tensor:
    """Match each latent dimension without letting a shared offset dominate."""
    prediction_flat = prediction.flatten(0, -2)
    target_flat = target.flatten(0, -2)
    prediction = (
        (prediction - prediction_flat.mean(0)) /
        prediction_flat.std(0, unbiased=False).clamp_min(1e-4))
    target = (
        (target - target_flat.mean(0)) /
        target_flat.std(0, unbiased=False).clamp_min(1e-4))
    return (prediction - target).square().mean()


def _variance_loss(values: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
    flat = values.flatten(0, -2)
    return torch.relu(floor - flat.std(dim=0, unbiased=False)).mean()


def _correlation_loss(values: torch.Tensor) -> torch.Tensor:
    """Penalize rank-one solutions even when every dimension has variance."""
    flat = values.flatten(0, -2)
    flat = (flat - flat.mean(0)) / flat.std(
        0, unbiased=False).clamp_min(1e-4)
    correlation = flat.T @ flat / max(1, flat.shape[0])
    diagonal = torch.diagonal(correlation)
    return (
        correlation.square().sum() - diagonal.square().sum()
    ) / correlation.numel()


def _effective_rank(values: torch.Tensor) -> float:
    flat = values.flatten(0, -2)
    centered = flat - flat.mean(0)
    singular = torch.linalg.svdvals(centered.float().cpu())
    probabilities = singular.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    return float(torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum()))


def _prediction_loss(prediction: torch.Tensor, target: torch.Tensor,
                     objective: str) -> torch.Tensor:
    if objective == "cosine":
        return _normalized_cosine_loss(prediction, target)
    if objective == "standardized":
        return _standardized_prediction_loss(prediction, target)
    raise ValueError(f"unknown predictive objective {objective!r}")


@torch.no_grad()
def _ema_update(target: nn.Module, online: nn.Module, decay: float) -> None:
    for target_parameter, online_parameter in zip(
            target.parameters(), online.parameters()):
        target_parameter.mul_(decay).add_(
            online_parameter, alpha=1.0 - decay)


def _future_targets(target_encoder: nn.Module, frames: torch.Tensor,
                    *, shuffled: bool, target_kind: str = "next"
                    ) -> torch.Tensor:
    batch, steps = frames.shape[:2]
    with torch.no_grad():
        if target_kind == "next":
            targets = target_encoder(frames[:, 1:].flatten(0, 1)).reshape(
                batch, steps - 1, -1)
        elif target_kind == "delta":
            encoded = target_encoder(frames.flatten(0, 1)).reshape(
                batch, steps, -1)
            targets = encoded[:, 1:] - encoded[:, :-1]
        else:
            raise ValueError(f"unknown target kind {target_kind!r}")
    if shuffled:
        targets = targets.roll(1, dims=0)
    return targets


@torch.no_grad()
def representation_metrics(agent: PredictiveStateAgent,
                           target_encoder: nn.Module,
                           frames: torch.Tensor, *,
                           objective: str, target_kind: str) -> dict[str, float]:
    states = agent.states(frames)
    predictions = agent.predictor(states[:, :-1])
    targets = _future_targets(
        target_encoder, frames, shuffled=False, target_kind=target_kind)
    flat = states.flatten(0, 1)
    normalized = nn.functional.normalize(flat, dim=-1)
    sample = normalized[:min(256, normalized.shape[0])]
    similarity = sample @ sample.T
    if sample.shape[0] > 1:
        mask = ~torch.eye(
            sample.shape[0], dtype=torch.bool, device=sample.device)
        pairwise_cosine = float(similarity[mask].mean())
    else:
        pairwise_cosine = 1.0
    shuffled_targets = targets.roll(1, dims=0)
    future_loss = float(_prediction_loss(predictions, targets, objective))
    shuffled_future_loss = float(
        _prediction_loss(predictions, shuffled_targets, objective))
    return {
        "heldout_future_loss": future_loss,
        "heldout_shuffled_future_loss": shuffled_future_loss,
        "future_alignment_margin": shuffled_future_loss - future_loss,
        "state_std_mean": float(flat.std(0, unbiased=False).mean()),
        "prediction_std_mean": float(
            predictions.flatten(0, 1).std(0, unbiased=False).mean()),
        "effective_rank": _effective_rank(states),
        "prediction_effective_rank": _effective_rank(predictions),
        "target_effective_rank": _effective_rank(targets),
        "distinct_lifetime_pairwise_cosine": pairwise_cosine,
    }


def pretrain(agent: PredictiveStateAgent, frames: torch.Tensor, *,
             steps: int, batch_size: int, learning_rate: float,
             shuffled: bool, objective: str, variance_weight: float,
             correlation_weight: float, target_kind: str,
             seed: int, device: torch.device
             ) -> tuple[nn.Module, dict[str, object]]:
    seed_everything(seed)
    agent.train()
    target = copy.deepcopy(agent.vision).to(device).eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    trainable = list(agent.vision.parameters())
    trainable += list(agent.recurrent.parameters())
    trainable += list(agent.predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 991)
    history = []
    last_gradient_norm = 0.0
    for step in range(1, steps + 1):
        indices = torch.randint(
            frames.shape[0], (min(batch_size, frames.shape[0]),),
            generator=generator)
        batch = frames[indices].to(device)
        states = agent.states(batch)
        predictions = agent.predictor(states[:, :-1])
        targets = _future_targets(
            target, batch, shuffled=shuffled, target_kind=target_kind)
        predictive_loss = _prediction_loss(
            predictions, targets, objective)
        variance_loss = _variance_loss(states) + _variance_loss(predictions)
        correlation_loss = (
            _correlation_loss(states) + _correlation_loss(predictions))
        loss = (
            predictive_loss + variance_weight * variance_loss +
            correlation_weight * correlation_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_gradient_norm = float(nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        _ema_update(target, agent.vision, 0.99)
        if step in {1, max(1, steps // 2), steps}:
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "predictive_loss": float(predictive_loss.detach()),
                "variance_loss": float(variance_loss.detach()),
                "correlation_loss": float(correlation_loss.detach()),
                "gradient_norm": last_gradient_norm,
            })
    return target, {"history": history, "last_gradient_norm": last_gradient_norm}


def reinforce_loss(logits: torch.Tensor, values: torch.Tensor,
                   sampled_actions: torch.Tensor,
                   verified_rewards: torch.Tensor,
                   entropy_weight: float = 0.01) -> torch.Tensor:
    """Policy loss consumes verifier rewards, never correct-action labels."""
    distribution = torch.distributions.Categorical(logits=logits)
    # A batch reward baseline is lower variance than an initially uncalibrated
    # value estimate and still contains only verifier outcomes.
    advantage = verified_rewards - verified_rewards.mean()
    actor = -(distribution.log_prob(sampled_actions) * advantage).mean()
    critic = 0.5 * (values - verified_rewards).square().mean()
    entropy = distribution.entropy().mean()
    return actor + critic - entropy_weight * entropy


@torch.no_grad()
def policy_accuracy(agent: PredictiveStateAgent, frames: torch.Tensor,
                    private_rules: torch.Tensor, batch_size: int,
                    device: torch.device) -> float:
    agent.eval()
    correct = 0
    for offset in range(0, frames.shape[0], batch_size):
        batch = frames[offset:offset + batch_size].to(device)
        logits, _ = agent.act(batch)
        actions = logits.argmax(-1).cpu()
        correct += int((actions == private_rules[offset:offset + batch_size]).sum())
    return correct / frames.shape[0]


def train_policy(agent: PredictiveStateAgent, train_frames: torch.Tensor,
                 private_rules: torch.Tensor, test_frames: torch.Tensor,
                 test_private_rules: torch.Tensor, *, batch_size: int,
                 learning_rate: float, seed: int, device: torch.device
                 ) -> dict[str, object]:
    seed_everything(seed)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    for module in (agent.policy, agent.value):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [*agent.policy.parameters(), *agent.value.parameters()],
        lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 177)
    order = torch.randperm(
        train_frames.shape[0],
        generator=torch.Generator().manual_seed(seed + 313))
    history = [{
        "unique_lifetimes": 0,
        "heldout_accuracy": policy_accuracy(
            agent, test_frames, test_private_rules, batch_size, device),
    }]
    seen = 0
    update = 0
    total_updates = math.ceil(train_frames.shape[0] / batch_size)
    evaluation_updates = {
        1, max(1, total_updates // 4), max(1, total_updates // 2),
        max(1, 3 * total_updates // 4), total_updates}
    agent.train()
    while seen < train_frames.shape[0]:
        end = min(seen + batch_size, train_frames.shape[0])
        indices = order[seen:end]
        frames = train_frames[indices].to(device)
        rules = private_rules[indices].to(device)
        logits, values = agent.act(frames)
        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probabilities, 1, generator=generator).squeeze(-1)
        # This equality is the private verifier boundary. No correct action is
        # provided to the network or differentiated through.
        rewards = (sampled == rules).to(logits.dtype)
        loss = reinforce_loss(logits, values, sampled, rewards)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [*agent.policy.parameters(), *agent.value.parameters()], 1.0)
        optimizer.step()
        seen = end
        update += 1
        if update in evaluation_updates:
            history.append({
                "unique_lifetimes": seen,
                "heldout_accuracy": policy_accuracy(
                    agent, test_frames, test_private_rules, batch_size, device),
                "mean_train_reward": float(rewards.mean()),
                "loss": float(loss.detach()),
            })
            agent.train()
    chance = 0.5
    aulc = sum(
        max(0.0, row["heldout_accuracy"] - chance) for row in history
    ) / len(history)
    return {
        "history": history,
        "heldout_accuracy_final": history[-1]["heldout_accuracy"],
        "reward_aulc_above_chance": aulc,
        "unique_lifetimes_to_threshold": {
            str(threshold): next((
                first["unique_lifetimes"]
                for first, second in zip(history, history[1:])
                if (first["heldout_accuracy"] >= threshold and
                    second["heldout_accuracy"] >= threshold)), None)
            for threshold in (0.60, 0.70, 0.80)
        },
    }


def _fit_discarded_probe(train_x: torch.Tensor, train_y: torch.Tensor,
                         test_x: torch.Tensor, test_y: torch.Tensor,
                         reversed_x: torch.Tensor, reversed_y: torch.Tensor,
                         *, nonlinear: bool, shuffled_labels: bool,
                         seed: int, device: torch.device
                         ) -> dict[str, float]:
    """Verifier-labeled localization instrument; weights are never retained."""
    seed_everything(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x = ((train_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    reversed_x = ((reversed_x - mean) / scale).to(device)
    labels = train_y
    if shuffled_labels:
        labels = labels[torch.randperm(
            labels.numel(),
            generator=torch.Generator().manual_seed(seed + 919))]
    labels = labels.to(device)
    if nonlinear:
        model = nn.Sequential(
            nn.Linear(train_x.shape[-1], 64), nn.GELU(), nn.Linear(64, 2))
    else:
        model = nn.Linear(train_x.shape[-1], 2)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(200):
        loss = nn.functional.cross_entropy(model(train_x), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_predictions = model(train_x).argmax(-1).cpu()
        normal_predictions = model(test_x).argmax(-1).cpu()
        reversed_predictions = model(reversed_x).argmax(-1).cpu()
    return {
        "train_accuracy": float((train_predictions == labels.cpu()).float().mean()),
        "heldout_accuracy": float(
            (normal_predictions == test_y).float().mean()),
        "reversed_relabeled_accuracy": float(
            (reversed_predictions == reversed_y).float().mean()),
        "reversed_stale_accuracy": float(
            (reversed_predictions == test_y).float().mean()),
        "prediction_flip_rate": float(
            (normal_predictions != reversed_predictions).float().mean()),
    }


@torch.no_grad()
def _frozen_states(agent: PredictiveStateAgent, frames: torch.Tensor,
                   batch_size: int, device: torch.device) -> torch.Tensor:
    agent.eval()
    parts = []
    for offset in range(0, frames.shape[0], batch_size):
        parts.append(agent.states(
            frames[offset:offset + batch_size].to(device))[:, -1].cpu())
    return torch.cat(parts)


def discarded_rule_diagnostic(agent: PredictiveStateAgent,
                              train_frames: torch.Tensor,
                              train_rules: torch.Tensor,
                              test_frames: torch.Tensor,
                              test_rules: torch.Tensor,
                              reversed_frames: torch.Tensor,
                              reversed_rules: torch.Tensor, *,
                              batch_size: int, seed: int,
                              device: torch.device) -> dict[str, object]:
    train_x = _frozen_states(agent, train_frames, batch_size, device)
    test_x = _frozen_states(agent, test_frames, batch_size, device)
    reversed_x = _frozen_states(agent, reversed_frames, batch_size, device)
    return {
        "status": (
            "discarded verifier-labeled localization probe; not capability"),
        "linear": _fit_discarded_probe(
            train_x, train_rules, test_x, test_rules,
            reversed_x, reversed_rules, nonlinear=False,
            shuffled_labels=False, seed=seed, device=device),
        "mlp": _fit_discarded_probe(
            train_x, train_rules, test_x, test_rules,
            reversed_x, reversed_rules, nonlinear=True,
            shuffled_labels=False, seed=seed, device=device),
        "shuffled_label_mlp": _fit_discarded_probe(
            train_x, train_rules, test_x, test_rules,
            reversed_x, reversed_rules, nonlinear=True,
            shuffled_labels=True, seed=seed, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-test-lifetimes", type=int, default=96)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--policy-lifetimes", type=int, default=510)
    parser.add_argument("--policy-test-lifetimes", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-learning-rate", type=float, default=3e-3)
    parser.add_argument(
        "--predictive-objective", choices=("cosine", "standardized"),
        default="standardized")
    parser.add_argument(
        "--target-kind", choices=("next", "delta"), default="delta")
    parser.add_argument("--variance-weight", type=float, default=2.0)
    parser.add_argument("--correlation-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    pretrain_frames = predictive_sequences(
        PRETRAIN_START, args.pretrain_lifetimes)
    pretrain_test_frames = predictive_sequences(
        PRETRAIN_TEST_START, args.pretrain_test_lifetimes,
        palettes=TEST_PALETTES, heldout=True)
    policy_train_frames, policy_train_rules = policy_sequences(
        POLICY_TRAIN_START, args.policy_lifetimes, heldout=False,
        palettes=TRAIN_PALETTES)
    policy_test_frames, policy_test_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES)
    reversed_policy_test_frames, reversed_policy_test_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES, reverse_events=True)
    data_seconds = time.perf_counter() - started

    seed_everything(args.seed)
    base = PredictiveStateAgent(args.hidden)
    initial_state = copy.deepcopy(base.state_dict())
    arms = {}
    for name, mode in (
            ("predictive", "normal"),
            ("fresh", "fresh"),
            ("shuffled_future", "shuffled")):
        agent = PredictiveStateAgent(args.hidden).to(device)
        agent.load_state_dict(initial_state)
        arm_started = time.perf_counter()
        if mode == "fresh":
            target = copy.deepcopy(agent.vision).to(device).eval()
            pretraining = {"history": [], "last_gradient_norm": 0.0}
        else:
            target, pretraining = pretrain(
                agent, pretrain_frames, steps=args.pretrain_steps,
                batch_size=args.batch_size,
                learning_rate=args.pretrain_learning_rate,
                shuffled=mode == "shuffled", seed=args.seed,
                objective=args.predictive_objective,
                variance_weight=args.variance_weight,
                correlation_weight=args.correlation_weight,
                target_kind=args.target_kind,
                device=device)
        agent.eval()
        representation = representation_metrics(
            agent, target, pretrain_test_frames.to(device),
            objective=args.predictive_objective,
            target_kind=args.target_kind)
        downstream = train_policy(
            agent, policy_train_frames, policy_train_rules,
            policy_test_frames, policy_test_rules,
            batch_size=args.batch_size,
            learning_rate=args.policy_learning_rate,
            seed=args.seed + 100, device=device)
        diagnostic = discarded_rule_diagnostic(
            agent, policy_train_frames, policy_train_rules,
            policy_test_frames, policy_test_rules,
            reversed_policy_test_frames, reversed_policy_test_rules,
            batch_size=args.batch_size, seed=args.seed + 500,
            device=device)
        arms[name] = {
            "pretraining": pretraining,
            "representation": representation,
            "downstream_reward_only": downstream,
            "discarded_rule_diagnostic": diagnostic,
            "seconds": time.perf_counter() - arm_started,
        }
        print(json.dumps({
            "arm": name,
            "representation": representation,
            "downstream": downstream,
            "seconds": arms[name]["seconds"],
        }, sort_keys=True), flush=True)

    predictive = arms["predictive"]["downstream_reward_only"]
    controls = [
        arms["fresh"]["downstream_reward_only"],
        arms["shuffled_future"]["downstream_reward_only"],
    ]
    best_control_aulc = max(
        arm["reward_aulc_above_chance"] for arm in controls)
    best_control_final = max(
        arm["heldout_accuracy_final"] for arm in controls)
    predictive_metrics = arms["predictive"]["representation"]
    shuffled_metrics = arms["shuffled_future"]["representation"]
    noncollapsed = (
        predictive_metrics["effective_rank"] >= max(4.0, args.hidden * 0.10)
        and predictive_metrics["state_std_mean"] >= 0.02
        and predictive_metrics["prediction_std_mean"] >= 0.02
    )
    report = {
        "schema": "zero-label-predictive-state-v1",
        "learner_inputs": ["rendered_rgb_stream", "sampled_action_reward"],
        "learner_forbidden_inputs": [
            "rule_label", "object_identity", "palette_id", "event_index",
            "game_state", "semantic_target"],
        "semantic_labels_used_for_training": False,
        "private_verifier_use": "sampled action equality converted to scalar reward",
        "configuration": vars(args) | {"report": str(args.report)},
        "data_generation_seconds": data_seconds,
        "arms": arms,
        "gate": {
            "prediction_beats_shuffled": (
                predictive_metrics["future_alignment_margin"] > 0.0
                and predictive_metrics["future_alignment_margin"] >
                shuffled_metrics["future_alignment_margin"]),
            "noncollapsed": noncollapsed,
            "reward_aulc_beats_both_controls": (
                predictive["reward_aulc_above_chance"] -
                best_control_aulc >= 0.02
                and predictive["heldout_accuracy_final"] >= 0.60
                and predictive["heldout_accuracy_final"] >
                best_control_final),
            "predictive_aulc_advantage": (
                predictive["reward_aulc_above_chance"] - best_control_aulc),
            "advance_to_three_minutes": False,
        },
        "total_seconds": time.perf_counter() - started,
    }
    report["gate"]["advance_to_three_minutes"] = bool(
        report["gate"]["prediction_beats_shuffled"]
        and report["gate"]["noncollapsed"]
        and report["gate"]["reward_aulc_beats_both_controls"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": report["gate"],
                      "total_seconds": report["total_seconds"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
