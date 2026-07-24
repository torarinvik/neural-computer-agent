"""Conservative adaptation tournament at the 32-verifier-bit frontier.

Every clone receives the same cached visual embeddings, attempted opaque
actions, scalar outcomes, initialization, optimizer-update budget, and blind
evaluation lifetimes.  The only experimental variable is which small part of
the predictive system may adapt.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .train import seed_everything
from .train_action_conditioned_success import selected_success_loss
from .train_identify_then_act import (
    ActionHistoryCore,
    PRETRAIN_START,
    TEST_START,
    TRAIN_START,
    identify_batch,
    make_readout,
    pretrain_core,
)


BLIND_START = TEST_START + 10_000_000


@dataclass(frozen=True)
class Candidate:
    name: str
    adaptation: str
    adapter_rank: int = 0
    readout_lr: float = 1e-3
    core_lr: float = 0.0


CANDIDATES = (
    Candidate("frozen_readout", "frozen"),
    Candidate("residual_rank4", "adapter", adapter_rank=4),
    Candidate("residual_rank8", "adapter", adapter_rank=8),
    Candidate("residual_rank16", "adapter", adapter_rank=16),
    Candidate(
        "action_embedding_lr1e4", "action_embedding", core_lr=1e-4),
    Candidate(
        "predictor_tail_lr1e4", "predictor_tail", core_lr=1e-4),
    Candidate(
        "action_predictor_tail_lr1e4", "action_predictor_tail",
        core_lr=1e-4),
    Candidate(
        "recurrent_predictor_tail_lr3e5",
        "recurrent_predictor_tail", core_lr=3e-5),
)


class ZeroResidualAdapter(nn.Module):
    """A zero-effect adapter which must earn every latent perturbation."""

    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, rank),
            nn.GELU(),
            nn.Linear(rank, width),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class CachedBehaviorSystem(nn.Module):
    """Predict actions from cached pixels while selectively adapting dynamics."""

    def __init__(
            self, base: ActionHistoryCore, candidate: Candidate, *,
            initialization_seed: int) -> None:
        super().__init__()
        self.hidden = base.hidden
        self.candidate = candidate
        self.action_embedding = copy.deepcopy(base.action_embedding)
        self.recurrent = copy.deepcopy(base.recurrent)
        self.predictor = copy.deepcopy(base.predictor)

        # Instantiate the common answer path under an identical seed before
        # constructing candidate-specific modules.
        seed_everything(initialization_seed)
        self.readout = make_readout("bottleneck", base.hidden * 3, 32)
        self.adapter = (
            ZeroResidualAdapter(base.hidden * 3, candidate.adapter_rank)
            if candidate.adapter_rank else None)
        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        for module in (
                self.action_embedding, self.recurrent, self.predictor):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        adaptation = self.candidate.adaptation
        if adaptation in ("action_embedding", "action_predictor_tail"):
            for parameter in self.action_embedding.parameters():
                parameter.requires_grad_(True)
        if adaptation in (
                "predictor_tail", "action_predictor_tail",
                "recurrent_predictor_tail"):
            final_linear = next(
                module for module in reversed(self.predictor)
                if isinstance(module, nn.Linear))
            for parameter in final_linear.parameters():
                parameter.requires_grad_(True)
        if adaptation == "recurrent_predictor_tail":
            for parameter in self.recurrent.parameters():
                parameter.requires_grad_(True)
        if adaptation not in (
                "frozen", "adapter", "action_embedding", "predictor_tail",
                "action_predictor_tail", "recurrent_predictor_tail"):
            raise ValueError(adaptation)

    def latent_features(
            self, visual: torch.Tensor,
            previous_actions: torch.Tensor) -> torch.Tensor:
        action = self.action_embedding(previous_actions)
        states = self.recurrent(torch.cat([visual, action], dim=-1))[0]
        state = states[:, -1]
        consequences = []
        for action_id in range(2):
            action_ids = torch.full(
                (state.shape[0],), action_id, dtype=torch.long,
                device=state.device)
            embedded = self.action_embedding(action_ids)
            consequences.append(
                self.predictor(torch.cat([state, embedded], dim=-1)))
        features = torch.cat([state, *consequences], dim=-1)
        if self.adapter is not None:
            features = features + self.adapter(features)
        return features

    def forward(
            self, visual: torch.Tensor,
            previous_actions: torch.Tensor) -> torch.Tensor:
        return self.readout(self.latent_features(visual, previous_actions))

    def optimizer_groups(self) -> list[dict[str, object]]:
        fast = list(self.readout.parameters())
        if self.adapter is not None:
            fast += list(self.adapter.parameters())
        groups: list[dict[str, object]] = [{
            "params": fast, "lr": self.candidate.readout_lr}]
        core = [
            parameter
            for module in (
                self.action_embedding, self.recurrent, self.predictor)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if core:
            groups.append({"params": core, "lr": self.candidate.core_lr})
        return groups


@torch.no_grad()
def cache_visual(
        vision: nn.Module, data: dict[str, torch.Tensor], *,
        device: torch.device) -> torch.Tensor:
    frames = data["frames"][:, :3].to(device)
    batch, steps = frames.shape[:2]
    vision.eval()
    return vision(frames.flatten(0, 1)).reshape(batch, steps, -1)


def _fit(
        base: ActionHistoryCore, candidate: Candidate,
        visual: torch.Tensor, previous: torch.Tensor,
        attempted: torch.Tensor, rewards: torch.Tensor, *,
        updates: int, batch_size: int, seed: int) -> CachedBehaviorSystem:
    model = CachedBehaviorSystem(
        base, candidate, initialization_seed=seed + 7).to(visual.device)
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(), weight_decay=1e-4)
    attempted = attempted.to(visual.device)
    rewards = rewards.to(visual.device)
    previous = previous.to(visual.device)
    generator = torch.Generator(device=visual.device).manual_seed(seed + 47)
    model.train()
    for _ in range(updates):
        indices = torch.randint(
            visual.shape[0],
            (min(batch_size, visual.shape[0]),),
            generator=generator, device=visual.device)
        logits = model(visual[indices], previous[indices])
        loss = selected_success_loss(
            logits, attempted[indices], rewards[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _metrics(
        model: CachedBehaviorSystem, visual: torch.Tensor,
        previous: torch.Tensor, correct: torch.Tensor) -> dict[str, object]:
    # Reuse the common evaluator through a tiny identity module.
    with torch.no_grad():
        logits = model(visual, previous.to(visual.device))
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


def _curve(
        base: ActionHistoryCore, candidate: Candidate,
        train_visual: torch.Tensor, train_previous: torch.Tensor,
        attempted: torch.Tensor, rewards: torch.Tensor,
        selection_visual: torch.Tensor, selection_previous: torch.Tensor,
        selection_correct: torch.Tensor, *,
        updates: int, batch_size: int, seed: int,
        ) -> tuple[dict[str, object], CachedBehaviorSystem]:
    points = []
    final_model = None
    for prefix in (8, 16, 32):
        model = _fit(
            base, candidate, train_visual[:prefix],
            train_previous[:prefix], attempted[:prefix], rewards[:prefix],
            updates=updates, batch_size=batch_size, seed=seed + prefix)
        result = _metrics(
            model, selection_visual, selection_previous,
            selection_correct)
        points.append({
            "unique_reward_bits": prefix,
            "unique_lifetimes": prefix,
            "optimizer_updates": updates,
            "examples_processed": updates * min(batch_size, prefix),
            "verified_accuracy": result["verified_accuracy"],
        })
        final_model = model
    stable = next((
        point["unique_reward_bits"]
        for index, point in enumerate(points)
        if all(
            later["verified_accuracy"] >= 0.75
            for later in points[index:])
    ), None)
    assert final_model is not None
    return {
        "curve": points,
        "final_accuracy": points[-1]["verified_accuracy"],
        "stable_bits_to_75": stable,
        "aulc_above_chance": sum(
            max(0.0, point["verified_accuracy"] - 0.5)
            for point in points) / len(points),
    }, final_model


def _rank(result: dict[str, object]) -> tuple[float, float, float]:
    stable = result["stable_bits_to_75"]
    return (
        1.0 if stable is not None else 0.0,
        -float(stable) if stable is not None else -math.inf,
        float(result["aulc_above_chance"]),
    )


def _audit(
        model: CachedBehaviorSystem,
        cached: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]],
        ) -> dict[str, float]:
    results = {
        name: _metrics(
            model, visual, data["previous_actions"],
            data["correct_actions"])
        for name, (visual, data) in cached.items()
    }
    normal = results["normal"]
    protocol = results["protocol_swap"]
    target = results["target_reverse"]
    return {
        "normal_accuracy": normal["verified_accuracy"],
        "normal_entropy": normal["normalized_policy_entropy"],
        "protocol_swap_accuracy": protocol["verified_accuracy"],
        "protocol_swap_prediction_flip": float(
            (normal["predictions"] !=
             protocol["predictions"]).float().mean()),
        "target_reverse_accuracy": target["verified_accuracy"],
        "target_reverse_prediction_flip": float(
            (normal["predictions"] !=
             target["predictions"]).float().mean()),
        "missing_consequence_accuracy": (
            results["missing"]["verified_accuracy"]),
        "missing_consequence_entropy": (
            results["missing"]["normalized_policy_entropy"]),
        "no_probe_effect_accuracy": (
            results["no_effect"]["verified_accuracy"]),
    }


def _train_control(
        base: ActionHistoryCore, candidate: Candidate,
        train_visual: torch.Tensor, train_previous: torch.Tensor,
        blind_visual: torch.Tensor, blind_previous: torch.Tensor,
        attempted: torch.Tensor, rewards: torch.Tensor,
        blind_correct: torch.Tensor, *, shuffle: str | None,
        updates: int, batch_size: int, seed: int,
        ) -> dict[str, object]:
    control_actions = attempted.clone()
    control_rewards = rewards.clone()
    permutation = torch.randperm(
        attempted.shape[0],
        generator=torch.Generator().manual_seed(seed + 1))
    if shuffle == "action":
        control_actions = control_actions[permutation]
    elif shuffle == "reward":
        control_rewards = control_rewards[permutation]
    elif shuffle is not None:
        raise ValueError(shuffle)
    model = _fit(
        base, candidate, train_visual, train_previous,
        control_actions, control_rewards, updates=updates,
        batch_size=batch_size, seed=seed + 2)
    return _metrics(
        model, blind_visual, blind_previous, blind_correct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--pretrain-lifetimes", type=int, default=128)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--selection-lifetimes", type=int, default=256)
    parser.add_argument("--blind-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--candidate", choices=tuple(
            candidate.name for candidate in CANDIDATES),
        help="Run one exact parent for a cheap replication.")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    candidates = (
        tuple(candidate for candidate in CANDIDATES
              if candidate.name == args.candidate)
        if args.candidate else CANDIDATES)
    started = time.perf_counter()
    seed_everything(args.seed)
    device = torch.device(args.device)

    pretrain = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False)
    train = identify_batch(TRAIN_START, 32, heldout=False)
    selection = identify_batch(
        TEST_START, args.selection_lifetimes, heldout=True)
    blind_data = {
        "normal": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True),
        "protocol_swap": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            swap_protocol=True),
        "target_reverse": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            reverse_target=True),
        "missing": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            missing_consequence=True),
        "no_effect": identify_batch(
            BLIND_START, args.blind_lifetimes, heldout=True,
            no_probe_effect=True),
    }

    base = ActionHistoryCore(64).to(device)
    pretraining = pretrain_core(
        base, pretrain, mode="action_conditioned",
        steps=args.pretrain_steps, batch_size=args.batch_size,
        learning_rate=3e-4, seed=args.seed, device=device)
    train_visual = cache_visual(base.vision, train, device=device)
    selection_visual = cache_visual(base.vision, selection, device=device)
    blind_cached = {
        name: (cache_visual(base.vision, data, device=device), data)
        for name, data in blind_data.items()
    }

    order = torch.randperm(
        32, generator=torch.Generator().manual_seed(args.seed + 59))
    train_visual = train_visual[order.to(device)]
    train_previous = train["previous_actions"][order].to(device)
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]

    population, models = {}, {}
    for index, candidate in enumerate(candidates):
        summary, model = _curve(
            base, candidate, train_visual, train_previous,
            attempted, rewards, selection_visual,
            selection["previous_actions"], selection["correct_actions"],
            updates=args.fit_updates, batch_size=args.batch_size,
            seed=args.seed + 1000)
        trainable = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad)
        population[candidate.name] = {
            "adaptation": candidate.adaptation,
            "adapter_rank": candidate.adapter_rank,
            "readout_learning_rate": candidate.readout_lr,
            "core_learning_rate": candidate.core_lr,
            "trainable_parameters": trainable,
            **summary,
        }
        models[candidate.name] = model
        print(json.dumps({
            "candidate": candidate.name, **population[candidate.name],
        }, sort_keys=True), flush=True)

    winner = max(
        candidates, key=lambda candidate: _rank(
            population[candidate.name]))
    winner_model = models[winner.name]
    audit = _audit(winner_model, blind_cached)
    controls = {
        "action_shuffled": _train_control(
            base, winner, train_visual, train_previous,
            blind_cached["normal"][0],
            blind_data["normal"]["previous_actions"],
            attempted, rewards, blind_data["normal"]["correct_actions"],
            shuffle="action", updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 4000),
        "reward_shuffled": _train_control(
            base, winner, train_visual, train_previous,
            blind_cached["normal"][0],
            blind_data["normal"]["previous_actions"],
            attempted, rewards, blind_data["normal"]["correct_actions"],
            shuffle="reward", updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 5000),
    }

    seed_everything(args.seed + 9000)
    fresh = ActionHistoryCore(64).to(device)
    fresh_train_visual = cache_visual(fresh.vision, train, device=device)[
        order.to(device)]
    fresh_blind_visual = cache_visual(
        fresh.vision, blind_data["normal"], device=device)
    controls["fully_fresh_core"] = _train_control(
        fresh, winner, fresh_train_visual, train_previous,
        fresh_blind_visual, blind_data["normal"]["previous_actions"],
        attempted, rewards, blind_data["normal"]["correct_actions"],
        shuffle=None, updates=args.fit_updates,
        batch_size=args.batch_size, seed=args.seed + 6000)

    admitted = bool(
        population[winner.name]["stable_bits_to_75"] is not None and
        audit["normal_accuracy"] >= 0.75 and
        audit["protocol_swap_accuracy"] >= 0.75 and
        audit["protocol_swap_prediction_flip"] >= 0.75 and
        audit["target_reverse_accuracy"] >= 0.75 and
        audit["target_reverse_prediction_flip"] >= 0.75 and
        audit["missing_consequence_accuracy"] <= 0.60 and
        audit["missing_consequence_entropy"] > audit["normal_entropy"] and
        audit["no_probe_effect_accuracy"] <= 0.60 and
        controls["action_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["reward_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["fully_fresh_core"]["verified_accuracy"] <= 0.60)

    report = {
        "schema": "learning-mechanism-tournament-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "visual_embeddings_cached_identically_across_clones": True,
        "population_size": len(candidates),
        "unique_reward_bits_per_clone": 32,
        "selection_and_blind_lifetimes_disjoint": True,
        "configuration": vars(args) | {
            "report": str(args.report),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
        },
        "pretraining": pretraining,
        "population": population,
        "winner": winner.name,
        "winner_blind_audit": audit,
        "winner_controls": {
            name: {
                key: value for key, value in result.items()
                if key != "predictions"
            }
            for name, result in controls.items()
        },
        "admitted_to_second_seed": admitted,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    if admitted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "learning-mechanism-checkpoint-v1",
            "candidate": winner.name,
            "candidate_configuration": winner.__dict__,
            "vision": {
                key: value.detach().cpu()
                for key, value in base.vision.state_dict().items()},
            "system": {
                key: value.detach().cpu()
                for key, value in winner_model.state_dict().items()},
        }, args.checkpoint_out)
    print(json.dumps({
        "winner": winner.name,
        "blind_audit": audit,
        "controls": report["winner_controls"],
        "admitted_to_second_seed": admitted,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
