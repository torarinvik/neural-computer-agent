"""Reward-free predictive-objective tournament at the 32-bit frontier.

All clones begin from one shared predictive core and see exactly the same
cached sensory transitions.  They differ only in a short, reward-free
refinement objective.  The downstream answer learner is then fitted from the
same 32 attempted actions and scalar verifier outcomes.
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
from torch.nn import functional as F

from .train import seed_everything
from .train_feature_interface_tournament import BLIND_START
from .train_identify_then_act import (
    ActionHistoryCore,
    PRETRAIN_START,
    TEST_START,
    TRAIN_START,
    evaluate,
    fit_readout,
    identify_batch,
    make_readout,
    pretrain_core,
)
from .train_zero_label_predictive_state import (
    _correlation_loss,
    _standardized_prediction_loss,
    _variance_loss,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    objective: str
    auxiliary_weight: float = 0.0


CANDIDATES = (
    Candidate("no_refinement", "none"),
    Candidate("delta_continue", "delta"),
    Candidate("delta_cosine", "cosine", 0.5),
    Candidate("delta_contrastive", "contrastive", 0.25),
    Candidate("action_decodable", "action", 0.25),
    Candidate("counterfactual_separation", "separation", 0.10),
    Candidate("contrastive_action", "contrastive_action", 0.25),
    Candidate("cosine_separation", "cosine_separation", 0.25),
)


@torch.no_grad()
def cache_visual(
        vision: nn.Module, data: dict[str, torch.Tensor], *,
        device: torch.device, frames: int = 4) -> torch.Tensor:
    values = data["frames"][:, :frames].to(device)
    batch, steps = values.shape[:2]
    vision.eval()
    return vision(values.flatten(0, 1)).reshape(batch, steps, -1)


def dynamics_features(
        core: ActionHistoryCore, visual: torch.Tensor,
        previous_actions: torch.Tensor) -> torch.Tensor:
    action = core.action_embedding(previous_actions.to(visual.device))
    states = core.recurrent(torch.cat([visual[:, :3], action], -1))[0]
    state = states[:, -1]
    consequences = []
    for action_id in range(2):
        actions = torch.full(
            (state.shape[0],), action_id, dtype=torch.long,
            device=state.device)
        consequences.append(core.predict(state, actions, passive=False))
    return torch.cat([state, *consequences], -1)


@torch.no_grad()
def frozen_dynamics_features(
        core: ActionHistoryCore, visual: torch.Tensor,
        previous_actions: torch.Tensor) -> torch.Tensor:
    core.eval()
    return dynamics_features(core, visual, previous_actions).detach()


def _contrastive_loss(
        prediction: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    desired = F.normalize(desired.detach(), dim=-1)
    logits = prediction @ desired.T / 0.10
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def _separation_loss(
        core: ActionHistoryCore, states: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(states.shape[0], dtype=torch.long, device=states.device)
    one = torch.ones(states.shape[0], dtype=torch.long, device=states.device)
    prediction_zero = F.normalize(
        core.predict(states, zero, passive=False), dim=-1)
    prediction_one = F.normalize(
        core.predict(states, one, passive=False), dim=-1)
    cosine = (prediction_zero * prediction_one).sum(-1)
    # Require only a modest action-dependent distinction; the verifier still
    # decides later whether that distinction is useful.
    return F.relu(cosine - 0.75).mean()


def refine_core(
        base: ActionHistoryCore, candidate: Candidate,
        visual: torch.Tensor, transitions: torch.Tensor,
        previous: torch.Tensor, *, steps: int, batch_size: int,
        learning_rate: float, seed: int) -> tuple[ActionHistoryCore, dict]:
    core = copy.deepcopy(base)
    for parameter in core.vision.parameters():
        parameter.requires_grad_(False)
    if candidate.objective == "none" or steps == 0:
        return core, {
            "optimizer_updates": 0,
            "examples_processed": 0,
            "history": [],
        }

    seed_everything(seed + 5)
    action_head = nn.Linear(core.hidden, 3).to(visual.device)
    parameters = [
        parameter for name, parameter in core.named_parameters()
        if not name.startswith("vision.")]
    if "action" in candidate.objective:
        parameters += list(action_head.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=visual.device).manual_seed(seed + 31)
    history = []
    core.train()
    for step in range(1, steps + 1):
        indices = torch.randint(
            visual.shape[0], (min(batch_size, visual.shape[0]),),
            generator=generator, device=visual.device)
        encoded = visual[indices]
        batch_previous = previous[indices.cpu()].to(visual.device)
        batch_actions = transitions[indices.cpu()].to(visual.device)
        embedded = core.action_embedding(batch_previous)
        states = core.recurrent(
            torch.cat([encoded[:, :3], embedded], -1))[0]
        selected_states = torch.cat([states[:, 0], states[:, 2]], 0)
        selected_actions = torch.cat([
            batch_actions[:, 0], batch_actions[:, 2]], 0)
        prediction = core.predict(
            selected_states, selected_actions, passive=False)
        desired = torch.cat([
            encoded[:, 1] - encoded[:, 0],
            encoded[:, 3] - encoded[:, 2],
        ], 0).detach()
        predictive = _standardized_prediction_loss(prediction, desired)
        variance = (
            _variance_loss(selected_states) + _variance_loss(prediction))
        correlation = _correlation_loss(selected_states)
        loss = predictive + 2.0 * variance + 0.5 * correlation
        auxiliary = torch.zeros((), device=visual.device)
        if candidate.objective in ("cosine", "cosine_separation"):
            auxiliary = auxiliary + (
                1.0 - F.cosine_similarity(prediction, desired, dim=-1)
            ).mean()
        if candidate.objective in (
                "contrastive", "contrastive_action"):
            auxiliary = auxiliary + _contrastive_loss(prediction, desired)
        if candidate.objective in (
                "action", "contrastive_action"):
            auxiliary = auxiliary + F.cross_entropy(
                action_head(prediction), selected_actions)
        if candidate.objective in (
                "separation", "cosine_separation"):
            auxiliary = auxiliary + _separation_loss(core, selected_states)
        loss = loss + candidate.auxiliary_weight * auxiliary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        optimizer.step()
        if step in (1, steps):
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "predictive_loss": float(predictive.detach()),
                "auxiliary_loss": float(auxiliary.detach()),
                "gradient_norm": gradient,
            })
    return core, {
        "optimizer_updates": steps,
        "examples_processed": steps * min(batch_size, visual.shape[0]) * 2,
        "history": history,
    }


def _initial_readout(
        *, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    seed_everything(seed)
    model = make_readout("bottleneck", 192, 32).to(device)
    return copy.deepcopy(model.state_dict())


def _curve(
        core: ActionHistoryCore, initial: dict[str, torch.Tensor],
        train_features: torch.Tensor, attempted: torch.Tensor,
        rewards: torch.Tensor, selection_features: torch.Tensor,
        selection_correct: torch.Tensor, *, updates: int,
        batch_size: int, seed: int) -> tuple[dict, nn.Module]:
    points = []
    final_model = None
    for prefix in (8, 16, 32):
        model = fit_readout(
            initial, train_features[:prefix], attempted[:prefix],
            rewards[:prefix], readout_kind="bottleneck",
            intention_width=32, updates=updates, batch_size=batch_size,
            learning_rate=1e-3, seed=seed + prefix)
        result = evaluate(model, selection_features, selection_correct)
        points.append({
            "unique_reward_bits": prefix,
            "unique_lifetimes": prefix,
            "optimizer_updates": updates,
            "examples_processed": updates * min(batch_size, prefix),
            "verified_accuracy": result["verified_accuracy"],
        })
        final_model = model
    stable = next((
        point["unique_reward_bits"] for index, point in enumerate(points)
        if all(later["verified_accuracy"] >= 0.75
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


def _rank(result: dict) -> tuple[float, float, float]:
    stable = result["stable_bits_to_75"]
    return (
        1.0 if stable is not None else 0.0,
        -float(stable) if stable is not None else -math.inf,
        float(result["aulc_above_chance"]),
    )


def _blind_audit(
        model: nn.Module, core: ActionHistoryCore,
        cached: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]],
        ) -> dict[str, float]:
    results = {}
    for name, (visual, data) in cached.items():
        features = frozen_dynamics_features(
            core, visual, data["previous_actions"])
        results[name] = evaluate(
            model, features, data["correct_actions"])
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


def _control(
        initial: dict[str, torch.Tensor],
        train_features: torch.Tensor, attempted: torch.Tensor,
        rewards: torch.Tensor, blind_features: torch.Tensor,
        blind_correct: torch.Tensor, *, shuffle: str | None,
        updates: int, batch_size: int, seed: int) -> dict:
    actions = attempted.clone()
    outcomes = rewards.clone()
    permutation = torch.randperm(
        actions.shape[0],
        generator=torch.Generator().manual_seed(seed + 1))
    if shuffle == "action":
        actions = actions[permutation]
    elif shuffle == "reward":
        outcomes = outcomes[permutation]
    elif shuffle is not None:
        raise ValueError(shuffle)
    model = fit_readout(
        initial, train_features, actions, outcomes,
        readout_kind="bottleneck", intention_width=32,
        updates=updates, batch_size=batch_size,
        learning_rate=1e-3, seed=seed + 2)
    return evaluate(model, blind_features, blind_correct)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--candidate", choices=tuple(
        candidate.name for candidate in CANDIDATES))
    parser.add_argument("--pretrain-lifetimes", type=int, default=128)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--refine-steps", type=int, default=16)
    parser.add_argument("--selection-lifetimes", type=int, default=256)
    parser.add_argument("--blind-lifetimes", type=int, default=256)
    parser.add_argument("--fit-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
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
    shared_pretraining = pretrain_core(
        base, pretrain, mode="action_conditioned",
        steps=args.pretrain_steps, batch_size=args.batch_size,
        learning_rate=3e-4, seed=args.seed, device=device)
    pretrain_visual = cache_visual(
        base.vision, pretrain, device=device)
    train_visual = cache_visual(
        base.vision, train, device=device, frames=3)
    selection_visual = cache_visual(
        base.vision, selection, device=device, frames=3)
    blind_cached = {
        name: (
            cache_visual(base.vision, data, device=device, frames=3), data)
        for name, data in blind_data.items()
    }
    order = torch.randperm(
        32, generator=torch.Generator().manual_seed(args.seed + 59))
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]
    initial = _initial_readout(seed=args.seed + 1000, device=device)

    population, cores, models = {}, {}, {}
    for candidate in candidates:
        core, refinement = refine_core(
            base, candidate, pretrain_visual,
            pretrain["transition_actions"],
            pretrain["previous_actions"],
            steps=args.refine_steps, batch_size=args.batch_size,
            learning_rate=1e-4, seed=args.seed + 2000)
        train_features = frozen_dynamics_features(
            core, train_visual, train["previous_actions"])[order.to(device)]
        selection_features = frozen_dynamics_features(
            core, selection_visual, selection["previous_actions"])
        summary, model = _curve(
            core, initial, train_features, attempted, rewards,
            selection_features, selection["correct_actions"],
            updates=args.fit_updates, batch_size=args.batch_size,
            seed=args.seed + 3000)
        population[candidate.name] = {
            "objective": candidate.objective,
            "auxiliary_weight": candidate.auxiliary_weight,
            "reward_free_refinement": refinement,
            **summary,
        }
        cores[candidate.name] = core
        models[candidate.name] = model
        print(json.dumps({
            "candidate": candidate.name, **population[candidate.name],
        }, sort_keys=True), flush=True)

    winner = max(
        candidates,
        key=lambda candidate: _rank(population[candidate.name]))
    winner_core = cores[winner.name]
    winner_model = models[winner.name]
    winner_train = frozen_dynamics_features(
        winner_core, train_visual,
        train["previous_actions"])[order.to(device)]
    winner_blind = frozen_dynamics_features(
        winner_core, blind_cached["normal"][0],
        blind_data["normal"]["previous_actions"])
    controls = {
        "action_shuffled": _control(
            initial, winner_train, attempted, rewards,
            winner_blind, blind_data["normal"]["correct_actions"],
            shuffle="action", updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 4000),
        "reward_shuffled": _control(
            initial, winner_train, attempted, rewards,
            winner_blind, blind_data["normal"]["correct_actions"],
            shuffle="reward", updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 5000),
    }
    seed_everything(args.seed + 9000)
    fresh = ActionHistoryCore(64).to(device)
    fresh_train_visual = cache_visual(
        fresh.vision, train, device=device, frames=3)
    fresh_blind_visual = cache_visual(
        fresh.vision, blind_data["normal"], device=device, frames=3)
    controls["fully_fresh_core"] = _control(
        initial,
        frozen_dynamics_features(
            fresh, fresh_train_visual,
            train["previous_actions"])[order.to(device)],
        attempted, rewards,
        frozen_dynamics_features(
            fresh, fresh_blind_visual,
            blind_data["normal"]["previous_actions"]),
        blind_data["normal"]["correct_actions"],
        shuffle=None, updates=args.fit_updates,
        batch_size=args.batch_size, seed=args.seed + 6000)

    audit = _blind_audit(
        winner_model, winner_core, blind_cached)
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
        "schema": "predictive-objective-tournament-v1",
        "semantic_labels_used_for_training": False,
        "correct_action_labels_used_for_training": False,
        "reward_used_during_core_refinement": False,
        "same_cached_sensory_experience_across_clones": True,
        "population_size": len(candidates),
        "unique_reward_bits_per_clone": 32,
        "shared_unlabeled_lifetimes": args.pretrain_lifetimes,
        "configuration": vars(args) | {
            "report": str(args.report),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
        },
        "shared_pretraining": shared_pretraining,
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
            "schema": "predictive-objective-checkpoint-v1",
            "candidate": winner.name,
            "candidate_configuration": winner.__dict__,
            "core": {
                key: value.detach().cpu()
                for key, value in winner_core.state_dict().items()},
            "readout": {
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
