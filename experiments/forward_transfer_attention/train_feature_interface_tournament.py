"""Eight-clone tournament at the 32-verifier-bit frontier.

All clones share the same sensory/predictive core and verifier experience.
They differ only in task-agnostic latent interfaces and readout size/rate.
Selection and final blind auditing use disjoint generated lifetimes.
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

from .train import seed_everything
from .train_identify_then_act import (
    ActionHistoryCore,
    PRETRAIN_START,
    TEST_START,
    TRAIN_START,
    decision_features,
    evaluate,
    fit_readout,
    identify_batch,
    make_readout,
    pretrain_core,
)


BLIND_START = TEST_START + 8_000_000


@dataclass(frozen=True)
class Candidate:
    name: str
    interface: str
    width: int
    learning_rate: float


CANDIDATES = (
    Candidate("concat_w64_lr3e3", "concat", 64, 3e-3),
    Candidate("concat_w32_lr1e3", "concat", 32, 1e-3),
    Candidate("state_w64_lr3e3", "state", 64, 3e-3),
    Candidate("delta_w64_lr3e3", "delta", 64, 3e-3),
    Candidate("state_delta_w64_lr3e3", "state_delta", 64, 3e-3),
    Candidate(
        "state_delta_product_w64_lr3e3",
        "state_delta_product", 64, 3e-3),
    Candidate(
        "consequence_relation_w64_lr3e3",
        "consequence_relation", 64, 3e-3),
    Candidate(
        "state_pairwise_w32_lr1e3",
        "state_pairwise", 32, 1e-3),
)

REFINED_CANDIDATES = (
    Candidate("concat_w24_lr5e4", "concat", 24, 5e-4),
    Candidate("concat_w32_lr5e4", "concat", 32, 5e-4),
    Candidate("concat_w32_lr1e3_parent", "concat", 32, 1e-3),
    Candidate("concat_w48_lr1e3", "concat", 48, 1e-3),
    Candidate("state_pairwise_w16_lr5e4", "state_pairwise", 16, 5e-4),
    Candidate("state_pairwise_w24_lr75e5", "state_pairwise", 24, 7.5e-4),
    Candidate(
        "state_pairwise_w32_lr1e3_parent",
        "state_pairwise", 32, 1e-3),
    Candidate("state_pairwise_w48_lr1e3", "state_pairwise", 48, 1e-3),
)


def feature_interface(features: torch.Tensor, interface: str) -> torch.Tensor:
    """Apply a generic relation interface to state/candidate consequences."""
    state, consequence_zero, consequence_one = features.chunk(3, dim=-1)
    delta = consequence_one - consequence_zero
    if interface == "concat":
        return features
    if interface == "state":
        return state
    if interface == "delta":
        return delta
    if interface == "state_delta":
        return torch.cat([state, delta], dim=-1)
    if interface == "state_delta_product":
        return torch.cat([state, delta, state * delta], dim=-1)
    if interface == "consequence_relation":
        return torch.cat([
            consequence_zero,
            consequence_one,
            delta,
            consequence_zero * consequence_one,
        ], dim=-1)
    if interface == "state_pairwise":
        return torch.cat([
            state,
            consequence_zero,
            consequence_one,
            state * consequence_zero,
            state * consequence_one,
        ], dim=-1)
    raise ValueError(interface)


def _initial_readout(
        input_width: int, candidate: Candidate, *,
        seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    seed_everything(seed)
    model = make_readout(
        "bottleneck", input_width, candidate.width).to(device)
    return copy.deepcopy(model.state_dict())


def _curve(
        candidate: Candidate,
        initial: dict[str, torch.Tensor],
        train_features: torch.Tensor,
        selection_features: torch.Tensor,
        attempted: torch.Tensor,
        rewards: torch.Tensor,
        correct: torch.Tensor,
        *,
        updates: int,
        batch_size: int,
        seed: int) -> tuple[dict[str, object], torch.nn.Module]:
    points = []
    final_model = None
    for prefix in (8, 16, 32):
        model = fit_readout(
            initial,
            train_features[:prefix],
            attempted[:prefix],
            rewards[:prefix],
            readout_kind="bottleneck",
            intention_width=candidate.width,
            updates=updates,
            batch_size=batch_size,
            learning_rate=candidate.learning_rate,
            seed=seed + prefix,
        )
        result = evaluate(model, selection_features, correct)
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
    summary = {
        "curve": points,
        "final_accuracy": points[-1]["verified_accuracy"],
        "stable_bits_to_75": stable,
        "aulc_above_chance": sum(
            max(0.0, point["verified_accuracy"] - 0.5)
            for point in points) / len(points),
    }
    assert final_model is not None
    return summary, final_model


def _single_control(
        candidate: Candidate,
        initial: dict[str, torch.Tensor],
        train_features: torch.Tensor,
        blind_features: torch.Tensor,
        attempted: torch.Tensor,
        rewards: torch.Tensor,
        correct: torch.Tensor,
        *,
        shuffle: str | None,
        updates: int,
        batch_size: int,
        seed: int) -> dict[str, object]:
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
    model = fit_readout(
        initial,
        train_features,
        control_actions,
        control_rewards,
        readout_kind="bottleneck",
        intention_width=candidate.width,
        updates=updates,
        batch_size=batch_size,
        learning_rate=candidate.learning_rate,
        seed=seed + 2,
    )
    return evaluate(model, blind_features, correct)


def _blind_audit(
        model: torch.nn.Module,
        core: ActionHistoryCore,
        interface: str,
        normal: dict[str, torch.Tensor],
        protocol_swap: dict[str, torch.Tensor],
        target_reverse: dict[str, torch.Tensor],
        missing: dict[str, torch.Tensor],
        no_effect: dict[str, torch.Tensor],
        *,
        device: torch.device) -> dict[str, float]:
    def features(data: dict[str, torch.Tensor]) -> torch.Tensor:
        return feature_interface(
            decision_features(
                core, data, passive=False, device=device),
            interface)

    normal_result = evaluate(
        model, features(normal), normal["correct_actions"])
    protocol_result = evaluate(
        model, features(protocol_swap),
        protocol_swap["correct_actions"])
    target_result = evaluate(
        model, features(target_reverse),
        target_reverse["correct_actions"])
    missing_result = evaluate(
        model, features(missing), missing["correct_actions"])
    no_effect_result = evaluate(
        model, features(no_effect), no_effect["correct_actions"])
    return {
        "normal_accuracy": normal_result["verified_accuracy"],
        "normal_entropy": normal_result["normalized_policy_entropy"],
        "protocol_swap_accuracy": protocol_result["verified_accuracy"],
        "protocol_swap_prediction_flip": float(
            (normal_result["predictions"] !=
             protocol_result["predictions"]).float().mean()),
        "target_reverse_accuracy": target_result["verified_accuracy"],
        "target_reverse_prediction_flip": float(
            (normal_result["predictions"] !=
             target_result["predictions"]).float().mean()),
        "missing_consequence_accuracy": (
            missing_result["verified_accuracy"]),
        "missing_consequence_entropy": (
            missing_result["normalized_policy_entropy"]),
        "no_probe_effect_accuracy": no_effect_result["verified_accuracy"],
    }


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
        "--generation", choices=("broad", "refined"), default="broad")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    candidates = (
        CANDIDATES if args.generation == "broad"
        else REFINED_CANDIDATES)
    started = time.perf_counter()
    seed_everything(args.seed)
    device = torch.device(args.device)

    pretrain = identify_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False)
    train = identify_batch(TRAIN_START, 32, heldout=False)
    selection = identify_batch(
        TEST_START, args.selection_lifetimes, heldout=True)
    blind = identify_batch(
        BLIND_START, args.blind_lifetimes, heldout=True)
    blind_protocol = identify_batch(
        BLIND_START, args.blind_lifetimes, heldout=True,
        swap_protocol=True)
    blind_target = identify_batch(
        BLIND_START, args.blind_lifetimes, heldout=True,
        reverse_target=True)
    blind_missing = identify_batch(
        BLIND_START, args.blind_lifetimes, heldout=True,
        missing_consequence=True)
    blind_no_effect = identify_batch(
        BLIND_START, args.blind_lifetimes, heldout=True,
        no_probe_effect=True)

    core = ActionHistoryCore(64).to(device)
    pretraining = pretrain_core(
        core, pretrain, mode="action_conditioned",
        steps=args.pretrain_steps, batch_size=args.batch_size,
        learning_rate=3e-4, seed=args.seed, device=device)
    base_train = decision_features(
        core, train, passive=False, device=device)
    base_selection = decision_features(
        core, selection, passive=False, device=device)
    base_blind = decision_features(
        core, blind, passive=False, device=device)

    order = torch.randperm(
        32, generator=torch.Generator().manual_seed(args.seed + 59))
    attempted = train["attempted_actions"][order]
    rewards = train["rewards"][order]

    population, models, initials = {}, {}, {}
    for index, candidate in enumerate(candidates):
        train_features = feature_interface(
            base_train, candidate.interface)[order.to(device)]
        selection_features = feature_interface(
            base_selection, candidate.interface)
        initial = _initial_readout(
            train_features.shape[-1], candidate,
            seed=args.seed + 1000 + index, device=device)
        summary, model = _curve(
            candidate, initial, train_features, selection_features,
            attempted, rewards, selection["correct_actions"],
            updates=args.fit_updates, batch_size=args.batch_size,
            seed=args.seed + 2000 + index * 100)
        population[candidate.name] = {
            "interface": candidate.interface,
            "readout_width": candidate.width,
            "learning_rate": candidate.learning_rate,
            **summary,
        }
        models[candidate.name] = model
        initials[candidate.name] = initial
        print(json.dumps({
            "candidate": candidate.name, **population[candidate.name],
        }, sort_keys=True), flush=True)

    def rank(candidate: Candidate) -> tuple[float, float, float]:
        result = population[candidate.name]
        stable = result["stable_bits_to_75"]
        return (
            1.0 if stable is not None else 0.0,
            -float(stable) if stable is not None else -math.inf,
            float(result["aulc_above_chance"]),
        )

    winner = max(candidates, key=rank)
    winner_model = models[winner.name]
    winner_train = feature_interface(
        base_train, winner.interface)[order.to(device)]
    winner_blind = feature_interface(base_blind, winner.interface)
    controls = {
        "action_shuffled": _single_control(
            winner, initials[winner.name], winner_train, winner_blind,
            attempted, rewards, blind["correct_actions"],
            shuffle="action", updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 4000),
        "reward_shuffled": _single_control(
            winner, initials[winner.name], winner_train, winner_blind,
            attempted, rewards, blind["correct_actions"],
            shuffle="reward", updates=args.fit_updates,
            batch_size=args.batch_size, seed=args.seed + 5000),
    }

    fresh_core = ActionHistoryCore(64).to(device)
    fresh_train = feature_interface(
        decision_features(
            fresh_core, train, passive=False, device=device),
        winner.interface)[order.to(device)]
    fresh_blind = feature_interface(
        decision_features(
            fresh_core, blind, passive=False, device=device),
        winner.interface)
    controls["fully_fresh_core"] = _single_control(
        winner, initials[winner.name], fresh_train, fresh_blind,
        attempted, rewards, blind["correct_actions"],
        shuffle=None, updates=args.fit_updates,
        batch_size=args.batch_size, seed=args.seed + 6000)

    audit = _blind_audit(
        winner_model, core, winner.interface,
        blind, blind_protocol, blind_target,
        blind_missing, blind_no_effect, device=device)
    admitted = bool(
        population[winner.name]["stable_bits_to_75"] is not None and
        audit["normal_accuracy"] >= 0.75 and
        audit["protocol_swap_accuracy"] >= 0.75 and
        audit["protocol_swap_prediction_flip"] >= 0.75 and
        audit["target_reverse_accuracy"] >= 0.75 and
        audit["target_reverse_prediction_flip"] >= 0.75 and
        audit["missing_consequence_accuracy"] <= 0.60 and
        audit["missing_consequence_entropy"] >
        audit["normal_entropy"] and
        audit["no_probe_effect_accuracy"] <= 0.60 and
        controls["action_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["reward_shuffled"]["verified_accuracy"] <= 0.60 and
        controls["fully_fresh_core"]["verified_accuracy"] <= 0.60)

    report = {
        "schema": "feature-interface-tournament-v1",
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
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
            "schema": "feature-interface-tournament-checkpoint-v1",
            "candidate": winner.name,
            "interface": winner.interface,
            "readout_width": winner.width,
            "core": {
                key: value.detach().cpu()
                for key, value in core.state_dict().items()},
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
