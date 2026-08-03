"""Probe online adaptation of a two-dimensional generic memory utility."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .legacy_model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import (
    evaluate_frequency_recency,
    frequency_recency_batch,
)
from .train_memory_replacement import _bank_reward


def _expanded_controller(
        payload: dict[str, object], *, device: torch.device
        ) -> tuple[UnifiedCognitiveController, dict[str, object]]:
    """Add one zero-initialized generic residual without changing old scores."""
    configuration = dict(payload["model_configuration"])
    if int(configuration["adaptive_memory_replace_features"]) != 6:
        raise ValueError("the parent must have the six-feature utility interface")
    configuration["adaptive_memory_replace_features"] = 7
    model = UnifiedCognitiveController(**configuration).to(device)
    state = {
        name: value
        for name, value in payload["state_dict"].items()
        if name != "memory_replacement_extra_gate.weight"}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["memory_replacement_extra_gate.weight"] or unexpected:
        raise ValueError(
            f"unexpected expanded-controller mismatch: {missing=}, "
            f"{unexpected=}")
    old_weight = payload["state_dict"][
        "memory_replacement_extra_gate.weight"].to(device)
    with torch.no_grad():
        model.memory_replacement_extra_gate.weight.zero_()
        model.memory_replacement_extra_gate.weight[:, :1].copy_(old_weight)
    return model, configuration


@torch.no_grad()
def _evaluate(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        noise_scale: float, weights: tuple[float, float, float],
        ablate_reliability_residual: bool = False) -> dict[str, object]:
    residual = model.memory_replacement_extra_gate.weight.detach().clone()
    if ablate_reliability_residual:
        model.memory_replacement_extra_gate.weight[0, 1] = 0.0
    report = evaluate_frequency_recency(
        model, banks=banks, capacity=capacity, seed=seed, device=device,
        write_threshold=write_threshold, noise_scale=noise_scale,
        recency_weight=weights[0], frequency_weight=weights[1],
        reliability_weight=weights[2])
    model.memory_replacement_extra_gate.weight.copy_(residual)
    report["residual_weights"] = residual.flatten().tolist()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=6901)
    parser.add_argument("--phase-steps", type=int, default=16)
    parser.add_argument("--batch-banks", type=int, default=128)
    parser.add_argument("--test-banks", type=int, default=2048)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=0.04)
    parser.add_argument("--perturbation", type=float, default=3.0)
    parser.add_argument("--step-size", type=float, default=1.5)
    parser.add_argument("--active-dimensions", type=int, choices=(1, 2),
                        default=2)
    parser.add_argument("--include-center-candidate", action="store_true")
    parser.add_argument("--shuffle-rewards", action="store_true")
    args = parser.parse_args()
    if args.phase_steps < 1:
        raise ValueError("phase steps must be positive")
    if args.perturbation <= 0.0 or args.step_size <= 0.0:
        raise ValueError("search scales must be positive")

    phases = [
        ("old_equal", (0.5, 0.5, 0.0)),
        ("reliability_dominant", (0.3, 0.3, 0.4)),
        ("old_return", (0.5, 0.5, 0.0)),
        ("all_equal", (1 / 3, 1 / 3, 1 / 3)),
    ]
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = _expanded_controller(payload, device=device)
    frozen, _ = _expanded_controller(payload, device=device)
    frozen.eval()
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.memory_replacement_extra_gate.weight.requires_grad_(True)

    started = time.perf_counter()
    endpoints = []
    history = []
    generated_contexts = 0
    future_bits = 0
    generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    for phase_index, (phase, weights) in enumerate(phases):
        phase_seed = args.seed * 10_000_000 + phase_index * 1_000_000
        for step in range(1, args.phase_steps + 1):
            data = frequency_recency_batch(
                model, banks=args.batch_banks,
                capacity=args.bank_capacity, seed=phase_seed + step,
                device=device, write_threshold=args.write_threshold,
                noise_scale=args.noise_scale,
                recency_weight=weights[0],
                frequency_weight=weights[1],
                reliability_weight=weights[2])
            generated_contexts += int(data["generated_contexts"])
            future_bits += args.batch_banks * args.bank_capacity
            direction = torch.randint(
                0, 2, (2,), generator=generator, device=device,
                dtype=torch.long).to(torch.float32) * 2.0 - 1.0
            if args.active_dimensions == 1:
                direction[1] = 0.0
            with torch.no_grad():
                current = (
                    model.memory_replacement_extra_gate.weight
                    .detach().clone())
                candidate_rewards = []
                candidate_actions = []
                candidate_signs = (
                    (1.0, 0.0, -1.0)
                    if args.include_center_candidate
                    else (1.0, -1.0))
                for sign in candidate_signs:
                    model.memory_replacement_extra_gate.weight.copy_(
                        current + sign * args.perturbation
                        * direction.unsqueeze(0))
                    actions = model.memory_replacement_scores(
                        data["option_features"]).argmax(-1)
                    candidate_actions.append(actions)
                    candidate_rewards.append(_bank_reward(
                        model, data, actions, device=device))
                rewards = torch.stack(candidate_rewards)
                if args.shuffle_rewards:
                    for bank in range(rewards.shape[1]):
                        order = torch.randperm(
                            rewards.shape[0], generator=generator,
                            device=device)
                        rewards[:, bank] = rewards[order, bank]
                mean_rewards = rewards.mean(1)
                winner_index = int(mean_rewards.argmax())
                winner = candidate_signs[winner_index]
                reward_gap = (
                    mean_rewards[winner_index]
                    - mean_rewards[
                        candidate_signs.index(0.0)
                        if 0.0 in candidate_signs
                        else 1 - winner_index])
                model.memory_replacement_extra_gate.weight.copy_(
                    current + winner * args.step_size
                    * direction.unsqueeze(0))
                history.append({
                    "phase": phase,
                    "step": step,
                    "reward_gap": float(reward_gap),
                    "winner": winner,
                    "direction": direction.tolist(),
                    "residual_weights": (
                        model.memory_replacement_extra_gate.weight
                        .flatten().tolist()),
                })

        evaluation_seed = args.seed + 90_000_000 + phase_index * 100_000
        learned = _evaluate(
            model, banks=args.test_banks, capacity=args.bank_capacity,
            seed=evaluation_seed, device=device,
            write_threshold=args.write_threshold,
            noise_scale=args.noise_scale, weights=weights)
        frozen_report = _evaluate(
            frozen, banks=args.test_banks, capacity=args.bank_capacity,
            seed=evaluation_seed, device=device,
            write_threshold=args.write_threshold,
            noise_scale=args.noise_scale, weights=weights)
        ablated = _evaluate(
            model, banks=args.test_banks, capacity=args.bank_capacity,
            seed=evaluation_seed, device=device,
            write_threshold=args.write_threshold,
            noise_scale=args.noise_scale, weights=weights,
            ablate_reliability_residual=True)
        endpoints.append({
            "phase": phase,
            "weights": {
                "recency": weights[0],
                "frequency": weights[1],
                "reliability": weights[2],
            },
            "learned": learned,
            "frozen": frozen_report,
            "reliability_residual_ablated": ablated,
        })

    binary = evaluate(
        model, count=2048, trials=6, seed=args.seed + 91_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=2048, trials=6, seed=args.seed + 92_000_000,
        device=device, task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial_state[name], value.detach().cpu())]
    reliability = endpoints[1]
    old_first = endpoints[0]["learned"]["target_eviction_rate"]
    old_return = endpoints[2]["learned"]["target_eviction_rate"]
    basic_phase_gates = [
        (
            endpoint["learned"]["target_eviction_rate"] >= 0.72
            and endpoint["learned"]["accuracy"]
            >= endpoint["learned"]["policy_accuracies"]["visible_oracle"]
            - 0.03)
        for endpoint in endpoints
    ]
    gate = {
        "each_phase_near_oracle_and_target_at_least_72":
            all(basic_phase_gates),
        "reliability_phase_beats_frozen_by_4_target_points":
            reliability["learned"]["target_eviction_rate"]
            >= reliability["frozen"]["target_eviction_rate"] + 0.04,
        "reliability_residual_adds_4_target_points": (
            args.active_dimensions == 1
            or reliability["learned"]["target_eviction_rate"]
            >= (
                reliability["reliability_residual_ablated"]
                ["target_eviction_rate"] + 0.04)),
        "old_utility_returns_within_5_points":
            old_return >= old_first - 0.05,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "only_extra_residual_changed":
            changed == ["memory_replacement_extra_gate.weight"],
    }
    gate["accepted"] = (
        all(gate.values())
        and args.active_dimensions == 2
        and not args.shuffle_rewards)
    report = {
        "schema": "unified-controller-multifeature-utility-adaptation-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "boundary_signal_visible_to_learner": False,
        "optimizer_reset_at_boundaries": False,
        "semantic_or_utility_labels_used_for_training": False,
        "training_signal":
            "symmetric_rademacher_verified_horse_race",
        "phases": [
            {"name": name, "weights": weights}
            for name, weights in phases],
        "history": history,
        "endpoint_reports": endpoints,
        "changed_parameters": changed,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "accounting": {
            "generated_training_contexts": generated_contexts,
            "future_query_training_bits": future_bits,
            "candidate_query_bits": (
                (3 if args.include_center_candidate else 2)
                * future_bits),
            "unique_training_verifier_bits":
                generated_contexts + (
                    3 if args.include_center_candidate else 2) * future_bits,
            "optimizer_updates": args.phase_steps * len(phases),
            "replayed_examples": 0,
        },
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    if gate["accepted"] and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint_saved": report["checkpoint_saved"],
        "gate": gate,
        "phase_endpoints": [
            {
                "phase": endpoint["phase"],
                "residual_weights":
                    endpoint["learned"]["residual_weights"],
                "accuracy": endpoint["learned"]["accuracy"],
                "target_eviction_rate":
                    endpoint["learned"]["target_eviction_rate"],
                "frozen_target_eviction_rate":
                    endpoint["frozen"]["target_eviction_rate"],
                "ablated_target_eviction_rate":
                    endpoint["reliability_residual_ablated"]
                    ["target_eviction_rate"],
            }
            for endpoint in endpoints
        ],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
