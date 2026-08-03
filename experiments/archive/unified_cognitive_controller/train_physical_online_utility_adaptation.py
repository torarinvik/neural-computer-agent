"""Train online utility adaptation from physical disk-backed episodes."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import torch

from .audit_frequency_recency_replacement import (
    _physical_policy,
    _retarget_future,
)
from .audit_multifeature_utility import _materialize_histories
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_memory_replacement import _bank_reward
from .train_multifeature_utility_adaptation import (
    _evaluate,
    _expanded_controller,
)


@torch.no_grad()
def _realize_on_disk(
        model, data: dict[str, object], root: Path, *,
        capacity: int, weights: tuple[float, float, float],
        device: torch.device) -> dict[str, object]:
    history_directory = root / "histories"
    history_directory.mkdir()
    started = time.perf_counter()
    (
        memories, access, successes, failures,
        persisted_exact, requested_exact,
    ) = _materialize_histories(
        model, data, history_directory, device=device)
    history_seconds = time.perf_counter() - started
    normalized_access = (
        torch.log1p(access.to(data["bank_ages"].dtype))
        / math.log(10.0))
    reliability = (
        (successes.to(data["bank_ages"].dtype) + 1.0)
        / (successes + failures + 2).to(data["bank_ages"].dtype))
    visible_utility = (
        weights[0] * data["bank_ages"] / capacity
        + weights[1] * normalized_access
        + weights[2] * reliability)
    realized_utility = visible_utility + data["utility_noise"]
    target_slot = realized_utility.argmin(-1)
    target_action = target_slot + 1
    future_batch, future_queries = _retarget_future(data, target_slot)
    features = data["option_features"].clone()
    features[:, 1:, 5] = normalized_access - 0.5
    features[:, 1:, 6] = reliability - 0.5
    realized_data = dict(data)
    realized_data.update({
        "option_features": features,
        "target_action": target_action,
        "future_batch": future_batch,
        "future_queries": future_queries,
    })
    return {
        "memories": memories,
        "data": realized_data,
        "target_action": target_action,
        "future_batch": future_batch,
        "future_queries": future_queries,
        "persisted_exact": persisted_exact,
        "requested_exact": requested_exact,
        "history_seconds": history_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7010)
    parser.add_argument("--phase-steps", type=int, default=8)
    parser.add_argument("--banks", type=int, default=32)
    parser.add_argument("--test-banks", type=int, default=2048)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--perturbation", type=float, default=3.0)
    parser.add_argument("--step-size", type=float, default=1.5)
    parser.add_argument("--shuffle-physical-rewards", action="store_true")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.phase_steps < 1 or args.banks < 1:
        raise ValueError("phase steps and banks must be positive")
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
    direction_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    history = []
    endpoints = []
    generated_contexts = 0
    future_bits = 0
    physical_banks = 0
    physical_candidate_evaluations = 0
    persisted_histories = 0
    maximum_parity_difference = 0.0
    parity_winner_matches = 0
    parity_equivalent_choices = 0
    maximum_cross_choice_regret = 0.0
    started = time.perf_counter()

    for phase_index, (phase, weights) in enumerate(phases):
        phase_seed = args.seed * 10_000_000 + phase_index * 1_000_000
        for step in range(1, args.phase_steps + 1):
            data = frequency_recency_batch(
                model, banks=args.banks, capacity=args.bank_capacity,
                seed=phase_seed + step, device=device,
                write_threshold=0.5, noise_scale=0.04,
                recency_weight=weights[0],
                frequency_weight=weights[1],
                reliability_weight=weights[2])
            generated_contexts += int(data["generated_contexts"])
            future_bits += args.banks * args.bank_capacity
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                realized = _realize_on_disk(
                    model, data, root, capacity=args.bank_capacity,
                    weights=weights, device=device)
                physical_banks += args.banks
                persisted_histories += int(realized["persisted_exact"])
                direction = torch.randint(
                    0, 2, (2,), generator=direction_generator,
                    device=device, dtype=torch.long).to(
                        torch.float32) * 2.0 - 1.0
                current = (
                    model.memory_replacement_extra_gate.weight
                    .detach().clone())
                signs = (1.0, 0.0, -1.0)
                physical_rewards = []
                tensor_rewards = []
                actions_by_candidate = []
                physical_seconds = 0.0
                for candidate_index, sign in enumerate(signs):
                    with torch.no_grad():
                        model.memory_replacement_extra_gate.weight.copy_(
                            current + sign * args.perturbation
                            * direction.unsqueeze(0))
                        actions = model.memory_replacement_scores(
                            realized["data"]["option_features"]).argmax(-1)
                        actions_by_candidate.append(actions)
                        tensor_rewards.append(float(_bank_reward(
                            model, realized["data"], actions,
                            device=device).mean()))
                        candidate_directory = (
                            root / f"candidate-{candidate_index}")
                        candidate_directory.mkdir()
                        candidate_started = time.perf_counter()
                        physical = _physical_policy(
                            model, realized["memories"], realized["data"],
                            actions, realized["future_batch"],
                            realized["future_queries"],
                            candidate_directory, device=device)
                        physical_seconds += (
                            time.perf_counter() - candidate_started)
                        physical_rewards.append(physical["accuracy"])
                        physical_candidate_evaluations += 1
                aligned_physical_rewards = list(physical_rewards)
                if args.shuffle_physical_rewards:
                    order = torch.randperm(
                        len(signs), generator=direction_generator,
                        device=device).tolist()
                    aligned_physical_rewards = [
                        physical_rewards[index] for index in order]
                physical_winner = max(
                    range(len(signs)),
                    key=aligned_physical_rewards.__getitem__)
                tensor_winner = max(
                    range(len(signs)), key=tensor_rewards.__getitem__)
                parity_winner_matches += int(
                    physical_winner == tensor_winner)
                physical_choice_tensor_regret = (
                    max(tensor_rewards)
                    - tensor_rewards[physical_winner])
                tensor_choice_physical_regret = (
                    max(physical_rewards)
                    - physical_rewards[tensor_winner])
                cross_choice_regret = max(
                    physical_choice_tensor_regret,
                    tensor_choice_physical_regret)
                parity_equivalent_choices += int(
                    cross_choice_regret <= 1e-6)
                maximum_cross_choice_regret = max(
                    maximum_cross_choice_regret,
                    cross_choice_regret)
                maximum_parity_difference = max(
                    maximum_parity_difference,
                    max(
                        abs(physical_reward - tensor_reward)
                        for physical_reward, tensor_reward in zip(
                            physical_rewards, tensor_rewards)))
                winner_sign = signs[physical_winner]
                with torch.no_grad():
                    model.memory_replacement_extra_gate.weight.copy_(
                        current + winner_sign * args.step_size
                        * direction.unsqueeze(0))
                winner_actions = actions_by_candidate[physical_winner]
                history.append({
                    "phase": phase,
                    "step": step,
                    "direction": direction.tolist(),
                    "physical_rewards": physical_rewards,
                    "aligned_physical_rewards": aligned_physical_rewards,
                    "tensor_rewards": tensor_rewards,
                    "physical_winner": physical_winner,
                    "tensor_winner": tensor_winner,
                    "cross_choice_regret": cross_choice_regret,
                    "residual_weights": (
                        model.memory_replacement_extra_gate.weight
                        .flatten().tolist()),
                    "winner_target_eviction_rate": float(
                        (
                            winner_actions
                            == realized["target_action"]).float().mean()),
                    "histories_persisted":
                        realized["persisted_exact"],
                    "history_seconds": realized["history_seconds"],
                    "physical_candidate_seconds": physical_seconds,
                    "elapsed_seconds": time.perf_counter() - started,
                })

        evaluation_seed = args.seed + 90_000_000 + phase_index * 100_000
        learned = _evaluate(
            model, banks=args.test_banks, capacity=args.bank_capacity,
            seed=evaluation_seed, device=device, write_threshold=0.5,
            noise_scale=0.04, weights=weights)
        frozen_report = _evaluate(
            frozen, banks=args.test_banks, capacity=args.bank_capacity,
            seed=evaluation_seed, device=device, write_threshold=0.5,
            noise_scale=0.04, weights=weights)
        ablated = _evaluate(
            model, banks=args.test_banks, capacity=args.bank_capacity,
            seed=evaluation_seed, device=device, write_threshold=0.5,
            noise_scale=0.04, weights=weights,
            ablate_reliability_residual=True)
        endpoints.append({
            "phase": phase,
            "weights": weights,
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
    phase_passes = [
        (
            endpoint["learned"]["target_eviction_rate"] >= 0.72
            and endpoint["learned"]["accuracy"]
            >= endpoint["learned"]["policy_accuracies"]["visible_oracle"]
            - 0.03)
        for endpoint in endpoints]
    gate = {
        "every_phase_near_oracle_and_target_at_least_72":
            all(phase_passes),
        "reliability_phase_beats_frozen_by_4_target_points":
            reliability["learned"]["target_eviction_rate"]
            >= reliability["frozen"]["target_eviction_rate"] + 0.04,
        "reliability_residual_adds_4_target_points":
            reliability["learned"]["target_eviction_rate"]
            >= (
                reliability["reliability_residual_ablated"]
                ["target_eviction_rate"] + 0.04),
        "old_utility_returns_within_5_points":
            endpoints[2]["learned"]["target_eviction_rate"]
            >= endpoints[0]["learned"]["target_eviction_rate"] - 0.05,
        "physical_and_tensor_choices_equivalent_within_1e_6":
            parity_equivalent_choices == args.phase_steps * len(phases),
        "maximum_physical_tensor_reward_difference_at_most_1_point":
            maximum_parity_difference <= 0.01,
        "every_physical_history_persisted":
            persisted_histories == physical_banks,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "only_extra_residual_changed":
            changed == ["memory_replacement_extra_gate.weight"],
    }
    gate["accepted"] = (
        all(gate.values()) and not args.shuffle_physical_rewards)
    report = {
        "schema": "unified-controller-physical-online-adaptation-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "training_signal": "three_candidate_physical_verified_horse_race",
        "tensor_arena_role": "parity_audit_only",
        "boundary_signal_visible_to_learner": False,
        "optimizer_reset_at_boundaries": False,
        "semantic_or_utility_labels_used_for_training": False,
        "phases": [
            {"name": name, "weights": weights}
            for name, weights in phases],
        "history": history,
        "endpoint_reports": endpoints,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "accounting": {
            "generated_training_contexts": generated_contexts,
            "future_query_training_bits": future_bits,
            "candidate_query_bits": 3 * future_bits,
            "unique_training_verifier_bits":
                generated_contexts + 3 * future_bits,
            "physical_banks": physical_banks,
            "physical_candidate_evaluations":
                physical_candidate_evaluations,
            "persisted_histories": persisted_histories,
            "optimizer_updates": args.phase_steps * len(phases),
            "replayed_examples": 0,
        },
        "maximum_physical_tensor_reward_difference":
            maximum_parity_difference,
        "physical_tensor_winner_matches": parity_winner_matches,
        "physical_tensor_equivalent_choices": parity_equivalent_choices,
        "maximum_cross_choice_regret": maximum_cross_choice_regret,
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
                "target_eviction_rate":
                    endpoint["learned"]["target_eviction_rate"],
                "frozen_target_eviction_rate":
                    endpoint["frozen"]["target_eviction_rate"],
            }
            for endpoint in endpoints],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
