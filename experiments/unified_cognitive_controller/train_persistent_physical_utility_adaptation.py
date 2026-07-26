"""Adapt memory utility from reward while the same physical banks persist."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .audit_multifeature_utility import _materialize_histories
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_physical_stream import (
    _apply_winner,
    _candidate_batch,
    _initial_rows,
    _physical_rewards,
    _stream_features,
    _tensor_rewards,
)
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_multifeature_utility_adaptation import _expanded_controller


def _clone_model(
        payload: dict[str, object], device: torch.device,
        ) -> UnifiedCognitiveController:
    model, _ = _expanded_controller(payload, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7031)
    parser.add_argument("--banks", type=int, default=8)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--rounds-per-phase", type=int, default=3)
    parser.add_argument("--perturbation", type=float, default=3.0)
    parser.add_argument("--step-size", type=float, default=1.5)
    parser.add_argument("--shuffle-physical-rewards", action="store_true")
    parser.add_argument(
        "--reset-banks-each-round", action="store_true",
        help="fresh-bank control: rematerialize physical histories each round")
    parser.add_argument(
        "--target-intervention",
        choices=("none", "cold", "empty_history", "shuffled_history"),
        default="none",
        help="one-time intervention immediately before reliability transfer")
    parser.add_argument(
        "--curriculum",
        choices=("standard", "gradual_reliability"), default="standard")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.banks < 2 or args.rounds_per_phase < 1:
        raise ValueError("at least two banks and one round are required")

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = _expanded_controller(payload, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    frozen = _clone_model(payload, device)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    initial_residual = (
        model.memory_replacement_extra_gate.weight.detach().clone())
    if args.curriculum == "gradual_reliability":
        phases = [
            ("mild_reliability", (0.35, 0.35, 0.3)),
            ("reliability_dominant", (0.3, 0.3, 0.4)),
            ("mild_reliability_return", (0.35, 0.35, 0.3)),
        ]
    else:
        phases = [
            ("old_equal", (0.5, 0.5, 0.0)),
            ("reliability_dominant", (0.3, 0.3, 0.4)),
            ("old_return", (0.5, 0.5, 0.0)),
        ]
    started = time.perf_counter()
    initial = frequency_recency_batch(
        model, banks=args.banks, capacity=args.bank_capacity,
        seed=args.seed * 10_000, device=device,
        write_threshold=0.5, noise_scale=0.04,
        recency_weight=0.5, frequency_weight=0.5)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        initial_directory = root / "initial"
        initial_directory.mkdir()
        memories, _, _, _, initial_exact, requested_exact = (
            _materialize_histories(
                model, initial, initial_directory, device=device))
        row_batch, row_queries = _initial_rows(
            initial, banks=args.banks, capacity=args.bank_capacity,
            device=device)
        direction_generator = torch.Generator(device=device).manual_seed(
            args.seed + 70_000_000)
        trace = []
        phase_summaries = []
        state_exact = initial_exact
        candidate_exact = 0
        transition_exact = True
        maximum_parity_difference = 0.0
        maximum_cross_choice_regret = 0.0
        total_replacements = 0
        rounds = 0

        for phase_index, (phase, weights) in enumerate(phases):
            if phase == "reliability_dominant":
                if args.target_intervention == "cold":
                    model.memory_replacement_extra_gate.weight.copy_(
                        initial_residual)
            phase_rows = []
            for round_index in range(args.rounds_per_phase):
                rounds += 1
                seed = (
                    args.seed * 10_000_000
                    + phase_index * 1_000_000 + round_index + 1)
                data = frequency_recency_batch(
                    model, banks=args.banks,
                    capacity=args.bank_capacity, seed=seed,
                    device=device, write_threshold=0.5,
                    noise_scale=0.04,
                    recency_weight=weights[0],
                    frequency_weight=weights[1],
                    reliability_weight=weights[2])
                if args.reset_banks_each_round:
                    fresh_directory = (
                        root / f"round-{rounds:03d}-fresh-initial")
                    fresh_directory.mkdir()
                    (
                        memories, _, _, _, exact, fresh_requested_exact,
                    ) = _materialize_histories(
                        model, data, fresh_directory, device=device)
                    state_exact += exact
                    requested_exact += fresh_requested_exact
                    row_batch, row_queries = _initial_rows(
                        data, banks=args.banks,
                        capacity=args.bank_capacity, device=device)
                candidate_batch = _candidate_batch(
                    data, banks=args.banks,
                    capacity=args.bank_capacity, device=device)
                candidate_queries = data["query_group"][:, -1]
                generator = torch.Generator(
                    device=device).manual_seed(seed + 70_000_000)
                noise = (
                    torch.rand(
                        args.banks, args.bank_capacity,
                        generator=generator, device=device) * 2 - 1
                ) * 0.04
                features, target = _stream_features(
                    model, memories, data["candidate_key"],
                    data["candidate_strength"], weights=weights,
                    noise=noise)
                if (
                        phase == "reliability_dominant"
                        and args.target_intervention in (
                            "empty_history", "shuffled_history")):
                    visible_memories = []
                    for source in memories:
                        visible = DiskLatentMemory.__new__(
                            DiskLatentMemory)
                        visible.store = source.store.clone()
                        for field in (
                                "access_count", "success_count",
                                "failure_count"):
                            values = getattr(visible.store, field)
                            if args.target_intervention == "empty_history":
                                values[:visible.count].zero_()
                            else:
                                values[:visible.count] = values[
                                    :visible.count].roll(1)
                        visible_memories.append(visible)
                    features, _ = _stream_features(
                        model, visible_memories, data["candidate_key"],
                        data["candidate_strength"], weights=weights,
                        noise=noise)
                current = (
                    model.memory_replacement_extra_gate.weight
                    .detach().clone())
                direction = torch.randint(
                    0, 2, (2,), generator=direction_generator,
                    device=device).float() * 2 - 1
                signs = (1.0, 0.0, -1.0)
                physical_means = []
                tensor_means = []
                actions_by_candidate = []
                for candidate_index, sign in enumerate(signs):
                    model.memory_replacement_extra_gate.weight.copy_(
                        current + sign * args.perturbation
                        * direction.unsqueeze(0))
                    actions = model.memory_replacement_scores(
                        features).argmax(-1)
                    actions_by_candidate.append(actions)
                    directory = (
                        root / f"round-{rounds:03d}-candidate-"
                        f"{candidate_index}")
                    directory.mkdir()
                    physical, exact = _physical_rewards(
                        model, memories, data, row_batch, row_queries,
                        candidate_batch, candidate_queries, actions, target,
                        directory, device=device)
                    tensor = _tensor_rewards(
                        model, memories, data, row_batch, row_queries,
                        candidate_batch, candidate_queries, actions, target,
                        device=device)
                    candidate_exact += exact
                    physical_means.append(float(physical.mean()))
                    tensor_means.append(float(tensor.mean()))
                aligned_rewards = list(physical_means)
                if args.shuffle_physical_rewards:
                    order = torch.randperm(
                        3, generator=direction_generator,
                        device=device).tolist()
                    aligned_rewards = [
                        physical_means[index] for index in order]
                physical_winner = max(
                    range(3), key=aligned_rewards.__getitem__)
                tensor_winner = max(
                    range(3), key=tensor_means.__getitem__)
                maximum_parity_difference = max(
                    maximum_parity_difference,
                    max(abs(a - b) for a, b in zip(
                        physical_means, tensor_means)))
                cross_regret = max(
                    max(tensor_means) - tensor_means[physical_winner],
                    max(physical_means) - physical_means[tensor_winner])
                maximum_cross_choice_regret = max(
                    maximum_cross_choice_regret, cross_regret)
                winner_sign = signs[physical_winner]
                model.memory_replacement_extra_gate.weight.copy_(
                    current + winner_sign * args.step_size
                    * direction.unsqueeze(0))
                learned_actions = actions_by_candidate[physical_winner]
                frozen_actions = frozen.memory_replacement_scores(
                    features).argmax(-1)
                frozen_directory = root / f"round-{rounds:03d}-frozen"
                frozen_directory.mkdir()
                frozen_rewards, exact = _physical_rewards(
                    frozen, memories, data, row_batch, row_queries,
                    candidate_batch, candidate_queries, frozen_actions,
                    target, frozen_directory, device=device)
                candidate_exact += exact
                state_directory = root / f"round-{rounds:03d}-state"
                state_directory.mkdir()
                (
                    memories, row_batch, row_queries, exact, replacements,
                    exact_transition,
                ) = _apply_winner(
                    model, memories, data, row_batch, row_queries,
                    candidate_batch, candidate_queries, learned_actions,
                    target, state_directory, device=device)
                state_exact += exact
                transition_exact = transition_exact and exact_transition
                total_replacements += replacements
                row = {
                    "phase": phase,
                    "round": round_index + 1,
                    "physical_rewards": physical_means,
                    "aligned_physical_rewards": aligned_rewards,
                    "tensor_rewards": tensor_means,
                    "physical_winner": physical_winner,
                    "learned_reward": physical_means[physical_winner],
                    "frozen_reward": float(frozen_rewards.mean()),
                    "learned_target_rate": float(
                        (learned_actions == target).float().mean()),
                    "frozen_target_rate": float(
                        (frozen_actions == target).float().mean()),
                    "residual_weights": (
                        model.memory_replacement_extra_gate.weight
                        .flatten().tolist()),
                    "replacements": replacements,
                }
                trace.append(row)
                phase_rows.append(row)
            phase_summaries.append({
                "phase": phase,
                "weights": weights,
                "learned_reward": sum(
                    row["learned_reward"] for row in phase_rows)
                    / len(phase_rows),
                "frozen_reward": sum(
                    row["frozen_reward"] for row in phase_rows)
                    / len(phase_rows),
                "learned_target_rate": sum(
                    row["learned_target_rate"] for row in phase_rows)
                    / len(phase_rows),
                "frozen_target_rate": sum(
                    row["frozen_target_rate"] for row in phase_rows)
                    / len(phase_rows),
                "first_round_learned_target_rate":
                    phase_rows[0]["learned_target_rate"],
                "last_round_learned_target_rate":
                    phase_rows[-1]["learned_target_rate"],
            })

        binary = evaluate(
            model, count=512, trials=6, seed=args.seed + 91_000_000,
            device=device, task="binary_mapping", feedback_trials=1)
        four_rule = evaluate(
            model, count=512, trials=6, seed=args.seed + 92_000_000,
            device=device, task="four_rule", feedback_trials=2)
        changed = [
            name for name, value in model.state_dict().items()
            if not torch.equal(
                initial_state[name], value.detach().cpu())]
        expected_state = args.banks * (
            rounds + 1
            + (rounds if args.reset_banks_each_round else 0))
        expected_candidates = args.banks * rounds * 4
        reliability = phase_summaries[1]
        old_return = phase_summaries[2]
        gate = {
            "every_state_save_reload_exact":
                state_exact == expected_state,
            "every_candidate_remained_bounded":
                candidate_exact == expected_candidates,
            "every_history_transition_exact": transition_exact,
            "physical_tensor_rewards_match_within_1e_6":
                maximum_parity_difference <= 1e-6,
            "physical_tensor_choices_equivalent_within_1e_6":
                maximum_cross_choice_regret <= 1e-6,
            "reliability_switch_improves_or_matches_frozen_reward":
                reliability["learned_reward"]
                >= reliability["frozen_reward"] - 1e-6,
            "reliability_switch_improves_or_matches_frozen_target_rate":
                reliability["learned_target_rate"]
                >= reliability["frozen_target_rate"],
            "old_utility_recovery_not_worse_than_frozen_by_over_13_points":
                old_return["learned_target_rate"]
                >= old_return["frozen_target_rate"] - 0.13,
            "at_least_one_replacement_persisted":
                total_replacements > 0,
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "only_extra_residual_changed":
                set(changed).issubset({
                    "memory_replacement_extra_gate.weight"}),
        }
        gate["accepted"] = (
            all(gate.values()) and not args.shuffle_physical_rewards)

    report = {
        "schema":
            "unified-controller-persistent-physical-adaptation-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "training_signal":
            "three_candidate_persistent_physical_verified_horse_race",
        "memory_lifetime": (
            "one_decision_fresh_control"
            if args.reset_banks_each_round
            else "persistent_across_all_decisions"),
        "semantic_or_utility_labels_used_for_training": False,
        "physical_reward_is_sovereign": True,
        "tensor_arena_role": "parity_audit_only",
        "trace": trace,
        "phase_summaries": phase_summaries,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "accounting": {
            "persistent_banks": args.banks,
            "physical_rounds": rounds,
            "candidate_save_reloads": expected_candidates,
            "state_save_reloads": expected_state,
            "total_replacements": total_replacements,
            "optimizer_updates": rounds,
            "replayed_examples": 0,
            "requested_initial_histories_reproduced_exactly":
                requested_exact,
        },
        "maximum_physical_tensor_reward_difference":
            maximum_parity_difference,
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
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint_saved": report["checkpoint_saved"],
        "gate": gate,
        "phase_summaries": phase_summaries,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
