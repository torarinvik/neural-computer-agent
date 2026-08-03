"""Test online adaptation when memory utility changes without a task signal."""
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


@torch.no_grad()
def _target_diagnostic(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        noise_scale: float, recency_weight: float,
        frequency_weight: float) -> dict[str, float]:
    data = frequency_recency_batch(
        model, banks=banks, capacity=capacity, seed=seed,
        device=device, write_threshold=write_threshold,
        noise_scale=noise_scale, recency_weight=recency_weight,
        frequency_weight=frequency_weight)
    actions = model.memory_replacement_scores(
        data["option_features"]).argmax(-1)
    return {
        "target_eviction_rate": float(
            (actions == data["target_action"]).float().mean()),
        "visible_oracle_target_rate": float(
            (
                data["visible_oracle_action"]
                == data["target_action"]).float().mean()),
        "replace_rate": float((actions > 0).float().mean()),
        "frequency_weight_parameter": float(
            model.memory_replacement_extra_gate.weight[
                0, 0].detach()),
        "generated_contexts": int(data["generated_contexts"]),
    }


def _new_optimizer(
        model: UnifiedCognitiveController, *, learning_rate: float,
        beta1: float) -> torch.optim.Optimizer:
    assert model.memory_replacement_extra_gate is not None
    return torch.optim.Adam(
        model.memory_replacement_extra_gate.parameters(),
        lr=learning_rate, betas=(beta1, 0.999))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=6801)
    parser.add_argument("--phase-steps", type=int, default=20)
    parser.add_argument("--batch-banks", type=int, default=128)
    parser.add_argument("--trace-banks", type=int, default=512)
    parser.add_argument("--test-banks", type=int, default=2048)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=1.5)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--exploration-temperature", type=float, default=4.0)
    parser.add_argument("--replacement-cost", type=float, default=0.01)
    parser.add_argument("--trace-every", type=int, default=2)
    parser.add_argument("--dominant-weight", type=float, default=0.75)
    update = parser.add_mutually_exclusive_group()
    update.add_argument("--paired-greedy-baseline", action="store_true")
    update.add_argument("--pairwise-preference-update", action="store_true")
    update.add_argument(
        "--symmetric-perturbation-update", action="store_true")
    parser.add_argument("--include-equal-return", action="store_true")
    parser.add_argument("--perturbation", type=float, default=3.0)
    parser.add_argument("--es-step-size", type=float, default=1.5)
    parser.add_argument("--shuffle-rewards", action="store_true")
    args = parser.parse_args()
    if args.phase_steps < 1 or args.trace_every < 1:
        raise ValueError("phase and trace intervals must be positive")
    if not 0.0 <= args.beta1 < 1.0:
        raise ValueError("beta1 must be in [0, 1)")
    if args.exploration_temperature < 1.0:
        raise ValueError("exploration temperature must be at least one")
    if not 0.5 < args.dominant_weight < 1.0:
        raise ValueError("dominant weight must be between 0.5 and 1")
    if args.perturbation <= 0.0 or args.es_step_size <= 0.0:
        raise ValueError("evolution-strategy scales must be positive")
    weak_weight = 1.0 - args.dominant_weight
    phases = [
        ("recency_dominant", args.dominant_weight, weak_weight),
        ("frequency_dominant", weak_weight, args.dominant_weight),
        ("recency_return", args.dominant_weight, weak_weight),
    ]
    if args.include_equal_return:
        phases.append(("equal_return", 0.5, 0.5))

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    if model.memory_replacement_extra_gate is None:
        raise ValueError("checkpoint has no adaptive utility residual")
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    initial_weight = float(
        model.memory_replacement_extra_gate.weight[0, 0].detach())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.memory_replacement_extra_gate.parameters():
        parameter.requires_grad_(True)
    optimizer = _new_optimizer(
        model, learning_rate=args.learning_rate, beta1=args.beta1)

    # Frozen controls share exact weights and exact evaluation streams.
    frozen = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    frozen.load_state_dict(payload["state_dict"])
    frozen.eval()

    history = []
    endpoint_reports = []
    generated_contexts = 0
    future_query_bits = 0
    diagnostic_contexts = 0
    started = time.perf_counter()
    global_step = 0
    for phase_index, (phase, recency_weight, frequency_weight) in enumerate(
            phases):
        phase_seed = args.seed * 10_000_000 + phase_index * 1_000_000
        trace = _target_diagnostic(
            model, banks=args.trace_banks,
            capacity=args.bank_capacity, seed=phase_seed + 900_000,
            device=device, write_threshold=args.write_threshold,
            noise_scale=args.noise_scale,
            recency_weight=recency_weight,
            frequency_weight=frequency_weight)
        diagnostic_contexts += int(trace["generated_contexts"])
        trace.update({
            "phase": phase,
            "phase_step": 0,
            "global_step": global_step,
            "elapsed_seconds": time.perf_counter() - started,
        })
        history.append(trace)
        print(json.dumps(trace, sort_keys=True), flush=True)

        for phase_step in range(1, args.phase_steps + 1):
            global_step += 1
            model.train()
            data = frequency_recency_batch(
                model, banks=args.batch_banks,
                capacity=args.bank_capacity,
                seed=phase_seed + phase_step,
                device=device, write_threshold=args.write_threshold,
                noise_scale=args.noise_scale,
                recency_weight=recency_weight,
                frequency_weight=frequency_weight)
            generated_contexts += int(data["generated_contexts"])
            future_query_bits += (
                args.batch_banks * args.bank_capacity)
            if args.symmetric_perturbation_update:
                with torch.no_grad():
                    current_weight = float(
                        model.memory_replacement_extra_gate.weight[0, 0])
                    candidate_rewards = []
                    candidate_actions = []
                    for offset in (
                            args.perturbation, -args.perturbation):
                        model.memory_replacement_extra_gate.weight.fill_(
                            current_weight + offset)
                        candidate_logits = model.memory_replacement_scores(
                            data["option_features"])
                        candidate_action = candidate_logits.argmax(-1)
                        candidate_reward = _bank_reward(
                            model, data, candidate_action, device=device)
                        candidate_reward = candidate_reward - (
                            args.replacement_cost
                            * (candidate_action > 0).to(
                                candidate_reward.dtype))
                        candidate_rewards.append(candidate_reward)
                        candidate_actions.append(candidate_action)
                    plus_reward, minus_reward = candidate_rewards
                    if args.shuffle_rewards:
                        swap = torch.rand(
                            plus_reward.shape, device=device) < 0.5
                        original_plus = plus_reward.clone()
                        plus_reward = torch.where(
                            swap, minus_reward, plus_reward)
                        minus_reward = torch.where(
                            swap, original_plus, minus_reward)
                    reward_difference = (
                        plus_reward - minus_reward).mean()
                    direction = float(torch.sign(reward_difference))
                    model.memory_replacement_extra_gate.weight.fill_(
                        current_weight + args.es_step_size * direction)
                    if direction >= 0.0:
                        reward = plus_reward
                        actions = candidate_actions[0]
                    else:
                        reward = minus_reward
                        actions = candidate_actions[1]
                    loss = -reward_difference
                    policy_entropy = 0.0
            else:
                logits = model.memory_replacement_scores(
                    data["option_features"])
                distribution = torch.distributions.Categorical(
                    logits=(
                        torch.zeros_like(logits)
                        if args.pairwise_preference_update
                        else logits / args.exploration_temperature))
                actions = distribution.sample()
                with torch.no_grad():
                    reward = _bank_reward(
                        model, data, actions, device=device)
                    reward = reward - (
                        args.replacement_cost
                        * (actions > 0).to(reward.dtype))
                    if (
                            args.paired_greedy_baseline
                            or args.pairwise_preference_update):
                        greedy_actions = logits.argmax(-1)
                        baseline_reward = _bank_reward(
                            model, data, greedy_actions, device=device)
                        baseline_reward = baseline_reward - (
                            args.replacement_cost
                            * (greedy_actions > 0).to(
                                baseline_reward.dtype))
                    if args.shuffle_rewards:
                        reward = reward.roll(1)
                advantage = (
                    reward - baseline_reward
                    if (
                        args.paired_greedy_baseline
                        or args.pairwise_preference_update)
                    else reward - reward.mean())
                if args.pairwise_preference_update:
                    sampled_scores = torch.gather(
                        logits, 1, actions.unsqueeze(1)).squeeze(1)
                    greedy_scores = torch.gather(
                        logits, 1, greedy_actions.unsqueeze(1)).squeeze(1)
                    loss = -(
                        advantage.detach()
                        * (sampled_scores - greedy_scores)).mean()
                else:
                    loss = -(
                        advantage * distribution.log_prob(actions)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.memory_replacement_extra_gate.parameters(), 1.0)
                optimizer.step()
                policy_entropy = float(
                    distribution.entropy().mean().detach())

            if (
                    phase_step % args.trace_every == 0
                    or phase_step == args.phase_steps):
                trace = _target_diagnostic(
                    model, banks=args.trace_banks,
                    capacity=args.bank_capacity,
                    seed=phase_seed + 900_000 + phase_step,
                    device=device, write_threshold=args.write_threshold,
                    noise_scale=args.noise_scale,
                    recency_weight=recency_weight,
                    frequency_weight=frequency_weight)
                diagnostic_contexts += int(trace["generated_contexts"])
                trace.update({
                    "phase": phase,
                    "phase_step": phase_step,
                    "global_step": global_step,
                    "loss": float(loss.detach()),
                    "sampled_reward": float(reward.mean()),
                    "sampled_target_eviction_rate": float(
                        (
                            actions
                            == data["target_action"]).float().mean()),
                    "policy_entropy": policy_entropy,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                history.append(trace)
                print(json.dumps(trace, sort_keys=True), flush=True)

        evaluation_seed = args.seed + 90_000_000 + phase_index * 100_000
        learned_report = evaluate_frequency_recency(
            model, banks=args.test_banks,
            capacity=args.bank_capacity, seed=evaluation_seed,
            device=device, write_threshold=args.write_threshold,
            noise_scale=args.noise_scale,
            recency_weight=recency_weight,
            frequency_weight=frequency_weight)
        frozen_report = evaluate_frequency_recency(
            frozen, banks=args.test_banks,
            capacity=args.bank_capacity, seed=evaluation_seed,
            device=device, write_threshold=args.write_threshold,
            noise_scale=args.noise_scale,
            recency_weight=recency_weight,
            frequency_weight=frequency_weight)
        endpoint_reports.append({
            "phase": phase,
            "recency_weight": recency_weight,
            "frequency_weight": frequency_weight,
            "frequency_weight_parameter": float(
                model.memory_replacement_extra_gate.weight[
                    0, 0].detach()),
            "learned": learned_report,
            "frozen": frozen_report,
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
    only_residual_changed = (
        changed == ["memory_replacement_extra_gate.weight"])

    def endpoint_pass(endpoint: dict[str, object]) -> bool:
        learned = endpoint["learned"]
        visible = learned["policy_accuracies"]["visible_oracle"]
        return (
            learned["target_eviction_rate"] >= 0.78
            and learned["accuracy"] >= visible - 0.03)

    phase_passes = [
        endpoint_pass(endpoint) for endpoint in endpoint_reports]
    first_target = endpoint_reports[0]["learned"]["target_eviction_rate"]
    middle_target = endpoint_reports[1]["learned"]["target_eviction_rate"]
    return_target = endpoint_reports[2]["learned"]["target_eviction_rate"]
    phase_weights = [
        endpoint["frequency_weight_parameter"]
        for endpoint in endpoint_reports]
    shifted_endpoints = endpoint_reports[:3]
    shifted_improvements = [
        (
            endpoint["learned"]["accuracy"]
            > endpoint["frozen"]["accuracy"]
            and endpoint["learned"]["target_eviction_rate"]
            >= endpoint["frozen"]["target_eviction_rate"] + 0.04)
        for endpoint in shifted_endpoints]
    equal_return_passed = True
    if args.include_equal_return:
        equal_endpoint = endpoint_reports[-1]
        equal_return_passed = (
            equal_endpoint["learned"]["target_eviction_rate"]
            >= equal_endpoint["frozen"]["target_eviction_rate"] - 0.03
            and equal_endpoint["learned"]["gate"]["accepted"])
    gate = {
        "each_phase_within_3_points_of_oracle_and_target_at_least_78":
            all(phase_passes),
        "each_shift_beats_frozen_and_adds_4_target_points":
            all(shifted_improvements),
        "frequency_phase_target_at_least_78":
            middle_target >= 0.78,
        "returned_recency_within_5_points":
            return_target >= first_target - 0.05,
        "coefficient_tracks_both_switches":
            phase_weights[0] > initial_weight
            and phase_weights[1] < phase_weights[0]
            and phase_weights[2] > phase_weights[1],
        "equal_mixture_skill_restored": equal_return_passed,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "only_utility_residual_changed": only_residual_changed,
    }
    gate["accepted"] = all(gate.values()) and not args.shuffle_rewards
    report = {
        "schema": "unified-controller-online-utility-adaptation-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "phases": [
            {
                "name": name,
                "recency_weight": recency,
                "frequency_weight": frequency,
            }
            for name, recency, frequency in phases
        ],
        "boundary_signal_visible_to_learner": False,
        "optimizer_reset_at_boundaries": False,
        "semantic_or_utility_labels_used_for_training": False,
        "training_signal": (
            "symmetric_perturbation_verified_horse_race"
            if args.symmetric_perturbation_update else (
                "within_bank_pairwise_verified_preference"
                if args.pairwise_preference_update
                else "centered_future_verified_success_minus_replacement_cost")),
        "reward_alignment_control": (
            "shuffled_across_banks" if args.shuffle_rewards else "intact"),
        "initial_frequency_weight_parameter": initial_weight,
        "history": history,
        "endpoint_reports": endpoint_reports,
        "accounting": {
            "generated_training_contexts": generated_contexts,
            "future_query_training_bits": future_query_bits,
            "paired_baseline_query_bits": (
                future_query_bits
                if (
                    args.paired_greedy_baseline
                    or args.pairwise_preference_update
                    or args.symmetric_perturbation_update) else 0),
            "unique_training_verifier_bits":
                generated_contexts + future_query_bits * (
                    2 if (
                        args.paired_greedy_baseline
                        or args.pairwise_preference_update
                        or args.symmetric_perturbation_update) else 1),
            "diagnostic_generated_contexts": diagnostic_contexts,
            "optimizer_updates": global_step,
            "replayed_examples": 0,
            "training_and_trace_seconds":
                history[-1]["elapsed_seconds"] if history else 0.0,
        },
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "changed_parameters": changed,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    if gate["accepted"] and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": payload["model_configuration"],
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
                "frequency_weight_parameter":
                    endpoint["frequency_weight_parameter"],
                "accuracy": endpoint["learned"]["accuracy"],
                "target_eviction_rate":
                    endpoint["learned"]["target_eviction_rate"],
                "frozen_accuracy": endpoint["frozen"]["accuracy"],
            }
            for endpoint in endpoint_reports
        ],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
