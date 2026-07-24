"""Pixel-space missing-evidence audit for the success-model milestone."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from .train import seed_everything
from .train_action_conditioned_success import (
    NonlinearActionHead,
    evaluate_action_head,
    frozen_final_states,
    reversal_audit,
)
from .train_fixed_reward_replay_sweep import (
    _fit_head,
    uniform_logged_buffer,
)
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


def ablate_policy_frames(frames: torch.Tensor, mode: str) -> torch.Tensor:
    """Remove real pixel evidence without manufacturing hidden states."""
    result = frames.clone()
    if mode == "normal":
        return result
    if mode == "no_feedback":
        result[:, 2].zero_()
    elif mode == "no_order":
        result[:, :2].zero_()
    elif mode == "feedback_only":
        keep = result[:, 2].clone()
        result.zero_()
        result[:, 2] = keep
    elif mode == "order_only":
        keep = result[:, :2].clone()
        result.zero_()
        result[:, :2] = keep
    elif mode == "support_only":
        result[:, 3:].zero_()
    elif mode == "query_only":
        result[:, :3].zero_()
    else:
        raise ValueError(f"unknown ablation {mode!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pretrain-lifetimes", type=int, default=252)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--policy-lifetimes", type=int, default=510)
    parser.add_argument("--policy-test-lifetimes", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--fit-updates", type=int, default=200)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
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
    train_frames, train_rules = policy_sequences(
        POLICY_TRAIN_START, args.policy_lifetimes, heldout=False,
        palettes=TRAIN_PALETTES)
    test_frames, test_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES)
    reversed_frames, reversed_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES, reverse_events=True)
    support_reversed_frames, support_reversed_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES, reverse_support_only=True)
    query_reversed_frames, query_reversed_rules = policy_sequences(
        POLICY_TEST_START, args.policy_test_lifetimes, heldout=True,
        palettes=TEST_PALETTES, reverse_query_only=True)

    seed_everything(args.seed)
    agent = PredictiveStateAgent(args.hidden).to(device)
    _, pretraining = pretrain(
        agent, pretrain_frames, steps=args.pretrain_steps,
        batch_size=args.batch_size,
        learning_rate=args.pretrain_learning_rate,
        shuffled=False, objective="standardized",
        variance_weight=args.variance_weight,
        correlation_weight=args.correlation_weight,
        target_kind="delta", seed=args.seed, device=device)
    train_states = frozen_final_states(
        agent, train_frames, args.batch_size, device)
    train_states, ordered_rules, actions, rewards, propensities = (
        uniform_logged_buffer(
            train_states, train_rules, seed=args.seed + 300))

    seed_everything(args.seed + 700)
    initial_head = NonlinearActionHead(args.hidden).to(device)
    initial_head_state = copy.deepcopy(initial_head.state_dict())
    head, accounting = _fit_head(
        initial_head_state, train_states, actions, rewards,
        hidden=args.hidden, updates=args.fit_updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate, mode="success",
        seed=args.seed + 800 + args.policy_lifetimes)

    audits: dict[str, dict[str, float]] = {}
    states_by_mode = {}
    for mode in (
            "normal", "no_feedback", "no_order", "feedback_only",
            "order_only", "support_only", "query_only"):
        states = frozen_final_states(
            agent, ablate_policy_frames(test_frames, mode),
            args.batch_size, device)
        states_by_mode[mode] = states
        audits[mode] = evaluate_action_head(
            head, states, test_rules, args.batch_size)
    reversed_states = frozen_final_states(
        agent, reversed_frames, args.batch_size, device)
    reversal = reversal_audit(
        head, states_by_mode["normal"], test_rules,
        reversed_states, reversed_rules, args.batch_size)
    support_reversed_states = frozen_final_states(
        agent, support_reversed_frames, args.batch_size, device)
    support_only_reversal = reversal_audit(
        head, states_by_mode["normal"], test_rules,
        support_reversed_states, support_reversed_rules, args.batch_size)
    query_reversed_states = frozen_final_states(
        agent, query_reversed_frames, args.batch_size, device)
    query_only_reversal = reversal_audit(
        head, states_by_mode["normal"], test_rules,
        query_reversed_states, query_reversed_rules, args.batch_size)

    normal_accuracy = audits["normal"]["verified_accuracy"]
    normal_entropy = audits["normal"]["mean_action_entropy"]
    decisive_modes = ("no_feedback", "no_order")
    accuracy_drops = {
        mode: normal_accuracy - audits[mode]["verified_accuracy"]
        for mode in decisive_modes}
    entropy_increases = {
        mode: audits[mode]["mean_action_entropy"] - normal_entropy
        for mode in decisive_modes}
    gate = {
        "grounded_evidence_dependence": all(
            accuracy_drops[mode] >= 0.15 for mode in decisive_modes),
        "missing_evidence_uncertainty": all(
            entropy_increases[mode] >= 0.05 for mode in decisive_modes),
        "reversal_causality": (
            reversal["reversed_relabeled_accuracy"] >= 0.60
            and reversal["prediction_flip_rate"] >= 0.50),
        "support_binding_causality": (
            support_only_reversal["reversed_relabeled_accuracy"] >= 0.60
            and support_only_reversal["prediction_flip_rate"] >= 0.50),
        "query_invariance": (
            query_only_reversal["reversed_relabeled_accuracy"] >= 0.60
            and query_only_reversal["prediction_flip_rate"] <= 0.25),
    }
    gate["passes_all"] = bool(all(gate.values()))
    report = {
        "schema": "success-missing-evidence-audit-v1",
        "learner_training_inputs": [
            "rendered_rgb_stream", "frozen_recurrent_latent",
            "attempted_action", "logging_propensity_0.5",
            "scalar_verifier_reward"],
        "audit_interventions": (
            "pixel-space frame removal only; no hidden-state swaps"),
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "configuration": vars(args) | {"report": str(args.report)},
        "pretraining": pretraining,
        "training_accounting": accounting | {
            "logging_mean_propensity": float(propensities.mean()),
            "logged_reward_rate": float(rewards.mean()),
            "ordered_private_rules_used_only_by_verifier": int(
                ordered_rules.numel()),
        },
        "audits": audits,
        "accuracy_drops": accuracy_drops,
        "entropy_increases": entropy_increases,
        "reversal": reversal,
        "support_only_reversal": support_only_reversal,
        "query_only_reversal": query_only_reversal,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "audits": audits,
        "accuracy_drops": accuracy_drops,
        "entropy_increases": entropy_increases,
        "reversal": reversal,
        "support_only_reversal": support_only_reversal,
        "query_only_reversal": query_only_reversal,
        "gate": gate,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
