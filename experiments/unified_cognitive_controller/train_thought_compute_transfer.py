"""Transfer a learned compute-advantage head to optional recurrent thought."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION, generate_lifetimes
from .train import evaluate, seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import (
    ComputeAdvantageHead,
    attempted_advantage_target,
)


ARM_NAMES = (
    "inherited", "reset", "reward_shuffled",
    "feature_shuffled", "missing_evidence")


@torch.no_grad()
def _thought_chunk(
        model, *, count: int, seed: int, device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = generate_lifetimes(
        count, 3, seed=seed, heldout=True, task="four_rule",
        appearance="dot_pairs", support_trials=2, device=device)
    state = model.initial_state(count, device=device)
    previous_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    previous_reward = torch.zeros(count, device=device)
    for trial in range(2):
        output, state = model.step(
            batch.frames[:, trial], state, previous_action,
            previous_reward,
            torch.full_like(previous_reward, float(trial > 0)))
        previous_action = output.logits.argmax(-1)
        previous_reward = (
            previous_action == batch.correct_actions[:, trial]).float()

    before = state
    immediate, after = model.step(
        batch.frames[:, 2], state, previous_action,
        previous_reward, torch.ones_like(previous_reward))
    probabilities = torch.softmax(immediate.logits, dim=-1)
    ranked = probabilities.topk(2, dim=-1).values
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
    hidden_change = (
        (after.hidden - before.hidden).square().mean(-1).sqrt())
    features = torch.stack((
        ranked[:, 0],
        ranked[:, 0] - ranked[:, 1],
        entropy,
        hidden_change,
    ), dim=-1)
    immediate_action = immediate.logits.argmax(-1)
    immediate_outcome = (
        immediate_action == batch.correct_actions[:, 2]).float()

    null_action = torch.full_like(previous_action, NULL_ACTION)
    zeros = torch.zeros_like(previous_reward)
    thought, _ = model.step(
        batch.frames[:, 2], after, null_action, zeros, zeros)
    thought_action = thought.logits.argmax(-1)
    thought_outcome = (
        thought_action == batch.correct_actions[:, 2]).float()
    return features, immediate_outcome, thought_outcome


@torch.no_grad()
def balanced_thought_dataset(
        model, *, selected_count: int, seed: int,
        device: torch.device, chunk_size: int = 4096,
        max_screened: int = 250_000,
        ) -> dict[str, torch.Tensor | int]:
    """Privately balance cases where thought helps versus harms."""
    if selected_count % 2:
        raise ValueError("selected count must be even")
    needed = selected_count // 2
    helpful: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harmful: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    helpful_count = 0
    harmful_count = 0
    screened = 0
    while (
            (helpful_count < needed or harmful_count < needed)
            and screened < max_screened):
        count = min(chunk_size, max_screened - screened)
        features, immediate, thought = _thought_chunk(
            model, count=count, seed=seed + screened,
            device=device)
        help_mask = thought > immediate
        harm_mask = thought < immediate
        if help_mask.any() and helpful_count < needed:
            take = min(
                needed - helpful_count, int(help_mask.sum()))
            indices = help_mask.nonzero().flatten()[:take]
            helpful.append((
                features[indices], immediate[indices], thought[indices]))
            helpful_count += take
        if harm_mask.any() and harmful_count < needed:
            take = min(
                needed - harmful_count, int(harm_mask.sum()))
            indices = harm_mask.nonzero().flatten()[:take]
            harmful.append((
                features[indices], immediate[indices], thought[indices]))
            harmful_count += take
        screened += count
    if helpful_count < needed or harmful_count < needed:
        raise RuntimeError(
            f"insufficient decision cases after {screened}: "
            f"{helpful_count=} {harmful_count=}")

    features = torch.cat(
        [part[0] for part in helpful + harmful])
    immediate = torch.cat(
        [part[1] for part in helpful + harmful])
    thought = torch.cat(
        [part[2] for part in helpful + harmful])
    generator = torch.Generator(device=device).manual_seed(seed + 88_019)
    permutation = torch.randperm(
        selected_count, generator=generator, device=device)
    return {
        "features": features[permutation],
        "immediate_outcomes": immediate[permutation],
        "thought_outcomes": thought[permutation],
        "screened_logical_lifetimes": screened,
        "private_screening_verifier_bits": screened * 2,
        "selected_helpful": needed,
        "selected_harmful": needed,
    }


def _active_features(
        features: torch.Tensor, arm: str,
        permutation: torch.Tensor | None = None) -> torch.Tensor:
    if arm in ("inherited", "reset", "reward_shuffled"):
        return features
    if arm == "missing_evidence":
        return torch.zeros_like(features)
    if arm == "feature_shuffled":
        if permutation is None:
            raise ValueError("feature shuffle requires a permutation")
        return features[permutation]
    raise ValueError(f"unknown arm: {arm}")


@torch.no_grad()
def _metrics(
        head: ComputeAdvantageHead, features: torch.Tensor,
        immediate: torch.Tensor, thought: torch.Tensor, *,
        thought_cost: float) -> dict[str, float]:
    advantage = head(features)
    chosen = (advantage > 0).long()
    actual = torch.stack((immediate, thought - thought_cost), dim=1)
    oracle = actual.argmax(-1)
    utility = actual.gather(1, chosen[:, None]).squeeze(1)
    fixed = max(
        float(actual[:, 0].mean()), float(actual[:, 1].mean()))
    ceiling = float(actual.max(-1).values.mean())
    achieved = float(utility.mean())
    gap = ceiling - fixed
    return {
        "compute_choice_accuracy": float((chosen == oracle).float().mean()),
        "verified_utility": achieved,
        "always_answer_utility": float(actual[:, 0].mean()),
        "always_think_utility": float(actual[:, 1].mean()),
        "strongest_fixed_utility": fixed,
        "oracle_utility": ceiling,
        "available_oracle_gap": gap,
        "captured_oracle_gap_fraction": (
            (achieved - fixed) / gap if gap > 1e-8 else 0.0),
        "thought_rate": float(chosen.float().mean()),
    }


def _stable_bits(history: list[dict[str, float]]) -> int | None:
    def passes(row: dict[str, float]) -> bool:
        return (
            row["compute_choice_accuracy"] >= 0.65
            and row["verified_utility"]
            >= row["strongest_fixed_utility"] + 0.10
            and row["captured_oracle_gap_fraction"] >= 0.20)
    for index, row in enumerate(history):
        if passes(row) and all(passes(later) for later in history[index:]):
            return int(row["unique_verifier_bits"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--advantage-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7801)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--thought-cost", type=float, default=0.01)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size % 2 or args.test_contexts % 2:
        raise ValueError("balanced datasets require even counts")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    model = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    inherited_payload = torch.load(
        args.advantage_checkpoint, map_location=device,
        weights_only=False)
    if inherited_payload.get("schema") != (
            "shadow-compute-advantage-head-v1"):
        raise ValueError("unsupported advantage checkpoint")
    hidden = int(inherited_payload["head_hidden"])
    inherited = ComputeAdvantageHead(hidden).to(device)
    inherited.load_state_dict(inherited_payload["head_state_dict"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        reset = ComputeAdvantageHead(hidden).to(device)
    heads = {
        "inherited": inherited,
        "reset": reset,
        "reward_shuffled": copy.deepcopy(inherited),
        "feature_shuffled": copy.deepcopy(inherited),
        "missing_evidence": copy.deepcopy(inherited),
    }
    optimizers = {
        name: torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, head in heads.items()}

    started = time.perf_counter()
    train = balanced_thought_dataset(
        model, selected_count=args.steps * args.batch_size,
        seed=args.seed * 1_000_000, device=device)
    test = balanced_thought_dataset(
        model, selected_count=args.test_contexts,
        seed=args.seed + 90_000_000, device=device)
    train_features = train["features"]
    train_immediate = train["immediate_outcomes"]
    train_thought = train["thought_outcomes"]
    test_features = test["features"]
    test_immediate = test["immediate_outcomes"]
    test_thought = test["thought_outcomes"]

    histories = {name: [] for name in ARM_NAMES}
    gradient_norms = {name: [] for name in ARM_NAMES}
    utility_sum = 0.0
    utility_count = 0

    def record(step: int) -> None:
        reverse = torch.arange(
            args.test_contexts - 1, -1, -1, device=device)
        for name, head in heads.items():
            histories[name].append({
                "step": step,
                "unique_verifier_bits": step * args.batch_size,
                **_metrics(
                    head, _active_features(
                        test_features, name, reverse),
                    test_immediate, test_thought,
                    thought_cost=args.thought_cost),
            })

    record(0)
    action_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    feature_generator = torch.Generator(device=device).manual_seed(
        args.seed + 72_000_000)
    for step in range(args.steps):
        start = step * args.batch_size
        end = start + args.batch_size
        features = train_features[start:end]
        immediate = train_immediate[start:end]
        thought = train_thought[start:end]
        actions = torch.randint(
            0, 2, (args.batch_size,), generator=action_generator,
            device=device)
        outcomes = torch.where(actions.bool(), thought, immediate)
        observed_utility = (
            outcomes - args.thought_cost * actions)
        utility_sum += float(observed_utility.sum())
        utility_count += args.batch_size
        targets = attempted_advantage_target(
            actions, observed_utility,
            baseline=utility_sum / utility_count, propensity=0.5)
        reward_permutation = torch.randperm(
            args.batch_size, generator=reward_generator, device=device)
        feature_permutation = torch.randperm(
            args.batch_size, generator=feature_generator, device=device)
        for name, head in heads.items():
            active = _active_features(
                features, name, feature_permutation)
            active_targets = (
                targets[reward_permutation]
                if name == "reward_shuffled" else targets)
            loss = nn.functional.smooth_l1_loss(
                head(active), active_targets)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)))
            optimizers[name].step()
        prefix = step + 1
        if (
                prefix % args.evaluate_every == 0
                or prefix == args.steps):
            record(prefix)

    final = {name: rows[-1] for name, rows in histories.items()}
    stable = {
        name: _stable_bits(rows)
        for name, rows in histories.items()}
    shuffle_generator = torch.Generator(device=device).manual_seed(
        args.seed + 79_000_000)
    shuffled_features = test_features[torch.randperm(
        args.test_contexts, generator=shuffle_generator, device=device)]
    evidence_shuffled = _metrics(
        heads["inherited"], shuffled_features,
        test_immediate, test_thought,
        thought_cost=args.thought_cost)
    inherited_final = final["inherited"]
    control_margin = min(
        inherited_final["verified_utility"]
        - final[name]["verified_utility"]
        for name in (
            "reward_shuffled", "feature_shuffled",
            "missing_evidence"))
    shuffle_cost = (
        inherited_final["verified_utility"]
        - evidence_shuffled["verified_utility"])
    inherited_bits = stable["inherited"]
    reset_bits = stable["reset"]

    persistence = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, head in heads.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(head.state_dict(), path)
            restored = ComputeAdvantageHead(hidden).to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            with torch.no_grad():
                persistence[name] = torch.equal(
                    head(test_features), restored(test_features))
    binary = evaluate(
        model, count=128, trials=6, seed=args.seed + 93_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=128, trials=6, seed=args.seed + 94_000_000,
        device=device, task="four_rule", feedback_trials=2)
    gate = {
        "inherited_final_choice_at_least_0_65":
            inherited_final["compute_choice_accuracy"] >= 0.65,
        "inherited_final_beats_fixed_by_0_10":
            inherited_final["verified_utility"]
            >= inherited_final["strongest_fixed_utility"] + 0.10,
        "inherited_captures_20_percent_gap":
            inherited_final["captured_oracle_gap_fraction"] >= 0.20,
        "inherited_strictly_faster_than_reset": (
            inherited_bits is not None
            and (reset_bits is None or inherited_bits < reset_bits)),
        "causal_controls_cost_at_least_0_05_utility":
            control_margin >= 0.05,
        "evidence_shuffle_costs_at_least_0_05_utility":
            shuffle_cost >= 0.05,
        "all_gradients_live": all(
            min(values) > 0 for values in gradient_norms.values()),
        "all_round_trips_exact": all(persistence.values()),
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
    }
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "thought-compute-transfer-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "advantage_checkpoint": str(args.advantage_checkpoint),
            "report": str(args.report),
        },
        "learner_visible": [
            "immediate_action_confidence",
            "immediate_top_two_margin",
            "immediate_action_entropy",
            "controller_hidden_change_norm",
            "opaque_attempted_compute_action",
            "exact_logging_propensity_0_5",
            "attempted_action_scalar_verified_outcome",
            "generic_normalized_thought_cost",
        ],
        "hidden_from_learner": [
            "unattempted_outcome", "correct_compute_action",
            "help_or_harm_curriculum_label", "semantic_task_identity",
            "correct_answer",
        ],
        "curriculum": {
            "private_selection":
                "equal extra-thought-helps and extra-thought-harms",
            "train": {
                key: value for key, value in train.items()
                if isinstance(value, int)},
            "test": {
                key: value for key, value in test.items()
                if isinstance(value, int)},
        },
        "histories": histories,
        "stable_unique_verifier_bits": stable,
        "final_metrics": final,
        "evidence_shuffled_audit": evidence_shuffled,
        "gradient_norms": gradient_norms,
        "persistence_exact": persistence,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "gate": gate,
        "accounting": {
            "learner_visible_unique_lifetimes":
                args.steps * args.batch_size,
            "learner_visible_unique_verifier_bits":
                args.steps * args.batch_size,
            "private_curriculum_screening_lifetimes":
                train["screened_logical_lifetimes"]
                + test["screened_logical_lifetimes"],
            "private_curriculum_screening_verifier_bits":
                train["private_screening_verifier_bits"]
                + test["private_screening_verifier_bits"],
            "private_test_both_action_bits":
                args.test_contexts * 2,
            "optimizer_updates_per_arm": args.steps,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "stable_unique_verifier_bits": stable,
        "final_metrics": final,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
