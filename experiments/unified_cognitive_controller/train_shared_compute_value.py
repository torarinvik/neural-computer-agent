"""Race a shared compute-value trunk across read and recurrent-thought actions."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .train import evaluate, seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import attempted_advantage_target
from .train_shadow_compute_critic import _logged_batch
from .train_thought_compute_transfer import (
    _metrics,
    balanced_thought_dataset,
)


class SharedComputeValue(nn.Module):
    """One generic latent value trunk with tiny operation-specific outputs."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, hidden),
            nn.GELU(),
        )
        self.adapters = nn.ModuleDict({
            "read": nn.Linear(hidden, 1),
            "thought": nn.Linear(hidden, 1),
        })
        for adapter in self.adapters.values():
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(
            self, features: torch.Tensor, operation: str) -> torch.Tensor:
        return self.adapters[operation](self.trunk(features)).squeeze(-1)


def initialize_from_advantage(
        model: SharedComputeValue, payload: dict) -> None:
    """Copy the learned generic trunk and source-operation output exactly."""
    state = payload["head_state_dict"]
    model.trunk[0].load_state_dict({
        "weight": state["network.0.weight"],
        "bias": state["network.0.bias"],
    })
    model.trunk[1].load_state_dict({
        "weight": state["network.1.weight"],
        "bias": state["network.1.bias"],
    })
    model.adapters["read"].load_state_dict({
        "weight": state["network.3.weight"],
        "bias": state["network.3.bias"],
    })


class OperationView(nn.Module):
    def __init__(self, model: SharedComputeValue, operation: str) -> None:
        super().__init__()
        self.model = model
        self.operation = operation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features, self.operation)


def _stable_bits(history: list[dict[str, float]]) -> int | None:
    def passes(row: dict[str, float]) -> bool:
        return (
            row["compute_choice_accuracy"] >= 0.60
            and row["verified_utility"]
            >= row["strongest_fixed_utility"] + 0.08
            and row["captured_oracle_gap_fraction"] >= 0.15)
    for index, row in enumerate(history):
        if passes(row) and all(passes(later) for later in history[index:]):
            return int(row["operation_verifier_bits"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--advantage-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=7811)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--compute-cost", type=float, default=0.01)
    parser.add_argument("--evaluate-every", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    payload = torch.load(
        args.advantage_checkpoint, map_location=device,
        weights_only=False)
    hidden = int(payload["head_hidden"])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        reset = SharedComputeValue(hidden).to(device)
    inherited = copy.deepcopy(reset)
    initialize_from_advantage(inherited, payload)
    shuffled = copy.deepcopy(inherited)
    missing = copy.deepcopy(inherited)
    models = {
        "inherited_shared": inherited,
        "reset_shared": reset,
        "reward_shuffled": shuffled,
        "missing_evidence": missing,
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, model in models.items()}

    thought_train = balanced_thought_dataset(
        controller, selected_count=args.steps * args.batch_size,
        seed=args.seed * 1_000_000, device=device)
    thought_test = balanced_thought_dataset(
        controller, selected_count=args.test_contexts,
        seed=args.seed + 90_000_000, device=device)
    read_test, _, _, read_no, read_yes = _logged_batch(
        controller, count=255, capacity=3,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=0.5)

    histories = {
        name: {"read": [], "thought": []} for name in models}
    gradient_norms = {name: [] for name in models}
    baseline_sums = {"read": 0.0, "thought": 0.0}
    baseline_counts = {"read": 0, "thought": 0}
    started = time.perf_counter()

    def operation_metrics(
            model: SharedComputeValue, operation: str,
            features: torch.Tensor, first: torch.Tensor,
            second: torch.Tensor) -> dict[str, float]:
        return _metrics(
            OperationView(model, operation), features, first, second,
            thought_cost=args.compute_cost)

    def record(step: int) -> None:
        for name, model in models.items():
            thought_features = (
                torch.zeros_like(thought_test["features"])
                if name == "missing_evidence"
                else thought_test["features"])
            read_features = (
                torch.zeros_like(read_test)
                if name == "missing_evidence" else read_test)
            histories[name]["thought"].append({
                "step": step,
                "operation_verifier_bits": step * args.batch_size,
                **operation_metrics(
                    model, "thought", thought_features,
                    thought_test["immediate_outcomes"],
                    thought_test["thought_outcomes"]),
            })
            histories[name]["read"].append({
                "step": step,
                "operation_verifier_bits": step * args.batch_size,
                **operation_metrics(
                    model, "read", read_features, read_no, read_yes),
            })

    record(0)
    action_generator = torch.Generator(device=device).manual_seed(
        args.seed + 70_000_000)
    shuffle_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    for step in range(args.steps):
        start = step * args.batch_size
        end = start + args.batch_size
        thought_features = thought_train["features"][start:end]
        thought_actions = torch.randint(
            0, 2, (args.batch_size,), generator=action_generator,
            device=device)
        thought_outcomes = torch.where(
            thought_actions.bool(),
            thought_train["thought_outcomes"][start:end],
            thought_train["immediate_outcomes"][start:end])
        read_features, read_actions, read_outcomes, _, _ = _logged_batch(
            controller, count=args.batch_size, capacity=3,
            seed=args.seed * 2_000_000 + step, device=device,
            write_threshold=0.5)
        batches = {
            "read": (read_features, read_actions, read_outcomes),
            "thought": (
                thought_features, thought_actions, thought_outcomes),
        }
        targets = {}
        for operation, (_, actions, outcomes) in batches.items():
            utility = outcomes - args.compute_cost * actions
            baseline_sums[operation] += float(utility.sum())
            baseline_counts[operation] += utility.numel()
            targets[operation] = attempted_advantage_target(
                actions, utility,
                baseline=(
                    baseline_sums[operation]
                    / baseline_counts[operation]),
                propensity=0.5)
        permutation = torch.randperm(
            args.batch_size, generator=shuffle_generator, device=device)
        for name, model in models.items():
            loss = 0.0
            for operation, (features, _, _) in batches.items():
                active = (
                    torch.zeros_like(features)
                    if name == "missing_evidence" else features)
                target = (
                    targets[operation][permutation]
                    if name == "reward_shuffled"
                    else targets[operation])
                loss = loss + nn.functional.smooth_l1_loss(
                    model(active, operation), target)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            optimizers[name].step()
        if (
                (step + 1) % args.evaluate_every == 0
                or step + 1 == args.steps):
            record(step + 1)

    final = {
        name: {
            operation: rows[-1]
            for operation, rows in by_operation.items()}
        for name, by_operation in histories.items()}
    stable = {
        name: {
            operation: _stable_bits(rows)
            for operation, rows in by_operation.items()}
        for name, by_operation in histories.items()}
    inherited_thought = final["inherited_shared"]["thought"]
    reset_thought = final["reset_shared"]["thought"]
    inherited_read = final["inherited_shared"]["read"]
    control_cost = min(
        inherited_thought["verified_utility"]
        - final[name]["thought"]["verified_utility"]
        for name in ("reward_shuffled", "missing_evidence"))
    gate = {
        "thought_choice_at_least_0_60":
            inherited_thought["compute_choice_accuracy"] >= 0.60,
        "thought_beats_fixed_by_0_08":
            inherited_thought["verified_utility"]
            >= inherited_thought["strongest_fixed_utility"] + 0.08,
        "thought_captures_15_percent_gap":
            inherited_thought["captured_oracle_gap_fraction"] >= 0.15,
        "inherited_thought_faster_than_reset": (
            stable["inherited_shared"]["thought"] is not None
            and (
                stable["reset_shared"]["thought"] is None
                or stable["inherited_shared"]["thought"]
                < stable["reset_shared"]["thought"])),
        "read_remains_useful":
            inherited_read["verified_utility"]
            >= inherited_read["strongest_fixed_utility"] + 0.05,
        "controls_cost_at_least_0_05": control_cost >= 0.05,
        "all_gradients_live": all(
            min(values) > 0 for values in gradient_norms.values()),
    }
    binary = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 93_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        controller, count=128, trials=6,
        seed=args.seed + 94_000_000, device=device,
        task="four_rule", feedback_trials=2)
    gate["binary_retained"] = binary["gate"]["accepted"]
    gate["four_rule_retained"] = four_rule["gate"]["accepted"]
    gate["accepted_for_replication"] = all(gate.values())

    persistence = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, model in models.items():
            path = Path(directory) / f"{name}.pt"
            torch.save(model.state_dict(), path)
            restored = SharedComputeValue(hidden).to(device)
            restored.load_state_dict(torch.load(
                path, map_location=device, weights_only=True))
            persistence[name] = torch.equal(
                model(thought_test["features"], "thought"),
                restored(thought_test["features"], "thought"))
    gate["all_round_trips_exact"] = all(persistence.values())
    gate["accepted_for_replication"] = all(gate.values())

    report = {
        "schema": "shared-compute-value-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "advantage_checkpoint": str(args.advantage_checkpoint),
            "report": str(args.report),
        },
        "learner_visible": [
            "four_generic_controller_statistics",
            "available_opaque_compute_operation_interface",
            "attempted_action", "logging_propensity_0_5",
            "attempted_action_scalar_outcome", "compute_cost",
        ],
        "hidden_from_learner": [
            "unattempted_outcome", "correct_compute_action",
            "help_or_harm_label", "correct_answer",
            "semantic_cognitive_task_identity",
        ],
        "histories": histories,
        "stable_operation_verifier_bits": stable,
        "final_metrics": final,
        "gradient_norms": gradient_norms,
        "persistence_exact": persistence,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "gate": gate,
        "accounting": {
            "learner_visible_unique_lifetimes":
                args.steps * args.batch_size * 2,
            "learner_visible_unique_verifier_bits":
                args.steps * args.batch_size * 2,
            "bits_per_operation": args.steps * args.batch_size,
            "private_thought_screening_lifetimes":
                thought_train["screened_logical_lifetimes"]
                + thought_test["screened_logical_lifetimes"],
            "private_thought_screening_verifier_bits":
                thought_train["private_screening_verifier_bits"]
                + thought_test["private_screening_verifier_bits"],
            "optimizer_updates_per_arm": args.steps,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "stable_operation_verifier_bits": stable,
        "final_metrics": final,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
