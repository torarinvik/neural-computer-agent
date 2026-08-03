"""Train explicit compute cost on reads, then transfer the trunk to re-query."""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .probe_requery_operation import requery_batch
from .train import evaluate, seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_shadow_compute_advantage import attempted_advantage_target
from .train_shadow_compute_critic import _logged_batch
from .train_thought_compute_transfer import _metrics


class CostAwareComputeValue(nn.Module):
    """Shared generic evidence trunk with explicit cost and action adapters."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.evidence_norm = nn.LayerNorm(4)
        self.hidden = nn.Linear(5, hidden)
        self.activation = nn.GELU()
        self.adapters = nn.ModuleDict({
            "read": nn.Linear(hidden, 1),
            "requery": nn.Linear(hidden, 1),
        })
        for adapter in self.adapters.values():
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)

    def forward(
            self, evidence: torch.Tensor, cost: torch.Tensor,
            operation: str) -> torch.Tensor:
        normalized = self.evidence_norm(evidence)
        latent = self.activation(self.hidden(torch.cat((
            normalized, cost.reshape(-1, 1)), dim=-1)))
        return self.adapters[operation](latent).squeeze(-1)


def initialize_from_four_feature(
        model: CostAwareComputeValue, payload: dict) -> None:
    state = payload["head_state_dict"]
    model.evidence_norm.load_state_dict({
        "weight": state["network.0.weight"],
        "bias": state["network.0.bias"],
    })
    with torch.no_grad():
        model.hidden.weight[:, :4].copy_(state["network.1.weight"])
        model.hidden.weight[:, 4].zero_()
        model.hidden.bias.copy_(state["network.1.bias"])
    model.adapters["read"].load_state_dict({
        "weight": state["network.3.weight"],
        "bias": state["network.3.bias"],
    })


class RequeryView(nn.Module):
    def __init__(self, model: CostAwareComputeValue, cost: float) -> None:
        super().__init__()
        self.model = model
        self.cost = cost

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        cost = torch.full(
            (evidence.shape[0],), self.cost / 0.30,
            device=evidence.device, dtype=evidence.dtype)
        return self.model(evidence, cost, "requery")


def _stable_bits(history: list[dict[str, float]]) -> int | None:
    def passes(row: dict[str, float]) -> bool:
        return (
            row["compute_choice_accuracy"] >= 0.65
            and row["verified_utility"]
            >= row["strongest_fixed_utility"] + 0.03
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
    parser.add_argument("--seed", type=int, default=7901)
    parser.add_argument("--source-steps", type=int, default=12)
    parser.add_argument("--target-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--requery-cost", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=0.003)
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
    ancestor = CostAwareComputeValue(hidden).to(device)
    initialize_from_four_feature(ancestor, payload)
    source_trained = copy.deepcopy(ancestor)
    source_cost_shuffled = copy.deepcopy(ancestor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 10_000)
        reset = CostAwareComputeValue(hidden).to(device)

    source_models = {
        "source_trained": source_trained,
        "source_cost_shuffled": source_cost_shuffled,
    }
    source_optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, model in source_models.items()}
    source_gradient_norms = {name: [] for name in source_models}
    source_sums = {name: 0.0 for name in source_models}
    source_counts = {name: 0 for name in source_models}
    action_generator = torch.Generator(device=device).manual_seed(
        args.seed + 60_000_000)
    cost_generator = torch.Generator(device=device).manual_seed(
        args.seed + 61_000_000)
    cost_choices = torch.tensor(
        [0.01, 0.08, 0.16, 0.24],
        device=device, dtype=torch.float32)
    started = time.perf_counter()

    for step in range(1, args.source_steps + 1):
        evidence, actions, outcomes, _, _ = _logged_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step,
            device=device, write_threshold=0.5)
        cost_indices = torch.randint(
            0, len(cost_choices), (args.batch_size,),
            generator=cost_generator, device=device)
        costs = cost_choices[cost_indices]
        shuffled_costs = costs.roll(1)
        utility = outcomes - costs * actions
        for name, model in source_models.items():
            visible_costs = (
                shuffled_costs
                if name == "source_cost_shuffled" else costs)
            source_sums[name] += float(utility.sum())
            source_counts[name] += utility.numel()
            targets = attempted_advantage_target(
                actions, utility,
                baseline=source_sums[name] / source_counts[name],
                propensity=0.5)
            loss = nn.functional.smooth_l1_loss(
                model(evidence, visible_costs / 0.30, "read"), targets)
            source_optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            source_gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            source_optimizers[name].step()

    source_cost_telemetry = {}
    with torch.no_grad():
        for name, model in source_models.items():
            zeros = torch.zeros(args.batch_size, device=device)
            ones = torch.ones(args.batch_size, device=device)
            source_cost_telemetry[name] = {
                "cost_column_l2": float(
                    model.hidden.weight[:, 4].norm()),
                "mean_absolute_read_prediction_change_cost_0_to_0_30":
                    float((
                        model(evidence, zeros, "read")
                        - model(evidence, ones, "read")
                    ).abs().mean()),
            }

    # A new operation never inherits the old action adapter.
    for model in (source_trained, source_cost_shuffled, ancestor):
        nn.init.zeros_(model.adapters["requery"].weight)
        nn.init.zeros_(model.adapters["requery"].bias)
    target_models = {
        "cost_aware": source_trained,
        "source_cost_shuffled": source_cost_shuffled,
        "ancestor": ancestor,
        "reset": reset,
        "reward_shuffled": copy.deepcopy(source_trained),
        "feature_shuffled": copy.deepcopy(source_trained),
        "missing_evidence": copy.deepcopy(source_trained),
    }
    target_optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, model in target_models.items()}
    test_evidence, test_first, test_second, _ = requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=0.5)
    histories = {name: [] for name in target_models}
    target_gradient_norms = {name: [] for name in target_models}
    target_sum = 0.0
    target_count = 0

    def record(step: int) -> None:
        reverse = torch.arange(
            args.test_contexts - 1, -1, -1, device=device)
        for name, model in target_models.items():
            evidence = test_evidence
            if name == "missing_evidence":
                evidence = torch.zeros_like(evidence)
            elif name == "feature_shuffled":
                evidence = evidence[reverse]
            histories[name].append({
                "step": step,
                "unique_verifier_bits": step * args.batch_size,
                **_metrics(
                    RequeryView(model, args.requery_cost),
                    evidence, test_first, test_second,
                    thought_cost=args.requery_cost),
            })

    record(0)
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 71_000_000)
    feature_generator = torch.Generator(device=device).manual_seed(
        args.seed + 72_000_000)
    for step in range(1, args.target_steps + 1):
        evidence, first, second, _ = requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 2_000_000 + step,
            device=device, write_threshold=0.5)
        actions = torch.randint(
            0, 2, (args.batch_size,), generator=action_generator,
            device=device)
        outcomes = torch.where(actions.bool(), second, first)
        utility = outcomes - args.requery_cost * actions
        target_sum += float(utility.sum())
        target_count += utility.numel()
        targets = attempted_advantage_target(
            actions, utility, baseline=target_sum / target_count,
            propensity=0.5)
        reward_permutation = torch.randperm(
            args.batch_size, generator=reward_generator, device=device)
        feature_permutation = torch.randperm(
            args.batch_size, generator=feature_generator, device=device)
        costs = torch.full(
            (args.batch_size,), args.requery_cost / 0.30,
            device=device)
        for name, model in target_models.items():
            active = evidence
            if name == "missing_evidence":
                active = torch.zeros_like(active)
            elif name == "feature_shuffled":
                active = active[feature_permutation]
            active_targets = (
                targets[reward_permutation]
                if name == "reward_shuffled" else targets)
            loss = nn.functional.smooth_l1_loss(
                model(active, costs, "requery"), active_targets)
            target_optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            target_gradient_norms[name].append(float(
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
            target_optimizers[name].step()
        if step % args.evaluate_every == 0 or step == args.target_steps:
            record(step)

    final = {name: rows[-1] for name, rows in histories.items()}
    stable = {name: _stable_bits(rows) for name, rows in histories.items()}
    main = final["cost_aware"]
    main_bits = stable["cost_aware"]
    evidence_cost = min(
        main["verified_utility"] - final[name]["verified_utility"]
        for name in ("feature_shuffled", "missing_evidence"))
    gate = {
        "choice_at_least_0_65":
            main["compute_choice_accuracy"] >= 0.65,
        "beats_fixed_by_0_03":
            main["verified_utility"]
            >= main["strongest_fixed_utility"] + 0.03,
        "captures_20_percent_gap":
            main["captured_oracle_gap_fraction"] >= 0.20,
        "stable_before_120_bits":
            main_bits is not None and main_bits < 120,
        "faster_than_ancestor": (
            main_bits is not None and (
                stable["ancestor"] is None
                or main_bits < stable["ancestor"])),
        "faster_than_reset": (
            main_bits is not None and (
                stable["reset"] is None
                or main_bits < stable["reset"])),
        "faster_than_cost_shuffled_source": (
            main_bits is not None and (
                stable["source_cost_shuffled"] is None
                or main_bits < stable["source_cost_shuffled"])),
        "evidence_controls_cost_0_02": evidence_cost >= 0.02,
        "source_gradients_live": all(
            min(values) > 0 for values in source_gradient_norms.values()),
        "target_gradients_live": all(
            min(values) > 0 for values in target_gradient_norms.values()),
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
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cost_aware.pt"
        torch.save(source_trained.state_dict(), path)
        restored = CostAwareComputeValue(hidden).to(device)
        restored.load_state_dict(torch.load(
            path, map_location=device, weights_only=True))
        persistence = torch.equal(
            source_trained(
                test_evidence,
                torch.full(
                    (args.test_contexts,),
                    args.requery_cost / 0.30, device=device),
                "requery"),
            restored(
                test_evidence,
                torch.full(
                    (args.test_contexts,),
                    args.requery_cost / 0.30, device=device),
                "requery"))
    gate["round_trip_exact"] = persistence
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "cost-aware-requery-transfer-v1",
        "configuration": vars(args) | {
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "advantage_checkpoint": str(args.advantage_checkpoint),
            "report": str(args.report),
        },
        "source_costs": cost_choices.tolist(),
        "learner_visible": [
            "four_generic_memory_statistics",
            "normalized_compute_cost", "physical_operation_adapter",
            "attempted_action", "logging_propensity_0_5",
            "attempted_action_scalar_outcome",
        ],
        "hidden_from_learner": [
            "unattempted_outcome", "correct_compute_action",
            "correct_answer", "semantic_task_identity",
        ],
        "histories": histories,
        "stable_target_verifier_bits": stable,
        "final_metrics": final,
        "source_gradient_norms": source_gradient_norms,
        "source_cost_telemetry": source_cost_telemetry,
        "target_gradient_norms": target_gradient_norms,
        "binary_retention": binary,
        "four_rule_retention": four_rule,
        "persistence_exact": persistence,
        "gate": gate,
        "accounting": {
            "source_unique_lifetimes":
                args.source_steps * args.batch_size,
            "source_unique_verifier_bits":
                args.source_steps * args.batch_size,
            "target_unique_lifetimes":
                args.target_steps * args.batch_size,
            "target_unique_verifier_bits":
                args.target_steps * args.batch_size,
            "source_optimizer_updates": args.source_steps,
            "target_optimizer_updates": args.target_steps,
            "replayed_examples": 0,
            "private_target_both_action_bits":
                args.test_contexts * 2,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "stable_target_verifier_bits": stable,
        "final_metrics": final,
        "gate": gate,
        "accounting": report["accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
