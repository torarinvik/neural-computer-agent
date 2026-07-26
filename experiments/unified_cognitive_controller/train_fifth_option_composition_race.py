"""Third-generation composition: verified four-action hierarchy vs read five."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .audit_fourth_option_composition import load_router
from .audit_option_composition import load_option
from .probe_requery_operation import ranked_requery_batch
from .train import seed_everything
from .train_fourth_option_composition_race import composed_physical_actions
from .train_option_composition_race import OptionValueHead
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import _load_head


class FlatFiveActionValueHead(nn.Module):
    def __init__(self, input_width: int = 7, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width), nn.Linear(input_width, hidden),
            nn.GELU(), nn.Linear(hidden, 5))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_values(features).argmax(-1)


@torch.no_grad()
def four_action_hierarchy(
        router4: OptionValueHead, option3: OptionValueHead,
        champion: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return composed_physical_actions(
        router4, option3, champion, features)


@torch.no_grad()
def five_action_hierarchy(
        router5: OptionValueHead, router4: OptionValueHead,
        option3: OptionValueHead, champion: nn.Module,
        features: torch.Tensor) -> torch.Tensor:
    width = int(router5.network[0].normalized_shape[0])
    use_fifth = router5(features[:, :width]).bool()
    old = four_action_hierarchy(
        router4, option3, champion, features)
    return torch.where(use_fifth, torch.full_like(old, 4), old)


@torch.no_grad()
def metrics(actions: torch.Tensor, utilities: torch.Tensor) -> dict[str, float]:
    chosen = utilities.gather(1, actions[:, None]).squeeze(1)
    return {
        "verified_utility": float(chosen.mean()),
        "oracle_action_accuracy": float(
            (actions == utilities.argmax(1)).float().mean()),
        "fifth_read_rate": float((actions == 4).float().mean()),
    }


def target_bits(
        rows: list[dict[str, object]], *, stable: bool) -> int | None:
    """Return the first passing bit count, optionally requiring no regression."""
    for index, row in enumerate(rows):
        if not row["reaches_target"]:
            continue
        if not stable or all(
                later["reaches_target"] for later in rows[index:]):
            return int(row["verifier_bits"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--three-option", type=Path, required=True)
    parser.add_argument("--four-router", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8071)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--replay-updates", type=int, default=4)
    parser.add_argument("--router-input-width", type=int, default=7)
    parser.add_argument(
        "--feedback-mode",
        choices=("bandit", "paired-population"),
        default="bandit",
        help=(
            "bandit observes one randomized outcome per context; "
            "paired-population lets temporary clones try every competing "
            "choice and charges every observed outcome as a verifier bit."))
    parser.add_argument(
        "--fifth-train-cost-start", type=float, default=0.04,
        help=(
            "Verifier cost for action five on the first training batch. "
            "It is annealed to the true 0.04 evaluation cost."))
    parser.add_argument("--practical-gain", type=float, default=0.02)
    args = parser.parse_args()
    if args.test_contexts % args.capacity:
        raise ValueError("test contexts must divide by capacity")
    if args.batch_size % args.capacity:
        raise ValueError("batch size must divide by capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    champion = _load_head(args.champion_head, device)
    option3 = load_option(args.three_option, device)
    router4 = load_router(args.four_router, device)
    for module in (champion, option3, router4):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    router5 = OptionValueHead(args.router_input_width, 32).to(device)
    flat = FlatFiveActionValueHead(args.router_input_width, 32).to(device)
    optimizers = {
        "composition": torch.optim.AdamW(
            router5.parameters(), lr=args.learning_rate, weight_decay=1e-4),
        "flat": torch.optim.AdamW(
            flat.parameters(), lr=args.learning_rate, weight_decay=1e-4),
    }
    costs = torch.tensor(
        [0.0, 0.01, 0.02, 0.03, 0.04], device=device)
    test_features, test_outcomes, _ = ranked_requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 80_000_000, device=device, write_threshold=0.5,
        candidate_count=5, include_rank_features=True)
    test_utilities = test_outcomes - costs
    old_actions = four_action_hierarchy(
        router4, option3, champion, test_features)
    old_metrics = metrics(old_actions, test_utilities)
    target = old_metrics["verified_utility"] + args.practical_gain
    oracle = float(test_utilities.max(1).values.mean())
    histories = {"option_composition": [], "flat_reset": []}

    bits_per_context = {
        "option_composition": (
            2 if args.feedback_mode == "paired-population" else 1),
        "flat_reset": (
            5 if args.feedback_mode == "paired-population" else 1),
    }

    def record(step: int) -> None:
        actions = {
            "option_composition": five_action_hierarchy(
                router5, router4, option3, champion, test_features),
            "flat_reset": flat(
                test_features[:, :args.router_input_width]),
        }
        for name, action in actions.items():
            row = metrics(action, test_utilities)
            row.update({
                "step": step,
                "verifier_bits": (
                    step * args.batch_size * bits_per_context[name]),
                "reaches_target": row["verified_utility"] >= target,
            })
            histories[name].append(row)

    record(0)
    generators = {
        name: torch.Generator(device=device).manual_seed(args.seed + offset)
        for name, offset in {
            "composition_action": 70_000_000,
            "flat_action": 70_000_000,
            "composition_replay": 71_000_000,
            "flat_replay": 72_000_000,
        }.items()
    }
    replay: dict[str, list[tuple[torch.Tensor, ...]]] = {
        "composition": [], "flat": []}
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        features, outcomes, _ = ranked_requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=0.5, candidate_count=5,
            include_rank_features=True)
        train_costs = costs.clone()
        curriculum_fraction = (
            (step - 1) / max(1, args.steps - 1))
        train_costs[4] = (
            args.fifth_train_cost_start
            + curriculum_fraction * (float(costs[4]) -
                                     args.fifth_train_cost_start))
        utilities = outcomes - train_costs
        old = four_action_hierarchy(
            router4, option3, champion, features)
        if args.feedback_mode == "paired-population":
            old_observed = utilities.gather(
                1, old[:, None]).squeeze(1)
            fifth_observed = utilities[:, 4]
            replay["composition"].append((
                features[:, :args.router_input_width].detach(),
                (fifth_observed - old_observed).detach(),
                torch.empty(0, device=device)))
            replay["flat"].append((
                features[:, :args.router_input_width].detach(),
                utilities.detach(),
                torch.empty(0, device=device)))
        else:
            attempted_option = torch.randint(
                0, 2, (args.batch_size,),
                generator=generators["composition_action"], device=device)
            attempted_physical = torch.where(
                attempted_option.bool(), torch.full_like(old, 4), old)
            observed = utilities.gather(
                1, attempted_physical[:, None]).squeeze(1)
            replay["composition"].append((
                features[:, :args.router_input_width].detach(),
                attempted_option.detach(),
                observed.detach()))
            attempted_flat = torch.randint(
                0, 5, (args.batch_size,),
                generator=generators["flat_action"], device=device)
            flat_observed = utilities.gather(
                1, attempted_flat[:, None]).squeeze(1)
            replay["flat"].append((
                features[:, :args.router_input_width].detach(),
                attempted_flat.detach(),
                flat_observed.detach()))
        for name, head in (("composition", router5), ("flat", flat)):
            all_features = torch.cat([row[0] for row in replay[name]])
            all_actions = torch.cat([row[1] for row in replay[name]])
            all_outcomes = (
                torch.cat([row[2] for row in replay[name]])
                if args.feedback_mode == "bandit" else None)
            for _ in range(args.replay_updates):
                indices = torch.randint(
                    0, all_features.shape[0], (args.batch_size,),
                    generator=generators[f"{name}_replay"], device=device)
                q_values = head.q_values(all_features[indices])
                if args.feedback_mode == "paired-population":
                    if name == "composition":
                        predicted_advantage = (
                            q_values[:, 1] - q_values[:, 0])
                        loss = nn.functional.smooth_l1_loss(
                            predicted_advantage, all_actions[indices])
                    else:
                        loss = nn.functional.smooth_l1_loss(
                            q_values, all_actions[indices])
                else:
                    prediction = q_values.gather(
                        1, all_actions[indices, None]).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(
                        prediction, all_outcomes[indices])
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizers[name].step()
        if step % 2 == 0 or step == args.steps:
            record(step)

    first_target = {}
    stable_target = {}
    for name, rows in histories.items():
        first_target[name] = target_bits(rows, stable=False)
        stable_target[name] = target_bits(rows, stable=True)
    composed_bits = stable_target["option_composition"]
    flat_bits = stable_target["flat_reset"]
    gate = {
        "oracle_headroom": oracle >= target + 0.02,
        "composition_reaches_target": composed_bits is not None,
        "composition_reaches_before_flat": (
            composed_bits is not None
            and (flat_bits is None or composed_bits < flat_bits)),
        "composition_final_retains_gain": (
            histories["option_composition"][-1]["verified_utility"]
            >= target),
    }
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "fifth-option-composition-race-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "three_option": str(args.three_option),
            "four_router": str(args.four_router),
            "report": str(args.report),
            "checkpoint": (
                str(args.checkpoint) if args.checkpoint else None),
        },
        "old_option": old_metrics,
        "target_utility": target,
        "oracle_utility": oracle,
        "first_target_bits": first_target,
        "stable_target_bits": stable_target,
        "histories": histories,
        "gate": gate,
        "accounting": {
            "verifier_bits_per_arm": {
                name: args.steps * args.batch_size * multiplier
                for name, multiplier in bits_per_context.items()
            },
            "replay_optimizer_updates_per_arm":
                args.steps * args.replay_updates,
            "replayed_examples_per_arm":
                args.steps * args.replay_updates * args.batch_size,
            "unique_logical_lifetimes_per_arm":
                args.steps * args.batch_size,
            "stable_transfer_ratio_flat_over_composition": (
                flat_bits / composed_bits
                if flat_bits is not None and composed_bits is not None
                else None),
            "wall_seconds": time.perf_counter() - started,
            "latency_seconds_per_logical_lifetime": (
                (time.perf_counter() - started)
                / (2 * args.steps * args.batch_size)),
        },
    }
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "fifth-option-router-v1",
            "input_width": args.router_input_width,
            "hidden": router5.network[1].out_features,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in router5.state_dict().items()},
            "four_router": str(args.four_router),
            "training_seed": args.seed,
            "verifier_bits": (
                args.steps * args.batch_size
                * bits_per_context["option_composition"]),
            "replay_updates": args.replay_updates,
            "feedback_mode": args.feedback_mode,
        }, args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
