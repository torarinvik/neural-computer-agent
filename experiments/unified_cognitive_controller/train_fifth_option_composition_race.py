"""Third-generation composition: verified four-action hierarchy vs read five."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn

from .audit_fourth_option_composition import load_router
from .audit_option_composition import load_option
from .probe_requery_operation import ranked_requery_batch
from .replay_stopping_probe import load_probe, predict_replay_benefit
from .train import seed_everything
from .train_fourth_option_composition_race import composed_physical_actions
from .train_option_composition_race import OptionValueHead
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import _load_head


class FlatFiveActionValueHead(nn.Module):
    def __init__(
            self, input_width: int = 7, hidden: int = 32,
            actions: int = 5) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width), nn.Linear(input_width, hidden),
            nn.GELU(), nn.Linear(hidden, actions))
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
def metrics(
        actions: torch.Tensor, utilities: torch.Tensor,
        new_action: int = 4) -> dict[str, float]:
    chosen = utilities.gather(1, actions[:, None]).squeeze(1)
    return {
        "verified_utility": float(chosen.mean()),
        "oracle_action_accuracy": float(
            (actions == utilities.argmax(1)).float().mean()),
        "new_action_rate": float((actions == new_action).float().mean()),
        "fifth_read_rate": float((actions == new_action).float().mean()),
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
    parser.add_argument(
        "--fifth-router", type=Path,
        help="Verified five-action router; enables a six-action race.")
    parser.add_argument(
        "--candidate-count", type=int, choices=(5, 6), default=5)
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
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help="Optional task-agnostic weight averaging for both race arms.")
    parser.add_argument(
        "--adaptive-replay-loss", type=float,
        help=(
            "If set, treat --replay-updates as a maximum and stop replay "
            "once full observed-experience loss is at or below this value."))
    parser.add_argument(
        "--record-replay-trace", action="store_true",
        help=(
            "Record task-agnostic state before and verified loss reduction "
            "after every replay update for learned-stopping diagnostics."))
    parser.add_argument(
        "--replay-stopper-checkpoint", type=Path, action="append",
        help=(
            "Frozen cross-generation predictor of marginal replay value. "
            "May be repeated; an ensemble stops only on unanimous evidence."))
    parser.add_argument(
        "--replay-compute-cost", type=float, default=0.0,
        help=(
            "Generic loss-equivalent price of one replay update; the learned "
            "stopper continues only when predicted benefit exceeds it."))
    parser.add_argument(
        "--replay-stopper-aggregation",
        choices=("unanimous", "mean"), default="unanimous",
        help=(
            "How an ensemble combines benefit predictions. Unanimous uses "
            "the conservative maximum; mean uses their arithmetic mean."))
    parser.add_argument(
        "--replay-min-updates", type=int, default=1,
        help="Minimum composition updates per new experience batch.")
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
        "--fifth-train-cost-start", type=float,
        help=(
            "Optional verifier cost for the new action on the first training "
            "batch. It is annealed to the true evaluation cost."))
    parser.add_argument("--practical-gain", type=float, default=0.02)
    args = parser.parse_args()
    if args.candidate_count == 6 and args.fifth_router is None:
        raise ValueError("six-action mode requires --fifth-router")
    if args.candidate_count == 5 and args.fifth_router is not None:
        raise ValueError("--fifth-router is only valid in six-action mode")
    if (args.replay_stopper_checkpoint is not None
            and args.adaptive_replay_loss is not None):
        raise ValueError(
            "learned replay stopping and fixed loss stopping are exclusive")
    if args.replay_min_updates < 1:
        raise ValueError("replay minimum updates must be positive")
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
    router5_parent = None
    if args.fifth_router is not None:
        from .audit_fifth_option_composition import load_fifth_router
        router5_parent = load_fifth_router(args.fifth_router, device)
    inherited_modules = [champion, option3, router4]
    if router5_parent is not None:
        inherited_modules.append(router5_parent)
    for module in inherited_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    new_router = OptionValueHead(args.router_input_width, 32).to(device)
    flat = FlatFiveActionValueHead(
        args.router_input_width, 32, args.candidate_count).to(device)
    replay_stoppers = []
    replay_stopper_normalizations = []
    if args.replay_stopper_checkpoint is not None:
        for path in args.replay_stopper_checkpoint:
            model, normalization = load_probe(path, device)
            replay_stoppers.append(model)
            replay_stopper_normalizations.append(normalization)
    replay_stopper_horizons = {
        int(row["target_horizon"])
        for row in replay_stopper_normalizations}
    if len(replay_stopper_horizons) > 1:
        raise ValueError("replay-stopper ensemble horizons must match")
    replay_stopper_horizon = (
        next(iter(replay_stopper_horizons))
        if replay_stopper_horizons else 1)
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    ema_router = copy.deepcopy(new_router)
    ema_flat = copy.deepcopy(flat)
    for module in (ema_router, ema_flat):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    optimizers = {
        "composition": torch.optim.AdamW(
            new_router.parameters(), lr=args.learning_rate,
            weight_decay=1e-4),
        "flat": torch.optim.AdamW(
            flat.parameters(), lr=args.learning_rate, weight_decay=1e-4),
    }
    costs = torch.arange(
        args.candidate_count, device=device, dtype=torch.float32) * 0.01
    new_action = args.candidate_count - 1
    train_cost_start = (
        float(costs[new_action])
        if args.fifth_train_cost_start is None
        else args.fifth_train_cost_start)

    @torch.no_grad()
    def inherited_actions(features: torch.Tensor) -> torch.Tensor:
        if router5_parent is None:
            return four_action_hierarchy(
                router4, option3, champion, features)
        return five_action_hierarchy(
            router5_parent, router4, option3, champion, features)

    @torch.no_grad()
    def composed_actions(
            features: torch.Tensor,
            router: OptionValueHead | None = None) -> torch.Tensor:
        selected_router = new_router if router is None else router
        width = int(selected_router.network[0].normalized_shape[0])
        use_new = selected_router(features[:, :width]).bool()
        old = inherited_actions(features)
        return torch.where(
            use_new, torch.full_like(old, new_action), old)

    test_features, test_outcomes, _ = ranked_requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 80_000_000, device=device, write_threshold=0.5,
        candidate_count=args.candidate_count, include_rank_features=True)
    test_utilities = test_outcomes - costs
    old_actions = inherited_actions(test_features)
    old_metrics = metrics(old_actions, test_utilities, new_action)
    target = old_metrics["verified_utility"] + args.practical_gain
    oracle = float(test_utilities.max(1).values.mean())
    histories = {"option_composition": [], "flat_reset": []}

    bits_per_context = {
        "option_composition": (
            2 if args.feedback_mode == "paired-population" else 1),
        "flat_reset": (
            args.candidate_count
            if args.feedback_mode == "paired-population" else 1),
    }

    def record(step: int) -> None:
        eval_router = (
            ema_router if args.ema_decay > 0 else new_router)
        eval_flat = ema_flat if args.ema_decay > 0 else flat
        actions = {
            "option_composition": composed_actions(
                test_features, eval_router),
            "flat_reset": eval_flat(
                test_features[:, :args.router_input_width]),
        }
        for name, action in actions.items():
            row = metrics(action, test_utilities, new_action)
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
    optimizer_updates = {"composition": 0, "flat": 0}
    replayed_examples = {"composition": 0, "flat": 0}
    final_replay_loss: dict[str, float | None] = {
        "composition": None, "flat": None}
    replay_trace: list[dict[str, int | float | str]] = []
    previous_loss_reduction = {"composition": 0.0, "flat": 0.0}
    previous_gradient_norm = {"composition": 0.0, "flat": 0.0}
    predicted_replay_stops = {"composition": 0, "flat": 0}
    predicted_replay_benefits: dict[str, list[float]] = {
        "composition": [], "flat": []}

    def replay_loss(
            name: str, head: nn.Module, features: torch.Tensor,
            actions: torch.Tensor,
            outcomes: torch.Tensor | None) -> torch.Tensor:
        q_values = head.q_values(features)
        if args.feedback_mode == "paired-population":
            if name == "composition":
                prediction = q_values[:, 1] - q_values[:, 0]
                return nn.functional.smooth_l1_loss(
                    prediction, actions)
            return nn.functional.smooth_l1_loss(q_values, actions)
        prediction = q_values.gather(
            1, actions[:, None]).squeeze(1)
        assert outcomes is not None
        return nn.functional.smooth_l1_loss(prediction, outcomes)

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        features, outcomes, _ = ranked_requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=0.5, candidate_count=args.candidate_count,
            include_rank_features=True)
        train_costs = costs.clone()
        curriculum_fraction = (
            (step - 1) / max(1, args.steps - 1))
        train_costs[new_action] = (
            train_cost_start
            + curriculum_fraction * (float(costs[new_action]) -
                                     train_cost_start))
        utilities = outcomes - train_costs
        old = inherited_actions(features)
        if args.feedback_mode == "paired-population":
            old_observed = utilities.gather(
                1, old[:, None]).squeeze(1)
            fifth_observed = utilities[:, new_action]
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
                attempted_option.bool(),
                torch.full_like(old, new_action), old)
            observed = utilities.gather(
                1, attempted_physical[:, None]).squeeze(1)
            replay["composition"].append((
                features[:, :args.router_input_width].detach(),
                attempted_option.detach(),
                observed.detach()))
            attempted_flat = torch.randint(
                0, args.candidate_count, (args.batch_size,),
                generator=generators["flat_action"], device=device)
            flat_observed = utilities.gather(
                1, attempted_flat[:, None]).squeeze(1)
            replay["flat"].append((
                features[:, :args.router_input_width].detach(),
                attempted_flat.detach(),
                flat_observed.detach()))
        for name, head in (("composition", new_router), ("flat", flat)):
            all_features = torch.cat([row[0] for row in replay[name]])
            all_actions = torch.cat([row[1] for row in replay[name]])
            all_outcomes = (
                torch.cat([row[2] for row in replay[name]])
                if args.feedback_mode == "bandit" else None)
            for replay_index in range(args.replay_updates):
                loss_before = None
                learned_stopping = (
                    name == "composition" and bool(replay_stoppers))
                if args.record_replay_trace or learned_stopping:
                    with torch.no_grad():
                        loss_before = float(replay_loss(
                            name, head, all_features, all_actions,
                            all_outcomes))
                if (learned_stopping
                        and replay_index >= args.replay_min_updates
                        and replay_index % replay_stopper_horizon == 0):
                    assert loss_before is not None
                    predictions = [
                        predict_replay_benefit(
                            model, normalization,
                            loss_before=loss_before,
                            previous_loss_reduction=
                                previous_loss_reduction[name],
                            previous_gradient_norm=
                                previous_gradient_norm[name],
                            observed_examples=int(all_features.shape[0]),
                            replay_index=replay_index,
                            replay_updates=args.replay_updates,
                            device=device)
                        for model, normalization in zip(
                            replay_stoppers,
                            replay_stopper_normalizations)
                    ]
                    predicted_benefit = (
                        max(predictions)
                        if args.replay_stopper_aggregation == "unanimous"
                        else sum(predictions) / len(predictions))
                    predicted_replay_benefits[name].append(
                        predicted_benefit)
                    if predicted_benefit <= args.replay_compute_cost:
                        predicted_replay_stops[name] += 1
                        final_replay_loss[name] = loss_before
                        break
                indices = torch.randint(
                    0, all_features.shape[0], (args.batch_size,),
                    generator=generators[f"{name}_replay"], device=device)
                loss = replay_loss(
                    name, head, all_features[indices],
                    all_actions[indices],
                    (all_outcomes[indices]
                     if all_outcomes is not None else None))
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(
                    head.parameters(), 1.0))
                optimizers[name].step()
                optimizer_updates[name] += 1
                replayed_examples[name] += args.batch_size
                if args.ema_decay > 0:
                    ema_head = (
                        ema_router if name == "composition" else ema_flat)
                    with torch.no_grad():
                        for averaged, current in zip(
                                ema_head.parameters(), head.parameters()):
                            averaged.mul_(args.ema_decay).add_(
                                current, alpha=1.0 - args.ema_decay)
                if (args.adaptive_replay_loss is not None
                        or args.record_replay_trace
                        or learned_stopping):
                    with torch.no_grad():
                        observed_loss = replay_loss(
                            name, head, all_features, all_actions,
                            all_outcomes)
                    final_replay_loss[name] = float(observed_loss)
                    if args.record_replay_trace or learned_stopping:
                        assert loss_before is not None
                        reduction = loss_before - float(observed_loss)
                        if args.record_replay_trace:
                            replay_trace.append({
                                "arm": name,
                                "experience_step": step,
                                "replay_index": replay_index,
                                "updates_before":
                                    optimizer_updates[name] - 1,
                                "observed_examples":
                                    int(all_features.shape[0]),
                                "loss_before": loss_before,
                                "previous_loss_reduction":
                                    previous_loss_reduction[name],
                                "previous_gradient_norm":
                                    previous_gradient_norm[name],
                                "loss_after": float(observed_loss),
                                "loss_reduction": reduction,
                            })
                        previous_loss_reduction[name] = reduction
                        previous_gradient_norm[name] = gradient_norm
                    if (args.adaptive_replay_loss is not None
                            and observed_loss <= args.adaptive_replay_loss):
                        break
            if args.adaptive_replay_loss is None:
                with torch.no_grad():
                    final_replay_loss[name] = float(replay_loss(
                        name, head, all_features, all_actions,
                        all_outcomes))
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
        "schema": (
            "sixth-option-composition-race-v1"
            if args.candidate_count == 6
            else "fifth-option-composition-race-v1"),
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "three_option": str(args.three_option),
            "four_router": str(args.four_router),
            "fifth_router": (
                str(args.fifth_router) if args.fifth_router else None),
            "report": str(args.report),
            "checkpoint": (
                str(args.checkpoint) if args.checkpoint else None),
            "replay_stopper_checkpoint": (
                [str(path) for path in args.replay_stopper_checkpoint]
                if args.replay_stopper_checkpoint else None),
        },
        "old_option": old_metrics,
        "target_utility": target,
        "oracle_utility": oracle,
        "first_target_bits": first_target,
        "stable_target_bits": stable_target,
        "histories": histories,
        "replay_trace": replay_trace,
        "gate": gate,
        "accounting": {
            "verifier_bits_per_arm": {
                name: args.steps * args.batch_size * multiplier
                for name, multiplier in bits_per_context.items()
            },
            "replay_optimizer_updates_per_arm": optimizer_updates,
            "replayed_examples_per_arm": replayed_examples,
            "final_observed_replay_loss": final_replay_loss,
            "predicted_replay_stops": predicted_replay_stops,
            "predicted_replay_benefit_mean": {
                name: (
                    sum(values) / len(values) if values else None)
                for name, values in predicted_replay_benefits.items()
            },
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
        saved_router = (
            ema_router if args.ema_decay > 0 else new_router)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": (
                "sixth-option-router-v1"
                if args.candidate_count == 6
                else "fifth-option-router-v1"),
            "input_width": args.router_input_width,
            "hidden": saved_router.network[1].out_features,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in saved_router.state_dict().items()},
            "parent_router": (
                str(args.fifth_router)
                if args.fifth_router else str(args.four_router)),
            "training_seed": args.seed,
            "verifier_bits": (
                args.steps * args.batch_size
                * bits_per_context["option_composition"]),
            "replay_updates": args.replay_updates,
            "actual_optimizer_updates":
                optimizer_updates["composition"],
            "actual_replayed_examples":
                replayed_examples["composition"],
            "adaptive_replay_loss": args.adaptive_replay_loss,
            "replay_stopper_checkpoint": (
                [str(path) for path in args.replay_stopper_checkpoint]
                if args.replay_stopper_checkpoint else None),
            "replay_compute_cost": args.replay_compute_cost,
            "replay_min_updates": args.replay_min_updates,
            "feedback_mode": args.feedback_mode,
            "ema_decay": args.ema_decay,
        }, args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
