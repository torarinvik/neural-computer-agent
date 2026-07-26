"""Second-generation composition: verified three-action option vs read four."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .audit_option_composition import load_option
from .probe_requery_operation import ranked_requery_batch
from .train import seed_everything
from .train_option_composition_race import (
    OptionValueHead,
    option_physical_actions,
)
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import _load_head


class FlatFourActionValueHead(nn.Module):
    def __init__(self, input_width: int = 9, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width), nn.Linear(input_width, hidden),
            nn.GELU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_values(features).argmax(-1)


@torch.no_grad()
def previous_option_actions(
        previous: OptionValueHead, champion: nn.Module,
        features: torch.Tensor) -> torch.Tensor:
    return option_physical_actions(
        previous, champion, features[:, :7])


@torch.no_grad()
def composed_physical_actions(
        router: OptionValueHead, previous: OptionValueHead,
        champion: nn.Module, features: torch.Tensor) -> torch.Tensor:
    router_width = int(router.network[0].normalized_shape[0])
    use_fourth = router(features[:, :router_width]).bool()
    old = previous_option_actions(previous, champion, features)
    return torch.where(
        use_fourth, torch.full_like(old, 3), old)


@torch.no_grad()
def metrics(actions: torch.Tensor, utilities: torch.Tensor) -> dict[str, float]:
    chosen = utilities.gather(1, actions[:, None]).squeeze(1)
    return {
        "verified_utility": float(chosen.mean()),
        "oracle_action_accuracy": float(
            (actions == utilities.argmax(1)).float().mean()),
        "fourth_read_rate": float((actions == 3).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--previous-option", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8061)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--replay-updates", type=int, default=4)
    parser.add_argument("--router-input-width", type=int, default=9)
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
    previous = load_option(args.previous_option, device)
    for module in (champion, previous):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    router = OptionValueHead(
        input_width=args.router_input_width, hidden=32).to(device)
    flat = FlatFourActionValueHead(
        input_width=args.router_input_width, hidden=32).to(device)
    optimizers = {
        "composition": torch.optim.AdamW(
            router.parameters(), lr=args.learning_rate, weight_decay=1e-4),
        "flat": torch.optim.AdamW(
            flat.parameters(), lr=args.learning_rate, weight_decay=1e-4),
    }
    costs = torch.tensor([0.0, 0.01, 0.02, 0.03], device=device)

    test_features, test_outcomes, _ = ranked_requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 80_000_000, device=device, write_threshold=0.5,
        candidate_count=4, include_rank_features=True)
    test_utilities = test_outcomes - costs
    old_actions = previous_option_actions(
        previous, champion, test_features)
    old_metrics = metrics(old_actions, test_utilities)
    target = old_metrics["verified_utility"] + args.practical_gain
    oracle = float(test_utilities.max(1).values.mean())
    histories = {"option_composition": [], "flat_reset": []}

    def record(step: int) -> None:
        actions = {
            "option_composition": composed_physical_actions(
                router, previous, champion, test_features),
            "flat_reset": flat(
                test_features[:, :args.router_input_width]),
        }
        for name, action in actions.items():
            row = metrics(action, test_utilities)
            row.update({
                "step": step,
                "verifier_bits": step * args.batch_size,
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
            write_threshold=0.5, candidate_count=4,
            include_rank_features=True)
        utilities = outcomes - costs
        old = previous_option_actions(previous, champion, features)

        attempted_option = torch.randint(
            0, 2, (args.batch_size,),
            generator=generators["composition_action"], device=device)
        attempted_physical = torch.where(
            attempted_option.bool(), torch.full_like(old, 3), old)
        observed = utilities.gather(
            1, attempted_physical[:, None]).squeeze(1)
        replay["composition"].append((
            features.detach(), attempted_option.detach(), observed.detach()))

        attempted_flat = torch.randint(
            0, 4, (args.batch_size,),
            generator=generators["flat_action"], device=device)
        flat_observed = utilities.gather(
            1, attempted_flat[:, None]).squeeze(1)
        replay["flat"].append((
            features.detach(), attempted_flat.detach(),
            flat_observed.detach()))

        for name, head in (("composition", router), ("flat", flat)):
            all_features = torch.cat([row[0] for row in replay[name]])
            all_actions = torch.cat([row[1] for row in replay[name]])
            all_outcomes = torch.cat([row[2] for row in replay[name]])
            for _ in range(args.replay_updates):
                indices = torch.randint(
                    0, all_actions.numel(), (args.batch_size,),
                    generator=generators[f"{name}_replay"], device=device)
                prediction = head.q_values(
                    all_features[
                        indices, :args.router_input_width]).gather(
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
    for name, rows in histories.items():
        reached = [row for row in rows if row["reaches_target"]]
        first_target[name] = (
            reached[0]["verifier_bits"] if reached else None)
    composed_bits = first_target["option_composition"]
    flat_bits = first_target["flat_reset"]
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
        "schema": "fourth-option-composition-race-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "previous_option": str(args.previous_option),
            "report": str(args.report),
            "checkpoint": (
                str(args.checkpoint) if args.checkpoint else None),
        },
        "old_option": old_metrics,
        "target_utility": target,
        "oracle_utility": oracle,
        "first_target_bits": first_target,
        "histories": histories,
        "gate": gate,
        "accounting": {
            "verifier_bits_per_arm": args.steps * args.batch_size,
            "replay_optimizer_updates_per_arm":
                args.steps * args.replay_updates,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "fourth-option-router-v1",
            "input_width": args.router_input_width,
            "hidden": router.network[1].out_features,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in router.state_dict().items()},
            "previous_option": str(args.previous_option),
            "training_seed": args.seed,
            "verifier_bits": args.steps * args.batch_size,
            "replay_updates": args.replay_updates,
        }, args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
