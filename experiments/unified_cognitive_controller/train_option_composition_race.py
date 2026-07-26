"""Compare a retrieved champion option with flat three-action relearning."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .probe_requery_operation import ranked_requery_batch
from .train import seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import _load_head
from .train_three_way_requery_race import RankedActionValueHead


class OptionValueHead(nn.Module):
    """Value the retrieved old skill against one genuinely new operation."""

    def __init__(self, input_width: int = 7, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_values(features).argmax(-1)


@torch.no_grad()
def champion_actions(champion: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return (champion(features[:, :4]) > 0).long()


@torch.no_grad()
def option_physical_actions(
        option_head: OptionValueHead, champion: nn.Module,
        features: torch.Tensor) -> torch.Tensor:
    use_new = option_head(features).bool()
    return torch.where(
        use_new, torch.full_like(use_new, 2, dtype=torch.long),
        champion_actions(champion, features))


@torch.no_grad()
def metrics(
        physical_actions: torch.Tensor,
        utilities: torch.Tensor) -> dict[str, float]:
    selected = utilities.gather(
        1, physical_actions[:, None]).squeeze(1)
    return {
        "verified_utility": float(selected.mean()),
        "oracle_action_accuracy": float(
            (physical_actions == utilities.argmax(1)).float().mean()),
        "third_read_rate": float(
            (physical_actions == 2).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8050)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--second-cost", type=float, default=0.01)
    parser.add_argument("--third-cost", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--practical-gain", type=float, default=0.02)
    parser.add_argument(
        "--replay-updates", type=int, default=1,
        help="Optimizer minibatches drawn per newly verified outcome batch.")
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
    option = OptionValueHead().to(device)
    flat = RankedActionValueHead(32, input_width=7).to(device)
    option_optimizer = torch.optim.AdamW(
        option.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    flat_optimizer = torch.optim.AdamW(
        flat.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    costs = torch.tensor(
        [0.0, args.second_cost, args.third_cost], device=device)

    test_features, test_outcomes, _ = ranked_requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 80_000_000, device=device,
        write_threshold=args.write_threshold, candidate_count=3,
        include_rank_features=True)
    test_utilities = test_outcomes - costs
    old_actions = champion_actions(champion, test_features)
    old_metrics = metrics(old_actions, test_utilities)
    target = old_metrics["verified_utility"] + args.practical_gain
    oracle = float(test_utilities.max(1).values.mean())
    histories = {"option_composition": [], "flat_reset": []}

    def record(step: int) -> None:
        actions = {
            "option_composition": option_physical_actions(
                option, champion, test_features),
            "flat_reset": flat(test_features),
        }
        for name, physical in actions.items():
            row = metrics(physical, test_utilities)
            row.update({
                "step": step,
                "verifier_bits": step * args.batch_size,
                "reaches_target": row["verified_utility"] >= target,
            })
            histories[name].append(row)

    record(0)
    generators = {
        "option": torch.Generator(device=device).manual_seed(
            args.seed + 70_000_000),
        "flat": torch.Generator(device=device).manual_seed(
            args.seed + 70_000_000),
        "option_replay": torch.Generator(device=device).manual_seed(
            args.seed + 71_000_000),
        "flat_replay": torch.Generator(device=device).manual_seed(
            args.seed + 72_000_000),
    }
    replay = {"option": [], "flat": []}
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        features, outcomes, _ = ranked_requery_batch(
            controller, count=args.batch_size, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold, candidate_count=3,
            include_rank_features=True)
        utilities = outcomes - costs

        attempted_option = torch.randint(
            0, 2, (args.batch_size,),
            generator=generators["option"], device=device)
        old = champion_actions(champion, features)
        attempted_physical = torch.where(
            attempted_option.bool(),
            torch.full_like(attempted_option, 2), old)
        observed = utilities.gather(
            1, attempted_physical[:, None]).squeeze(1)
        replay["option"].append((
            features.detach(), attempted_option.detach(), observed.detach()))

        attempted_flat = torch.randint(
            0, 3, (args.batch_size,),
            generator=generators["flat"], device=device)
        flat_observed = utilities.gather(
            1, attempted_flat[:, None]).squeeze(1)
        replay["flat"].append((
            features.detach(), attempted_flat.detach(),
            flat_observed.detach()))

        for name, head, optimizer in (
                ("option", option, option_optimizer),
                ("flat", flat, flat_optimizer)):
            replay_features = torch.cat([
                row[0] for row in replay[name]])
            replay_actions = torch.cat([
                row[1] for row in replay[name]])
            replay_outcomes = torch.cat([
                row[2] for row in replay[name]])
            for _ in range(args.replay_updates):
                replay_indices = torch.randint(
                    0, replay_actions.numel(), (args.batch_size,),
                    generator=generators[f"{name}_replay"],
                    device=device)
                prediction = head.q_values(
                    replay_features[replay_indices]).gather(
                        1, replay_actions[replay_indices, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(
                    prediction, replay_outcomes[replay_indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
        if step % 2 == 0 or step == args.steps:
            record(step)

    first_target = {}
    for name, rows in histories.items():
        reached = [row for row in rows if row["reaches_target"]]
        first_target[name] = (
            reached[0]["verifier_bits"] if reached else None)
    option_bits = first_target["option_composition"]
    flat_bits = first_target["flat_reset"]
    gate = {
        "oracle_headroom": oracle >= target + 0.02,
        "option_reaches_target": option_bits is not None,
        "option_reaches_before_flat": (
            option_bits is not None
            and (flat_bits is None or option_bits < flat_bits)),
        "option_final_retains_gain": (
            histories["option_composition"][-1]["verified_utility"]
            >= target),
    }
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "option-composition-race-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "report": str(args.report),
            "checkpoint": (
                str(args.checkpoint)
                if args.checkpoint is not None else None),
        },
        "learner_visible": [
            "seven_generic_rank_statistics",
            "randomized_attempted_latent_option",
            "attempted_option_scalar_outcome",
        ],
        "hidden_from_learner": [
            "unattempted_outcomes", "oracle_action", "correct_answer",
            "semantic_task_identity", "private_test_metrics",
        ],
        "champion": old_metrics,
        "target_utility": target,
        "oracle_utility": oracle,
        "first_target_bits": first_target,
        "histories": histories,
        "gate": gate,
        "accounting": {
            "verifier_bits_per_arm": args.steps * args.batch_size,
            "optimizer_updates_per_arm": args.steps,
            "replay_optimizer_updates_per_arm":
                args.steps * args.replay_updates,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "option-composition-head-v1",
            "input_width": 7,
            "hidden": option.network[1].out_features,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in option.state_dict().items()},
            "source_champion": str(args.champion_head),
            "training_seed": args.seed,
            "verifier_bits": args.steps * args.batch_size,
            "replay_updates": args.replay_updates,
        }, args.checkpoint)
        report["checkpoint"] = str(args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
