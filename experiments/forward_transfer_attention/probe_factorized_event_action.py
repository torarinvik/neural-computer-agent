"""Probe learned factorized composition of support rules and query events.

The frozen agent receives only its normal sensory stream. Verifier-side labels
train throwaway diagnostic heads for the support rule and the two candidate
actions. Passing this probe justifies integrating the generic factorized
architecture; it is not itself a behavioral capability claim.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_neural_computer.model import (
    FactorizedEventAnswerRouter)

from .environment import (
    generate_compositional_temporal_attention_lifetime,
    generate_temporal_attention_lifetime)
from .probe_temporal_rule_memory import _load
from .train import _forward, seed_everything
from .train_consolidator import _initial_memory


def _collect(model, consolidator, *, start: int, lifetimes: int,
             batch_size: int, shots: int, heldout: bool,
             device: torch.device,
             generator=generate_temporal_attention_lifetime
             ) -> dict[str, torch.Tensor]:
    fields: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "support", "first", "second", "rule", "first_action",
            "second_action", "action", "pair", "reversed")}
    model.eval()
    consolidator.eval()
    pair_cursor = 0
    for offset in range(0, lifetimes, batch_size):
        count = min(batch_size, lifetimes - offset)
        items = [
            generator(
                start + offset + index, heldout=heldout, query_count=1,
                feedback_mode="color-button")
            for index in range(count)]
        memory = _initial_memory(model, items, device)
        first_raw = None
        for shot in range(shots):
            output, _ = _forward(
                model, [item.supports[shot] for item in items], memory, device)
            if first_raw is None:
                first_raw = (
                    output.write_keys, output.write_values,
                    output.write_strengths)
            memory = memory.append(
                output.write_keys, output.write_values, output.write_strengths,
                torch.ones_like(output.write_strengths))
            memory = consolidator(memory)
            memory = memory.append(
                first_raw[0], first_raw[1], first_raw[2],
                torch.ones_like(first_raw[2]))

        captured = []
        if hasattr(model, "latest_row_answer_event_head"):
            handle = model.latest_row_answer_event_head.register_forward_pre_hook(
                lambda _module, inputs: captured.append(
                    inputs[0].detach().cpu()))
            capture_kind = "event_head"
        elif hasattr(model, "latest_row_factorized_router"):
            handle = model.latest_row_factorized_router.register_forward_pre_hook(
                lambda _module, inputs: captured.append(tuple(
                    value.detach().cpu() for value in inputs)))
            capture_kind = "factorized_router"
        else:
            raise ValueError("model has no supported event-answer input tap")
        originals = [item.future_queries[0] for item in items]
        with torch.no_grad():
            _, original_actions = _forward(model, originals, memory, device)
        original_input = captured[model.thought_steps - 1]
        captured.clear()

        reversed_episodes = []
        for item, episode, order in zip(
                items, originals, [item.query_features[0] for item in items]):
            answer = item.color_mapping[order[1 - item.rule]]
            reversed_episodes.append(replace(
                episode, frames=episode.frames[::-1].copy(),
                pcm=episode.pcm[::-1].copy(),
                actions=episode.actions * 0 + answer))
        with torch.no_grad():
            _, reversed_actions = _forward(
                model, reversed_episodes, memory, device)
        reversed_input = captured[model.thought_steps - 1]
        handle.remove()

        hidden = model.hidden
        pair = torch.arange(
            pair_cursor, pair_cursor + count, dtype=torch.long)
        pair_cursor += count
        rule = torch.tensor([item.rule for item in items], dtype=torch.long)
        first_action = torch.tensor([
            item.color_mapping[item.query_features[0][0]] for item in items],
            dtype=torch.long)
        second_action = torch.tensor([
            item.color_mapping[item.query_features[0][1]] for item in items],
            dtype=torch.long)
        support = first_raw[0].detach().cpu()
        for event_input, candidate_1, candidate_2, action, is_reversed in (
                (original_input, first_action, second_action,
                 original_actions.cpu(), False),
                (reversed_input, second_action, first_action,
                 reversed_actions.cpu(), True)):
            if capture_kind == "event_head":
                captured_support = support
                captured_first = event_input[:, :hidden]
                captured_second = event_input[:, hidden:hidden * 2]
            else:
                captured_support, captured_first, captured_second = event_input
            fields["support"].append(captured_support)
            fields["first"].append(captured_first)
            fields["second"].append(captured_second)
            fields["rule"].append(rule)
            fields["first_action"].append(candidate_1)
            fields["second_action"].append(candidate_2)
            fields["action"].append(action)
            fields["pair"].append(pair)
            fields["reversed"].append(torch.full(
                (count,), is_reversed, dtype=torch.bool))
    return {key: torch.cat(value) for key, value in fields.items()}


def _normalize(train: dict[str, torch.Tensor],
               test: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    statistics = {}
    for key in ("support", "first", "second"):
        mean = train[key].mean(0, keepdim=True)
        scale = train[key].std(0, keepdim=True).clamp_min(1e-5)
        statistics[key + "_mean"] = mean
        statistics[key + "_scale"] = scale
        train[key] = (train[key] - mean) / scale
        test[key] = (test[key] - mean) / scale
    return statistics


def _fit(train, test, *, seed: int, device: torch.device,
         aux_weight: float, shuffled_final: bool = False,
         ingredient_pretrain_steps: int = 0,
         freeze_ingredients: bool = False,
         route_regularizer_weight: float = 0.0,
         save_router: Path | None = None,
         normalization: dict[str, torch.Tensor] | None = None):
    seed_everything(seed)
    model = FactorizedEventAnswerRouter(
        train["support"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3)
    action_targets = train["action"]
    if shuffled_final:
        generator = torch.Generator().manual_seed(seed + 991)
        action_targets = action_targets[
            torch.randperm(action_targets.numel(), generator=generator)]
    train_device = {
        key: value.to(device) for key, value in train.items()
        if key not in ("pair", "reversed")}
    test_device = {
        key: value.to(device) for key, value in test.items()
        if key not in ("pair", "reversed")}
    action_targets = action_targets.to(device)
    for _ in range(ingredient_pretrain_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            train_device["support"], train_device["first"],
            train_device["second"])
        ingredient_loss = (
            nn.functional.cross_entropy(
                output["rule"], train_device["rule"]) +
            nn.functional.cross_entropy(
                output["first_action"], train_device["first_action"]) +
            nn.functional.cross_entropy(
                output["second_action"], train_device["second_action"]))
        ingredient_loss.backward()
        optimizer.step()
    if freeze_ingredients:
        for name, parameter in model.named_parameters():
            if not name.startswith("answer_gate."):
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.answer_gate.parameters(), lr=3e-3, weight_decay=1e-3)
    best = 0.0
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            train_device["support"], train_device["first"],
            train_device["second"])
        loss = nn.functional.cross_entropy(output["action"], action_targets)
        if route_regularizer_weight:
            route = output["route"].clamp_min(1e-8)
            sample_entropy = -(route * route.log()).sum(-1).mean()
            mean_route = route.mean(0)
            batch_entropy = -(mean_route * mean_route.log()).sum()
            loss = loss + route_regularizer_weight * (
                sample_entropy - batch_entropy)
        if aux_weight:
            loss = loss + aux_weight * (
                nn.functional.cross_entropy(
                    output["rule"], train_device["rule"]) +
                nn.functional.cross_entropy(
                    output["first_action"], train_device["first_action"]) +
                nn.functional.cross_entropy(
                    output["second_action"], train_device["second_action"]))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            test_output = model(
                test_device["support"], test_device["first"],
                test_device["second"])
            accuracy = float((
                test_output["action"].argmax(-1) == test_device["action"]
            ).float().mean())
            best = max(best, accuracy)
    with torch.no_grad():
        output = model(
            test_device["support"], test_device["first"],
            test_device["second"])
        predictions = output["action"].argmax(-1).cpu()
        hard_predictions = output["hard_action"].argmax(-1).cpu()
        generator = torch.Generator(device=device).manual_seed(seed + 1777)
        shuffled_support = test_device["support"][
            torch.randperm(
                test_device["support"].shape[0],
                generator=generator, device=device)]
        shuffled_support_predictions = model(
            shuffled_support, test_device["first"],
            test_device["second"])["action"].argmax(-1).cpu()
    if save_router is not None:
        if normalization is None:
            raise ValueError("saving a router requires normalization statistics")
        for name, value in normalization.items():
            getattr(model, name).copy_(value.to(device))
        save_router.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "hidden": int(train["support"].shape[1]),
            "width": 64,
            "supervised": True,
            "ingredient_pretrain_steps": ingredient_pretrain_steps,
            "route_regularizer_weight": route_regularizer_weight,
            "train_logical_lifetimes": int(
                train["action"].numel() // 2),
        }, save_router)
    normal = ~test["reversed"]
    reversed_mask = test["reversed"]
    original_by_pair = {
        int(pair): int(prediction)
        for pair, prediction in zip(
            test["pair"][normal], predictions[normal])}
    reversed_by_pair = {
        int(pair): int(prediction)
        for pair, prediction in zip(
            test["pair"][reversed_mask], predictions[reversed_mask])}
    hard_original_by_pair = {
        int(pair): int(prediction)
        for pair, prediction in zip(
            test["pair"][normal], hard_predictions[normal])}
    hard_reversed_by_pair = {
        int(pair): int(prediction)
        for pair, prediction in zip(
            test["pair"][reversed_mask], hard_predictions[reversed_mask])}
    flip_rate = sum(
        original_by_pair[pair] != reversed_by_pair[pair]
        for pair in original_by_pair) / len(original_by_pair)
    original_targets = {
        int(pair): int(target)
        for pair, target in zip(
            test["pair"][normal], test["action"][normal])}
    stale_reversal_accuracy = sum(
        reversed_by_pair[pair] == original_targets[pair]
        for pair in original_by_pair) / len(original_by_pair)
    hard_flip_rate = sum(
        hard_original_by_pair[pair] != hard_reversed_by_pair[pair]
        for pair in hard_original_by_pair) / len(hard_original_by_pair)
    hard_stale_reversal_accuracy = sum(
        hard_reversed_by_pair[pair] == original_targets[pair]
        for pair in hard_original_by_pair) / len(hard_original_by_pair)
    return {
        "best_action_accuracy": best,
        "action_accuracy": float((
            predictions == test["action"]).float().mean()),
        "hard_action_accuracy": float((
            hard_predictions == test["action"]).float().mean()),
        "rule_accuracy": float((
            output["rule"].argmax(-1).cpu() == test["rule"]).float().mean()),
        "first_action_accuracy": float((
            output["first_action"].argmax(-1).cpu() ==
            test["first_action"]).float().mean()),
        "second_action_accuracy": float((
            output["second_action"].argmax(-1).cpu() ==
            test["second_action"]).float().mean()),
        "counterfactual_prediction_flip_rate": flip_rate,
        "stale_label_reversal_accuracy": stale_reversal_accuracy,
        "hard_counterfactual_prediction_flip_rate": hard_flip_rate,
        "hard_stale_label_reversal_accuracy": hard_stale_reversal_accuracy,
        "shuffled_support_accuracy": float((
            shuffled_support_predictions == test["action"]).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path)
    parser.add_argument("--consolidator-checkpoint", type=Path)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path)
    parser.add_argument("--projection-transfer-checkpoint", type=Path)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--train-lifetimes", type=int, default=128)
    parser.add_argument("--test-lifetimes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--ingredient-pretrain-steps", type=int, default=0)
    parser.add_argument("--freeze-ingredients", action="store_true")
    parser.add_argument("--route-regularizer-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=337)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--save-router", type=Path)
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if args.train_cache or args.test_cache:
        if not (args.train_cache and args.test_cache):
            raise ValueError("both train and test caches are required")
        train_payload = torch.load(
            args.train_cache, map_location="cpu", weights_only=False)
        test_payload = torch.load(
            args.test_cache, map_location="cpu", weights_only=False)
        train = train_payload["tensors"]
        test = test_payload["tensors"]
        args.train_lifetimes = int(train["pair"].unique().numel())
        args.test_lifetimes = int(test["pair"].unique().numel())
        cache_provenance = {
            "train": train_payload["provenance"],
            "test": test_payload["provenance"],
        }
    else:
        required = (
            args.controller_checkpoint, args.consolidator_checkpoint,
            args.pairwise_transfer_checkpoint,
            args.projection_transfer_checkpoint)
        if not all(required):
            raise ValueError("controller/consolidator/transfer checkpoints are required")
        transfers = (
            str(args.pairwise_transfer_checkpoint),
            str(args.projection_transfer_checkpoint))
        model, consolidator = _load(
            args.controller_checkpoint, args.consolidator_checkpoint, device,
            transfer_paths=transfers, transfer_strength=args.transfer_strength)
        train = _collect(
            model, consolidator, start=17_000_000,
            lifetimes=args.train_lifetimes, batch_size=args.batch_size,
            shots=args.shots, heldout=False, device=device)
        test = _collect(
            model, consolidator, start=19_000_000,
            lifetimes=args.test_lifetimes, batch_size=args.batch_size,
            shots=args.shots, heldout=True, device=device)
        cache_provenance = None
    normalization = _normalize(train, test)
    result = {
        "factorized_auxiliary": _fit(
            train, test, seed=args.seed, device=device,
            aux_weight=args.aux_weight,
            ingredient_pretrain_steps=args.ingredient_pretrain_steps,
            freeze_ingredients=args.freeze_ingredients,
            route_regularizer_weight=args.route_regularizer_weight,
            save_router=args.save_router, normalization=normalization),
        "shuffled_final_labels": _fit(
            train, test, seed=args.seed, device=device,
            aux_weight=args.aux_weight, shuffled_final=True,
            ingredient_pretrain_steps=args.ingredient_pretrain_steps,
            freeze_ingredients=args.freeze_ingredients,
            route_regularizer_weight=args.route_regularizer_weight),
        "train_logical_lifetimes": args.train_lifetimes,
        "test_logical_lifetimes": args.test_lifetimes,
        "train_examples_with_counterfactuals": int(train["action"].numel()),
        "test_examples_with_counterfactuals": int(test["action"].numel()),
        "auxiliary_weight": args.aux_weight,
        "ingredient_pretrain_steps": args.ingredient_pretrain_steps,
        "ingredients_frozen_for_composition": args.freeze_ingredients,
        "route_regularizer_weight": args.route_regularizer_weight,
        "lifetime_disjoint": True,
        "pass_bar": {
            "action_accuracy": .65,
            "shuffled_action_near_chance": .125,
            "counterfactual_prediction_flip_rate": .80,
        },
        "schema": "factorized-event-action-probe-v1",
    }
    if cache_provenance is not None:
        result["cache_provenance"] = cache_provenance
    if args.save_router:
        result["router_checkpoint"] = str(args.save_router)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
