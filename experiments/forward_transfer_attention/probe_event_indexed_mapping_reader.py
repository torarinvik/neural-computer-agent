"""Test a generic event-indexed reader on four-identity mapping recall.

This is a disposable supervised architecture diagnostic. The frozen agent
produces two sensory-derived raw study rows and one query state. Labels train
only the reader; they never enter the agent, memory writer, or visual stream.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_neural_computer.model import (
    ContentAddressedEventMemoryReader, EventIndexedMemoryReader)
from .probe_mapping_representation import _extract
from .probe_temporal_rule_memory import _load
from .train import seed_everything


def _normalized(train, test):
    result = {}
    for key in ("reader_rows", "reader_queries"):
        reduce = tuple(range(train[key].ndim - 1))
        mean = train[key].mean(reduce, keepdim=True)
        scale = train[key].std(reduce, keepdim=True).clamp_min(1e-5)
        result[key + "_mean"] = mean
        result[key + "_scale"] = scale
    return result


def _fit(train, validation, test, *, seed, device, shuffle_labels=False,
         save_reader: Path | None = None, normalization=None,
         architecture: str = "relation", steps: int = 400):
    seed_everything(seed)
    reader_class = {
        "relation": EventIndexedMemoryReader,
        "content-addressed": ContentAddressedEventMemoryReader,
    }[architecture]
    model = reader_class(train["reader_queries"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3)
    rows = train["reader_rows"].to(device)
    queries = train["reader_queries"].to(device)
    labels = train["reader_actions"].to(device)
    if shuffle_labels:
        generator = torch.Generator().manual_seed(seed + 991)
        labels = labels[torch.randperm(
            labels.numel(), generator=generator).to(device)]
    with torch.no_grad():
        model.rows_mean.copy_(
            normalization["reader_rows_mean"].to(device))
        model.rows_scale.copy_(
            normalization["reader_rows_scale"].to(device))
        model.query_mean.copy_(
            normalization["reader_queries_mean"].to(device))
        model.query_scale.copy_(
            normalization["reader_queries_scale"].to(device))
    best_validation = -1.0
    best_state = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(rows, queries), labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            prediction = model(
                validation["reader_rows"].to(device),
                validation["reader_queries"].to(device)).argmax(-1).cpu()
            accuracy = float((
                prediction == validation["reader_actions"]).float().mean())
            if accuracy > best_validation:
                best_validation = accuracy
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    with torch.no_grad():
        train_prediction = model(rows, queries).argmax(-1).cpu()
        test_rows = test["reader_rows"].to(device)
        test_queries = test["reader_queries"].to(device)
        targets = test["reader_actions"]
        prediction = model(test_rows, test_queries).argmax(-1).cpu()
        swapped = model(test_rows.flip(1), test_queries).argmax(-1).cpu()
        generator = torch.Generator().manual_seed(seed + 1777)
        shuffled_rows = test_rows[
            torch.randperm(test_rows.shape[0], generator=generator).to(device)]
        shuffled = model(shuffled_rows, test_queries).argmax(-1).cpu()
    if save_reader is not None:
        if normalization is None:
            raise ValueError("saving a reader requires normalization")
        save_reader.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "hidden": int(train["reader_queries"].shape[-1]),
            "width": 64,
            "actions": 8,
            "architecture": architecture,
            "normalization": normalization,
            "supervised": True,
            "train_examples": int(train["reader_actions"].numel()),
            "optimizer_steps": steps,
        }, save_reader)
    return {
        "train_accuracy": float((
            train_prediction == train["reader_actions"]).float().mean()),
        "best_validation_accuracy": best_validation,
        "test_accuracy": float((prediction == targets).float().mean()),
        "row_swap_accuracy": float((swapped == targets).float().mean()),
        "row_swap_prediction_agreement": float((
            swapped == prediction).float().mean()),
        "shuffled_memory_accuracy": float((
            shuffled == targets).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--train-lifetimes", type=int, default=256)
    parser.add_argument("--test-lifetimes", type=int, default=256)
    parser.add_argument("--validation-lifetimes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=577)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--query-tap",
        choices=("in-situ-recurrent", "pre-memory-sensory"),
        default="in-situ-recurrent")
    parser.add_argument(
        "--architecture", choices=("relation", "content-addressed"),
        default="relation")
    parser.add_argument(
        "--train-query-surface",
        choices=("audit-card", "temporal-event", "mixed"),
        default="audit-card")
    parser.add_argument(
        "--evaluation-query-surface",
        choices=("audit-card", "temporal-event", "mixed"),
        default="audit-card")
    parser.add_argument("--train-start", type=int, default=63_000_000)
    parser.add_argument("--test-start", type=int, default=65_000_000)
    parser.add_argument("--validation-start", type=int, default=64_000_000)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--save-reader", type=Path)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ranges = (
        range(args.train_start, args.train_start + args.train_lifetimes),
        range(args.validation_start,
              args.validation_start + args.validation_lifetimes),
        range(args.test_start, args.test_start + args.test_lifetimes))
    for left in range(len(ranges)):
        for right in range(left + 1, len(ranges)):
            if max(ranges[left].start, ranges[right].start) < min(
                    ranges[left].stop, ranges[right].stop):
                raise ValueError(
                    "train/validation/test lifetime ranges overlap")
    device = torch.device(args.device)
    transfers = (
        str(args.pairwise_transfer_checkpoint),
        str(args.projection_transfer_checkpoint))
    agent, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfers, transfer_strength=args.transfer_strength)
    train = _extract(
        agent, start=args.train_start, lifetimes=args.train_lifetimes,
        batch_size=args.batch_size, heldout=False, device=device,
        query_tap=args.query_tap,
        reader_query_surface=args.train_query_surface)
    test = _extract(
        agent, start=args.test_start, lifetimes=args.test_lifetimes,
        batch_size=args.batch_size, heldout=True, device=device,
        query_tap=args.query_tap,
        reader_query_surface=args.evaluation_query_surface)
    validation = _extract(
        agent, start=args.validation_start,
        lifetimes=args.validation_lifetimes,
        batch_size=args.batch_size, heldout=True, device=device,
        query_tap=args.query_tap,
        reader_query_surface=args.evaluation_query_surface)
    normalization = _normalized(train, test)
    result = {
        "schema": "event-indexed-mapping-reader-probe-v1",
        "agent_weights_frozen": True,
        "sensory_only": True,
        "diagnostic_labels_visible_to_reader_only": True,
        "lifetime_disjoint": True,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "validation_lifetimes": args.validation_lifetimes,
        "train_start": args.train_start,
        "test_start": args.test_start,
        "validation_start": args.validation_start,
        "query_tap": args.query_tap,
        "train_query_surface": args.train_query_surface,
        "evaluation_query_surface": args.evaluation_query_surface,
        "architecture": args.architecture,
        "optimizer_steps": args.steps,
        "reader": _fit(
            train, validation, test, seed=args.seed, device=device,
            save_reader=args.save_reader, normalization=normalization,
            architecture=args.architecture, steps=args.steps),
        "shuffled_label_control": _fit(
            train, validation, test, seed=args.seed, device=device,
            shuffle_labels=True, normalization=normalization,
            architecture=args.architecture, steps=args.steps),
        "chance_action_accuracy": 1 / 8,
        "normalization_saved": sorted(normalization),
        "pass_bar": {
            "test_accuracy": .65,
            "row_swap_prediction_agreement": .95,
            "shuffled_memory_accuracy_max": .25,
            "shuffled_label_accuracy_max": .20,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
