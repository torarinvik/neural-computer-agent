"""Decode the demonstrated rewarded identity from frozen recurrent states.

This is a disposable supervised diagnostic.  Labels train only the probe and
never enter the controller, consolidator, or memory path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .environment import SHOTS, generate_temporal_attention_lifetime
from .probe_temporal_order import _fit_probe
from .probe_temporal_rule_memory import _load
from .train import _append, seed_everything
from .train_consolidator import _initial_memory


PROBE_SHOTS = tuple(shots for shots in SHOTS if shots > 0)


@torch.no_grad()
def _extract(model, consolidator, *, start: int, lifetimes: int,
             batch_size: int, heldout: bool, target: str,
             device: torch.device, reset_controller_hidden: bool = False):
    names = ("object_1_state", "object_2_state", "post_feedback_state",
             "support_sequence", "prewrite_state") + tuple(
                 f"thought_step_{index + 1}" for index in range(model.thought_steps))
    collected = {shots: {name: [] for name in names} for shots in PROBE_SHOTS}
    labels = {shots: [] for shots in PROBE_SHOTS}
    captured: dict[str, torch.Tensor] = {}
    write_handle = model.write_key.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("prewrite", inputs[0].detach()))
    observation_handle = model.observation_head.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("observations", inputs[0].detach()))
    thought_handle = model.thought_cell.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("thoughts", []).append(
            output.detach()))
    reset_handle = None
    if reset_controller_hidden:
        # Causal diagnostic only: preserve the exact visual stream, weights,
        # memory and probe, but prevent the observation GRU from carrying its
        # hidden state from one visual event into the next. Workspace and
        # persistent-memory paths remain intact, so this is a deliberately
        # narrow test of controller-state recurrence rather than a claim that
        # all recurrent computation has been removed.
        reset_handle = model.controller.register_forward_pre_hook(
            lambda _module, inputs: (inputs[0], torch.zeros_like(inputs[1])))
    try:
        for offset in range(0, lifetimes, batch_size):
            count = min(batch_size, lifetimes - offset)
            items = [generate_temporal_attention_lifetime(
                start + offset + index, heldout=heldout) for index in range(count)]
            memory = _initial_memory(model, items, device)
            cursor = 0
            for shots in PROBE_SHOTS:
                while cursor < shots:
                    captured["thoughts"] = []
                    appended, _, _ = _append(
                        model, [item.supports[cursor] for item in items], memory, device)
                    support_states = captured["observations"]
                    prewrite_state = captured["prewrite"]
                    memory = consolidator(appended)
                    cursor += 1
                values = {
                    "object_1_state": support_states[:, 0],
                    "object_2_state": support_states[:, 1],
                    "post_feedback_state": support_states[:, 2],
                    "support_sequence": support_states.flatten(1),
                    "prewrite_state": prewrite_state,
                }
                if len(captured["thoughts"]) != model.thought_steps:
                    raise RuntimeError(
                        f"captured {len(captured['thoughts'])} thought steps; "
                        f"expected {model.thought_steps}")
                values.update({
                    f"thought_step_{index + 1}": state
                    for index, state in enumerate(captured["thoughts"])
                })
                for name, value in values.items():
                    collected[shots][name].append(value.cpu())
                support_index = shots - 1
                if target == "rewarded-identity":
                    target_values = [
                        item.support_features[support_index][item.rule] for item in items
                    ]
                elif target == "rewarded-was-first":
                    target_values = [item.rule for item in items]
                elif target == "first-identity":
                    target_values = [
                        item.support_features[support_index][0] for item in items
                    ]
                else:
                    raise ValueError(f"unknown target {target!r}")
                labels[shots].append(torch.tensor(target_values, dtype=torch.long))
    finally:
        write_handle.remove()
        observation_handle.remove()
        thought_handle.remove()
        if reset_handle is not None:
            reset_handle.remove()
    return (
        {shots: {name: torch.cat(parts) for name, parts in taps.items()}
         for shots, taps in collected.items()},
        {shots: torch.cat(parts) for shots, parts in labels.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--test-lifetimes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--target",
                        choices=("rewarded-identity", "rewarded-was-first",
                                 "first-identity"),
                        default="rewarded-identity")
    parser.add_argument(
        "--controller-hidden-mode", choices=("normal", "reset-each-event"),
        default="normal",
        help="diagnostic ablation of observation-GRU hidden-state recurrence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, consolidator = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device)
    train_x, train_y = _extract(
        model, consolidator, start=15_000_000, lifetimes=args.train_lifetimes,
        batch_size=args.batch_size, heldout=False, target=args.target, device=device,
        reset_controller_hidden=args.controller_hidden_mode == "reset-each-event")
    test_x, test_y = _extract(
        model, consolidator, start=17_000_000, lifetimes=args.test_lifetimes,
        batch_size=args.batch_size, heldout=True, target=args.target, device=device,
        reset_controller_hidden=args.controller_hidden_mode == "reset-each-event")
    results = {}
    balances = {}
    for shots in PROBE_SHOTS:
        results[str(shots)] = {}
        balances[str(shots)] = {
            "train_label_1_rate": float(train_y[shots].float().mean()),
            "test_label_1_rate": float(test_y[shots].float().mean()),
        }
        for tap in train_x[shots]:
            results[str(shots)][tap] = {
                "linear": _fit_probe(
                    train_x[shots][tap], train_y[shots], test_x[shots][tap],
                    test_y[shots], nonlinear=False, device=device, seed=args.seed),
                "mlp": _fit_probe(
                    train_x[shots][tap], train_y[shots], test_x[shots][tap],
                    test_y[shots], nonlinear=True, device=device, seed=args.seed),
            }
    report = {
        "schema": "temporal-rewarded-identity-recurrent-probe-v1",
        "controller_frozen": True,
        "consolidator_frozen": True,
        "visual_only": True,
        "diagnostic_labels_visible_to_probe_only": True,
        "target": args.target,
        "controller_hidden_mode": args.controller_hidden_mode,
        "train_examples_per_shot": args.train_lifetimes,
        "test_examples_per_shot": args.test_lifetimes,
        "balance": balances,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
