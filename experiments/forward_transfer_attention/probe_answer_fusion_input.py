"""Probe whether the answer-fusion input contains the correct action.

This is a cheap localization test: it never updates the agent.  If a fresh
multiclass probe can decode the action from the exact `(controller, latest-row)`
input used by the fusion head, instability belongs to action-head training. If
it cannot, the query/relation binding is still missing before the head.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch
from torch import nn

from .consolidator import LatentConsolidator
from .environment import generate_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .train import _forward, seed_everything
from .train_consolidator import _initial_memory


def _collect(model, consolidator, *, start, lifetimes, batch_size, heldout,
             shots, device, preserve_raw_write=True, tap="fusion", target="action",
             include_counterfactual=False, preserve_first_raw_write=False):
    if preserve_raw_write and preserve_first_raw_write:
        raise ValueError("choose either the latest or first raw-write sidecar")
    xs, ys = [], []
    model.eval(); consolidator.eval()
    for offset in range(0, lifetimes, batch_size):
        n = min(batch_size, lifetimes - offset)
        items = [generate_temporal_attention_lifetime(
            start + offset + i, heldout=heldout, query_count=1,
            feedback_mode="color-button") for i in range(n)]
        memory = _initial_memory(model, items, device)
        support_raw_keys = []
        first_raw_write = None
        cursor = 0
        while cursor < shots:
            support = [item.supports[cursor] for item in items]
            out, _ = _forward(model, support, memory, device)
            raw_key, raw_value = out.write_keys, out.write_values
            support_raw_keys.append(raw_key.detach().cpu())
            if first_raw_write is None:
                first_raw_write = (raw_key, raw_value, out.write_strengths)
            memory = memory.append(out.write_keys, out.write_values,
                                   out.write_strengths,
                                   torch.ones_like(out.write_strengths))
            memory = consolidator(memory)
            if preserve_raw_write or preserve_first_raw_write:
                sidecar = (first_raw_write if preserve_first_raw_write else
                           (raw_key, raw_value, out.write_strengths))
                memory = memory.append(
                    sidecar[0], sidecar[1], sidecar[2],
                    torch.ones_like(sidecar[2]))
            cursor += 1
        captured = []
        handle = None
        if tap.startswith("support_raw_"):
            if tap == "support_raw_first":
                captured.append(support_raw_keys[0])
            elif tap == "support_raw_latest":
                captured.append(support_raw_keys[-1])
            else:
                captured.append(torch.cat(support_raw_keys, dim=-1))
        elif tap.startswith("memory_"):
            captured.append(
                (memory.keys[:, 0] if tap == "memory_first" else
                 memory.keys[:, -1]).detach().cpu())
        else:
            head = {
                "fusion": model.latest_row_answer_fusion_head,
                "pairwise": getattr(model, "latest_row_answer_pairwise_head", None),
                "event": getattr(model, "latest_row_answer_event_head", None),
                "event_support_raw_all": getattr(
                    model, "latest_row_answer_event_head", None),
            }[tap]
            if head is None:
                raise ValueError(f"checkpoint does not enable the {tap} answer tap")
            handle = head.register_forward_pre_hook(
                lambda _module, inputs: captured.append(inputs[0].detach().cpu()))
        episodes = [item.future_queries[0] for item in items]
        with torch.no_grad():
            _, action_targets = _forward(model, episodes, memory, device)
            counterfactual_action_targets = None
            if include_counterfactual:
                reversed_episodes = []
                for item, episode, order in zip(
                        items, episodes,
                        [item.query_features[0] for item in items]):
                    answer = item.color_mapping[order[1 - item.rule]]
                    reversed_episodes.append(replace(
                        episode, frames=episode.frames[::-1].copy(),
                        pcm=episode.pcm[::-1].copy(),
                        actions=episode.actions * 0 + answer))
                _, counterfactual_action_targets = _forward(
                    model, reversed_episodes, memory, device)
        if handle is not None:
            handle.remove()
        if not captured:
            raise RuntimeError("fusion head was not called")
        if tap == "event_support_raw_all":
            hidden = model.hidden
            enriched = []
            for event_input in captured:
                first = event_input[:, :hidden]
                second = event_input[:, hidden:hidden * 2]
                pieces = [first, second, first * second, (first - second).abs()]
                for raw_key in support_raw_keys:
                    pieces.extend((raw_key, first * raw_key, second * raw_key))
                enriched.append(torch.cat(pieces, dim=-1))
            captured = enriched
        # The answer head is called once per thought step; behavior uses the
        # final step, so probe that exact input rather than an earlier draft.
        if target == "action":
            targets = action_targets.cpu()
        elif target == "first":
            targets = torch.tensor(
                [item.query_features[0][0] for item in items], dtype=torch.long)
        elif target in ("first_action", "second_action"):
            position = 0 if target == "first_action" else 1
            targets = torch.tensor([
                item.color_mapping[item.query_features[0][position]]
                for item in items], dtype=torch.long)
        else:
            targets = torch.tensor([item.rule for item in items], dtype=torch.long)
        original_input = (captured[0] if tap.startswith(("memory_", "support_raw_"))
                          else captured[model.thought_steps - 1])
        xs.append(original_input); ys.append(targets)
        if include_counterfactual:
            if tap.startswith(("memory_", "support_raw_")):
                raise ValueError(
                    "counterfactual probing requires a query-dependent answer-head tap")
            if target == "action":
                counterfactual_targets = counterfactual_action_targets.cpu()
            elif target == "first":
                counterfactual_targets = torch.tensor(
                    [item.query_features[0][1] for item in items],
                    dtype=torch.long)
            elif target in ("first_action", "second_action"):
                position = 1 if target == "first_action" else 0
                counterfactual_targets = torch.tensor([
                    item.color_mapping[item.query_features[0][position]]
                    for item in items], dtype=torch.long)
            else:
                counterfactual_targets = targets
            xs.append(captured[-1])
            ys.append(counterfactual_targets)
    return torch.cat(xs), torch.cat(ys)


def _fit(train_x, train_y, test_x, test_y, *, nonlinear, seed, device):
    seed_everything(seed)
    mean, scale = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x, test_x = ((train_x - mean) / scale).to(device), ((test_x - mean) / scale).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    classes = int(torch.cat((train_y, test_y)).max()) + 1
    model = (nn.Sequential(nn.Linear(train_x.shape[1], 64), nn.GELU(),
                           nn.Linear(64, classes))
             if nonlinear else nn.Linear(train_x.shape[1], classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best = 0.0
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(train_x), train_y).backward(); opt.step()
        with torch.no_grad():
            best = max(best, float((model(test_x).argmax(-1) == test_y).float().mean()))
    with torch.no_grad():
        train_acc = float((model(train_x).argmax(-1) == train_y).float().mean())
        test_acc = float((model(test_x).argmax(-1) == test_y).float().mean())
    return {"train_accuracy": train_acc, "test_accuracy": test_acc, "best_test_accuracy": best}


def _fit_linear_state(train_x, train_y, *, seed, device):
    """Fit a linear probe and fold normalization into raw-space weights."""
    seed_everything(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    x = ((train_x - mean) / scale).to(device); y = train_y.to(device)
    head = nn.Linear(x.shape[1], 8).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(head(x), y).backward(); opt.step()
    weight = head.weight.detach().cpu() / scale
    bias = head.bias.detach().cpu() - (weight * mean).sum(1)
    return {"weight": weight, "bias": bias}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller-checkpoint", type=Path, required=True)
    p.add_argument("--consolidator-checkpoint", type=Path, required=True)
    p.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    p.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    p.add_argument("--transfer-strength", type=float, default=.01)
    p.add_argument("--train-lifetimes", type=int, default=128)
    p.add_argument("--test-lifetimes", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--shots", type=int, default=2)
    p.add_argument("--tap",
                   choices=("fusion", "pairwise", "event",
                            "event_support_raw_all",
                            "memory_first", "memory_latest",
                            "support_raw_first", "support_raw_latest",
                            "support_raw_all"),
                   default="fusion")
    p.add_argument("--target",
                   choices=("action", "first", "first_action",
                            "second_action", "rule"),
                   default="action")
    p.add_argument("--include-counterfactual", action="store_true")
    p.add_argument("--preserve-first-raw-write", action="store_true")
    p.add_argument("--seed", type=int, default=115)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--save-linear-head", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); seed_everything(a.seed); device = torch.device(a.device)
    paths = (str(a.pairwise_transfer_checkpoint), str(a.projection_transfer_checkpoint))
    model, consolidator = _load(a.controller_checkpoint, a.consolidator_checkpoint, device,
                                transfer_paths=paths, transfer_strength=a.transfer_strength)
    train_x, train_y = _collect(model, consolidator, start=11_000_000,
                                lifetimes=a.train_lifetimes, batch_size=a.batch_size,
                                heldout=False, shots=a.shots, device=device, tap=a.tap,
                                target=a.target,
                                include_counterfactual=a.include_counterfactual,
                                preserve_raw_write=not a.preserve_first_raw_write,
                                preserve_first_raw_write=a.preserve_first_raw_write)
    test_x, test_y = _collect(model, consolidator, start=13_000_000,
                               lifetimes=a.test_lifetimes, batch_size=a.batch_size,
                               heldout=True, shots=a.shots, device=device, tap=a.tap,
                               target=a.target,
                               include_counterfactual=a.include_counterfactual,
                               preserve_raw_write=not a.preserve_first_raw_write,
                               preserve_first_raw_write=a.preserve_first_raw_write)
    g = torch.Generator().manual_seed(a.seed + 77)
    shuffled = train_y[torch.randperm(train_y.numel(), generator=g)]
    result = {"normal": {"linear": _fit(train_x, train_y, test_x, test_y,
                                           nonlinear=False, seed=a.seed, device=device),
                          "mlp": _fit(train_x, train_y, test_x, test_y,
                                       nonlinear=True, seed=a.seed, device=device)},
              "shuffled_labels": _fit(train_x, shuffled, test_x, test_y,
                                       nonlinear=True, seed=a.seed, device=device),
              "train_examples": int(train_y.numel()), "test_examples": int(test_y.numel()),
              "shots": a.shots, "tap": a.tap, "target": a.target,
              "include_counterfactual": a.include_counterfactual,
              "preserve_first_raw_write": a.preserve_first_raw_write,
              "schema": "answer-fusion-input-probe-v1"}
    if a.save_linear_head:
        state = _fit_linear_state(train_x, train_y, seed=a.seed, device=device)
        torch.save(state, a.save_linear_head)
        result["linear_head_checkpoint"] = str(a.save_linear_head)
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
