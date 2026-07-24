"""Discarded supervised ceiling probe for the closed-loop intercept interface.

Private oracle actions are diagnostic-only.  Probe weights are never installed
in or used to initialize the deployed learner.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import time
from pathlib import Path

import torch
from torch import nn

from .train import seed_everything
from .train_actuator_transfer import SuccessSystem
from .train_closed_loop_intercept import (
    CURSOR_MAX_SPEED,
    PRETRAIN_START,
    STEP_PIXELS,
    TEST_START,
    TRAIN_START,
    _bounce,
    decision_features,
    execute_policy,
    predictive_metrics,
    pretrain_core,
    trajectory_batch,
)
from .train_micro_intercept import InterceptPredictiveCore, protocol_for_seed


DAGGER_START = 97_000_000


def oracle_labels(
        private_states: torch.Tensor,
        protocol: tuple[int, int, int]) -> torch.Tensor:
    """Return the first action of a minimum-final-distance plan."""

    @functools.lru_cache(maxsize=None)
    def best_distance(target_x: int, target_v: int, cursor_x: int,
                      cursor_v: int, remaining: int) -> int:
        if remaining == 0:
            return abs(target_x - cursor_x)
        values = []
        for command in range(3):
            next_cursor_v = max(
                -CURSOR_MAX_SPEED,
                min(CURSOR_MAX_SPEED, cursor_v + protocol[command]))
            next_cursor_x = max(
                18, min(
                    141, cursor_x + next_cursor_v * STEP_PIXELS))
            next_target_x, next_target_v = _bounce(target_x, target_v)
            values.append(best_distance(
                next_target_x, next_target_v,
                next_cursor_x, next_cursor_v, remaining - 1))
        return min(values)

    labels = torch.empty(
        private_states.shape[:2], dtype=torch.long)
    horizon = private_states.shape[1]
    for lifetime in range(private_states.shape[0]):
        for step in range(horizon):
            target_x, target_v, cursor_x, cursor_v = (
                int(value) for value in private_states[lifetime, step])
            remaining = horizon - step
            distances = []
            for command in range(3):
                next_cursor_v = max(
                    -CURSOR_MAX_SPEED,
                    min(CURSOR_MAX_SPEED,
                        cursor_v + protocol[command]))
                next_cursor_x = max(
                    18, min(
                        141,
                        cursor_x + next_cursor_v * STEP_PIXELS))
                next_target_x, next_target_v = _bounce(
                    target_x, target_v)
                distances.append(best_distance(
                    next_target_x, next_target_v,
                    next_cursor_x, next_cursor_v, remaining - 1))
            labels[lifetime, step] = min(
                range(3), key=lambda command: (
                    distances[command], abs(protocol[command]), command))
    return labels


def fit_probe(
        initial: dict[str, torch.Tensor], features: torch.Tensor,
        labels: torch.Tensor, *, intention_width: int, updates: int,
        batch_size: int, learning_rate: float, seed: int
        ) -> tuple[SuccessSystem, dict[str, float]]:
    flat_features = features.flatten(0, 1)
    flat_labels = labels.flatten().to(features.device)
    model = SuccessSystem(
        flat_features.shape[-1], intention_width, actions=3).to(
            features.device)
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=features.device).manual_seed(seed + 401)
    last_loss = 0.0
    for _ in range(updates):
        indices = torch.randint(
            flat_features.shape[0],
            (min(batch_size, flat_features.shape[0]),),
            generator=generator, device=features.device)
        loss = nn.functional.cross_entropy(
            model(flat_features[indices]), flat_labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    with torch.no_grad():
        accuracy = float(
            (model(flat_features).argmax(-1) == flat_labels).float().mean())
    return model, {"last_loss": last_loss, "train_oracle_accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intention-width", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--pretrain-lifetimes", type=int, default=96)
    parser.add_argument("--pretrain-steps", type=int, default=40)
    parser.add_argument("--probe-lifetimes", type=int, default=90)
    parser.add_argument("--test-lifetimes", type=int, default=96)
    parser.add_argument("--probe-updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    parser.add_argument("--probe-learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    protocol = protocol_for_seed(args.seed)
    pretrain = trajectory_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False,
        protocol=protocol, horizon=args.horizon)
    fixed = trajectory_batch(
        PRETRAIN_START, args.pretrain_lifetimes, heldout=False,
        protocol=protocol, horizon=args.horizon, fixed_stay=True)
    diagnostic = trajectory_batch(
        TRAIN_START, args.probe_lifetimes, heldout=False,
        protocol=protocol, horizon=args.horizon)
    heldout = trajectory_batch(
        TEST_START, args.test_lifetimes, heldout=True,
        protocol=protocol, horizon=args.horizon)
    labels = oracle_labels(diagnostic["private_states"], protocol)
    heldout_labels = oracle_labels(heldout["private_states"], protocol)

    initial_core = InterceptPredictiveCore(args.hidden).to(device)
    core_state = copy.deepcopy(initial_core.state_dict())
    cores, pretraining = {}, {}
    for name, mode, data in (
            ("action_conditioned", "action_conditioned", pretrain),
            ("passive", "passive", pretrain),
            ("shuffled_action", "shuffled_action", pretrain),
            ("fixed_no_action", "action_conditioned", fixed)):
        core = InterceptPredictiveCore(args.hidden).to(device)
        core.load_state_dict(core_state)
        pretraining[name] = pretrain_core(
            core, data, mode=mode, steps=args.pretrain_steps,
            batch_size=args.batch_size,
            learning_rate=args.pretrain_learning_rate,
            seed=args.seed, device=device)
        cores[name] = core
    fresh = InterceptPredictiveCore(args.hidden).to(device)
    fresh.load_state_dict(core_state)
    cores["fully_fresh"] = fresh
    passive_modes = {
        "action_conditioned": False,
        "passive": True,
        "shuffled_action": False,
        "fixed_no_action": False,
        "fully_fresh": False,
    }
    features = {
        name: decision_features(
            core, diagnostic["frames"][:, :-1],
            passive=passive_modes[name], device=device)
        for name, core in cores.items()
    }
    heldout_features = {
        name: decision_features(
            core, heldout["frames"][:, :-1],
            passive=passive_modes[name], device=device)
        for name, core in cores.items()
    }
    template = SuccessSystem(
        args.hidden * 4, args.intention_width, actions=3).to(device)
    initial_probe = copy.deepcopy(template.state_dict())
    results, fitted_models = {}, {}
    for name, core in cores.items():
        model, fit = fit_probe(
            initial_probe, features[name], labels,
            intention_width=args.intention_width,
            updates=args.probe_updates, batch_size=args.batch_size,
            learning_rate=args.probe_learning_rate,
            seed=args.seed + 500)
        with torch.no_grad():
            heldout_oracle_accuracy = float(
                (model(heldout_features[name].flatten(0, 1)).argmax(-1).cpu()
                 == heldout_labels.flatten()).float().mean())
        execution = execute_policy(
            core, model, start=TEST_START, count=args.test_lifetimes,
            protocol=protocol, horizon=args.horizon,
            passive=passive_modes[name], device=device)
        reverse = execute_policy(
            core, model, start=TEST_START, count=args.test_lifetimes,
            protocol=protocol, horizon=args.horizon,
            passive=passive_modes[name], device=device,
            reverse_motion=True)
        results[name] = {
            **fit,
            "heldout_oracle_accuracy": heldout_oracle_accuracy,
            "heldout_terminal_success": execution["terminal_success"],
            "reverse_terminal_success": reverse["terminal_success"],
            "reverse_action_sequence_change": float(
                (execution["actions"] != reverse["actions"]).float().mean()),
            "predictive_metrics": predictive_metrics(
                core, heldout, passive=passive_modes[name], device=device),
        }
        fitted_models[name] = model
        print(json.dumps({"arm": name, **results[name]}, sort_keys=True),
              flush=True)

    shuffled_labels = labels.flatten()[
        torch.randperm(
            labels.numel(),
            generator=torch.Generator().manual_seed(args.seed + 601))
    ].reshape_as(labels)
    shuffled_model, shuffled_fit = fit_probe(
        initial_probe, features["action_conditioned"], shuffled_labels,
        intention_width=args.intention_width,
        updates=args.probe_updates, batch_size=args.batch_size,
        learning_rate=args.probe_learning_rate, seed=args.seed + 602)
    shuffled_execution = execute_policy(
        cores["action_conditioned"], shuffled_model,
        start=TEST_START, count=args.test_lifetimes,
        protocol=protocol, horizon=args.horizon,
        passive=False, device=device)
    results["shuffled_oracle_labels"] = {
        **shuffled_fit,
        "heldout_terminal_success": (
            shuffled_execution["terminal_success"]),
    }

    # Diagnostic-only data aggregation: ask whether the interface can control
    # when the oracle labels states induced by the probe's own behavior.
    candidate_core = cores["action_conditioned"]
    induced = execute_policy(
        candidate_core, fitted_models["action_conditioned"],
        start=DAGGER_START, count=args.probe_lifetimes,
        protocol=protocol, horizon=args.horizon,
        passive=False, device=device, return_diagnostic_data=True)
    induced_labels = oracle_labels(
        induced["diagnostic_private_states"], protocol)
    induced_features = decision_features(
        candidate_core, induced["diagnostic_frames"],
        passive=False, device=device)
    dagger_model, dagger_fit = fit_probe(
        initial_probe,
        torch.cat([features["action_conditioned"], induced_features]),
        torch.cat([labels, induced_labels]),
        intention_width=args.intention_width,
        updates=args.probe_updates, batch_size=args.batch_size,
        learning_rate=args.probe_learning_rate, seed=args.seed + 701)
    dagger_execution = execute_policy(
        candidate_core, dagger_model,
        start=TEST_START, count=args.test_lifetimes,
        protocol=protocol, horizon=args.horizon,
        passive=False, device=device)
    dagger_reverse = execute_policy(
        candidate_core, dagger_model,
        start=TEST_START, count=args.test_lifetimes,
        protocol=protocol, horizon=args.horizon,
        passive=False, device=device, reverse_motion=True)
    results["diagnostic_dagger"] = {
        **dagger_fit,
        "induced_trajectories": args.probe_lifetimes,
        "heldout_terminal_success": (
            dagger_execution["terminal_success"]),
        "reverse_terminal_success": (
            dagger_reverse["terminal_success"]),
        "reverse_action_sequence_change": float(
            (dagger_execution["actions"] !=
             dagger_reverse["actions"]).float().mean()),
    }
    candidate = results["action_conditioned"]
    dagger = results["diagnostic_dagger"]
    gate = {
        "interface_supports_control": bool(
            dagger["heldout_terminal_success"] >= 0.60 and
            dagger["reverse_terminal_success"] >= 0.55 and
            dagger["reverse_action_sequence_change"] >= 0.35 and
            dagger["heldout_terminal_success"] >=
            results["shuffled_oracle_labels"][
                "heldout_terminal_success"] + 0.20),
        "offline_to_on_policy_gap": (
            candidate["heldout_oracle_accuracy"] -
            candidate["heldout_terminal_success"]),
        "dagger_terminal_gain": (
            dagger["heldout_terminal_success"] -
            candidate["heldout_terminal_success"]),
        "weights_must_be_discarded": True,
    }
    report = {
        "schema": "closed-loop-control-ceiling-probe-v1",
        "diagnostic_only": True,
        "semantic_oracle_labels_used": True,
        "probe_weights_enter_deployed_agent": False,
        "train_oracle_label_counts": torch.bincount(
            labels.flatten(), minlength=3).tolist(),
        "heldout_oracle_label_counts": torch.bincount(
            heldout_labels.flatten(), minlength=3).tolist(),
        "configuration": vars(args) | {"report": str(args.report)},
        "results": results,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate, "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
