"""High-precision retention audit for a protected n-back checkpoint.

This is deliberately a no-update audit.  Both checkpoints are evaluated with
the same three-stream controller/RAM configuration and independent seeds.  The
report keeps the parent/child comparison, reset control, and time-shuffle
control together so a harder-rung gain cannot be promoted while hiding
catastrophic forgetting in older rungs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .brainworkshop_gym import BrainWorkshopConfig
from .train_brainworkshop_policy import BrainWorkshopPolicy, _evaluate
from .legacy_model import UnifiedCognitiveController


def _load_policy(path: Path, device: torch.device) -> BrainWorkshopPolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_config = dict(payload["model_configuration"])
    model_config.pop("external_history_depth", None)
    model_config.pop("feedback_skill_adapter_width", None)
    model_config.pop("feedback_skill_history_depth", None)
    model_config.pop("relational_context_adapter_width", None)
    model_config.pop("relational_context_max_history", None)
    model_config.pop("relational_context_auxiliary_weight", None)
    model_config.pop("relational_context_output_adapter", None)
    model_config.pop("relational_context_use_controller_state", None)
    controller = UnifiedCognitiveController(**model_config).to(device)
    controller.vision = None
    controller.actuator = None
    state_dict = payload["state_dict"]
    controller_state = {
        key.removeprefix("controller."): value
        for key, value in state_dict.items()
        if key.startswith("controller.")
    }
    controller.load_state_dict(controller_state or state_dict)
    policy = BrainWorkshopPolicy(
        external_memory_adapter_width=64,
        external_history_depth=8,
        per_stream_external_history=True,
        per_stream_intention_adapter_width=64,
        feedback_skill_adapter_width=int(
            payload.get("model_configuration", {}).get(
                "feedback_skill_adapter_width", 0)),
        feedback_skill_history_depth=int(
            payload.get("model_configuration", {}).get(
                "feedback_skill_history_depth", 1)),
        factorized_output=True,
        factorized_reward=True,
        modalities=("vision", "audio", "text"),
        target_modalities=("text",),
        controller=controller,
    ).to(device)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def audit_checkpoint(
    path: Path,
    *,
    rungs: tuple[int, ...],
    count: int,
    trials: int,
    seed: int,
    device: torch.device,
) -> dict:
    policy = _load_policy(path, device)
    results: dict[str, dict] = {}
    for offset, n_back in enumerate(rungs):
        config = BrainWorkshopConfig(
            n_back=n_back,
            trials=trials,
            position_vocab=2,
            text_vocab=8,
            modalities=("vision", "audio", "text"),
            trial_ms=1000,
            balanced_matches=False,
        )
        common = dict(
            count=count,
            device=device,
            external_history=True,
            per_stream_external_history=True,
            external_history_depth=8,
        )
        normal = _evaluate(policy, config, seed=seed + offset * 3, **common)
        reset = _evaluate(
            policy, config, seed=seed + offset * 3 + 1,
            reset_history=True, **common)
        shuffled = _evaluate(
            policy, config, seed=seed + offset * 3 + 2,
            shuffle_time=True, **common)
        results[str(n_back)] = {
            "normal": normal,
            "history_reset": reset,
            "time_shuffle": shuffled,
        }
    return {"checkpoint": str(path), "count": count, "trials": trials,
            "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=48300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    rungs = (1, 5, 6, 7, 8)
    report = {
        "format": "nback_retention_audit.v1",
        "rungs": list(rungs),
        "parent": audit_checkpoint(
            args.parent, rungs=rungs, count=args.count, trials=args.trials,
            seed=args.seed, device=device),
        "child": audit_checkpoint(
            args.child, rungs=rungs, count=args.count, trials=args.trials,
            seed=args.seed + 100, device=device),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for side in ("parent", "child"):
        print(side, json.dumps(report[side]["results"], sort_keys=True))


if __name__ == "__main__":
    main()
