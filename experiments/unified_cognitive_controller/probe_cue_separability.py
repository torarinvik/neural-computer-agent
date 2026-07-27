"""Predict a rung's interference before paying for it.

A successor slot can only shut on a skill it can tell apart. Its gate reads the
frozen encoder's event features, so two operations the frozen encoder maps to
nearly the same place are indistinguishable to the gate no matter how it is
trained, and the new slot will perturb the old skill.

That makes interference predictable from the parent alone. This probe encodes
the same underlying events under each operation's cue and reports the pairwise
separation of the frozen features. Run it before adding a rung: a pair below the
warning threshold will interfere, and the fix is the cue, not the training.

The measured relationship on the four-skill controller was direct -- separation
4.05 gave +0.0007 retention delta, 1.36 gave about zero, and 0.25 gave -0.0303.

The encoder ends in a global average pool, so cues of equal area differing only
in position are nearly identical to it. Separate cues by area, not by position.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from .environment import _OPERATION_CUE_SLOTS, generate_lifetimes
from .model import UnifiedCognitiveController


DEFAULT_TASKS = (
    "visible_context", "visible_context_xor", "contextual_composition",
    "context_rule_xor", "contextual_override")


@torch.no_grad()
def cue_separations(
        model: UnifiedCognitiveController, tasks: tuple[str, ...], *,
        count: int, seed: int, support_trials: int,
        device: torch.device) -> dict[str, float]:
    """Pairwise distance between frozen event features, in feature sigmas."""
    model.eval()
    features = {}
    for task in tasks:
        batch = generate_lifetimes(
            count, 6, seed=seed, task=task, support_trials=support_trials,
            device=device)
        features[task] = model.vision(batch.frames[:, 0])
    scale = torch.cat(list(features.values())).std()
    if not torch.isfinite(scale) or scale == 0:
        raise RuntimeError("frozen features have no usable scale")
    return {
        f"{first}|{second}": float(
            (features[first] - features[second]).norm(dim=-1).mean() / scale)
        for first, second in itertools.combinations(tasks, 2)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--support-trials", type=int, default=2)
    parser.add_argument(
        "--warn-below", type=float, default=1.0,
        help="separations under this predicted measurable interference")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.parent, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    tasks = tuple(name for name in args.tasks.split(",") if name)
    separations = cue_separations(
        model, tasks, count=args.count, seed=args.seed,
        support_trials=args.support_trials, device=device)
    risky = {
        pair: value for pair, value in separations.items()
        if value < args.warn_below}
    report = {
        "schema": "cue-separability-probe-v1",
        "parent": str(args.parent),
        "tasks": list(tasks),
        "cue_slots": {
            task: _OPERATION_CUE_SLOTS.get(task) for task in tasks},
        "separations": separations,
        "warn_below": args.warn_below,
        "pairs_at_risk": risky,
        "safe": not risky,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
    for pair, value in sorted(separations.items(), key=lambda item: item[1]):
        flag = "  <-- will interfere" if value < args.warn_below else ""
        print(f"{value:8.4f}  {pair}{flag}")
    print(json.dumps({"safe": report["safe"], "pairs_at_risk": risky}))


if __name__ == "__main__":
    main()
