"""No-replay continual learning with isolated executable capability files.

The shared controller learns span two once.  Span three and span four are
then acquired independently in fresh generic growth slots over a frozen
parent.  Each learned slot is persisted as one opaque executable artifact.
At inference, an external memory-side address uses only the controller's
opaque context state and event-window occupancy to select one artifact; it
never sums unrelated skills into the same intention.

This is the controller-as-CPU / memory-as-files design under test.  The
address is a transport-level context signature, not a span or task label, and
the controller receives no raw frames, correct answers, or semantic IDs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    SequenceMemoryBatch,
    generate_sequence_memory_batch,
)
from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalOutputBus,
    ControllerFeedback,
    ExecutableArtifactMemory,
    OpaqueProtocolDecoder,
    load_growth_artifact,
)

from .canonical_growth_pressure_test import (
    FrameEventEncoder,
    _accuracy,
    _artifact,
    _copy_parent_weights,
    _digest_core,
    _freeze_except,
    _rollout,
    _runtime,
)


def _growth_runtime(
    *,
    seed: int,
    width: int = 64,
    growth_recurrent_from: int = 1,
    dynamic_growth: bool = False,
) -> AmodalControllerRuntime:
    if growth_recurrent_from not in {0, 1}:
        raise ValueError("growth_recurrent_from must be 0 or 1")
    if dynamic_growth:
        growth_recurrent_from = 0
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=32,
        workspace_slots=4,
        intention_width=16,
        feedback_width=2,
        event_window_capacity=16,
        reliability_hidden=16,
        growth_register_widths=(width, width),
        growth_prior_only_from=1,
        growth_recurrent_from=growth_recurrent_from,
        growth_gated=dynamic_growth,
        growth_from_intention=dynamic_growth,
        growth_gate_from_context=dynamic_growth,
    )
    return AmodalControllerRuntime(
        controller,
        encoders={"vision": FrameEventEncoder(32)},
        output_bus=AmodalOutputBus(
            {"action": OpaqueProtocolDecoder(16, 2, hidden=16)}
        ),
    )


def _train_current_span(
    runtime: AmodalControllerRuntime,
    *,
    span: int,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    audit_count: int,
    eval_every: int,
    snapshot_policy: str = "final",
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    if snapshot_policy not in {"final", "best_heldout"}:
        raise ValueError("snapshot_policy must be final or best_heldout")
    parameters = [
        parameter
        for name, parameter in runtime.named_parameters()
        if name.startswith("controller.growth_slots.0.") and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("no trainable growth-slot parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    progress: list[dict[str, float | int]] = []
    best_growth_state: dict[str, torch.Tensor] | None = None
    best_heldout_accuracy = float("-inf")
    runtime.train()
    for update in range(1, updates + 1):
        batch = generate_sequence_memory_batch(
            batch_size,
            span=span,
            distractors=1,
            seed=seed + update * 10_007,
            operation="forward",
        )
        result = _rollout(runtime, batch, train=True)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
            }
        )
        if update == updates or (eval_every > 0 and update % eval_every == 0):
            runtime.eval()
            heldout_accuracy = _accuracy(
                runtime,
                operation="forward",
                count=audit_count,
                span=span,
                seed=seed + 1_000_000 + update,
            )
            progress.append(
                {
                    "update": update,
                    "heldout_accuracy": heldout_accuracy,
                }
            )
            if snapshot_policy == "best_heldout" and heldout_accuracy > best_heldout_accuracy:
                best_heldout_accuracy = heldout_accuracy
                best_growth_state = {
                    name: value.detach().clone()
                    for name, value in runtime.controller.state_dict().items()
                    if name.startswith("growth_slots.0.")
                }
            runtime.train()
    if best_growth_state is not None:
        state = runtime.controller.state_dict()
        for name, value in best_growth_state.items():
            state[name].copy_(value)
    runtime.eval()
    return history, progress


def _stable_bits(
    progress: list[dict[str, float | int]],
    *,
    threshold: float,
    bits_per_update: int,
) -> int | None:
    for index, row in enumerate(progress):
        if all(
            float(later["heldout_accuracy"]) >= threshold
            for later in progress[index:]
        ):
            return int(row["update"]) * bits_per_update
    return None


def _quiet_feedback(batch_size: int) -> ControllerFeedback:
    zeros = torch.zeros(batch_size)
    return ControllerFeedback(
        torch.zeros(batch_size, 2), zeros, torch.ones(batch_size), zeros
    )


@torch.no_grad()
def _context_keys(
    runtime: AmodalControllerRuntime,
    batch: SequenceMemoryBatch,
    *,
    occupancy_scale: float,
) -> torch.Tensor:
    """Build opaque route keys from controller state and transport occupancy."""
    state = runtime.initial_state(batch.batch_size, device=batch.input_frames.device)
    feedback = _quiet_feedback(batch.batch_size)
    for frame in batch.input_frames.transpose(0, 1):
        _, state = runtime.step_streams({"vision": frame}, state, feedback)
    for frame in batch.distractor_frames.transpose(0, 1):
        _, state = runtime.step_streams({"vision": frame}, state, feedback)
    occupancy = state.event_window.present.to(state.hidden).float()
    return F.normalize(
        torch.cat([state.hidden, occupancy * occupancy_scale], dim=-1), dim=-1
    )


def _route_key(
    runtime: AmodalControllerRuntime,
    *,
    span: int,
    seed: int,
    count: int,
    occupancy_scale: float,
) -> torch.Tensor:
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation="forward",
    )
    return F.normalize(
        _context_keys(runtime, batch, occupancy_scale=occupancy_scale).mean(dim=0),
        dim=0,
    )


def _selected_row(
    bank: ExecutableArtifactMemory,
    queries: torch.Tensor,
) -> tuple[list[int], float]:
    rows: list[int] = []
    for query in queries:
        handle, _ = bank.promote(query)
        rows.append(handle.index)
    return rows, sum(row >= 0 for row in rows) / max(1, len(rows))


def _load_selected(
    parent: AmodalControllerRuntime,
    bank: ExecutableArtifactMemory,
    row: int,
    *,
    seed: int,
    growth_width: int,
    growth_recurrent_from: int,
    dynamic_growth: bool,
) -> AmodalControllerRuntime:
    runtime = _growth_runtime(
        seed=seed,
        width=growth_width,
        growth_recurrent_from=growth_recurrent_from,
        dynamic_growth=dynamic_growth,
    )
    _copy_parent_weights(parent, runtime)
    _, artifact = bank.promote_index(row)
    load_growth_artifact(
        runtime.controller,
        artifact,
        growth_prefixes=("growth_slots.0.",),
    )
    runtime.eval()
    return runtime


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    stages = tuple(args.stages)
    if stages != tuple(sorted(set(stages))) or len(stages) < 2:
        raise ValueError("stages must contain at least two strictly increasing spans")
    if min(args.updates_per_stage, args.batch_size, args.audit_count, args.growth_width) < 1:
        raise ValueError("updates, batch size, and audit count must be positive")
    if args.growth_recurrent_from not in {0, 1}:
        raise ValueError("growth-recurrent-from must be 0 or 1")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")

    parent = _runtime(seed=args.seed, growth=False)
    parent_parameters = [parameter for parameter in parent.parameters()]
    parent_optimizer = torch.optim.AdamW(
        parent_parameters, lr=args.learning_rate, weight_decay=1e-5
    )
    parent_history: list[dict[str, float | int]] = []
    parent_progress: list[dict[str, float | int]] = []
    parent.train()
    for update in range(1, args.updates_per_stage + 1):
        batch = generate_sequence_memory_batch(
            args.batch_size,
            span=stages[0],
            distractors=1,
            seed=args.seed + 100 + update * 10_007,
            operation="forward",
        )
        result = _rollout(parent, batch, train=True)
        parent_optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parent_parameters, 1.0)
        parent_optimizer.step()
        parent_history.append(
            {
                "update": update,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
            }
        )
        if update == args.updates_per_stage or (
            args.eval_every > 0 and update % args.eval_every == 0
        ):
            parent.eval()
            parent_progress.append(
                {
                    "update": update,
                    "heldout_accuracy": _accuracy(
                        parent,
                        operation="forward",
                        count=args.audit_count,
                        span=stages[0],
                        seed=args.seed + 10_000 + update,
                    ),
                }
            )
            parent.train()
    parent.eval()

    bank_path = args.report.parent / "artifact_memory"
    if bank_path.exists():
        shutil.rmtree(bank_path)
    bank = ExecutableArtifactMemory(
        bank_path,
        width=32 + 16,
        capacity=len(stages),
    )
    zero_runtime = _growth_runtime(
        seed=args.seed,
        width=args.growth_width,
        growth_recurrent_from=args.growth_recurrent_from,
        dynamic_growth=args.dynamic_growth,
    )
    _copy_parent_weights(parent, zero_runtime)
    zero_artifact = _artifact(zero_runtime, "growth_slots.0.")
    route_keys: dict[str, torch.Tensor] = {}
    route_keys[str(stages[0])] = _route_key(
        parent,
        span=stages[0],
        seed=args.seed + 50_000,
        count=args.address_count,
        occupancy_scale=args.occupancy_scale,
    )
    bank.put(route_keys[str(stages[0])], zero_artifact)

    artifacts: dict[str, dict[str, torch.Tensor]] = {}
    stage_history: dict[str, dict[str, list[dict[str, float | int]]]] = {
        str(stages[0]): {
            "training": parent_history,
            "progress": parent_progress,
        }
    }
    for index, span in enumerate(stages[1:], start=1):
        acquired = _growth_runtime(
            seed=args.seed + index,
            width=args.growth_width,
            growth_recurrent_from=args.growth_recurrent_from,
            dynamic_growth=args.dynamic_growth,
        )
        _copy_parent_weights(parent, acquired)
        _freeze_except(acquired, ("growth_slots.0.",))
        history, progress = _train_current_span(
            acquired,
            span=span,
            updates=args.updates_per_stage,
            batch_size=args.batch_size,
            seed=args.seed + 200 * index,
            learning_rate=args.learning_rate,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            snapshot_policy=args.snapshot_policy,
        )
        artifact = _artifact(acquired, "growth_slots.0.")
        artifacts[str(span)] = artifact
        route_keys[str(span)] = _route_key(
            parent,
            span=span,
            seed=args.seed + 50_000 + index,
            count=args.address_count,
            occupancy_scale=args.occupancy_scale,
        )
        bank.put(route_keys[str(span)], artifact)
        stage_history[str(span)] = {
            "training": history,
            "progress": progress,
        }

    bank = ExecutableArtifactMemory.load(bank_path)
    route_accuracy: dict[str, float] = {}
    selected_rows: dict[str, int] = {}
    selected_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    for index, span in enumerate(stages):
        batch = generate_sequence_memory_batch(
            args.audit_count,
            span=span,
            distractors=1,
            seed=args.seed + 60_000 + index,
            operation="forward",
        )
        queries = _context_keys(
            parent,
            batch,
            occupancy_scale=args.occupancy_scale,
        )
        rows, _ = _selected_row(bank, queries)
        expected = index
        route_accuracy[str(span)] = sum(row == expected for row in rows) / len(rows)
        row = max(set(rows), key=rows.count)
        selected_rows[str(span)] = row
        selected_runtime = _load_selected(
            parent,
            bank,
            row,
            seed=args.seed + 70_000 + index,
            growth_width=args.growth_width,
            growth_recurrent_from=args.growth_recurrent_from,
            dynamic_growth=args.dynamic_growth,
        )
        selected_behavior[str(span)] = _accuracy(
            selected_runtime,
            operation="forward",
            count=args.audit_count,
            span=span,
            seed=args.seed + 80_000 + index,
        )
        wrong_runtime = _load_selected(
            parent,
            bank,
            (row + 1) % len(stages),
            seed=args.seed + 90_000 + index,
            growth_width=args.growth_width,
            growth_recurrent_from=args.growth_recurrent_from,
            dynamic_growth=args.dynamic_growth,
        )
        wrong_behavior[str(span)] = _accuracy(
            wrong_runtime,
            operation="forward",
            count=args.audit_count,
            span=span,
            seed=args.seed + 80_000 + index,
        )

    final_runtime = _load_selected(
        parent,
        bank,
        selected_rows[str(stages[-1])],
        seed=args.seed + 100_000,
        growth_width=args.growth_width,
        growth_recurrent_from=args.growth_recurrent_from,
        dynamic_growth=args.dynamic_growth,
    )
    controls = {
        "blank_sequence": _accuracy(
            final_runtime,
            operation="forward",
            count=args.audit_count,
            span=stages[-1],
            seed=args.seed + 110_001,
            blank_sequence=True,
        ),
        "workspace_disabled": _accuracy(
            final_runtime,
            operation="forward",
            count=args.audit_count,
            span=stages[-1],
            seed=args.seed + 110_002,
            disable_workspace=True,
        ),
    }
    stable_bits_to_threshold = {
        str(stages[0]): _stable_bits(
            stage_history[str(stages[0])]["progress"],
            threshold=args.mastery_threshold,
            bits_per_update=args.batch_size * stages[0],
        )
    }
    for span in stages[1:]:
        stable_bits_to_threshold[str(span)] = _stable_bits(
            stage_history[str(span)]["progress"],
            threshold=args.mastery_threshold,
            bits_per_update=args.batch_size * span,
        )
    target_final = selected_behavior[str(stages[-1])]
    prior_retention = {
        str(span): selected_behavior[str(span)] >= args.mastery_threshold
        for span in stages[:-1]
    }
    core_unchanged = _digest_core(
        parent,
        ("growth_slots.0.", "growth_slots.1."),
    ) == _digest_core(final_runtime, ("growth_slots.0.", "growth_slots.1."))
    report = {
        "schema": "canonical-no-replay-artifact-bank-v1",
        "claim_boundary": (
            "A frozen parent and independently acquired generic growth artifacts "
            "retain an arbitrary strictly increasing span curriculum without "
            "replaying old controller examples. A memory-side opaque context "
            "router selects one file; artifacts are never summed."
        ),
        "seed": args.seed,
        "stages": list(stages),
        "updates_per_stage": args.updates_per_stage,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "address_count": args.address_count,
        "occupancy_scale": args.occupancy_scale,
        "growth_width": args.growth_width,
        "growth_recurrent_from": 0 if args.dynamic_growth else args.growth_recurrent_from,
        "dynamic_growth": args.dynamic_growth,
        "snapshot_policy": args.snapshot_policy,
        "artifact_memory": str(bank_path),
        "route_keys": {key: value.tolist() for key, value in route_keys.items()},
        "route_accuracy": route_accuracy,
        "selected_rows": selected_rows,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "prior_retention": prior_retention,
        "stable_bits_to_threshold": stable_bits_to_threshold,
        "controls": controls,
        "history": stage_history,
        "accounting": {
            "unique_logical_lifetimes": args.updates_per_stage * args.batch_size * len(stages),
            "unique_verifier_bits": sum(
                args.updates_per_stage * args.batch_size * span for span in stages
            ),
            "optimizer_updates": args.updates_per_stage * len(stages),
            "replayed_examples": 0,
            "stable_bits_to_threshold": stable_bits_to_threshold,
            "route_calibration_lifetimes": args.address_count * len(stages),
            "diagnostic_lifetimes_charged_to_budget": args.audit_count * len(stages),
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "no_replayed_examples": True,
            "all_context_routes_at_least_90": all(
                value >= 0.90 for value in route_accuracy.values()
            ),
            "all_prior_spans_retained": all(prior_retention.values()),
            "target_span_mastered": (
                target_final >= args.mastery_threshold
                and stable_bits_to_threshold[str(stages[-1])] is not None
            ),
            "wrong_artifact_is_causal": all(
                selected_behavior[key] > wrong_behavior[key] + 0.05
                for key in selected_behavior
                if key != str(stages[0])
            ),
            "blank_sequence_near_chance": controls["blank_sequence"] <= 0.65,
            "workspace_ablation_is_informative": controls["workspace_disabled"] < target_final - 0.05,
            "frozen_parent_core": core_unchanged,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69311)
    parser.add_argument("--stages", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--updates-per-stage", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--address-count", type=int, default=64)
    parser.add_argument("--occupancy-scale", type=float, default=8.0)
    parser.add_argument("--growth-width", type=int, default=64)
    parser.add_argument("--growth-recurrent-from", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dynamic-growth", action="store_true")
    parser.add_argument(
        "--snapshot-policy",
        choices=("final", "best_heldout"),
        default="final",
    )
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "accepted_diagnostic": report["accepted_diagnostic"],
                "route_accuracy": report["route_accuracy"],
                "selected_behavior": report["selected_behavior"],
                "wrong_behavior": report["wrong_behavior"],
                "controls": report["controls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
