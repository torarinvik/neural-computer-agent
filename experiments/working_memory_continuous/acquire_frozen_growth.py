"""Acquire one working-memory procedure into isolated external growth state.

This is the next rung after artifact persistence.  A compatibility fixture
provides a parent that already solves the inherited working-memory frontier.
The controller is frozen, a zero-output generic successor slot is appended,
and only that slot is trained from rendered events, opaque attempted actions,
and scalar verifier outcomes.  The learned slot is then written as an opaque
artifact, reloaded in a fresh model instance, and compared with the live
learner under retention and causal-ablation controls.

The archived controller remains a measurement fixture until the canonical
amodal runtime exposes the same generic growth-state execution seam.  The
artifact store and frozen-core loader used here are production interfaces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.audit_sequence_multi_skill_bank import (
    _model,
)
from experiments.archive.unified_cognitive_controller.legacy_model import (
    UnifiedCognitiveController,
)
from experiments.archive.unified_cognitive_controller.train_pair_relation_appearance_bridge import (
    _slot_prefixes,
)
from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
    rollout_sequence_memory,
)
from neural_computer import (
    ExecutableArtifactMemory,
    freeze_core,
    load_growth_artifact,
)


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _state_digest(
    model: torch.nn.Module,
    excluded_prefixes: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _address_key(
    model: torch.nn.Module,
    batch,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Derive an opaque address from the complete public write encounter."""
    state = model.initial_state(batch.batch_size, device=device)
    null = torch.full(
        (batch.batch_size,), 2, dtype=torch.long, device=device)
    zeros = torch.zeros(batch.batch_size, device=device)
    for index in range(batch.span):
        _, state = model.step(
            batch.input_frames[:, index], state, null, zeros, zeros
        )
    for index in range(batch.distractor_frames.shape[1]):
        _, state = model.step(
            batch.distractor_frames[:, index], state, null, zeros, zeros
        )
    _, state = model.step(
        batch.query_frames[:, 0], state, null, zeros, zeros
    )
    return F.normalize(state.hidden.mean(dim=0), dim=0).cpu()


def _growth_state(
    model: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def _build_successor(
    payload: dict[str, object],
    *,
    device: torch.device,
    slot_width: int,
) -> tuple[UnifiedCognitiveController, dict[str, object], int, tuple[str, ...]]:
    """Append one zero-output slot to any compatible parent, including zero."""
    configuration = dict(payload["model_configuration"])
    inherited = tuple(configuration.get("skill_adapter_widths", ()))
    slot = len(inherited)
    configuration["skill_adapter_widths"] = (*inherited, slot_width)
    prefixes = _slot_prefixes(slot)
    successor = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = successor.load_state_dict(
        payload["state_dict"], strict=False
    )
    expected_missing = {
        name for name in successor.state_dict() if name.startswith(prefixes)
    }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            "successor checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return successor, configuration, slot, prefixes


@torch.no_grad()
def _probe_insertion_behavior(
    parent: torch.nn.Module,
    child: torch.nn.Module,
    batch,
) -> bool:
    parent_result = rollout_sequence_memory(
        parent, batch, sample_actions=False)
    child_result = rollout_sequence_memory(
        child, batch, sample_actions=False)
    return bool(
        torch.equal(parent_result["logits"], child_result["logits"])
        and torch.equal(
            parent_result["final_workspace"], child_result["final_workspace"]
        )
    )


def _audit(
    model: torch.nn.Module,
    *,
    spans: tuple[int, ...],
    count: int,
    distractors: int,
    seed: int,
    device: torch.device,
    target_span: int,
    target_operation: str,
) -> dict[str, dict[str, object]]:
    return {
        str(span): evaluate_sequence_memory(
            model,
            count=count,
            span=span,
            distractors=distractors,
            seed=seed + index * 100_003,
            operation=(target_operation if span == target_span else "mixed"),
            device=device,
        )
        for index, span in enumerate(spans)
    }


def _corruption_control(
    memory: ExecutableArtifactMemory,
    key: torch.Tensor,
    destination: Path,
    *,
    device: torch.device,
) -> bool:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(memory.directory, destination)
    filename = memory.paths[0]
    if filename is None:
        raise RuntimeError("stored growth artifact has no path")
    path = destination / filename
    path.write_bytes(path.read_bytes() + b"corruption")
    try:
        reloaded = ExecutableArtifactMemory.load(destination, device=device)
        reloaded.promote(key)
    except ValueError as error:
        return "hash mismatch" in str(error)
    return False


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.target_span in args.rehearse_spans:
        raise ValueError("target span must not be rehearsed")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch and audit counts must be even")
    if min(args.steps, args.batch_size, args.audit_count) < 1:
        raise ValueError("steps, batch size, and audit count must be positive")

    device = torch.device(args.device)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    parent_payload = _load(args.parent, device)
    parent = _model(parent_payload, device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    child, configuration, slot, prefixes = _build_successor(
        parent_payload,
        device=device,
        slot_width=args.slot_width,
    )
    freeze_core(child, prefixes)
    trainable = [
        parameter for parameter in child.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("successor slot has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-5
    )

    insertion_batch = generate_sequence_memory_batch(
        args.batch_size,
        span=args.target_span,
        distractors=args.distractors,
        seed=args.seed + 1_000,
        operation=args.target_operation,
        device=device,
    )
    initial_behavior_exact = _probe_insertion_behavior(
        parent, child, insertion_batch
    )
    parent_core_digest = _state_digest(parent)
    child_core_digest_before = _state_digest(child, prefixes)

    schedule = (args.target_span, *args.rehearse_spans)
    history: list[dict[str, float | int]] = []
    seen_bits = 0
    started = perf_counter()
    for step in range(1, args.steps + 1):
        train_span = schedule[(step - 1) % len(schedule)]
        is_target = train_span == args.target_span
        batch = generate_sequence_memory_batch(
            args.batch_size,
            span=train_span,
            distractors=args.distractors,
            seed=args.seed + step * 10_007,
            operation=(args.target_operation if is_target else "mixed"),
            device=device,
        )
        result = rollout_sequence_memory(
            child,
            batch,
            sample_actions=True,
            exploration=args.exploration,
            shuffle_outcomes=args.shuffle_outcomes,
            loss_mode=args.loss_mode,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = result["loss"]
        if args.distill_old_weight and args.rehearse_spans:
            for rehearsal_index, rehearsal_span in enumerate(args.rehearse_spans):
                rehearsal_batch = generate_sequence_memory_batch(
                    args.batch_size,
                    span=rehearsal_span,
                    distractors=args.distractors,
                    seed=(
                        args.seed
                        + step * 20_011
                        + rehearsal_index * 7_919
                    ),
                    operation="mixed",
                    device=device,
                )
                rehearsal_result = rollout_sequence_memory(
                    child,
                    rehearsal_batch,
                    sample_actions=False,
                    return_slot_activity=bool(args.old_residual_penalty),
                )
                with torch.no_grad():
                    parent_logits = rollout_sequence_memory(
                        parent,
                        rehearsal_batch,
                        sample_actions=False,
                    )["logits"]
                loss = loss + args.distill_old_weight * F.mse_loss(
                    rehearsal_result["logits"], parent_logits
                )
                if args.old_residual_penalty:
                    residuals = rehearsal_result.get(
                        "skill_adapter_residual_norms"
                    )
                    if residuals is None:
                        raise RuntimeError(
                            "successor activity was not returned for penalty"
                        )
                    loss = loss + args.old_residual_penalty * residuals[..., -1].square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        seen_bits += args.batch_size * train_span
        history.append(
            {
                "update": step,
                "train_span": train_span,
                "unique_logical_lifetimes": step * args.batch_size,
                "unique_verifier_bits": seen_bits,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(loss.detach()),
            }
        )

    child.eval()
    child_core_digest_after = _state_digest(child, prefixes)
    artifact = _growth_state(child, prefixes)
    if not artifact:
        raise RuntimeError("trained successor did not produce growth state")

    address_batch = generate_sequence_memory_batch(
        args.batch_size,
        span=args.target_span,
        distractors=args.distractors,
        seed=args.seed + 2_000,
        operation=args.target_operation,
        device=device,
    )
    key = _address_key(parent, address_batch, device=device)
    memory = ExecutableArtifactMemory(
        args.memory,
        width=int(key.numel()),
        capacity=1,
        device=device,
        write_match_threshold=args.write_match_threshold,
    )
    memory.put(key, artifact)
    memory.validate()
    reloaded_memory = ExecutableArtifactMemory.load(args.memory, device=device)
    handle, loaded_artifact = reloaded_memory.promote(key)

    rehydrated, _, _, rehydrated_prefixes = _build_successor(
        parent_payload,
        device=device,
        slot_width=args.slot_width,
    )
    if rehydrated_prefixes != prefixes:
        raise RuntimeError("fresh growth state has a different slot boundary")
    freeze_core(rehydrated, prefixes)
    receipt = load_growth_artifact(
        rehydrated,
        loaded_artifact,
        growth_prefixes=prefixes,
    )
    rehydrated.eval()

    zeroed = _model(
        {"model_configuration": configuration, "state_dict": child.state_dict()},
        device,
    )
    with torch.no_grad():
        zeroed_state = zeroed.state_dict()
        for name in artifact:
            zeroed_state[name].zero_()
    zeroed.load_state_dict(zeroed_state, strict=True)
    zeroed.eval()

    audit_spans = (*args.rehearse_spans, args.target_span)
    parent_audit = _audit(
        parent,
        spans=audit_spans,
        count=args.audit_count,
        distractors=args.distractors,
        seed=args.seed + 3_000_000,
        device=device,
        target_span=args.target_span,
        target_operation=args.target_operation,
    )
    child_audit = _audit(
        child,
        spans=audit_spans,
        count=args.audit_count,
        distractors=args.distractors,
        seed=args.seed + 3_000_000,
        device=device,
        target_span=args.target_span,
        target_operation=args.target_operation,
    )
    rehydrated_audit = _audit(
        rehydrated,
        spans=audit_spans,
        count=args.audit_count,
        distractors=args.distractors,
        seed=args.seed + 3_000_000,
        device=device,
        target_span=args.target_span,
        target_operation=args.target_operation,
    )
    zeroed_audit = _audit(
        zeroed,
        spans=audit_spans,
        count=args.audit_count,
        distractors=args.distractors,
        seed=args.seed + 3_000_000,
        device=device,
        target_span=args.target_span,
        target_operation=args.target_operation,
    )
    retention_deltas = {
        str(span): float(
            child_audit[str(span)]["accuracy"]
            - parent_audit[str(span)]["accuracy"]
        )
        for span in args.rehearse_spans
    }
    target_key = str(args.target_span)
    target_gain = float(
        child_audit[target_key]["accuracy"]
        - parent_audit[target_key]["accuracy"]
    )
    corruption_rejected = _corruption_control(
        memory,
        key,
        args.report.parent / "memory_corrupted",
        device=device,
    )
    wall_seconds = perf_counter() - started
    report = {
        "schema": "frozen-controller-external-growth-acquisition-v1",
        "claim_boundary": (
            "The parent controller is frozen. Only one generic successor "
            "growth boundary is trained from rendered events, opaque "
            "attempted actions, and scalar outcomes. The artifact store "
            "does not interpret the learned tensors. This is not a claim "
            "of cold-start address discovery or general cognition."),
        "parent": str(args.parent),
        "memory": str(args.memory),
        "seed": args.seed,
        "target_span": args.target_span,
        "target_operation": args.target_operation,
        "rehearse_spans": list(args.rehearse_spans),
        "slot": slot,
        "slot_width": args.slot_width,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "learning_rate": args.learning_rate,
        "exploration": args.exploration,
        "loss_mode": args.loss_mode,
        "distill_old_weight": args.distill_old_weight,
        "old_residual_penalty": args.old_residual_penalty,
        "shuffle_outcomes": args.shuffle_outcomes,
        "unique_logical_lifetimes": args.steps * args.batch_size,
        "unique_verifier_bits": seen_bits,
        "optimizer_updates": args.steps,
        "replayed_examples": 0,
        "wall_seconds": wall_seconds,
        "latency_seconds_per_update": wall_seconds / args.steps,
        "gpu_seconds": None,
        "stable_bits_to_threshold": None,
        "retention_on_mastered_primitives": retention_deltas,
        "transfer_ratio_against_fresh_learner": None,
        "history": history,
        "artifact_tensor_names": sorted(artifact),
        "artifact_address_width": int(key.numel()),
        "artifact_route": {
            "index": handle.index,
            "confidence": handle.confidence,
            "margin": handle.margin,
            "version": handle.version,
        },
        "initial_behavior_exact": initial_behavior_exact,
        "parent_core_digest": parent_core_digest,
        "child_core_digest_before": child_core_digest_before,
        "child_core_digest_after": child_core_digest_after,
        "core_unchanged_during_training": (
            child_core_digest_before == child_core_digest_after
        ),
        "rehydrated_core_unchanged": receipt.core_unchanged,
        "parent_audit": parent_audit,
        "child_audit": child_audit,
        "rehydrated_audit": rehydrated_audit,
        "zeroed_growth_audit": zeroed_audit,
        "retention_deltas": retention_deltas,
        "target_gain": target_gain,
        "rehydrated_matches_child": all(
            rehydrated_audit[str(span)]["accuracy"]
            == child_audit[str(span)]["accuracy"]
            for span in audit_spans
        ),
        "corruption_rejected": corruption_rejected,
        "gates": {
            "initial_behavior_preserved": initial_behavior_exact,
            "core_unchanged": (
                child_core_digest_before == child_core_digest_after
                and receipt.core_unchanged
            ),
            "rehydration_matches_child": all(
                rehydrated_audit[str(span)]["accuracy"]
                == child_audit[str(span)]["accuracy"]
                for span in audit_spans
            ),
            "growth_is_causally_used": (
                zeroed_audit[target_key]["accuracy"]
                < child_audit[target_key]["accuracy"] - 0.05
            ),
            "retention_within_two_points": all(
                delta >= -0.02 for delta in retention_deltas.values()
            ),
            "corruption_rejected": corruption_rejected,
        },
    }
    report["accepted_smoke"] = all(report["gates"].values())
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument("--target-span", type=int, default=9)
    parser.add_argument(
        "--target-operation",
        choices=(
            "mixed", "forward", "reverse", "complement",
            "complement_reverse", "complement_rotate", "adjacent_xor",
            "prefix_parity", "global_parity", "rotate", "undo_complement",
            "producer_global_parity",
        ),
        default="mixed",
    )
    parser.add_argument("--rehearse-spans", type=int, nargs="*", default=(2, 4, 6, 8))
    parser.add_argument("--slot-width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=32)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument(
        "--loss-mode",
        choices=("bce", "reinforce", "success_only"),
        default="bce",
    )
    parser.add_argument(
        "--distill-old-weight",
        type=float,
        default=0.0,
        help="trainer-only parent-logit rehearsal on inherited spans",
    )
    parser.add_argument(
        "--old-residual-penalty",
        type=float,
        default=0.0,
        help="trainer-only penalty on successor activity during rehearsal",
    )
    parser.add_argument("--write-match-threshold", type=float, default=0.999)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "accepted_smoke": report["accepted_smoke"],
        "target_gain": report["target_gain"],
        "unique_verifier_bits": report["unique_verifier_bits"],
        "gates": report["gates"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
