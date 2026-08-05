"""Acquire a pure consumer factor over an external producer register.

The first factor is the promoted complement artifact. A second slot is
configured as a generic prior-only consumer: its input contains only the
preceding slot's learned hidden register, never raw event or recurrent-state
features. It is trained outcome-only on verifier-private sequence tasks where
the producer supplies a learned representation and the consumer performs the
new computation. This is a controlled producer->consumer chaining audit, not
a claim of arbitrary program synthesis.
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

from experiments.archive.unified_cognitive_controller.legacy_model import (
    UnifiedCognitiveController,
)
from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
    rollout_sequence_memory,
)
from experiments.working_memory_continuous.acquire_frozen_growth import (
    _slot_prefixes,
)
from neural_computer import (
    ExecutableArtifactMemory,
    compose_growth_artifacts,
    freeze_core,
    load_growth_artifact,
)


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _digest(
    module: torch.nn.Module,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _build_chain(
    payload: dict[str, object],
    *,
    device: torch.device,
) -> tuple[UnifiedCognitiveController, dict[str, object], tuple[str, ...], tuple[str, ...]]:
    configuration = dict(payload["model_configuration"])
    configuration.update(
        {
            "skill_adapter_widths": (256, 256),
            "skill_adapter_reads_prior": True,
            "skill_adapter_reads_prior_from": 1,
            "skill_adapter_prior_only_from": 1,
            "skill_adapter_recurrent_from": 1,
            "skill_adapter_prior_read_limit": 1,
        }
    )
    model = UnifiedCognitiveController(**configuration).to(device)
    producer_prefixes = _slot_prefixes(0)
    consumer_prefixes = _slot_prefixes(1)
    missing, unexpected = model.load_state_dict(
        payload["state_dict"], strict=False
    )
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith(producer_prefixes + consumer_prefixes)
    }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            "producer/consumer checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model, configuration, producer_prefixes, consumer_prefixes


def _load_artifact_from_memory(
    directory: Path,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    memory = ExecutableArtifactMemory.load(directory, device=device)
    rows = memory.address_rows()
    if len(rows) != 1:
        raise ValueError(f"expected one artifact in {directory}")
    _, artifact = memory.promote_index(rows[0][0])
    return artifact


def _zero_prefixes(
    model: UnifiedCognitiveController,
    prefixes: tuple[str, ...],
) -> UnifiedCognitiveController:
    state = model.state_dict()
    for name, value in state.items():
        if name.startswith(prefixes):
            value.zero_()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_into_slot(
    model: UnifiedCognitiveController,
    artifact: dict[str, torch.Tensor],
    *,
    target_slot: int,
    producer_prefixes: tuple[str, ...],
    consumer_prefixes: tuple[str, ...],
) -> dict[str, object]:
    target_prefixes = (
        producer_prefixes if target_slot == 0 else consumer_prefixes
    )
    mapped = compose_growth_artifacts(
        (artifact,),
        prefix_maps=(
            dict(zip(_slot_prefixes(0), target_prefixes, strict=True)),
        ),
    )
    freeze_core(model, producer_prefixes + consumer_prefixes)
    receipt = load_growth_artifact(
        model,
        mapped,
        growth_prefixes=producer_prefixes + consumer_prefixes,
    )
    return {
        "loaded_keys": len(receipt.loaded_keys),
        "core_unchanged": receipt.core_unchanged,
        "target_slot": target_slot,
    }


def _accuracy(
    model: UnifiedCognitiveController,
    *,
    operation: str,
    count: int,
    span: int,
    distractors: int,
    seed: int,
    device: torch.device,
) -> float:
    return float(
        evaluate_sequence_memory(
            model,
            count=count,
            span=span,
            distractors=distractors,
            seed=seed,
            operation=operation,
            device=device,
        )["accuracy"]
    )


def _train_consumer(
    model: UnifiedCognitiveController,
    *,
    consumer_prefixes: tuple[str, ...],
    steps: int,
    batch_size: int,
    span: int,
    distractors: int,
    operation: str,
    shuffle_outcomes: bool,
    seed: int,
    learning_rate: float,
    device: torch.device,
) -> list[dict[str, float | int]]:
    freeze_core(model, consumer_prefixes)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("consumer slot has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    for update in range(1, steps + 1):
        batch = generate_sequence_memory_batch(
            batch_size,
            span=span,
            distractors=distractors,
            seed=seed + update * 10_007,
            operation=operation,
            device=device,
        )
        result = rollout_sequence_memory(
            model,
            batch,
            sample_actions=True,
            exploration=0.10,
            shuffle_outcomes=shuffle_outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "unique_logical_lifetimes": update * batch_size,
                "unique_verifier_bits": update * batch_size * span,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
            }
        )
    model.eval()
    return history


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(args.steps, args.batch_size, args.audit_count, args.span) < 1:
        raise ValueError("steps, batch size, audit count, and span must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    device = torch.device(args.device)
    payload = _load(args.parent, device)
    parent, _, producer_prefixes, consumer_prefixes = _build_chain(
        payload, device=device
    )
    parent_digest = _digest(parent, excluded_prefixes=producer_prefixes + consumer_prefixes)

    producer_artifact = _load_artifact_from_memory(
        args.producer_memory, device=device
    )
    producer, _, producer_prefixes, consumer_prefixes = _build_chain(
        payload, device=device
    )
    producer_receipt = _load_into_slot(
        producer,
        producer_artifact,
        target_slot=0,
        producer_prefixes=producer_prefixes,
        consumer_prefixes=consumer_prefixes,
    )
    producer.eval()

    child, _, child_producer_prefixes, child_consumer_prefixes = _build_chain(
        payload, device=device
    )
    _load_into_slot(
        child,
        producer_artifact,
        target_slot=0,
        producer_prefixes=child_producer_prefixes,
        consumer_prefixes=child_consumer_prefixes,
    )
    history = _train_consumer(
        child,
        consumer_prefixes=child_consumer_prefixes,
        steps=args.steps,
        batch_size=args.batch_size,
        span=args.span,
        distractors=args.distractors,
        operation=args.operation,
        shuffle_outcomes=args.shuffle_outcomes,
        seed=args.seed,
        learning_rate=args.learning_rate,
        device=device,
    )
    child.eval()
    raw_consumer_artifact = {
        name: value.detach().cpu().clone()
        for name, value in child.state_dict().items()
        if name.startswith(child_consumer_prefixes)
    }
    if not raw_consumer_artifact:
        raise RuntimeError("consumer training produced no artifact")
    # Store every consumer artifact in the canonical source-slot namespace so
    # the memory backend never learns the destination slot of a future load.
    consumer_artifact = compose_growth_artifacts(
        (raw_consumer_artifact,),
        prefix_maps=(
            dict(zip(
                child_consumer_prefixes,
                _slot_prefixes(0),
                strict=True,
            )),
        ),
    )
    if args.consumer_memory.exists():
        shutil.rmtree(args.consumer_memory)
    consumer_memory = ExecutableArtifactMemory(
        args.consumer_memory,
        width=64,
        capacity=1,
        device=device,
    )
    consumer_key = F.normalize(
        torch.randn(64, generator=torch.Generator().manual_seed(args.seed + 1)),
        dim=0,
    )
    consumer_memory.put(consumer_key, consumer_artifact)
    consumer_memory = ExecutableArtifactMemory.load(
        args.consumer_memory, device=device
    )
    _, reloaded_consumer_artifact = consumer_memory.promote(consumer_key)

    fresh_composed, _, fresh_producer_prefixes, fresh_consumer_prefixes = _build_chain(
        payload, device=device
    )
    _load_into_slot(
        fresh_composed,
        producer_artifact,
        target_slot=0,
        producer_prefixes=fresh_producer_prefixes,
        consumer_prefixes=fresh_consumer_prefixes,
    )
    _load_into_slot(
        fresh_composed,
        reloaded_consumer_artifact,
        target_slot=1,
        producer_prefixes=fresh_producer_prefixes,
        consumer_prefixes=fresh_consumer_prefixes,
    )
    fresh_composed.eval()

    # A fresh consumer-only model proves that the consumer cannot bypass the
    # producer: its only input is the zero producer register.
    consumer_only, _, consumer_only_producer_prefixes, consumer_only_consumer_prefixes = _build_chain(
        payload, device=device
    )
    _load_into_slot(
        consumer_only,
        reloaded_consumer_artifact,
        target_slot=1,
        producer_prefixes=consumer_only_producer_prefixes,
        consumer_prefixes=consumer_only_consumer_prefixes,
    )
    consumer_only.eval()

    producer_zeroed = _zero_prefixes(
        _build_chain(payload, device=device)[0], producer_prefixes
    )
    _load_into_slot(
        producer_zeroed,
        reloaded_consumer_artifact,
        target_slot=1,
        producer_prefixes=producer_prefixes,
        consumer_prefixes=consumer_prefixes,
    )
    producer_zeroed.eval()

    audit_models = {
        "parent": parent,
        "producer_only": producer,
        "consumer_only": consumer_only,
        "composed": fresh_composed,
        "producer_zeroed": producer_zeroed,
    }
    behavior = {
        name: _accuracy(
            model,
            count=args.audit_count,
            span=args.span,
            distractors=args.distractors,
            operation=args.operation,
            seed=args.seed + 50_000,
            device=device,
        )
        for name, model in audit_models.items()
    }
    read_ablated = fresh_composed
    read_ablated.skill_adapter_ablate_prior_read_slot = 1
    read_ablated_accuracy = _accuracy(
        read_ablated,
        count=args.audit_count,
        span=args.span,
        distractors=args.distractors,
        operation=args.operation,
        seed=args.seed + 50_000,
        device=device,
    )
    read_ablated.skill_adapter_ablate_prior_read_slot = None
    composed_suite = evaluate_sequence_memory(
        fresh_composed,
        count=args.audit_count,
        span=args.span,
        distractors=args.distractors,
        seed=args.seed + 50_000,
        operation=args.operation,
        device=device,
    )

    composed_digest = _digest(
        fresh_composed,
        excluded_prefixes=fresh_producer_prefixes + fresh_consumer_prefixes,
    )
    consumer_reload_exact = all(
        torch.equal(consumer_artifact[name], reloaded_consumer_artifact[name])
        for name in consumer_artifact
    )
    controller_core_unchanged = parent_digest == composed_digest
    report = {
        "schema": "prior-only-consumer-growth-audit-v1",
        "claim_boundary": (
            "A generic prior-only consumer slot can be trained from scalar "
            "outcomes over a promoted producer artifact and can contribute "
            "to a new verifier-private sequence-level computation. This does "
            "not qualify arbitrary factor algebra or unrestricted programs."
        ),
        "parent": str(args.parent),
        "producer_memory": str(args.producer_memory),
        "consumer_memory": str(args.consumer_memory),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "span": args.span,
        "distractors": args.distractors,
        "operation": args.operation,
        "shuffle_outcomes": args.shuffle_outcomes,
        "producer_input_width": child.skill_adapters[0][0].in_features,
        "consumer_input_width": child.skill_adapters[1][0].in_features,
        "consumer_raw_bypass": child.skill_adapters[1][0].in_features > 256,
        "producer_receipt": producer_receipt,
        "consumer_artifact_entries": len(consumer_artifact),
        "consumer_reload_exact": consumer_reload_exact,
        "behavior": {
            **behavior,
            "consumer_prior_read_ablated": read_ablated_accuracy,
        },
        "composed_causal_audit": {
            key: composed_suite[key]
            for key in (
                "accuracy",
                "blank_sequence_accuracy",
                "blank_operation_cue_accuracy",
                "active_state_reset_accuracy",
                "all_memory_reset_accuracy",
                "reverse_operation_accuracy",
                "reverse_sequence_accuracy",
                "zero_distractor_accuracy",
                "workspace_disabled_accuracy",
            )
        },
        "history": history,
        "accounting": {
            "unique_logical_lifetimes": args.steps * args.batch_size,
            "unique_verifier_bits": args.steps * args.batch_size * args.span,
            "verifier_bits": args.steps * args.batch_size * args.span,
            "optimizer_updates": args.steps,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
            "stable_bits_to_threshold": None,
            "retention_on_mastered_primitives": None,
            "transfer_ratio_against_fresh_learner": None,
        },
        "controller_core_unchanged": controller_core_unchanged,
        "gates": {
            "producer_loaded": producer_receipt["loaded_keys"] > 0,
            "consumer_input_is_prior_only": (
                child.skill_adapters[1][0].in_features == 256
                and child.skill_adapter_reads_prior_from == 1
                and child.skill_adapter_prior_only_from == 1
            ),
            "consumer_has_no_raw_bypass": (
                child.skill_adapters[1][0].in_features == 256
            ),
            "consumer_reload_exact": consumer_reload_exact,
            "controller_core_unchanged": controller_core_unchanged,
            "consumer_beats_parent": (
                behavior["composed"] > behavior["parent"] + 0.05
            ),
            "consumer_beats_producer_only": (
                behavior["composed"] > behavior["producer_only"] + 0.05
            ),
            "consumer_only_near_chance": (
                abs(behavior["consumer_only"] - 0.5) <= 0.10
            ),
            "producer_ablation_is_causal": (
                behavior["producer_zeroed"]
                < behavior["composed"] - 0.05
            ),
            "prior_read_ablation_is_causal": (
                read_ablated_accuracy < behavior["composed"] - 0.05
            ),
            "missing_evidence_near_chance": (
                composed_suite["blank_sequence_accuracy"] <= 0.65
            ),
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--producer-memory", type=Path, required=True)
    parser.add_argument("--consumer-memory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69101)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--span", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=1)
    parser.add_argument(
        "--operation",
        choices=("undo_complement", "producer_global_parity"),
        default="undo_complement",
    )
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "accepted_diagnostic": report["accepted_diagnostic"],
        "behavior": report["behavior"],
        "gates": report["gates"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
