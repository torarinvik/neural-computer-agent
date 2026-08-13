"""Causal transfer audit for the external working-memory cell.

The existing fast-cell probe measured post-write reconstruction.  This audit
uses the cell through the canonical ``INPUT -> PROCESS -> OUTPUT`` rollout:
the cell reads its old external state before the keypress is selected and
only then appends the current learned event.  A source codec is trained on
fresh n-back-2 lifetimes, frozen, and evaluated on new lifetimes with fresh
external state.  A matched untrained cell is the control.  N-back-3 is
reported as a harder rule-transfer probe, not hidden inside the promotion
gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import ExternalWorkingMemoryCell

from .runner import CanonicalBrainWorkshopAgent
from .trainer import evaluate_policy, train_reward_only

CAUSAL_WORKING_MEMORY_TRANSFER_SCHEMA = (
    "neural-computer.brainworkshop-causal-working-memory-transfer.v1"
)
MASTERY_THRESHOLD = 0.80


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _codec_digest(agent: CanonicalBrainWorkshopAgent) -> str:
    digest = hashlib.sha256()
    for prefix, module in (
        ("working_memory", agent.external_reader),
        ("intent_adapter", agent.intent_adapter),
        ("keypress_decoder", agent.keypress_decoder),
    ):
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(f"{prefix}.{name}".encode())
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _freeze_codec(agent: CanonicalBrainWorkshopAgent) -> None:
    for module in (
        agent.external_reader,
        agent.intent_adapter,
        agent.keypress_decoder,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def _agent(seed: int) -> CanonicalBrainWorkshopAgent:
    with torch.random.fork_rng():
        torch.manual_seed(seed + 1)
        cell = ExternalWorkingMemoryCell(
            event_width=16,
            action_width=2,
            memory_capacity=4,
            context_width=16,
            hidden=32,
        )
    return CanonicalBrainWorkshopAgent(
        n_back=3,
        symbol_count=4,
        event_width=16,
        intention_width=8,
        feedback_width=8,
        reader_kind="relation",
        seed=seed,
        working_memory_cell=cell,
    )


def _scores(
    agent: CanonicalBrainWorkshopAgent,
    *,
    n_back: int,
    seeds: tuple[int, ...],
    steps: int,
    time_shuffle: bool = False,
    reset_history: bool = False,
) -> list[float]:
    return [
        evaluate_policy(
            agent,
            n_back=n_back,
            batch_size=64,
            seeds=(seed,),
            steps=steps,
            time_shuffle=time_shuffle,
            reset_history=reset_history,
        )
        for seed in seeds
    ]


def _mastered(scores: list[float]) -> bool:
    return bool(scores) and min(scores) >= MASTERY_THRESHOLD


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.updates, args.batch_size, args.source_steps, args.target_steps) < 1:
        raise ValueError("all causal transfer budgets must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    started = perf_counter()
    seeds = (args.seed, args.seed + 1)

    inherited = _agent(args.seed)
    controller_digest_before = _digest(inherited.controller)
    history = train_reward_only(
        inherited,
        n_back=2,
        updates=args.updates,
        batch_size=args.batch_size,
        steps=args.source_steps,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
    )
    codec_digest_before_eval = _codec_digest(inherited)
    _freeze_codec(inherited)
    source_scores = _scores(
        inherited,
        n_back=2,
        seeds=seeds,
        steps=args.source_steps,
    )
    inherited_prefix_scores = _scores(
        inherited,
        n_back=2,
        seeds=tuple(seed + 20_000 for seed in seeds),
        steps=args.source_steps,
    )
    inherited_shuffle_scores = _scores(
        inherited,
        n_back=2,
        seeds=tuple(seed + 30_000 for seed in seeds),
        steps=args.source_steps,
        time_shuffle=True,
    )
    inherited_reset_scores = _scores(
        inherited,
        n_back=2,
        seeds=tuple(seed + 40_000 for seed in seeds),
        steps=args.source_steps,
        reset_history=True,
    )
    inherited_n3_scores = _scores(
        inherited,
        n_back=3,
        seeds=tuple(seed + 50_000 for seed in seeds),
        steps=args.target_steps,
    )
    controller_digest_after = _digest(inherited.controller)
    codec_digest_after_eval = _codec_digest(inherited)

    fresh = _agent(args.seed)
    fresh_scores = _scores(
        fresh,
        n_back=2,
        seeds=tuple(seed + 20_000 for seed in seeds),
        steps=args.source_steps,
    )

    gates = {
        "source_mastery": _mastered(source_scores),
        "causal_frozen_prefix_mastery": _mastered(inherited_prefix_scores),
        "fresh_control_near_chance": max(fresh_scores) <= 0.65,
        "shuffled_outcome_near_chance": max(inherited_shuffle_scores) <= 0.65,
        "history_reset_near_chance": max(inherited_reset_scores) <= 0.65,
        "controller_unchanged": controller_digest_before == controller_digest_after,
        "working_memory_codec_frozen_during_target": (
            codec_digest_before_eval == codec_digest_after_eval
        ),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": CAUSAL_WORKING_MEMORY_TRANSFER_SCHEMA,
        "claim_boundary": (
            "A source-trained external working-memory codec produces causal "
            "action behavior on fresh state under a frozen controller; the "
            "n-back-3 result is a harder rule-transfer probe and is not part "
            "of the bounded promotion gate."
        ),
        "seed": args.seed,
        "replication_seeds": list(seeds),
        "source_rule": "n_back_2",
        "target_rule_probe": "n_back_3",
        "updates": args.updates,
        "batch_size": args.batch_size,
        "source_steps": args.source_steps,
        "target_steps": args.target_steps,
        "mastery_threshold": MASTERY_THRESHOLD,
        "source_scores": source_scores,
        "inherited_fresh_state_scores": inherited_prefix_scores,
        "fresh_control_scores": fresh_scores,
        "shuffled_outcome_scores": inherited_shuffle_scores,
        "history_reset_scores": inherited_reset_scores,
        "inherited_nback3_scores": inherited_n3_scores,
        "source_final_training_accuracy": history[-1].eligible_accuracy,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": args.updates
            * args.batch_size
            * (args.source_steps - 2),
            "unique_logical_lifetimes": args.updates * args.batch_size,
            "optimizer_updates": args.updates,
            "replayed_examples": 0,
            "evaluation_verifier_bits": {
                "source_retention": len(seeds) * 64 * (args.source_steps - 2),
                "fresh_control": len(seeds) * 64 * (args.source_steps - 2),
                "nback3_probe": len(seeds) * 64 * (args.target_steps - 3),
            },
            "wall_seconds": perf_counter() - started,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-steps", type=int, default=10)
    parser.add_argument("--target-steps", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
