"""Pressure-test online admission of an unregistered alignment transform."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from experiments.external_register_composition_amodal.train import (
    _batch,
    _module_digest,
    _rollout,
    _stable_bits,
)
from experiments.outcome_only_alignment_cell_stream.train import (
    ExternalAlignmentCellBank,
    OpaqueAlignmentRouter,
    TRANSFORM_SEEDS,
    _evaluate_router,
    _return_scores,
    _score,
    _source_setup,
    _train_cell,
    _train_router,
)
from experiments.outcome_only_alignment_cell_stream.train import (
    ACTION_WIDTH,
    EVENT_WIDTH,
    OPERATION,
)


ONLINE_TRANSFORM_SEED = 710_123
MASTERY_THRESHOLD = 0.8


class ExternalAlignmentKeyBank(nn.Module):
    """Immutable opaque event signatures used for protected cell addressing."""

    schema = "neural-computer.external-alignment-key-bank.v1"

    def __init__(self, context_width: int) -> None:
        super().__init__()
        self.context_width = int(context_width)
        self.register_buffer("keys", torch.empty(0, context_width))
        self.logical_ids: list[str] = []

    def add(self, logical_id: str, key: torch.Tensor) -> None:
        if not logical_id or logical_id in self.logical_ids:
            raise ValueError("alignment key logical ID must be unique and nonempty")
        if key.ndim != 1 or key.shape[0] != self.context_width:
            raise ValueError("alignment key has the wrong shape")
        if not bool(torch.isfinite(key).all()):
            raise ValueError("alignment key must be finite")
        self.keys = torch.cat(
            (self.keys, key.detach().to(self.keys).unsqueeze(0)),
            dim=0,
        )
        self.logical_ids.append(logical_id)

    def select(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError("alignment key query has the wrong shape")
        if not len(self.logical_ids):
            raise ValueError("cannot select from an empty alignment key bank")
        return torch.cdist(context, self.keys).argmin(dim=-1)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "context_width": self.context_width,
            "logical_ids": tuple(self.logical_ids),
        }


@torch.no_grad()
def _estimate_key(parent, transform_seed: int, *, seed: int) -> torch.Tensor:
    contexts = []
    for index in range(16):
        batch = _batch(
            OPERATION,
            count=16,
            span=4,
            seed=seed + index * 1009,
        )
        contexts.append(
            _context_for_transform(parent, batch, transform_seed).mean(dim=0)
        )
    return torch.stack(contexts).mean(dim=0)


@torch.no_grad()
def _evaluate_key_router(
    parent,
    machine,
    decoder,
    bank: ExternalAlignmentCellBank,
    key_bank: ExternalAlignmentKeyBank,
    *,
    transform_seeds: tuple[int, ...],
    count: int,
    span: int,
    seed: int,
) -> dict[str, object]:
    rows = []
    for target_index, transform_seed in enumerate(transform_seeds):
        batch = _batch(
            OPERATION,
            count=count,
            span=span,
            seed=seed + target_index * 1009,
        )
        context = _context_for_transform(parent, batch, transform_seed).mean(
            dim=0,
            keepdim=True,
        )
        selected_index = int(key_bank.select(context).item())
        action_accuracy = _score(
            parent,
            machine,
            decoder,
            bank.cell(f"cell_{selected_index}"),
            transform_seed=transform_seed,
            count=count,
            span=span,
            seed=seed + target_index * 1009,
        )
        rows.append(
            {
                "target_index": target_index,
                "selected_index": selected_index,
                "routing_correct": selected_index == target_index,
                "action_accuracy": action_accuracy,
            }
        )
    return {
        "rows": rows,
        "routing_accuracy": sum(
            bool(row["routing_correct"]) for row in rows
        ) / len(rows),
        "action_mastery": all(
            float(row["action_accuracy"]) >= MASTERY_THRESHOLD for row in rows
        ),
    }


def _expand_router(
    router: OpaqueAlignmentRouter,
    *,
    old_count: int,
) -> OpaqueAlignmentRouter:
    expanded = OpaqueAlignmentRouter(
        router.network[0].in_features,
        old_count + 1,
        hidden=router.network[0].out_features,
    )
    expanded.network[0].load_state_dict(router.network[0].state_dict())
    expanded.network[2].weight.data[:old_count].copy_(
        router.network[2].weight.data
    )
    expanded.network[2].bias.data[:old_count].copy_(
        router.network[2].bias.data
    )
    for parameter in expanded.network[:-1].parameters():
        parameter.requires_grad_(False)
    return expanded


def _train_new_router_head(
    parent,
    machine,
    decoder,
    bank: ExternalAlignmentCellBank,
    router: OpaqueAlignmentRouter,
    *,
    transform_seed: int,
    new_index: int,
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    learning_rate: float,
    eval_every: int,
    shuffle_outcomes: bool,
) -> list[dict[str, float | int]]:
    """Train only the new head; restore old rows after every optimizer step."""

    output_layer = router.network[2]
    old_weight = output_layer.weight.data[:new_index].clone()
    old_bias = output_layer.bias.data[:new_index].clone()
    optimizer = torch.optim.AdamW(
        [output_layer.weight, output_layer.bias],
        lr=learning_rate,
        weight_decay=0.0,
    )
    progress: list[dict[str, float | int]] = []
    baseline = 0.5
    previous_reward = 0.5
    for update in range(1, updates + 1):
        batch = _batch(
            OPERATION,
            count=batch_size,
            span=span,
            seed=seed + update * 10_007,
        )
        context = _context_for_transform(parent, batch, transform_seed).mean(
            dim=0,
            keepdim=True,
        )
        distribution = torch.distributions.Categorical(logits=router(context))
        selected_index = distribution.sample()
        with torch.no_grad():
            _, rewards = _rollout(
                parent,
                machine,
                decoder,
                batch,
                (machine.instructions[0],),
                basis_slots=(0,),
                train_decoder=False,
                credit_mode="attempted_bce",
                event_bridge=bank.cell(f"cell_{int(selected_index.item())}"),
                bridge_event_mode="composed_orthogonal",
                bridge_state_mode="zero",
                bridge_transform_seed=transform_seed,
            )
        reward = float(rewards.mean())
        delivered_reward = previous_reward if shuffle_outcomes else reward
        advantage = delivered_reward - baseline
        loss = -advantage * distribution.log_prob(selected_index).mean()
        loss = loss - 0.01 * distribution.entropy().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            output_layer.weight[:new_index].copy_(old_weight)
            output_layer.bias[:new_index].copy_(old_bias)
        baseline = 0.95 * baseline + 0.05 * delivered_reward
        previous_reward = reward
        if update % eval_every == 0 or update == updates:
            progress.append(
                {
                    "update": update,
                    "delivered_reward": delivered_reward,
                    "router_loss": float(loss.detach()),
                }
            )
    return progress


@torch.no_grad()
def _context_for_transform(parent, batch, transform_seed: int) -> torch.Tensor:
    from experiments.external_register_composition_amodal.train import (
        _apply_bridge_event_transform,
    )

    frames = torch.cat(
        (batch.input_frames, batch.distractor_frames, batch.query_frames),
        dim=1,
    )
    batch_size, frame_count = frames.shape[:2]
    encoded = parent.encoders["vision"](
        frames.reshape(batch_size * frame_count, *frames.shape[2:])
    ).reshape(batch_size, frame_count, EVENT_WIDTH)
    transformed = _apply_bridge_event_transform(
        encoded.reshape(-1, EVENT_WIDTH),
        "composed_orthogonal",
        transform_seed,
    ).reshape_as(encoded)
    return torch.cat(
        (transformed.mean(dim=1), transformed.std(dim=1, unbiased=False)),
        dim=-1,
    )


def _initial_cells(args, parent, machine, decoder) -> tuple[
    ExternalAlignmentCellBank,
    list[int | None],
    list[dict[str, object]],
]:
    bank = ExternalAlignmentCellBank()
    stable_bits: list[int | None] = []
    reports: list[dict[str, object]] = []
    for index, transform_seed in enumerate(TRANSFORM_SEEDS):
        bridge, progress = _train_cell(
            parent,
            machine,
            decoder,
            transform_seed=transform_seed,
            updates=args.bridge_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 40_000 + index * 100_003,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
            shuffle_outcomes=False,
        )
        shuffled_bridge, shuffled_progress = _train_cell(
            parent,
            machine,
            decoder,
            transform_seed=transform_seed,
            updates=args.bridge_updates,
            batch_size=args.batch_size,
            span=args.span,
            seed=args.seed + 60_000 + index * 100_003,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            audit_count=args.audit_count,
            shuffle_outcomes=True,
        )
        bank.add(f"cell_{index}", bridge)
        bank.freeze(f"cell_{index}")
        stable = _stable_bits(
            progress,
            threshold=MASTERY_THRESHOLD,
            bits_per_update=args.batch_size * args.span,
        )
        stable_bits.append(stable)
        reports.append(
            {
                "logical_id": f"cell_{index}",
                "transform_seed": transform_seed,
                "target_after": _score(
                    parent,
                    machine,
                    decoder,
                    bridge,
                    transform_seed=transform_seed,
                    count=args.audit_count,
                    span=args.span,
                    seed=args.seed + 50_000 + index * 1009,
                ),
                "shuffled_after": _score(
                    parent,
                    machine,
                    decoder,
                    shuffled_bridge,
                    transform_seed=transform_seed,
                    count=args.audit_count,
                    span=args.span,
                    seed=args.seed + 70_000 + index * 1009,
                ),
                "stable_bits_to_threshold": stable,
                "shuffled_stable_bits_to_threshold": _stable_bits(
                    shuffled_progress,
                    threshold=MASTERY_THRESHOLD,
                    bits_per_update=args.batch_size * args.span,
                ),
            }
        )
    return bank, stable_bits, reports


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    if min(
        args.parent_updates,
        args.source_updates,
        args.source_restarts,
        args.bridge_updates,
        args.router_updates,
        args.growth_router_updates,
        args.batch_size,
        args.span,
        args.audit_count,
        args.eval_every,
    ) < 1:
        raise ValueError("all update, batch, span, and audit counts must be positive")
    (
        parent,
        machine,
        decoder,
        parent_digest_before,
        machine_digest_before,
        decoder_digest_before,
        source_accuracy,
        source_attempts,
    ) = _source_setup(args)
    bank, initial_stable_bits, initial_reports = _initial_cells(
        args,
        parent,
        machine,
        decoder,
    )
    initial_router, initial_router_progress = _train_router(
        parent,
        machine,
        decoder,
        bank,
        transform_seeds=TRANSFORM_SEEDS,
        updates=args.router_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 110_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        shuffle_outcomes=False,
    )
    initial_routing = _evaluate_router(
        parent,
        machine,
        decoder,
        bank,
        initial_router,
        transform_seeds=TRANSFORM_SEEDS,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 120_000,
    )
    key_bank = ExternalAlignmentKeyBank(EVENT_WIDTH * 2)
    for index, transform_seed in enumerate(TRANSFORM_SEEDS):
        key_bank.add(
            f"cell_{index}",
            _estimate_key(
                parent,
                transform_seed,
                seed=args.seed + 130_000 + index * 10_003,
            ),
        )
    initial_key_routing = _evaluate_key_router(
        parent,
        machine,
        decoder,
        bank,
        key_bank,
        transform_seeds=TRANSFORM_SEEDS,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 140_000,
    )
    new_bridge, new_progress = _train_cell(
        parent,
        machine,
        decoder,
        transform_seed=ONLINE_TRANSFORM_SEED,
        updates=args.bridge_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 340_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        shuffle_outcomes=False,
    )
    shuffled_new_bridge, shuffled_new_progress = _train_cell(
        parent,
        machine,
        decoder,
        transform_seed=ONLINE_TRANSFORM_SEED,
        updates=args.bridge_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 360_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        shuffle_outcomes=True,
    )
    new_target_after = _score(
        parent,
        machine,
        decoder,
        new_bridge,
        transform_seed=ONLINE_TRANSFORM_SEED,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 350_000,
    )
    shuffled_new_after = _score(
        parent,
        machine,
        decoder,
        shuffled_new_bridge,
        transform_seed=ONLINE_TRANSFORM_SEED,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 370_000,
    )
    new_stable_bits = _stable_bits(
        new_progress,
        threshold=MASTERY_THRESHOLD,
        bits_per_update=args.batch_size * args.span,
    )
    bank.add("cell_3", new_bridge)
    bank.freeze("cell_3")
    all_transform_seeds = (*TRANSFORM_SEEDS, ONLINE_TRANSFORM_SEED)
    key_bank.add(
        "cell_3",
        _estimate_key(
            parent,
            ONLINE_TRANSFORM_SEED,
            seed=args.seed + 440_000,
        ),
    )
    expanded_key_routing = _evaluate_key_router(
        parent,
        machine,
        decoder,
        bank,
        key_bank,
        transform_seeds=all_transform_seeds,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 450_000,
    )
    corrupted_key_bank = copy.deepcopy(key_bank)
    with torch.no_grad():
        corrupted_key_bank.keys[[1, 3]] = corrupted_key_bank.keys[[3, 1]]
    corrupted_key_routing = _evaluate_key_router(
        parent,
        machine,
        decoder,
        bank,
        corrupted_key_bank,
        transform_seeds=all_transform_seeds,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 450_000,
    )
    expanded_router = _expand_router(initial_router, old_count=3)
    expanded_router_progress = _train_new_router_head(
        parent,
        machine,
        decoder,
        bank,
        expanded_router,
        transform_seed=ONLINE_TRANSFORM_SEED,
        new_index=3,
        updates=args.growth_router_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 410_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        shuffle_outcomes=False,
    )
    shuffled_expanded_router = _expand_router(initial_router, old_count=3)
    shuffled_expanded_progress = _train_new_router_head(
        parent,
        machine,
        decoder,
        bank,
        shuffled_expanded_router,
        transform_seed=ONLINE_TRANSFORM_SEED,
        new_index=3,
        updates=args.growth_router_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 430_000,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        shuffle_outcomes=True,
    )
    expanded_routing = _evaluate_router(
        parent,
        machine,
        decoder,
        bank,
        expanded_router,
        transform_seeds=all_transform_seeds,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 420_000,
    )
    shuffled_expanded_routing = _evaluate_router(
        parent,
        machine,
        decoder,
        bank,
        shuffled_expanded_router,
        transform_seeds=all_transform_seeds,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 440_000,
    )
    returns = _return_scores(
        bank,
        all_transform_seeds,
        parent,
        machine,
        decoder,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 80_000,
    )
    source_after = _score(
        parent,
        machine,
        decoder,
        bank.cell("cell_0"),
        transform_seed=TRANSFORM_SEEDS[0],
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 100_000,
    )
    parent_digest_after = _module_digest(parent.controller)
    machine_digest_after = _module_digest(machine)
    decoder_digest_after = _module_digest(decoder)
    old_rows = expanded_routing["rows"][:3]
    shared_head_old_routes_retained = all(
        bool(row["routing_correct"])
        and float(row["action_accuracy"]) >= MASTERY_THRESHOLD
        for row in old_rows
    )
    gates = {
        "source_mastered": source_accuracy >= MASTERY_THRESHOLD,
        "initial_router_mastered": bool(initial_routing["action_mastery"])
        and float(initial_routing["routing_accuracy"]) == 1.0,
        "initial_key_addressing_mastered": bool(
            initial_key_routing["action_mastery"]
        )
        and float(initial_key_routing["routing_accuracy"]) == 1.0,
        "initial_cells_stable": all(value is not None for value in initial_stable_bits),
        "new_unregistered_cell_mastered": new_target_after >= MASTERY_THRESHOLD,
        "new_cell_stable": new_stable_bits is not None,
        "new_shuffled_rejected": shuffled_new_after < MASTERY_THRESHOLD
        and _stable_bits(
            shuffled_new_progress,
            threshold=MASTERY_THRESHOLD,
            bits_per_update=args.batch_size * args.span,
        )
        is None,
        "new_route_mastered": bool(expanded_routing["rows"][3]["routing_correct"])
        and float(expanded_routing["rows"][3]["action_accuracy"])
        >= MASTERY_THRESHOLD,
        "protected_key_addressing_after_growth": bool(
            expanded_key_routing["action_mastery"]
        )
        and float(expanded_key_routing["routing_accuracy"]) == 1.0,
        "key_corruption_is_detected": (
            float(corrupted_key_routing["routing_accuracy"])
            < float(expanded_key_routing["routing_accuracy"])
        ),
        "expanded_router_shuffled_rejected": not bool(
            shuffled_expanded_routing["action_mastery"]
        ),
        "source_retained": source_after >= source_accuracy - 0.02,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "frozen_source_machine": machine_digest_before == machine_digest_after,
        "frozen_source_decoder": decoder_digest_before == decoder_digest_after,
    }
    report = {
        "schema": "neural-computer.outcome-only-online-alignment-growth-report.v1",
        "claim_boundary": (
            "An external alignment bank can admit one unregistered transform "
            "with a protected immutable key and no replay of old streams; a "
            "shared-head expansion is retained as a negative control. This is "
            "not unrestricted or general continual learning."
        ),
        "seed": args.seed,
        "configuration": {
            "initial_transform_seeds": list(TRANSFORM_SEEDS),
            "online_transform_seed": ONLINE_TRANSFORM_SEED,
            "key_bank": key_bank.configuration(),
            "bridge_updates": args.bridge_updates,
            "router_updates": args.router_updates,
            "growth_router_updates": args.growth_router_updates,
            "batch_size": args.batch_size,
            "span": args.span,
            "audit_count": args.audit_count,
        },
        "results": {
            "source_accuracy": source_accuracy,
            "source_attempts": source_attempts,
            "source_after": source_after,
            "initial_cells": initial_reports,
            "initial_routing": initial_routing,
            "initial_key_routing": initial_key_routing,
            "new_cell": {
                "target_after": new_target_after,
                "shuffled_after": shuffled_new_after,
                "stable_bits_to_threshold": new_stable_bits,
                "progress": new_progress,
                "shuffled_progress": shuffled_new_progress,
            },
            "expanded_returns": returns,
            "expanded_routing": expanded_routing,
            "shared_head_diagnostic": {
                "old_routes_retained": shared_head_old_routes_retained,
                "expanded_router": expanded_routing,
            },
            "expanded_key_routing": expanded_key_routing,
            "corrupted_key_routing": corrupted_key_routing,
            "shuffled_expanded_routing": shuffled_expanded_routing,
            "initial_router_progress": initial_router_progress,
            "expanded_router_progress": expanded_router_progress,
            "shuffled_expanded_progress": shuffled_expanded_progress,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.source_updates * args.batch_size * args.span * 2
                + len(TRANSFORM_SEEDS) * args.bridge_updates * args.batch_size * args.span * 2
                + args.router_updates * args.batch_size * args.span * 2
                + args.bridge_updates * args.batch_size * args.span * 2
                + args.growth_router_updates * args.batch_size * args.span * 2
            ),
            "unique_logical_lifetimes": (
                args.source_updates * args.batch_size
                + len(TRANSFORM_SEEDS) * args.bridge_updates * args.batch_size * 2
                + args.router_updates * args.batch_size * 2
                + args.bridge_updates * args.batch_size * 2
                + args.growth_router_updates * args.batch_size * 2
            ),
            "optimizer_updates": (
                args.source_updates
                + len(TRANSFORM_SEEDS) * args.bridge_updates * 2
                + args.router_updates * 2
                + args.bridge_updates * 2
                + args.growth_router_updates * 2
            ),
            "replayed_examples": 0,
        },
        "digests": {
            "parent_before": parent_digest_before,
            "parent_after": parent_digest_after,
            "source_machine_before": machine_digest_before,
            "source_machine_after": machine_digest_after,
            "source_decoder_before": decoder_digest_before,
            "source_decoder_after": decoder_digest_after,
            "initial_router": _module_digest(initial_router),
            "expanded_router": _module_digest(expanded_router),
            "key_bank": _module_digest(key_bank),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "elapsed_seconds": perf_counter() - started,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates", type=int, default=192)
    parser.add_argument("--source-restarts", type=int, default=2)
    parser.add_argument("--bridge-updates", type=int, default=256)
    parser.add_argument("--router-updates", type=int, default=256)
    parser.add_argument("--growth-router-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
