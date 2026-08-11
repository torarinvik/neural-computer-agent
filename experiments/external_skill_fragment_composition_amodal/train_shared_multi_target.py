"""Pressure-test one shared external learner across many programs.

The four opaque fragments are acquired sequentially exactly once.  After the
bank is frozen, one external trace combiner and one opaque output decoder learn
several target programs together.  Additional program orders remain held out
from that learner.  Target order, operation names, and correct actions stay in
the trainer-owned verifier; the combiner sees only post-instruction traces and
padding masks.

This experiment is deliberately stricter than allocating one combiner per
target: it tests whether the external learner itself is reusable rather than
whether the bank can feed an ever-growing collection of target adapters.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from experiments.external_skill_fragment_composition_amodal.train import (
    ACTION_WIDTH,
    REGISTER_WIDTH,
    _batch,
    _digest,
    _fragment_bank,
    _machine,
    _rollout,
    _stable_bits,
    _train_stage,
)
from experiments.external_skill_fragment_composition_amodal.train_multi import (
    PRIMITIVES,
    _append_fragment,
    _bank_with_fragments,
    _corruption_audit,
    _eval_points,
    _retention,
    _set_fragment_stage,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _runtime,
)
from neural_computer import (
    ExternalSkillFragmentBank,
    ExternalSkillFragmentOperatorCombiner,
    ExternalSkillFragmentSegmentCombiner,
    ExternalSkillFragmentSerialCombiner,
    OpaqueProtocolDecoder,
)

TRAIN_ORDERS = (
    (3, 2, 0, 1),
    (1, 0, 3, 2),
    (2, 3, 1, 0),
)
HELDOUT_ORDERS = ((0, 1, 3, 2), (0, 3, 2, 1), (1, 2, 3, 0))
CURRICULUM_TRAIN_ORDERS = (
    ((0,), (1,), (2,), (3,))
    + (
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (0, 3),
        (3, 0),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
    )
    + (
        (3, 2, 0),
        (1, 0, 3),
        (2, 3, 1),
        (0, 1, 2),
        (0, 2, 3),
        (1, 3, 0),
    )
    + TRAIN_ORDERS
)
CURRICULUM_HELDOUT_ORDERS = (
    (0, 1, 3),
    (0, 3, 1),
    (1, 2, 0),
    *HELDOUT_ORDERS,
)


def _order_sets(
    curriculum: bool,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    if curriculum:
        return CURRICULUM_TRAIN_ORDERS, CURRICULUM_HELDOUT_ORDERS
    return TRAIN_ORDERS, HELDOUT_ORDERS


def _program(order: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(PRIMITIVES[index] for index in order)


def _rotate_orders(
    orders: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the same routes with their first operator moved to the end."""

    return tuple(order if len(order) < 2 else order[1:] + order[:1] for order in orders)


def _group_composition_ids(target_count: int, examples_per_target: int) -> torch.Tensor:
    """Assign contiguous rows to each target for reliable per-target audits."""

    if target_count < 1 or examples_per_target < 1:
        raise ValueError("composition target and example counts must be positive")
    return torch.arange(target_count, dtype=torch.long).repeat_interleave(
        examples_per_target
    )


def _subset_batch(batch, rows: torch.Tensor):
    """Keep a balanced subset without changing the rendered transport ABI."""

    rows = rows.to(device=batch.input_frames.device, dtype=torch.long)
    return replace(
        batch,
        input_frames=batch.input_frames.index_select(0, rows),
        distractor_frames=batch.distractor_frames.index_select(0, rows),
        query_frames=batch.query_frames.index_select(0, rows),
        correct_actions=batch.correct_actions.index_select(0, rows),
        sequence=batch.sequence.index_select(0, rows),
        operation_bits=batch.operation_bits.index_select(0, rows),
        seeds=batch.seeds.index_select(0, rows),
    )


def _select_causal_rows(
    batch,
    causal_signal: torch.Tensor,
    composition_ids: torch.Tensor,
    *,
    examples_per_target: int,
    mode: str,
    seed: int,
) -> tuple[object, dict[str, object]]:
    """Select answer-changing examples, with a matched passive control.

    ``causal_signal`` is produced by common-render, common-policy
    leave-one-transition-out verifier outcomes.  The selection is trainer-only
    data curation: neither the signal nor the verifier answers cross the
    deployed combiner/decoder boundary.  Passive selection still pays for the
    same candidate probe and chooses an equal-budget random subset.
    """

    if mode not in ("active", "passive"):
        raise ValueError("causal row selection mode must be active or passive")
    if examples_per_target < 1:
        raise ValueError("causal selection examples per target must be positive")
    if causal_signal.ndim != 3 or causal_signal.shape[0] != batch.batch_size:
        raise ValueError("causal signal must have shape [batch, queries, segments]")
    if composition_ids.shape != (batch.batch_size,):
        raise ValueError("composition IDs must align with candidate rows")
    scores = causal_signal.float().mean(dim=(1, 2))
    target_count = int(composition_ids.max().item()) + 1
    generator = torch.Generator(device=composition_ids.device)
    generator.manual_seed(seed)
    selected: list[torch.Tensor] = []
    for target in range(target_count):
        candidates = torch.nonzero(composition_ids == target, as_tuple=False).flatten()
        if candidates.numel() < examples_per_target:
            raise ValueError("causal candidate pool is smaller than the requested subset")
        if mode == "active":
            chosen = candidates[torch.topk(
                scores[candidates], examples_per_target, largest=True, sorted=True
            ).indices]
        else:
            permutation = torch.randperm(candidates.numel(), generator=generator)
            chosen = candidates[permutation[:examples_per_target]]
        selected.append(chosen)
    rows = torch.cat(selected)
    selected_scores = scores[rows]
    stats = {
        "mode": mode,
        "candidate_rows": batch.batch_size,
        "selected_rows": int(rows.numel()),
        "candidate_mean_causal_signal": float(scores.mean()),
        "selected_mean_causal_signal": float(selected_scores.mean()),
        "selected_max_causal_signal": float(selected_scores.max()),
    }
    return _subset_batch(batch, rows), stats


def _causal_selection_bits(
    specs: tuple[tuple[tuple[int, ...], tuple[str, ...]], ...],
    *,
    updates: int,
    batch_size: int,
    span: int,
    candidate_multiplier: int,
    probe_samples: int,
) -> int:
    """Count verifier bits spent probing candidates plus interventions."""

    target_count = len(specs)
    segment_count = sum(len(program) for _, program in specs)
    candidate_rows = batch_size * candidate_multiplier * target_count
    return updates * span * probe_samples * (
        candidate_rows + batch_size * candidate_multiplier * segment_count
    )


def _specs(
    orders: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], tuple[str, ...]], ...]:
    return tuple((order, _program(order)) for order in orders)


def _spec_groups_by_length(
    specs: tuple[tuple[tuple[int, ...], tuple[str, ...]], ...],
    *,
    max_group_size: int = 2,
) -> tuple[tuple[tuple[tuple[int, ...], tuple[str, ...]], ...], ...]:
    """Batch only programs with equal executable lengths.

    The rendered composition transport can carry one or two opaque programs
    per batch, but the register machine requires their executable traces to
    have equal lengths.  A curriculum intentionally mixes depths, so grouping
    by length is part of the transport ABI rather than an experiment-specific
    assumption.
    """

    if max_group_size < 1:
        raise ValueError("composition spec group size must be positive")
    groups_by_length: dict[int, list[tuple[tuple[int, ...], tuple[str, ...]]]] = {}
    length_order: list[int] = []
    for spec in specs:
        length = len(spec[0])
        if length < 1:
            raise ValueError("composition programs cannot be empty")
        if length not in groups_by_length:
            groups_by_length[length] = []
            length_order.append(length)
        groups_by_length[length].append(spec)
    groups: list[tuple[tuple[tuple[int, ...], tuple[str, ...]], ...]] = []
    for length in length_order:
        entries = groups_by_length[length]
        groups.extend(
            tuple(entries[start : start + max_group_size])
            for start in range(0, len(entries), max_group_size)
        )
    return tuple(groups)


def _causal_prefix_targets(
    batch,
    programs: tuple[tuple[str, ...], ...],
    composition_ids: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Render verifier outcomes for every causal prefix of each program.

    The prefix targets remain trainer-owned deterministic verifier answers.
    The learner receives only the scalar action utilities produced from these
    fresh prefix lifetimes; no operation name or target tensor crosses the
    combiner boundary.
    """

    if not programs or len({len(program) for program in programs}) != 1:
        raise ValueError("prefix targets require nonempty equal-depth programs")
    prefix_batches = []
    for depth in range(1, len(programs[0]) + 1):
        prefixes = tuple(program[:depth] for program in programs)
        prefix_batch = _batch(
            operation="generated_composition",
            count=batch.batch_size,
            span=batch.span,
            seed=seed + depth * 7_919,
            generated_compositions=prefixes,
            generated_composition_ids_override=composition_ids,
            operation_bits_override=batch.operation_bits,
            sequence_override=batch.sequence,
        )
        prefix_batches.append(prefix_batch.correct_actions)
    return torch.stack(prefix_batches, dim=1)


def _make_combiner(mode: str) -> nn.Module | None:
    if mode == "trace":
        # The register interpreter already emits a learned terminal state.
        # This diagnostic keeps that state intact and tests whether an extra
        # learned composition codec is destroying ordered information.
        return None
    if mode == "segment":
        return ExternalSkillFragmentSegmentCombiner(
            REGISTER_WIDTH, 16, REGISTER_WIDTH, hidden=64
        )
    if mode == "serial":
        combiner = ExternalSkillFragmentSerialCombiner(
            REGISTER_WIDTH, 16, REGISTER_WIDTH, hidden=64
        )
        for _ in PRIMITIVES:
            combiner.append_step_slot()
        return combiner
    if mode == "serial_shared":
        combiner = ExternalSkillFragmentSerialCombiner(
            REGISTER_WIDTH,
            16,
            REGISTER_WIDTH,
            hidden=64,
            step_sharing="shared",
        )
        combiner.append_step_slot()
        return combiner
    if mode == "operator":
        return ExternalSkillFragmentOperatorCombiner(
            REGISTER_WIDTH, 16, REGISTER_WIDTH, hidden=64, operator_rank=8
        )
    raise ValueError(f"unsupported composition combiner mode: {mode}")


def _make_prefix_credit_head() -> nn.Module:
    """Create an external transition-use policy with near-identity startup."""

    head = nn.Sequential(
        nn.Linear(REGISTER_WIDTH, 64),
        nn.GELU(),
        nn.LayerNorm(64),
        nn.Linear(64, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, 2.0)
    return head


def _parameters(module: nn.Module | None) -> tuple[nn.Parameter, ...]:
    """Return trainable parameters for a codec, including the trace baseline."""

    return () if module is None else tuple(module.parameters())


def _shared_train_stage(
    parent,
    machine,
    bank: ExternalSkillFragmentBank,
    combiner: nn.Module | None,
    decoder: OpaqueProtocolDecoder,
    *,
    specs: tuple[tuple[tuple[int, ...], tuple[str, ...]], ...],
    updates: int,
    batch_size: int,
    span: int,
    seed: int,
    trainable: list[nn.Parameter],
    eval_every: int,
    audit_count: int,
    audit_seed: int,
    shuffle_outcomes: bool = False,
    order_contrast_weight: float = 0.0,
    prefix_credit_weight: float = 0.0,
    leave_one_out_credit_weight: float = 0.0,
    credit_head: nn.Module | None = None,
    causal_selection: str = "none",
    causal_candidate_multiplier: int = 2,
    causal_probe_temperature: float = 0.5,
    causal_probe_samples: int = 4,
) -> list[dict[str, object]]:
    """Train one combiner/decoder on fresh samples from every target order."""

    parameters = [parameter for parameter in trainable if parameter.requires_grad]
    if not parameters:
        raise ValueError("shared composition stage has no trainable parameters")
    if not math.isfinite(order_contrast_weight) or order_contrast_weight < 0.0:
        raise ValueError("order contrast weight must be finite and non-negative")
    if not math.isfinite(prefix_credit_weight) or prefix_credit_weight < 0.0:
        raise ValueError("prefix credit weight must be finite and non-negative")
    if (
        not math.isfinite(leave_one_out_credit_weight)
        or leave_one_out_credit_weight < 0.0
    ):
        raise ValueError(
            "leave-one-out credit weight must be finite and non-negative"
        )
    if leave_one_out_credit_weight and credit_head is None:
        raise ValueError("leave-one-out credit requires an external credit head")
    if credit_head is not None and not leave_one_out_credit_weight:
        raise ValueError("an external credit head requires a positive weight")
    if causal_selection not in ("none", "active", "passive"):
        raise ValueError("causal selection must be none, active, or passive")
    if causal_selection != "none" and causal_candidate_multiplier < 1:
        raise ValueError("causal candidate multiplier must be positive")
    if not math.isfinite(causal_probe_temperature) or causal_probe_temperature <= 0.0:
        raise ValueError("causal probe temperature must be finite and positive")
    if (
        not isinstance(causal_probe_samples, int)
        or isinstance(causal_probe_samples, bool)
        or causal_probe_samples < 1
    ):
        raise ValueError("causal probe samples must be a positive integer")
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=1e-5)
    history: list[dict[str, object]] = []
    for update in range(1, updates + 1):
        losses: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        selection_observations: list[dict[str, object]] = []
        for group_index, group in enumerate(_spec_groups_by_length(specs)):
            orders = tuple(order for order, _ in group)
            programs = tuple(program for _, program in group)
            count = batch_size * len(group)
            composition_ids = _group_composition_ids(len(group), batch_size)
            selection_stats = None
            if causal_selection == "none":
                batch = _batch(
                    operation="generated_composition",
                    count=count,
                    span=span,
                    seed=seed + update * 10_007 + group_index * 1_000_003,
                    generated_compositions=programs,
                    generated_composition_ids_override=composition_ids,
                )
            else:
                candidate_per_target = batch_size * causal_candidate_multiplier
                candidate_ids = _group_composition_ids(
                    len(group), candidate_per_target
                )
                candidate_batch = _batch(
                    operation="generated_composition",
                    count=candidate_per_target * len(group),
                    span=span,
                    seed=seed + update * 10_007 + group_index * 1_000_003,
                    generated_compositions=programs,
                    generated_composition_ids_override=candidate_ids,
                )
                _, _, causal_signal = _rollout(
                    parent,
                    machine,
                    bank,
                    decoder,
                    candidate_batch,
                    selected=None,
                    train_decoder=True,
                    combiner=combiner,
                    route_programs=orders,
                    include_instruction_codes=True,
                    credit_head=credit_head,
                    leave_one_out_credit_weight=leave_one_out_credit_weight,
                    return_causal_signal=True,
                    counterfactual_temperature=causal_probe_temperature,
                    counterfactual_samples=causal_probe_samples,
                )
                batch, selection_stats = _select_causal_rows(
                    candidate_batch,
                    causal_signal,
                    candidate_ids,
                    examples_per_target=batch_size,
                    mode=causal_selection,
                    seed=seed + update * 97_003 + group_index * 1_000_033,
                )
                composition_ids = _group_composition_ids(len(group), batch_size)
            prefix_targets = (
                _causal_prefix_targets(
                    batch,
                    programs,
                    composition_ids,
                    seed=seed + update * 30_011 + group_index * 2_000_003,
                )
                if prefix_credit_weight
                else None
            )
            loss, target_rewards = _rollout(
                parent,
                machine,
                bank,
                decoder,
                batch,
                selected=None,
                train_decoder=True,
                shuffle_outcomes=shuffle_outcomes,
                combiner=combiner,
                route_programs=orders,
                include_instruction_codes=True,
                prefix_targets=prefix_targets,
                prefix_loss_weight=prefix_credit_weight,
                credit_head=credit_head,
                leave_one_out_credit_weight=leave_one_out_credit_weight,
            )
            losses.append(loss)
            rewards.append(target_rewards)
            if selection_stats is not None:
                selection_observations.append(selection_stats)
            if order_contrast_weight and any(len(order) > 1 for order in orders):
                contrast_loss, _ = _rollout(
                    parent,
                    machine,
                    bank,
                    decoder,
                    batch,
                    selected=None,
                    train_decoder=True,
                    shuffle_outcomes=shuffle_outcomes,
                    combiner=combiner,
                    route_programs=_rotate_orders(orders),
                    include_instruction_codes=True,
                    invert_targets=True,
                    prefix_targets=prefix_targets,
                    prefix_loss_weight=prefix_credit_weight,
                    credit_head=credit_head,
                    leave_one_out_credit_weight=leave_one_out_credit_weight,
                )
                losses.append(order_contrast_weight * contrast_loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        row: dict[str, object] = {
            "update": update,
            "training_accuracy": float(torch.cat(rewards).mean()),
            "loss": float(loss.detach()),
        }
        row["unique_verifier_bits"] = update * batch_size * span * 2 * len(specs)
        row["counterfactual_rollouts"] = (
            update
            * batch_size
            * span
            * 2
            * sum(1 for order, _ in specs if len(order) > 1)
            if order_contrast_weight
            else 0
        )
        row["prefix_credit_rollouts"] = (
            update
            * batch_size
            * span
            * sum(len(program) for _, program in specs)
            if prefix_credit_weight
            else 0
        )
        row["leave_one_out_credit_rollouts"] = (
            update
            * batch_size
            * span
            * sum(len(program) for _, program in specs)
            if leave_one_out_credit_weight
            else 0
        )
        row["causal_selection"] = causal_selection
        row["causal_candidate_multiplier"] = causal_candidate_multiplier
        row["causal_probe_temperature"] = causal_probe_temperature
        row["causal_probe_samples"] = causal_probe_samples
        if selection_observations:
            row["causal_selection_candidate_rows"] = sum(
                int(observation["candidate_rows"])
                for observation in selection_observations
            )
            row["causal_selection_selected_rows"] = sum(
                int(observation["selected_rows"])
                for observation in selection_observations
            )
            row["causal_selection_candidate_mean"] = float(
                sum(
                    float(observation["candidate_mean_causal_signal"])
                    for observation in selection_observations
                )
                / len(selection_observations)
            )
            row["causal_selection_selected_mean"] = float(
                sum(
                    float(observation["selected_mean_causal_signal"])
                    for observation in selection_observations
                )
                / len(selection_observations)
            )
        if eval_every and (update % eval_every == 0 or update == updates):
            heldout = _evaluate_specs(
                parent,
                machine,
                bank,
                combiner,
                decoder,
                specs=specs,
                count=audit_count,
                span=span,
                seed=audit_seed + update * 1_009,
                shuffle_outcomes=shuffle_outcomes,
            )
            row["heldout_by_target"] = heldout
            row["heldout_accuracy"] = min(heldout)
        history.append(row)
    return history


def _evaluate_specs(
    parent,
    machine,
    bank: ExternalSkillFragmentBank,
    combiner: nn.Module | None,
    decoder: OpaqueProtocolDecoder,
    *,
    specs: tuple[tuple[tuple[int, ...], tuple[str, ...]], ...],
    count: int,
    span: int,
    seed: int,
    shuffle_outcomes: bool = False,
    zero_codes: bool = False,
    blank_sequence: bool = False,
) -> list[float]:
    """Evaluate target rows in opaque pairs to reuse frozen traversal work."""

    values: list[float] = []
    with torch.no_grad():
        for group_index, group in enumerate(_spec_groups_by_length(specs)):
            orders = tuple(order for order, _ in group)
            programs = tuple(program for _, program in group)
            group_count = count * len(group)
            batch = _batch(
                operation="generated_composition",
                count=group_count,
                span=span,
                seed=seed + group_index * 10_007,
                generated_compositions=programs,
                blank_sequence=blank_sequence,
                generated_composition_ids_override=_group_composition_ids(
                    len(group), count
                ),
            )
            rewards = _rollout(
                parent,
                machine,
                bank,
                decoder,
                batch,
                selected=None,
                train_decoder=False,
                shuffle_outcomes=shuffle_outcomes,
                zero_codes=zero_codes,
                combiner=combiner,
                route_programs=orders,
                include_instruction_codes=True,
            )[1]
            for target_index in range(len(group)):
                start = target_index * count
                values.append(float(rewards[start : start + count].mean()))
    return values


def _train_four_fragments(parent, args: argparse.Namespace):
    machine = _machine()
    bank = _fragment_bank(args.seed + 1)
    decoders: list[OpaqueProtocolDecoder] = []
    histories: list[list[dict[str, object]]] = []
    retention_by_stage: list[dict[str, float]] = []
    for index, operation in enumerate(PRIMITIVES):
        if index > 0:
            _append_fragment(bank, args.seed + 2_000 + index)
        decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
        decoders.append(decoder)
        trainable = _set_fragment_stage(machine, bank, index)
        histories.append(
            _train_stage(
                parent,
                machine,
                bank,
                decoder,
                operation=operation,
                selected=(index,),
                updates=args.primitive_updates,
                batch_size=args.batch_size,
                span=args.span,
                seed=args.seed + 10_000 + index * 20_003,
                trainable=[*trainable, *decoder.parameters()],
                eval_every=args.eval_every,
                audit_count=args.audit_count,
                audit_seed=args.seed + 30_000 + index * 20_003,
            )
        )
        bank.protect(index)
        retention_by_stage.append(
            _retention(
                parent,
                machine,
                bank,
                decoders,
                count=args.audit_count,
                span=args.span,
                seed=args.seed + 50_000 + index * 20_003,
            )
        )
    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    for parameter in bank.parameters():
        parameter.requires_grad_(False)
    return machine, bank, decoders, histories, retention_by_stage


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(1)
    if (
        min(
            args.parent_updates,
            args.primitive_updates,
            args.composition_updates,
            args.batch_size,
            args.audit_count,
            args.eval_every,
        )
        < 1
    ):
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    if not math.isfinite(args.order_contrast_weight) or args.order_contrast_weight < 0.0:
        raise ValueError("order contrast weight must be finite and non-negative")
    if not math.isfinite(args.prefix_credit_weight) or args.prefix_credit_weight < 0.0:
        raise ValueError("prefix credit weight must be finite and non-negative")
    if (
        args.prefix_credit_weight or args.leave_one_out_credit_weight
    ) and args.combiner_mode not in ("serial", "serial_shared"):
        raise ValueError("causal serial credit requires a serial combiner mode")
    if args.causal_selection != "none" and args.combiner_mode not in (
        "serial",
        "serial_shared",
    ):
        raise ValueError("causal selection requires a serial combiner mode")
    if args.prefix_credit_weight and args.leave_one_out_credit_weight:
        raise ValueError("choose direct prefix credit or leave-one-out credit")
    if (
        not math.isfinite(args.leave_one_out_credit_weight)
        or args.leave_one_out_credit_weight < 0.0
    ):
        raise ValueError(
            "leave-one-out credit weight must be finite and non-negative"
        )
    if args.causal_selection not in ("none", "active", "passive"):
        raise ValueError("causal selection must be none, active, or passive")
    if args.causal_selection != "none" and args.causal_candidate_multiplier < 1:
        raise ValueError("causal candidate multiplier must be positive")
    if not math.isfinite(args.causal_probe_temperature) or args.causal_probe_temperature <= 0.0:
        raise ValueError("causal probe temperature must be finite and positive")
    if args.causal_probe_samples < 1:
        raise ValueError("causal probe samples must be positive")

    parent = _runtime(seed=args.seed, growth=False)
    _, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 100,
        learning_rate=3e-3,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
    )
    parent.eval()
    parent_digest_before = _digest(parent.controller)
    machine, bank, source_decoders, primitive_histories, retention_by_stage = (
        _train_four_fragments(parent, args)
    )
    source_before = retention_by_stage[-1]
    bank_digest_before = bank.payload()["sha256"]
    train_orders, heldout_orders = _order_sets(args.curriculum)
    train_specs = _specs(train_orders)
    heldout_specs = _specs(heldout_orders)
    target_count = len(train_specs)
    contrast_spec_count = sum(1 for order, _ in train_specs if len(order) > 1)
    bits_per_update = args.batch_size * args.span * 2

    torch.manual_seed(args.seed + 100_000)
    shared_combiner = _make_combiner(args.combiner_mode)
    shared_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    shared_credit_head = (
        _make_prefix_credit_head()
        if args.leave_one_out_credit_weight
        else None
    )
    shared_history = _shared_train_stage(
        parent,
        machine,
        bank,
        shared_combiner,
        shared_decoder,
        specs=train_specs,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 100_000,
        trainable=[
            *_parameters(shared_combiner),
            *shared_decoder.parameters(),
            *(
                shared_credit_head.parameters()
                if shared_credit_head is not None
                else ()
            ),
        ],
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 101_000,
        order_contrast_weight=args.order_contrast_weight,
        prefix_credit_weight=args.prefix_credit_weight,
        leave_one_out_credit_weight=args.leave_one_out_credit_weight,
        credit_head=shared_credit_head,
        causal_selection=args.causal_selection,
        causal_candidate_multiplier=args.causal_candidate_multiplier,
        causal_probe_temperature=args.causal_probe_temperature,
        causal_probe_samples=args.causal_probe_samples,
    )
    shared_train_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        shared_combiner,
        shared_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 102_000,
    )
    shared_heldout_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        shared_combiner,
        shared_decoder,
        specs=heldout_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 103_000,
    )
    wrong_specs = tuple(
        (tuple(order[1:]) + (order[0],), program)
        for order, program in train_specs
    )
    shared_wrong_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        shared_combiner,
        shared_decoder,
        specs=wrong_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 104_000,
    )
    shared_zero_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        shared_combiner,
        shared_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 105_000,
        zero_codes=True,
    )
    shared_missing_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        shared_combiner,
        shared_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 106_000,
        blank_sequence=True,
    )

    torch.manual_seed(args.seed + 110_000)
    shuffled_combiner = _make_combiner(args.combiner_mode)
    shuffled_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    shuffled_credit_head = (
        _make_prefix_credit_head()
        if args.leave_one_out_credit_weight
        else None
    )
    _shared_train_stage(
        parent,
        machine,
        bank,
        shuffled_combiner,
        shuffled_decoder,
        specs=train_specs,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 110_000,
        trainable=[
            *_parameters(shuffled_combiner),
            *shuffled_decoder.parameters(),
            *(
                shuffled_credit_head.parameters()
                if shuffled_credit_head is not None
                else ()
            ),
        ],
        eval_every=0,
        audit_count=args.audit_count,
        audit_seed=args.seed + 111_000,
        shuffle_outcomes=True,
        order_contrast_weight=args.order_contrast_weight,
        prefix_credit_weight=args.prefix_credit_weight,
        leave_one_out_credit_weight=args.leave_one_out_credit_weight,
        credit_head=shuffled_credit_head,
        causal_selection=args.causal_selection,
        causal_candidate_multiplier=args.causal_candidate_multiplier,
        causal_probe_temperature=args.causal_probe_temperature,
        causal_probe_samples=args.causal_probe_samples,
    )
    shuffled_accuracy = _evaluate_specs(
        parent,
        machine,
        bank,
        shuffled_combiner,
        shuffled_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 111_000,
        shuffle_outcomes=True,
    )

    torch.manual_seed(args.seed + 120_000)
    fresh_machine = _machine()
    fresh_bank = _bank_with_fragments(args.seed + 2, len(PRIMITIVES))
    fresh_combiner = _make_combiner(args.combiner_mode)
    fresh_decoder = OpaqueProtocolDecoder(REGISTER_WIDTH, ACTION_WIDTH, hidden=16)
    fresh_credit_head = (
        _make_prefix_credit_head() if args.leave_one_out_credit_weight else None
    )
    fresh_history = _shared_train_stage(
        parent,
        fresh_machine,
        fresh_bank,
        fresh_combiner,
        fresh_decoder,
        specs=train_specs,
        updates=args.composition_updates,
        batch_size=args.batch_size,
        span=args.span,
        seed=args.seed + 120_000,
        trainable=[
            *fresh_machine.parameters(),
            fresh_bank.shared_basis,
            *fresh_bank.coefficients,
            *_parameters(fresh_combiner),
            *fresh_decoder.parameters(),
            *(
                fresh_credit_head.parameters()
                if fresh_credit_head is not None
                else ()
            ),
        ],
        eval_every=args.eval_every,
        audit_count=args.audit_count,
        audit_seed=args.seed + 121_000,
        order_contrast_weight=args.order_contrast_weight,
        prefix_credit_weight=args.prefix_credit_weight,
        leave_one_out_credit_weight=args.leave_one_out_credit_weight,
        credit_head=fresh_credit_head,
        causal_selection=args.causal_selection,
        causal_candidate_multiplier=args.causal_candidate_multiplier,
        causal_probe_temperature=args.causal_probe_temperature,
        causal_probe_samples=args.causal_probe_samples,
    )
    fresh_train_accuracy = _evaluate_specs(
        parent,
        fresh_machine,
        fresh_bank,
        fresh_combiner,
        fresh_decoder,
        specs=train_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 122_000,
    )
    fresh_heldout_accuracy = _evaluate_specs(
        parent,
        fresh_machine,
        fresh_bank,
        fresh_combiner,
        fresh_decoder,
        specs=heldout_specs,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 123_000,
    )
    source_after = _retention(
        parent,
        machine,
        bank,
        source_decoders,
        count=args.audit_count,
        span=args.span,
        seed=args.seed + 130_000,
    )
    bank_digest_after = bank.payload()["sha256"]

    # Keep concurrent seed runs isolated.  A shared sibling directory makes
    # corruption audits race: one process can overwrite the checkpoint while
    # another is loading it, producing a false persistence failure.
    persistence_dir = args.report_out.with_name(
        f"{args.report_out.stem}.persistence"
    )
    if persistence_dir.exists():
        shutil.rmtree(persistence_dir)
    persistence_dir.mkdir(parents=True, exist_ok=True)
    persistence_exact = _corruption_audit(bank, persistence_dir / "fragment-bank.pt")
    parent_digest_after = _digest(parent.controller)
    primitive_stable_bits = {
        PRIMITIVES[index]: _stable_bits(
            history,
            threshold=args.mastery_threshold,
            bits_per_update=bits_per_update,
        )
        for index, history in enumerate(primitive_histories)
    }
    shared_stable_bits = _stable_bits(
        shared_history,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update * len(train_specs),
    )
    fresh_stable_bits = _stable_bits(
        fresh_history,
        threshold=args.mastery_threshold,
        bits_per_update=bits_per_update * len(train_specs),
    )
    composition_eval_points = _eval_points(args.composition_updates, args.eval_every)
    parent_eval_points = _eval_points(args.parent_updates, args.eval_every)
    training_batches = (
        args.parent_updates
        + len(PRIMITIVES) * args.primitive_updates
        + 3 * target_count * args.composition_updates
    )
    prefix_credit_training_bits = (
        3
        * args.composition_updates
        * args.batch_size
        * args.span
        * 2
        * sum(len(program) for _, program in train_specs)
        if args.prefix_credit_weight
        else 0
    )
    leave_one_out_credit_training_bits = (
        3
        * args.composition_updates
        * args.batch_size
        * args.span
        * sum(len(program) for _, program in train_specs)
        if args.leave_one_out_credit_weight
        else 0
    )
    causal_selection_training_bits = (
        3
        * _causal_selection_bits(
            train_specs,
            updates=args.composition_updates,
            batch_size=args.batch_size,
            span=args.span,
            candidate_multiplier=args.causal_candidate_multiplier,
            probe_samples=args.causal_probe_samples,
        )
        if args.causal_selection != "none"
        else 0
    )
    causal_selection_candidate_lifetimes = (
        3
        * args.composition_updates
        * args.batch_size
        * args.causal_candidate_multiplier
        * target_count
        if args.causal_selection != "none"
        else 0
    )
    audit_batches = (
        parent_eval_points
        + len(PRIMITIVES) * _eval_points(args.primitive_updates, args.eval_every)
        + 2 * target_count * composition_eval_points
        + sum(range(1, len(PRIMITIVES) + 1))
        + 2 * len(PRIMITIVES)
        + 8 * target_count
    )
    gates = {
        "primitives_mastered": min(source_before.values()) >= args.mastery_threshold,
        "primitives_stable": all(
            value is not None for value in primitive_stable_bits.values()
        ),
        "primitives_retained": min(source_after.values()) >= args.mastery_threshold,
        "shared_train_targets_mastered": min(shared_train_accuracy)
        >= args.mastery_threshold,
        "shared_train_targets_stable": shared_stable_bits is not None,
        "shared_fresh_control_stable": fresh_stable_bits is not None,
        "positive_stable_transfer": (
            shared_stable_bits is not None
            and fresh_stable_bits is not None
            and fresh_stable_bits > shared_stable_bits
        ),
        "heldout_orders_generalize": min(shared_heldout_accuracy)
        >= args.mastery_threshold,
        "wrong_orders_rejected": max(shared_wrong_accuracy) < args.mastery_threshold,
        "no_fragment_bypass": max(shared_zero_accuracy) < args.mastery_threshold,
        "missing_evidence_rejected": max(shared_missing_accuracy)
        < args.mastery_threshold,
        "reward_shuffled_rejected": max(shuffled_accuracy) < args.mastery_threshold,
        "one_shared_external_learner": True,
        "frozen_parent": parent_digest_before == parent_digest_after,
        "frozen_acquired_bank": bank_digest_before == bank_digest_after,
        "persistence_exact_and_corruption_rejected": persistence_exact,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-skill-fragment-shared-multi-target-report.v1",
        "claim_boundary": (
            "One shared external trace combiner and decoder reuse four acquired "
            f"opaque fragments across {len(train_specs)} training orders and "
            f"{len(heldout_specs)} held-out orders while the acquired bank and "
            "parent remain frozen. This does "
            "not establish arbitrary program induction, unrestricted growth, "
            "compression, or general continual learning."
        ),
        "seed": args.seed,
        "primitives": list(PRIMITIVES),
        "training_orders": [list(order) for order in train_orders],
        "heldout_orders": [list(order) for order in heldout_orders],
        "parent_progress": parent_progress,
        "retention_by_stage": retention_by_stage,
        "primitive_histories": primitive_histories,
        "primitive_stable_bits": primitive_stable_bits,
        "source_before": source_before,
        "source_after": source_after,
        "shared": {
            "train_accuracy": shared_train_accuracy,
            "heldout_accuracy": shared_heldout_accuracy,
            "wrong_accuracy": shared_wrong_accuracy,
            "zero_fragment_accuracy": shared_zero_accuracy,
            "missing_evidence_accuracy": shared_missing_accuracy,
            "history": shared_history,
            "combiner_mode": args.combiner_mode,
            "combiner_configuration": (
                shared_combiner.configuration()
                if hasattr(shared_combiner, "configuration")
                else None
            ),
            "combiner_count": 1,
            "decoder_count": 1,
            "order_contrast_weight": args.order_contrast_weight,
            "order_contrast_training_specs": contrast_spec_count,
            "prefix_credit_weight": args.prefix_credit_weight,
            "prefix_credit_protocol": (
                "causal_prefix_verifier_outcomes_v1"
                if args.prefix_credit_weight
                else None
            ),
            "leave_one_out_credit_weight": args.leave_one_out_credit_weight,
            "leave_one_out_credit_protocol": (
                "common_random_leave_one_prefix_out_v1"
                if args.leave_one_out_credit_weight
                else None
            ),
            "causal_selection": args.causal_selection,
            "causal_selection_protocol": (
                "common_render_stochastic_answer_change_top_k_v2"
                if args.causal_selection == "active"
                else "common_render_stochastic_matched_random_k_v2"
                if args.causal_selection == "passive"
                else None
            ),
            "causal_candidate_multiplier": args.causal_candidate_multiplier,
            "causal_probe_temperature": args.causal_probe_temperature,
            "causal_probe_samples": args.causal_probe_samples,
        },
        "reward_shuffled": {"accuracy": shuffled_accuracy},
        "fresh": {
            "train_accuracy": fresh_train_accuracy,
            "heldout_accuracy": fresh_heldout_accuracy,
            "history": fresh_history,
            "combiner_mode": args.combiner_mode,
            "combiner_count": 1,
            "decoder_count": 1,
            "order_contrast_weight": args.order_contrast_weight,
            "prefix_credit_weight": args.prefix_credit_weight,
            "leave_one_out_credit_weight": args.leave_one_out_credit_weight,
            "causal_selection": args.causal_selection,
            "causal_candidate_multiplier": args.causal_candidate_multiplier,
            "causal_probe_temperature": args.causal_probe_temperature,
            "causal_probe_samples": args.causal_probe_samples,
        },
        "acquired_bank": {
            "sha256_before_targets": bank_digest_before,
            "sha256_after_targets": bank_digest_after,
        },
        "stable_bits_to_threshold": shared_stable_bits,
        "fresh_stable_bits_to_threshold": fresh_stable_bits,
        "transfer_ratio_fresh_over_shared": (
            float(fresh_stable_bits) / float(shared_stable_bits)
            if shared_stable_bits and fresh_stable_bits
            else None
        ),
        "accounting": {
            "unique_verifier_bits": (training_batches + audit_batches)
            * bits_per_update
            + prefix_credit_training_bits
            + leave_one_out_credit_training_bits
            + causal_selection_training_bits,
            "training_unique_verifier_bits": training_batches * bits_per_update
            + prefix_credit_training_bits
            + leave_one_out_credit_training_bits
            + causal_selection_training_bits,
            "prefix_credit_unique_verifier_bits": prefix_credit_training_bits,
            "leave_one_out_credit_unique_verifier_bits": (
                leave_one_out_credit_training_bits
            ),
            "causal_selection_unique_verifier_bits": causal_selection_training_bits,
            "audit_unique_verifier_bits": audit_batches
            * args.audit_count
            * args.span
            * 2,
            "unique_logical_lifetimes": training_batches * args.batch_size,
            "causal_selection_candidate_lifetimes": causal_selection_candidate_lifetimes,
            "audit_logical_lifetimes": audit_batches * args.audit_count,
            "optimizer_updates": (
                args.parent_updates
                + len(PRIMITIVES) * args.primitive_updates
                + 3 * args.composition_updates
            ),
            "counterfactual_rollouts": (
                3 * contrast_spec_count * args.composition_updates
                if args.order_contrast_weight
                else 0
            ),
            "prefix_credit_rollouts": (
                3
                * args.composition_updates
                * args.batch_size
                * args.span
                * sum(len(program) for _, program in train_specs)
                if args.prefix_credit_weight
                else 0
            ),
            "leave_one_out_credit_rollouts": (
                3
                * args.composition_updates
                * args.batch_size
                * args.span
                * sum(len(program) for _, program in train_specs)
                if args.leave_one_out_credit_weight
                else 0
            ),
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=64)
    parser.add_argument("--primitive-updates", type=int, default=256)
    parser.add_argument("--composition-updates", type=int, default=128)
    parser.add_argument(
        "--combiner-mode",
        choices=("trace", "segment", "serial", "serial_shared", "operator"),
        default="segment",
        help="external composition codec or direct interpreter trace to pressure-test",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="train on shorter programs before the held-out depth rung",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--span", type=int, default=3)
    parser.add_argument("--audit-count", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument(
        "--order-contrast-weight",
        type=float,
        default=0.0,
        help="weight a trainer-only wrong-order counterfactual loss",
    )
    parser.add_argument(
        "--prefix-credit-weight",
        type=float,
        default=0.0,
        help="weight fresh causal verifier outcomes for every execution prefix",
    )
    parser.add_argument(
        "--leave-one-out-credit-weight",
        type=float,
        default=0.0,
        help="weight common-random final utility for omitting each transition",
    )
    parser.add_argument(
        "--causal-selection",
        choices=("none", "active", "passive"),
        default="none",
        help="select answer-changing candidates or use the matched passive control",
    )
    parser.add_argument(
        "--causal-candidate-multiplier",
        type=int,
        default=2,
        help="candidate rows probed per trained row for causal selection",
    )
    parser.add_argument(
        "--causal-probe-temperature",
        type=float,
        default=0.5,
        help="temperature for stochastic trainer-only causal probes",
    )
    parser.add_argument(
        "--causal-probe-samples",
        type=int,
        default=4,
        help="common-random samples per candidate causal probe",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
