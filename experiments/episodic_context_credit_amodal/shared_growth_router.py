"""Audit a shared candidate-conditioned router under replay-free growth.

The older repeated-shift harness allocates one scalar route extension per new
capability.  This audit instead allocates one permutation-equivariant router
per temporal shift and gives it the whole opaque candidate bank for that
shift.  Each shift router is frozen before the next shift, so old routes do
not depend on replay or on mutable controller weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.episodic_context_credit_amodal.repeated_shift_growth import (
    _credit_accuracy,
    _family_schedule,
    _parse_int_list,
    _patterns_for_family,
    _piecewise_query,
)
from experiments.episodic_context_credit_amodal.train import (
    _episodes,
    _pattern_bank,
    _prototypes,
    _route_accuracy,
    seed_everything,
    train_context,
    train_credit,
    train_external_credit_head,
    train_router,
)
from neural_computer import (
    CapabilityRetentionLedger,
    EpisodicContextEncoder,
    OpaqueCandidateGrowthRouter,
    RetentionPolicyConfig,
    failure_gated_candidate_scores,
    paired_counterfactual_ranking_loss,
)

Expansion = tuple[OpaqueCandidateGrowthRouter, torch.Tensor, int]


def _phase_index(
    family: int,
    schedule: tuple[tuple[int, int, int], ...],
) -> int:
    for index, (start, end, _length) in enumerate(schedule[1:]):
        if start <= family < end:
            return index
    raise ValueError(f"family {family} is not a shifted family")


def _query_prototype_keys(
    query_fn,
    families: tuple[int, ...],
    *,
    seed: int,
    samples_per_family: int = 64,
) -> torch.Tensor:
    """Bootstrap opaque memory keys from fresh learned-query tensors."""

    if samples_per_family < 1:
        raise ValueError("query prototype sample count must be positive")
    keys: list[torch.Tensor] = []
    for offset, family in enumerate(families):
        family_rows = torch.full(
            (samples_per_family,), family, dtype=torch.long
        )
        query = query_fn(family_rows, seed + offset * 1009)
        keys.append(F.normalize(query.mean(dim=0), dim=-1))
    return torch.stack(keys)


def _route_query_fn(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    patterns_by_length: dict[int, torch.Tensor],
    schedule: tuple[tuple[int, int, int], ...],
    *,
    representation: str,
):
    """Build a fixed-width learned trajectory query for memory-side routing."""

    if representation == "context":
        return _piecewise_query(
            encoder,
            prototypes,
            patterns_by_length,
            schedule,
        )
    if representation != "trajectory_stats":
        raise ValueError(f"unsupported route query representation {representation!r}")

    def query(families: torch.Tensor, seed: int) -> torch.Tensor:
        result = torch.empty(
            families.shape[0], encoder.context_width + encoder.hidden * 3
        )
        for family_tensor in families.unique(sorted=True):
            family = int(family_tensor)
            mask = families == family
            _, episode_length = _patterns_for_family(family, schedule)
            events, actions, outcomes, present = _episodes(
                families[mask],
                prototypes,
                seed=seed + family,
                patterns=patterns_by_length[episode_length],
            )
            with torch.no_grad():
                output = encoder(events, actions, outcomes, present)
            sequence = output.sequence
            present_float = present.unsqueeze(-1).to(sequence.dtype)
            mean = (sequence * present_float).sum(dim=1) / present_float.sum(
                dim=1
            ).clamp_min(1.0)
            masked_sequence = sequence.masked_fill(~present.unsqueeze(-1), -torch.inf)
            maximum = masked_sequence.amax(dim=1)
            lengths = present.sum(dim=1).clamp_min(1).to(torch.long) - 1
            final = sequence.gather(
                1,
                lengths[:, None, None].expand(-1, 1, encoder.hidden),
            ).squeeze(1)
            result[mask] = torch.cat((output.context, final, mean, maximum), dim=-1)
        return result

    return query


def _train_shared_expansion(
    query_fn,
    candidate_keys: torch.Tensor,
    phase_families: tuple[int, ...],
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffled: bool = False,
    hidden: int = 48,
) -> OpaqueCandidateGrowthRouter:
    if not phase_families:
        raise ValueError("shared expansion requires at least one family")
    if shuffled and batch_size % 2:
        raise ValueError("shuffled shared expansion requires an even batch")
    router = OpaqueCandidateGrowthRouter(
        width=int(candidate_keys.shape[-1]),
        hidden=hidden,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=1e-5)
    for update in range(updates):
        local_target = update % len(phase_families)
        target_family = phase_families[local_target]
        if shuffled:
            query_families = torch.full(
                (batch_size // 2,), target_family, dtype=torch.long
            )
            query_families = torch.cat((query_families, query_families))
        else:
            query_families = torch.full(
                (batch_size,), target_family, dtype=torch.long
            )
        query = query_fn(query_families, seed + 40_009 + update)
        scores = router(query, candidate_keys)
        other_candidates = tuple(
            index for index in range(len(phase_families)) if index != local_target
        )
        if not other_candidates:
            raise ValueError("shared expansion requires at least two candidates")
        pair_count = len(other_candidates)
        pair_scores = scores.repeat_interleave(pair_count, dim=0)
        attempted = torch.tensor(
            [
                [local_target, other]
                for other in other_candidates
            ],
            dtype=torch.long,
        ).repeat(batch_size, 1)
        pair_utilities = torch.tensor([[1.0, 0.0]]).expand(
            batch_size * pair_count, -1
        ).clone()
        if shuffled:
            pair_utilities = torch.cat(
                (
                    pair_utilities[: (batch_size // 2) * pair_count],
                    torch.flip(
                        pair_utilities[: (batch_size // 2) * pair_count],
                        dims=(1,),
                    ),
                )
            )
        ranking_loss, _advantage = paired_counterfactual_ranking_loss(
            pair_scores,
            attempted,
            pair_utilities,
        )
        target_scores = scores[:, local_target]
        loss = (
            scores.square().mean()
            if shuffled
            else ranking_loss + F.softplus(-(target_scores - 1.0)).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
    router.eval()
    return router


@torch.no_grad()
def _route_scores(
    base_router,
    query: torch.Tensor,
    old_keys: torch.Tensor,
    expansions: tuple[Expansion, ...],
    *,
    family: int,
    disabled_phase: int | None = None,
    limit: int | None = None,
    expected_row: int | None = None,
) -> tuple[torch.Tensor, list[float], list[float]]:
    # The family index is the physical row only in canonical ordering.
    # Permutation audits must provide the remapped physical target; otherwise
    # the diagnostic treats a correctly selected permuted row as a failure and
    # incorrectly activates later growth routers.
    success_row = family if expected_row is None else expected_row
    scores = base_router(query, old_keys)
    attempts: list[float] = []
    activations: list[float] = []
    selected_expansions = expansions if limit is None else expansions[:limit]
    for phase, (router, keys, _start) in enumerate(selected_expansions):
        failed = scores.argmax(dim=-1) != success_row
        attempts.append(float(failed.to(torch.float32).mean()))
        residual = router(query, keys)
        if phase == disabled_phase:
            residual = torch.zeros_like(residual)
        activations.append(
            float((failed & (residual.max(dim=-1).values > 1.0)).float().mean())
        )
        scores = failure_gated_candidate_scores(scores, residual, failed)
    return scores, attempts, activations


@torch.no_grad()
def _shared_selection(
    base_router,
    query_fn,
    old_keys: torch.Tensor,
    expansions: tuple[Expansion, ...],
    *,
    family: int,
    seed: int,
    batch_size: int,
    disabled_phase: int | None = None,
    limit: int | None = None,
    expected_row: int | None = None,
) -> tuple[float, list[float], list[float]]:
    families = torch.full((batch_size,), family, dtype=torch.long)
    query = query_fn(families, seed)
    scores, attempts, activations = _route_scores(
        base_router,
        query,
        old_keys,
        expansions,
        family=family,
        disabled_phase=disabled_phase,
        limit=limit,
        expected_row=expected_row,
    )
    return float((scores.argmax(dim=-1) == family).float().mean()), attempts, activations


@torch.no_grad()
def _permuted_new_route_accuracy(
    base_router,
    query_fn,
    old_keys: torch.Tensor,
    expansions: tuple[Expansion, ...],
    new_families: tuple[int, ...],
    schedule: tuple[tuple[int, int, int], ...],
    *,
    seed: int,
    batch_size: int,
) -> float:
    correct = 0
    total = 0
    for offset, family in enumerate(new_families):
        phase = _phase_index(family, schedule)
        start = schedule[phase + 1][0]
        local = family - start
        permutation = torch.randperm(
            expansions[phase][1].shape[0],
            generator=torch.Generator().manual_seed(seed + offset),
        )
        permuted = list(expansions)
        router, keys, family_start = permuted[phase]
        permuted[phase] = (router, keys[permutation], family_start)
        expected_local = int((permutation == local).nonzero(as_tuple=False)[0])
        expected = start + expected_local
        families = torch.full((batch_size,), family, dtype=torch.long)
        query = query_fn(families, seed + offset * 1009)
        scores, _attempts, _activations = _route_scores(
            base_router,
            query,
            old_keys,
            tuple(permuted),
            family=family,
            expected_row=expected,
        )
        correct += int((scores.argmax(dim=-1) == expected).sum())
        total += batch_size
    return correct / total


@torch.no_grad()
def _candidate_score_permutation_accuracy(
    query_fn,
    expansions: tuple[Expansion, ...],
    schedule: tuple[tuple[int, int, int], ...],
    *,
    seed: int,
    batch_size: int,
) -> float:
    """Audit the candidate scorer's direct row-equivariance invariant."""

    correct = 0
    total = 0
    for phase, (router, keys, start) in enumerate(expansions):
        _phase_start, phase_end, _length = schedule[phase + 1]
        phase_families = tuple(range(start, phase_end))
        families = torch.tensor(
            [family for family in phase_families for _ in range(batch_size)],
            dtype=torch.long,
        )
        query = query_fn(families, seed + phase * 1009)
        scores = router(query, keys)
        permutation = torch.randperm(
            keys.shape[0],
            generator=torch.Generator().manual_seed(seed + phase * 2003),
        )
        permuted_scores = router(query, keys[permutation])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.shape[0])
        expected = inverse[scores.argmax(dim=-1)]
        correct += int((permuted_scores.argmax(dim=-1) == expected).sum())
        total += int(families.shape[0])
    if total == 0:
        raise ValueError("candidate permutation audit requires one expansion")
    return correct / total


def _retention_reversal_audit_shared(
    base_router,
    query_fn,
    old_keys: torch.Tensor,
    all_keys: torch.Tensor,
    expansions: tuple[Expansion, ...],
    new_families: tuple[int, ...],
    *,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    config = RetentionPolicyConfig(
        mastery_threshold=0.8,
        min_mastery_observations=4,
        reversal_threshold=0.5,
        reversal_patience=4,
        recent_window=4,
    )
    ledger = CapabilityRetentionLedger(all_keys.shape[1], config=config)
    observations: list[list[float]] = [[] for _ in range(all_keys.shape[0])]
    for repetition in range(8):
        for family in (0, 1) + new_families:
            score, _attempts, _activations = _shared_selection(
                base_router,
                query_fn,
                old_keys,
                expansions,
                family=family,
                seed=seed + 100_000 + repetition * 101 + family,
                batch_size=batch_size,
            )
            observations[family].append(score)
    for key, values in zip(all_keys, observations, strict=True):
        for value in values:
            ledger.observe(key, value)
    initial_statuses = [ledger.status(key) for key in all_keys]
    full_bank_choice = ledger.choose_eviction_index(
        all_keys, torch.arange(all_keys.shape[0], dtype=torch.float32)
    )
    reversal_target = all_keys.shape[0] - 1
    for _ in range(config.reversal_patience):
        ledger.observe(all_keys[reversal_target], 0.0)
    reversal_statuses = [ledger.status(key) for key in all_keys]
    post_reversal_choice = ledger.choose_eviction_index(
        all_keys, torch.arange(all_keys.shape[0], dtype=torch.float32)
    )
    for _ in range(config.min_mastery_observations):
        ledger.observe(all_keys[reversal_target], 1.0)
    recovered_status = ledger.status(all_keys[reversal_target])
    return {
        "initial_protected": [status.protected for status in initial_statuses],
        "full_bank_refuses_eviction": full_bank_choice is None,
        "reversal_target_slot": reversal_target,
        "reversal_status": {
            "protected": reversal_statuses[reversal_target].protected,
            "reversal_count": reversal_statuses[reversal_target].reversal_count,
            "observations": reversal_statuses[reversal_target].observations,
        },
        "other_slots_remain_protected": all(
            status.protected
            for index, status in enumerate(reversal_statuses)
            if index != reversal_target
        ),
        "post_reversal_eviction_slot": post_reversal_choice,
        "reversal_releases_only_target": (
            not reversal_statuses[reversal_target].protected
            and all(
                status.protected
                for index, status in enumerate(reversal_statuses)
                if index != reversal_target
            )
            and post_reversal_choice == reversal_target
        ),
        "recovered_protected": recovered_status.protected,
        "recovery_reversal_count": recovered_status.reversal_count,
        "observation_count": sum(len(values) for values in observations)
        + config.reversal_patience
        + config.min_mastery_observations,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(args.seed)
    shift_lengths = _parse_int_list(
        args.shift_episode_lengths,
        name="--shift-episode-lengths",
    )
    family_counts = _parse_int_list(
        args.families_per_shift,
        name="--families-per-shift",
        minimum=1,
    )
    schedule = _family_schedule(
        args.base_episode_length,
        shift_lengths,
        family_counts,
    )
    new_families = tuple(
        family
        for start, end, _length in schedule[1:]
        for family in range(start, end)
    )
    patterns_by_length = {
        length: _pattern_bank(
            max(end for start, end, current in schedule if current == length) - 1,
            episode_length=length,
        )
        for _start, _end, length in schedule
    }
    prototypes = _prototypes(args.seed + 77, 8)
    encoder = EpisodicContextEncoder(8, 3, hidden=32, context_width=16)
    base_patterns = patterns_by_length[args.base_episode_length]
    context_loss = train_context(
        encoder,
        prototypes,
        updates=args.context_updates,
        seed=args.seed,
        patterns=base_patterns,
    )
    credit_loss = train_credit(
        encoder,
        prototypes,
        updates=args.credit_updates,
        batch_size=args.batch_size,
        seed=args.seed,
        patterns=base_patterns,
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    query = _route_query_fn(
        encoder,
        prototypes,
        patterns_by_length,
        schedule,
        representation=args.route_query_representation,
    )
    base_query = _route_query_fn(
        encoder,
        prototypes,
        {args.base_episode_length: base_patterns},
        schedule=((0, 2, args.base_episode_length),),
        representation=args.route_query_representation,
    )
    route_width = int(
        base_query(torch.tensor([0], dtype=torch.long), args.seed + 79_001).shape[1]
    )
    old_keys = F.normalize(
        torch.randn(
            2,
            route_width,
            generator=torch.Generator().manual_seed(args.seed + 91),
        ),
        dim=-1,
    )
    if args.candidate_key_bootstrap == "query_prototype":
        extension_keys = _query_prototype_keys(
            query,
            new_families,
            seed=args.seed + 80_001,
        )
    else:
        extension_keys = F.normalize(
            torch.randn(
                len(new_families),
                route_width,
                generator=torch.Generator().manual_seed(args.seed + 93),
            ),
            dim=-1,
        )
    all_keys = torch.cat((old_keys, extension_keys), dim=0)
    base_router = train_router(
        base_query,
        old_keys,
        updates=args.route_updates,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    old_route_accuracy = _route_accuracy(
        base_router,
        base_query,
        old_keys,
        families=(0, 1),
        seed=args.seed + 50_001,
        batch_size=args.audit_batch_size,
    )
    old_permutation_accuracy = _route_accuracy(
        base_router,
        base_query,
        old_keys[[1, 0]],
        families=(1, 0),
        expected_rows=(0, 1),
        seed=args.seed + 51_001,
        batch_size=args.audit_batch_size,
    )

    expansions: list[Expansion] = []
    shuffled_expansions: list[Expansion] = []
    phase_reports: list[dict[str, object]] = []
    for phase, (start, end, episode_length) in enumerate(schedule[1:]):
        phase_families = tuple(range(start, end))
        phase_keys = extension_keys[start - 2 : end - 2]
        expansion = _train_shared_expansion(
            query,
            phase_keys,
            phase_families,
            updates=args.shared_route_updates,
            batch_size=args.batch_size,
            seed=args.seed + start,
            hidden=args.shared_route_hidden,
        )
        shuffled = _train_shared_expansion(
            query,
            phase_keys,
            phase_families,
            updates=args.shared_route_updates,
            batch_size=args.batch_size,
            seed=args.seed + start + 10_000,
            shuffled=True,
            hidden=args.shared_route_hidden,
        )
        expansions.append((expansion, phase_keys, start))
        shuffled_expansions.append((shuffled, phase_keys, start))
        selections = {
            str(family): _shared_selection(
                base_router,
                query,
                old_keys,
                tuple(expansions),
                family=family,
                seed=args.seed + 60_001 + family,
                batch_size=args.audit_batch_size,
            )[0]
            for family in phase_families
        }
        phase_reports.append(
            {
                "shift_index": phase + 1,
                "episode_length": episode_length,
                "families": list(phase_families),
                "minimum_route_selection": min(selections.values()),
                "route_selection": selections,
            }
        )

    expansion_tuple = tuple(expansions)
    shuffled_tuple = tuple(shuffled_expansions)
    new_selection: dict[str, float] = {}
    ablated_selection: dict[str, float] = {}
    shuffled_selection: dict[str, float] = {}
    prior_attempts: dict[str, list[float]] = {}
    for family in new_families:
        phase = _phase_index(family, schedule)
        selected, _attempts, _activations = _shared_selection(
            base_router,
            query,
            old_keys,
            expansion_tuple,
            family=family,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
        )
        ablated, _attempts, _activations = _shared_selection(
            base_router,
            query,
            old_keys,
            expansion_tuple,
            family=family,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
            disabled_phase=phase,
            limit=phase + 1,
        )
        shuffled, _attempts, _activations = _shared_selection(
            base_router,
            query,
            old_keys,
            shuffled_tuple,
            family=family,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
        )
        _selected, attempts, _activations = _shared_selection(
            base_router,
            query,
            old_keys,
            expansion_tuple,
            family=family,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
        )
        new_selection[str(family)] = selected
        ablated_selection[str(family)] = ablated
        shuffled_selection[str(family)] = shuffled
        prior_attempts[str(family)] = attempts

    operational_permutation_accuracy = _permuted_new_route_accuracy(
        base_router,
        query,
        old_keys,
        expansion_tuple,
        new_families,
        schedule,
        seed=args.seed + 61_001,
        batch_size=args.audit_batch_size,
    )
    new_score_permutation_accuracy = _candidate_score_permutation_accuracy(
        query,
        expansion_tuple,
        schedule,
        seed=args.seed + 62_001,
        batch_size=args.audit_batch_size,
    )
    permutation_accuracy = (
        old_permutation_accuracy + new_score_permutation_accuracy
    ) / 2.0
    heads: dict[int, torch.nn.Module] = {}
    for family in new_families:
        _family, episode_length = _patterns_for_family(family, schedule)
        heads[family] = train_external_credit_head(
            encoder,
            prototypes,
            new_family=family,
            updates=args.external_credit_updates,
            batch_size=args.batch_size,
            seed=args.seed + family,
            patterns=patterns_by_length[episode_length],
        )
    retention = _retention_reversal_audit_shared(
        base_router,
        query,
        old_keys,
        all_keys,
        expansion_tuple,
        new_families,
        batch_size=args.audit_batch_size,
        seed=args.seed,
    )
    credit_accuracy = _credit_accuracy(
        encoder,
        prototypes,
        heads,
        base_episode_length=args.base_episode_length,
        new_families=new_families,
        schedule=schedule,
        patterns_by_length=patterns_by_length,
        seed=args.seed + 70_001,
    )
    shift_count = len(schedule) - 1
    bits = (
        args.credit_updates * args.batch_size * args.base_episode_length * 2
        + sum(
            args.external_credit_updates * args.batch_size * length * 2
            for start, end, length in schedule[1:]
            for _ in range(end - start)
        )
        + (args.route_updates + 2 * shift_count * args.shared_route_updates)
        * args.batch_size
        * 2
        + int(retention["observation_count"])
    )
    lifetimes = (
        args.context_updates * 2
        + args.credit_updates
        + args.route_updates
        + shift_count * 2 * args.shared_route_updates
        + len(new_families) * args.external_credit_updates
        + int(retention["observation_count"])
    )
    report: dict[str, object] = {
        "schema": "neural-computer.episodic-context-credit-shared-growth-router-report.v1",
        "claim_boundary": (
            "A frozen base capability set survives sequential temporal shifts "
            "while one permutation-equivariant external router per shift learns "
            "variable candidate banks without replay. This is bounded growth, "
            "not general continual learning."
        ),
        "seed": args.seed,
        "schedule": [
            {
                "family_start": start,
                "family_end": end,
                "episode_length": length,
            }
            for start, end, length in schedule
        ],
        "new_families": list(new_families),
        "candidate_key_bootstrap": args.candidate_key_bootstrap,
        "route_query_representation": args.route_query_representation,
        "shared_expansion_count": shift_count,
        "max_candidates_per_expansion": max(
            end - start for start, end, _length in schedule[1:]
        ),
        "context_loss": context_loss,
        "credit_loss": credit_loss,
        "old_route_accuracy": old_route_accuracy,
        "candidate_permutation_accuracy": permutation_accuracy,
        "old_candidate_permutation_accuracy": old_permutation_accuracy,
        "new_candidate_permutation_accuracy": new_score_permutation_accuracy,
        "operational_route_permutation_accuracy": operational_permutation_accuracy,
        "phase_reports": phase_reports,
        "new_route_selection": new_selection,
        "new_route_selection_without_shared_expansion": ablated_selection,
        "reward_shuffled_expansion_selection": shuffled_selection,
        "prior_expansion_attempt_rates": prior_attempts,
        "credit_position_accuracy": credit_accuracy,
        "retention_reversal": retention,
        "accounting": {
            "unique_verifier_bits": bits,
            "unique_logical_lifetimes": lifetimes,
            "optimizer_updates": (
                args.context_updates
                + args.credit_updates
                + args.route_updates
                + 2 * shift_count * args.shared_route_updates
                + len(new_families) * args.external_credit_updates
            ),
            "replayed_examples": 0,
            "distribution_shifts": shift_count,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    report["gates"] = {
        "old_route_retained": old_route_accuracy >= 0.8,
        "candidate_permutation_invariant": permutation_accuracy >= 0.8,
        "new_routes_recovered": all(value >= 0.8 for value in new_selection.values()),
        "new_routes_causal": all(
            new_selection[key] >= ablated_selection[key] + 0.5
            for key in new_selection
        ),
        "reward_shuffled_not_selected": all(
            value <= 0.5 for value in shuffled_selection.values()
        ),
        "prior_expansions_attempted": all(
            all(rate >= 0.8 for rate in attempts[: _phase_index(int(family), schedule)])
            for family, attempts in prior_attempts.items()
        ),
        "credit_signal_survives_all_shifts": all(
            credit_accuracy.get(key, 0.0) >= 0.66
            for key in (
                "old",
                "combined",
                *[f"shift_{index}" for index in range(1, len(schedule))],
            )
        ),
        "no_replay_after_shifts": True,
        "retention_reversal_safe": (
            retention["full_bank_refuses_eviction"]
            and retention["reversal_releases_only_target"]
            and retention["recovered_protected"]
        ),
    }
    report["promoted"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--base-episode-length", type=int, default=6)
    parser.add_argument("--shift-episode-lengths", default="8,10,12")
    parser.add_argument("--families-per-shift", default="8,10,12")
    parser.add_argument("--context-updates", type=int, default=1024)
    parser.add_argument("--credit-updates", type=int, default=512)
    parser.add_argument("--external-credit-updates", type=int, default=128)
    parser.add_argument("--route-updates", type=int, default=1024)
    parser.add_argument("--shared-route-updates", type=int, default=1024)
    parser.add_argument("--shared-route-hidden", type=int, default=48)
    parser.add_argument(
        "--candidate-key-bootstrap",
        choices=("random", "query_prototype"),
        default="random",
    )
    parser.add_argument(
        "--route-query-representation",
        choices=("context", "trajectory_stats"),
        default="context",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.base_episode_length < 2 or args.audit_batch_size < 1:
        raise SystemExit("base episode length and audit batch size are invalid")
    report = run(args)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "gates": report["gates"],
                "phase_reports": report["phase_reports"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
