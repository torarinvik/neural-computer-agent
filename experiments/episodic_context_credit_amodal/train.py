"""Audit reusable episodic context and outcome-only external route growth."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from neural_computer import (
    CapabilityRetentionLedger,
    EpisodicContextEncoder,
    EpisodicCreditHead,
    ExternalGrowthPrior,
    FactorizedOpaqueAddressRouter,
    OpaqueViewRouteExtension,
    RetentionPolicyConfig,
    credit_weights_from_logits,
    episodic_context_contrastive_loss,
    failure_gated_view_scores,
    paired_counterfactual_policy_loss,
    paired_counterfactual_ranking_loss,
    paired_event_credit_loss,
)

PATTERNS = torch.tensor(
    [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ],
    dtype=torch.long,
)
EXTENDED_PATTERNS = torch.tensor(
    [
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1],
    ],
    dtype=torch.long,
)


def _pattern_bank(
    max_family: int,
    *,
    episode_length: int | None = None,
) -> torch.Tensor:
    if episode_length is not None:
        if episode_length < 2:
            raise ValueError("episode length must be at least two")
        ones = max(1, episode_length // 2)
        total_patterns = math.comb(episode_length, ones)
        if max_family >= total_patterns:
            raise ValueError("requested family exceeds the generated pattern bank")
        # Preserve the historical full-bank tensor for small experiments, but
        # avoid materializing a combinatorial tail that the current schedule
        # never addresses.  Family IDs are assigned in combinations() order,
        # so the bounded prefix is exactly the same deterministic namespace.
        materialized_count = (
            total_patterns
            if total_patterns <= 100_000
            else max_family + 1
        )
        rows = []
        for index, positions in enumerate(combinations(range(episode_length), ones)):
            if index >= materialized_count:
                break
            row = [0] * episode_length
            for position in positions:
                row[position] = 1
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)
    if max_family < len(PATTERNS):
        return PATTERNS
    if max_family < len(EXTENDED_PATTERNS):
        return EXTENDED_PATTERNS
    raise ValueError("requested family exceeds the available pattern bank")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prototypes(seed: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = F.normalize(torch.randn(width, generator=generator), dim=0)
    second = torch.randn(width, generator=generator)
    second = second - first * torch.dot(first, second)
    return torch.stack((first, F.normalize(second, dim=0)))


def _base_events(
    families: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    patterns: torch.Tensor = PATTERNS,
) -> torch.Tensor:
    pattern = patterns.to(device=families.device)[families]
    return prototypes.to(families.device)[pattern]


def _episodes(
    families: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    seed: int,
    noise: float = 0.25,
    action_width: int = 3,
    patterns: torch.Tensor = PATTERNS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    events = _base_events(families, prototypes, patterns=patterns)
    events = events + noise * torch.randn(events.shape, generator=generator)
    actions = torch.zeros(
        families.shape[0], patterns.shape[1], action_width, dtype=torch.float32
    )
    actions[..., 0] = 1.0
    outcomes = torch.zeros(families.shape[0], patterns.shape[1])
    present = torch.ones_like(outcomes, dtype=torch.bool)
    return events, actions, outcomes, present


def _encode(
    encoder: EpisodicContextEncoder,
    families: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    seed: int,
    patterns: torch.Tensor = PATTERNS,
) -> torch.Tensor:
    events, actions, outcomes, present = _episodes(
        families, prototypes, seed=seed, patterns=patterns
    )
    with torch.no_grad():
        return encoder(events, actions, outcomes, present).context.detach()


def train_context(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    *,
    updates: int,
    seed: int,
    patterns: torch.Tensor = PATTERNS,
) -> float:
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=3e-3, weight_decay=1e-5)
    losses: list[float] = []
    for update in range(updates):
        families = torch.tensor([0, 1], dtype=torch.long)
        base = _base_events(families, prototypes, patterns=patterns)
        generator = torch.Generator(device="cpu").manual_seed(seed + update * 97)
        events_left = base + 0.30 * torch.randn(base.shape, generator=generator)
        events_right = base + 0.30 * torch.randn(base.shape, generator=generator)
        actions = torch.zeros(2, patterns.shape[1], 3)
        actions[..., 0] = 1.0
        outcomes = torch.zeros(2, patterns.shape[1])
        present = torch.ones_like(outcomes, dtype=torch.bool)
        left = encoder(events_left, actions, outcomes, present).context
        right = encoder(events_right, actions, outcomes, present).context
        loss = episodic_context_contrastive_loss(left, right, temperature=0.12)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses[-min(len(losses), 16) :]))


def train_credit(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    patterns: torch.Tensor = PATTERNS,
) -> float:
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-3, weight_decay=1e-5)
    losses: list[float] = []
    for update in range(updates):
        generator = torch.Generator(device="cpu").manual_seed(seed + 10_001 + update)
        families = torch.randint(0, 2, (batch_size,), generator=generator)
        events, actions, outcomes, present = _episodes(
            families,
            prototypes,
            seed=seed + 20_003 + update,
            patterns=patterns,
        )
        output = encoder(events, actions, outcomes, present)
        utilities = torch.zeros(batch_size, patterns.shape[1], 2)
        decisive = families.remainder(patterns.shape[1])
        utilities[torch.arange(batch_size), decisive, 0] = 1.0
        loss, _ = paired_event_credit_loss(
            output.credit_logits, utilities, present=present
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses[-min(len(losses), 16) :]))


def train_external_credit_head(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    *,
    new_family: int,
    updates: int,
    batch_size: int,
    seed: int,
    patterns: torch.Tensor = PATTERNS,
) -> EpisodicCreditHead:
    """Acquire isolated credit state for one appended capability."""
    head = EpisodicCreditHead(encoder.hidden, encoder.context_width)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-5)
    for update in range(updates):
        families = torch.full((batch_size,), new_family, dtype=torch.long)
        events, actions, outcomes, present = _episodes(
            families,
            prototypes,
            seed=seed + 80_001 + update,
            patterns=patterns,
        )
        with torch.no_grad():
            encoded = encoder(events, actions, outcomes, present)
        logits = head(
            encoded.sequence,
            encoded.context,
            outcomes,
            present,
        )
        utilities = torch.zeros(batch_size, patterns.shape[1], 2)
        decisive = torch.full((batch_size,), new_family, dtype=torch.long).remainder(
            patterns.shape[1]
        )
        utilities[torch.arange(batch_size), decisive, 0] = 1.0
        loss, _ = paired_event_credit_loss(
            logits, utilities, present=present
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    head.eval()
    return head


def train_router(
    query_fn: Callable[[torch.Tensor, int], torch.Tensor],
    keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffled: bool = False,
) -> FactorizedOpaqueAddressRouter:
    router = FactorizedOpaqueAddressRouter(width=int(keys.shape[-1]), hidden=48)
    optimizer = torch.optim.AdamW(router.parameters(), lr=4e-3, weight_decay=1e-5)
    pair_schedule = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    router.train()
    for update in range(updates):
        family = update % 2
        families = torch.full((batch_size,), family, dtype=torch.long)
        query = query_fn(families, seed + 30_007 + update)
        attempted = pair_schedule[update % len(pair_schedule)].expand(batch_size, -1)
        utilities = (attempted == family).to(torch.float32)
        if shuffled:
            generator = torch.Generator(device="cpu").manual_seed(seed + update)
            utilities = utilities[torch.randperm(batch_size, generator=generator)]
        loss, _ = paired_counterfactual_ranking_loss(
            router(query, keys), attempted, utilities
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
    router.eval()
    return router


def train_extension(
    query_fn: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    new_family: int,
    updates: int,
    batch_size: int,
    seed: int,
    shuffled: bool = False,
    antithetic_shuffled: bool = False,
    negative_families: tuple[int, ...] = (),
    growth_prior: ExternalGrowthPrior | None = None,
    growth_prior_reset_prefixes: tuple[str, ...] = ("score.",),
    growth_prior_mix: float = 1.0,
) -> OpaqueViewRouteExtension:
    """Acquire one route gate with fresh positive and anti-interference outcomes."""

    if negative_families and (batch_size < 2 or batch_size % 2):
        raise ValueError(
            "anti-interference extension training requires an even batch size"
        )
    if antithetic_shuffled and not shuffled:
        raise ValueError("antithetic shuffled training requires shuffled=True")
    if antithetic_shuffled and batch_size % 2:
        raise ValueError("antithetic shuffled training requires an even batch")
    extension = OpaqueViewRouteExtension(width=16, hidden=48)
    if growth_prior is not None:
        growth_prior.load_into(
            extension,
            reset_prefixes=growth_prior_reset_prefixes,
            mix=growth_prior_mix,
        )
    optimizer = torch.optim.AdamW(extension.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed + 70_013)
    for update in range(updates):
        if negative_families:
            positive_count = batch_size // 2
            negative_count = batch_size - positive_count
            negative = torch.tensor(
                [
                    negative_families[
                        (update * negative_count + index) % len(negative_families)
                    ]
                    for index in range(negative_count)
                ],
                dtype=torch.long,
            )
            families = torch.cat(
                (
                    torch.full((positive_count,), new_family, dtype=torch.long),
                    negative,
                )
            )
            utilities = torch.zeros(batch_size, 2)
            utilities[:positive_count, 0] = 1.0
            utilities[positive_count:, 1] = 1.0
        else:
            families = torch.full((batch_size,), new_family, dtype=torch.long)
            utilities = torch.tensor([[1.0, 0.0]]).expand(batch_size, -1).clone()
        query = query_fn(families, seed + 40_009 + update)
        if shuffled:
            if antithetic_shuffled:
                # Duplicate each query with exactly contradictory outcomes.
                # The null credit is then exactly zero, rather than merely
                # zero in expectation over a finite random shuffle.
                families = torch.full(
                    (batch_size // 2,), new_family, dtype=torch.long
                )
                families = torch.cat((families, families))
                utilities = torch.zeros(batch_size, 2)
                utilities[: batch_size // 2, 0] = 1.0
                utilities[batch_size // 2 :, 1] = 1.0
            else:
                if batch_size % 2:
                    raise ValueError("shuffled extension control needs an even batch")
                utilities = torch.zeros(batch_size, 2)
                utilities[: batch_size // 2, 0] = 1.0
                utilities[batch_size // 2 :, 1] = 1.0
                utilities = utilities[torch.randperm(batch_size, generator=generator)]
        scores = extension(query)
        _policy_loss, advantage = paired_counterfactual_policy_loss(
            scores, utilities
        )
        # The route transaction activates only above this opaque verifier
        # boundary. Train the score against that same boundary instead of
        # merely learning its sign; otherwise a mixed positive/negative batch
        # can be directionally correct while remaining below activation.
        loss = (
            scores.square().mean()
            if shuffled and antithetic_shuffled
            else _policy_loss
            if shuffled
            else F.softplus(-(advantage * (scores - 1.0))).mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extension.parameters(), 1.0)
        optimizer.step()
    extension.eval()
    return extension


def _route_accuracy(
    router: FactorizedOpaqueAddressRouter,
    query_fn: Callable[[torch.Tensor, int], torch.Tensor],
    keys: torch.Tensor,
    *,
    families: tuple[int, ...],
    expected_rows: tuple[int, ...] | None = None,
    seed: int,
    batch_size: int,
) -> float:
    if expected_rows is not None and len(expected_rows) != len(families):
        raise ValueError("expected route rows must align with families")
    correct = 0
    total = 0
    for index, family in enumerate(families):
        current = torch.full((batch_size,), family, dtype=torch.long)
        scores = router(query_fn(current, seed + index * 1009), keys)
        expected = family if expected_rows is None else expected_rows[index]
        correct += int((scores.argmax(dim=-1) == expected).sum())
        total += batch_size
    return correct / total


def _extension_selection(
    router: FactorizedOpaqueAddressRouter,
    query_fn: Callable[[torch.Tensor, int], torch.Tensor],
    keys: torch.Tensor,
    *,
    family: int,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    seed: int,
    batch_size: int,
    disabled_extension: int | None = None,
    activation_threshold: float = 1.0,
) -> tuple[float, float, list[float], list[float]]:
    families = torch.full((batch_size,), family, dtype=torch.long)
    query = query_fn(families, seed)
    route_scores = router(query, keys)
    old_failure = route_scores.argmax(dim=-1) != family
    prior_attempt_rates: list[float] = []
    activation_rates: list[float] = []
    for extension_index, extension in enumerate(extensions):
        extension_scores = (
            torch.zeros(batch_size)
            if disabled_extension == extension_index
            else extension(query)
        )
        prior_failed = route_scores.argmax(dim=-1) != family
        prior_attempt_rates.append(float(prior_failed.to(torch.float32).mean()))
        extension_active = prior_failed & (extension_scores > activation_threshold)
        activation_rates.append(float(extension_active.to(torch.float32).mean()))
        route_scores = failure_gated_view_scores(
            route_scores,
            extension_scores,
            extension_active,
        )
    selected = route_scores.argmax(dim=-1)
    return (
        float((selected == family).to(torch.float32).mean()),
        float((~old_failure).to(torch.float32).mean()),
        prior_attempt_rates,
        activation_rates,
    )


def _retention_reversal_audit(
    context_router: FactorizedOpaqueAddressRouter,
    extension_query: Callable[[torch.Tensor, int], torch.Tensor],
    keys: torch.Tensor,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    *,
    new_families: tuple[int, ...],
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    """Audit opaque protection, reversal, and recovery after sequential growth."""

    config = RetentionPolicyConfig(
        mastery_threshold=0.8,
        min_mastery_observations=4,
        reversal_threshold=0.5,
        reversal_patience=4,
        recent_window=4,
    )
    ledger = CapabilityRetentionLedger(keys.shape[1], config=config)
    observations: list[list[float]] = [[] for _ in range(keys.shape[0])]
    for repetition in range(8):
        for slot, family in enumerate((0, 1)):
            score = _route_accuracy(
                context_router,
                extension_query,
                keys[:2],
                families=(family,),
                seed=seed + 100_000 + repetition * 101 + family,
                batch_size=batch_size,
            )
            observations[slot].append(score)
        for offset, family in enumerate(new_families, start=2):
            score, _old_success, _attempts, _activations = _extension_selection(
                context_router,
                extension_query,
                keys[:2],
                family=family,
                extensions=extensions,
                seed=seed + 110_000 + repetition * 101 + family,
                batch_size=batch_size,
            )
            observations[offset].append(score)
    for key, values in zip(keys, observations, strict=True):
        for value in values:
            ledger.observe(key, value)
    initial_statuses = [ledger.status(key) for key in keys]
    full_bank_choice = ledger.choose_eviction_index(
        keys, torch.arange(keys.shape[0], dtype=torch.float32)
    )
    reversal_target = keys.shape[0] - 1
    for _ in range(config.reversal_patience):
        ledger.observe(keys[reversal_target], 0.0)
    reversal_statuses = [ledger.status(key) for key in keys]
    post_reversal_choice = ledger.choose_eviction_index(
        keys, torch.arange(keys.shape[0], dtype=torch.float32)
    )
    for _ in range(config.min_mastery_observations):
        ledger.observe(keys[reversal_target], 1.0)
    recovered_status = ledger.status(keys[reversal_target])
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


def run(
    seed: int,
    args: argparse.Namespace,
    *,
    new_families: tuple[int, ...],
) -> dict[str, object]:
    seed_everything(seed)
    expected_families = tuple(range(2, 2 + len(new_families)))
    if new_families != expected_families:
        raise ValueError("new families must be contiguous and start at family 2")
    patterns = _pattern_bank(
        max(new_families),
        episode_length=args.episode_length,
    )
    external_credit_updates = (
        args.credit_updates
        if args.external_credit_updates is None
        else args.external_credit_updates
    )
    if external_credit_updates < 1:
        raise ValueError("external credit updates must be positive")
    started = time.perf_counter()
    event_width = 8
    context_width = 16
    prototypes = _prototypes(seed + 77, event_width)
    encoder = EpisodicContextEncoder(
        event_width,
        3,
        hidden=32,
        context_width=context_width,
    )
    context_loss = train_context(
        encoder,
        prototypes,
        updates=args.context_updates,
        seed=seed,
        patterns=patterns,
    )
    credit_loss = train_credit(
        encoder,
        prototypes,
        updates=args.credit_updates,
        batch_size=args.batch_size,
        seed=seed,
        patterns=patterns,
    )
    encoder.eval()

    def context_query(families: torch.Tensor, query_seed: int) -> torch.Tensor:
        return _encode(
            encoder,
            families,
            prototypes,
            seed=query_seed,
            patterns=patterns,
        )

    def pooled_query(families: torch.Tensor, query_seed: int) -> torch.Tensor:
        events, _, _, _ = _episodes(
            families,
            prototypes,
            seed=query_seed,
            patterns=patterns,
        )
        return events.mean(dim=1)

    keys = F.normalize(
        torch.randn(2, context_width, generator=torch.Generator().manual_seed(seed + 91)),
        dim=-1,
    )
    extension_keys = F.normalize(
        torch.randn(
            len(new_families),
            context_width,
            generator=torch.Generator().manual_seed(seed + 93),
        ),
        dim=-1,
    )
    capability_keys = torch.cat((keys, extension_keys), dim=0)
    context_router = train_router(
        context_query,
        keys,
        updates=args.route_updates,
        batch_size=args.batch_size,
        seed=seed,
    )
    pooled_keys = F.normalize(
        torch.randn(2, event_width, generator=torch.Generator().manual_seed(seed + 92)),
        dim=-1,
    )
    pooled_router = train_router(
        pooled_query,
        pooled_keys,
        updates=args.route_updates,
        batch_size=args.batch_size,
        seed=seed + 1,
    )
    old_context_accuracy = _route_accuracy(
        context_router,
        context_query,
        keys,
        families=(0, 1),
        seed=seed + 50_001,
        batch_size=args.batch_size,
    )
    pooled_accuracy = _route_accuracy(
        pooled_router,
        pooled_query,
        pooled_keys,
        families=(0, 1),
        seed=seed + 50_001,
        batch_size=args.batch_size,
    )
    permutation = torch.tensor([1, 0])
    permutation_accuracy = _route_accuracy(
        context_router,
        context_query,
        keys[permutation],
        families=(1, 0),
        expected_rows=(0, 1),
        seed=seed + 51_001,
        batch_size=args.batch_size,
    )

    extension_query = context_query
    extensions = tuple(
        train_extension(
            extension_query,
            new_family=family,
            updates=args.extension_updates,
            batch_size=args.batch_size,
            seed=seed + family,
            negative_families=tuple(range(family)),
        )
        for family in new_families
    )
    extension_credit_heads = tuple(
        train_external_credit_head(
            encoder,
            prototypes,
            new_family=family,
            updates=external_credit_updates,
            batch_size=args.batch_size,
            seed=seed + family,
            patterns=patterns,
        )
        for family in new_families
    )
    shuffled_extensions = tuple(
        train_extension(
            extension_query,
            new_family=family,
            updates=args.extension_updates,
            batch_size=args.batch_size,
            seed=seed + family + 10,
            shuffled=True,
        )
        for family in new_families
    )
    new_selection: dict[str, float] = {}
    new_route_without_extension: dict[str, float] = {}
    shuffled_selection: dict[str, float] = {}
    old_failure: dict[str, float] = {}
    prior_attempt_rates: dict[str, list[float]] = {}
    extension_activation_rates: dict[str, list[float]] = {}
    for family in new_families:
        selected, old_success, attempts, activations = _extension_selection(
            context_router,
            extension_query,
            keys,
            family=family,
            extensions=extensions,
            seed=seed + 60_001 + family,
            batch_size=args.batch_size,
        )
        new_selection[str(family)] = selected
        old_failure[str(family)] = 1.0 - old_success
        prior_attempt_rates[str(family)] = attempts
        extension_activation_rates[str(family)] = activations
        disabled = family - 2
        ablated, _, _, _ = _extension_selection(
            context_router,
            extension_query,
            keys,
            family=family,
            extensions=extensions,
            disabled_extension=disabled,
            seed=seed + 60_001 + family,
            batch_size=args.batch_size,
        )
        new_route_without_extension[str(family)] = ablated
        shuffled, _, _, _ = _extension_selection(
            context_router,
            extension_query,
            keys,
            family=family,
            extensions=shuffled_extensions,
            seed=seed + 60_001 + family,
            batch_size=args.batch_size,
        )
        shuffled_selection[str(family)] = shuffled
    retention_reversal = _retention_reversal_audit(
        context_router,
        extension_query,
        capability_keys,
        extensions,
        new_families=new_families,
        batch_size=args.batch_size,
        seed=seed,
    )
    credit_eval_families = torch.arange(2 + len(new_families))
    credit_events, credit_actions, credit_outcomes, credit_present = _episodes(
        credit_eval_families,
        prototypes,
        seed=seed + 70_001,
        patterns=patterns,
    )
    with torch.no_grad():
        credit_output = encoder(
            credit_events, credit_actions, credit_outcomes, credit_present
        )
        old_credit_weights = credit_output.credit_weights[:2]
        new_credit_weights = []
        for index, head in enumerate(extension_credit_heads):
            logits = head(
                credit_output.sequence[2 + index : 3 + index],
                credit_output.context[2 + index : 3 + index],
                credit_outcomes[2 + index : 3 + index],
                credit_present[2 + index : 3 + index],
            )
            new_credit_weights.append(
                credit_weights_from_logits(
                    logits, credit_present[2 + index : 3 + index]
                )
            )
        new_credit_weights_tensor = torch.cat(new_credit_weights, dim=0)
    old_credit_position_accuracy = float(
        (
            old_credit_weights.argmax(dim=-1)
            == credit_eval_families[:2]
        ).to(torch.float32).mean()
    )
    new_credit_position_accuracy = float(
        (
            new_credit_weights_tensor.argmax(dim=-1)
            == credit_eval_families[2:].remainder(patterns.shape[1])
        ).to(torch.float32).mean()
    )
    credit_position_accuracy = float(
        (
            torch.cat((old_credit_weights, new_credit_weights_tensor), dim=0)
            .argmax(dim=-1)
            == credit_eval_families.remainder(patterns.shape[1])
        ).to(torch.float32).mean()
    )
    report: dict[str, object] = {
        "schema": "neural-computer.episodic-context-credit-multistep-report.v1",
        "seed": seed,
        "event_width": event_width,
        "episode_length": int(patterns.shape[1]),
        "pattern_bank_size": int(patterns.shape[0]),
        "pattern_ones": int(patterns[0].sum()),
        "old_families": [0, 1],
        "new_families": list(new_families),
        "context_updates": args.context_updates,
        "credit_updates": args.credit_updates,
        "external_credit_updates": external_credit_updates,
        "route_updates": args.route_updates,
        "extension_updates": args.extension_updates,
        "batch_size": args.batch_size,
        "context_contrastive_loss": context_loss,
        "credit_loss": credit_loss,
        "context_old_route_accuracy": old_context_accuracy,
        "pooled_baseline_old_route_accuracy": pooled_accuracy,
        "candidate_permutation_accuracy": permutation_accuracy,
        "new_route_selection": new_selection,
        "old_route_failure_on_new": old_failure,
        "prior_extension_attempt_rates": prior_attempt_rates,
        "extension_activation_rates": extension_activation_rates,
        "extension_activation_threshold": 1.0,
        "new_route_selection_without_extension": new_route_without_extension,
        "reward_shuffled_extension_selection": shuffled_selection,
        "credit_position_accuracy": credit_position_accuracy,
        "old_credit_position_accuracy": old_credit_position_accuracy,
        "new_credit_position_accuracy": new_credit_position_accuracy,
        "retention_reversal": retention_reversal,
        "accounting": {
            "self_supervised_context_updates": args.context_updates,
            "self_supervised_context_lifetimes": args.context_updates * 2,
            "counterfactual_credit_updates": args.credit_updates,
            "counterfactual_credit_lifetimes": args.credit_updates * args.batch_size,
            "counterfactual_credit_verifier_bits": (
                args.credit_updates * args.batch_size * patterns.shape[1] * 2
            ),
            "external_credit_head_updates": external_credit_updates * len(new_families),
            "external_credit_head_lifetimes": external_credit_updates * args.batch_size * len(new_families),
            "external_credit_head_verifier_bits": (
                external_credit_updates
                * args.batch_size
                * patterns.shape[1]
                * 2
                * len(new_families)
            ),
            "route_optimizer_updates": args.route_updates,
            "route_lifetimes": args.route_updates * args.batch_size,
            "route_verifier_bits": args.route_updates * args.batch_size * 2,
            "extension_optimizer_updates": args.extension_updates * len(new_families),
            "shuffled_extension_optimizer_updates": args.extension_updates * len(new_families),
            "extension_lifetimes": args.extension_updates * args.batch_size * len(new_families),
            "extension_verifier_bits": (
                args.extension_updates
                * args.batch_size
                * 2
                * len(new_families)
            ),
            "unique_logical_lifetimes": (
                args.context_updates * 2
                + (
                    args.credit_updates
                    + external_credit_updates * len(new_families)
                    + args.route_updates
                    + 2 * len(new_families) * args.extension_updates
                )
                * args.batch_size
                + retention_reversal["observation_count"]
            ),
            "unique_verifier_bits": (
                (
                    args.credit_updates
                    + external_credit_updates * len(new_families)
                )
                * args.batch_size
                * patterns.shape[1]
                * 2
                + (args.route_updates + 2 * len(new_families) * args.extension_updates)
                * args.batch_size
                * 2
                + retention_reversal["observation_count"]
            ),
            "optimizer_updates": (
                args.context_updates
                + args.credit_updates
                + external_credit_updates * len(new_families)
                + args.route_updates
                + 2 * len(new_families) * args.extension_updates
            ),
            "retention_observations": retention_reversal["observation_count"],
            "replayed_examples_after_extension": 0,
            "replayed_examples": 0,
            "privileged_task_labels_seen_by_model": 0,
            "privileged_correct_rows_seen_by_model": 0,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    report["gates"] = {
        "context_route_mastered": old_context_accuracy >= 0.8,
        "context_beats_pooled_baseline": old_context_accuracy >= pooled_accuracy + 0.2,
        "candidate_permutation_invariant": permutation_accuracy >= 0.8,
        "new_route_recovered": all(
            value >= 0.8 for value in new_selection.values()
        ),
        "new_route_causal": all(
            new_selection[key] >= new_route_without_extension[key] + 0.5
            for key in new_selection
        ),
        "old_route_retained": old_context_accuracy >= 0.8,
        "reward_shuffled_extension_not_selected": all(
            value <= 0.5 for value in shuffled_selection.values()
        ),
        "prior_extensions_attempted": all(
            all(
                rate >= 0.8
                for rate in attempts[: int(family_text) - 2]
            )
            for family_text, attempts in prior_attempt_rates.items()
        ),
        "credit_position_signal": credit_position_accuracy >= 0.66,
        "isolated_new_credit_signal": new_credit_position_accuracy >= 0.66,
        "no_replay_after_extension": True,
        "retention_reversal_safe": (
            retention_reversal["full_bank_refuses_eviction"]
            and retention_reversal["reversal_releases_only_target"]
            and retention_reversal["recovered_protected"]
        ),
    }
    report["promoted"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--context-updates", type=int, default=128)
    parser.add_argument("--credit-updates", type=int, default=128)
    parser.add_argument("--external-credit-updates", type=int, default=None)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--extension-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="generate a same-statistics pattern bank of this temporal length",
    )
    parser.add_argument(
        "--new-families",
        type=str,
        default="2,3",
        help="comma-separated contiguous family IDs acquired after the frozen base",
    )
    args = parser.parse_args()
    try:
        new_families = tuple(
            int(value.strip())
            for value in args.new_families.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise SystemExit("--new-families must be a comma-separated integer list") from error
    if not new_families:
        raise SystemExit("--new-families must contain at least one family")
    report = run(args.seed, args, new_families=new_families)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
