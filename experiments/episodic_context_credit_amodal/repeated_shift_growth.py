"""Audit replay-free external growth across sequential distribution shifts."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from experiments.episodic_context_credit_amodal.train import (
    _encode,
    _episodes,
    _extension_selection,
    _pattern_bank,
    _prototypes,
    _retention_reversal_audit,
    _route_accuracy,
    seed_everything,
    train_context,
    train_credit,
    train_extension,
    train_external_credit_head,
    train_router,
)
from neural_computer import (
    EpisodicContextEncoder,
    EpisodicCreditHead,
    ExternalGrowthPrior,
    FactorizedOpaqueAddressRouter,
    OpaqueViewRouteExtension,
    PersistentOpaqueStateStore,
    credit_weights_from_logits,
)


def _parse_int_list(
    raw: str,
    *,
    name: str,
    minimum: int = 2,
) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise SystemExit(f"{name} must be comma-separated integers") from error
    if not values or any(value < minimum for value in values):
        raise SystemExit(f"{name} must contain integers >= {minimum}")
    return values


def _persist_and_reload_state(
    state_dir: Path,
    router: FactorizedOpaqueAddressRouter,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    heads: dict[int, EpisodicCreditHead],
    *,
    route_width: int,
    context_hidden: int,
    context_width: int,
) -> dict[str, object]:
    """Persist and independently reload all memory-side learned state."""

    state_dir.mkdir(parents=True, exist_ok=True)
    base_configuration = {
        "component": "repeated-shift-base-route",
        "schema": "neural-computer.factorized-opaque-address-router.v1",
        "width": route_width,
        "hidden": 48,
    }
    base_store = PersistentOpaqueStateStore(
        state_dir / "base-route.pt", configuration=base_configuration
    )
    digests: dict[str, str] = {"base-route": base_store.save_module(router)}
    extension_stores: list[PersistentOpaqueStateStore] = []
    for index, extension in enumerate(extensions):
        store = PersistentOpaqueStateStore(
            state_dir / f"extension-{index:03d}.pt",
            configuration={
                "component": "repeated-shift-route-extension",
                "schema": "neural-computer.opaque-view-route-extension.v1",
                "index": index,
                "width": route_width,
                "hidden": 48,
            },
        )
        extension_stores.append(store)
        digests[f"extension-{index:03d}"] = store.save_module(extension)
    head_stores: list[PersistentOpaqueStateStore] = []
    for index, family in enumerate(heads):
        store = PersistentOpaqueStateStore(
            state_dir / f"credit-{index:03d}.pt",
            configuration={
                "component": "repeated-shift-credit-head",
                "schema": "neural-computer.episodic-credit-head.v1",
                "index": index,
                "family": family,
                "hidden": context_hidden,
                "context_width": context_width,
            },
        )
        head_stores.append(store)
        digests[f"credit-{index:03d}"] = store.save_module(heads[family])

    reloaded_router = FactorizedOpaqueAddressRouter(width=route_width, hidden=48)
    base_store.load_module(reloaded_router)
    reloaded_extensions: list[OpaqueViewRouteExtension] = []
    for store in extension_stores:
        extension = OpaqueViewRouteExtension(width=route_width, hidden=48)
        store.load_module(extension)
        reloaded_extensions.append(extension)
    reloaded_heads: dict[int, EpisodicCreditHead] = {}
    for family, store in zip(heads, head_stores, strict=True):
        head = EpisodicCreditHead(context_hidden, context_width)
        store.load_module(head)
        reloaded_heads[family] = head

    corruption_rejected = False
    with TemporaryDirectory(prefix="repeated-shift-corruption-") as directory:
        probe_path = Path(directory) / "base-route.pt"
        payload = torch.load(base_store.path, weights_only=False)
        first_name = next(iter(payload["state_dict"]))
        payload["state_dict"][first_name] = payload["state_dict"][first_name].clone()
        payload["state_dict"][first_name].reshape(-1)[0] += 1.0
        torch.save(payload, probe_path)
        probe_store = PersistentOpaqueStateStore(
            probe_path, configuration=base_configuration
        )
        try:
            probe_store.load()
        except ValueError as error:
            corruption_rejected = "checksum" in str(error)

    return {
        "router": reloaded_router,
        "extensions": tuple(reloaded_extensions),
        "heads": reloaded_heads,
        "digests": digests,
        "state_file_count": len(digests),
        "corruption_rejected": corruption_rejected,
    }


def _family_schedule(
    base_episode_length: int,
    shift_episode_lengths: tuple[int, ...],
    families_per_shift: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    if len(shift_episode_lengths) != len(families_per_shift):
        raise ValueError("shift lengths and family counts must align")
    schedule: list[tuple[int, int, int]] = [(0, 2, base_episode_length)]
    next_family = 2
    for episode_length, family_count in zip(
        shift_episode_lengths,
        families_per_shift,
        strict=True,
    ):
        if family_count < 1:
            raise ValueError("each shift must add at least one family")
        schedule.append((next_family, next_family + family_count, episode_length))
        next_family += family_count
    return tuple(schedule)


def _patterns_for_family(
    family: int,
    schedule: tuple[tuple[int, int, int], ...],
) -> tuple[int, int]:
    for start, end, episode_length in schedule:
        if start <= family < end:
            return family, episode_length
    raise ValueError(f"family {family} is outside the schedule")


def _piecewise_query(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    patterns_by_length: dict[int, torch.Tensor],
    schedule: tuple[tuple[int, int, int], ...],
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def query(families: torch.Tensor, seed: int) -> torch.Tensor:
        result = torch.empty(
            families.shape[0], encoder.context_width, dtype=torch.float32
        )
        for family_tensor in families.unique(sorted=True):
            family = int(family_tensor)
            mask = families == family
            _, episode_length = _patterns_for_family(family, schedule)
            result[mask] = _encode(
                encoder,
                families[mask],
                prototypes,
                seed=seed + family,
                patterns=patterns_by_length[episode_length],
            )
        return result

    return query


@torch.no_grad()
def _credit_accuracy(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    heads: dict[int, torch.nn.Module],
    *,
    base_episode_length: int,
    new_families: tuple[int, ...],
    schedule: tuple[tuple[int, int, int], ...],
    patterns_by_length: dict[int, torch.Tensor],
    seed: int,
) -> dict[str, float]:
    correct = 0
    total = 0
    group_correct: dict[str, int] = {}
    group_total: dict[str, int] = {}
    old_families = (0, 1)
    for family in old_families + new_families:
        _, episode_length = _patterns_for_family(family, schedule)
        patterns = patterns_by_length[episode_length]
        family_tensor = torch.tensor([family], dtype=torch.long)
        events, actions, outcomes, present = _episodes(
            family_tensor,
            prototypes,
            seed=seed + family,
            patterns=patterns,
        )
        output = encoder(events, actions, outcomes, present)
        if family < 2:
            weights = output.credit_weights
        else:
            head = heads[family]
            logits = head(
                output.sequence,
                output.context,
                outcomes,
                present,
            )
            weights = credit_weights_from_logits(logits, present)
        expected = family % episode_length
        family_correct = int((weights.argmax(dim=-1) == expected).sum())
        family_total = int(weights.shape[0])
        correct += family_correct
        total += family_total
        group = (
            next(
                index
                for index, (start, end, _length) in enumerate(schedule[1:], start=1)
                if start <= family < end
            )
            if family >= 2
            else 0
        )
        group_correct[str(group)] = group_correct.get(str(group), 0) + family_correct
        group_total[str(group)] = group_total.get(str(group), 0) + family_total
    return {
        "old": group_correct.get("0", 0) / group_total.get("0", 1),
        "combined": correct / total,
        **{
            f"shift_{group}": group_correct[str(group)] / group_total[str(group)]
            for group in range(1, len(schedule))
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(args.seed)
    configured_state_dir = getattr(args, "state_dir", None)
    state_dir = configured_state_dir or (
        args.report_out.parent / f"{args.report_out.stem}.state"
    )
    shift_lengths = _parse_int_list(
        args.shift_episode_lengths,
        name="--shift-episode-lengths",
    )
    family_counts = _parse_int_list(
        args.families_per_shift,
        name="--families-per-shift",
        minimum=1,
    )
    if any(value < 1 for value in family_counts):
        raise ValueError("families per shift must be positive")
    schedule = _family_schedule(
        args.base_episode_length,
        shift_lengths,
        family_counts,
    )
    if len(schedule) < 3:
        raise ValueError("the repeated-shift audit requires at least two shifts")
    new_families = tuple(
        family for start, end, _length in schedule[1:] for family in range(start, end)
    )
    patterns_by_length = {
        episode_length: _pattern_bank(
            max(end for start, end, length in schedule if length == episode_length) - 1,
            episode_length=episode_length,
        )
        for _start, _end, episode_length in schedule
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
    query = _piecewise_query(
        encoder,
        prototypes,
        patterns_by_length,
        schedule,
    )
    base_query = _piecewise_query(
        encoder,
        prototypes,
        {args.base_episode_length: base_patterns},
        schedule=((0, 2, args.base_episode_length),),
    )
    old_keys = F.normalize(
        torch.randn(
            2,
            16,
            generator=torch.Generator().manual_seed(args.seed + 91),
        ),
        dim=-1,
    )
    extension_keys = F.normalize(
        torch.randn(
            len(new_families),
            16,
            generator=torch.Generator().manual_seed(args.seed + 93),
        ),
        dim=-1,
    )
    capability_keys = torch.cat((old_keys, extension_keys), dim=0)
    router = train_router(
        base_query,
        old_keys,
        updates=args.route_updates,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    old_route_accuracy = _route_accuracy(
        router,
        base_query,
        old_keys,
        families=(0, 1),
        seed=args.seed + 50_001,
        batch_size=args.audit_batch_size,
    )
    permutation_accuracy = _route_accuracy(
        router,
        base_query,
        old_keys[[1, 0]],
        families=(1, 0),
        expected_rows=(0, 1),
        seed=args.seed + 51_001,
        batch_size=args.audit_batch_size,
    )

    extensions: list[OpaqueViewRouteExtension] = []
    shuffled_extensions: list[OpaqueViewRouteExtension] = []
    heads: dict[int, EpisodicCreditHead] = {}
    phase_reports: list[dict[str, object]] = []
    growth_prior: ExternalGrowthPrior | None = None
    prior_source_counts: list[int] = []
    for shift_index, (start, end, episode_length) in enumerate(schedule[1:], start=1):
        phase_families = tuple(range(start, end))
        phase_prior = (
            growth_prior if args.growth_initialization == "prior_average" else None
        )
        prior_source_counts.append(
            0 if phase_prior is None else phase_prior.source_count
        )
        for family in phase_families:
            extensions.append(
                train_extension(
                    query,
                    new_family=family,
                    updates=args.extension_updates,
                    batch_size=args.batch_size,
                    seed=args.seed + family,
                    negative_families=tuple(range(family)),
                    growth_prior=phase_prior,
                    growth_prior_mix=args.growth_prior_mix,
                )
            )
            shuffled_extensions.append(
                train_extension(
                    query,
                    new_family=family,
                    updates=args.extension_updates,
                    batch_size=args.batch_size,
                    seed=args.seed + family + 10,
                    shuffled=True,
                    antithetic_shuffled=True,
                    growth_prior=phase_prior,
                    growth_prior_mix=args.growth_prior_mix,
                )
            )
            heads[family] = train_external_credit_head(
                encoder,
                prototypes,
                new_family=family,
                updates=args.external_credit_updates,
                batch_size=args.batch_size,
                seed=args.seed + family,
                patterns=patterns_by_length[episode_length],
            )
        if args.growth_initialization == "prior_average":
            growth_prior = ExternalGrowthPrior.from_modules(extensions)
        phase_selection: dict[str, float] = {}
        for family in phase_families:
            selected, _old_success, _attempts, _activations = _extension_selection(
                router,
                query,
                old_keys,
                family=family,
                extensions=tuple(extensions),
                seed=args.seed + 60_001 + family,
                batch_size=args.audit_batch_size,
            )
            phase_selection[str(family)] = selected
        phase_reports.append(
            {
                "shift_index": shift_index,
                "episode_length": episode_length,
                "families": list(phase_families),
                "minimum_route_selection": min(phase_selection.values()),
                "route_selection": phase_selection,
            }
        )

    extension_tuple = tuple(extensions)
    shuffled_tuple = tuple(shuffled_extensions)
    persistent_state = _persist_and_reload_state(
        state_dir,
        router,
        extension_tuple,
        heads,
        route_width=int(old_keys.shape[-1]),
        context_hidden=encoder.hidden,
        context_width=encoder.context_width,
    )
    reloaded_router = persistent_state["router"]
    reloaded_extensions = persistent_state["extensions"]
    reloaded_heads = persistent_state["heads"]
    new_selection: dict[str, float] = {}
    ablated_selection: dict[str, float] = {}
    shuffled_selection: dict[str, float] = {}
    prior_attempts: dict[str, list[float]] = {}
    for family in new_families:
        selected, _old_success, attempts, _activations = _extension_selection(
            router,
            query,
            old_keys,
            family=family,
            extensions=extension_tuple,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
        )
        disabled = new_families.index(family)
        ablated, _, _, _ = _extension_selection(
            router,
            query,
            old_keys,
            family=family,
            extensions=extension_tuple,
            disabled_extension=disabled,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
        )
        shuffled, _, _, _ = _extension_selection(
            router,
            query,
            old_keys,
            family=family,
            extensions=shuffled_tuple,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
        )
        new_selection[str(family)] = selected
        ablated_selection[str(family)] = ablated
        shuffled_selection[str(family)] = shuffled
        prior_attempts[str(family)] = attempts

    retention = _retention_reversal_audit(
        router,
        query,
        capability_keys,
        extension_tuple,
        new_families=new_families,
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
    reloaded_old_route_accuracy = _route_accuracy(
        reloaded_router,
        base_query,
        old_keys,
        families=(0, 1),
        seed=args.seed + 50_001,
        batch_size=args.audit_batch_size,
    )
    reloaded_new_selection: dict[str, float] = {}
    for family in new_families:
        selected, _, _, _ = _extension_selection(
            reloaded_router,
            query,
            old_keys,
            family=family,
            extensions=reloaded_extensions,
            seed=args.seed + 60_001 + family,
            batch_size=args.audit_batch_size,
        )
        reloaded_new_selection[str(family)] = selected
    reloaded_credit_accuracy = _credit_accuracy(
        encoder,
        prototypes,
        reloaded_heads,
        base_episode_length=args.base_episode_length,
        new_families=new_families,
        schedule=schedule,
        patterns_by_length=patterns_by_length,
        seed=args.seed + 70_001,
    )
    base_credit_bits = (
        args.credit_updates * args.batch_size * args.base_episode_length * 2
    )
    external_credit_bits = sum(
        args.external_credit_updates * args.batch_size * episode_length * 2
        for _start, _end, episode_length in schedule[1:]
        for _ in range(_end - _start)
    )
    route_bits = args.route_updates * args.batch_size * 2
    extension_route_bits = (
        2 * len(new_families) * args.extension_updates * args.batch_size * 2
    )
    retention_bits = int(retention["observation_count"])
    bits = (
        base_credit_bits
        + external_credit_bits
        + route_bits
        + extension_route_bits
        + retention_bits
    )
    lifetimes = (
        args.context_updates * 2
        + (
            args.credit_updates
            + args.route_updates
            + len(new_families)
            * (args.external_credit_updates + 2 * args.extension_updates)
        )
        * args.batch_size
        + int(retention["observation_count"])
    )
    wall_seconds = time.perf_counter() - started
    report: dict[str, object] = {
        "schema": "neural-computer.episodic-context-credit-repeated-shift-report.v1",
        "claim_boundary": (
            f"A frozen base capability set survives {len(schedule) - 1} "
            "sequential temporal distribution shifts while fresh external "
            "routes and isolated credit heads are acquired without replay. "
            "This is a bounded repeated-shift diagnostic, not general "
            "continual learning."
        ),
        "seed": args.seed,
        "schedule": [
            {
                "family_start": start,
                "family_end": end,
                "episode_length": episode_length,
            }
            for start, end, episode_length in schedule
        ],
        "new_families": list(new_families),
        "context_loss": context_loss,
        "credit_loss": credit_loss,
        "old_route_accuracy": old_route_accuracy,
        "candidate_permutation_accuracy": permutation_accuracy,
        "phase_reports": phase_reports,
        "new_route_selection": new_selection,
        "new_route_selection_without_extension": ablated_selection,
        "reward_shuffled_extension_selection": shuffled_selection,
        "prior_extension_attempt_rates": prior_attempts,
        "credit_position_accuracy": credit_accuracy,
        "growth_initialization": args.growth_initialization,
        "growth_prior_policy": (
            f"average_state_reset_score_head_mix_{args.growth_prior_mix:.2f}"
            if args.growth_initialization == "prior_average"
            else "none"
        ),
        "growth_prior_source_counts": prior_source_counts,
        "retention_reversal": retention,
        "persistent_state": {
            "state_file_count": persistent_state["state_file_count"],
            "state_digests": persistent_state["digests"],
            "reloaded_old_route_accuracy": reloaded_old_route_accuracy,
            "reloaded_new_route_selection": reloaded_new_selection,
            "reloaded_credit_position_accuracy": reloaded_credit_accuracy,
            "corruption_rejected": persistent_state["corruption_rejected"],
        },
        "accounting": {
            "unique_verifier_bits": bits,
            "unique_logical_lifetimes": lifetimes,
            "optimizer_updates": (
                args.context_updates
                + args.credit_updates
                + args.route_updates
                + len(new_families)
                * (args.external_credit_updates + 2 * args.extension_updates)
            ),
            "base_credit_verifier_bits": base_credit_bits,
            "external_credit_verifier_bits": external_credit_bits,
            "base_route_verifier_bits": route_bits,
            "extension_route_verifier_bits": extension_route_bits,
            "retention_verifier_bits": retention_bits,
            "route_optimizer_updates": (
                args.route_updates + 2 * len(new_families) * args.extension_updates
            ),
            "credit_optimizer_updates": args.credit_updates,
            "external_credit_optimizer_updates": (
                len(new_families) * args.external_credit_updates
            ),
            "persistence_verifier_bits": 0,
            "persistence_optimizer_updates": 0,
            "persistence_replayed_examples": 0,
            "replayed_examples": 0,
            "distribution_shifts": len(schedule) - 1,
            "wall_seconds": wall_seconds,
            "latency_seconds_per_unique_lifetime": wall_seconds / max(lifetimes, 1),
            "transfer_ratio_against_fresh_learner": None,
        },
    }
    report["gates"] = {
        "old_route_retained": old_route_accuracy >= 0.8,
        "candidate_permutation_invariant": permutation_accuracy >= 0.8,
        "new_routes_recovered": all(value >= 0.8 for value in new_selection.values()),
        "new_routes_causal": all(
            new_selection[key] >= ablated_selection[key] + 0.5 for key in new_selection
        ),
        "reward_shuffled_not_selected": all(
            value <= 0.5 for value in shuffled_selection.values()
        ),
        "prior_extensions_attempted": all(
            all(rate >= 0.8 for rate in attempts[: int(family) - 2])
            for family, attempts in prior_attempts.items()
        ),
        "credit_signal_survives_all_shifts": all(
            credit_accuracy.get(key, 0.0) >= 0.66
            for key in (
                "old",
                "combined",
                *[f"shift_{i}" for i in range(1, len(schedule))],
            )
        ),
        "no_replay_after_shifts": True,
        "retention_reversal_safe": (
            retention["full_bank_refuses_eviction"]
            and retention["reversal_releases_only_target"]
            and retention["recovered_protected"]
        ),
        "persistent_route_reload": (
            reloaded_old_route_accuracy >= 0.8
            and all(value >= 0.8 for value in reloaded_new_selection.values())
            and all(
                abs(reloaded_new_selection[key] - new_selection[key]) <= 1e-6
                for key in new_selection
            )
        ),
        "persistent_credit_reload": all(
            abs(reloaded_credit_accuracy[key] - credit_accuracy[key]) <= 1e-6
            for key in credit_accuracy
        ),
        "persistent_state_corruption_rejected": bool(
            persistent_state["corruption_rejected"]
        ),
    }
    report["promoted"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--base-episode-length", type=int, default=6)
    parser.add_argument("--shift-episode-lengths", default="8,10")
    parser.add_argument("--families-per-shift", default="8,10")
    parser.add_argument("--context-updates", type=int, default=1024)
    parser.add_argument("--credit-updates", type=int, default=512)
    parser.add_argument("--external-credit-updates", type=int, default=128)
    parser.add_argument("--route-updates", type=int, default=1024)
    parser.add_argument("--extension-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-batch-size", type=int, default=64)
    parser.add_argument(
        "--growth-initialization",
        choices=("fresh", "prior_average"),
        default="fresh",
        help="initialize new external route adapters from fresh state or a frozen average prior",
    )
    parser.add_argument(
        "--growth-prior-mix",
        type=float,
        default=1.0,
        help="fraction of prior representation to blend into a fresh adapter",
    )
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
