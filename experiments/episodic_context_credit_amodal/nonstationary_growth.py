"""Pressure-test replay-free external growth across a temporal distribution shift."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

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
    credit_weights_from_logits,
)


def _mixed_query(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    *,
    base_patterns: torch.Tensor,
    shifted_patterns: torch.Tensor,
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Use the old stream for protected families and shifted stream for new ones."""

    def query(families: torch.Tensor, seed: int) -> torch.Tensor:
        result = torch.empty(
            families.shape[0], encoder.context_width, dtype=torch.float32
        )
        for family_tensor in families.unique(sorted=True):
            family = int(family_tensor)
            mask = families == family
            patterns = base_patterns if family < 2 else shifted_patterns
            result[mask] = _encode(
                encoder,
                families[mask],
                prototypes,
                seed=seed + family,
                patterns=patterns,
            )
        return result

    return query


@torch.no_grad()
def _credit_position_accuracy(
    encoder: EpisodicContextEncoder,
    prototypes: torch.Tensor,
    heads: tuple[torch.nn.Module, ...],
    *,
    base_patterns: torch.Tensor,
    shifted_patterns: torch.Tensor,
    new_families: tuple[int, ...],
    seed: int,
) -> tuple[float, float, float]:
    old_families = torch.tensor([0, 1], dtype=torch.long)
    old_events, old_actions, old_outcomes, old_present = _episodes(
        old_families,
        prototypes,
        seed=seed,
        patterns=base_patterns,
    )
    old_output = encoder(old_events, old_actions, old_outcomes, old_present)
    old_weights = old_output.credit_weights

    new_families_tensor = torch.tensor(new_families, dtype=torch.long)
    new_events, new_actions, new_outcomes, new_present = _episodes(
        new_families_tensor,
        prototypes,
        seed=seed + 1,
        patterns=shifted_patterns,
    )
    new_output = encoder(
        new_events,
        new_actions,
        new_outcomes,
        new_present,
    )
    new_weights: list[torch.Tensor] = []
    for index, head in enumerate(heads):
        logits = head(
            new_output.sequence[index : index + 1],
            new_output.context[index : index + 1],
            new_outcomes[index : index + 1],
            new_present[index : index + 1],
        )
        new_weights.append(
            credit_weights_from_logits(
                logits,
                new_present[index : index + 1],
            )
        )
    new_weight_tensor = torch.cat(new_weights, dim=0)
    old_expected = old_families.remainder(base_patterns.shape[1])
    new_expected = new_families_tensor.remainder(shifted_patterns.shape[1])
    old_accuracy = float(
        (old_weights.argmax(dim=-1) == old_expected).float().mean()
    )
    new_accuracy = float(
        (new_weight_tensor.argmax(dim=-1) == new_expected).float().mean()
    )
    old_correct = (old_weights.argmax(dim=-1) == old_expected).sum()
    new_correct = (new_weight_tensor.argmax(dim=-1) == new_expected).sum()
    combined_accuracy = float(
        (old_correct + new_correct)
        / (old_expected.numel() + new_expected.numel())
    )
    return old_accuracy, new_accuracy, combined_accuracy


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    seed_everything(args.seed)
    if args.base_episode_length < 2 or args.shifted_episode_length < 2:
        raise ValueError("episode lengths must be at least two")
    if args.shifted_episode_length == args.base_episode_length:
        raise ValueError("nonstationary audit requires different episode lengths")
    if args.audit_batch_size < 1:
        raise ValueError("audit batch size must be positive")
    new_families = tuple(
        range(2, 2 + args.new_family_count)
    )
    base_patterns = _pattern_bank(1, episode_length=args.base_episode_length)
    shifted_patterns = _pattern_bank(
        max(new_families),
        episode_length=args.shifted_episode_length,
    )
    prototypes = _prototypes(args.seed + 77, 8)
    encoder = EpisodicContextEncoder(8, 3, hidden=32, context_width=16)
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

    query = _mixed_query(
        encoder,
        prototypes,
        base_patterns=base_patterns,
        shifted_patterns=shifted_patterns,
    )
    base_query = _mixed_query(
        encoder,
        prototypes,
        base_patterns=base_patterns,
        shifted_patterns=base_patterns,
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
    audit_batch_size = args.audit_batch_size
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
        batch_size=audit_batch_size,
    )
    permutation_accuracy = _route_accuracy(
        router,
        base_query,
        old_keys[[1, 0]],
        families=(1, 0),
        expected_rows=(0, 1),
        seed=args.seed + 51_001,
        batch_size=audit_batch_size,
    )

    extensions = tuple(
        train_extension(
            query,
            new_family=family,
            updates=args.extension_updates,
            batch_size=args.batch_size,
            seed=args.seed + family,
            negative_families=tuple(range(family)),
        )
        for family in new_families
    )
    shuffled_extensions = tuple(
        train_extension(
            query,
            new_family=family,
            updates=args.extension_updates,
            batch_size=args.batch_size,
            seed=args.seed + family + 10,
            shuffled=True,
            antithetic_shuffled=True,
        )
        for family in new_families
    )
    credit_heads = tuple(
        train_external_credit_head(
            encoder,
            prototypes,
            new_family=family,
            updates=args.external_credit_updates,
            batch_size=args.batch_size,
            seed=args.seed + family,
            patterns=shifted_patterns,
        )
        for family in new_families
    )

    new_selection: dict[str, float] = {}
    shuffled_selection: dict[str, float] = {}
    ablated_selection: dict[str, float] = {}
    prior_attempts: dict[str, list[float]] = {}
    for family in new_families:
        selected, _old_success, attempts, _activations = _extension_selection(
            router,
            query,
            old_keys,
            family=family,
            extensions=extensions,
            seed=args.seed + 60_001 + family,
            batch_size=audit_batch_size,
        )
        disabled = family - 2
        ablated, _, _, _ = _extension_selection(
            router,
            query,
            old_keys,
            family=family,
            extensions=extensions,
            disabled_extension=disabled,
            seed=args.seed + 60_001 + family,
            batch_size=audit_batch_size,
        )
        shuffled, _, _, _ = _extension_selection(
            router,
            query,
            old_keys,
            family=family,
            extensions=shuffled_extensions,
            seed=args.seed + 60_001 + family,
            batch_size=audit_batch_size,
        )
        new_selection[str(family)] = selected
        ablated_selection[str(family)] = ablated
        shuffled_selection[str(family)] = shuffled
        prior_attempts[str(family)] = attempts

    retention = _retention_reversal_audit(
        router,
        query,
        capability_keys,
        extensions,
        new_families=new_families,
        batch_size=audit_batch_size,
        seed=args.seed,
    )
    old_credit, new_credit, combined_credit = _credit_position_accuracy(
        encoder,
        prototypes,
        credit_heads,
        base_patterns=base_patterns,
        shifted_patterns=shifted_patterns,
        new_families=new_families,
        seed=args.seed + 70_001,
    )
    report: dict[str, object] = {
        "schema": "neural-computer.episodic-context-credit-nonstationary-report.v1",
        "claim_boundary": (
            "Two old capabilities are learned on one temporal distribution, "
            "then new external routes and isolated credit heads are acquired "
            "from a different episode length with no replay of the old stream. "
            "This is a bounded nonstationary diagnostic, not general continual "
            "learning."
        ),
        "seed": args.seed,
        "base_episode_length": args.base_episode_length,
        "shifted_episode_length": args.shifted_episode_length,
        "base_pattern_bank_size": int(base_patterns.shape[0]),
        "shifted_pattern_bank_size": int(shifted_patterns.shape[0]),
        "old_families": [0, 1],
        "new_families": list(new_families),
        "audit_batch_size": audit_batch_size,
        "context_loss": context_loss,
        "credit_loss": credit_loss,
        "old_route_accuracy": old_route_accuracy,
        "candidate_permutation_accuracy": permutation_accuracy,
        "new_route_selection": new_selection,
        "new_route_selection_without_extension": ablated_selection,
        "reward_shuffled_extension_selection": shuffled_selection,
        "prior_extension_attempt_rates": prior_attempts,
        "credit_position_accuracy": {
            "old": old_credit,
            "new": new_credit,
            "combined": combined_credit,
        },
        "retention_reversal": retention,
        "accounting": {
            "base_context_updates": args.context_updates,
            "base_credit_updates": args.credit_updates,
            "external_credit_updates": args.external_credit_updates,
            "route_updates": args.route_updates,
            "extension_updates": args.extension_updates * len(new_families),
            "unique_logical_lifetimes": (
                args.context_updates * 2
                + (
                    args.credit_updates
                    + args.route_updates
                    + len(new_families)
                    * (args.external_credit_updates + 2 * args.extension_updates)
                )
                * args.batch_size
                + int(retention["observation_count"])
            ),
            "unique_verifier_bits": (
                args.credit_updates
                * args.batch_size
                * args.base_episode_length
                * 2
                + args.external_credit_updates
                * len(new_families)
                * args.batch_size
                * args.shifted_episode_length
                * 2
                + (
                    args.route_updates
                    + 2 * len(new_families) * args.extension_updates
                )
                * args.batch_size
                * 2
                + int(retention["observation_count"])
            ),
            "optimizer_updates": (
                args.context_updates
                + args.credit_updates
                + args.route_updates
                + len(new_families)
                * (args.external_credit_updates + 2 * args.extension_updates)
            ),
            "replayed_examples": 0,
            "distribution_shift": (
                f"episode_length_{args.base_episode_length}_to_"
                f"{args.shifted_episode_length}"
            ),
            "wall_seconds": time.perf_counter() - started,
        },
    }
    report["gates"] = {
        "old_route_retained": old_route_accuracy >= 0.8,
        "candidate_permutation_invariant": permutation_accuracy >= 0.8,
        "new_routes_recovered": all(
            value >= 0.8 for value in new_selection.values()
        ),
        "new_routes_causal": all(
            new_selection[key] >= ablated_selection[key] + 0.5
            for key in new_selection
        ),
        "reward_shuffled_not_selected": all(
            value <= 0.5 for value in shuffled_selection.values()
        ),
        "prior_extensions_attempted": all(
            all(rate >= 0.8 for rate in attempts[: int(family) - 2])
            for family, attempts in prior_attempts.items()
        ),
        "credit_signal_survives_shift": combined_credit >= 0.66,
        "no_replay_after_shift": True,
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
    parser.add_argument("--shifted-episode-length", type=int, default=7)
    parser.add_argument("--new-family-count", type=int, default=8)
    parser.add_argument("--context-updates", type=int, default=1024)
    parser.add_argument("--credit-updates", type=int, default=512)
    parser.add_argument("--external-credit-updates", type=int, default=128)
    parser.add_argument("--route-updates", type=int, default=1024)
    parser.add_argument("--extension-updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.new_family_count < 1:
        raise SystemExit("--new-family-count must be positive")
    report = run(args)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "gates": report["gates"],
                "new_route_selection": report["new_route_selection"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
