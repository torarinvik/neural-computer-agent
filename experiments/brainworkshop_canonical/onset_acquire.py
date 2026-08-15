"""Unused-seed onset lease for the AND combinator.

Onset is the first public rule in this catalog that no single admitted
family solves: retrieve of the delay file and invert of that file both stay
below threshold, and only ``and(invert(delay), prototype)`` clears it. The
lease measures whether search selects that combination on seeds no earlier
onset, current-symbol, Dual, or founding run has consumed.

This never writes ``AgentBrain.bank``. An AND child is not admitted here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

from .bank_program import (
    bind_live_prototype,
    install_temporal_artifact,
    invert_artifact,
)
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import (
    BOUND_FRONTEND_SEEDS,
    DEVELOPMENT_SEEDS,
    DIAGNOSTIC_SEEDS,
    FRONTEND_SEED,
    LEASE_SESSIONS,
    MINIMUM_BITS,
    PREVIOUS_ACQUIRE_SEEDS,
    SEARCH_LEASE_SEEDS,
    STEPS,
    THRESHOLD,
    _encoders,
    _machine,
    _stable_hold_bits,
    _summary,
    curated_frontend,
    require_controller,
)
from .dual_promotion import KNOWN_USED_SEEDS
from .program_search import search_temporal_programs
from .rendered_environment import RenderedBrainWorkshopConfig
from .rendered_live import run_rendered_live_lifetime

EXPERIMENT_ID = "brainworkshop-onset-search-lease-2026-08-15"
ONSET_LEASE_SEEDS = (125_017, 126_017, 127_017)
# The 48-step lease was rejected because the single-family prototype-only
# control is base-rate bound near 0.75 and crossed 0.8 on one seed with 47
# eligible trials. The long lease changes episode length only, on a fresh
# seed block, to test whether that crossing was sampling noise.
LONG_LEASE_ID = "brainworkshop-onset-search-lease-long-2026-08-15"
LONG_LEASE_SEEDS = (128_017, 129_017, 130_017)
LONG_STEPS = 192
DUAL_HOLDOUT_SEEDS = frozenset({113_017, 114_017, 115_017})
TARGET_SYMBOL = 0


def onset_config(*, steps: int = STEPS) -> RenderedBrainWorkshopConfig:
    return RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        match_rule="onset",
        target_symbol=TARGET_SYMBOL,
    )


def assert_unused_onset_seeds(
    seeds: tuple[int, ...],
    *,
    sessions: int = LEASE_SESSIONS,
    additional_used: frozenset[int] = frozenset(),
) -> None:
    """Fail closed if any lifetime of this lease reuses an earlier lifetime."""

    used = (
        set(KNOWN_USED_SEEDS)
        | set(DEVELOPMENT_SEEDS)
        | set(DIAGNOSTIC_SEEDS)
        | set(PREVIOUS_ACQUIRE_SEEDS)
        | set(BOUND_FRONTEND_SEEDS)
        | set(SEARCH_LEASE_SEEDS)
        | set(DUAL_HOLDOUT_SEEDS)
        | set(additional_used)
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("onset lease seeds must be unique")
    if len(seeds) < 3:
        raise ValueError("onset lease needs at least three seeds")
    consumed: set[int] = set()
    for seed in seeds:
        span = set(range(seed, seed + sessions + 1))
        overlap = span & used
        if overlap:
            raise ValueError(
                f"onset lease lifetimes collide with used seeds: {sorted(overlap)}"
            )
        if span & consumed:
            raise ValueError("onset lease replicates overlap each other")
        consumed |= span
    # Earlier leases also consumed seed..seed+sessions around their block.
    for earlier in (
        *PREVIOUS_ACQUIRE_SEEDS,
        *BOUND_FRONTEND_SEEDS,
        *SEARCH_LEASE_SEEDS,
        *DUAL_HOLDOUT_SEEDS,
        *additional_used,
    ):
        span = set(range(earlier, earlier + sessions + 1))
        if span & consumed:
            raise ValueError(
                f"onset lease reuses lifetimes of an earlier lease around {earlier}"
            )


def run_onset_lease_replicate(
    controller_payload: dict[str, object],
    bank_path: Path,
    encoders,
    *,
    seed: int,
    steps: int = STEPS,
    sessions: int = LEASE_SESSIONS,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    """Search the closed grammar on onset, then hold the winner frozen."""

    machine = _machine(controller_payload, learn=False)
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    config = onset_config(steps=steps)
    started = time.perf_counter()

    def acquire(proposal):
        """Two-phase acquire: act as invert, learn the prototype from rewards."""

        del proposal
        machine.learning_enabled = True
        machine.sample = False
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=seed, learn=True, sample=False
        )
        machine.learning_enabled = False
        machine.sample = False
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
            "program_file_updates": report.program_file_updates,
        }

    def evaluate(proposal):
        del proposal
        report = run_rendered_live_lifetime(
            machine, encoders, config, seed=seed + 1, learn=False, sample=False
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    search = search_temporal_programs(
        bank,
        machine,
        evaluate,
        threshold=THRESHOLD,
        minimum_bits=MINIMUM_BITS,
        acquire=acquire,
        encoders=encoders,
    )
    winner = search["winner"]
    holds: list[dict[str, Any]] = []
    if winner is not None:
        holds.append(
            {
                "session": 0,
                "seed": seed + 1,
                "accuracy": winner["accuracy"],
                "unique_verifier_bits": winner["unique_verifier_bits"],
            }
        )
        for index in range(1, sessions):
            report = run_rendered_live_lifetime(
                machine,
                encoders,
                config,
                seed=seed + 1 + index,
                learn=False,
                sample=False,
            )
            holds.append(
                {
                    "session": index,
                    "seed": seed + 1 + index,
                    "accuracy": report.eligible_accuracy,
                    "unique_verifier_bits": report.unique_verifier_bits,
                    "program_file_updates": report.program_file_updates,
                }
            )
    hold_seed = seed + sessions

    learned = machine.prototype.detach().clone()
    with torch.no_grad():
        machine.prototype.data.zero_()
    zeros = run_rendered_live_lifetime(
        machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )
    with torch.no_grad():
        machine.prototype.data.copy_(learned)
    reversed_actions = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=hold_seed,
        learn=False,
        sample=False,
        action_permutation=(1, 0),
    )
    shuffled = run_rendered_live_lifetime(
        machine,
        encoders,
        config,
        seed=hold_seed,
        learn=False,
        sample=False,
        randomized_outcome_seed=seed,
    )
    other = _encoders(machine)
    crossed = run_rendered_live_lifetime(
        machine, other, config, seed=hold_seed, learn=False, sample=False
    )

    retrieve_machine = _machine(controller_payload, learn=False)
    install_temporal_artifact(retrieve_machine, bank, bank.artifact(0))
    retrieved = run_rendered_live_lifetime(
        retrieve_machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )
    invert_machine = _machine(controller_payload, learn=False)
    install_temporal_artifact(
        invert_machine, bank, invert_artifact(bank.artifact(0))
    )
    inverted = run_rendered_live_lifetime(
        invert_machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )
    prototype_machine = _machine(controller_payload, learn=False)
    prototype_only_artifact = bind_live_prototype(machine, encoders)
    install_temporal_artifact(
        prototype_machine, bank, prototype_only_artifact, encoders=encoders
    )
    prototype_only = run_rendered_live_lifetime(
        prototype_machine, encoders, config, seed=hold_seed, learn=False, sample=False
    )

    coverage = {
        "proposal_count": int(search["proposal_count"]),
        "executed": int(search["executed"]),
        "illegal_compose": int(search["illegal_compose"]),
        "not_installable": [
            {"label": row["label"], "reason": row.get("reason")}
            for row in search["attempts"]
            if not row["executed"] and row["kind"] != "illegal_compose"
        ],
    }
    stable = _stable_hold_bits(holds)
    frozen_holds = all(
        int(row.get("program_file_updates", 0)) == 0 for row in holds[1:]
    )
    accepted = bool(
        winner is not None
        and winner["kind"] == "and"
        and winner.get("frontend_digest") == encoders.digest()
        and stable is not None
        and frozen_holds
        and zeros.eligible_accuracy < THRESHOLD
        and reversed_actions.eligible_accuracy < THRESHOLD
        and shuffled.eligible_accuracy < THRESHOLD
        and crossed.eligible_accuracy < THRESHOLD
        and retrieved.eligible_accuracy < THRESHOLD
        and inverted.eligible_accuracy < THRESHOLD
        and prototype_only.eligible_accuracy < THRESHOLD
        and machine.controller_digest() == bank.controller_digest
    )
    return {
        "schema": "neural-computer.onset-search-lease-replicate.v1",
        "experiment_id": experiment_id,
        "seed": seed,
        "steps": steps,
        "sessions": sessions,
        "search": search,
        "grammar_coverage": coverage,
        "holds": holds,
        "stable_bits_to_threshold": stable,
        "frozen_holds": frozen_holds,
        "frontend_digest": encoders.digest(),
        "cross_frontend_digest": other.digest(),
        "controller_digest": machine.controller_digest(),
        "prototype_norm": float(learned.norm()),
        "zeros": _summary("zeros", zeros),
        "action_reversed": _summary("action_reversed", reversed_actions),
        "reward_shuffled": _summary("reward_shuffled", shuffled),
        "cross_encoder": _summary("cross_encoder", crossed),
        "retrieve_slot0": _summary("retrieve_slot0", retrieved),
        "invert_slot0": _summary("invert_slot0", inverted),
        "prototype_only": _summary("prototype_only", prototype_only),
        "elapsed_seconds": time.perf_counter() - started,
        "accepted": accepted,
    }


def run_onset_lease(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    seeds: tuple[int, ...] = ONSET_LEASE_SEEDS,
    steps: int = STEPS,
    sessions: int = LEASE_SESSIONS,
    frontend_path: Path | None = None,
    frontend_seed: int = FRONTEND_SEED,
    experiment_id: str = EXPERIMENT_ID,
    additional_used: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Run the unused onset population. Never writes the bank."""

    assert_unused_onset_seeds(
        seeds, sessions=sessions, additional_used=additional_used
    )
    controller_sha = require_controller(controller_path)
    before = sha256_file(bank_path)
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    slot0 = bank.artifact(0).digest()
    program_count = bank.program_count
    payload = load_temporal_controller_artifact(controller_path)
    probe = _machine(payload, learn=False)
    encoders = curated_frontend(probe, seed=frontend_seed, path=frontend_path)
    frontend_digest = encoders.digest()
    output_directory.mkdir(parents=True, exist_ok=True)
    replicates: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        report = run_onset_lease_replicate(
            payload,
            bank_path,
            encoders,
            seed=seed,
            steps=steps,
            sessions=sessions,
            experiment_id=experiment_id,
        )
        path = output_directory / f"seed{seed}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        replicates.append(report)
    after = sha256_file(bank_path)
    restored = ExternalTemporalProgramBank.load_bank(bank_path)
    if (
        after != before
        or restored.artifact(0).digest() != slot0
        or restored.program_count != program_count
    ):
        raise RuntimeError("onset lease mutated AgentBrain.bank")
    accepted = all(row["accepted"] for row in replicates)

    def _below(name: str) -> bool:
        return all(float(row[name]["accuracy"]) < THRESHOLD for row in replicates)

    campaign = {
        "schema": "neural-computer.onset-search-lease.v1",
        "experiment_id": experiment_id,
        "status": "replicated_not_admitted" if accepted else "rejected",
        "controller_sha256": controller_sha,
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "slot0_digest": slot0,
        "frontend_digest": frontend_digest,
        "frontend_shared": all(
            row["frontend_digest"] == frontend_digest for row in replicates
        ),
        "admitted": False,
        "selection": "search_and_of_invert_and_acquired_prototype",
        "winner_kinds": [
            (row["search"]["winner"] or {}).get("kind") for row in replicates
        ],
        "grammar_coverage": {
            f"seed{row['seed']}": row["grammar_coverage"] for row in replicates
        },
        "seeds": list(seeds),
        "steps": steps,
        "sessions": sessions,
        "threshold": THRESHOLD,
        "minimum_bits": MINIMUM_BITS,
        "replicates": replicates,
        "accepted": accepted,
        "elapsed_seconds": time.perf_counter() - started,
        "unique_verifier_bits": {
            f"hold_seed{row['seed']}": sum(
                int(item["unique_verifier_bits"]) for item in row["holds"]
            )
            for row in replicates
        },
        "stable_bits_to_threshold": {
            f"seed{row['seed']}": row["stable_bits_to_threshold"]
            for row in replicates
        },
        "optimizer_updates": 0,
        "replayed_examples": 0,
    }
    (output_directory / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n"
    )
    ledger = {
        "schema": "neural-computer.sample-efficiency-ledger.v1",
        "experiment_id": experiment_id,
        "status": campaign["status"],
        "source": "in_repository_run",
        "seed_blocks": [[seed, seed + sessions] for seed in seeds],
        "replication_count": len(seeds),
        "unique_verifier_bits": campaign["unique_verifier_bits"],
        "unique_logical_lifetimes_all_arms": len(seeds) * (sessions + 8),
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": {
            f"seed{row['seed']}": row["elapsed_seconds"] for row in replicates
        },
        "stable_bits_to_threshold": campaign["stable_bits_to_threshold"],
        "retention_on_mastered_primitive": (
            "onset AND child held frozen for the lease sessions with zero "
            "program file updates after acquire"
        ),
        "transfer_ratio_against_fresh_learner": None,
        "transfer_ratio_note": (
            "retrieve, invert, and prototype-only are single-family reject "
            "controls, not a fresh-learner climb"
        ),
        "controls": {
            "zeros_below_threshold": _below("zeros"),
            "action_reversed_below_threshold": _below("action_reversed"),
            "reward_shuffled_below_threshold": _below("reward_shuffled"),
            "cross_encoder_below_threshold": _below("cross_encoder"),
            "retrieve_slot0_below_threshold": _below("retrieve_slot0"),
            "invert_slot0_below_threshold": _below("invert_slot0"),
            "prototype_only_below_threshold": _below("prototype_only"),
            "frozen_holds": all(row["frozen_holds"] for row in replicates),
            "frontend_bound": all(
                (row["search"]["winner"] or {}).get("frontend_digest")
                == frontend_digest
                for row in replicates
            ),
            "controller_frozen": all(
                row["controller_digest"] == bank.controller_digest
                for row in replicates
            ),
            "bank_unchanged": after == before,
            "admitted": False,
        },
        "unresolved": {
            "bank_admission": "the AND child is not a curated AgentBrain slot",
            "open_program_induction": (
                "grammar is still retrieve, compose, invert, and, invent"
            ),
            "learned_proposer": "search still enumerates the closed grammar",
            "depth_two_coverage": (
                "the prototype-capable machine cannot install recursive "
                "depth-2 files, so retrieve slot 2, compose, and invert "
                "slot 2 were proposed but not executed"
            ),
            "dual_2back_transfer": "not this experiment",
        },
    }
    (output_directory / "sample_efficiency_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    )
    checksums = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output_directory.glob("*.json"))
    ]
    (output_directory / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_onset_search_lease_2026-08-15"
        ),
    )
    parser.add_argument(
        "--frontend",
        type=Path,
        default=(
            repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
        ),
    )
    parser.add_argument("--frontend-seed", type=int, default=FRONTEND_SEED)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sessions", type=int, default=LEASE_SESSIONS)
    parser.add_argument(
        "--long",
        action="store_true",
        help=(
            "longer-episode arm on a fresh seed block; changes episode length "
            "only, so the base-rate-bound single-family control is measured on "
            "more eligible trials"
        ),
    )
    arguments = parser.parse_args()
    default_output = (
        repository / "session_records" / "brainworkshop_onset_search_lease_2026-08-15"
    )
    seeds = LONG_LEASE_SEEDS if arguments.long else ONSET_LEASE_SEEDS
    experiment_id = LONG_LEASE_ID if arguments.long else EXPERIMENT_ID
    additional_used = frozenset(ONSET_LEASE_SEEDS) if arguments.long else frozenset()
    steps = arguments.steps
    if steps is None:
        steps = LONG_STEPS if arguments.long else STEPS
    if arguments.long and arguments.output_dir == default_output:
        arguments.output_dir = (
            repository
            / "session_records"
            / "brainworkshop_onset_search_lease_long_2026-08-15"
        )
    campaign = run_onset_lease(
        arguments.controller_artifact,
        arguments.bank,
        arguments.output_dir,
        seeds=seeds,
        steps=steps,
        sessions=arguments.sessions,
        frontend_path=arguments.frontend,
        frontend_seed=arguments.frontend_seed,
        experiment_id=experiment_id,
        additional_used=additional_used,
    )
    print(
        json.dumps(
            {
                "accepted": campaign["accepted"],
                "status": campaign["status"],
                "admitted": campaign["admitted"],
                "bank_unchanged": campaign["bank_unchanged"],
                "winner_kinds": campaign["winner_kinds"],
                "seeds": campaign["seeds"],
                "elapsed_seconds": campaign["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
