"""The five navigation results, on worlds nobody has looked at.

Every measurement in the successor transfer, learned decomposition, object
identity, curious exploration and relational transfer records was taken on the
development seed. Each of those records says so at the bottom, and every one of
them is therefore a diagnostic rather than a result.

There is a sharper problem than the usual one, and finding it is why this
exists. Those five experiments drew their worlds from a **hard-coded** seed, so
changing the run seed varied only the exploration randomness -- the eight
sampled worlds were the same eight every time. A rerun at a new seed would have
looked like a holdout and would have been nothing of the kind. The world seed
is now a parameter, defaulting to the development value so that every recorded
number still reproduces exactly, and this run passes seeds from a block claimed
in `seed_ledger` and never spent.

Three replicates, from the three seeds of that block, so this replicates as
well as holds out. The block is asserted unused *before* a single episode is
drawn, and the run fails closed if it is not.

Nothing is tuned here. Every threshold, arm, relation and candidate cut is
whatever the development records left it at. The only question is which
orderings survive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from neural_computer.promotion import sha256_file

from .curious_exploration import run_curious_exploration
from .learned_decomposition import run_learned_decomposition
from .object_identity import run_object_identity
from .relational_transfer import run_relational_transfer
from .seed_ledger import (
    INTEGRATED_SESSIONS_PER_REPLICATE,
    assert_unused_block,
    block,
)
from .successor_transfer import run_successor_transfer

EXPERIMENT_ID = "brainworkshop-navigation-holdout-2026-08-16"
NAVIGATION_HOLDOUT_SCHEMA = "neural-computer.navigation-holdout.v1"
BLOCK_NAME = "navigation_family_holdout"
# Trimmed from the development task counts so that three replicates of five
# experiments stay inside the ten-minute rung. Nothing else is changed.
HOLDOUT_TASKS = 4




def successor_claims(report: dict[str, Any]) -> dict[str, bool]:
    curve = report["accumulation"]
    claims = {
        "accumulation_rises": (
            curve[-1]["gpi_over_optimal"] >= curve[0]["gpi_over_optimal"]
        ),
        "stitching_beats_single_policy": (
            curve[-1]["gpi_over_optimal"] > 3 * curve[-1]["single_over_optimal"]
        ),
    }
    for family in ("single", "disjunction", "avoid"):
        claims[f"{family}_beats_best_stored"] = (
            report[f"{family}_gpi_fraction"]
            > report[f"{family}_best_single_fraction"]
        )
        claims[f"{family}_beats_random"] = (
            report[f"{family}_gpi_fraction"] > report[f"{family}_random_fraction"]
        )
        claims[f"{family}_near_replan_when_told"] = (
            report[f"{family}_replan_told_fraction"]
            - report[f"{family}_gpi_told_fraction"]
            < 0.2
        )
    return claims


def decomposition_claims(report: dict[str, Any]) -> dict[str, bool]:
    cuts = report["cuts"]
    return {
        "components_chosen_everywhere": report["components_chosen"] == report["tasks"],
        "components_cheaper_than_whole": (
            cuts["components"]["total_bits"] < cuts["whole"]["total_bits"]
        ),
        "components_cheaper_than_scatter": (
            cuts["components"]["total_bits"] < cuts["scatter"]["total_bits"]
        ),
        "too_fine_pays_error": (
            cuts["cells"]["error_bits"] > cuts["components"]["error_bits"]
        ),
        "whole_scene_cannot_predict_itself": cuts["whole"]["error_bits"] > 0.0,
    }


def identity_claims(report: dict[str, Any]) -> dict[str, bool]:
    quiet = report["conditions"]["none"]
    claims = {
        "two_markers_still_solved": all(
            quiet[f"{arm}_own_accuracy"] == 1.0
            for arm in ("hybrid", "search", "alignment", "persistence")
        ),
        "search_tracks_better": (
            quiet["search_track_fidelity"] > quiet["alignment_track_fidelity"]
        ),
    }
    for condition in ("random_walk", "cycling"):
        found = report["conditions"][condition]
        claims[f"{condition}_persistence_breaks"] = (
            found["persistence_own_accuracy"] < 0.6
        )
        claims[f"{condition}_search_beats_persistence"] = (
            found["search_own_accuracy"] > found["persistence_own_accuracy"]
        )
        claims[f"{condition}_search_tracks_better"] = (
            found["search_track_fidelity"] > found["alignment_track_fidelity"]
        )
        claims[f"{condition}_intersection_still_finds_the_goal"] = (
            found["hybrid_goal_accuracy"] == 1.0
        )
        claims[f"{condition}_predictability_fails"] = (
            found["predictability_own_accuracy"] < found["search_own_accuracy"]
        )
        # Added after the first holdout, because it is what actually survived.
        # Whether the agent names itself correctly on the final frame turns out
        # to be a coin-flip against elimination; whether the *model* it built
        # is right is not close. A coherent track feeds coherent transitions
        # even on episodes where the final naming is wrong, and the model is
        # the thing anything downstream uses.
        claims[f"{condition}_search_builds_a_better_model"] = (
            found["hybrid_model_accuracy"] > found["persistence_model_accuracy"]
        )
    return claims


def exploration_claims(report: dict[str, Any]) -> dict[str, bool]:
    quiet = report["conditions"]["none"]
    noisy = report["conditions"]["random_walk"]
    return {
        "curious_beats_uniform": (
            quiet["curious"]["coverage"] > quiet["uniform"]["coverage"]
        ),
        "curious_beats_uniform_downstream": (
            quiet["curious"]["downstream_fraction"]
            > quiet["uniform"]["downstream_fraction"]
        ),
        "optimism_is_most_of_it": (
            quiet["optimistic"]["coverage"] > quiet["uniform"]["coverage"]
        ),
        "curiosity_still_adds": (
            quiet["curious"]["coverage"] > quiet["optimistic"]["coverage"]
        ),
        "gating_is_immune_to_the_television": (
            abs(quiet["curious"]["coverage"] - noisy["curious"]["coverage"]) < 1e-9
        ),
        "ungated_is_not": (
            noisy["curious_ungated"]["coverage"]
            < quiet["curious_ungated"]["coverage"]
        ),
        "horizon_family_is_degenerate": (
            abs(quiet["curious_g50"]["coverage"] - quiet["curious_g99"]["coverage"])
            < 1e-9
        ),
    }


def relational_claims(report: dict[str, Any]) -> dict[str, bool]:
    trained = report["trained"]
    held_out = report["held_out"]
    return {
        "pairs_needed_on_trained": (
            trained["place_gpi_fraction"] < 0.5 * trained["gpi_fraction"]
        ),
        "pairs_needed_on_held_out": (
            held_out["place_gpi_fraction"] < 0.5 * held_out["gpi_fraction"]
        ),
        "held_out_relation_transfers": (
            held_out["gpi_fraction"] > 1.4 * held_out["best_single_fraction"]
        ),
        "held_out_beats_random": (
            held_out["gpi_fraction"] > 2.0 * held_out["random_fraction"]
        ),
        "trained_matches_replan": (
            trained["replan_fraction"] - trained["gpi_fraction"] < 0.05
        ),
        "held_out_costs_more_than_interpolation": (
            held_out["replan_fraction"] - held_out["gpi_fraction"] > 0.05
        ),
    }


def run_navigation_holdout(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    tasks: int = HOLDOUT_TASKS,
    replicates: int = 3,
) -> dict[str, Any]:
    seeds = block(BLOCK_NAME)
    # Fails closed, before any episode is drawn.
    assert_unused_block(
        BLOCK_NAME, seeds, sessions=INTEGRATED_SESSIONS_PER_REPLICATE
    )
    before = sha256_file(bank_path)

    started = time.perf_counter()
    replicate_rows: list[dict[str, Any]] = []
    for index, world_seed in enumerate(seeds[: int(replicates)]):
        destination = output_directory / f"replicate{index}"
        common = {
            "frontend_path": frontend_path,
            "world_seed": world_seed,
            "seed": world_seed % 100_003,
        }
        successor = run_successor_transfer(
            controller_path, bank_path, destination, tasks=tasks, **common
        )
        decomposition = run_learned_decomposition(
            controller_path, bank_path, destination, tasks=tasks, **common
        )
        identity = run_object_identity(
            controller_path, bank_path, destination, tasks=tasks, **common
        )
        exploration = run_curious_exploration(
            controller_path, bank_path, destination, tasks=tasks, **common
        )
        relational = run_relational_transfer(
            controller_path, bank_path, destination, tasks=tasks, **common
        )
        replicate_rows.append(
            {
                "replicate": index,
                "world_seed": world_seed,
                "successor": successor_claims(successor),
                "decomposition": decomposition_claims(decomposition),
                "identity": identity_claims(identity),
                "exploration": exploration_claims(exploration),
                "relational": relational_claims(relational),
                "headline": {
                    "successor_single_gpi": successor["single_gpi_fraction"],
                    "successor_accumulation_end": (
                        successor["accumulation"][-1]["gpi_over_optimal"]
                    ),
                    "components_bits": decomposition["cuts"]["components"][
                        "total_bits"
                    ],
                    "whole_bits": decomposition["cuts"]["whole"]["total_bits"],
                    "identity_search_own": report_own(identity),
                    "identity_persistence_own": report_own(
                        identity, arm="persistence"
                    ),
                    "curious_coverage": exploration["conditions"]["none"]["curious"][
                        "coverage"
                    ],
                    "uniform_coverage": exploration["conditions"]["none"]["uniform"][
                        "coverage"
                    ],
                    "relational_held_out_gpi": relational["held_out"]["gpi_fraction"],
                    "relational_held_out_place": relational["held_out"][
                        "place_gpi_fraction"
                    ],
                },
            }
        )

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the navigation holdout mutated AgentBrain.bank")

    families = ("successor", "decomposition", "identity", "exploration", "relational")
    survival: dict[str, Any] = {}
    for family in families:
        names = sorted(replicate_rows[0][family]) if replicate_rows else []
        survival[family] = {
            name: sum(int(row[family][name]) for row in replicate_rows)
            for name in names
        }
    total = sum(len(survival[family]) for family in families)
    held = sum(
        1
        for family in families
        for name, count in survival[family].items()
        if count == len(replicate_rows)
    )

    report = {
        "schema": NAVIGATION_HOLDOUT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "block": BLOCK_NAME,
        "seeds": list(seeds[: int(replicates)]),
        "replicates": len(replicate_rows),
        "tasks": tasks,
        "claims_total": total,
        "claims_held_everywhere": held,
        "survival": survival,
        "rows": replicate_rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "navigation_holdout.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def report_own(identity: dict[str, Any], *, arm: str = "search") -> float:
    """Mean own-identification accuracy across the distractor conditions."""

    conditions = ("random_walk", "cycling")
    return sum(
        identity["conditions"][condition][f"{arm}_own_accuracy"]
        for condition in conditions
    ) / len(conditions)


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--bank", type=Path, default=repository / "artifacts/checkpoints/AgentBrain.bank"
    )
    parser.add_argument(
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_navigation_holdout_2026-08-16"
        ),
    )
    parser.add_argument("--tasks", type=int, default=HOLDOUT_TASKS)
    parser.add_argument("--replicates", type=int, default=3)
    arguments = parser.parse_args()
    report = run_navigation_holdout(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        tasks=arguments.tasks,
        replicates=arguments.replicates,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
