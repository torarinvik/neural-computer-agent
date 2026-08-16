"""Spend fresh seeds on composition, and keep what it builds.

Every composition number so far is on a spent development seed with a scratch
library, which under this repository's own rules makes it a diagnostic however
many controls it carries. This is the run that makes it a measurement: three
replicates on the previously unspent `compositional_transfer_holdout` block,
and the composing arm's library persisted, checksummed, and reloaded.

What gets kept is different in kind from what the integrated agent's holdout
kept. Those were programs induced from feedback. Some of these were never
induced at all -- they are products the agent assembled out of files it already
had, confirmed in the environment, and admitted as files in their own right.
That is what makes the store a library rather than a cache, and it is why the
artifact is worth having on disk rather than in a table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from neural_computer.induced_library import (
    INDUCED_LIBRARY_EXTENSION,
    InducedProgramLibrary,
)
from neural_computer.promotion import sha256_file

from .compositional_transfer import EXPERIMENT_ID, run_transfer
from .seed_ledger import (
    INTEGRATED_SESSIONS_PER_REPLICATE,
    assert_unused_block,
    block,
)

HOLDOUT_SCHEMA = "neural-computer.compositional-transfer-holdout.v1"
BLOCK_NAME = "compositional_transfer_holdout"
COMPOSITES = 8


def run_holdout(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    library_directory: Path,
    *,
    frontend_path: Path | None = None,
    composites: int = COMPOSITES,
) -> dict[str, Any]:
    """Three replicates on unspent seeds; keep every composing arm's library."""

    seeds = block(BLOCK_NAME)
    assert_unused_block(
        BLOCK_NAME, seeds, sessions=INTEGRATED_SESSIONS_PER_REPLICATE
    )

    bank_before = sha256_file(bank_path)
    library_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    replicates: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        library_path = (
            library_directory
            / f"composed_programs_seed{seed}{INDUCED_LIBRARY_EXTENSION}"
        )
        report = run_transfer(
            controller_path,
            bank_path,
            output_directory / f"replicate_{index}",
            frontend_path=frontend_path,
            seed=seed,
            composites=composites,
            library_path=library_path,
        )
        stored = InducedProgramLibrary.load(library_path)
        composing = report["arms"]["composing"]
        if stored.record_count != composing["final_library_size"]:
            raise RuntimeError("persisted library disagrees with the run")
        # How many of the kept files were *built* rather than induced. This is
        # the number the whole record is about, and it is read back off disk.
        assembled = sum(
            1
            for record in stored.records()
            if record.provenance.get("source") == "composed"
        )
        replicates.append(
            {
                "seed": seed,
                "library_path": str(library_path),
                "library_sha256": sha256_file(library_path),
                "stored_programs": stored.record_count,
                "stored_composed": assembled,
                "composite_acquisition": report["composite_acquisition"],
                "composition_ratio_against_recognition": report[
                    "composition_ratio_against_recognition"
                ],
                "composition_ratio_against_control": report[
                    "composition_ratio_against_control"
                ],
                "disjoint_ratio_against_its_control": report[
                    "disjoint_ratio_against_its_control"
                ],
                "composites": {
                    arm: {
                        key: report["arms"][arm]["composites"][key]
                        for key in (
                            "tasks",
                            "trivial",
                            "solved_nontrivial",
                            "composed",
                            "recognised",
                            "induced",
                            "unsolved",
                            "false_recognitions",
                            "combiner_recovered",
                            "parts_recovered",
                            "max_hypotheses",
                        )
                    }
                    for arm in ("composing", "recognising", "control", "disjoint", "shuffled")
                },
            }
        )

    bank_after = sha256_file(bank_path)
    if bank_after != bank_before:
        raise RuntimeError("the compositional holdout mutated AgentBrain.bank")

    def across(arm: str, key: str) -> int:
        return sum(int(item["composites"][arm][key]) for item in replicates)

    control_ratios = [
        item["composition_ratio_against_control"] for item in replicates
    ]
    retrieval_ratios = [
        item["composition_ratio_against_recognition"] for item in replicates
    ]
    disjoint_ratios = [
        item["disjoint_ratio_against_its_control"] for item in replicates
    ]
    summary = {
        "schema": HOLDOUT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed_block": BLOCK_NAME,
        "seeds": list(seeds),
        "replicates": replicates,
        "composites": across("composing", "tasks"),
        "trivial_composites": across("composing", "trivial"),
        "solved_nontrivial": across("composing", "solved_nontrivial"),
        "composed": across("composing", "composed"),
        "combiner_recovered": across("composing", "combiner_recovered"),
        "parts_recovered": across("composing", "parts_recovered"),
        "false_recognitions": across("composing", "false_recognitions"),
        "shuffled_solved_nontrivial": across("shuffled", "solved_nontrivial"),
        "control_ratio_mean": sum(control_ratios) / len(control_ratios),
        "control_ratio_worst": max(control_ratios),
        "retrieval_ratio_mean": sum(retrieval_ratios) / len(retrieval_ratios),
        "retrieval_ratio_worst": max(retrieval_ratios),
        "disjoint_ratio_mean": sum(disjoint_ratios) / len(disjoint_ratios),
        "disjoint_ratio_best": min(disjoint_ratios),
        "programs_kept": sum(int(item["stored_programs"]) for item in replicates),
        "programs_kept_that_were_assembled": sum(
            int(item["stored_composed"]) for item in replicates
        ),
        "agent_bank_sha256": bank_before,
        "agent_bank_unchanged": bank_after == bank_before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "holdout.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


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
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
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
            / "brainworkshop_compositional_transfer_2026-08-15"
        ),
    )
    parser.add_argument(
        "--library-directory",
        type=Path,
        default=repository / "artifacts/checkpoints",
    )
    parser.add_argument("--composites", type=int, default=COMPOSITES)
    arguments = parser.parse_args()
    summary = run_holdout(
        arguments.controller,
        arguments.bank,
        arguments.output,
        arguments.library_directory,
        frontend_path=arguments.frontend,
        composites=arguments.composites,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
