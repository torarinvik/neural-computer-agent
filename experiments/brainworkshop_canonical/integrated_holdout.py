"""Spend fresh seeds, and keep what the agent learns on them.

Every record in this session ends with the same two sentences: nothing is
admitted, and the bank is unchanged. They are honest and they are the reason
none of it describes an agent. A capability that is measured and discarded is a
ceiling with better manners.

This is the run that does not discard. It differs from the development curve in
exactly two ways, and both are the point:

**The seeds are fresh.** `integrated_agent_holdout` is registered in the seed
ledger and cleared against every lifetime any earlier campaign consumed. Three
replicates, far enough apart that no task's stream can reach another's, because
one replicate of this agent strides thousands of seeds rather than the seven a
lease replicate spans.

**The library persists.** The growing arm writes a checksummed
`.library` file under `artifacts/checkpoints/`, and it is loaded and verified
after the run. What is in it is not a checkpoint of a training procedure; it is
a set of programs, each of which was induced from the agent's own feedback,
confirmed on episodes it never learned from, and refused admission if the
library already pressed that way.

`AgentBrain.bank` is still not written, and that is a deliberate boundary rather
than an omission. The temporal family it holds cannot express most of these
rules -- that was measured, not assumed -- so writing induced counter programs
into it would mean either corrupting its schema or storing something it cannot
execute. The induced library is a second store with its own discipline, and the
bank's digest is checked before and after to prove it was left alone.
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

from .integrated_agent import EXPERIMENT_ID, run_integrated
from .seed_ledger import (
    INTEGRATED_SESSIONS_PER_REPLICATE,
    assert_unused_block,
    block,
)

HOLDOUT_SCHEMA = "neural-computer.integrated-agent-holdout.v1"
BLOCK_NAME = "integrated_agent_holdout"
STREAM_LENGTH = 24
POOL_SIZE = 6


def run_holdout(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    library_directory: Path,
    *,
    frontend_path: Path | None = None,
    stream_length: int = STREAM_LENGTH,
    pool_size: int = POOL_SIZE,
) -> dict[str, Any]:
    """Three replicates on unspent seeds; keep the first replicate's library."""

    seeds = block(BLOCK_NAME)
    # Fail closed before a single episode is drawn. A holdout measurement taken
    # on a seed some diagnostic already looked at is not a holdout measurement,
    # and finding that out afterwards is finding it out too late.
    assert_unused_block(
        BLOCK_NAME, seeds, sessions=INTEGRATED_SESSIONS_PER_REPLICATE
    )

    bank_before = sha256_file(bank_path)
    library_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    replicates: list[dict[str, Any]] = []
    library_paths: list[str] = []
    for index, seed in enumerate(seeds):
        library_path = (
            library_directory
            / f"induced_programs_seed{seed}{INDUCED_LIBRARY_EXTENSION}"
        )
        report = run_integrated(
            controller_path,
            bank_path,
            output_directory / f"replicate_{index}",
            frontend_path=frontend_path,
            seed=seed,
            stream_length=stream_length,
            pool_size=pool_size,
            library_path=library_path,
        )
        # The kept file is reloaded from disk and checked, so "admitted" means
        # a program that survives a round trip through the store rather than
        # one that merely reached the end of the loop.
        stored = InducedProgramLibrary.load(library_path)
        if stored.record_count != report["growing"]["admitted"]:
            raise RuntimeError("persisted library disagrees with the run's admissions")
        replicates.append(
            {
                "seed": seed,
                "library_path": str(library_path),
                "library_sha256": sha256_file(library_path),
                "library_digest": stored.digest(),
                "stored_programs": stored.record_count,
                "acquisition_ratio": report["acquisition_ratio"],
                "verifier_step_ratio": report["verifier_step_ratio"],
                "growing": {
                    key: report["growing"][key]
                    for key in (
                        "solved",
                        "solved_nontrivial",
                        "nontrivial_tasks",
                        "trivial_tasks",
                        "recognised",
                        "induced",
                        "admitted",
                        "false_recognitions",
                        "total_acquisition_steps",
                        "total_verifier_steps",
                    )
                },
                "control": {
                    key: report["control"][key]
                    for key in (
                        "solved",
                        "solved_nontrivial",
                        "total_acquisition_steps",
                        "total_verifier_steps",
                    )
                },
                "reward_shuffled": {
                    key: report["reward_shuffled"][key]
                    for key in ("solved", "solved_nontrivial", "admitted")
                },
            }
        )
        library_paths.append(str(library_path))

    bank_after = sha256_file(bank_path)
    if bank_after != bank_before:
        raise RuntimeError("the holdout run mutated AgentBrain.bank")

    ratios = [item["acquisition_ratio"] for item in replicates]
    summary = {
        "schema": HOLDOUT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed_block": BLOCK_NAME,
        "seeds": list(seeds),
        "stream_length": stream_length,
        "pool_size": pool_size,
        "replicates": replicates,
        "acquisition_ratio_mean": sum(ratios) / len(ratios),
        "acquisition_ratio_worst": max(ratios),
        "total_programs_kept": sum(
            int(item["stored_programs"]) for item in replicates
        ),
        "false_recognitions": sum(
            int(item["growing"]["false_recognitions"]) for item in replicates
        ),
        "reward_shuffled_solved": sum(
            int(item["reward_shuffled"]["solved"]) for item in replicates
        ),
        "library_paths": library_paths,
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
            repository / "session_records" / "brainworkshop_integrated_agent_2026-08-15"
        ),
    )
    parser.add_argument(
        "--library-directory",
        type=Path,
        default=repository / "artifacts/checkpoints",
    )
    parser.add_argument("--stream-length", type=int, default=STREAM_LENGTH)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    arguments = parser.parse_args()
    summary = run_holdout(
        arguments.controller,
        arguments.bank,
        arguments.output,
        arguments.library_directory,
        frontend_path=arguments.frontend,
        stream_length=arguments.stream_length,
        pool_size=arguments.pool_size,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
