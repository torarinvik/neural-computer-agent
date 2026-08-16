"""Development audit for evidence-trained composition proposal routing.

The existing compositional agent evaluates every stored pair under every
combiner.  This audit keeps its confirmation and admission rules unchanged,
but gives search a learned opaque shortlist first and measures the safe
exhaustive fallback on a stranger.  It is a proposal-throughput result, not a
new capability admission.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from neural_computer.composition_proposer import LearnedCompositionProposer
from neural_computer.induced_library import (
    InducedProgramLibrary,
    InducedProgramRecord,
    canonical_signature_stream,
)
from neural_computer.promotion import sha256_file

from .compositional_recognition import search_compositions
from .compositional_rules import product_rule, sample_primitive_pool
from .counter_state_programs import compile_rule, initial_counters, predict_symbols
from .identification_ceiling import Trace
from .rule_automata import RuleAutomaton

EXPERIMENT_ID = "brainworkshop-learned-composition-proposer-2026-08-16"
EXPERIMENT_SCHEMA = "neural-computer.learned-composition-proposer-audit.v1"
DEVELOPMENT_SEED = 41
ALPHABET = 4
LIBRARY_SIZE = 32
TRACE_COUNT = 8
TRACE_LENGTH = 16


def _record(machine: RuleAutomaton) -> InducedProgramRecord:
    program = compile_rule(
        machine,
        channel_of_symbol=tuple(range(ALPHABET)),
        cluster_count=ALPHABET,
    )
    initial = initial_counters(
        program, cluster_count=ALPHABET, states=machine.state_count
    )
    signature, _ = predict_symbols(
        program,
        canonical_signature_stream(ALPHABET),
        cluster_count=ALPHABET,
        initial_counters=initial,
    )
    return InducedProgramRecord(
        program=program,
        initial_counters=initial,
        alphabet=ALPHABET,
        signature=signature,
        provenance={"machine": machine.payload()},
    ).validate()


def _traces(machine: RuleAutomaton, *, seed: int) -> tuple[Trace, ...]:
    import torch

    generator = torch.Generator().manual_seed(int(seed))
    traces = []
    for _ in range(TRACE_COUNT):
        symbols = torch.randint(
            0, ALPHABET, (TRACE_LENGTH,), generator=generator
        ).tolist()
        traces.append(
            Trace(
                symbols=tuple(symbols),
                outputs=tuple(machine.expected(symbols)),
                eligible=(True,) * TRACE_LENGTH,
                symbol_count=ALPHABET,
            )
        )
    return tuple(traces)


def run_learned_proposer_audit(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
) -> dict[str, Any]:
    started = time.perf_counter()
    pool = sample_primitive_pool(
        symbol_count=ALPHABET, count=LIBRARY_SIZE, seed=8000 + seed
    )
    library = InducedProgramLibrary(alphabet=ALPHABET)
    for machine in pool:
        library.append(_record(machine))
    target = product_rule(pool[0], pool[1], "and")
    traces = _traces(target, seed=seed)
    exhaustive_started = time.perf_counter()
    exhaustive, exhaustive_report = search_compositions(library, traces)
    exhaustive_seconds = time.perf_counter() - exhaustive_started

    proposer = LearnedCompositionProposer(
        slot_budget=8, candidate_budget=128, fallback_on_miss=True
    )
    learned_started = time.perf_counter()
    learned, learned_report = search_compositions(
        library, traces, proposer=proposer
    )
    learned_seconds = time.perf_counter() - learned_started
    if learned is not None and learned.combiner is not None:
        proposer.observe(learned.combiner, accepted=True)

    stranger = sample_primitive_pool(
        symbol_count=ALPHABET, count=1, seed=900_000 + seed
    )[0]
    stranger_started = time.perf_counter()
    stranger_found, stranger_report = search_compositions(
        library, _traces(stranger, seed=seed + 1), proposer=proposer
    )
    stranger_seconds = time.perf_counter() - stranger_started
    bank_path = Path(__file__).parents[2] / "artifacts/checkpoints/AgentBrain.bank"
    bank_digest = sha256_file(bank_path) if bank_path.is_file() else None
    report = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "library_size": library.record_count,
        "target": "held_out_pair_slot_0_and_slot_1",
        "exhaustive": {
            "winner": exhaustive.payload() if exhaustive else None,
            "hypotheses": exhaustive_report["hypotheses"],
            "seconds": exhaustive_seconds,
        },
        "learned_proposer": {
            "winner": learned.payload() if learned else None,
            "hypotheses": learned_report["hypotheses"],
            "proposal_budget": learned_report.get("proposal_budget"),
            "fallback_hypotheses": learned_report.get("fallback_hypotheses"),
            "seconds": learned_seconds,
            "configuration": proposer.configuration(),
        },
        "stranger_control": {
            "found": stranger_found.payload() if stranger_found else None,
            "hypotheses": stranger_report["hypotheses"],
            "fallback_hypotheses": stranger_report.get("fallback_hypotheses"),
            "fallback_used": bool(proposer.last_budget.fallback),
            "seconds": stranger_seconds,
        },
        "hypothesis_reduction": (
            1.0
            - learned_report["hypotheses"] / max(1, exhaustive_report["hypotheses"])
        ),
        "wall_seconds": time.perf_counter() - started,
        "agent_bank_sha256": bank_digest,
        "agent_bank_unchanged": True,
        "claim_status": "development_proposal_throughput_diagnostic_not_promoted",
        "claim_boundary": (
            "The proposer only changes candidate ordering and shortlisting. "
            "Verifier-backed confirmation remains mandatory, the stranger "
            "control uses exhaustive fallback, and no new program is admitted "
            "or written to the curated bank."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "learned_proposer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "session_records"
        / "brainworkshop_learned_proposer_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_learned_proposer_audit(arguments.output, seed=arguments.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
