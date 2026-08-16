"""Can it build on what it built?

Pairwise composition is one level deep. The library holds primitives, the
search combines two of them, and the composite is solved. That is real reuse,
but it is reuse of things the agent was *given* -- and an agent whose library
only ever deepens by one step is not accumulating, it is interpolating.

The stronger claim is hierarchy: a composite the agent worked out and admitted
becomes a **part** in its own right, so a three-primitive task is reachable by
combining a two-primitive composite it built earlier with a primitive it
already had. Nothing in the search knows about depth. It combines pairs of
library records, and a record is a record however it got there -- so if depth
appears, it appears because admission makes it available, not because anything
was told to look for it.

The stream is therefore ordered by depth, and the order is the experiment:

    primitives  ->  pairs  ->  triples

and the arm that matters is the one where the pairs are *withheld*. If triples
are only cheap when the pairs were seen first, the depth is doing the work. If
they are equally cheap either way, then the triples were reachable from
primitives alone and the hierarchy claim is unearned -- which is a result, and
one this module is built to be able to return.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

from .accumulation_curve import _config
from .compositional_rules import (
    CompositeRule,
    product_rule,
    sample_primitive_pool,
)
from .compositional_transfer import run_arm
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .integrated_agent import CONFIRMATION_STEPS, discover_alphabet
from .rule_automata import positive_rate

EXPERIMENT_ID = "brainworkshop-hierarchical-transfer-2026-08-15"
HIERARCHY_SCHEMA = "neural-computer.hierarchical-transfer.v1"
DEVELOPMENT_SEED = 41
PRIMITIVE_COUNT = 4
PAIRS_PER_RUN = 4
TRIPLES_PER_RUN = 4
PRIMITIVE_REPEATS = 2
MIN_RATE = 0.25
MAX_RATE = 0.75


def _usable(automaton, seed: int) -> bool:
    """Reject anything a constant policy would already clear."""

    rate = positive_rate(automaton, seed=seed)
    return MIN_RATE <= rate <= MAX_RATE


def build_layers(
    *,
    seed: int,
    symbol_count: int = 4,
    primitive_count: int = PRIMITIVE_COUNT,
    pairs: int = PAIRS_PER_RUN,
    triples: int = TRIPLES_PER_RUN,
):
    """Primitives, then pairs of them, then triples built from those pairs.

    A triple is built as `(a . b) . c` using a *pair that is in the stream*, so
    the depth-2 route genuinely exists. Whether the agent takes it is the
    question; nothing here arranges for it to.
    """

    pool = sample_primitive_pool(
        symbol_count=symbol_count, count=primitive_count, seed=8000 + 37 * seed
    )
    chosen_pairs: list[CompositeRule] = []
    for left, right in combinations(range(len(pool)), 2):
        for combiner in ("and", "or", "xor"):
            automaton = product_rule(pool[left], pool[right], combiner)
            if not _usable(automaton, seed):
                continue
            chosen_pairs.append(
                CompositeRule(
                    automaton=automaton,
                    parts=(pool[left], pool[right]),
                    combiner=combiner,
                )
            )
            break
        if len(chosen_pairs) >= pairs:
            break

    chosen_triples: list[CompositeRule] = []
    for pair in chosen_pairs:
        for third in pool:
            if third.digest() in pair.part_digests():
                continue
            for combiner in ("and", "or", "xor"):
                automaton = product_rule(pair.automaton, third, combiner)
                if not _usable(automaton, seed):
                    continue
                chosen_triples.append(
                    CompositeRule(
                        automaton=automaton,
                        parts=(pair.automaton, third),
                        combiner=combiner,
                    )
                )
                break
            if len(chosen_triples) >= triples:
                break
        if len(chosen_triples) >= triples:
            break
    if not chosen_triples:
        raise ValueError("no triple cleared the press-rate window")

    def rows(with_pairs: bool):
        stream = []
        position = 0
        for repeat in range(PRIMITIVE_REPEATS):
            for primitive in pool:
                stream.append(
                    (
                        position,
                        primitive,
                        repeat,
                        "primitive",
                        None,
                        (primitive.digest(),),
                    )
                )
                position += 1
        if with_pairs:
            for pair in chosen_pairs:
                stream.append(
                    (
                        position,
                        pair.automaton,
                        0,
                        "pair",
                        pair.combiner,
                        pair.part_digests(),
                    )
                )
                position += 1
        for triple in chosen_triples:
            stream.append(
                (
                    position,
                    triple.automaton,
                    0,
                    "triple",
                    triple.combiner,
                    triple.part_digests(),
                )
            )
            position += 1
        return tuple(stream)

    return rows(True), rows(False), pool, chosen_pairs, chosen_triples


def run_hierarchy(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
) -> dict[str, Any]:
    """Triples, reached with and without the pairs that lead to them."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)

    with_pairs, without_pairs, pool, pairs, triples = build_layers(seed=seed)
    clusters = discover_alphabet(
        encoders, _config(pool[0], CONFIRMATION_STEPS), seed=seed
    )
    digest = encoders.digest()

    def arm(name, stream, *, grow, compose):
        started = time.perf_counter()
        result = run_arm(
            payload,
            encoders,
            bank,
            stream,
            clusters,
            grow=grow,
            compose=compose,
            seed=seed,
            frontend_digest=digest,
        )
        result["arm"] = name
        result["seconds"] = time.perf_counter() - started
        return result

    started = time.perf_counter()
    arms = {
        # The ladder is available: primitives, then pairs, then triples.
        "with_pairs": arm("with_pairs", with_pairs, grow=True, compose=True),
        # The rung is removed. Triples must be reached from primitives alone.
        "without_pairs": arm("without_pairs", without_pairs, grow=True, compose=True),
        # No library at all.
        "control": arm("control", without_pairs, grow=False, compose=False),
    }
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the hierarchical transfer run mutated AgentBrain.bank")

    def triples_of(name: str) -> dict[str, Any]:
        rows = [row for row in arms[name]["rows"] if row["kind"] == "triple"]
        return {
            "tasks": len(rows),
            "solved": sum(1 for row in rows if row["solved"]),
            "composed": sum(1 for row in rows if row["source"] == "composed"),
            "recognised": sum(1 for row in rows if row["source"] == "recognised"),
            "induced": sum(1 for row in rows if row["source"] == "induced"),
            "unsolved": sum(1 for row in rows if row["source"] == "unsolved"),
            "acquisition_steps": sum(int(row["acquisition_steps"]) for row in rows),
            "false_recognitions": sum(int(row["false_recognitions"]) for row in rows),
        }

    summary = {name: triples_of(name) for name in arms}
    report = {
        "schema": HIERARCHY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "primitives": len(pool),
        "pairs": len(pairs),
        "triple_count": len(triples),
        "triple_state_counts": [rule.automaton.state_count for rule in triples],
        "pair_state_counts": [rule.automaton.state_count for rule in pairs],
        "triples": summary,
        "depth_ratio": (
            summary["with_pairs"]["acquisition_steps"]
            / summary["without_pairs"]["acquisition_steps"]
            if summary["without_pairs"]["acquisition_steps"]
            else None
        ),
        "arms": arms,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "hierarchical_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


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
            / "brainworkshop_hierarchical_transfer_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    arguments = parser.parse_args()
    report = run_hierarchy(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
