"""What the current closed grammar does against sampled rules.

This is a diagnostic, not a lease. It runs on an already-consumed development
seed and admits nothing; its purpose is to place the existing search on the
complexity axis that `rule_automata` provides, so later work has a baseline to
beat.

Each sampled rule is run through the same searcher the leases use. A rule
counts as solved when the search gates a winner at the usual threshold. The
never-press baseline is reported beside it, because a rule whose positive rate
is far from a half can be half-solved by a constant policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import (
    FRONTEND_SEED,
    MINIMUM_BITS,
    THRESHOLD,
    _machine,
    curated_frontend,
)
from .program_search import search_temporal_programs
from .rendered_environment import RenderedBrainWorkshopConfig
from .rendered_live import run_rendered_live_lifetime
from .rule_automata import known_rule, positive_rate, sample_rule

EXPERIMENT_ID = "brainworkshop-sampled-rule-baseline-2026-08-15"
# Already consumed by earlier development work; nothing fresh is spent here.
DEVELOPMENT_SEED = 41
STATE_COUNTS = (1, 2, 3, 4, 5, 6)
RULES_PER_STATE_COUNT = 3
SYMBOL_COUNT = 4
STEPS = 448


def _run_one(
    payload: dict[str, object],
    bank: ExternalTemporalProgramBank,
    encoders,
    rule,
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        symbol_count=rule.symbol_count,
        match_rule="automaton",
        rule=rule,
    )
    machine = _machine(payload, learn=False)

    def acquire(proposal):
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
    best = max(
        (float(row.get("accuracy", 0.0)) for row in search["attempts"] if row["executed"]),
        default=0.0,
    )
    rate = positive_rate(rule, seed=seed)
    return {
        "rule_digest": rule.digest(),
        "state_count": rule.state_count,
        "positive_rate": rate,
        "constant_policy_accuracy": max(rate, 1.0 - rate),
        "winner_kind": winner["kind"] if winner else None,
        "best_executed_accuracy": best,
        "solved": winner is not None,
    }


def run_baseline(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    state_counts: tuple[int, ...] = STATE_COUNTS,
    rules_per_state_count: int = RULES_PER_STATE_COUNT,
    symbol_count: int = SYMBOL_COUNT,
    steps: int = STEPS,
    seed: int = DEVELOPMENT_SEED,
    frontend_path: Path | None = None,
) -> dict[str, Any]:
    """Sweep sampled rules by complexity. Never writes the bank."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for states in state_counts:
        for index in range(rules_per_state_count):
            rule = sample_rule(
                symbol_count=symbol_count,
                state_count=states,
                seed=6000 + 100 * states + index,
            )
            if rule is None:
                continue
            rows.append(
                _run_one(payload, bank, encoders, rule, seed=seed, steps=steps)
            )
    # The four hand-written rules, for comparison on the same axis.
    hand_rows: list[dict[str, Any]] = []
    for name, kwargs in (
        ("current_symbol", {}),
        ("onset", {}),
        ("changed", {}),
        ("n_back", {"n_back": 1}),
    ):
        rule = known_rule(name, symbol_count=symbol_count, **kwargs)
        row = _run_one(payload, bank, encoders, rule, seed=seed, steps=steps)
        row["hand_written_rule"] = name + (
            f"-{kwargs['n_back']}" if kwargs else ""
        )
        hand_rows.append(row)
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("rule baseline mutated AgentBrain.bank")
    by_states = {
        states: [row for row in rows if row["state_count"] == states]
        for states in state_counts
    }
    report = {
        "schema": "neural-computer.sampled-rule-baseline.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic",
        "note": (
            "development seed, already consumed; nothing admitted and no "
            "holdout claim"
        ),
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "seed": seed,
        "steps": steps,
        "symbol_count": symbol_count,
        "threshold": THRESHOLD,
        "sampled": rows,
        "hand_written": hand_rows,
        "solved_by_state_count": {
            str(states): {
                "solved": sum(1 for row in group if row["solved"]),
                "of": len(group),
            }
            for states, group in by_states.items()
        },
        "hand_written_solved": sum(1 for row in hand_rows if row["solved"]),
        "hand_written_of": len(hand_rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "baseline.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    checksums = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output_directory.glob("*.json"))
    ]
    (output_directory / "checksums.sha256").write_text("\n".join(checksums) + "\n")
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser()
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
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_sampled_rule_baseline_2026-08-15"
        ),
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    report = run_baseline(
        arguments.controller_artifact,
        arguments.bank,
        arguments.output_dir,
        steps=arguments.steps,
        frontend_path=arguments.frontend,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "bank_unchanged": report["bank_unchanged"],
                "solved_by_state_count": report["solved_by_state_count"],
                "hand_written_solved": (
                    f"{report['hand_written_solved']}/{report['hand_written_of']}"
                ),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
