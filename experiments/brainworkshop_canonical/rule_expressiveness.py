"""Is a failed rule unreachable by the grammar, or unrepresentable at all?

`program_search.py` warns that an inexpressible target looks exactly like slow
search, and the sampled-rule baseline cannot tell the two apart. This module
answers it by enumeration rather than by searching harder, against two
independent ceilings.

**Machine ceiling.** Every program the current geometry can hold is executed
directly: each temporal address, each with and without inverted intention,
each prototype template the acquire rule could form, and each AND of the two.
Search is bypassed entirely, so a rule that no enumerated program solves is
not a search failure.

**Window ceiling.** Independently of this machine, the best accuracy any
policy could reach while seeing only the last `w` symbols. This is the
Bayes-optimal windowed predictor: for each window context, always answer with
that context's majority press. It bounds every architecture with that much
memory, so if it is already low, no proposer and no program family fixes the
rule at `max_history = 4` and the answer is a blueprint change.

Both ceilings run against sampled rules on a consumed development seed. This
is a diagnostic and admits nothing.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file
from neural_computer.temporal_program import PROTOTYPE_MATCH_EXECUTION_SCHEMA

from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, THRESHOLD, _machine, curated_frontend
from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    render_position,
)
from .rendered_live import run_rendered_live_lifetime
from .rule_automata import RuleAutomaton, known_rule, sample_rule

EXPERIMENT_ID = "brainworkshop-rule-expressiveness-2026-08-15"
DEVELOPMENT_SEED = 41
ADDRESS_EXECUTION_SCHEMA = "neural-computer.relative-history-select.v1"
STATE_COUNTS = (1, 2, 3, 4, 5, 6)
RULES_PER_STATE_COUNT = 3
SYMBOL_COUNT = 4
STEPS = 448
WINDOWS = (1, 2, 3, 4, 5)
WINDOW_PROBE_STEPS = 40_000


def windowed_ceiling(
    rule: RuleAutomaton, *, window: int, seed: int, steps: int = WINDOW_PROBE_STEPS
) -> float:
    """Best accuracy any policy can reach seeing only the last `window` symbols.

    The optimal windowed policy answers each context with its majority press,
    so its accuracy is the share of ticks that fall in the majority. No
    architecture with this much memory can beat it.
    """

    if window < 1:
        raise ValueError("a window ceiling needs at least one symbol")
    generator = torch.Generator().manual_seed(int(seed))
    symbols = torch.randint(0, rule.symbol_count, (steps,), generator=generator).tolist()
    expected = rule.expected(symbols)
    contexts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for position in range(window - 1, len(symbols)):
        context = tuple(symbols[position - window + 1 : position + 1])
        contexts[context][expected[position]] += 1
    scored = sum(sum(counter.values()) for counter in contexts.values())
    best = sum(max(counter.values()) for counter in contexts.values())
    return best / scored if scored else 0.0


def _symbol_events(encoders, *, symbol_count: int, frame_size: int) -> torch.Tensor:
    """One learned event per symbol, straight from the frozen frontend."""

    with torch.no_grad():
        frames = torch.stack(
            [render_position(symbol, size=frame_size) for symbol in range(symbol_count)]
        )
        return encoders.vision(frames)


def prototype_cover(events: torch.Tensor) -> tuple[tuple[tuple[int, ...], torch.Tensor], ...]:
    """Every template the acquire rule could converge on.

    Acquisition averages the events it is rewarded for, so a reachable
    template is the mean of some non-empty subset of the symbol events. With a
    small alphabet the cover is exhaustive rather than a sample.
    """

    symbol_count = events.shape[0]
    cover: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for size in range(1, symbol_count + 1):
        for subset in combinations(range(symbol_count), size):
            cover.append((subset, events[list(subset)].mean(dim=0)))
    return tuple(cover)


def _score(machine, encoders, config, *, seed: int) -> float:
    report = run_rendered_live_lifetime(
        machine, encoders, config, seed=seed, learn=False, sample=False
    )
    return float(report.eligible_accuracy)


def machine_ceiling(
    payload: dict[str, object],
    encoders,
    rule: RuleAutomaton,
    *,
    seed: int,
    steps: int,
    frame_size: int = 36,
) -> dict[str, Any]:
    """Execute every program this geometry can hold; keep the best."""

    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        symbol_count=rule.symbol_count,
        match_rule="automaton",
        rule=rule,
    )
    machine = _machine(payload, learn=False)
    machine.learning_enabled = False
    machine.sample = False
    events = _symbol_events(encoders, symbol_count=rule.symbol_count, frame_size=frame_size)
    cover = prototype_cover(events)
    best = {"accuracy": 0.0, "program": None}
    evaluated = 0

    def consider(accuracy: float, program: str) -> None:
        nonlocal best
        if accuracy > best["accuracy"]:
            best = {"accuracy": accuracy, "program": program}

    # Temporal addresses, with and without inverted intention.
    for address in range(machine.max_history):
        for invert in (False, True):
            with torch.no_grad():
                machine.relative_address_logits.data.fill_(-20.0)
                machine.relative_address_logits.data[address] = 20.0
                machine.prototype.data.zero_()
            machine._execution_schema = ADDRESS_EXECUTION_SCHEMA
            machine._invert_intention = invert
            machine._combine_and = False
            consider(
                _score(machine, encoders, config, seed=seed),
                f"{'invert ' if invert else ''}address {address}",
            )
            evaluated += 1

    # Prototype templates, with and without inverted intention.
    for subset, template in cover:
        for invert in (False, True):
            with torch.no_grad():
                machine.prototype.data.copy_(template)
            machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
            machine._invert_intention = invert
            machine._combine_and = False
            consider(
                _score(machine, encoders, config, seed=seed),
                f"{'invert ' if invert else ''}prototype {subset}",
            )
            evaluated += 1

    # AND of an inverted address with a template.
    for address in range(machine.max_history):
        for subset, template in cover:
            with torch.no_grad():
                machine.relative_address_logits.data.fill_(-20.0)
                machine.relative_address_logits.data[address] = 20.0
                machine.prototype.data.copy_(template)
            machine._execution_schema = PROTOTYPE_MATCH_EXECUTION_SCHEMA
            machine._invert_intention = True
            machine._combine_and = True
            consider(
                _score(machine, encoders, config, seed=seed),
                f"and(invert address {address}, prototype {subset})",
            )
            evaluated += 1

    return {
        "best_accuracy": best["accuracy"],
        "best_program": best["program"],
        "programs_evaluated": evaluated,
    }


def run_expressiveness(
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
    """Both ceilings, on the same rules the baseline swept. Admits nothing."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    def measure(rule: RuleAutomaton, label: str | None) -> dict[str, Any]:
        ceiling = machine_ceiling(
            payload, encoders, rule, seed=seed + 1, steps=steps
        )
        windows = {
            str(window): windowed_ceiling(rule, window=window, seed=seed)
            for window in WINDOWS
        }
        machine_best = float(ceiling["best_accuracy"])
        window_best = float(windows[str(max(WINDOWS))])
        return {
            "rule_digest": rule.digest(),
            "hand_written_rule": label,
            "state_count": rule.state_count,
            "machine_ceiling": machine_best,
            "machine_best_program": ceiling["best_program"],
            "programs_evaluated": ceiling["programs_evaluated"],
            "window_ceilings": windows,
            "expressible": machine_best >= THRESHOLD,
            # A rule no windowed policy can reach is not a proposer problem.
            "reachable_by_any_window_policy": window_best >= THRESHOLD,
            "verdict": (
                "expressible"
                if machine_best >= THRESHOLD
                else (
                    "unreachable_at_this_memory"
                    if window_best < THRESHOLD
                    else "expressible_window_but_not_by_this_program_family"
                )
            ),
        }

    for states in state_counts:
        for index in range(rules_per_state_count):
            rule = sample_rule(
                symbol_count=symbol_count,
                state_count=states,
                seed=6000 + 100 * states + index,
            )
            if rule is None:
                continue
            rows.append(measure(rule, None))
    hand_rows = [
        measure(known_rule(name, symbol_count=symbol_count, **kwargs), name)
        for name, kwargs in (
            ("current_symbol", {}),
            ("onset", {}),
            ("changed", {}),
            ("n_back", {"n_back": 1}),
        )
    ]
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("expressiveness diagnostic mutated AgentBrain.bank")
    verdicts = Counter(row["verdict"] for row in rows)
    report = {
        "schema": "neural-computer.rule-expressiveness.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic",
        "note": "development seed, already consumed; nothing admitted",
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "seed": seed,
        "steps": steps,
        "threshold": THRESHOLD,
        "windows": list(WINDOWS),
        "window_probe_steps": WINDOW_PROBE_STEPS,
        "sampled": rows,
        "hand_written": hand_rows,
        "verdicts": dict(verdicts),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "expressiveness.json").write_text(
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
            / "brainworkshop_rule_expressiveness_2026-08-15"
        ),
    )
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    report = run_expressiveness(
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
                "verdicts": report["verdicts"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
