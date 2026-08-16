"""What it costs to learn when being wrong stops naming the answer.

Two actions make a scalar reward a full supervision signal: `1 - action` is
the target whenever the reward is zero. Every learning result in this session
was collected under that, so every one of them is a supervised result wearing
a bandit's clothes, and the whole stack's dependence on it was invisible
because nothing ever tested a third action.

This measures the two things that change.

**Information per step falls.** A success names the target; a failure rules out
one of `k` and leaves `k-1`. The count of steps where the target is *known*
rather than merely constrained is reported directly, because that -- not the
episode count -- is what the learner actually has.

**The probe policy stops being free.** Under two actions it does not matter
what the agent tries. Under `k` a policy that always plays the same thing
learns only where that thing is wrong, and never what is right. So the fixed
policy is run as a control at every point, and the gap between it and uniform
exploration is the size of the problem that two actions were hiding.

Both are read against `best_constant_rate` rather than against zero: what a
learner has to beat is the best single fixed answer, and at four actions that
is around a third rather than around a half.

Every episode is drawn once at the largest budget and the smaller budgets are
prefixes of it, so a budget curve is a curve through one world rather than
through several.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file

from .accumulation_curve import _config
from .choice_agent import fixed_policy_episode, probe_episode, run_choice_machine
from .choice_induction import ChoiceTrace, induce_from_choices
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, THRESHOLD, _machine, curated_frontend
from .integrated_agent import discover_alphabet
from .rule_automata import best_constant_rate, sample_rule

EXPERIMENT_ID = "brainworkshop-choice-ceiling-2026-08-15"
CHOICE_CEILING_SCHEMA = "neural-computer.choice-ceiling.v1"
DEVELOPMENT_SEED = 41
PROBE_STEPS = 16
CONFIRMATION_STEPS = 448
ACTION_COUNTS = (2, 3, 4, 5)
STATE_COUNTS = (1, 2, 3, 4)
BUDGETS = (4, 8, 16, 28, 56)
# Beyond this a rule is too close to a constant answer to separate a learner
# from one, which is the same rejection the binary sampler always applied.
MAX_CONSTANT_RATE = 0.6


def shuffled_rewards(traces, *, seed: int):
    """The missing-evidence control, in the currency this task actually has.

    Permuting the rewards keeps how often the agent was right and destroys
    which choices it was right *about*. An agent still learning under this is
    reading the reward's frequency rather than its content.
    """

    generator = torch.Generator().manual_seed(int(seed))
    produced = []
    for trace in traces:
        order = torch.randperm(len(trace.rewards), generator=generator).tolist()
        produced.append(
            ChoiceTrace(
                symbols=trace.symbols,
                actions=trace.actions,
                rewards=tuple(trace.rewards[index] for index in order),
                eligible=trace.eligible,
                symbol_count=trace.symbol_count,
                action_count=trace.action_count,
            )
        )
    return tuple(produced)


def measure_rule(
    encoders,
    clusters: torch.Tensor,
    rule,
    *,
    seed: int,
    budgets: tuple[int, ...] = BUDGETS,
) -> list[dict[str, Any]]:
    """One rule, every probe policy, every budget as a prefix of one draw."""

    action_count = rule.action_count
    probe_config = _config(rule, PROBE_STEPS)
    full = _config(rule, CONFIRMATION_STEPS)
    largest = max(budgets)

    uniform = [
        probe_episode(
            encoders,
            probe_config,
            clusters,
            seed=seed + 1000 + index,
            policy_seed=seed + 7000 + index,
            action_count=action_count,
        )
        for index in range(largest)
    ]
    fixed = [
        fixed_policy_episode(
            encoders,
            probe_config,
            clusters,
            seed=seed + 1000 + index,
            action=0,
            action_count=action_count,
        )
        for index in range(largest)
    ]
    shuffled = shuffled_rewards(uniform, seed=seed + 11)

    constant = best_constant_rate(rule, seed=seed)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for name, traces in (
            ("uniform", uniform[:budget]),
            ("fixed", fixed[:budget]),
            ("shuffled", shuffled[:budget]),
        ):
            fit = induce_from_choices(tuple(traces))
            accuracy = (
                run_choice_machine(
                    fit.machine, encoders, full, clusters, seed=seed + 500
                )["accuracy"]
                if fit is not None
                else 0.0
            )
            rows.append(
                {
                    "action_count": action_count,
                    "state_count": rule.state_count,
                    "policy": name,
                    "budget": budget,
                    "labelled_steps": budget * PROBE_STEPS,
                    # Steps where the target is known outright. The quantity
                    # that actually falls as the action set grows.
                    "resolved_steps": sum(trace.resolved for trace in traces),
                    "fit_states": None if fit is None else fit.machine.state_count,
                    "fit_error_rate": None if fit is None else fit.error_rate,
                    "accuracy": accuracy,
                    "best_constant": constant,
                    "beats_constant": accuracy > constant,
                    "solved": accuracy >= THRESHOLD,
                    "identified": accuracy == 1.0,
                    "rule_digest": rule.digest(),
                }
            )
    return rows


def run_ceiling(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    action_counts: tuple[int, ...] = ACTION_COUNTS,
    state_counts: tuple[int, ...] = STATE_COUNTS,
    budgets: tuple[int, ...] = BUDGETS,
) -> dict[str, Any]:
    """The whole grid: actions by states by evidence, three policies each."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    anchor = sample_rule(symbol_count=4, state_count=2, seed=1)
    clusters = discover_alphabet(
        encoders, _config(anchor, CONFIRMATION_STEPS), seed=seed
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, int]] = []
    for action_count in action_counts:
        for state_count in state_counts:
            rule = sample_rule(
                symbol_count=4,
                state_count=state_count,
                seed=1200 + 10 * action_count + state_count,
                action_count=action_count,
                maximum_constant_rate=MAX_CONSTANT_RATE,
            )
            if rule is None:
                skipped.append(
                    {"action_count": action_count, "state_count": state_count}
                )
                continue
            rows.extend(
                measure_rule(
                    encoders, clusters, rule, seed=seed, budgets=budgets
                )
            )
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the choice ceiling run mutated AgentBrain.bank")

    def subset(**query):
        return [
            row
            for row in rows
            if all(row[key] == value for key, value in query.items())
        ]

    largest = max(budgets)
    summary = {
        "schema": CHOICE_CEILING_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "action_counts": list(action_counts),
        "state_counts": list(state_counts),
        "budgets": list(budgets),
        "rules": len({row["rule_digest"] for row in rows}),
        "skipped": skipped,
        "identified_at_largest_budget": {
            str(count): sum(
                1
                for row in subset(action_count=count, policy="uniform", budget=largest)
                if row["identified"]
            )
            for count in action_counts
        },
        "rules_per_action_count": {
            str(count): len(
                subset(action_count=count, policy="uniform", budget=largest)
            )
            for count in action_counts
        },
        "fixed_policy_identified_at_largest_budget": {
            str(count): sum(
                1
                for row in subset(action_count=count, policy="fixed", budget=largest)
                if row["identified"]
            )
            for count in action_counts
        },
        "shuffled_beats_constant": sum(
            1 for row in rows if row["policy"] == "shuffled" and row["beats_constant"]
        ),
        "shuffled_rows": sum(1 for row in rows if row["policy"] == "shuffled"),
        "mean_resolved_fraction": {
            str(count): (
                sum(
                    row["resolved_steps"] / row["labelled_steps"]
                    for row in subset(action_count=count, policy="uniform")
                )
                / max(1, len(subset(action_count=count, policy="uniform")))
            )
            for count in action_counts
        },
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "choice_ceiling.json").write_text(
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
            repository / "session_records" / "brainworkshop_choice_ceiling_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    arguments = parser.parse_args()
    report = run_ceiling(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
