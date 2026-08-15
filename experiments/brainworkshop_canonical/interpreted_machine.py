"""Give the induced machine to the controller, and see what it costs.

`DECISION_CONTROLLER_IS_THE_INTERPRETER.md` names the controller as the thing
that executes external programs, and every result in this session bypasses it.
The counter bridge, the induced programs, the integrated agent -- all of them
decide presses in a Python executor while the pretrained interpreter sits in
`artifacts/checkpoints/` doing nothing. Worse, the accumulation curve now bends
the right way *without* it, which weakens rather than supports the decision.

This is the measurement that settles it, and it is cheap now that there is
something real to interpret: compile an induced Mealy machine into an
`InterpretedProgram`, run it through the interpreter, and compare.

Three things had to be true for this to be possible, and none of them is a
special case for finite-state rules.

**The condition has to be the one the controller was taught.** The interpreter
controller knows exactly one thing: name the operator in an instruction's first
field when the current event matches the workspace, and the second field
otherwise. So the workspace holds one slot, `load_const` puts a symbol
prototype into it, and the very next instruction's condition reads "is the
current symbol this one". Nothing else is asked of the network.

**The machine's state has to live outside the network.** It lives in the
program counter: one block of instructions per state, and a transition is
where the tick parks the pointer. `halt_at` ends a tick at a chosen row, so the
state is carried between ticks by the runtime rather than by any hidden
activation.

**Growing the instruction set must not touch the controller.** `load_const`
and `halt_at` were added after the controller was frozen. They are two more
rows in a table the *program* carries, and the controller's digest is asserted
unchanged across the whole measurement -- which is the invariant the decision
rests on, tested against a real capability rather than a reference program.

What this cannot do is make the controller good at interpreting. It reports
what the controller actually achieves, next to what the same program achieves
when the operators are read off the instruction instead of predicted. If those
two numbers differ, the difference is the cost of the decision, and it belongs
in the record rather than in an argument.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file

from .accumulation_curve import _config, curriculum_rules
from .controller_pretraining import load_temporal_controller_artifact
from .counter_state_programs import compile_rule as compile_counter
from .counter_state_programs import initial_counters, run_counter_program
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .interpreter_controller import (
    OPERATOR_NAMES,
    InterpretedProgram,
    InterpreterController,
    operator_handles,
    run_tick,
)
from .interpreter_pretraining import MATCH_TOLERANCE, load_interpreter
from .prototype_templates import cluster_events, observe_events
from .rendered_environment import RenderedBrainWorkshopVerifier
from .rule_automata import RuleAutomaton

EXPERIMENT_ID = "brainworkshop-interpreted-machine-2026-08-15"
INTERPRETED_MACHINE_SCHEMA = "neural-computer.interpreted-machine.v1"
DEVELOPMENT_SEED = 41
STEPS = 448
# Longest path through one state's block is two rows per symbol plus the
# handler pair, so this is generous rather than tight.
MICROSTEP_BUDGET = 64


def compile_machine(
    machine: RuleAutomaton,
    clusters: torch.Tensor,
    *,
    seed: int,
    budget: int = MICROSTEP_BUDGET,
) -> InterpretedProgram:
    """A Mealy machine as instruction rows, with its state in the pointer.

    One block per state. Inside a block, each symbol is tested by loading its
    prototype and letting the condition decide whether to enter that symbol's
    handler. A handler emits the press and parks the pointer at the next
    state's block.

    This is a general compiler over the rule class. Nothing in it is chosen for
    any particular rule, and nothing in it is visible to the controller, which
    sees only opaque instruction rows and a handle table.
    """

    machine.validate()
    if clusters.ndim != 2 or int(clusters.shape[0]) != machine.symbol_count:
        raise ValueError("prototypes must cover the machine's alphabet exactly")
    width = int(clusters.shape[1])
    handles = operator_handles(width, seed=seed)
    names = OPERATOR_NAMES
    op = {name: names.index(name) for name in names}

    states = machine.state_count
    alphabet = machine.symbol_count
    block_length = 2 * alphabet + 1

    def block(state: int) -> int:
        return state * block_length

    def handler(state: int, symbol: int) -> int:
        return states * block_length + 2 * (state * alphabet + symbol)

    met: list[int] = []
    unmet: list[int] = []
    operands: list[int] = []

    def emit_row(met_op: str, unmet_op: str, operand: int) -> None:
        met.append(op[met_op])
        unmet.append(op[unmet_op])
        operands.append(int(operand))

    for state in range(states):
        for symbol in range(alphabet):
            # Put this symbol's prototype where the condition can see it.
            emit_row("load_const", "load_const", symbol)
            # The only learned decision in the whole program: is the current
            # event that prototype? If so enter the handler, otherwise try the
            # next symbol.
            emit_row("jump", "advance", handler(state, symbol))
        # No prototype matched. Fail closed rather than press on a guess.
        emit_row("halt", "halt", 0)
    for state in range(states):
        for symbol in range(alphabet):
            emit_row("emit", "emit", int(machine.outputs[state][symbol]))
            emit_row(
                "halt_at", "halt_at", block(int(machine.transitions[state][symbol]))
            )

    instructions = torch.stack(
        [
            torch.cat((handles[met[row]], handles[unmet[row]]))
            for row in range(len(met))
        ]
    )
    return InterpretedProgram(
        handles=handles,
        operators=names,
        instructions=instructions,
        operator_index=tuple(met),
        operands=tuple(operands),
        workspace_slots=1,
        microstep_budget=budget,
        alternate_index=tuple(unmet),
        constants=clusters.detach().clone(),
    ).validate()


def run_interpreted(
    program: InterpretedProgram,
    encoders,
    config,
    *,
    seed: int,
    controller: InterpreterController | None = None,
    mode: str = "teacher",
    match_tolerance: float = MATCH_TOLERANCE,
    resolve_within_instruction: bool = False,
) -> dict[str, Any]:
    """Drive one episode's presses through the interpreter."""

    config = config.validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=int(seed))
    workspace = torch.zeros(program.workspace_slots, program.event_width)
    pointer = 0
    hits = 0
    scored = 0
    microsteps = 0
    ticks = 0
    statuses: set[str] = set()
    silent = 0
    while not verifier.done:
        frame = verifier.observation().vision
        with torch.no_grad():
            event = encoders.vision(frame.unsqueeze(0))
        result = run_tick(
            program,
            controller,
            event,
            workspace,
            mode=mode,
            match_tolerance=match_tolerance,
            start_pointer=pointer,
            resolve_within_instruction=resolve_within_instruction,
        )
        statuses.add(result.status)
        microsteps += result.microsteps
        ticks += 1
        pointer = result.next_pointer
        # A tick that emitted nothing still has to answer the verifier. It
        # answers "no press" and the silence is counted, rather than being
        # laundered into a decision the program never made.
        if result.press is None:
            silent += 1
        press = 0 if result.press is None else int(result.press)
        step = verifier.score(torch.tensor([press], dtype=torch.long))
        if bool(step.eligible.item()):
            hits += int(step.reward.item())
            scored += 1
    return {
        "accuracy": hits / scored if scored else 0.0,
        "scored": scored,
        "statuses": ",".join(sorted(statuses)),
        "silent_ticks": silent,
        "microsteps_per_tick": microsteps / ticks if ticks else 0.0,
        "instructions": int(program.instructions.shape[0]),
    }


def clusters_in_symbol_order(encoders, clusters: torch.Tensor, *, symbol_count: int):
    """Reorder discovered prototypes so index k is the verifier's symbol k.

    **This is an experimenter's oracle and it is only sound here.** A learner
    never needs it: a machine induced from feedback is already expressed in
    cluster indices, so the identity map is correct for it by construction, and
    `induced_counter_program` exists to make exactly that point.

    This module is not measuring learning. It is measuring whether three
    execution paths agree on the same program, and for that the program has to
    be one whose behaviour is known independently -- which means the sampled
    rule, whose symbols are the verifier's rather than the frontend's. Without
    this reordering the compiled program implements a permutation of the rule,
    the absolute accuracies mean nothing, and only the agreement between paths
    survives. A first version of this module made that mistake.
    """

    from .counter_state_programs import cluster_symbol_map

    channel_of_symbol = cluster_symbol_map(
        encoders, clusters, symbol_count=symbol_count
    )
    if sorted(channel_of_symbol) != list(range(symbol_count)):
        raise ValueError("prototypes do not correspond one-to-one with symbols")
    return clusters[list(channel_of_symbol)]


def decode_audit(
    program: InterpretedProgram,
    controller: InterpreterController,
    events: torch.Tensor,
    clusters: torch.Tensor,
    *,
    match_tolerance: float = MATCH_TOLERANCE,
) -> dict[str, Any]:
    """Where the controller disagrees with the instruction it is reading.

    "The controller cannot interpret this" is not actionable. Whether it picks
    the wrong *field* of an instruction, or a handle that is in neither field,
    says different things: the first is the decoding skill the pretraining is
    supposed to install, the second is a content-addressing failure that a
    larger operator table would make worse.

    Rather than following one execution trace, this questions the controller on
    every row of the program under *both* branch outcomes, using real encoder
    events and the program's own constants as the workspace. That covers rows a
    trace might never reach, and it is the stronger test.
    """

    alphabet = int(clusters.shape[0])
    rows = int(program.instructions.shape[0])
    totals = {"rows": 0, "agreed": 0, "wrong_field": 0, "off_table": 0}
    conditional = {"rows": 0, "agreed": 0}
    for row in range(rows):
        instruction = program.instructions[row].reshape(1, -1)
        fields = {int(program.operator_index[row]), int(program.unmet_index(row))}
        for held in range(alphabet):
            workspace = clusters[held].reshape(1, -1)
            for symbol in range(alphabet):
                # A real observed event of this symbol, not a synthetic vector.
                sample = events[symbol].reshape(1, -1)
                met = (
                    float(torch.linalg.vector_norm(sample - workspace))
                    <= match_tolerance
                )
                expected = (
                    int(program.operator_index[row])
                    if met
                    else program.unmet_index(row)
                )
                with torch.no_grad():
                    intention = controller(sample, instruction, workspace)
                chosen = int(
                    (program.handles @ intention.reshape(-1)).argmax().item()
                )
                totals["rows"] += 1
                totals["agreed"] += int(chosen == expected)
                if chosen != expected:
                    if chosen in fields:
                        totals["wrong_field"] += 1
                    else:
                        totals["off_table"] += 1
                if len(fields) > 1:
                    conditional["rows"] += 1
                    conditional["agreed"] += int(chosen == expected)
    return {
        "decoded_rows": totals["rows"],
        "decode_accuracy": totals["agreed"] / totals["rows"] if totals["rows"] else 0.0,
        "wrong_field": totals["wrong_field"],
        "off_table": totals["off_table"],
        "conditional_rows": conditional["rows"],
        "conditional_accuracy": (
            conditional["agreed"] / conditional["rows"] if conditional["rows"] else 0.0
        ),
    }


def compare_paths(
    payload: dict[str, object],
    encoders,
    controller: InterpreterController,
    machine: RuleAutomaton,
    clusters: torch.Tensor,
    symbol_events: torch.Tensor,
    config,
    *,
    seed: int,
) -> dict[str, Any]:
    """The same machine, three ways: counters, interpreter, learned interpreter."""

    counter_program = compile_counter(
        machine,
        channel_of_symbol=tuple(range(int(clusters.shape[0]))),
        cluster_count=int(clusters.shape[0]),
    )
    counters = run_counter_program(
        counter_program,
        encoders,
        config,
        clusters,
        seed=seed,
        initial_counters=initial_counters(
            counter_program,
            cluster_count=int(clusters.shape[0]),
            states=machine.state_count,
        ),
    )
    interpreted = compile_machine(machine, clusters, seed=FRONTEND_SEED)
    teacher = run_interpreted(interpreted, encoders, config, seed=seed, mode="teacher")
    learned = run_interpreted(
        interpreted,
        encoders,
        config,
        seed=seed,
        controller=controller,
        mode="learned",
    )
    # The same controller, the same program, the same episode -- resolving the
    # intention only among the operators the row actually names.
    narrowed = run_interpreted(
        interpreted,
        encoders,
        config,
        seed=seed,
        controller=controller,
        mode="learned",
        resolve_within_instruction=True,
    )
    audit = decode_audit(interpreted, controller, symbol_events, clusters)
    return {
        "states": machine.state_count,
        "decode_accuracy": audit["decode_accuracy"],
        "conditional_accuracy": audit["conditional_accuracy"],
        "wrong_field": audit["wrong_field"],
        "off_table": audit["off_table"],
        "counter_accuracy": float(counters["accuracy"]),
        "counter_instructions": len(counter_program.instructions),
        "teacher_accuracy": teacher["accuracy"],
        "learned_accuracy": learned["accuracy"],
        "narrowed_accuracy": narrowed["accuracy"],
        "narrowed_statuses": narrowed["statuses"],
        "narrowed_silent_ticks": narrowed["silent_ticks"],
        "interpreted_instructions": teacher["instructions"],
        "teacher_statuses": teacher["statuses"],
        "learned_statuses": learned["statuses"],
        "learned_silent_ticks": learned["silent_ticks"],
        "microsteps_per_tick": teacher["microsteps_per_tick"],
        "behaviour_preserved": teacher["accuracy"] == float(counters["accuracy"]),
        "interpretation_cost": float(counters["accuracy"]) - learned["accuracy"],
        "narrowed_cost": float(counters["accuracy"]) - narrowed["accuracy"],
    }


def run_comparison(
    controller_path: Path,
    interpreter_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    steps: int = STEPS,
) -> dict[str, Any]:
    """Every sampled rule, executed by counters and by the controller."""

    bank_before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    controller, _ = load_interpreter(interpreter_path)
    controller_before = controller.digest()

    rules = curriculum_rules()
    clusters = clusters_in_symbol_order(
        encoders,
        cluster_events(observe_events(encoders, _config(rules[0], steps), seed=seed)),
        symbol_count=rules[0].symbol_count,
    )
    # One real observed event per symbol, for the decode audit. Reads no
    # reward and is not used by any execution path.
    observed = observe_events(encoders, _config(rules[0], steps), seed=seed)
    assignment = torch.cdist(observed, clusters).argmin(dim=1)
    symbol_events = torch.stack(
        [observed[assignment == symbol][0] for symbol in range(int(clusters.shape[0]))]
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for rule in rules:
        row = compare_paths(
            payload,
            encoders,
            controller,
            rule,
            clusters,
            symbol_events,
            _config(rule, steps),
            seed=seed + 1,
        )
        row["rule_digest"] = rule.digest()
        rows.append(row)

    controller_after = controller.digest()
    if controller_after != controller_before:
        raise RuntimeError("interpreting changed the controller's parameters")
    if sha256_file(bank_path) != bank_before:
        raise RuntimeError("the interpreted-machine comparison mutated AgentBrain.bank")

    preserved = sum(1 for row in rows if row["behaviour_preserved"])
    report = {
        "schema": INTERPRETED_MACHINE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "rules": len(rows),
        "behaviour_preserved": preserved,
        "operators_after_freezing": len(OPERATOR_NAMES),
        "controller_digest": controller_before,
        "controller_digest_unchanged": controller_after == controller_before,
        "mean_counter_accuracy": sum(row["counter_accuracy"] for row in rows)
        / len(rows),
        "mean_teacher_accuracy": sum(row["teacher_accuracy"] for row in rows)
        / len(rows),
        "mean_learned_accuracy": sum(row["learned_accuracy"] for row in rows)
        / len(rows),
        "worst_interpretation_cost": max(row["interpretation_cost"] for row in rows),
        "mean_narrowed_accuracy": sum(row["narrowed_accuracy"] for row in rows)
        / len(rows),
        "worst_narrowed_cost": max(row["narrowed_cost"] for row in rows),
        "narrowed_exact": sum(1 for row in rows if row["narrowed_cost"] == 0.0),
        "mean_decode_accuracy": sum(row["decode_accuracy"] for row in rows) / len(rows),
        "mean_conditional_accuracy": sum(
            row["conditional_accuracy"] for row in rows
        ) / len(rows),
        "decode_errors_off_table": sum(int(row["off_table"]) for row in rows),
        "decode_errors_wrong_field": sum(int(row["wrong_field"]) for row in rows),
        "rows": rows,
        "agent_bank_sha256": bank_before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "interpreted_machine.json").write_text(
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
        "--interpreter",
        type=Path,
        default=repository / "artifacts/checkpoints/interpreter_controller_seed1001.pt",
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
            / "brainworkshop_interpreted_machine_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    arguments = parser.parse_args()
    report = run_comparison(
        arguments.controller,
        arguments.interpreter,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
