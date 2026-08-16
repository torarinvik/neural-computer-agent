"""Counter programs whose output is a choice rather than a bit.

`counter_state_programs` reserves counter zero for the press and reads it as
"greater than zero". That is the right ABI for a two-action world and it cannot
express a third answer, so everything learned in `choice_induction` stopped at
the hypothesis: it could be fitted and stepped, but not compiled, not stored,
and therefore not kept or reused.

This is the same ABI widened at exactly one place. Counters `0..k-1` are the
answer, one per action, and the runtime reads the largest. Everything after
them is unchanged in shape: one input channel per discovered cluster, then free
working state. A program still ends by halting, still runs under an explicit
step budget, and is still an ordinary `ControlFlowProgram` that the existing
executor runs without knowing any of this.

The binary layout is left exactly as it was rather than being folded into this
one. Every program in the induced libraries on disk was compiled under it, and
a wider layout would renumber their counters -- so the two coexist, and which
applies is decided by the machine's own action count.
"""

from __future__ import annotations

import torch

from neural_computer.control_flow import ControlFlowInstruction, ControlFlowProgram

from .counter_state_programs import DEFAULT_STEP_BUDGET, nearest_cluster
from .rendered_environment import RenderedBrainWorkshopVerifier
from .rule_automata import RuleAutomaton

CHOICE_PROGRAM_SCHEMA = "neural-computer.choice-counter-program.v1"


def choice_layout(
    *, action_count: int, cluster_count: int, working_counters: int
) -> dict[str, int]:
    """Where the answer ends and the input begins."""

    if action_count < 2 or cluster_count < 1 or working_counters < 1:
        raise ValueError("a choice layout needs actions, inputs and working state")
    return {
        "first_output": 0,
        "action_count": int(action_count),
        "first_input": int(action_count),
        "first_working": int(action_count) + int(cluster_count),
        "counter_count": int(action_count) + int(cluster_count) + int(working_counters),
    }


def compile_choice_rule(
    machine: RuleAutomaton, *, cluster_count: int
) -> ControlFlowProgram:
    """A Mealy machine with `k` outputs, as a counter program.

    State is one-hot across persistent counters, so each cell is one test. Each
    tick evaluates the block for the active state and active input, increments
    the counter naming the answer, writes the next state, then swaps next into
    current. General over the rule class; nothing about any particular rule.
    """

    machine.validate()
    states = machine.state_count
    actions = machine.action_count
    layout = choice_layout(
        action_count=actions,
        cluster_count=cluster_count,
        working_counters=2 * states,
    )
    first_working = layout["first_working"]

    def state_counter(index: int) -> int:
        return first_working + index

    def next_counter(index: int) -> int:
        return first_working + states + index

    instructions: list[ControlFlowInstruction] = []
    patches: list[tuple[int, str]] = []
    labels: dict[str, int] = {}

    def emit(instruction: ControlFlowInstruction, label: str | None = None) -> None:
        instructions.append(instruction)
        if label is not None:
            patches.append((len(instructions) - 1, label))

    for state in range(states):
        for symbol in range(machine.symbol_count):
            end = f"block_{state}_{symbol}"
            emit(
                ControlFlowInstruction(
                    "jump_if_zero", counter=state_counter(state), target=0
                ),
                end,
            )
            emit(
                ControlFlowInstruction(
                    "jump_if_zero",
                    counter=layout["first_input"] + symbol,
                    target=0,
                ),
                end,
            )
            # The one widened line: name the answer instead of raising a flag.
            emit(
                ControlFlowInstruction(
                    "inc", counter=int(machine.outputs[state][symbol])
                )
            )
            emit(
                ControlFlowInstruction(
                    "inc", counter=next_counter(int(machine.transitions[state][symbol]))
                )
            )
            labels[end] = len(instructions)

    for state in range(states):
        labels[f"clear_{state}"] = len(instructions)
        emit(
            ControlFlowInstruction(
                "jump_if_zero", counter=state_counter(state), target=0
            ),
            f"cleared_{state}",
        )
        emit(ControlFlowInstruction("dec", counter=state_counter(state)))
        emit(ControlFlowInstruction("jump", target=0), f"clear_{state}")
        labels[f"cleared_{state}"] = len(instructions)
    for state in range(states):
        labels[f"move_{state}"] = len(instructions)
        emit(
            ControlFlowInstruction(
                "jump_if_zero", counter=next_counter(state), target=0
            ),
            f"moved_{state}",
        )
        emit(ControlFlowInstruction("dec", counter=next_counter(state)))
        emit(ControlFlowInstruction("inc", counter=state_counter(state)))
        emit(ControlFlowInstruction("jump", target=0), f"move_{state}")
        labels[f"moved_{state}"] = len(instructions)
    emit(ControlFlowInstruction("halt"))

    resolved = list(instructions)
    for position, label in patches:
        instruction = resolved[position]
        resolved[position] = ControlFlowInstruction(
            instruction.op, counter=instruction.counter, target=labels[label]
        )
    return ControlFlowProgram(layout["counter_count"], tuple(resolved)).validate()


def choice_initial_counters(
    machine: RuleAutomaton, *, cluster_count: int
) -> tuple[int, ...]:
    """Start in state zero, matching the compiler's one-hot convention."""

    layout = choice_layout(
        action_count=machine.action_count,
        cluster_count=cluster_count,
        working_counters=2 * machine.state_count,
    )
    counters = [0] * layout["counter_count"]
    counters[layout["first_working"]] = 1
    return tuple(counters)


def _answer(counters, action_count: int) -> int:
    """The largest output counter. Ties go to the lowest action."""

    best = 0
    for action in range(1, int(action_count)):
        if counters[action] > counters[best]:
            best = action
    return best


def predict_choice_symbols(
    program: ControlFlowProgram,
    symbols,
    *,
    action_count: int,
    cluster_count: int,
    initial_counters: tuple[int, ...],
    step_budget: int = DEFAULT_STEP_BUDGET,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """What a program would answer on a stream already in hand.

    Costs no episode and reads no reward, which is what makes recognition free
    however large the library gets.
    """

    program.validate()
    counters = list(initial_counters)
    answers: list[int] = []
    statuses: set[str] = set()
    for symbol in symbols:
        index = int(symbol)
        if not 0 <= index < cluster_count:
            raise ValueError("stream symbol is outside the program's alphabet")
        for action in range(action_count):
            counters[action] = 0
        for channel in range(cluster_count):
            counters[action_count + channel] = int(channel == index)
        execution = program.execute(counters, max_steps=step_budget)
        statuses.add(execution.status)
        counters = list(execution.counters)
        answers.append(_answer(counters, action_count))
    return tuple(answers), tuple(sorted(statuses))


def run_choice_program(
    program: ControlFlowProgram,
    encoders,
    config,
    clusters: torch.Tensor,
    *,
    action_count: int,
    seed: int,
    initial_counters: tuple[int, ...],
    step_budget: int = DEFAULT_STEP_BUDGET,
) -> dict[str, float | int | str]:
    """Drive one episode's answers from a compiled program."""

    config = config.validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=int(seed))
    stream = config.streams[0]
    cluster_count = int(clusters.shape[0])
    counters = list(initial_counters)
    hits = scored = 0
    statuses: set[str] = set()
    while not verifier.done:
        observation = verifier.observation()
        frame = observation.vision if stream == "vision" else observation.audio
        if frame is None:
            raise ValueError("choice program found no frame on the bound stream")
        with torch.no_grad():
            event = (
                encoders.vision(frame.unsqueeze(0))
                if stream == "vision"
                else encoders.audio(frame.unsqueeze(0))
            )
        index = int(nearest_cluster(event, clusters).item())
        for action in range(action_count):
            counters[action] = 0
        for channel in range(cluster_count):
            counters[action_count + channel] = int(channel == index)
        execution = program.execute(counters, max_steps=step_budget)
        statuses.add(execution.status)
        counters = list(execution.counters)
        answer = _answer(counters, action_count)
        step = verifier.score(torch.tensor([answer], dtype=torch.long))
        if bool(step.eligible.item()):
            hits += int(step.reward.item())
            scored += 1
    return {
        "accuracy": hits / scored if scored else 0.0,
        "scored": scored,
        "statuses": ",".join(sorted(statuses)),
    }
