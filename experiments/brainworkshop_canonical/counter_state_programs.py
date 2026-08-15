"""Give the program family persistent state, using the substrate that exists.

Eleven of eighteen sampled rules have no representation in the temporal
family: its whole use of a four-tick history is to test equality against one
lagged symbol, which scores `-0.003` against a memoryless policy. The missing
ingredient is not memory depth but accumulated state.

`control_flow.py` already provides that — a two-counter-machine ABI with a
fail-closed executor, step budgets, and an admission boundary — but nothing
connects it to the rendered stream. This module is that bridge, and it fixes
only an interface, not a rule class:

- counter 0 is the **press**, read after halt and cleared before each tick;
- counters 1..k are **input channels**, one per event cluster discovered from
  observation, set one-hot to the current event's nearest cluster;
- every later counter is **persistent working state**, carried across ticks
  untouched by the runtime.

Nothing about that layout encodes a task. A program may use the working
counters however it likes, and the substrate stays Turing-complete in the
limit, bounded per tick by an explicit step budget.

`compile_rule` is an **experimenter's oracle**, not a learner. It takes a rule
the agent never sees and emits a program that implements it, which is how this
module measures whether the family *can* express something — exactly as the
enumeration ceiling did. Compiled programs are diagnostics and are never
admitted to a bank.
"""

from __future__ import annotations

import torch

from neural_computer.control_flow import ControlFlowInstruction, ControlFlowProgram

from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    RenderedBrainWorkshopVerifier,
    render_position,
)
from .rule_automata import RuleAutomaton

PRESS_COUNTER = 0
DEFAULT_STEP_BUDGET = 8192


def counter_layout(cluster_count: int, working_counters: int) -> dict[str, int]:
    """Where the fixed interface ends and free working state begins."""

    if cluster_count < 1 or working_counters < 1:
        raise ValueError("a counter layout needs inputs and working state")
    return {
        "press": PRESS_COUNTER,
        "first_input": 1,
        "first_working": 1 + cluster_count,
        "counter_count": 1 + cluster_count + working_counters,
    }


def nearest_cluster(events: torch.Tensor, clusters: torch.Tensor) -> torch.Tensor:
    """Quantise each observed event to its closest discovered cluster."""

    return torch.cdist(events, clusters).argmin(dim=1)


def run_counter_program(
    program: ControlFlowProgram,
    encoders: RenderedBrainWorkshopEncoders,
    config: RenderedBrainWorkshopConfig,
    clusters: torch.Tensor,
    *,
    seed: int,
    step_budget: int = DEFAULT_STEP_BUDGET,
    initial_counters: tuple[int, ...] | None = None,
) -> dict[str, float | int | str]:
    """Drive one episode's presses from a counter program's persistent state."""

    config = config.validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=int(seed))
    stream = config.streams[0]
    cluster_count = int(clusters.shape[0])
    layout = counter_layout(cluster_count, program.counter_count - 1 - cluster_count)
    if program.counter_count != layout["counter_count"]:
        raise ValueError("program counter count does not match the cluster layout")
    counters = (
        list(initial_counters)
        if initial_counters is not None
        else [0] * program.counter_count
    )
    if len(counters) != program.counter_count:
        raise ValueError("initial counters do not match the program")
    hits = 0
    scored = 0
    statuses: set[str] = set()
    while not verifier.done:
        observation = verifier.observation()
        frame = observation.vision if stream == "vision" else observation.audio
        if frame is None:
            raise ValueError("counter program found no frame on the bound stream")
        with torch.no_grad():
            event = (
                encoders.vision(frame.unsqueeze(0))
                if stream == "vision"
                else encoders.audio(frame.unsqueeze(0))
            )
        index = int(nearest_cluster(event, clusters).item())
        counters[PRESS_COUNTER] = 0
        for channel in range(cluster_count):
            counters[layout["first_input"] + channel] = int(channel == index)
        execution = program.execute(counters, max_steps=step_budget)
        statuses.add(execution.status)
        counters = list(execution.counters)
        press = 1 if counters[PRESS_COUNTER] > 0 else 0
        step = verifier.score(torch.tensor([press], dtype=torch.long))
        hits += int(step.reward.item())
        scored += 1
    return {
        "accuracy": hits / scored if scored else 0.0,
        "scored": scored,
        "statuses": ",".join(sorted(statuses)),
    }


def predict_symbols(
    program: ControlFlowProgram,
    symbols,
    *,
    cluster_count: int,
    initial_counters: tuple[int, ...],
    step_budget: int = DEFAULT_STEP_BUDGET,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Run a program over a symbol stream with no environment attached.

    `run_counter_program` needs a verifier because it scores. Recognition and
    signatures need neither: they ask what a program *would* press given a
    stream that is already in hand, which costs no episode and reads no
    reward. Same ABI, same executor, no environment.
    """

    program.validate()
    if len(initial_counters) != program.counter_count:
        raise ValueError("initial counters do not match the program")
    counters = list(initial_counters)
    presses: list[int] = []
    statuses: set[str] = set()
    for symbol in symbols:
        index = int(symbol)
        if not 0 <= index < cluster_count:
            raise ValueError("stream symbol is outside the program's alphabet")
        counters[PRESS_COUNTER] = 0
        for channel in range(cluster_count):
            counters[1 + channel] = int(channel == index)
        execution = program.execute(counters, max_steps=step_budget)
        statuses.add(execution.status)
        counters = list(execution.counters)
        presses.append(1 if counters[PRESS_COUNTER] > 0 else 0)
    return tuple(presses), tuple(sorted(statuses))


def cluster_symbol_map(
    encoders: RenderedBrainWorkshopEncoders,
    clusters: torch.Tensor,
    *,
    symbol_count: int,
    frame_size: int = 36,
) -> tuple[int, ...]:
    """Which input channel each symbol lands on. Oracle-side only."""

    with torch.no_grad():
        frames = torch.stack(
            [render_position(symbol, size=frame_size) for symbol in range(symbol_count)]
        )
        events = encoders.vision(frames)
    return tuple(int(index) for index in nearest_cluster(events, clusters))


def compile_rule(
    rule: RuleAutomaton,
    *,
    channel_of_symbol: tuple[int, ...],
    cluster_count: int,
) -> ControlFlowProgram:
    """Oracle: emit a counter program implementing one rule.

    State is held one-hot across persistent counters, which keeps every test a
    single instruction. Each tick evaluates the block for the active state and
    active input, writes the press and the next state, then swaps next into
    current. This is a general compiler over the rule class, not a program
    written for any particular rule, and it is never given to the learner.
    """

    rule.validate()
    states = rule.state_count
    if len(channel_of_symbol) != rule.symbol_count:
        raise ValueError("symbol-to-channel map does not cover the alphabet")
    layout = counter_layout(cluster_count, 2 * states)
    first_working = layout["first_working"]

    def state_counter(index: int) -> int:
        return first_working + index

    def next_counter(index: int) -> int:
        return first_working + states + index

    instructions: list[ControlFlowInstruction] = []
    patches: list[tuple[int, str]] = []

    def emit(instruction: ControlFlowInstruction, label: str | None = None) -> None:
        instructions.append(instruction)
        if label is not None:
            patches.append((len(instructions) - 1, label))

    labels: dict[str, int] = {}

    for state in range(states):
        for symbol in range(rule.symbol_count):
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
                    counter=layout["first_input"] + channel_of_symbol[symbol],
                    target=0,
                ),
                end,
            )
            if rule.outputs[state][symbol]:
                emit(ControlFlowInstruction("inc", counter=PRESS_COUNTER))
            emit(
                ControlFlowInstruction(
                    "inc", counter=next_counter(int(rule.transitions[state][symbol]))
                )
            )
            labels[end] = len(instructions)

    # Clear the current state, then move next into current.
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


def initial_counters(program: ControlFlowProgram, *, cluster_count: int, states: int) -> tuple[int, ...]:
    """Start in state zero, matching the compiler's one-hot convention."""

    layout = counter_layout(cluster_count, 2 * states)
    counters = [0] * program.counter_count
    counters[layout["first_working"]] = 1
    return tuple(counters)


def _search_space_log10(counter_count: int, length: int) -> float:
    """log10 of how many programs enumeration would sift at this length.

    Reported as a logarithm because the count overflows a float long before
    the program lengths this substrate actually needs.
    """

    from math import log10

    basis = 2 * counter_count + length * (1 + 2 * counter_count)
    return (length - 1) * log10(float(basis))


def run_ceiling(
    output_directory,
    *,
    frontend_path,
    state_counts=(1, 2, 3, 4, 5, 6),
    rules_per_state_count: int = 3,
    symbol_count: int = 4,
    steps: int = 448,
    seed: int = 41,
):
    """Does the counter substrate express the rules the temporal family cannot?"""

    import json
    import time
    from pathlib import Path

    from neural_computer.promotion import sha256_file

    from .prototype_templates import cluster_events, observe_events
    from .rule_automata import known_rule, sample_rule

    encoders = RenderedBrainWorkshopEncoders.load(Path(frontend_path))
    rows = []
    started = time.perf_counter()

    def measure(rule: RuleAutomaton, label: str | None):
        config = RenderedBrainWorkshopConfig(
            n_back=1,
            steps=steps,
            streams=("vision",),
            symbol_count=rule.symbol_count,
            match_rule="automaton",
            rule=rule,
        )
        clusters = cluster_events(observe_events(encoders, config, seed=seed))
        channels = cluster_symbol_map(
            encoders, clusters, symbol_count=rule.symbol_count
        )
        program = compile_rule(
            rule,
            channel_of_symbol=channels,
            cluster_count=int(clusters.shape[0]),
        )
        start = initial_counters(
            program, cluster_count=int(clusters.shape[0]), states=rule.state_count
        )
        result = run_counter_program(
            program, encoders, config, clusters, seed=seed + 1, initial_counters=start
        )
        length = len(program.instructions)
        return {
            "rule_digest": rule.digest(),
            "hand_written_rule": label,
            "state_count": rule.state_count,
            "clusters_discovered": int(clusters.shape[0]),
            "instructions": length,
            "counter_count": program.counter_count,
            "accuracy": float(result["accuracy"]),
            "execution_status": result["statuses"],
            "solved": float(result["accuracy"]) >= 0.999,
            "enumeration_space_log10": _search_space_log10(
                program.counter_count, length
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
    hand = [
        measure(known_rule(name, symbol_count=symbol_count, **kwargs), name)
        for name, kwargs in (
            ("current_symbol", {}),
            ("onset", {}),
            ("changed", {}),
            ("n_back", {"n_back": 1}),
            ("n_back", {"n_back": 2}),
        )
    ]
    report = {
        "schema": "neural-computer.counter-state-ceiling.v1",
        "experiment_id": "brainworkshop-counter-state-ceiling-2026-08-15",
        "status": "diagnostic",
        "note": (
            "compiled by an experimenter oracle from rules the learner never "
            "sees; nothing admitted and no search performed"
        ),
        "seed": seed,
        "steps": steps,
        "sampled": rows,
        "hand_written": hand,
        "sampled_solved": sum(1 for row in rows if row["solved"]),
        "sampled_of": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ceiling.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (directory / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(directory.glob("*.json"))
        )
        + "\n"
    )
    return report


def main() -> None:
    import json
    from pathlib import Path

    repository = Path(__file__).parents[2]
    report = run_ceiling(
        repository / "session_records" / "brainworkshop_counter_state_ceiling_2026-08-15",
        frontend_path=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "sampled_solved": f"{report['sampled_solved']}/{report['sampled_of']}",
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
