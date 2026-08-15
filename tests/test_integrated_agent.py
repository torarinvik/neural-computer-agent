from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.accumulation_curve import curriculum_rules
from experiments.brainworkshop_canonical.counter_state_programs import (
    compile_rule,
    initial_counters,
    predict_symbols,
)
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from experiments.brainworkshop_canonical.integrated_agent import (
    PROBE_STEPS,
    THRESHOLD,
    proves_competence,
    recognise,
    run_integrated,
    shuffled_feedback,
    task_stream,
)
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer.induced_library import (
    InducedProgramLibrary,
    InducedProgramRecord,
    canonical_signature_stream,
)
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
MODULE = REPOSITORY / "experiments/brainworkshop_canonical/integrated_agent.py"

FORBIDDEN_NAMES = {"rule", "cluster_symbol_map", "positive_rate"}
FORBIDDEN_ATTRIBUTES = {"rule", "expected", "cluster_symbol_map"}


def _oracle_references(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Attribute) and item.attr in FORBIDDEN_ATTRIBUTES:
            found.add(item.attr)
        elif isinstance(item, ast.Name) and item.id in FORBIDDEN_NAMES:
            found.add(item.id)
    return found


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(MODULE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in the integrated agent module")


def _rule(state_count: int, offset: int = 0):
    rule = sample_rule(
        symbol_count=4, state_count=state_count, seed=6000 + 100 * state_count + offset
    )
    assert rule is not None
    return rule


def _record(rule, alphabet: int = 4) -> InducedProgramRecord:
    program = compile_rule(
        rule, channel_of_symbol=tuple(range(alphabet)), cluster_count=alphabet
    )
    start = initial_counters(program, cluster_count=alphabet, states=rule.state_count)
    signature, _ = predict_symbols(
        program,
        canonical_signature_stream(alphabet),
        cluster_count=alphabet,
        initial_counters=start,
    )
    return InducedProgramRecord(
        program=program,
        initial_counters=start,
        alphabet=alphabet,
        signature=signature,
    ).validate()


def _traces(rule, count: int, seed: int = 3, length: int = PROBE_STEPS):
    import torch

    generator = torch.Generator().manual_seed(seed)
    produced = []
    for _ in range(count):
        stream = torch.randint(0, 4, (length,), generator=generator).tolist()
        produced.append(
            Trace(
                symbols=tuple(stream),
                outputs=tuple(rule.expected(stream)),
                eligible=tuple([True] * length),
                symbol_count=4,
            )
        )
    return produced


# --- the statistical discipline the loop turns on -------------------------


def test_recognition_demands_evidence_for_competence_not_absence_against() -> None:
    """The bug that produced a wrong program adopted at 0.73.

    Sixteen perfect labels do not separate a correct program from a slightly
    wrong one, and a test that only asks whether competence can be *ruled out*
    accepts on them. This asks the other question.
    """

    assert not proves_competence(16, 16, threshold=THRESHOLD)
    assert proves_competence(32, 32, threshold=THRESHOLD)
    # Good but not perfect is not enough at the same count.
    assert not proves_competence(30, 32, threshold=THRESHOLD)
    # And no amount of evidence rescues a rate under the gate.
    assert not proves_competence(700, 1000, threshold=THRESHOLD)
    assert not proves_competence(0, 0, threshold=THRESHOLD)


def test_a_stored_program_is_recognised_only_once_evidence_supports_it() -> None:
    rule = _rule(3)
    library = InducedProgramLibrary(alphabet=4)
    library.append(_record(rule))
    assert recognise(library, _traces(rule, 1)) is None
    found = recognise(library, _traces(rule, 2))
    assert found is not None and found[0] == 0


def test_a_wrong_program_is_never_recognised_however_much_evidence() -> None:
    """The failure that matters: adopting the library's answer to a different
    question. A confident library is worse than no library."""

    library = InducedProgramLibrary(alphabet=4)
    library.append(_record(_rule(3)))
    library.append(_record(_rule(4)))
    assert recognise(library, _traces(_rule(5), 28)) is None


def test_a_refused_slot_is_not_offered_again() -> None:
    rule = _rule(3)
    library = InducedProgramLibrary(alphabet=4)
    library.append(_record(rule))
    traces = _traces(rule, 4)
    assert recognise(library, traces) is not None
    assert recognise(library, traces, exclude=frozenset({0})) is None


# --- the stream ------------------------------------------------------------


def test_the_task_stream_repeats_and_spreads_across_complexity() -> None:
    rules = curriculum_rules()
    stream = task_stream(rules, length=24, pool_size=6, seed=41)
    assert len(stream) == 24
    assert len({rule.digest() for _, rule, _ in stream}) <= 6
    # A pool taken as a prefix would be nothing but one- and two-state rules.
    assert len({rule.state_count for _, rule, _ in stream}) >= 4
    assert max(repeat for _, _, repeat in stream) >= 1
    with pytest.raises(ValueError, match="pool size"):
        task_stream(rules, length=4, pool_size=0, seed=1)


def test_shuffling_feedback_preserves_the_label_marginal() -> None:
    """Otherwise the control would be testing label frequency, not structure."""

    trace = _traces(_rule(4), 1)[0]
    corrupted = shuffled_feedback(7)(trace, 0)
    assert sorted(corrupted.outputs) == sorted(trace.outputs)
    assert corrupted.symbols == trace.symbols
    assert corrupted.outputs != trace.outputs


# --- the boundary ----------------------------------------------------------


def test_the_agent_loop_reads_nothing_of_the_rule_but_its_name() -> None:
    """Sharper than "does not mention the rule", which it cannot satisfy.

    `solve_task` is handed the rule because the *verifier* needs it to generate
    episodes, exactly as every live experiment in this repository does. What
    must not happen is the agent consulting it. So the check is on what is read
    off the object: an identity for the record, a state count for the table,
    and nothing that could answer the question the agent is being asked.
    """

    permitted = {"digest", "state_count"}
    node = _function("solve_task")
    read = {
        item.attr
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and item.value.id == "rule"
    }
    assert read <= permitted, f"solve_task reads rule.{sorted(read - permitted)}"


def test_no_agent_function_touches_an_oracle() -> None:
    for name in ("recognise", "discover_alphabet", "admit", "proves_competence"):
        leaked = _oracle_references(_function(name))
        assert not leaked, f"{name} reads {sorted(leaked)}"


def test_the_walker_would_catch_a_leak() -> None:
    leaked = _oracle_references(ast.parse("presses = config.rule.expected(symbols)"))
    assert leaked == {"rule", "expected"}


# --- end to end ------------------------------------------------------------


@pytest.mark.skipif(not BANK.is_file(), reason="curated bank is not present")
def test_the_library_makes_acquisition_cheaper_and_keeps_what_it_learns(
    tmp_path,
) -> None:
    """The claim, end to end, against its own matched control.

    Small enough to run in a test; the record's numbers come from a longer
    stream. What must hold at any size is the direction and the controls.
    """

    before = sha256_file(BANK)
    report = run_integrated(
        CONTROLLER,
        BANK,
        tmp_path,
        frontend_path=FRONTEND,
        seed=41,
        stream_length=8,
        pool_size=3,
        library_path=tmp_path / "induced.library",
    )
    growing, control = report["growing"], report["control"]

    # It solves the tasks, and it solves them without the library helping the
    # control arm by accident.
    assert growing["solved"] == growing["tasks"]
    assert control["solved"] == control["tasks"]
    assert growing["recognised"] > 0
    assert control["recognised"] == 0

    # Cheaper where the claim lives, and never at the cost of being wrong.
    assert report["acquisition_ratio"] < 0.8
    assert growing["false_recognitions"] == 0

    # It kept something, and the kept thing is on disk and loads.
    assert growing["admitted"] >= 1
    library = (tmp_path / "induced.library")
    assert library.is_file()
    assert InducedProgramLibrary.load(library).record_count == growing["admitted"]

    # Destroying the feedback destroys the learning.
    assert report["reward_shuffled"]["solved"] == 0
    assert report["reward_shuffled"]["admitted"] == 0

    # And nothing touched the brain.
    assert report["agent_bank_unchanged"]
    assert sha256_file(BANK) == before
