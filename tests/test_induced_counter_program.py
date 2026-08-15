from __future__ import annotations

import ast
from pathlib import Path

import pytest

from experiments.brainworkshop_canonical.controller_pretraining import (
    load_temporal_controller_artifact,
)
from experiments.brainworkshop_canonical.current_symbol_acquire import (
    FRONTEND_SEED,
    _machine,
    curated_frontend,
)
from experiments.brainworkshop_canonical.induced_counter_program import (
    induce_program,
    run_induction,
)
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
)
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY
    / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
MODULE = (
    REPOSITORY
    / "experiments/brainworkshop_canonical/induced_counter_program.py"
)

# What names oracle access, as opposed to reading the agent's own hypothesis.
# `machine.state_count` is the inferred machine and is fine; `config.rule` and
# anything derived from it is not, so the check is on the object rather than
# on the attribute name.
FORBIDDEN_NAMES = {"rule", "cluster_symbol_map", "compile_rule"}
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
    raise AssertionError(f"{name} not found in the induced program module")


def test_the_induction_cannot_see_the_rule() -> None:
    """Structural, not hopeful.

    The whole claim is that these programs are learned rather than compiled
    from the answer. "Nobody looked" is weaker than "the code does not name
    it", so this walks the function body.
    """

    leaked = _oracle_references(_function("induce_program"))
    assert not leaked, (
        f"induce_program reads {sorted(leaked)}; an induced program may not "
        "consult the rule it is supposed to have inferred"
    )


def test_the_attribute_walker_would_catch_a_leak() -> None:
    # Guard against a vacuous pass: a function that does read the rule must
    # be seen to.
    # run_induction records the true state count for the table, and says so,
    # so the walker must see `rule` there.
    assert _oracle_references(_function("run_induction")) == {"rule"}


def test_a_program_is_induced_and_executed_without_reading_the_rule() -> None:
    payload = load_temporal_controller_artifact(CONTROLLER)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    rule = sample_rule(symbol_count=4, state_count=3, seed=6300)
    assert rule is not None
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=448,
        streams=("vision",),
        symbol_count=4,
        match_rule="automaton",
        rule=rule,
    )
    row = induce_program(payload, encoders, bank, config, seed=41)
    assert row["identified"] is True
    # The hypothesis found the right complexity from feedback alone.
    assert row["inferred_state_count"] == rule.state_count
    assert row["accuracy"] == 1.0
    assert row["solved"] is True
    # Fail-closed executor: every tick halted inside its step budget.
    assert row["executor_statuses"] == "halted"
    assert row["episodes_spent"] == 2


def test_an_unidentifiable_rule_reports_rather_than_guesses() -> None:
    payload = load_temporal_controller_artifact(CONTROLLER)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    rule = sample_rule(symbol_count=4, state_count=6, seed=6600)
    assert rule is not None
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=448,
        streams=("vision",),
        symbol_count=4,
        match_rule="automaton",
        rule=rule,
    )
    row = induce_program(payload, encoders, bank, config, seed=41, node_budget=50_000)
    assert row["identified"] is False
    assert row["solved"] is False
    assert row["accuracy"] is None
    # It spent the probe and stopped, rather than compiling a guess.
    assert row["episodes_spent"] == 1
    assert "within budget" in row["reason"]


def test_the_sweep_records_and_leaves_the_curated_bank_alone(tmp_path) -> None:
    before = sha256_file(BANK)
    report = run_induction(
        CONTROLLER,
        BANK,
        tmp_path / "record",
        frontend_path=FRONTEND,
        node_budget=50_000,
    )
    assert sha256_file(BANK) == before
    assert report["bank_unchanged"] is True
    assert report["status"] == "diagnostic"
    assert report["of"] == 18
    assert (tmp_path / "record" / "induction.json").is_file()
    assert (tmp_path / "record" / "checksums.sha256").is_file()
    # Two episodes when a machine is found, one when the probe settles it.
    for row in report["rules"]:
        assert row["episodes_spent"] == (2 if row["identified"] else 1)


@pytest.mark.parametrize("state_count,seed", [(1, 6000), (2, 6200), (3, 6300)])
def test_induced_programs_match_the_compiled_ceiling_shape(state_count, seed) -> None:
    """Induced programs should be the same size as oracle-compiled ones.

    The counter bridge recorded 21-22 instructions at one state, 43 at two,
    63-66 at three. Inducing the machine from feedback rather than reading it
    must not change what the compiler emits.
    """

    payload = load_temporal_controller_artifact(CONTROLLER)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )
    bank = ExternalTemporalProgramBank.load_bank(BANK)
    rule = sample_rule(symbol_count=4, state_count=state_count, seed=seed)
    assert rule is not None
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=448,
        streams=("vision",),
        symbol_count=4,
        match_rule="automaton",
        rule=rule,
    )
    row = induce_program(payload, encoders, bank, config, seed=41)
    expected = {1: range(21, 23), 2: range(43, 44), 3: range(63, 67)}[state_count]
    assert row["instructions"] in expected
    assert row["counters"] == 5 + 2 * state_count
