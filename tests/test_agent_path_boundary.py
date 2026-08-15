"""The agent path may not import an oracle, a diagnostic, or a rule.

`DECISION_CONTROLLER_IS_THE_INTERPRETER.md` invariant 6 says nothing in the
deployed path may consult a rule, a task identity, or an oracle. Several
modules in this repository do exactly that on purpose — the rule compiler
reads the answer, the samplers define the task, the sweeps grade it — and the
only thing keeping them out of the agent is that nobody imported them.

This test makes that structural instead of hopeful, by walking the transitive
import graph rather than the top-level imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "experiments" / "brainworkshop_canonical"

# Reads the rule, defines the task, or grades an experiment.
FORBIDDEN = {
    "counter_state_programs",  # compile_rule reads the rule directly
    "rule_automata",           # defines and samples the task distribution
    "rule_baseline",           # experiment sweep
    "rule_expressiveness",     # experiment sweep
}

# What the agent is: the interpreter and what it needs to run.
AGENT_PATH = ("interpreter_controller", "interpreter_pretraining")


def _first_party_imports(module: str) -> set[str]:
    path = PACKAGE / f"{module}.py"
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            for alias in node.names:
                found.add(alias.name)
            if node.module:
                found.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("experiments.brainworkshop_canonical."):
                found.add(node.module.split(".")[-1])
    return {name for name in found if (PACKAGE / f"{name}.py").exists()}


def _transitive(module: str) -> set[str]:
    seen: set[str] = set()
    frontier = [module]
    while frontier:
        current = frontier.pop()
        for dependency in _first_party_imports(current):
            if dependency not in seen:
                seen.add(dependency)
                frontier.append(dependency)
    return seen


@pytest.mark.parametrize("module", AGENT_PATH)
def test_the_agent_path_never_reaches_an_oracle(module: str) -> None:
    reached = _transitive(module)
    leaked = reached & FORBIDDEN
    assert not leaked, (
        f"{module} transitively imports {sorted(leaked)}; the agent path may "
        "not consult a rule, a task identity, or an oracle"
    )


def test_the_forbidden_modules_still_exist_and_are_the_right_ones() -> None:
    # If one is renamed away, this test must fail loudly rather than pass
    # vacuously by guarding a module that no longer exists.
    for module in FORBIDDEN:
        assert (PACKAGE / f"{module}.py").exists(), module
    # And each really does read something the agent must not see.
    compiler = (PACKAGE / "counter_state_programs.py").read_text()
    assert "def compile_rule" in compiler
    assert "oracle" in compiler.lower()


def test_the_import_walker_would_actually_catch_a_leak() -> None:
    # Guard against the walker silently returning nothing: a module that does
    # import an oracle must be seen to.
    reached = _transitive("rule_baseline")
    assert "rule_automata" in reached
    assert reached & FORBIDDEN
