from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.accumulation_curve import _config
from experiments.brainworkshop_canonical.interpreted_machine import (
    compile_machine,
    run_comparison,
    run_interpreted,
)
from experiments.brainworkshop_canonical.interpreter_controller import (
    OPERATOR_NAMES,
    operator_handles,
    resolve_operator,
)
from experiments.brainworkshop_canonical.interpreter_pretraining import load_interpreter
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer.promotion import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]
BANK = REPOSITORY / "artifacts/checkpoints/AgentBrain.bank"
CONTROLLER = (
    REPOSITORY / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
)
INTERPRETER = REPOSITORY / "artifacts/checkpoints/interpreter_controller_seed1001.pt"
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _rule(state_count: int, offset: int = 0):
    rule = sample_rule(
        symbol_count=4, state_count=state_count, seed=6000 + 100 * state_count + offset
    )
    assert rule is not None
    return rule


def _prototypes(width: int = 16, alphabet: int = 4) -> torch.Tensor:
    """Well-separated stand-ins for discovered clusters."""

    return torch.eye(alphabet, width) * 8.0


def test_a_compiled_machine_carries_its_state_in_the_pointer() -> None:
    """No hidden activation holds the machine's state; the runtime does."""

    machine = _rule(3)
    clusters = _prototypes()
    program = compile_machine(machine, clusters, seed=1001)
    # One block per state plus one handler pair per cell.
    assert program.instructions.shape[0] == 3 * (2 * 4 + 1) + 2 * (3 * 4)
    assert program.workspace_slots == 1
    assert program.constants is not None
    assert program.constants.shape[0] == 4


def test_the_operators_a_machine_needs_were_added_after_freezing() -> None:
    """The invariant the whole decision rests on."""

    controller, _ = load_interpreter(INTERPRETER)
    before = controller.digest()
    program = compile_machine(_rule(2), _prototypes(), seed=1001)
    used = {program.operators[index] for index in program.operator_index}
    assert {"load_const", "halt_at"} <= used
    assert controller.digest() == before


def test_narrowed_resolution_only_ever_rules_out_what_the_row_cannot_mean() -> None:
    handles = operator_handles(16, seed=5)
    intention = handles[3]
    assert resolve_operator(intention, handles) == 3
    # Restricted to a set that excludes the true nearest, the best of the
    # offered candidates wins rather than an error or a silent fallback.
    assert resolve_operator(intention, handles, among=(0, 1)) in (0, 1)
    assert resolve_operator(intention, handles, among=(3, 1)) == 3
    with pytest.raises(ValueError, match="at least one candidate"):
        resolve_operator(intention, handles, among=())


@pytest.mark.skipif(not INTERPRETER.is_file(), reason="interpreter is not present")
def test_interpretation_preserves_behaviour_and_the_controller_can_do_it(
    tmp_path,
) -> None:
    """The measurement the decision asked for, over the whole rule population.

    Three claims, and the third is the one that was in doubt: the interpreted
    program is the same program, the operator table can grow without touching
    the controller, and the controller can actually run it.
    """

    before = sha256_file(BANK)
    report = run_comparison(
        CONTROLLER, INTERPRETER, BANK, tmp_path, frontend_path=FRONTEND, seed=41
    )

    # The compiled program is the program.
    assert report["behaviour_preserved"] == report["rules"]
    assert report["mean_teacher_accuracy"] == pytest.approx(1.0)

    # Growing the instruction set left the controller alone.
    assert report["controller_digest_unchanged"]
    assert report["operators_after_freezing"] == len(OPERATOR_NAMES)

    # Resolving among the operators a row names, the frozen controller is exact.
    assert report["narrowed_exact"] == report["rules"]
    assert report["mean_narrowed_accuracy"] == pytest.approx(1.0)

    # And resolving against the whole table is measurably worse, which is the
    # finding rather than an implementation detail.
    assert report["mean_learned_accuracy"] < 0.9
    # Every error was an operator the row could not have meant.
    assert report["decode_errors_wrong_field"] == 0
    assert report["decode_errors_off_table"] > 0

    assert sha256_file(BANK) == before


@pytest.mark.skipif(not INTERPRETER.is_file(), reason="interpreter is not present")
def test_a_tick_that_decodes_nothing_presses_nothing() -> None:
    """Fail-closed, still, with the pointer external.

    A budget too small to reach an emit must not fall back to a default press.
    """

    from experiments.brainworkshop_canonical.controller_pretraining import (
        load_temporal_controller_artifact,
    )
    from experiments.brainworkshop_canonical.current_symbol_acquire import (
        FRONTEND_SEED,
        _machine,
        curated_frontend,
    )

    payload = load_temporal_controller_artifact(CONTROLLER)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=FRONTEND
    )
    machine = _rule(2)
    clusters = _prototypes(width=encoders.event_width)
    starved = compile_machine(machine, clusters, seed=1001, budget=1)
    result = run_interpreted(
        starved, encoders, _config(machine, 32), seed=42, mode="teacher"
    )
    assert result["silent_ticks"] == 32
    assert "budget_exhausted" in result["statuses"]
