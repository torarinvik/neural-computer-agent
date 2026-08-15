from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.interpreter_controller import (
    InterpretedProgram,
    InterpreterController,
    one_back_program,
    operator_handles,
    resolve_operator,
    run_tick,
)
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    RenderedBrainWorkshopVerifier,
)

REPOSITORY = Path(__file__).resolve().parents[1]
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"


def _encoders() -> RenderedBrainWorkshopEncoders:
    return RenderedBrainWorkshopEncoders.load(FRONTEND)


def _interpret(program, encoders, *, steps: int = 448, seed: int = 42, mode="teacher", controller=None):
    config = RenderedBrainWorkshopConfig(
        n_back=1, steps=steps, streams=("vision",), symbol_count=4
    ).validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=seed)
    workspace = torch.zeros(program.workspace_slots, encoders.event_width)
    hits = 0
    scored = 0
    statuses: set[str] = set()
    while not verifier.done:
        with torch.no_grad():
            event = encoders.vision(verifier.observation().vision.unsqueeze(0))
        result = run_tick(program, controller, event, workspace, mode=mode)
        statuses.add(result.status)
        press = 0 if result.press is None else result.press
        step = verifier.score(torch.tensor([press], dtype=torch.long))
        if bool(step.eligible.item()):
            hits += int(step.reward.item())
            scored += 1
    return hits / scored, statuses


def test_interpreting_a_program_reproduces_the_recorded_capability() -> None:
    encoders = _encoders()
    program = one_back_program(encoders.event_width, seed=1001)
    accuracy, statuses = _interpret(program, encoders)
    # The 1-back lease recorded 1.000; interpretation must not lose it.
    assert accuracy == pytest.approx(1.0)
    assert statuses == {"halted"}


def test_adding_an_operator_leaves_the_controller_untouched() -> None:
    encoders = _encoders()
    controller = InterpreterController(encoders.event_width)
    before = controller.digest()
    program = one_back_program(encoders.event_width, seed=1001)
    widened = program.with_operator(
        "compare_two", operator_handles(encoders.event_width, seed=7)[0]
    )
    assert widened.handles.shape[0] == program.handles.shape[0] + 1
    # The invariant that keeps the vocabulary in the bank rather than the net.
    assert controller.digest() == before
    assert sum(p.numel() for p in controller.parameters()) == sum(
        p.numel() for p in InterpreterController(encoders.event_width).parameters()
    )
    accuracy, _ = _interpret(widened, encoders)
    assert accuracy == pytest.approx(1.0)


def test_more_workspace_does_not_resize_the_controller() -> None:
    encoders = _encoders()
    controller = InterpreterController(encoders.event_width)
    before = controller.digest()
    program = one_back_program(encoders.event_width, seed=1001)
    roomier = InterpretedProgram(
        handles=program.handles,
        operators=program.operators,
        instructions=program.instructions,
        operator_index=program.operator_index,
        operands=program.operands,
        workspace_slots=16,
        microstep_budget=program.microstep_budget,
    ).validate()
    assert controller.digest() == before
    accuracy, _ = _interpret(roomier, encoders)
    assert accuracy == pytest.approx(1.0)


def test_a_starved_budget_fails_closed_instead_of_defaulting() -> None:
    encoders = _encoders()
    program = one_back_program(encoders.event_width, seed=1001, budget=1)
    workspace = torch.zeros(program.workspace_slots, encoders.event_width)
    config = RenderedBrainWorkshopConfig(
        n_back=1, steps=32, streams=("vision",), symbol_count=4
    ).validate()
    verifier = RenderedBrainWorkshopVerifier(config, seed=42)
    with torch.no_grad():
        event = encoders.vision(verifier.observation().vision.unsqueeze(0))
    result = run_tick(program, None, event, workspace, mode="teacher")
    assert result.status == "budget_exhausted"
    assert result.press is None


def test_an_operand_off_the_end_fails_closed() -> None:
    encoders = _encoders()
    program = one_back_program(encoders.event_width, seed=1001)
    broken = InterpretedProgram(
        handles=program.handles,
        operators=program.operators,
        instructions=program.instructions,
        operator_index=(program.operators.index("store"),) + program.operator_index[1:],
        operands=(99,) + program.operands[1:],
        workspace_slots=program.workspace_slots,
        microstep_budget=program.microstep_budget,
    ).validate()
    workspace = torch.zeros(program.workspace_slots, encoders.event_width)
    result = run_tick(
        broken, None, torch.zeros(1, encoders.event_width), workspace, mode="teacher"
    )
    assert result.status == "invalid_operand"
    assert result.press is None


def test_operators_are_resolved_by_content_not_by_index() -> None:
    handles = operator_handles(16, seed=3)
    for index in range(handles.shape[0]):
        assert resolve_operator(handles[index], handles) == index
    # A vector near a handle resolves to it; identity is by proximity alone.
    nudged = handles[2] + 0.01 * torch.randn(16)
    assert resolve_operator(nudged, handles) == 2


def test_an_untrained_controller_does_not_accidentally_interpret() -> None:
    encoders = _encoders()
    program = one_back_program(encoders.event_width, seed=1001)
    controller = InterpreterController(encoders.event_width)
    accuracy, _ = _interpret(
        program, encoders, steps=128, mode="learned", controller=controller
    )
    # Interpretation is a learned skill this controller has not been given.
    # If this ever passes, the test is measuring something other than skill.
    assert accuracy < 0.8


def test_a_program_must_be_internally_consistent() -> None:
    handles = operator_handles(16, seed=3)
    with pytest.raises(ValueError, match="operator outside the table"):
        InterpretedProgram(
            handles=handles,
            operators=("advance",) * handles.shape[0],
            instructions=torch.zeros(2, 16),
            operator_index=(0, 99),
            operands=(0, 0),
            workspace_slots=1,
        ).validate()
    with pytest.raises(ValueError, match="instruction width"):
        InterpretedProgram(
            handles=handles,
            operators=("advance",) * handles.shape[0],
            instructions=torch.zeros(2, 8),
            operator_index=(0, 0),
            operands=(0, 0),
            workspace_slots=1,
        ).validate()
