from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.brainworkshop_canonical.interpreter_controller import (
    OPERATOR_NAMES,
    one_back_program,
    operator_handles,
    run_tick,
)
from experiments.brainworkshop_canonical.interpreter_pretraining import (
    evaluate_generalisation,
    interpretation_accuracy,
    load_interpreter,
    sample_batch,
)
from experiments.brainworkshop_canonical.rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    RenderedBrainWorkshopVerifier,
)

REPOSITORY = Path(__file__).resolve().parents[1]
FRONTEND = REPOSITORY / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
INTERPRETER = REPOSITORY / "artifacts/checkpoints/interpreter_controller_seed1001.pt"


def _interpret(program, encoders, controller, *, steps: int = 448, seed: int = 42):
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
        result = run_tick(program, controller, event, workspace, mode="learned")
        statuses.add(result.status)
        step = verifier.score(
            torch.tensor([0 if result.press is None else result.press], dtype=torch.long)
        )
        if bool(step.eligible.item()):
            hits += int(step.reward.item())
            scored += 1
    return hits / scored, statuses


def test_the_pretrained_controller_interprets_a_verified_capability() -> None:
    encoders = RenderedBrainWorkshopEncoders.load(FRONTEND)
    controller, record = load_interpreter(INTERPRETER)
    assert not any(p.requires_grad for p in controller.parameters())
    program = one_back_program(encoders.event_width, seed=1001)
    accuracy, statuses = _interpret(program, encoders, controller)
    # 1-back's lease recorded 1.000; interpretation by a trained controller
    # must reach the same value, not merely beat chance.
    assert accuracy == pytest.approx(1.0)
    assert statuses == {"halted"}
    assert record["parameters"] == sum(p.numel() for p in controller.parameters())


def test_operators_invented_after_freezing_still_interpret() -> None:
    encoders = RenderedBrainWorkshopEncoders.load(FRONTEND)
    controller, _ = load_interpreter(INTERPRETER)
    before = controller.digest()
    program = one_back_program(encoders.event_width, seed=1001)
    for index in range(10):
        program = program.with_operator(
            f"invented_{index}", operator_handles(encoders.event_width, seed=500 + index)[0]
        )
    # Relative, not absolute: the base operator table itself grows as the
    # machine is taught to do more, and hard-coding its size would make that
    # growth read as a regression.
    assert program.handles.shape[0] == len(OPERATOR_NAMES) + 10
    accuracy, _ = _interpret(program, encoders, controller)
    assert accuracy == pytest.approx(1.0)
    assert controller.digest() == before


def test_interpretation_generalises_to_unseen_vocabulary_sizes() -> None:
    controller, record = load_interpreter(INTERPRETER)
    assert record["operators_seen_in_training"] == 8
    report = evaluate_generalisation(
        controller, event_width=record["event_width"], seed=7, batch=1024
    )
    for row in report["held_out_handles"]:
        # Every handle here is freshly drawn and never trained on, and 16 and
        # 32 operators are vocabularies four times what training used.
        assert row["accuracy"] >= 0.99, row
        assert row["condition_met"] >= 0.99, row
        assert row["condition_unmet"] >= 0.99, row


def test_the_skill_is_branching_not_just_copying_a_field() -> None:
    controller, record = load_interpreter(INTERPRETER)
    generator = torch.Generator().manual_seed(3)
    batch = sample_batch(
        batch=2048, event_width=record["event_width"], operators=8, generator=generator
    )
    # A controller that ignored the condition and always copied the first
    # field would score near chance on the unmet half.
    met = batch["condition"]
    assert 0.3 < float(met.float().mean()) < 0.7
    assert interpretation_accuracy(controller, batch) >= 0.99
    always_primary = (batch["target_index"] == batch["target_index"][met][0]).float().mean()
    assert float(always_primary) < 0.9


def test_training_problems_carry_no_task_signal() -> None:
    generator = torch.Generator().manual_seed(11)
    batch = sample_batch(batch=64, event_width=16, operators=4, generator=generator)
    # Nothing in a training problem references a rule, a reward, or a stimulus.
    assert set(batch) == {
        "event",
        "instruction",
        "workspace",
        "handles",
        "target_index",
        "target",
        "condition",
    }
    assert batch["instruction"].shape == (64, 32)
    assert batch["handles"].shape == (64, 4, 16)
