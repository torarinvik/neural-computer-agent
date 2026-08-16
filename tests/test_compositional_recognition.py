from __future__ import annotations

import torch

from experiments.brainworkshop_canonical.compositional_recognition import (
    Candidate,
    composed_machine,
    flatten_targets,
    machine_of,
    record_predictions,
    search_compositions,
)
from experiments.brainworkshop_canonical.compositional_rules import (
    product_rule,
    sample_primitive_pool,
)
from experiments.brainworkshop_canonical.counter_state_programs import (
    compile_rule,
    initial_counters,
    predict_symbols,
)
from experiments.brainworkshop_canonical.identification_ceiling import Trace
from neural_computer.induced_library import (
    InducedProgramLibrary,
    InducedProgramRecord,
    canonical_signature_stream,
)

ALPHABET = 4


def _record(machine, **extra) -> InducedProgramRecord:
    program = compile_rule(
        machine, channel_of_symbol=tuple(range(ALPHABET)), cluster_count=ALPHABET
    )
    start = initial_counters(
        program, cluster_count=ALPHABET, states=machine.state_count
    )
    signature, _ = predict_symbols(
        program,
        canonical_signature_stream(ALPHABET),
        cluster_count=ALPHABET,
        initial_counters=start,
    )
    return InducedProgramRecord(
        program=program,
        initial_counters=start,
        alphabet=ALPHABET,
        signature=signature,
        provenance={"machine": machine.payload(), **extra},
    ).validate()


def _traces(machine, count: int, length: int = 16, seed: int = 3):
    generator = torch.Generator().manual_seed(seed)
    produced = []
    for _ in range(count):
        stream = torch.randint(0, ALPHABET, (length,), generator=generator).tolist()
        produced.append(
            Trace(
                symbols=tuple(stream),
                outputs=tuple(machine.expected(stream)),
                eligible=tuple([True] * length),
                symbol_count=ALPHABET,
            )
        )
    return produced


def _noisy(traces, *, rate: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    produced = []
    for trace in traces:
        flips = (
            torch.rand(len(trace.outputs), generator=generator) < rate
        ).tolist()
        produced.append(
            Trace(
                symbols=trace.symbols,
                outputs=tuple(
                    value ^ int(flip)
                    for value, flip in zip(trace.outputs, flips, strict=True)
                ),
                eligible=trace.eligible,
                symbol_count=trace.symbol_count,
            )
        )
    return produced


def _pool(count: int = 4, seed: int = 8000):
    return sample_primitive_pool(symbol_count=ALPHABET, count=count, seed=seed)


def _library(machines) -> InducedProgramLibrary:
    library = InducedProgramLibrary(alphabet=ALPHABET)
    for machine in machines:
        library.append(_record(machine))
    return library


def test_a_composite_is_found_from_the_parts_that_built_it() -> None:
    """The claim: a task never seen, solved out of things already held."""

    pool = _pool()
    library = _library(pool)
    for combiner in ("and", "or", "xor"):
        target = product_rule(pool[0], pool[1], combiner)
        found, report = search_compositions(library, _traces(target, 8))
        assert found is not None, combiner
        assert found.kind == "pair"
        assert set(found.slots) == {0, 1}
        assert found.combiner == combiner
        assert report["hypotheses"] > report["singles_examined"]


def test_a_single_record_still_wins_when_it_explains_the_evidence() -> None:
    """Composition must not talk the agent out of the simple answer."""

    pool = _pool()
    library = _library(pool)
    found, _ = search_compositions(library, _traces(pool[2], 8))
    assert found is not None
    assert found.kind == "single"
    assert found.slots == (2,)


def test_nothing_is_offered_for_a_task_the_library_cannot_build() -> None:
    """The failure that would make every number here meaningless."""

    library = _library(_pool())
    stranger = _pool(count=2, seed=500_000)[0]
    found, report = search_compositions(library, _traces(stranger, 28))
    assert found is None
    assert report["hypotheses"] > 20


def test_the_multiplicity_correction_is_available_and_off_by_default() -> None:
    """Off because it was measured, not because it is unprincipled.

    It prevents about five false adoptions in eight hundred unrelated targets,
    all of which confirmation refuses anyway, and costs six of eight composable
    tasks at 10% label noise. The second gate is not optional, so the free one
    is not tightened to duplicate it.
    """

    library = _library(_pool(count=6))
    _, default = search_compositions(library, _traces(_pool()[0], 4))
    _, corrected = search_compositions(
        library, _traces(_pool()[0], 4), correct_for_multiplicity=True
    )
    assert default["hypotheses"] == corrected["hypotheses"]
    assert default["effective_alpha"] == default["alpha"]
    assert corrected["effective_alpha"] < default["effective_alpha"]
    assert corrected["effective_alpha"] == default["alpha"] / corrected["hypotheses"]


def test_composition_survives_label_noise_the_correction_would_not() -> None:
    """The trade the default is chosen on, at the scale a test can hold."""

    pool = _pool()
    library = _library(pool)
    target = product_rule(pool[0], pool[1], "and")
    noisy = _noisy(_traces(target, 8), rate=0.10, seed=5)
    found, _ = search_compositions(library, noisy)
    assert found is not None
    assert set(found.slots) == {0, 1}
    assert found.combiner == "and"


def test_a_refused_slot_takes_its_combinations_with_it() -> None:
    pool = _pool()
    library = _library(pool)
    target = product_rule(pool[0], pool[1], "xor")
    traces = _traces(target, 8)
    assert search_compositions(library, traces)[0] is not None
    found, _ = search_compositions(library, traces, exclude=frozenset({0}))
    assert found is None or 0 not in found.slots


def test_a_found_combination_becomes_a_real_machine() -> None:
    """Behavioural agreement is not executable; the composite has to be built."""

    pool = _pool()
    library = _library(pool)
    target = product_rule(pool[0], pool[2], "or")
    found, _ = search_compositions(library, _traces(target, 8))
    assert found is not None
    built = composed_machine(library, found)
    assert built is not None
    stream = list(range(ALPHABET)) * 30
    assert built.expected(stream) == target.expected(stream)


def test_a_record_without_a_hypothesis_cannot_be_composed() -> None:
    """The library stores programs and keeps provenance opaque; composition is
    the one thing that needs to look inside, and it must fail closed."""

    pool = _pool()
    library = InducedProgramLibrary(alphabet=ALPHABET)
    program = compile_rule(
        pool[0], channel_of_symbol=tuple(range(ALPHABET)), cluster_count=ALPHABET
    )
    start = initial_counters(
        program, cluster_count=ALPHABET, states=pool[0].state_count
    )
    signature, _ = predict_symbols(
        program,
        canonical_signature_stream(ALPHABET),
        cluster_count=ALPHABET,
        initial_counters=start,
    )
    library.append(
        InducedProgramRecord(
            program=program,
            initial_counters=start,
            alphabet=ALPHABET,
            signature=signature,
            provenance={"source": "opaque"},
        )
    )
    assert machine_of(library.record(0)) is None
    assert composed_machine(library, Candidate("single", (0,), None, 1, 1)) is None


def test_predictions_line_up_with_the_labels_they_are_scored_against() -> None:
    """Flattening once is what makes combination free; misaligning it would
    make every agreement meaningless and nothing downstream would notice."""

    pool = _pool()
    library = _library(pool)
    traces = _traces(pool[1], 4)
    targets, positions = flatten_targets(traces)
    predicted = record_predictions(library.record(1), traces, positions)
    assert len(predicted) == len(targets)
    assert predicted == targets


def test_an_empty_library_offers_nothing_rather_than_failing() -> None:
    empty = InducedProgramLibrary(alphabet=ALPHABET)
    found, report = search_compositions(empty, _traces(_pool()[0], 4))
    assert found is None
    assert report["hypotheses"] == 0
    found, report = search_compositions(_library(_pool()), ())
    assert found is None
    assert report["trials"] == 0
