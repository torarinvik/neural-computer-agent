from __future__ import annotations

import json

import pytest

from experiments.brainworkshop_canonical.counter_state_programs import (
    compile_rule,
    initial_counters,
    predict_symbols,
)
from experiments.brainworkshop_canonical.rule_automata import sample_rule
from neural_computer.induced_library import (
    InducedProgramLibrary,
    InducedProgramRecord,
    canonical_signature_stream,
    signature_distance,
)


def _rule(state_count: int, offset: int = 0):
    rule = sample_rule(
        symbol_count=4, state_count=state_count, seed=6000 + 100 * state_count + offset
    )
    assert rule is not None
    return rule


def _record(rule, alphabet: int = 4, **provenance) -> InducedProgramRecord:
    program = compile_rule(
        rule, channel_of_symbol=tuple(range(alphabet)), cluster_count=alphabet
    )
    start = initial_counters(
        program, cluster_count=alphabet, states=rule.state_count
    )
    signature, statuses = predict_symbols(
        program,
        canonical_signature_stream(alphabet),
        cluster_count=alphabet,
        initial_counters=start,
    )
    assert statuses == ("halted",)
    return InducedProgramRecord(
        program=program,
        initial_counters=start,
        alphabet=alphabet,
        signature=signature,
        provenance=dict(provenance),
    ).validate()


def test_a_compiled_program_reproduces_its_machine_exactly() -> None:
    """The certificate the signature is supposed to be.

    Signatures are taken from the compiled program rather than the hypothesis,
    so a compiler that quietly changed behaviour would show up here rather
    than as an unexplained recognition failure much later.
    """

    stream = canonical_signature_stream(4)
    for state_count in range(1, 7):
        rule = _rule(state_count)
        record = _record(rule)
        assert record.signature == tuple(rule.expected(list(stream)))


def test_the_signature_stream_depends_only_on_the_alphabet() -> None:
    assert canonical_signature_stream(4) == canonical_signature_stream(4)
    assert canonical_signature_stream(4) != canonical_signature_stream(5)
    assert set(canonical_signature_stream(3)) <= {0, 1, 2}


def test_appending_never_disturbs_what_is_already_there() -> None:
    """The property the whole store exists for."""

    library = InducedProgramLibrary(alphabet=4)
    first = _record(_rule(2))
    slot = library.append(first)
    before = library.record(slot).digest()
    for state_count in (3, 4, 5, 6):
        library.append(_record(_rule(state_count)))
    assert library.record(slot).digest() == before
    assert library.record_count == 5


def test_a_behavioural_duplicate_is_detected_without_executing_anything() -> None:
    library = InducedProgramLibrary(alphabet=4)
    rule = _rule(3)
    slot = library.append(_record(rule, note="first"))
    # A differently annotated compilation of the same rule presses identically.
    twin = _record(rule, note="second")
    assert twin.digest() != library.record(slot).digest()
    assert library.duplicate_of(twin.signature) == slot
    assert library.duplicate_of(_record(_rule(4)).signature) is None


def test_nearest_ranks_by_signature_and_bounds_what_gets_executed() -> None:
    library = InducedProgramLibrary(alphabet=4)
    for state_count in (1, 2, 3, 4, 5, 6):
        library.append(_record(_rule(state_count)))
    target = library.record(3).signature
    ranked = library.nearest(target, limit=3)
    assert len(ranked) == 3
    assert ranked[0] == (3, 0)
    assert [distance for _, distance in ranked] == sorted(
        distance for _, distance in ranked
    )


def test_signatures_over_different_streams_are_refused() -> None:
    with pytest.raises(ValueError, match="not comparable"):
        signature_distance((0, 1), (0, 1, 0))


def test_a_library_round_trips_through_disk_with_its_checksum(tmp_path) -> None:
    library = InducedProgramLibrary(alphabet=4, frontend_digest="a" * 64)
    for state_count in (2, 4):
        library.append(_record(_rule(state_count), states=state_count))
    path = tmp_path / "induced.library"
    library.save(path)
    assert path.with_suffix(path.suffix + ".sha256").is_file()
    loaded = InducedProgramLibrary.load(path)
    assert loaded.digest() == library.digest()
    assert loaded.record_count == 2
    assert loaded.record(0).provenance["states"] == 2


def test_a_tampered_library_refuses_to_load(tmp_path) -> None:
    """The failure this store must never have: silent history rewriting."""

    library = InducedProgramLibrary(alphabet=4)
    library.append(_record(_rule(3)))
    path = tmp_path / "induced.library"
    library.save(path)

    payload = json.loads(path.read_text())
    payload["records"][0]["provenance"]["forged"] = True
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="file checksum mismatch"):
        InducedProgramLibrary.load(path)

    # Repairing the sidecar is not enough: the payload carries its own digest.
    library.save(path)
    payload = json.loads(path.read_text())
    payload["records"][0]["provenance"]["forged"] = True
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text(text)
    import hashlib

    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        InducedProgramLibrary.load(path)


def test_a_missing_checksum_is_not_a_loadable_library(tmp_path) -> None:
    library = InducedProgramLibrary(alphabet=4)
    library.append(_record(_rule(2)))
    path = tmp_path / "induced.library"
    library.save(path)
    path.with_suffix(path.suffix + ".sha256").unlink()
    with pytest.raises(ValueError, match="checksum is missing"):
        InducedProgramLibrary.load(path)


def test_a_program_over_the_wrong_alphabet_is_refused() -> None:
    library = InducedProgramLibrary(alphabet=4)
    rule = sample_rule(symbol_count=3, state_count=2, seed=6202)
    assert rule is not None
    with pytest.raises(ValueError, match="alphabet does not match"):
        library.append(_record(rule, alphabet=3))


def test_degenerate_construction_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least two symbols"):
        InducedProgramLibrary(alphabet=1)
    with pytest.raises(ValueError, match="SHA-256"):
        InducedProgramLibrary(alphabet=4, frontend_digest="short")
    library = InducedProgramLibrary(alphabet=4)
    with pytest.raises(ValueError, match="extension"):
        library.save(tmp_path / "induced.json")
    with pytest.raises(IndexError):
        library.record(0)
