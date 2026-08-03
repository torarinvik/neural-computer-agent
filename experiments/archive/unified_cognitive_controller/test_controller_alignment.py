from .audit_controller_alignment import summarize_gates


def test_repertoire_requires_every_requested_capability() -> None:
    summary = summarize_gates({
        "working_memory": {"passed": True},
        "persistent_memory": {"passed": False},
    })
    assert summary["passed_count"] == 1
    assert summary["requested_count"] == 2
    assert not summary["one_controller_repertoire_passed"]


def test_repertoire_passes_only_for_one_complete_checkpoint() -> None:
    summary = summarize_gates({
        "working_memory": {"passed": True},
        "persistent_memory": {"passed": True},
    })
    assert summary["one_controller_repertoire_passed"]
