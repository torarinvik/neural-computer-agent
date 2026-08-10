import pytest
import torch

from neural_computer import (
    ExternalEntryBindingRepertoire,
    ExternalEntryRepertoire,
)


def test_external_entry_repertoire_grows_proposes_and_persists() -> None:
    repertoire = ExternalEntryRepertoire(2, merge_cosine=0.99)
    first = repertoire.observe(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        utility=torch.tensor([1.0, 0.0]),
        propensity=torch.tensor([0.5, 1.0]),
        timestamp=torch.tensor([7, 8]),
    )
    assert first.added == (True, True)

    duplicate = repertoire.observe(
        torch.tensor([[2.0, 0.0]]),
        utility=0.25,
        propensity=0.25,
        timestamp=9,
    )
    assert duplicate.entry_indices == (0,)
    assert duplicate.added == (False,)
    assert repertoire.record_count == 2
    stats = repertoire.statistics()
    assert stats["attempts"].tolist() == [2, 1]
    assert stats["outcome_counts"].tolist() == [2, 1]
    assert torch.allclose(
        stats["inverse_propensity_utility_sums"],
        torch.tensor([3.0, 0.0], dtype=torch.float64),
    )

    proposal = repertoire.propose(device="cpu")
    assert proposal.entries.shape == (2, 2)
    assert proposal.source_indices == (0, 1)
    assert torch.allclose(proposal.propensities, torch.full((2,), 0.5))

    restored = ExternalEntryRepertoire.from_payload(repertoire.payload())
    assert restored.content_digest() == repertoire.content_digest()
    corrupt = repertoire.payload()
    corrupt["entries"] = corrupt["entries"].clone()
    corrupt["entries"][0, 0] += 0.1
    with pytest.raises(ValueError, match="checksum"):
        ExternalEntryRepertoire.from_payload(corrupt)


def test_external_entry_admission_is_copy_on_write_and_retention_safe() -> None:
    repertoire = ExternalEntryRepertoire(2)
    repertoire.observe(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    candidate_entry = torch.tensor([0.5, 0.5])

    accepted = repertoire.admit_verified(
        candidate_entry,
        lambda candidate: bool(
            candidate.observe(candidate_entry, utility=1.0).outcome_observed
        ),
    )
    assert accepted.accepted
    assert accepted.entry_index == 2
    assert repertoire.record_count == 3

    rejected_digest = repertoire.content_digest()
    rejected = repertoire.admit_verified(
        torch.tensor([0.25, -0.75]),
        lambda _candidate: False,
    )
    assert not rejected.accepted
    assert repertoire.content_digest() == rejected_digest

    mutation_rejected = repertoire.admit_verified(
        torch.tensor([-0.5, 0.5]),
        lambda candidate: bool(
            candidate.observe(torch.tensor([1.0, 0.0])).record_count == 3
        ),
    )
    assert not mutation_rejected.accepted
    assert repertoire.content_digest() == rejected_digest


def test_external_entry_repertoire_keeps_opposite_polarities_distinct() -> None:
    repertoire = ExternalEntryRepertoire(1)
    repertoire.observe(torch.tensor([[1.0], [-1.0], [0.0]]))
    assert repertoire.record_count == 3
    proposal = repertoire.propose()
    assert torch.equal(proposal.entries, torch.tensor([[1.0], [-1.0], [0.0]]))


def test_external_entry_binding_repertoire_keeps_pairs_atomic() -> None:
    repertoire = ExternalEntryBindingRepertoire(2, 1)
    first = repertoire.observe(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0], [-1.0]]),
        utility=torch.tensor([1.0, 0.0]),
    )
    assert first.added == (True, True)
    proposal = repertoire.propose()
    assert torch.equal(
        proposal.intentions,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert torch.equal(proposal.entries, torch.tensor([[1.0], [-1.0]]))
    assert proposal.source_indices == (0, 1)

    candidate_intention = torch.tensor([0.5, 0.5])
    candidate_entry = torch.tensor([1.0])
    accepted = repertoire.admit_verified(
        candidate_intention,
        candidate_entry,
        lambda candidate: bool(
            candidate.observe(
                candidate_intention,
                candidate_entry,
                utility=1.0,
            ).outcome_observed
        ),
    )
    assert accepted.accepted
    assert accepted.entry_index == 2
    assert repertoire.record_count == 3
    assert repertoire.logical_ids == (0, 1, 2)

    restored = ExternalEntryBindingRepertoire.from_payload(repertoire.payload())
    assert restored.content_digest() == repertoire.content_digest()
    assert restored.logical_ids == repertoire.logical_ids
    corrupt = repertoire.payload()
    corrupt["store"] = dict(corrupt["store"])
    corrupt["store"]["entries"] = corrupt["store"]["entries"].clone()
    corrupt["store"]["entries"][0, 0] += 0.1
    with pytest.raises(ValueError, match="checksum"):
        ExternalEntryBindingRepertoire.from_payload(corrupt)
