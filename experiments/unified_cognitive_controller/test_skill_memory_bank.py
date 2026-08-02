import torch

from .skill_memory_bank import SkillArtifactBank


def test_skill_bank_roundtrip_promotion_and_eviction(tmp_path):
    bank = SkillArtifactBank(tmp_path / "bank", width=4, capacity=2)
    real = {"schema": "test", "value": torch.arange(3)}
    decoy = {"schema": "test", "value": torch.zeros(3)}
    real_index = bank.put(torch.tensor([1.0, 0.0, 0.0, 0.0]), real,
                          name="real.pt")
    bank.put(torch.tensor([0.0, 1.0, 0.0, 0.0]), decoy, name="decoy.pt")
    bank.save()

    restored = SkillArtifactBank.load(tmp_path / "bank")
    assert restored.artifact_sha256[real_index]
    index, confidence, artifact = restored.promote(
        torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert index == real_index
    assert confidence == 1.0
    assert torch.equal(artifact["value"], real["value"])
    assert restored.hot_indices == (real_index,)
    assert int(restored.memory.store.access_count[real_index]) == 1
    restored.evict_hot(real_index)
    assert restored.hot_indices == ()


def test_skill_bank_replaces_least_used_row(tmp_path):
    bank = SkillArtifactBank(tmp_path / "bank", width=2, capacity=1)
    bank.put(torch.tensor([1.0, 0.0]), {"value": torch.tensor(1)},
            name="first.pt")
    bank.put(torch.tensor([0.0, 1.0]), {"value": torch.tensor(2)},
            name="second.pt")
    index, _, artifact = bank.promote(torch.tensor([0.0, 1.0]))
    assert index == 0
    assert int(artifact["value"]) == 2


def test_skill_bank_rejects_tampered_cold_artifact(tmp_path):
    bank = SkillArtifactBank(tmp_path / "bank", width=2, capacity=1)
    bank.put(torch.tensor([1.0, 0.0]), {"value": torch.tensor(1)},
            name="skill.pt")
    bank.save()

    artifact_path = tmp_path / "bank" / "skill.pt"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
    restored = SkillArtifactBank.load(tmp_path / "bank")
    try:
        restored.promote(torch.tensor([1.0, 0.0]))
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered artifact was promoted")


def test_skill_bank_can_abstain_on_ambiguous_address(tmp_path):
    bank = SkillArtifactBank(tmp_path / "bank", width=2, capacity=2)
    bank.put(torch.tensor([1.0, 0.0]), {"value": torch.tensor(1)},
            name="horizontal.pt")
    bank.put(torch.tensor([0.0, 1.0]), {"value": torch.tensor(2)},
            name="vertical.pt")
    bank.save()
    restored = SkillArtifactBank.load(tmp_path / "bank")
    try:
        restored.promote(torch.tensor([1.0, 1.0]), min_margin=0.1)
    except LookupError as error:
        assert "abstaining" in str(error)
        assert int(restored.memory.store.access_count.sum()) == 0
    else:
        raise AssertionError("ambiguous address selected a skill")


def test_skill_bank_can_abstain_on_low_confidence_address(tmp_path):
    bank = SkillArtifactBank(tmp_path / "bank", width=2, capacity=1)
    bank.put(torch.tensor([1.0, 0.0]), {"value": torch.tensor(1)},
            name="skill.pt")
    bank.save()
    restored = SkillArtifactBank.load(tmp_path / "bank")
    try:
        restored.promote(torch.tensor([-1.0, 0.0]), min_confidence=0.0)
    except LookupError as error:
        assert "abstaining" in str(error)
        assert int(restored.memory.store.access_count.sum()) == 0
    else:
        raise AssertionError("low-confidence address selected a skill")
