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
