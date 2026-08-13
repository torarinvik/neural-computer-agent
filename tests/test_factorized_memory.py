from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_computer import (
    MemoryCandidates,
    MemoryQuery,
    PersistentSharedBasisContentAddressedMemory,
    SharedBasisContentAddressedMemory,
)


def _payloads(seed: int = 17) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(12, 8, generator=generator)
    latent = torch.randn(12, 2, generator=generator)
    shared = torch.randn(2, 8, generator=generator)
    shared = torch.linalg.qr(shared.transpose(0, 1)).Q[:, :2].transpose(0, 1)
    noise = 1e-3 * torch.randn(12, 8, generator=generator)
    return keys, latent @ shared + noise


def _write(
    memory: SharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    receipt = memory.write(
        keys,
        values,
        torch.ones(keys.shape[0]),
    )
    assert bool(receipt.committed.all())


def _routes_match(
    memory: SharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    tolerance: float,
) -> bool:
    for key, expected in zip(keys, values, strict=True):
        read = memory.read(MemoryQuery(key.reshape(1, -1)))
        if not bool(read.hit.item()):
            return False
        if not torch.allclose(read.value[0], expected, atol=tolerance, rtol=0.0):
            return False
    return True


def test_shared_basis_grows_without_losing_independent_routes() -> None:
    keys, values = _payloads()
    memory = SharedBasisContentAddressedMemory(
        8,
        write_threshold=0.0,
        write_match_threshold=0.999,
        basis_tolerance=1e-8,
    )

    _write(memory, keys, values)

    assert memory.record_count == 12
    assert memory.basis_count == 8
    assert memory.dense_value_scalar_count == 96
    assert memory.physical_value_scalar_count == 160
    assert _routes_match(memory, keys, values, tolerance=1e-6)


def test_shared_basis_compression_is_copy_on_write_and_verifier_gated() -> None:
    keys, values = _payloads()
    memory = SharedBasisContentAddressedMemory(
        8,
        write_threshold=0.0,
        write_match_threshold=0.999,
        basis_tolerance=1e-8,
    )
    _write(memory, keys, values)
    before = {name: tensor.detach().clone() for name, tensor in memory.state_dict().items()}
    version = int(memory.store_version.item())

    lossy = memory.compression_candidate(1)
    rejected = memory.replace_from_candidate(
        lossy,
        expected_version=version,
        retention_probe=lambda _candidate: False,
    )

    assert not rejected.accepted
    assert int(memory.store_version.item()) == version
    for name, tensor in before.items():
        assert torch.equal(memory.state_dict()[name], tensor)

    candidate = memory.compression_candidate(2)
    assert memory.max_value_error(candidate) < 0.01
    accepted = memory.replace_from_candidate(
        candidate,
        expected_version=version,
        retention_probe=lambda current: _routes_match(
            current,
            keys,
            values,
            tolerance=0.01,
        ),
    )

    assert accepted.accepted
    assert accepted.basis_rows_before == 8
    assert accepted.basis_rows_after == 2
    assert accepted.rows_before == accepted.rows_after == 12
    assert memory.physical_value_scalar_count < memory.dense_value_scalar_count
    assert _routes_match(memory, keys, values, tolerance=0.01)
    with pytest.raises(RuntimeError, match="stale"):
        memory.replace_from_candidate(
            memory.compression_candidate(2),
            expected_version=version,
        )


def test_persistent_shared_basis_memory_reloads_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    keys, values = _payloads(18)
    path = tmp_path / "shared-basis-memory.pt"
    memory = PersistentSharedBasisContentAddressedMemory(
        8,
        path,
        write_threshold=0.0,
        write_match_threshold=0.999,
        basis_tolerance=1e-8,
    )
    _write(memory, keys, values)
    candidate = memory.compression_candidate(2)
    receipt = memory.replace_from_candidate(
        candidate,
        expected_version=int(memory.store_version.item()),
        retention_probe=lambda current: _routes_match(
            current,
            keys,
            values,
            tolerance=0.01,
        ),
    )
    assert receipt.accepted

    restored = PersistentSharedBasisContentAddressedMemory(
        8,
        path,
        write_threshold=0.0,
        write_match_threshold=0.999,
        basis_tolerance=1e-8,
    )
    assert restored.record_count == 12
    assert restored.basis_count == 2
    assert _routes_match(restored, keys, values, tolerance=0.01)

    payload = torch.load(path, weights_only=False)
    payload["state_dict"]["coefficients"] = payload["state_dict"][
        "coefficients"
    ].clone()
    payload["state_dict"]["coefficients"][0, 0] += 0.1
    corrupt_path = tmp_path / "corrupt-shared-basis-memory.pt"
    torch.save(payload, corrupt_path)
    with pytest.raises(ValueError, match="checksum"):
        PersistentSharedBasisContentAddressedMemory(
            8,
            corrupt_path,
            write_threshold=0.0,
            write_match_threshold=0.999,
            basis_tolerance=1e-8,
        )


def test_persistent_shared_basis_memory_rolls_back_on_snapshot_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys, values = _payloads(19)
    path = tmp_path / "rollback-shared-basis-memory.pt"
    memory = PersistentSharedBasisContentAddressedMemory(
        8,
        path,
        write_threshold=0.0,
        write_match_threshold=0.999,
        basis_tolerance=1e-8,
    )
    _write(memory, keys[:1], values[:1])
    before = {name: tensor.detach().clone() for name, tensor in memory.state_dict().items()}

    def fail_snapshot(_path: Path) -> None:
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(memory, "snapshot", fail_snapshot)
    with pytest.raises(OSError, match="persistence"):
        memory.write(keys[1:2], values[1:2], torch.ones(1))

    for name, tensor in before.items():
        assert torch.equal(memory.state_dict()[name], tensor)


def test_shared_basis_rewrite_replaces_logical_rows_verifier_gated() -> None:
    keys, values = _payloads(23)
    memory = SharedBasisContentAddressedMemory(
        8,
        write_threshold=0.0,
        write_match_threshold=0.999,
        basis_tolerance=1e-8,
    )
    _write(memory, keys[:6], values[:6])
    replacement = MemoryCandidates(
        keys=keys[6:].unsqueeze(0),
        values=values[6:].unsqueeze(0),
        strengths=torch.ones(1, 6),
        timestamps=torch.zeros(1, 6),
        occupied=torch.ones(1, 6, dtype=torch.bool),
    )
    candidate = memory.rewrite_candidate(replacement, basis_rows=2)
    version = int(memory.store_version.item())
    before = {
        name: tensor.detach().clone() for name, tensor in memory.state_dict().items()
    }

    rejected = memory.replace_from_rewrite_candidate(
        candidate,
        expected_version=version,
        retention_probe=lambda _candidate: False,
    )
    assert not rejected.accepted
    assert int(memory.store_version.item()) == version
    for name, tensor in before.items():
        assert torch.equal(memory.state_dict()[name], tensor)

    accepted = memory.replace_from_rewrite_candidate(
        candidate,
        expected_version=version,
        retention_probe=lambda current: _routes_match(
            current,
            keys[6:],
            values[6:],
            tolerance=0.01,
        ),
    )
    assert accepted.accepted
    assert accepted.rows_before == accepted.rows_after == 6
    assert accepted.basis_rows_after == 2
    assert _routes_match(memory, keys[6:], values[6:], tolerance=0.01)
    with pytest.raises(RuntimeError, match="stale"):
        memory.replace_from_rewrite_candidate(
            candidate,
            expected_version=version,
        )
