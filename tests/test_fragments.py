import pytest
import torch

from neural_computer import (
    EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA,
    EXTERNAL_SKILL_FRAGMENT_COMPOSITION_SCHEMA,
    ExternalCapabilityRegisterMachine,
    ExternalSkillFragmentArtifact,
    ExternalSkillFragmentBank,
)


def _bank() -> ExternalSkillFragmentBank:
    torch.manual_seed(441)
    bank = ExternalSkillFragmentBank(
        instruction_width=6,
        basis_count=4,
        key_width=6,
        router_hidden=12,
        max_fragment_steps=4,
    )
    with torch.no_grad():
        bank.shared_basis.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10.0)
    bank.add_fragment(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    bank.add_fragment(
        torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    bank.add_fragment(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    return bank


def test_fragment_artifact_round_trips_with_integrity_digest() -> None:
    artifact = ExternalSkillFragmentArtifact(
        coefficients=torch.eye(3),
        key=torch.ones(5),
    )

    restored = ExternalSkillFragmentArtifact.from_payload(artifact.payload())

    assert restored.configuration() == artifact.configuration()
    assert restored.digest() == artifact.digest()
    assert torch.equal(restored.coefficients, artifact.coefficients)


def test_fragment_bank_routes_and_composes_reusable_shared_basis_data() -> None:
    bank = _bank()
    queries = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    route = bank.route(queries)
    composition = bank.compose_queries(queries.unsqueeze(0))

    assert bank.configuration()["schema"] == EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA
    assert route.indices[:, 0].tolist() == [0, 1]
    assert composition.schema == EXTERNAL_SKILL_FRAGMENT_COMPOSITION_SCHEMA
    assert composition.fragment_indices.tolist() == [[0, 1]]
    expected = torch.cat((bank.fragment_codes(0), bank.fragment_codes(1)), dim=0)
    assert torch.allclose(composition.codes[0, : expected.shape[0]], expected)
    assert composition.mask[0, : expected.shape[0]].all()


def test_fragment_route_is_permutation_equivariant_and_has_no_task_identity() -> None:
    bank = _bank()
    # Build the permuted bank through the public artifact boundary.  This avoids
    # exposing or relying on a semantic slot label in the route query.
    permuted = ExternalSkillFragmentBank(
        instruction_width=bank.instruction_width,
        basis_count=bank.basis_count,
        key_width=bank.key_width,
        router_hidden=bank.router_hidden,
        max_fragment_steps=bank.max_fragment_steps,
    )
    permuted.shared_basis.data.copy_(bank.shared_basis.data)
    permuted.router.load_state_dict(bank.router.state_dict())
    for index in (2, 1, 0):
        artifact = bank.artifact(index)
        permuted.add_artifact(artifact)

    query = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    original_digest = bank.artifact(int(bank.route(query).indices[0, 0])).digest()
    permuted_digest = permuted.artifact(int(permuted.route(query).indices[0, 0])).digest()

    assert original_digest == permuted_digest


def test_fragment_bank_learned_router_uses_only_outcome_pairs_and_grows_without_resize() -> None:
    bank = _bank()
    basis_shape = tuple(bank.shared_basis.shape)
    bank.enable_learned_routing()
    query = torch.stack((bank.keys[0].detach(), bank.keys[1].detach()))
    outcomes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    loss, comparisons = bank.outcome_ranking_loss(query, outcomes)
    loss.backward()
    before = bank.artifact(0).digest()
    bank.add_fragment(torch.zeros(1, 4), torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

    assert comparisons > 0
    assert bank.router.query_encoder[0].weight.grad is not None
    assert tuple(bank.shared_basis.shape) == basis_shape
    assert bank.artifact(0).digest() == before
    assert bank.fragment_count == 4


def test_fragment_bank_grows_shared_basis_without_changing_old_fragment_codes() -> None:
    bank = _bank()
    bank.freeze_basis_prefix(bank.basis_count)
    before = bank.fragment_codes(0).detach().clone()
    old_basis_count = bank.basis_count

    new_rows = bank.grow_basis(2)

    assert new_rows == (old_basis_count, old_basis_count + 1)
    assert bank.basis_count == old_basis_count + 2
    assert bank.shared_basis.shape == (old_basis_count + 2, bank.instruction_width)
    assert bank.coefficients[0].shape == (2, old_basis_count + 2)
    assert torch.allclose(bank.fragment_codes(0), before)
    bank.shared_basis.grad = None
    loss = bank.shared_basis.square().sum()
    loss.backward()
    assert torch.allclose(
        bank.shared_basis.grad[:old_basis_count],
        torch.zeros_like(bank.shared_basis.grad[:old_basis_count]),
    )
    assert bank.shared_basis.grad[old_basis_count:].abs().sum() > 0


def test_fragment_bank_persistence_preserves_basis_routes_and_composition() -> None:
    bank = _bank()
    bank.enable_learned_routing()
    payload = bank.payload()
    restored = ExternalSkillFragmentBank.from_payload(payload)
    query = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert restored.payload()["sha256"] == payload["sha256"]
    assert restored.route(query).indices.tolist() == bank.route(query).indices.tolist()
    assert torch.allclose(
        restored.compose_queries(query.unsqueeze(0)).codes,
        bank.compose_queries(query.unsqueeze(0)).codes,
    )


def test_fragment_bank_persistence_restores_expandable_basis_protection(tmp_path) -> None:
    bank = _bank()
    old_codes = bank.fragment_codes(0).detach().clone()
    bank.grow_basis(2)
    bank.freeze_basis_prefix(4)
    path = tmp_path / "expanded-bank.pt"

    bank.save(path)
    restored = ExternalSkillFragmentBank.load(path)

    assert restored.basis_count == 6
    assert restored.payload()["state"]["frozen_basis_rows"] == 4
    assert torch.allclose(restored.fragment_codes(0), old_codes)
    restored.shared_basis.grad = None
    restored.shared_basis.square().sum().backward()
    assert torch.allclose(
        restored.shared_basis.grad[:4],
        torch.zeros_like(restored.shared_basis.grad[:4]),
    )


def test_fragment_bank_disk_persistence_is_atomic_and_restartable(tmp_path) -> None:
    bank = _bank()
    bank.protect(1)
    bank.enable_learned_routing()
    path = tmp_path / "memory" / "fragment-bank.pt"

    digest = bank.save(path)
    restored = ExternalSkillFragmentBank.load(path)

    assert path.is_file()
    assert digest == bank.payload()["sha256"]
    assert restored.payload()["sha256"] == digest
    assert restored.protection_mask().tolist() == [False, True, False]
    assert restored.logical_ids == bank.logical_ids


def test_fragment_bank_checksum_covers_lifecycle_metadata() -> None:
    payload = _bank().payload()
    payload["state"]["protected"][0] = True

    with pytest.raises(ValueError, match="checksum"):
        ExternalSkillFragmentBank.from_payload(payload)


def test_fragment_bank_rejects_checksum_corruption_before_mutation() -> None:
    payload = _bank().payload()
    payload["state"]["fragments"][0]["coefficients"] = payload["state"][
        "fragments"
    ][0]["coefficients"].clone()
    payload["state"]["fragments"][0]["coefficients"][0, 0] += 1.0

    try:
        ExternalSkillFragmentBank.from_payload(payload)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("expected fragment-bank checksum rejection")


def test_shared_register_interpreter_executes_variable_length_fragment_chain() -> None:
    bank = _bank()
    composition = bank.compose_indices(torch.tensor([[0, 1], [1, 2]]))
    machine = ExternalCapabilityRegisterMachine(
        event_width=1,
        action_width=1,
        intention_width=2,
        register_width=6,
        instruction_width=6,
        interpreter_hidden=8,
    )

    result = machine.execute_fragment_composition(
        torch.zeros(2, 6),
        composition,
    )

    assert result.shape == (2, 6)
    assert torch.isfinite(result).all()


def test_shared_register_interpreter_rejects_forged_out_of_bank_fragment() -> None:
    bank = _bank()
    composition = bank.compose_indices(torch.tensor([[0, 1]]))
    forged = type(composition)(
        fragment_indices=torch.tensor([[0, 3]], dtype=torch.int64),
        route_scores=composition.route_scores,
        codes=composition.codes,
        mask=composition.mask,
        bank_fragment_count=composition.bank_fragment_count,
    )
    machine = ExternalCapabilityRegisterMachine(
        event_width=1,
        action_width=1,
        intention_width=2,
        register_width=6,
        instruction_width=6,
        interpreter_hidden=8,
    )

    with pytest.raises(ValueError, match="outside the bank"):
        machine.execute_fragment_composition(torch.zeros(1, 6), forged)
