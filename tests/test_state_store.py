import pytest
import torch
from torch import nn

from neural_computer import PersistentOpaqueStateStore


def _configuration() -> dict[str, object]:
    return {
        "component": "opaque-route-policy",
        "schema": "neural-computer.opaque-address-router.v1",
        "width": 4,
        "hidden": 8,
    }


def test_persistent_opaque_state_store_round_trips_a_replaceable_module(tmp_path) -> None:
    source = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))
    path = tmp_path / "route-state.pt"
    store = PersistentOpaqueStateStore(path, configuration=_configuration())

    digest = store.save_module(source)
    restored = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))
    restored_digest = store.load_module(restored)

    assert digest == restored_digest
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in source.state_dict().items()
    )


def test_persistent_opaque_state_store_rejects_corruption_without_mutating_module(
    tmp_path,
) -> None:
    source = nn.Linear(4, 2)
    path = tmp_path / "route-state.pt"
    store = PersistentOpaqueStateStore(path, configuration=_configuration())
    store.save_module(source)

    payload = torch.load(path, weights_only=False)
    payload["state_dict"]["weight"][0, 0] += 1.0
    torch.save(payload, path)
    restored = nn.Linear(4, 2)
    before = {
        name: value.detach().clone() for name, value in restored.state_dict().items()
    }

    with pytest.raises(ValueError, match="checksum"):
        store.load_module(restored)

    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in before.items())


def test_persistent_opaque_state_store_rejects_configuration_and_shape_mismatch(
    tmp_path,
) -> None:
    path = tmp_path / "route-state.pt"
    source = nn.Linear(4, 2)
    store = PersistentOpaqueStateStore(path, configuration=_configuration())
    store.save_module(source)

    mismatched_configuration = PersistentOpaqueStateStore(
        path,
        configuration={**_configuration(), "hidden": 16},
    )
    with pytest.raises(ValueError, match="configuration"):
        mismatched_configuration.load()

    mismatched_module = nn.Linear(4, 3)
    with pytest.raises(ValueError, match="shape"):
        store.load_module(mismatched_module)


def test_persistent_opaque_state_store_rejects_nonfinite_state(tmp_path) -> None:
    store = PersistentOpaqueStateStore(
        tmp_path / "route-state.pt",
        configuration=_configuration(),
    )

    with pytest.raises(ValueError, match="finite"):
        store.save({"weight": torch.tensor([float("nan")])})
