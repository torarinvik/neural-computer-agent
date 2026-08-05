from __future__ import annotations

import pytest
import torch
from torch import nn

from neural_computer import (
    compose_growth_artifacts,
    freeze_core,
    load_growth_artifact,
)


class _Processor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = nn.Linear(3, 3)
        self.growth = nn.Linear(3, 3)


def test_growth_loader_changes_only_declared_external_state() -> None:
    processor = _Processor()
    freeze_core(processor, ("growth.",))
    core_before = {name: value.detach().clone() for name, value in processor.core.state_dict().items()}
    artifact = {
        f"growth.{name}": torch.ones_like(value)
        for name, value in processor.growth.state_dict().items()
    }

    receipt = load_growth_artifact(
        processor, artifact, growth_prefixes=("growth.",)
    )

    assert receipt.core_unchanged
    assert all(not parameter.requires_grad for name, parameter in processor.named_parameters() if name.startswith("core."))
    assert all(parameter.requires_grad for name, parameter in processor.named_parameters() if name.startswith("growth."))
    for name, value in processor.core.state_dict().items():
        assert torch.equal(value, core_before[name])
    for value in processor.growth.state_dict().values():
        assert torch.equal(value, torch.ones_like(value))


def test_growth_loader_rejects_core_entries_and_shape_mismatches() -> None:
    processor = _Processor()
    with pytest.raises(ValueError, match="outside the growth boundary"):
        load_growth_artifact(
            processor,
            {"core.weight": processor.core.weight.detach().clone()},
            growth_prefixes=("growth.",),
        )
    with pytest.raises(ValueError, match="wrong shape"):
        load_growth_artifact(
            processor,
            {"growth.weight": torch.zeros(2, 2)},
            growth_prefixes=("growth.",),
        )


def test_growth_composition_remaps_disjoint_namespaces_and_clones_values() -> None:
    first = {"skill.0.weight": torch.tensor([[1.0]])}
    second = {"skill.0.weight": torch.tensor([[2.0]])}

    composed = compose_growth_artifacts(
        (first, second),
        prefix_maps=(
            {"skill.0.": "skill.0."},
            {"skill.0.": "skill.1."},
        ),
    )

    assert set(composed) == {"skill.0.weight", "skill.1.weight"}
    assert torch.equal(composed["skill.0.weight"], first["skill.0.weight"])
    assert torch.equal(composed["skill.1.weight"], second["skill.0.weight"])
    assert composed["skill.0.weight"].device.type == "cpu"
    assert composed["skill.0.weight"] is not first["skill.0.weight"]


def test_growth_composition_rejects_namespace_collisions() -> None:
    with pytest.raises(ValueError, match="namespace collision"):
        compose_growth_artifacts(
            ({"growth.weight": torch.ones(1)}, {"growth.weight": torch.zeros(1)})
        )
