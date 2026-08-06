from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    _load_stack_artifact,
    _stack_artifact,
)
from experiments.generated_composition_capability_amodal.train_pipeline import (
    _new_stack,
)
from experiments.parent_conditioned_artifact_bank_amodal.train import _new_capability
from neural_computer import (
    compose_growth_artifacts,
    compress_growth_artifact,
    decompress_growth_artifact,
    freeze_core,
    load_growth_artifact,
    select_growth_artifact_view,
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


def test_composition_artifact_loader_infers_expanded_slot_count() -> None:
    stack = _new_stack(seed=17, program_count=3, stack="routed")
    _unused_program, decoder = _new_capability(18)
    artifact = _stack_artifact(stack, decoder)

    restored_stack, restored_decoder = _load_stack_artifact(artifact)

    assert len(restored_stack.programs) == 3
    for name, value in stack.state_dict().items():
        assert torch.equal(restored_stack.state_dict()[name], value)
    for name, value in decoder.state_dict().items():
        assert torch.equal(restored_decoder.state_dict()[name], value)


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


def test_growth_view_projects_one_opaque_namespace() -> None:
    artifact = {
        "growth_slots.0.weight": torch.ones(1, 1),
        "growth_slots.1.weight": torch.full((1, 1), 2.0),
    }
    selected = select_growth_artifact_view(
        artifact,
        source_prefix="growth_slots.1.",
    )
    assert set(selected) == {"growth_slots.0.weight"}
    assert torch.equal(selected["growth_slots.0.weight"], torch.full((1, 1), 2.0))


def test_growth_compression_casts_floating_payloads_without_changing_names() -> None:
    artifact = {
        "growth.weight": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
        "growth.index": torch.tensor([1, 2], dtype=torch.int64),
    }

    compressed = compress_growth_artifact(artifact)

    assert set(compressed) == set(artifact)
    assert compressed["growth.weight"].dtype == torch.float16
    assert compressed["growth.index"].dtype == torch.int64
    assert compressed["growth.weight"].numel() * compressed["growth.weight"].element_size() < artifact["growth.weight"].numel() * artifact["growth.weight"].element_size()
    assert torch.equal(compressed["growth.index"], artifact["growth.index"])


def test_growth_compression_can_preserve_verifier_selected_tensors() -> None:
    artifact = {
        "growth.weight": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
        "growth.bias": torch.tensor([0.25], dtype=torch.float32),
    }

    compressed = compress_growth_artifact(
        artifact,
        preserve_names=("growth.bias",),
    )

    assert compressed["growth.weight"].dtype == torch.float16
    assert compressed["growth.bias"].dtype == torch.float32
    assert torch.equal(compressed["growth.bias"], artifact["growth.bias"])


def test_growth_compression_supports_loss_aware_dtype_overrides() -> None:
    artifact = {
        "growth.critical": torch.linspace(-1.0, 1.0, 16),
        "growth.other": torch.linspace(-2.0, 2.0, 16),
    }

    compressed = compress_growth_artifact(
        artifact,
        preserve_names=("growth.critical",),
        dtype_overrides={"growth.other": torch.int8},
    )

    assert compressed["growth.critical"].dtype == torch.float32
    assert compressed["growth.other"].dtype == torch.int8
    restored = decompress_growth_artifact(compressed)
    assert torch.equal(restored["growth.critical"], artifact["growth.critical"])
    assert torch.allclose(restored["growth.other"], artifact["growth.other"], atol=0.02)


def test_growth_loader_can_explicitly_cast_compressed_growth_state() -> None:
    processor = _Processor()
    compressed = {
        name: value.to(torch.float16)
        for name, value in {
            f"growth.{name}": tensor
            for name, tensor in processor.growth.state_dict().items()
        }.items()
    }

    receipt = load_growth_artifact(
        processor,
        compressed,
        growth_prefixes=("growth.",),
        allow_dtype_cast=True,
    )

    assert receipt.core_unchanged


def test_growth_loader_rejects_dtype_cast_by_default() -> None:
    processor = _Processor()
    compressed = {
        f"growth.{name}": tensor.to(torch.float16)
        for name, tensor in processor.growth.state_dict().items()
    }

    with pytest.raises(ValueError, match="wrong dtype"):
        load_growth_artifact(
            processor,
            compressed,
            growth_prefixes=("growth.",),
        )


def test_growth_int8_codec_round_trips_with_explicit_scales() -> None:
    artifact = {
        "growth.weight": torch.tensor([[1.25, -2.5]], dtype=torch.float32),
        "growth.bias": torch.tensor([0.0], dtype=torch.float32),
    }

    compressed = compress_growth_artifact(artifact, dtype=torch.int8)
    restored = decompress_growth_artifact(compressed)

    assert compressed["growth.weight"].dtype == torch.int8
    assert "growth.weight.__scale__" in compressed
    assert restored["growth.weight"].dtype == torch.float32
    assert torch.allclose(
        restored["growth.weight"], artifact["growth.weight"], atol=0.02
    )
    assert torch.equal(restored["growth.bias"], artifact["growth.bias"])


def test_growth_int8_codec_rejects_unscaled_int8_entries() -> None:
    with pytest.raises(ValueError, match="missing its scale"):
        decompress_growth_artifact(
            {"growth.weight": torch.ones(1, dtype=torch.int8)}
        )


def test_growth_int4_codec_packs_and_restores_row_scaled_tensors() -> None:
    artifact = {
        "growth.weight": torch.tensor(
            [[1.25, -2.5, 0.0], [0.2, 0.4, -0.1]], dtype=torch.float32
        ),
        "growth.bias": torch.tensor([0.0, 0.5, -0.25], dtype=torch.float32),
    }

    compressed = compress_growth_artifact(artifact, dtype="int4")
    restored = decompress_growth_artifact(compressed)

    assert compressed["growth.weight"].dtype == torch.uint8
    assert compressed["growth.weight.__shape__"].tolist() == [2, 3]
    assert compressed["growth.weight"].numel() * 2 >= artifact["growth.weight"].numel()
    assert restored["growth.weight"].shape == artifact["growth.weight"].shape
    assert restored["growth.bias"].shape == artifact["growth.bias"].shape
    assert torch.allclose(restored["growth.weight"], artifact["growth.weight"], atol=0.2)
