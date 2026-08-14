from __future__ import annotations

from pathlib import Path

import torch

from experiments.brainworkshop_canonical import (
    build_pretrained_controller_program_machine,
    build_recursive_temporal_program_machine,
    load_temporal_controller_artifact,
    pretrain_previous_event_controller,
    save_temporal_controller_artifact,
)


def test_temporal_controller_artifact_round_trip(tmp_path: Path) -> None:
    payload, report = pretrain_previous_event_controller(
        frontend_families=2,
        lifetimes_per_family=1,
        steps_per_lifetime=4,
        heldout_frontends=1,
        event_width=8,
        source_key_width=3,
        max_history=2,
        intention_width=8,
        hidden=8,
    )
    path = tmp_path / "controller.pt"
    save_temporal_controller_artifact(payload, path)
    loaded = load_temporal_controller_artifact(path)
    machine = build_pretrained_controller_program_machine(loaded)

    assert report.unique_verifier_bits == 6
    assert report.optimizer_updates == 6
    assert report.replayed_examples == 0
    assert machine.controller_digest() == report.controller_digest
    assert machine.optimizer_updates == 0
    assert machine.program_file_updates == 0
    assert machine.learning_target == "external_temporal_address_program"
    assert torch.equal(
        machine.relative_address_logits,
        torch.zeros_like(machine.relative_address_logits),
    )

    inherited = build_pretrained_controller_program_machine(
        loaded, inherit_program_prior=True
    )
    assert torch.equal(
        inherited.relative_address_logits,
        inherited.inherited_program_prior,
    )
    assert inherited.controller_digest() == machine.controller_digest()

    recursive = build_recursive_temporal_program_machine(loaded)
    grown = build_recursive_temporal_program_machine(loaded, max_history=8)
    assert recursive.legacy_controller_digest() == machine.controller_digest()
    assert recursive.controller_digest() != machine.controller_digest()
    assert recursive.composition_depth == 1
    assert grown.max_history == 8
    assert grown.legacy_controller_digest() != machine.controller_digest()
    assert all(
        torch.equal(
            recursive.state_dict()[name], grown.state_dict()[name]
        )
        for name in recursive.state_dict()
        if name not in {"relative_address_logits", "inherited_program_prior"}
    )
