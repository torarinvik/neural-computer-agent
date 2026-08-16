"""Collection policy for the fast regression suite.

The repository contains two different kinds of tests:

* fast contract/regression tests, which should run on every change; and
* campaign tests, which deliberately execute multi-minute research audits and
  verify archived evidence.

Campaign modules stay in the repository and remain runnable, but are marked
so the default suite does not silently spend twenty minutes rerunning every
historical audit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CAMPAIGN_MODULES = frozenset(
    {
        "test_accumulation_curve.py",
        "test_brainworkshop_canonical.py",
        "test_brainworkshop_active_disambiguation.py",
        "test_brainworkshop_live.py",
        "test_brainworkshop_physical_bank_curriculum.py",
        "test_brainworkshop_physical_train.py",
        "test_brainworkshop_rendered_live.py",
        "test_behaviour_signature.py",
        "test_choice_induction.py",
        "test_class_escalation.py",
        "test_composition_accumulation.py",
        "test_compositional_transfer.py",
        "test_control_flow.py",
        "test_control_flow_runtime.py",
        "test_curious_exploration.py",
        "test_current_symbol_acquire.py",
        "test_dual_promotion.py",
        "test_founding_promotion.py",
        "test_generated_vocabulary_transfer.py",
        "test_identification_ceiling.py",
        "test_environment_widening.py",
        "test_induced_counter_program.py",
        "test_integrated_navigation.py",
        "test_integrated_navigation_v3.py",
        "test_integrated_agent.py",
        "test_interpreter_pretraining.py",
        "test_interpreted_machine.py",
        "test_learned_decomposition.py",
        "test_live_executive_admission.py",
        "test_live_executive_router.py",
        "test_live_identity_assignment_learned.py",
        "test_live_identity_assignment_pixel.py",
        "test_machine_factorization.py",
        "test_navigation_holdout.py",
        "test_navigation_transfer.py",
        "test_neural_workshop_live.py",
        "test_object_identity.py",
        "test_object_navigation.py",
        "test_operator_world_transfer.py",
        "test_noise_tolerant_induction.py",
        "test_onset_acquire.py",
        "test_persistent_identity_v2_experiment.py",
        "test_persistent_identity_v3_experiment.py",
        "test_relational_transfer.py",
        "test_program_search.py",
        "test_prototype_match.py",
        "test_recipe_expressibility_audit.py",
        "test_self_model_adversarial.py",
        "test_successor_transfer.py",
        "test_two_speed_battery.py",
        "test_physical_dual_loopback.py",
    }
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    campaign = pytest.mark.campaign
    for item in items:
        if Path(str(item.fspath)).name in CAMPAIGN_MODULES:
            item.add_marker(campaign)
