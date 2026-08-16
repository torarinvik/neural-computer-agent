from __future__ import annotations

from experiments.brainworkshop_canonical.maze_transfer import (
    run_cross_task_transfer,
    run_maze_transfer,
)


def test_maze_transfer_uses_one_shared_agent_boundary(tmp_path) -> None:
    report = run_maze_transfer(
        tmp_path,
        replicates=1,
        training_episodes=2,
        evaluation_episodes=1,
        steps=8,
    )
    assert report["claim_status"] == "development_diagnostic"
    assert report["shared_agent_boundary"]["one_controller"] is True
    assert report["shared_agent_boundary"]["one_amodal_event_bus"] is True
    arms = report["replicates"][0]["arms"]
    assert arms["workshop_warm"]["same_agent_object_for_workshop_and_maze"] is True
    assert arms["workshop_warm"]["controller_unchanged"] is True
    assert arms["fresh"]["controller_unchanged"] is True
    assert arms["stale_world_model"]["controller_unchanged"] is True
    assert (tmp_path / "maze_transfer.json").is_file()


def test_cross_task_transfer_reuses_one_agent_in_both_directions(tmp_path) -> None:
    report = run_cross_task_transfer(
        tmp_path,
        replicates=1,
        workshop_lifetimes=1,
        training_episodes=2,
        evaluation_episodes=1,
        steps=8,
    )
    assert report["claim_status"] == "development_diagnostic"
    assert report["shared_agent_boundary"]["one_controller_across_both_tasks"]
    assert report["shared_agent_boundary"]["one_amodal_event_bus"]
    assert report["shared_agent_boundary"]["one_intention_bus"]
    same_agent = report["replicates"][0]["same_agent"]
    assert same_agent["same_core_instance_across_workshop_and_maze"]
    assert same_agent["controller_unchanged"]
    assert same_agent["workshop_before_maze"]["lifetimes"] == 1
    assert same_agent["workshop_after_maze"]["lifetimes"] == 1
    assert "maze_only_shared_operator" in report["replicates"][0]
    assert report["replicates"][0]["maze_only_shared_operator"]["curve"]
    assert (tmp_path / "cross_task_transfer.json").is_file()
