from __future__ import annotations

from experiments.brainworkshop_canonical.maze_transfer import run_maze_transfer


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
