from __future__ import annotations

import torch

from experiments.brainworkshop_canonical.maze_environment import (
    MazeTask,
    MazeVerifier,
    render_maze,
    sample_maze_task,
)


def test_sampled_maze_is_connected_and_goal_is_not_rendered() -> None:
    task = sample_maze_task(seed=7101, minimum_distance=5)
    assert task is not None
    task.validate()
    assert task.distances()[task.start] >= 5
    alternate_goal = next(
        place for place in range(task.place_count) if place not in {task.start, task.goal}
    )
    alternate = MazeTask(
        walls=task.walls,
        open_positions=task.open_positions,
        transitions=task.transitions,
        action_permutation=task.action_permutation,
        goal=alternate_goal,
        start=task.start,
        grid_size=task.grid_size,
    ).validate()
    assert torch.equal(
        render_maze(task, task.start), render_maze(alternate, alternate.start)
    )


def test_maze_verifier_moves_only_through_open_cells() -> None:
    task = sample_maze_task(seed=7102, minimum_distance=5)
    assert task is not None
    verifier = MazeVerifier(task, steps=3)
    before = verifier.observation()
    outcome = verifier.score(torch.tensor([0], dtype=torch.long))
    after = verifier.observation()
    assert before.shape == after.shape == (3, 42, 42)
    assert outcome.reward.shape == outcome.eligible.shape == (1,)
