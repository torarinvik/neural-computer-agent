"""Finding where to cut a scene, instead of being told.

The object agent was handed its decomposition. Connected components on a colour
mask is a legitimate encoder change -- `AGENTS.md` asks only that simultaneous
streams stay separately bindable -- but nothing in that experiment *discovered*
that the scene had parts, and the record said so plainly.

MONet, IODINE and Slot Attention discover parts by competing to reconstruct the
image: K slots fight to explain the pixels, and the split that wins is the one
where each part is separately predictable. All three are gradient-trained
autoencoders with a decoder, and this repository has neither. What it has
instead is something those models mostly do without: **actions, and time**.

So the criterion changes from reconstruction to compression of the *dynamics*.
A cut is good when the pieces move independently -- when each piece's next
symbol follows from that piece's own current symbol and the action, with no
reference to the others. Write that down as a table per piece and the good cut
is simply the one whose tables are cheapest, in the ordinary description-bits
plus error-bits sense used everywhere else here.

That criterion has teeth, because the two obvious failure directions are
punished by different terms:

- a cut that is **too coarse** -- the whole scene as one symbol -- pays in
  description bits, because there are far more scenes than places. It also pays
  in error bits, and this is the interesting part: with identical markers the
  whole scene does not even determine what happens next, since *agent at a,
  goal at g* and *agent at g, goal at a* are one picture with two futures. The
  aliasing found in the object-navigation record shows up here as a cost rather
  than as a remark.
- a cut that is **too fine** -- nine grid cells, each occupied or not -- has
  tiny tables and cannot predict, because whether a cell becomes occupied
  depends on where the marker is now, which is a fact about a different cell.
  It pays in error bits.

Between them sits the cut nobody specified: one part per marker. Candidates are
scored, not assumed, and connected components is simply one entrant.

Two honest limits, stated before the numbers. The search is over a **fixed
family** of candidate cuts rather than over all partitions of the image, so
this selects a decomposition and does not invent one. And the component cut
needs alignment to be scoreable at all, because its parts arrive in positional
order -- so `slot_alignment` is load-bearing here, and a cut that could not be
tracked would lose for a reason that is not about the cut.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file

from .controller_pretraining import load_temporal_controller_artifact
from .counter_state_programs import nearest_cluster
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .navigation_environment import sample_navigation_task
from .object_scene import (
    GRID_INK,
    PLACE_COUNT,
    render_scene,
    scene_parts,
)
from .prototype_templates import cluster_events, estimated_tolerance
from .slot_alignment import Tracker

EXPERIMENT_ID = "brainworkshop-learned-decomposition-2026-08-16"
LEARNED_DECOMPOSITION_SCHEMA = "neural-computer.learned-decomposition.v1"
DEVELOPMENT_SEED = 41
FRAME_SIZE = 36
EPISODE_STEPS = 40


def _blank(size: int) -> torch.Tensor:
    frame = torch.zeros(3, size, size)
    cell = size // 3
    frame[:, cell - 1 : cell + 1, :] = GRID_INK
    frame[:, 2 * cell - 1 : 2 * cell + 1, :] = GRID_INK
    frame[:, :, cell - 1 : cell + 1] = GRID_INK
    frame[:, :, 2 * cell - 1 : 2 * cell + 1] = GRID_INK
    return frame


def _isolated(frame: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The frame with everything outside `mask` replaced by empty grid.

    Every candidate part is handed to the encoder as a whole frame with the
    same background, so no cut is advantaged or disadvantaged by presenting the
    encoder with a differently shaped input than the others.
    """

    size = int(frame.shape[-1])
    isolated = _blank(size)
    keep = mask.unsqueeze(0).expand_as(frame)
    return torch.where(keep, frame, isolated)


def _region_mask(size: int, rows: tuple[int, int], columns: tuple[int, int]):
    mask = torch.zeros(size, size, dtype=torch.bool)
    mask[rows[0] : rows[1], columns[0] : columns[1]] = True
    return mask


def _grid_cut(divisions: int):
    def cut(frame: torch.Tensor):
        size = int(frame.shape[-1])
        step = size // divisions
        parts = []
        for row in range(divisions):
            for column in range(divisions):
                rows = (row * step, (row + 1) * step if row + 1 < divisions else size)
                columns = (
                    column * step,
                    (column + 1) * step if column + 1 < divisions else size,
                )
                mask = _region_mask(size, rows, columns)
                centroid = ((rows[0] + rows[1]) / 2.0, (columns[0] + columns[1]) / 2.0)
                parts.append((centroid, _isolated(frame, mask)))
        return tuple(parts)

    return cut


def _strip_cut(vertical: bool, divisions: int = 2):
    def cut(frame: torch.Tensor):
        size = int(frame.shape[-1])
        step = size // divisions
        parts = []
        for index in range(divisions):
            span = (index * step, (index + 1) * step if index + 1 < divisions else size)
            full = (0, size)
            mask = (
                _region_mask(size, full, span)
                if vertical
                else _region_mask(size, span, full)
            )
            centre = (span[0] + span[1]) / 2.0
            centroid = (size / 2.0, centre) if vertical else (centre, size / 2.0)
            parts.append((centroid, _isolated(frame, mask)))
        return tuple(parts)

    return cut


def _whole_cut(frame: torch.Tensor):
    size = int(frame.shape[-1])
    return ((((size - 1) / 2.0, (size - 1) / 2.0), frame.clone()),)


def _component_cut(frame: torch.Tensor):
    return scene_parts(frame)


def _scatter_cut(seed: int, groups: int = 2):
    """A partition that respects nothing: pixels dealt out at random.

    The control that says the ordering below is about the *structure* of the
    cut and not about how many pieces it has. It has the same piece count as
    the component cut and none of its content.
    """

    def cut(frame: torch.Tensor):
        size = int(frame.shape[-1])
        generator = torch.Generator().manual_seed(int(seed))
        labels = torch.randint(0, groups, (size, size), generator=generator)
        parts = []
        for group in range(groups):
            mask = labels == group
            parts.append(((float(group), 0.0), _isolated(frame, mask)))
        return tuple(parts)

    return cut


@dataclass(frozen=True)
class Cut:
    """One candidate answer to "what are the things in this picture?"."""

    name: str
    parts: Any
    tracked: bool

    def __call__(self, frame: torch.Tensor):
        return self.parts(frame)


def candidate_cuts(seed: int = 0) -> tuple[Cut, ...]:
    return (
        Cut("whole", _whole_cut, tracked=False),
        Cut("components", _component_cut, tracked=True),
        Cut("halves_vertical", _strip_cut(vertical=True), tracked=False),
        Cut("halves_horizontal", _strip_cut(vertical=False), tracked=False),
        Cut("quadrants", _grid_cut(2), tracked=False),
        Cut("cells", _grid_cut(3), tracked=False),
        Cut("scatter", _scatter_cut(seed), tracked=False),
    )


# --- the stream every cut is scored on -------------------------------------


def wander(task, *, start: int, goal: int, steps: int, seed: int):
    """One episode of frames and the actions taken between them.

    Generated once and replayed for every candidate, so no cut is scored on
    different experience than another. The policy is uniform: this measures
    what a decomposition costs to describe, not how well anything behaves.
    """

    generator = torch.Generator().manual_seed(int(seed))
    place = int(start)
    frames = [render_scene(place, goal, size=FRAME_SIZE)]
    actions: list[int] = []
    for _ in range(int(steps)):
        action = int(
            torch.randint(0, task.action_count, (1,), generator=generator).item()
        )
        place = int(task.transitions[action][place])
        actions.append(action)
        frames.append(render_scene(place, goal, size=FRAME_SIZE))
    return frames, actions


def _encode_parts(encoders, parts) -> torch.Tensor:
    with torch.no_grad():
        return encoders.vision(torch.stack([frame for _, frame in parts]))


def _bits(keys: dict, alphabet: int) -> tuple[float, float]:
    """Description bits for the table, error bits for what it gets wrong."""

    width = math.log2(max(2, int(alphabet)))
    description = len(keys) * width
    mistakes = sum(
        sum(counts.values()) - max(counts.values()) for counts in keys.values()
    )
    return description, mistakes * width


def _canonical_order(traces: Sequence[Sequence[int]]) -> list[list[int]]:
    """Relabel this episode's parts so index means the same thing every time.

    Slot order is positional, which carries no identity, so part 0 is the agent
    in one episode and the goal in the next. Pooling tables across episodes
    then mixes two different objects under one key and charges the difference
    as error -- measured, that was the whole of the component cut's error term.

    Ordering by how often a part changes symbol is a canonicalisation of the
    same kind the Mealy machines get by relabelling states breadth-first, and
    it is applied to every candidate cut identically, so it cannot favour one.
    """

    width = max((len(row) for row in traces), default=0)
    if width < 2:
        return [list(row) for row in traces]
    columns: list[list[int]] = []
    for part in range(width):
        columns.append([row[part] for row in traces if part < len(row)])

    def motion(column: Sequence[int]) -> float:
        changes = sum(1 for before, after in pairwise(column) if before != after)
        return changes / max(1, len(column) - 1)

    order = sorted(
        range(width), key=lambda part: (-motion(columns[part]), columns[part][0])
    )
    return [[row[part] for part in order if part < len(row)] for row in traces]


def measure_cut(cut: Cut, encoders, episodes) -> dict[str, Any]:
    """What the dynamics cost to write down, once the scene is cut this way."""

    catalogue = []
    for frames, _ in episodes:
        for frame in frames:
            catalogue.extend(cut(frame))
    if not catalogue:
        return {"cut": cut.name, "status": "empty"}

    events = _encode_parts(encoders, catalogue)
    tolerance = estimated_tolerance(events)
    if tolerance is None:
        # The estimator refuses when within-part and between-part distances do
        # not separate, which is a real answer about the cut and not a failure
        # to be worked around.
        return {"cut": cut.name, "status": "no alphabet"}
    clusters = cluster_events(events, tolerance=tolerance, maximum_clusters=512)
    alphabet = int(clusters.shape[0])

    def symbols(frame) -> tuple:
        parts = cut(frame)
        read = nearest_cluster(_encode_parts(encoders, parts), clusters)
        return tuple(
            (centroid, int(index))
            for (centroid, _), index in zip(parts, read, strict=True)
        )

    tables: dict[int, dict[tuple[int, int], dict[int, int]]] = {}
    steps = 0
    for frames, actions in episodes:
        readings = [symbols(frame) for frame in frames]
        if cut.tracked:
            tracker = Tracker.started(readings[0])
            traces = [list(tracker.reading())]
            for step, reading in enumerate(readings[1:]):
                tracker.update(reading, action=actions[step])
                traces.append(list(tracker.reading()))
        else:
            traces = [[symbol for _, symbol in reading] for reading in readings]
        traces = _canonical_order(traces)
        for index, action in enumerate(actions):
            before, after = traces[index], traces[index + 1]
            steps += 1
            for part in range(min(len(before), len(after))):
                cell = tables.setdefault(part, {}).setdefault(
                    (before[part], action), {}
                )
                cell[after[part]] = cell.get(after[part], 0) + 1

    description = 0.0
    error = 0.0
    for keys in tables.values():
        part_description, part_error = _bits(keys, alphabet)
        description += part_description
        error += part_error
    total = description + error
    return {
        "cut": cut.name,
        "status": "scored",
        "parts": len(tables),
        "alphabet": alphabet,
        "description_bits": description,
        "error_bits": error,
        "total_bits": total,
        "bits_per_step": total / steps if steps else 0.0,
        "steps": steps,
    }


def run_learned_decomposition(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    tasks: int = 4,
    episodes: int = 4,
    steps: int = EPISODE_STEPS,
) -> dict[str, Any]:
    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(tasks):
        task = sample_navigation_task(seed=9000 + 37 * index)
        if task is None:
            continue
        generator = torch.Generator().manual_seed(seed + index)
        stream = [
            wander(
                task,
                start=int(
                    torch.randint(0, PLACE_COUNT, (1,), generator=generator).item()
                ),
                goal=int(
                    torch.randint(0, PLACE_COUNT, (1,), generator=generator).item()
                ),
                steps=steps,
                seed=seed + 100 * index + episode,
            )
            for episode in range(episodes)
        ]
        measured = [
            measure_cut(cut, encoders, stream) for cut in candidate_cuts(seed + index)
        ]
        scored = [row for row in measured if row["status"] == "scored"]
        chosen = min(scored, key=lambda row: row["total_bits"]) if scored else None
        rows.append(
            {
                "task": index,
                "cuts": measured,
                "chosen": chosen["cut"] if chosen else None,
            }
        )

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the learned decomposition run mutated AgentBrain.bank")

    names = [cut.name for cut in candidate_cuts(seed)]
    summary: dict[str, Any] = {}
    for name in names:
        scored = [
            row
            for task_row in rows
            for row in task_row["cuts"]
            if row["cut"] == name and row["status"] == "scored"
        ]
        if not scored:
            summary[name] = {"scored_tasks": 0}
            continue
        summary[name] = {
            "scored_tasks": len(scored),
            "alphabet": sum(row["alphabet"] for row in scored) / len(scored),
            "parts": sum(row["parts"] for row in scored) / len(scored),
            "description_bits": sum(row["description_bits"] for row in scored)
            / len(scored),
            "error_bits": sum(row["error_bits"] for row in scored) / len(scored),
            "total_bits": sum(row["total_bits"] for row in scored) / len(scored),
        }

    chosen_counts: dict[str, int] = {}
    for row in rows:
        if row["chosen"]:
            chosen_counts[row["chosen"]] = chosen_counts.get(row["chosen"], 0) + 1

    report = {
        "schema": LEARNED_DECOMPOSITION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "tasks": len(rows),
        "episodes_per_task": episodes,
        "episode_steps": steps,
        "cuts": summary,
        "chosen_counts": chosen_counts,
        "components_chosen": chosen_counts.get("components", 0),
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "learned_decomposition.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--bank", type=Path, default=repository / "artifacts/checkpoints/AgentBrain.bank"
    )
    parser.add_argument(
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_learned_decomposition_2026-08-16"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=4)
    arguments = parser.parse_args()
    report = run_learned_decomposition(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        tasks=arguments.tasks,
        episodes=arguments.episodes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
