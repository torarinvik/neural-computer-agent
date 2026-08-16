"""Scenes with more than one thing in them, and a frontend that says so.

Every observation in this repository has been atomic. One place, one symbol,
one event -- so "what is in the scene" and "which scene is this" were the same
question, and nothing ever had to be decomposed.

Here a scene holds two markers: where the agent is, and where it has been asked
to go. That single change does two things at once.

**It makes a goal something you can be given.** The goal is not a number passed
in a function call and not a reward to be stumbled upon; it is *shown*, through
the same pixels and the same frozen encoder as everything else.

**It makes the observation compositional.** Eight places for the agent and
eight for the goal is sixty-four distinct scenes. An agent that reads a scene
whole must learn all sixty-four and can carry nothing from one goal to another;
an agent that decomposes it into two objects learns eight and should transfer
to combinations it has never seen. That gap is the measurement.

Two design choices were forced by measurement rather than taste.

**The markers share a colour.** Drawn in different colours, "goal at place
three" and "self at place three" encode to different events, and an agent could
never tell it had arrived -- the two objects would live in incomparable
alphabets. One colour puts every marker in the same eight-place alphabet, so
comparing them is meaningful, and the frontend separates them by **connected
component** rather than by hue.

**Nothing says which slot is which.** Slots arrive ordered by position, which
changes as the agent moves, so slot index carries no identity. Which object you
*are* is left to be worked out from the only evidence there is: across a
wandering episode, one of them stays put.

The decomposition living in the frontend is the honest place for it.
`AGENTS.md` requires that "simultaneous streams remain separately bindable
rather than blindly averaged", and emitting one event per detected object
instead of one per frame is exactly that -- an encoder change, not a controller
change.
"""

from __future__ import annotations

from collections import deque

import torch

from .rendered_environment import _GRID_POSITIONS

OBJECT_SCENE_SCHEMA = "neural-computer.object-scene.v1"
PLACE_COUNT = len(_GRID_POSITIONS)
MARKER_COLOUR = (0.25, 0.70, 1.0)
GRID_INK = 0.12


def _grid(size: int) -> torch.Tensor:
    frame = torch.zeros(3, size, size)
    cell = size // 3
    frame[:, cell - 1 : cell + 1, :] = GRID_INK
    frame[:, 2 * cell - 1 : 2 * cell + 1, :] = GRID_INK
    frame[:, :, cell - 1 : cell + 1] = GRID_INK
    frame[:, :, 2 * cell - 1 : 2 * cell + 1] = GRID_INK
    return frame


def _draw(frame: torch.Tensor, place: int, *, size: int) -> None:
    cell = size // 3
    row, column = _GRID_POSITIONS[place]
    margin = max(2, cell // 5)
    frame[
        :,
        row * cell + margin : (row + 1) * cell - margin,
        column * cell + margin : (column + 1) * cell - margin,
    ] = torch.tensor(MARKER_COLOUR).view(3, 1, 1)


def render_markers(places, *, size: int = 36) -> torch.Tensor:
    """The grid, with a marker at each of `places`.

    Any number of them, all the same colour, drawn in the order given so that
    overlapping markers coincide rather than occlude. Two markers on one place
    is one marker, which is what "arrived" looks like from outside and is the
    same thing the reward is about.
    """

    ordered = [int(place) for place in places]
    if not ordered:
        raise ValueError("a scene needs at least one marker")
    for place in ordered:
        if not 0 <= place < PLACE_COUNT:
            raise ValueError("a marker is outside the grid")
    frame = _grid(size)
    for place in ordered:
        _draw(frame, place, size=size)
    return frame


def render_scene(agent: int, goal: int, *, size: int = 36) -> torch.Tensor:
    """The grid, with a marker where the agent is and one where it should be.

    The two-marker case, kept as its own name because it is the one every
    earlier measurement was taken on and the pixels must not move.
    """

    return render_markers((goal, agent), size=size)


def _components(mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    """Connected marker regions, found without knowing the grid layout."""

    height, width = mask.shape
    seen = torch.zeros_like(mask, dtype=torch.bool)
    regions: list[list[tuple[int, int]]] = []
    for row in range(height):
        for column in range(width):
            if not bool(mask[row, column]) or bool(seen[row, column]):
                continue
            region: list[tuple[int, int]] = []
            frontier = deque([(row, column)])
            seen[row, column] = True
            while frontier:
                y, x = frontier.popleft()
                region.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    if bool(mask[ny, nx]) and not bool(seen[ny, nx]):
                        seen[ny, nx] = True
                        frontier.append((ny, nx))
            regions.append(region)
    return regions


def scene_parts(frame: torch.Tensor, *, tolerance: float = 0.12):
    """`(centroid, isolated frame)` per marker, ordered by position only.

    Ordering by position means the index flips as the agent moves past the
    goal, so it cannot stand in for identity. That is the point: a slot is a
    thing in the scene, not a role in the task. The centroid is *where* the
    part is on the image plane, which is not an identity either -- it is what
    an aligner needs in order to work one out.
    """

    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("a scene is one RGB frame")
    size = int(frame.shape[-1])
    reference = torch.tensor(MARKER_COLOUR).view(3, 1, 1)
    mask = (frame - reference).abs().sum(dim=0) <= tolerance
    parts = []
    for region in _components(mask):
        centroid = (
            sum(y for y, _ in region) / len(region),
            sum(x for _, x in region) / len(region),
        )
        isolated = _grid(size)
        for y, x in region:
            isolated[:, y, x] = reference.reshape(3)
        parts.append((centroid, isolated))
    parts.sort(key=lambda item: item[0])
    return tuple(parts)


def scene_slots(frame: torch.Tensor, *, tolerance: float = 0.12):
    """One isolated frame per marker, ordered by position and nothing else."""

    return tuple(isolated for _, isolated in scene_parts(frame, tolerance=tolerance))


def encode_slots(encoders, frame: torch.Tensor) -> torch.Tensor:
    """Encode each marker separately, through the unchanged vision encoder."""

    slots = scene_slots(frame)
    if not slots:
        raise ValueError("a scene must contain at least one marker")
    with torch.no_grad():
        return encoders.vision(torch.stack(slots))


def encode_scene(encoders, frame: torch.Tensor) -> torch.Tensor:
    """Encode the whole scene as one event, the way everything else does.

    The control's frontend. It sees exactly the same pixels and reads them
    without decomposing, so any difference between the two agents is the
    decomposition and not the information available.
    """

    with torch.no_grad():
        return encoders.vision(frame.unsqueeze(0))
