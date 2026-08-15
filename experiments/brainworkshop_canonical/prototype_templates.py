"""Candidate templates, instead of trusting one acquired average.

The expressiveness diagnostic showed seven sampled rules were expressible by
programs the machine already supports while search found two. The gap is not
subtle: the searcher tries exactly one prototype, whatever the acquire rule's
reward-weighted average converges to, while the enumeration tried every
template the frontend can form. This module supplies the candidates.

Templates are built only from what the learner may see. One observation pass
encodes the rendered stream through its own frozen frontend; the resulting
events are clustered by distance alone, with no alphabet size, symbol label,
or verifier state involved, and candidates are the means of subsets of those
clusters. A subset mean is exactly the shape acquisition would converge to if
it happened to be rewarded on that subset, so this widens the same hypothesis
class rather than introducing a new one.
"""

from __future__ import annotations

from itertools import combinations

import torch

from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
    RenderedBrainWorkshopVerifier,
)

DEFAULT_TOLERANCE = 0.5
DEFAULT_MAXIMUM_CLUSTERS = 8
DEFAULT_MAXIMUM_SUBSET = 4


def observe_events(
    encoders: RenderedBrainWorkshopEncoders,
    config: RenderedBrainWorkshopConfig,
    *,
    seed: int,
) -> torch.Tensor:
    """Encode one episode's rendered stream through the learner's frontend.

    The pass emits a constant action because it is an observation pass; the
    outcomes it produces are not read. It costs one lifetime, which callers
    must account for.
    """

    verifier = RenderedBrainWorkshopVerifier(config.validate(), seed=int(seed))
    stream = config.streams[0]
    frames: list[torch.Tensor] = []
    while not verifier.done:
        observation = verifier.observation()
        frame = observation.vision if stream == "vision" else observation.audio
        if frame is None:
            raise ValueError("observation pass found no frame on the bound stream")
        frames.append(frame)
        verifier.score(torch.zeros(1, dtype=torch.long))
    with torch.no_grad():
        batch = torch.stack(frames)
        if stream == "vision":
            return encoders.vision(batch)
        return encoders.audio(batch)


def cluster_events(
    events: torch.Tensor,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    maximum_clusters: int = DEFAULT_MAXIMUM_CLUSTERS,
) -> torch.Tensor:
    """Group observed events by distance and return one mean per group.

    Greedy single-pass clustering: an event joins the first group within
    tolerance, or starts a new one. Nothing here knows how many distinct
    stimuli exist; the count is discovered from the stream.
    """

    if events.ndim != 2 or events.shape[0] < 1:
        raise ValueError("clustering needs a batch of observed events")
    centres: list[torch.Tensor] = []
    members: list[list[torch.Tensor]] = []
    for event in events:
        placed = False
        for index, centre in enumerate(centres):
            if float(torch.linalg.vector_norm(event - centre)) <= tolerance:
                members[index].append(event)
                centres[index] = torch.stack(members[index]).mean(dim=0)
                placed = True
                break
        if not placed:
            if len(centres) >= maximum_clusters:
                continue
            centres.append(event.clone())
            members.append([event.clone()])
    if not centres:
        raise ValueError("clustering produced no groups")
    return torch.stack(centres)


def candidate_templates(
    clusters: torch.Tensor,
    *,
    maximum_subset: int = DEFAULT_MAXIMUM_SUBSET,
) -> tuple[tuple[tuple[int, ...], torch.Tensor], ...]:
    """Every subset mean up to a size cap, smallest subsets first.

    Ordering matters: a single-cluster template is the simplest hypothesis and
    is tried before any mixture, so the cheapest explanation still wins first.
    """

    if clusters.ndim != 2 or clusters.shape[0] < 1:
        raise ValueError("candidate templates need at least one cluster")
    count = clusters.shape[0]
    cap = max(1, min(int(maximum_subset), count))
    candidates: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for size in range(1, cap + 1):
        for subset in combinations(range(count), size):
            candidates.append((subset, clusters[list(subset)].mean(dim=0)))
    return tuple(candidates)


def observed_templates(
    encoders: RenderedBrainWorkshopEncoders,
    config: RenderedBrainWorkshopConfig,
    *,
    seed: int,
    tolerance: float = DEFAULT_TOLERANCE,
    maximum_subset: int = DEFAULT_MAXIMUM_SUBSET,
    maximum_clusters: int = DEFAULT_MAXIMUM_CLUSTERS,
) -> tuple[tuple[tuple[int, ...], torch.Tensor], ...]:
    """One observation pass to candidate templates, for the searcher."""

    events = observe_events(encoders, config, seed=seed)
    clusters = cluster_events(
        events, tolerance=tolerance, maximum_clusters=maximum_clusters
    )
    return candidate_templates(clusters, maximum_subset=maximum_subset)
