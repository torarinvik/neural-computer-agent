"""Every seed block a recorded campaign has consumed, in one place.

A block is consumed as soon as any of its lifetimes has been observed,
including diagnostics run to explain a failure. Re-using one is no longer a
holdout measurement, so each campaign names its own block here and the guard
refuses any overlap with every other block.

Each replicate of a lease burns `seed .. seed + LEASE_SESSIONS`, not just the
start seed, so the guard compares spans rather than start values. Blocks are
1000 apart to leave that room.
"""

from __future__ import annotations

# Widest span one replicate consumes, in seeds, counted from its start seed.
LIFETIMES_PER_REPLICATE = 7

BLOCKS: dict[str, tuple[int, ...]] = {
    "dual_holdout": (113_017, 114_017, 115_017),
    "current_symbol_unbound_acquire": (116_017, 117_017, 118_017),
    "current_symbol_bound_frontend": (119_017, 120_017, 121_017),
    "current_symbol_search_lease": (122_017, 123_017, 124_017),
    "onset_lease_48": (125_017, 126_017, 127_017),
    "onset_lease_192": (128_017, 129_017, 130_017),
    "current_symbol_lease_discriminating": (131_017, 132_017, 133_017),
    "onset_lease_discriminating": (134_017, 135_017, 136_017),
    # The integrated agent walks a stream of tasks and strides `seed` by
    # TASK_SEED_STRIDE per task, so one replicate consumes a span three orders
    # of magnitude wider than a lease does. Placed far above every earlier
    # block, and far enough apart from each other, that no span can meet
    # another whatever stream length is chosen.
    "integrated_agent_holdout": (3_000_017, 3_500_017, 4_000_017),
    # Composition walks the same wide per-task stride, and its stream is longer
    # than the integrated agent's, so it is placed above that block with the
    # same clearance.
    "compositional_transfer_holdout": (5_000_017, 5_500_017, 6_000_017),
}

# One replicate of the integrated agent consumes `stride * tasks` seeds plus
# the probe offset, not the seven a lease replicate spans.
INTEGRATED_SESSIONS_PER_REPLICATE = 250_000


def block(name: str) -> tuple[int, ...]:
    """Start seeds of one named block."""

    if name not in BLOCKS:
        raise KeyError(f"unknown seed block: {name}")
    return BLOCKS[name]


def _span(seed: int, *, sessions: int) -> set[int]:
    return set(range(seed, seed + sessions + 1))


def consumed_seeds(
    *, exclude: str = "", sessions: int = LIFETIMES_PER_REPLICATE
) -> frozenset[int]:
    """Every lifetime consumed by recorded blocks other than `exclude`."""

    if exclude and exclude not in BLOCKS:
        raise KeyError(f"unknown seed block: {exclude}")
    consumed: set[int] = set()
    for name, seeds in BLOCKS.items():
        if name == exclude:
            continue
        for seed in seeds:
            consumed |= _span(seed, sessions=sessions)
    return frozenset(consumed)


def assert_unused_block(
    name: str,
    seeds: tuple[int, ...],
    *,
    sessions: int = LIFETIMES_PER_REPLICATE,
    also_used: frozenset[int] = frozenset(),
) -> None:
    """Fail closed if this block overlaps any other recorded lifetime."""

    if name in BLOCKS and tuple(seeds) != BLOCKS[name]:
        raise ValueError(
            f"block {name} is recorded as {BLOCKS[name]}, not {tuple(seeds)}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed block entries must be unique")
    if len(seeds) < 3:
        raise ValueError("a lease population needs at least three seeds")
    # An unregistered name is a candidate block: it must clear everything.
    known = name if name in BLOCKS else ""
    used = set(consumed_seeds(exclude=known, sessions=sessions)) | set(also_used)
    claimed: set[int] = set()
    for seed in seeds:
        span = _span(seed, sessions=sessions)
        overlap = span & used
        if overlap:
            raise ValueError(
                f"block {name} collides with recorded lifetimes: {sorted(overlap)}"
            )
        if span & claimed:
            raise ValueError(f"block {name} replicates overlap each other")
        claimed |= span
