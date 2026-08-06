# Parent-conditioned frozen-core transfer — 2026-08-06

Status: promoted narrow two-seed parent-conditioned external-transfer result.

The inherited learner starts from a span-2 forward parent and acquires a new
span-4 reverse procedure without changing the controller core. Its external
slot has recurrent temporal state, receives the frozen controller's learned
intention as an additional opaque representation, and has a context-conditioned
output gate. The gate reads only learned controller context; it has no task,
modality, protocol, correct-action, or semantic label input.

Both inherited and fresh arms receive the same fresh target episodes and fresh
forward parent-task rehearsal episodes. The fresh arm is deliberately more
plastic: all parameters are trainable. No old examples are replayed.

Across seeds 69316 and 69317:

- inherited stable target bits: `9,216` / `9,216`;
- fresh stable target bits: `15,360` / `12,288`;
- fresh-over-inherited transfer ratio: `1.667x` / `1.333x`;
- transferred target accuracy: `0.848` / `0.809`;
- parent retention: `1.000` / `1.000`;
- frozen-core digest: unchanged for both seeds;
- replayed examples: `0`.

The reward-shuffled arm was evaluated against the parent’s pre-growth target
baseline rather than a hard chance threshold, because the frozen parent can
already carry partial target behavior. Shuffled training added no target gain,
and the transferred arm beat it by `0.188` and `0.250`.

This promotes a reusable parent-conditioned external computation boundary,
not general continual learning. The next gates are more independent parent
primitives, more than one simultaneously stored external program, persistent
artifact reload, and transfer under genuinely varied rendered streams.

Earlier recurrent-only, width, stronger-parent, intention-only, and paired
credit failures are retained under `controls/`.
