# Replay-free staged growth for three depth-eight procedures (2026-08-07)

Status: replicated promoted bounded continual-memory result.

This audit uses staged external slot growth rather than shared-gradient
consolidation. Each new procedure is trained only in a newly appended slot;
old slots, old route slices, and the frozen controller are not updated. A
stage is adopted only after fresh outcome probes verify the new procedure and
every earlier alias. The final bank is then tested through reload, reversal,
corruption, and target-growth controls.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| source sequence | `0 -> 1 -> 2` | `0 -> 1 -> 2` |
| depth | `8` | `8` |
| staged rewrites adopted | `2/2` | `2/2` |
| final source reload behavior | `1.0000/1.0000/0.8789` | `1.0000/1.0000/0.8047` |
| target reload behavior | `1.0000` | `1.0000` |
| target admission | grew `1 -> 2` | grew `1 -> 2` |
| reversal/recovery | passed | passed |
| corruption/reload/frozen-core | passed | passed |
| replayed examples | `0` | `0` |

The final source behavior values are independent held-out summaries; the
promotion gate also required every fresh retention probe and every alias to
remain protected after reload. Each replica consumed `323,584` unique
verifier bits, `108,544` logical lifetimes, `3,456` optimizer updates, and
`52` retention observations. Wall time was `394.6s` and `392.9s`.

This is the strongest current evidence for continual learning without
catastrophic forgetting in the external-memory boundary: new knowledge is
added in isolated mutable state while old knowledge remains frozen and
behavior-verified. It is still bounded: the procedure registry, eight-step
renderer, slot blueprint, and tested horizon are finite. Arbitrary new
computation, unrestricted memory growth, and general continual learning remain
unqualified.
