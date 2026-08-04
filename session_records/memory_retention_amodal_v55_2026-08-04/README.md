# v55 parent write-policy freeze diagnostic

v55 freezes the generic memory-write policy during parent acquisition and
unfreezes it at the retention transition. This is the explicit update-path
separation hypothesis for parent/write-policy co-adaptation; the controller
still receives only ordinary events, opaque actions, and scalar outcomes.

On the matched seed-19 1,024/256 short rung, the arm reaches the narrow gate
at `17,920` verifier bits, with `0.999` intact recall, `0.480` clear-memory
recall, `0.519` corrupt-memory recall, and `1.000` mastered-parent retention.
That is not better than the matched unfrozen control, so the mechanism is
rejected for promotion and is not scaled to the longer transfer rung.

The phase-specific gradient route remains available as training-only
diagnostic infrastructure. The production controller is unchanged.
