# Online interleaved reliability and delay

This pressure test feeds two factual streams in an interleaved order. Scalar
verifier outcomes update the reliability statistics only after each route, so
the committed-slot veto must learn online rather than from a separate
calibration fixture. After the warm-up, a low-error corrupted revisit is
rejected; clean reversals still route to both historical slots. A fresh
gate-disabled control matches the same corruption.

Incomplete timestamp windows are interleaved into the same run and update the
wait statistics once. Delayed incomplete evidence becomes worth waiting for,
while fast absence becomes releasable. The controller and factual bank remain
frozen after source fitting, with no old-evidence replay.

This is a bounded online reliability/delay result, not general continual
learning, unrestricted memory growth, or arbitrary new computation.
