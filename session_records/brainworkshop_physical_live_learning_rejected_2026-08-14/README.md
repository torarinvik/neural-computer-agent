# Physical live Position 1-Back learning rejection (2026-08-14)

Status: **I/O mechanism retained; checkpoint architecture-invalid and rejected**.

This run trained one fresh controller directly against the frontmost public Brain
Workshop 5.0 macOS window. The learner received only display-captured RGB
events, emitted ordinary `A` key decisions, and updated from the same visible
green/red/blue feedback used by a human. Private session data was inspected
only after the run as a discarded diagnostic and never entered the learner.

Two interface calibration failures were corrected before the recorded run:

- neutral trials are absent outcomes under Brain Workshop's default scoring,
  rather than fabricated positive rewards;
- the macOS renderer's observed correct-feedback pixels are saturated green
  `(0, 255, 0)`, despite the lighter fallback named in the source config;
- spatially moved pulses no longer cancel in the onset gate when a short blank
  interval falls between screen samples.

The final precommitted run used Position 1-Back, twelve one-second trials per
session, ten sessions, a 12 Hz cognitive loop, persistent weights and optimizer
state, fresh temporal history per session, and zero replay. It recorded 55
unique public verifier bits and exactly 55 optimizer updates. Four were visible
green successes. Cumulative public score was `0.0727`; the final 32-bit rolling
score was `0.09375`, and the last session returned to zero. Brain Workshop's
own post-run session percentages agreed with the public evidence score up to a
final action that cannot be resolved without a later stimulus boundary.

The I/O evidence establishes that physical RGB input, ordinary key output,
delayed receipt credit, checkpoint/resume, and immediate online updates work
together. It is not a valid task-acquisition run for the normative architecture:
the 55 updates changed controller weights instead of an independently persisted
external program file. It also did not show stable learning, mastery, retention,
or transfer. The checkpoint is therefore not curated under
`artifacts/checkpoints/` and must not be used as inherited controller state.
