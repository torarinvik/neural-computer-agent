# Outcome-only source-trust calibration

This promotion tests whether one canonical controller can use a scalar
verifier outcome to calibrate which of two contradictory event sources should
be trusted on later ticks. The reliable source is hidden but stable within an
episode. Frontends are frozen and independent; the controller sees only
standardized event tokens, its previous opaque action, and the previous scalar
outcome.

Runtime v14 maintains a generic source-key trust state. A learned
outcome-conditioned credit policy reads prior event tokens, generic source
keys, and opaque feedback, then updates that state before the next binding
decision. A direct, low-gain source-key/trust similarity path makes that state
actionable without naming a modality, task, source, or action protocol.

Seeds 17, 18, and 19 passed the preregistered 512-update promotion rung:

| seed | clean post-calibration | shuffled stream order | no feedback | feedback shuffled | action shuffled | intention shuffled | intention zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 1.0000 | 1.0000 | 0.4998 | 0.5008 | 0.4950 | 0.4992 | 0.5008 |
| 18 | 0.7552 | 0.7599 | 0.4888 | 0.5020 | 0.5031 | 0.5073 | 0.4958 |
| 19 | 1.0000 | 1.0000 | 0.4982 | 0.4831 | 0.4987 | 0.4982 | 0.5007 |

The reward-shuffled seed-17 negative control remained at 0.4927 clean and
0.5015 under stream-order shuffling, so it did not promote. Stable threshold
crossing occurred at 327,680, 262,144, and 196,608 verifier bits for seeds
17, 18, and 19 respectively.

This promotes a narrow temporal source-trust reuse mechanism through the
canonical amodal boundary. It does not establish arbitrary trust reversal,
learned delay compensation, natural-language or speech grounding, or a
general contradiction solver.
