# Diagnostic: peripheral skill leakage quantified (2026-08-06)

After end-to-end Snake acquisition, the core is frozen and Snake's
peripherals are replaced with fresh random modules retrained through the
frozen core, against a matched random-frozen-core control.

| recovery mastery | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| original (trained end to end) | 0.8555 | 0.9297 |
| fresh peripherals + trained core, 100 updates | 0.3184 | 0.0449 |
| fresh peripherals + trained core, 300 updates | 0.9375 | 0.8320 |
| fresh peripherals + random core, 300 updates | 0.5254 | 0.6680 |

Findings: (a) fresh peripherals recover full or near-full mastery through
the trained core at half the original budget - the strategy needed to
play is overwhelmingly core-resident and the peripheral contribution is
re-derivable; (b) the random-core control reaching 0.53-0.67 shows
peripherals CAN carry substantial competence through any fixed core, so
peripheral purity is not automatic - the trained-core advantage
(+0.41/+0.16) bounds the core's genuine contribution. Zero replay.

Implication for the storage rule: peripheral leakage is real but bounded
and recoverable; the externalization harness's ignorance training is the
existing mechanism that squeezes it further.
