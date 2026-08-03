# Missing-evidence population race (2026-08-03)

The single accepted missing-evidence recipe has replicated at 1,024
lifetime audits, but its gain is seed-variable. This population race tests
whether independent verifier-driven experience can find a stronger member
without changing the architecture or selecting on a tiny screen.

All arms start from the same independent span-11 parent
`span11_prior_adaptive_seed996033.pt` and use:

- 512 fresh span-11 mixed-operation lifetimes;
- 512 span-10, 512 span-9, and 512 blank span-11 protected lifetimes;
- 32 epochs, batch 512, learning rate 0.0005;
- binary complement and scalar-outcome critic losses;
- gate/logit protection 0.1 and no additional reads or semantic labels.

Four independent model/data seeds are trained. Every arm is audited on the
same lifetime-disjoint 1,024-lifetime audit with acquisition, causal,
retention, blank, and reset gates. A member is only a candidate for the
4,096-lifetime audit if its paired child-over-span-10 interval is positive;
the common audit is the selection boundary, not the training report.

## Result

All four arms passed the common 1,024-lifetime audit:

| Seed | Child-over-parent | 95% interval | Span-10 Δ | Span-9 Δ |
| ---: | ---: | ---: | ---: | ---: |
| 996060 | +2.15 pp | [+1.54, +2.73] | +0.12 pp | −0.50 pp |
| 996061 | +2.27 pp | [+1.66, +2.89] | +0.08 pp | −0.36 pp |
| 996062 | **+2.33 pp** | [+1.70, +2.95] | +0.11 pp | −0.29 pp |
| 996063 | +2.04 pp | [+1.45, +2.65] | +0.23 pp | −0.34 pp |

The selected seed `996062` passed the 4,096-lifetime audit at **+2.02 pp**
(95% interval **[+1.71, +2.32] pp**). An independent second arm, seed
`996061`, also passed at **+1.56 pp** (95% interval **[+1.27, +1.85] pp**).
Both remained blank/reset safe and within the two-point old-span retention
gate. The selected child reached 70.16% versus 68.14% for the span-10
parent; this is a stronger high-power frontier than the prior +1.77 pp run,
but it is not 90% mastery.

The population is promoted as a verified frontier, not as a guarantee that a
random seed will pass. The top checkpoint is curated at
`artifacts/checkpoints/span11_missing_evidence_population_seed996062.pt`.
Its SHA-256 is
`c00a81ce929ddb55ce8dfb644805974e0c33d9301540041157f03fc558c1cf6b`.

## Accounting

Each arm charged 2,048 unique logical lifetimes and 20,992 unique verifier
bits: 512 target, 512 span-10, 512 span-9, and 512 missing-evidence
lifetimes. Each used 1,312 optimizer updates and 1,024 old-span rehearsal
lifetimes. Across the four independent arms this is 8,192 lifetimes, 83,968
training verifier bits, 5,248 optimizer updates, and 4,096 replayed
examples. The audits used additional lifetime-disjoint verifier episodes;
online latency and fresh-learner transfer were not measured. Replay savings
remain zero, so this is a capability frontier rather than an autonomous
replay-efficiency stop.
