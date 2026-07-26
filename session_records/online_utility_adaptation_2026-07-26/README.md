# Online utility adaptation milestone — 2026-07-26

## Result

One controller now adapts its bounded-memory replacement utility online when
the environment changes what makes stored information useful. A continuous
capacity-six stream changed from recency-dominant to frequency-dominant, back
to recency-dominant, and finally to equal weighting. The learner received no
phase identity, boundary signal, utility labels, correct actions, optimizer
reset, or replay.

Two independent 64-update runs passed every registered online, causal, and
retention gate:

| Seed | Seconds | Recency target | Frequency target | Recency return | Equal return |
|---:|---:|---:|---:|---:|---:|
| 6809 | 28.66 | 90.67% | 86.43% | 91.16% | 90.53% |
| 6810 | 28.89 | 90.82% | 87.16% | 91.31% | 89.99% |

Each run used 64 updates, 114,688 generated contexts, and 212,992 unique
verifier bits counting both candidates. Only one existing utility-residual
coefficient changed. Binary mapping and four-rule skills remained intact.

Selected checkpoint:
`artifacts/checkpoints/unified_memory_online_utility_seed6810.pt`

SHA-256:
`c3e837c6512a30c11b1c861b79242296b76cfa0cd9fe62aa414d3e5b2aa10750`

Independent replica:
`artifacts/checkpoints/unified_memory_online_utility_seed6809.pt`

SHA-256:
`d25d26c4d34ff86e50474b5ff38c630a2d92b782dea10d4782b01a363bb64a81`

## Why this mechanism won

The useful update is a two-clone horse race over one controller coefficient.
On each fresh memory bank, temporary candidates at `w + 3` and `w - 3` make
real greedy replacement decisions. The candidate producing more later
verified success determines a 1.5-unit step. Only the winner's coefficient
survives; there is still one controller.

Earlier gradient variants failed because their stochastic training objective
did not match the greedy behavior being selected. An exact local probe showed
that high exploration temperatures could reverse the gradient sign even
though direct coefficient sweeps showed a clear behavioral optimum.

## Adversarial control

The matched seed-6808 control randomly swapped the paired verified outcomes
before choosing a winner. It moved the coefficient in the wrong direction:
frequency-dominant target eviction fell to 57.71%, below the frozen parent's
69.53%. The run failed its gates and saved no checkpoint. This shows that the
successful movement depends on the alignment between candidate behavior and
future verified outcomes.

## Physical disk audit

The selected seed-6810 checkpoint was evaluated on 1,024 independent physical
disk banks:

- learned 96.94%, visible oracle 96.81%, full oracle 97.41%;
- age shuffled 92.74%, frequency shuffled 88.66%;
- 6,144 rows before and after, zero growth;
- all 1,024 access histories survived save/reload exactly;
- no weights changed during the audit.

## Honest boundary

Online adaptation itself used tensorized memory banks for speed. The final
adapted controller, not the adaptation process, was audited against physical
serialized disk. This is a replicated proof of rapid verifier-driven
adaptation for one generic coefficient. It is not yet evidence for a learned
general meta-optimizer, a high-dimensional latent utility policy, or online
adaptation inside an unbounded disk stream.

## Next atom

Add one generic utility feature and one coefficient. Race the one- and
two-parameter controllers at matched verifier bits on gradual relevance
changes. Promote only if the extra dimension lowers post-switch regret,
survives feature corruption controls, and preserves every old skill.
