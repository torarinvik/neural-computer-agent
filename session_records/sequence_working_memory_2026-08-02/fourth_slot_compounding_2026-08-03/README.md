# Verified fourth-slot complement compounding — 2026-08-03

This is the first follow-up that clears the registered causal promotion bar
after the population/continuation diagnostics. Starting from the promoted
three-slot parent `complement_population_winner_seed93785.pt`, a fresh fourth
slot was appended and trained on 1,536 new complement lifetimes. The learner
still received only controller-visible latent features, opaque attempted
actions, and scalar attempted-action outcomes. Span-nine and span-ten
rehearsal used 256 lifetimes per stream; residual, gate, and logit replay
penalties were all 0.03. No semantic operation label or correct unattempted
action entered the learner's buffer.

## Full audit of the promoted candidate

Checkpoint: `artifacts/checkpoints/complement_population_fourth_slot_seed93871.pt`

| Audit seed | Parent | Candidate | Causal gain | Span 9 Δ | Span 10 Δ | Reset | Blank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 495171 | 66.58% | 71.90% | +5.31 pp | +0.04 pp | −0.01 pp | 49.96% | 49.98% |
| 495172 | 66.12% | 72.07% | +5.95 pp | +0.04 pp | −0.05 pp | 49.97% | 49.99% |
| 495173 | 66.89% | 72.69% | +5.80 pp | +0.02 pp | −0.04 pp | 49.99% | 49.99% |

The matched outcome-shuffled child scored **54.48%** on the first full audit,
below the 55% adversarial ceiling. The zeroed-fourth-slot control exactly
returned to the parent on every audit. Thus the improvement is causal,
reward-dependent, reset-safe, and retention-safe across three independent
audit seeds. The promoted candidate's SHA-256 is
`61f97ee8f7ce0d2ec32e065aeaa6c72ce05a8ff7332698a3b1b89f0f58fcf262`.

## Replication and interpretation

A second independent training seed with the same 1,536-lifetime recipe
produced a smaller but positive **+2.97-point** causal gain and therefore did
not clear the +5 promotion bar. This means the exact acquisition seed remains
variable, while the promoted checkpoint itself is robust on held-out audits.
The result should be described as a verified single-checkpoint compounding
milestone, not as a guaranteed outcome for every seed. The unpromoted replica
is retained as `complement_population_fourth_slot_replica_seed93873_unpromoted.pt`.

The preceding 1,024-lifetime protected append recipe produced +4.25 and +3.19
points on two streams, with shuffled control at 52.35%; increasing to 1,536
on the stronger parent was the smallest rung that crossed the causal bar.

## Decision and next frontier

This result validates a useful form of compounding: after the parent has
already acquired the adjacent complement primitive, a new zero-impact slot
can improve that primitive with 1,536 fresh lifetimes while leaving the older
spans unchanged. It does **not** yet prove that every new slot is cheaper, nor
that the system has mastered the complement operation.

The next rung should be a carefully audited fifth-slot or a genuinely new
primitive, with the same gradual data ladder and all causal, shuffled, blank,
reset, and old-skill retention gates. Do not jump directly to a large blind
run; first replicate the acquisition/promotion procedure around this
checkpoint and measure accepted improvement per fresh verifier bit.
