# Value-diverse strategy memory

Date: 2026-07-26

## Mechanism

The four-slot strategy bank now has an optional task-agnostic admission rule.
Among candidates already paid for by the physical verifier, it retains the
candidate that maximizes minimum pairwise distance in latent strategy space.
When full, it chooses the replacement producing the most separated bank.

Sixteen context-metric perturbation directions are screened using only the
agent's own action disagreement. Exactly one pair is sent to the physical
verifier. Screening therefore adds compute but no verifier bits, semantic
labels, correct-action labels, or hidden game state.

The context and policy perturbation random generators are independent. This
fix was required for genuinely matched winner-only controls.

## Sub-minute ladder

At six rounds on seed 7072:

- winner-only: 2/5 action-divergent, 1/5 reward-divergent;
- value-diverse: 3/5 action-divergent, 2/5 reward-divergent;
- verifier bits per informative pair improved from 672 to 336.

The exact mechanistic change replicated on seed 7073. A shuffled-reward arm
failed its capability and retention gates.

## Promoted 54-round result

| Seed | Admission | Reward-divergent pairs | Bits/informative pair | Old-return target |
|---|---:|---:|---:|---:|
| 7073 | winner | 13.2% | 918.9 | 25.0% |
| 7073 | value-diverse | 56.6% | 214.4 | 95.8% |
| 7072 | winner | 11.3% | 1,072.0 | 0.0% |
| 7072 | value-diverse | 17.0% | 714.7 | 27.8% |

On seed 7073 the diverse arm also reached 41.7% reliability target accuracy
versus 8.3% for the frozen policy. Shuffling reward alignment reduced
old-return target accuracy to 0% and failed the overall gate.

All intact diverse arms passed binary/four-rule retention, physical/tensor
parity, bounded persistence, and exact save/reload gates.

## Verdict

Promoted, with magnitude variability explicitly unresolved. Preserving latent
strategy diversity makes verifier feedback more informative and can improve
later behavior without more unique experience. The next small experiments
should localize why seed 7073 benefits much more than seed 7072 before changing
capacity or enabling dynamic allocation.
