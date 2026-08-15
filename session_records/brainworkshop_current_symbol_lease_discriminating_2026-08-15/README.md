# Current-symbol search lease at the trial floor (2026-08-15)

Status: **replicated, not admitted**. Pre-registered in
`session_records/PREREGISTRATION_discriminating_leases_2026-08-15.md`, which
was committed before this ran and predicted `invent` on every seed.

This replaces `brainworkshop_current_symbol_search_lease_2026-08-15` as the
standing result. That record was measured before `and` entered the grammar
ahead of `invent`, and at 47 eligible trials it could not have separated the
two; its seeds are spent, so this campaign uses the fresh block
`current_symbol_lease_discriminating` (131017, 132017, 133017) at 448 steps,
so 448 eligible trials per session. `AgentBrain.bank` was not written and
nothing was admitted.

## Results

Winner is `invent` on every seed, bound to frontend `1ce405a0…`. Controller
digest `59c9ef2b…`. Slot 0 stayed `90e20193…`.

| Seed | Six frozen holds | Stable prefix | zeros | reversed | reward shuffled | other encoder | delay slot 0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 131017 | 1.000 × 6 (2688 bits) | 448 | 0.500 | 0.000 | 0.478 | 0.500 | 0.701 |
| 132017 | 1.000 × 6 (2688 bits) | 448 | 0.500 | 0.000 | 0.525 | 0.500 | 0.730 |
| 133017 | 1.000 × 6 (2688 bits) | 448 | 0.500 | 0.000 | 0.522 | 0.500 | 0.708 |

Every pre-registered criterion held: predicted winner on every seed, bound
frontend, stable prefix, zero program-file updates after acquire, every reject
control below `0.8`, controller digest unchanged, bank byte identical.

## What changed against the superseded record

Search still reaches `invent`, but now it has to get past `and` to do so. At
448 eligible trials the AND arm does not gate on this task, so the searcher
falls through to invent and the acquired template holds perfectly. At 48
trials it did gate, on two of three seeds, at `0.812` — a base-rate near miss
that the old episode length could not exclude.

The winner scored `448` of `448` on every session. The delay file, the closest
admitted alternative, sits at `0.701`-`0.730`.

## Not claimed

- no admission: the bound prototype is not a curated `AgentBrain.bank` slot;
- no bits-to-threshold transfer ratio against a fresh learner; zeros and delay
  slot 0 are reject controls, not a fresh-learner climb;
- no learned proposer; search still enumerates the closed grammar;
- the template is bound to `1ce405a0…` and does not survive a frontend swap,
  which is what the cross-encoder arm at `0.500` shows.
