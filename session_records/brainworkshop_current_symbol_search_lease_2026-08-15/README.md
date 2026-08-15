# Current-symbol search invent lease (2026-08-15)

Status: **replicated, not admitted** — but see the reproduction note; this
record predates the grammar it was measured under.

## Reproduction note (added after the onset lease)

`and` was added to the proposer after this campaign ran, and it is enumerated
before `invent`. Re-running the same seeds under the current code at 48 steps
selects `and` at `0.812` on 122017 and 123017 and returns `rejected`; only
124017 still reaches `invent`. The same re-run at 96 and 192 steps returns
`invent` at `1.000` on all three seeds and accepts.

The grammar is not the weak part: `0.8` over 48 or fewer eligible trials does
not separate a base-rate near-miss from a solution, which is the same failure
that rejected the 48-step onset lease
(`session_records/brainworkshop_onset_search_lease_2026-08-15/`). The numbers
below are left exactly as measured; they are not a claim about the current
grammar at 48 steps. Re-establishing this result under the current grammar
needs a fresh unused-seed block, which has not been run.

Unused seeds 122017, 123017, and 124017 selected a program by search:
delay retrieves failed, invent acquired on the curated frontend
`rendered_frontend_seed1001.pt` (`1ce405a0…`), then held for six frozen
sessions. `AgentBrain.bank` was not written. This population does not
reuse 116017–121017 or the Dual lease.

## Results

Winner is `invent` on every seed, bound to frontend `1ce405a0…`.
Controller digest `59c9ef2b…`. Slot 0 stayed `90e20193…`.

| Seed | Acquire | Six frozen holds | Stable prefix | Zeros | Reverse | Shuffle | Delay slot 0 | Other encoder |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 122017 | 0.958 | 1.000 × 6 (288 bits) | 48 | 0.500 | 0.000 | 0.604 | 0.792 | 0.500 |
| 123017 | 0.938 | 1.000 × 6 (288 bits) | 48 | 0.500 | 0.000 | 0.542 | 0.708 | 0.500 |
| 124017 | 0.938 | 1.000 × 6 (288 bits) | 48 | 0.500 | 0.000 | 0.562 | 0.667 | 0.500 |

Stable bits are the first hold prefix that remains at every later
measured prefix. All later holds stayed at `1.000`.

## Limits

- not a curated AgentBrain slot;
- grammar is still retrieve, compose, invent;
- not a Dual 2-back bits-to-threshold transfer;
- not desktop Dual.

## Superseded

`brainworkshop_current_symbol_lease_discriminating_2026-08-15` is the standing
result for this claim: a fresh block (131017-133017) at 448 eligible trials,
pre-registered, winner `invent` at `1.000` on every seed. This record's seeds
are spent and its length is below the trial floor, so it is history, not
evidence.
