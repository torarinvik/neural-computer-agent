# Promoted: six-regime held-out prefix retention

Date: 2026-08-10
Seeds: `85401`, `85402`
Schema: `neural-computer.policy-free-intention-prefix-growth.v2`

## Result

Both replicated seeds pass the bounded six-regime retention promotion gates:

- all six opaque regimes master after delayed feedback settles;
- every new cell passes an eight-context held-out verifier prefix;
- prior content remains above the `0.90` floor;
- prior route probability remains above the `0.15` floor;
- sparse materialization, reward-shuffled, missing-evidence, corruption,
  exact-persistence, frozen-controller, and zero-replay controls pass.

The retained content floors are `0.9596` and `0.9617`; route floors are
`0.5884` and `0.6726`. This is a bounded stable-prefix result, not a claim of
positive transfer, unrestricted memory growth, arbitrary new computation, or
general continual learning.

## Architectural finding

The breakthrough is the separation of three lifecycle phases: an unprotected
cell receives causal evidence, a held-out verifier qualifies it, and only then
does its frozen context prototype become an address prior. Protection freezes
both generator content and route state until a relevant reversal releases the
cell. The planner and controller remain unchanged.

The matched fresh accounting is deliberately retained. Positive transfer did
not occur on every disjoint successor (`false` for both seeds); the system
learned to retain and address the growing bank, but it has not yet shown that
the bank makes every new regime faster than a fresh learner.

Reports: `seed-85401.json`, `seed-85402.json`.
