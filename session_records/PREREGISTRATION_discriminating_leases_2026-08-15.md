# Pre-registration: discriminating leases (2026-08-15)

Written and committed **before** either campaign below was run. Whatever the
runs produce is recorded, including rejection. If a campaign is rejected, this
file is not edited to match it.

## Why

Both leases so far were gated on `accuracy >= 0.8` with no statement of how
many trials that gate needs. On these tasks the strongest wrong answer is not
chance; it is a single-family policy sitting at the rule's base rate, near
`0.75`. Over few eligible trials such a policy crosses `0.8` often:

| Eligible trials | P(a 0.75 policy reaches 0.8) |
| ---: | ---: |
| 47 (48-step lease) | 0.228 |
| 191 (192-step lease) | 0.059 |
| 447 (this protocol) | 0.006 |

At 47 trials, one of three onset seeds produced exactly that spurious pass,
which is what rejected `brainworkshop_onset_search_lease_2026-08-15`. The same
weakness is why the recorded current-symbol search lease no longer reproduces
once `and` entered the grammar ahead of `invent`.

## The rule, from here on

A campaign is accepted only if, in addition to its existing controls, its
episodes are long enough that a near-miss policy rarely fakes a pass:

- near-miss rate: `threshold - 0.05`, i.e. `0.75` against the `0.8` gate;
- alpha: `0.01`;
- required eligible trials: `411`, the first length past which the binomial
  upper tail stays at or below alpha (it wobbles below that because the pass
  count is `ceil(threshold * trials)`; `379` clears alpha but `380`-`383` do
  not);
- a campaign must clear alpha at its own exact length as well.

Enforced in `experiments/brainworkshop_canonical/lease_discrimination.py` and
recorded in every campaign and ledger as `discrimination`. Existing records
below the floor keep their measured numbers and are not accepted under this
rule.

## The two campaigns

Both run at **448 steps** (448 eligible trials for current-symbol, 447 for
onset; tails `0.0061` and `0.0066`), six sessions, on the frozen controller
`93a4dbb7…`, bank `07319eb1…`, curated frontend `1ce405a0…`. Neither writes
`AgentBrain.bank` and neither admits a file.

| Campaign | Block | Seeds | Predicted winner |
| --- | --- | --- | --- |
| current-symbol search lease | `current_symbol_lease_discriminating` | 131017, 132017, 133017 | `invent` |
| onset lease | `onset_lease_discriminating` | 134017, 135017, 136017 | `and` |

Blocks are registered in `experiments/brainworkshop_canonical/seed_ledger.py`,
which refuses any overlap with every earlier block, counting all seven
lifetimes each replicate consumes rather than only the start seed.

## Acceptance criteria, fixed in advance

Both campaigns:

- the search winner is the predicted kind on every seed, bound to the curated
  frontend digest;
- a stable hold prefix exists: some hold session after which every later
  measured session stayed at or above `0.8`;
- holds after the acquire session make zero program-file updates;
- every reject control stays below `0.8`: zeros, action-reversed,
  reward-shuffled, cross-encoder, and retrieve of slot 0;
- onset additionally requires invert of slot 0 and the prototype alone to stay
  below `0.8`, since the claim is that onset needs two families;
- the controller digest is unchanged and the bank file is byte-identical
  before and after.

A single seed missing any of these rejects the campaign. The blocks are then
spent and a further attempt needs new ones.
