# Fourth-generation compounding: verified six-action hierarchy

## Result

The verified five-action hierarchy was retrieved as one opaque option and a
new binary router learned whether to invoke it or execute a sixth physical
read. Two frozen replications learned the sixth-action frontier with fewer
scalar verifier outcomes than a flat six-action learner, retained old skills,
passed causal interventions, and were committed as children of the verified
five-action memory lineage.

No semantic action label, correct-action hook, task ID, or hidden game state
entered the persistent learner. Temporary population members attempted the
competing actions and shared their scalar verified outcomes. Every outcome was
charged.

## Viability and localization

Across eight preflight streams:

- inherited five-action utility was approximately `0.82–0.84`;
- the best old-vs-sixth option utility was approximately `0.95–0.97`;
- action six beat the inherited hierarchy in `12.5–13.7%` of contexts.

A discarded-weight probe showed that the relation was learnable:

| Generic feature width | Preference accuracy | Held-out utility |
|---:|---:|---:|
| 9 | `90.93%` | `0.89047` |
| 11 | `91.74%` | `0.89317` |
| 13 | `91.86%` | `0.89268` |

The first 3,840-context race failed. At 7,680 contexts and 512 optimizer
updates, composition crossed the +2-point target transiently at 11,040 bits
but regressed by the final checkpoint. Lower learning rate, EMA, and the other
verified five-action parent did not repair it.

The decisive fork reused the same 7,680 unique contexts for 2,048 optimizer
updates. No additional verifier outcomes were purchased. This converted the
transient relation into stable behavior. The important resource distinction
is therefore:

> Sample-efficient learning requires both economical acquisition of unique
> experience and sufficient internal processing/replay of that experience.

## Frozen replications

Composition observes two outcomes per logical lifetime: inherited hierarchy
and action six. Flat reset observes all six. Stable means the first measured
+2-point threshold after which every later checkpoint passes.

| Seed | Inherited | Final composition | Composition stable bits | Flat stable bits | Transfer ratio |
|---:|---:|---:|---:|---:|---:|
| 8098 | `0.83166` | `0.88497` | 5,280 | 10,800 | `2.05x` |
| 8099 | `0.83447` | `0.88922` | 5,760 | 6,480 | `1.125x` |

Each arm used:

- 7,680 unique logical lifetimes;
- 2,048 optimizer updates;
- 122,880 replayed examples;
- 15,360 composition verifier bits or 46,080 flat verifier bits;
- roughly four seconds of measured GPU experiment time.

## Independent adversarial audits

Each router was evaluated on eight unseen 2,040-context streams plus a
separate 2,400-record randomized-action confirmation set.

| Audit | Seed 8098 router | Seed 8099 router |
|---|---:|---:|
| inherited hierarchy | `0.83263` | `0.82642` |
| six-action composition | `0.88636` | `0.88384` |
| absolute gain | `+5.37` points | `+5.74` points |
| shuffled router features | `0.73208` | `0.73689` |
| reversed router decision | `0.41300` | `0.41127` |
| randomized improvement estimate | `+4.76` points | `+6.05` points |
| randomized lower 95% bound | `+3.18` points | `+4.51` points |
| all eight streams improve | yes | yes |
| binary mapping retained | yes | yes |
| four-rule task retained | yes | yes |

These controls reject constant action-six use, feature-independent shortcuts,
and evaluation-only selection.

## Persistent lineage

Both routers are children of verified five-action skill
`31c77292b2935bf0ce30`.

- Seed 8098 child: `32136ff777cf65f9888d`
- Seed 8099 child: `7f1b092c30c09c5396b9`

Both stores passed exact fresh-process reload, child-corruption detection,
parent survival after child corruption, and loading of every prior ancestor.

## Conclusion

This is the fourth consecutive verified generation of hierarchical option
growth: two actions became three, three became four, four became five, and
five became six. The same small controller now reuses an increasingly capable
external-memory lineage instead of relearning the full action space.

The new frontier is adaptive experience processing: learn how many replay or
thought updates each new experience deserves, so the controller can obtain the
six-action gain without a fixed 16-replay schedule.
