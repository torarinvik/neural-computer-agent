# Verified-use volatility protects useful memory without blocking growth

## Question

Can each persistent latent memory row learn how editable it should be, so that
verified useful experience is protected while failed or stale experience
remains available for reuse?

The key distinction is **verified usefulness, not access frequency**. A bad row
may be read often. Reads alone must not let it freeze itself.

## Bounded negative before the breakthrough

We first applied plasticity to the previous neural skill slot during acquisition
of the next primitive. Fixed whole-slot, gradient-alignment, and per-hidden-unit
importance variants all failed the replication gate:

- fixed volatility had seed-specific oracle value but no universal level;
- early reward telemetry selected the wrong level on fresh seeds;
- acquisition/retention gradient alignment was weak or negative;
- per-unit retention importance was safe at low scale but improved the new skill
  by only 1.12 points on average, below the two-point gate;
- larger per-unit updates harmed acquisition.

Conclusion: the current gated skill ladder already isolates weight changes well.
Rewriting inherited weights is not presently the high-ROI location for
plasticity.

## Persistent-memory mechanism

`PersistentMemory` schema v4 adds one scalar `volatility` per physical row:

- a new row starts fully plastic (`1.0`);
- verified success lowers volatility;
- verified failure raises it;
- unused rows slowly thaw toward `1.0`;
- access alone never protects a row;
- elastic replacement rewrites keys, values, and strength in proportion to the
  row's volatility;
- the field is preserved by clone/select/growth/save/load;
- v1-v3 stores load backward-compatibly with volatility `1.0`.

The implementation initially exposed a useful audit failure: stale-row thaw was
written with PyTorch advanced indexing and therefore modified a temporary
tensor. The causal probe caught the unchanged stale volatility, the update was
corrected, and tests now cover protection, failure thaw, stale thaw, persistence,
and elastic replacement.

## Non-stationary causal atom

Each eight-row bank contained:

- three stable, repeatedly successful skills;
- three equally frequent, repeatedly failing decoys;
- two once-useful but now stale skills.

Four new skills then had to enter the bounded bank. The learner-facing memory
saw only latent keys/values, access statistics, and scalar verifier outcomes.
The stable/decoy/stale labels existed only in the private audit.

Across 64 seeds:

| policy | final score | stable retained | new acquired | failed decoys rewritten |
|---|---:|---:|---:|---:|
| verified-use volatility | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| uniform plasticity | 77.46% | 47.40% | 100.00% | 50.52% |
| access-only plasticity | 71.43% | 100.00% | 50.00% | 29.17% |
| shuffled volatility/row | 79.02% | 51.04% | 100.00% | 50.00% |

Stable and failing decoy rows had identical access counts. The result therefore
cannot be explained by frequency. Shuffling scalar-to-row correspondence
destroys the retention benefit, establishing row-specific causality.

## Reward-only learned selector

The diagnostic argmax was then removed. A 321-parameter generic policy received
four physical-memory features per candidate row:

- volatility;
- normalized access count;
- normalized age;
- write strength.

It sampled four replacement actions and trained with REINFORCE from the final
scalar verifier score. It received no semantic task ID, stable-row label, or
correct replacement action.

Four independent training seeds (`13000`, `14000`, `15000`, `16000`) used 1,024
unique training environments and 192 updates. Every run achieved:

- 100% perfect episodes over 512 held-out environments;
- 100% stable retention in 128 real physical-memory audits;
- 100% new-skill acquisition;
- all five pre-registered behavioral and causal gates.

Shuffling volatility across physical rows reduced stable retention to
51.04–53.39%. Access-only or uniform features lost about one stable row per
episode. The selector therefore learned to act on verified-use plasticity from
reward; it did not merely inherit a hard-coded replacement rule.

## Scope and next frontier

This is a significant mechanistic breakthrough: external memory can now be
selectively stable and selectively plastic at the same time, and a small
reward-only controller can learn to exploit that state.

It is not yet evidence that the full visual cognitive controller learns optimal
volatility end to end. The next rung is to expose volatility as one generic
feature of the existing in-controller replacement head, initialize it as a
no-op, and run a real mixed cognitive stream with:

1. stable old skills;
2. misleading frequently accessed memories;
3. a genuine task shift;
4. new skills competing for bounded disk capacity;
5. retention, acquisition-speed, save/reload, shuffle, corruption, and
   memory-removal audits.

Only promote if the full controller learns the new skills faster without
crossing the old-skill retention gate.
