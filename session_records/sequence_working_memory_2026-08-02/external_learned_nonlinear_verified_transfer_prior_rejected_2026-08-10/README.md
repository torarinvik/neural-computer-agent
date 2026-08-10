# Verified transfer-vs-fresh prior selection (2026-08-10)

This audit tested whether a new nonlinear factual-memory candidate should be
initialized from an existing slot or from a fresh model. The router created
copy-on-write transfer and fresh challengers, trained both only on the current
bundle, selected the lower verified probe loss, and then continued training
the selected candidate. The controller and committed slots remained frozen;
no old-regime replay or raw provisional rows were used.

The transfer challenger was isolated and persisted exactly. It was not a
promoted capability gain:

| arm | seed 82601 | seed 82602 | seed 82603 |
| --- | --- | --- | --- |
| automatic prior: held-out gate | pass | fail | pass |
| verified transfer prior: held-out gate | pass | fail | fail |
| revisit identity | 0/6 | 2/6 | 2/6 |
| promoted | no | no | no |

The verified probe selected transfer for every novel regime after the first,
but this did not make the learned nonlinear model reliably retain or route
old regimes. It improved one seed's held-out curve, partially improved one,
and worsened the third relative to the matched automatic-prior control. The
route proposal remained non-authoritative and the factual fallback preserved
correctness, but neither path supplied a reusable identity representation.

## Interpretation

This rejects transfer-prior selection as the next capability mechanism, not
copy-on-write isolation or verifier-gated challenger evaluation. The export's
strong result is consistent: storing factual transition knowledge and deriving
behavior by goal-conditioned search is safer than carrying a preferential
policy or initialization. A local probe can choose a better starting point,
but it cannot create the missing representation needed to identify a nonlinear
regime.

The active direction remains a representation-stable or meta-learned factual
model with independently verified route identity. Evidence is reported in
the paired auto and verified JSON files in this directory.
