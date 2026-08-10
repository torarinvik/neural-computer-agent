# Active factual-model disambiguation probe — promoted

This three-seed audit tests causal probe addressing. Two opaque external
factual models agree on the current observation and differ only in the
consequence of one available intention. The planner selects the intention
with maximal predicted disagreement, and the observed consequence routes the
hidden model. A uniform random-intention and random-tie-break control is the
floor.

| metric | seed 83001 | seed 83002 | seed 83003 |
| --- | ---: | ---: | ---: |
| active-probe routing accuracy | 1.000 | 1.000 | 1.000 |
| random-control routing accuracy | 0.750 | 0.773 | 0.730 |
| causal probe margin | 0.250 | 0.227 | 0.270 |
| controller updates | 0 | 0 | 0 |
| raw replayed examples | 0 | 0 | 0 |
| exact persistence | true | true | true |

All seeds pass the narrow probe gate. The probe and external bank are
inference-only during the audit; the controller remains frozen and model
queries do not mutate state.

Claim boundary: this qualifies active factual disambiguation for a tiny
two-regime synthetic fixture. It does not establish learned probe selection,
multimodal probe usefulness, arbitrary action spaces, or general continual
learning. The next audit must test a larger candidate space, noisy outcomes,
and a causal held-out environment where the probe is not hand-constructed.

Reports are protected by `SHA256SUMS`.
