# Learned anonymous stream binding

This two-seed pressure test trains one generic transition-context encoder from
paired same-stream views, freezes it, and then binds three interleaved streams
without caller-supplied stream keys. The external binding memory maintains only
bounded anonymous prefixes, opaque prototypes, delay estimates, and
verifier-calibrated reliability. The bound key feeds one shared factual
multi-stream transition router; the controller is frozen throughout.

| seed | identity loss | learned consistency | fresh consistency | missing | order control | delay | reload |
| ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 2301 | 0.000195 | 1.000 | 0.167 | pass | pass | pass | pass |
| 2302 | 0.000215 | 1.000 | 0.167 | pass | pass | pass | pass |

Both runs rejected checksum corruption, updated external reliability from
scalar verifier outcomes, kept the controller byte-stable and used zero
replay. Trainer-only stream indices were used to make positive pairs and score
the diagnostic; they are not part of the deployed binding state.

## Claim boundary

This promotes a bounded learned identity/binding boundary over a shared factual
bank. It does not establish open-set identity, unrestricted memory growth,
general learned delay policy, natural-language grounding, or general
continual learning. The next rung must vary encoders, stream counts, delay laws,
open-set arrivals, and contradictory evidence while preserving held-out factual
promotion and complete-retention gates.

Full accounting is in `sample_efficiency_ledger.json`; raw reports are in
`report_seed2301.json` and `report_seed2302.json`.
