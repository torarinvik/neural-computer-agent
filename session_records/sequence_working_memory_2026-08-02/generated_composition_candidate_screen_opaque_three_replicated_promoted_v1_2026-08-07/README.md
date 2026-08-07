# Learned-event compute-candidate screening (2026-08-07)

This audit adds `ExternalComputeCandidateScreen` to the fresh-verified
reusable-compute admission path. The screen receives only a learned event
summary, opaque physical-candidate indices, and scalar verifier outcomes. It
orders future trials and may stop after the first candidate that clears the
fresh probe floor; it never authorizes reuse by itself.

The three-procedure opaque pressure test passes at both promoted seeds:

- seed `69316`: final behavior `1.0000/0.8516/1.0000`, with two physical
  compute modules and three logical bindings;
- seed `69317`: final behavior `1.0000/0.7500/0.9961`, with one physical
  compute module and three logical bindings.

Matched no-screen controls produce identical final and reload behavior for
each seed. When seed `69316` reaches an ambiguous two-module bank, screening
stops after the first fresh pass and reduces optimizer updates from `1344` to
`1088` (19.0%), and unique verifier bits from `124928` to `100352`. Seed
`69317` has only one physical candidate, so the screen is correctly neutral:
both policies use `832` optimizer updates.

All promoted gates pass: frozen controller, old-binding retention, exact
reload, corruption detection/recovery, candidate-screen persistence, and zero
replay. This promotes a conditional trial-cost reduction for bounded external
compute reuse. It does not establish general continual learning, unrestricted
memory growth, or arbitrary new computation. Full accounting and paired
controls are in the JSON reports and `sample_efficiency_ledger.json`.
