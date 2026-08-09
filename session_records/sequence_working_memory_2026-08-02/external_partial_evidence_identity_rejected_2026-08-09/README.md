# Variable-prefix online context identity — rejected

This rung trained the context encoder to align noisy prefixes of source
transition bundles with their full opaque keys, then reduced online admission
to seven rows out of fourteen. The router's new active-stream continuation
state prevented immediate duplicate-slot admission and source slots were
write-protected.

The identity gates mostly worked: both seeds formed short-prefix admissions,
source slots stayed at `1.0` and byte-stable, and wrong-context factual error
passed. The capability gate did not:

| metric | seed 70511 | seed 70512 |
| --- | ---: | ---: |
| target-C warm updates | 80 | 50 |
| target-D warm updates | 80 | 40 |
| target-C fresh updates | 35 | 36 |
| target-D fresh updates | 46 | 37 |
| target-C mastery | 0.0 | 1.0 |
| target-D mastery | 0.333 | 0.333 |

Verdict: reject promotion. Short-prefix context identity is not sufficient
for fast acquisition; the selected model needs a principled credit-assignment
and evidence-accumulation policy before it can learn a full disjoint dynamics
regime from a stream. The router change is retained as infrastructure, but
this experiment is not evidence of solved streaming continual learning.

Reports are protected by `SHA256SUMS`.
