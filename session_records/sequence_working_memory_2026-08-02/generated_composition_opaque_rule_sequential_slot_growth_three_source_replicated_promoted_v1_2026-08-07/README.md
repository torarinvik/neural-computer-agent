# Staged growth with runtime-generated opaque operators (2026-08-07)

Status: replicated promoted primitive-family expansion.

The generated-composition interface now accepts verifier-private `rule:xx`
tokens. Each token denotes an 8-bit local neighborhood rule selected from a
256-member family. The event renderer exposes only a generic binary barcode
and ordinal band; the controller never receives the rule token, truth table,
task ID, or verifier answer. The staged external-memory protocol then learns
three eight-step procedures one new slot at a time.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| staged procedures | `3` | `3` |
| staged rewrites adopted | `2/2` | `2/2` |
| final source reload behavior | `1.0000/0.8867/1.0000` | `1.0000/0.8945/0.8984` |
| target reload behavior | `1.0000` | `1.0000` |
| target fresh behavior | `0.9844` | `0.9805` |
| reversal/recovery | passed | passed |
| corruption/reload/frozen-core | passed | passed |
| replayed examples | `0` | `0` |

Every promotion gate passed in both seeds. Each replica consumed `323,584`
unique verifier bits, `108,544` logical lifetimes, `3,456` optimizer updates,
and `52` retention observations. Wall time was `389.7s` and `397.5s`.

This is evidence that the external memory boundary can acquire and retain
procedures built from an unseen parameterized operator family, rather than
only recombining the seven named primitives. It is still not arbitrary
Turing-complete program synthesis: the local-rule family, eight-step
renderer, slot blueprint, and tested horizon remain bounded.
