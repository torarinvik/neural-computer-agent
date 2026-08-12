# Staged growth with runtime-generated opaque operators (2026-08-07)

Status: audit-history artifact superseded by the strict isolated-slot rerun.

The generated-composition interface accepts verifier-private `rule:xx`
tokens. Each token denotes an 8-bit local neighborhood rule selected from a
256-member family. The event renderer exposes only a generic binary barcode
and ordinal band; the controller never receives the rule token, truth table,
task ID, or verifier answer. This predecessor used an all-slot acquisition
mask, so its results do not prove that a new procedure learned in fresh
capacity. The authoritative strict rerun is in
`generated_composition_opaque_rule_sequential_slot_growth_three_source_strict_isolated_replicated_promoted_v1_2026-08-07/`.

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

Every original promotion gate passed in both seeds, but the predecessor did
not include the fresh-slot-only causal gate. Each replica consumed `323,584`
unique verifier bits, `108,544` logical lifetimes, `3,456` optimizer updates,
and `52` retention observations. Wall time was `389.7s` and `397.5s`.

It is evidence for the historical permissive route path, not an isolated-slot
continual-learning promotion. It is still not arbitrary Turing-complete
program synthesis: the local-rule family, eight-step renderer, slot blueprint,
and tested horizon remain bounded.
