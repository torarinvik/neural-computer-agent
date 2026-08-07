# Strict isolated-slot opaque-operator growth (2026-08-07)

Status: replicated promoted correction to the opaque-operator audit.

The earlier audit used an all-slot mask while training the appended source.
That allowed a new procedure to borrow frozen old slots, so its “isolated
slot” description was too strong. This rerun binds each new source to the
newly appended slot during acquisition and binds the resulting alias to that
slot after consolidation. Existing aliases retain their prior bindings.

The verifier-private family is unchanged: each `rule:xx` token denotes one of
256 eight-bit functions over a three-cell binary neighborhood. The renderer
exposes only a generic barcode and ordinal event band; the controller receives
neither rule tokens nor verifier answers.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| staged procedures | `3` | `3` |
| staged rewrites adopted | `2/2` | `2/2` |
| final source reload behavior | `1.0000/1.0000/1.0000` | `1.0000/0.9844/1.0000` |
| target reload behavior | `1.0000` | `1.0000` |
| target fresh behavior | `0.9844` | `0.9805` |
| new-slot bindings | `2`, `3` | `2`, `3` |
| reversal/recovery | passed | passed |
| corruption/reload/frozen-core | passed | passed |
| replayed examples | `0` | `0` |

Every strict promotion gate passed in both seeds. Each replica consumed
`323,584` unique verifier bits, `108,544` logical lifetimes, `3,456`
optimizer updates, and `52` retention observations. Wall time was `379.7s`
and `385.7s`.

This promotes replay-free acquisition in genuinely fresh external slots and
behavior-verified retention of earlier aliases. It remains bounded continual
memory over a finite operator family and eight-step horizon; arbitrary
program induction, unrestricted growth, and general continual learning remain
unqualified.

The permissive predecessor is retained as an audit-history artifact, but this
strict archive is the authoritative evidence for the isolated-slot claim.

The underbudget rung (`32/64/128/64` parent/source/consolidation/target
updates) was rejected before promotion because source mastery and retention
were not stable. Its safe rejection is recorded in
`report_strict_short_control_seed69316.json`.
