# Frozen-controller relation-memory breakthrough — 2026-08-01

## Question

Can a frozen controller reuse its visual same/different representation while
external memory acquires a new, context-conditioned action convention?  Each
visible context privately maps the relation to normal or reversed opaque
actions.  The controller, vision encoder, and decoder are immutable; only
serializable memory rows change.

## Memory-only mechanism

For each context, the agent attempts opaque actions on two complementary
relation supports and their alternatives.  The memory receives only the
controller-produced event key, the attempted action, and the scalar success or
failure outcome.  Successful action intentions are stored as values.  At query
time a generic content-addressed disk read returns a value to the existing
frozen output decoder.  The private context ID and target action are used only
by the verifier to generate the scalar outcomes and score the audit.

This is a non-parametric external-memory baseline.  It intentionally does not
claim that the native recurrent `retrieved_memory` vector path can compose a
new relation; that remains a separate interface question.

## Three independent seeds

All runs used 1,024 held-out contexts, 4,096 stored rows, actual disk
save/reload, and MPS.  Every controller parameter hash was identical before
and after the run.

| seed | disk | reversed | flip rate | no memory | shuffled | corrupted |
|---:|---:|---:|---:|---:|---:|---:|
| 19001 | 98.14% | 98.14% | 100% | 49.61% | 48.14% | 51.95% |
| 19002 | 98.14% | 98.14% | 100% | 49.80% | 49.51% | 48.24% |
| 19003 | 98.34% | 98.34% | 100% | 49.22% | 52.83% | 48.63% |

Every pre-registered gate passed on every seed.  The complete JSON outputs are
`seed19001.json`, `seed19002.json`, and `seed19003.json` in this directory.

## Interpretation

This closes the strongest current frozen-weight claim: a visual relation
primitive can be retained in the frozen controller while a new context-private
action rule is acquired entirely in external memory.  The result is causal:
memory removal, value shuffling, and value corruption all collapse to chance,
and pixel-identical private-rule reversal flips predictions.

The next frontier is to move the same capability into the native latent
memory-read path—without storing protocol action prototypes as values—and then
to compare verifier bits against a reset learner.  That will test whether the
controller's amodal memory interface itself supports compositional rule
acquisition, rather than relying on a non-parametric action cache.
