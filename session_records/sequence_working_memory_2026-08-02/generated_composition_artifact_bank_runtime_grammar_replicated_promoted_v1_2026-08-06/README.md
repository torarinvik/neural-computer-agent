# Replicated runtime-grammar append-only acquisition (2026-08-06)

Status: replicated promoted mechanism-transfer result.

The generated-composition renderer and route-key path now accept a
verifier-private runtime grammar rather than requiring programs to be present
in the default static table. The audit supplied four four-primitive programs
at runtime:

1. `forward -> reverse -> complement -> rotate`
2. `rotate -> complement -> reverse -> forward`
3. `complement -> rotate -> forward -> reverse`
4. `reverse -> forward -> rotate -> complement`

These programs were not entries in the prior two- and three-primitive default
grammar. The controller still received only rendered learned events; program
specifications were used only by the verifier-private renderer and outcome
generator. Each artifact was acquired in an isolated external stack, admitted
only after stable fresh retention outcomes, and routed through the frozen
append-only chain.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| runtime artifact 0 behavior | 0.9219 | 0.9375 |
| runtime artifact 1 behavior | 0.9141 | 0.9648 |
| runtime artifact 2 behavior | 0.9609 | 0.9609 |
| runtime artifact 3 behavior | 0.9766 | 0.9492 |
| causal route accuracy | 1.0000 | 1.0000 |
| candidate-key permutation accuracy | 1.0000 | 1.0000 |
| cold-start old-route accuracy | 1.0000 | 1.0000 |
| stage-specific shuffled controls | 0.0000 | 0.0000 |

All artifact rows were stable and protected. Reload, corruption rejection,
frozen-core, and zero-replay gates passed in both runs. A short 16/32-update
rung correctly refused the first append because the new artifact was not yet
protected; the full acquisition budget was required for this longer program
family.

This promotes replicated acquisition across a runtime-supplied grammar and
four-primitive computation length. It is stronger evidence for mechanism
transfer than adding another predeclared composition ID, but it remains
bounded external growth: the artifact blueprint and append-only capacity are
finite, and arbitrary unconstrained program induction is not yet shown.
