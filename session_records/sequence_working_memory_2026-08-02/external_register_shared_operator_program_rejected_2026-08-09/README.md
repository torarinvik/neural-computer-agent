# Shared operator program: rejected transfer intervention

Date: 2026-08-09

Seeds: `69316`, `69317`

Mode tested: temporary `factorized_shared_operator_program`

The intervention gave each opaque instruction a short latent program over
common low-rank transition factors. It was intended to make a new operation a
composition of reusable atomic transitions while keeping the controller
frozen. It exposed no task names, protocol actions, execution positions,
correct answers, or verifier-private metadata to the deployed path.

## Result

The matched full audit used the same source and target budgets as the retained
shared-operator-basis screen: `576` joint source updates, `512` target updates,
two held-out direct targets, and two held-out compositions.

Seed `69316` passed all target behavior and retention gates, but inherited
learning was slower than fresh on both compositions: `40,960` versus `24,576`
stable verifier bits, and `24,576` versus `16,384`.

Seed `69317` failed one composition behavior gate (`0.7422` final accuracy,
`0.75` consolidation probe) and had no inherited stable prefix for that
program. Its other composition required `40,960` versus `16,384` fresh bits.

Both seeds retained source capabilities with exact zero retention deltas and
passed the causal controls where the target behavior gate passed. Each used
`1,067,008` unique verifier bits, `8,160` optimizer updates, and zero replayed
examples.

## Decision

Reject and remove the operator-program branch. Extra latent execution depth
increased optimization burden without producing reusable faster learning; it
is not evidence for general continual learning. The next intervention should
optimize the adaptation rule itself—meta-learned plasticity with protected
old-state boundaries—rather than adding untrained execution depth.

The full disposable reports were emitted under `/tmp/neural-computer-next/`:
`operator_program_interleaved_full512_seed69316.json` and
`operator_program_interleaved_full512_seed69317.json`.
