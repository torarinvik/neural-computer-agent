# Promoted learned transition-evidence admission rung

Two seeds (`69701`, `69702`) show that an external learned evidence evaluator
can make context reuse tolerant to small noise in learned next-state tensors.
The noisy duplicate was accepted as reuse, performed zero writes, and left the
source memory prediction exactly unchanged. The fixed exact-match resolver
rejected the same noisy evidence and allocated a duplicate context.

Contradictory evidence still admitted a new context, wrong-context and fresh
memory factual controls passed, persistence passed, and the controller stayed
unchanged. Target adaptation used zero optimizer updates and zero replayed
examples.

The evaluator itself was pre-trained on 1,024 synthetic verifier rows for 500
updates; its repeated training-row use is recorded explicitly in each report
(`510,976` replayed training rows). This is therefore a boundary robustness
promotion, not evidence of replay-free learning of the evaluator.

The result remains bounded. The evaluator is trained from synthetic opaque
transition tensors and does not yet learn context from raw modalities, adapt
its threshold online, or solve unbounded memory consolidation.
