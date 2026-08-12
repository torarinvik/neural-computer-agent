# Held-out composition transfer — promoted

Four opaque instructions are acquired sequentially for four verifier-side
primitive procedures. The shared register interpreter is then frozen. A new
held-out composition/order uses the existing instruction vectors:

`prefix_parity → complement → reverse → adjacent_xor`

Only a fresh decoder is trained for the inherited path; no new instruction
code is added. A matched fresh interpreter of identical size learns the same
composition directly. Both replicated seeds `69316` and `69317` promote.

Inherited stable composition mastery is `8,192` verifier bits on both seeds.
Fresh learners require `12,288` bits on both, giving a replicated `1.5x`
fresh-over-inherited transfer ratio. Final inherited composition accuracies
are `0.8867` and `0.8750`. All source-retention, composition, shuffled-null,
missing-evidence, exact-reload, checksum-corruption, frozen-parent, and
zero-replay gates pass.

The matched 256-update source rung is retained as a control: it ties the
transfer gate on seed `69316` and leaves one source below mastery on seed
`69317`. The promoted result is bounded reusable compositional computation,
not arbitrary program induction or general continual learning.
