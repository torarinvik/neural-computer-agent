# Replay-free adaptive control-flow growth

This audit promotes a memory-side curriculum for the external CPU/files
boundary. A generic control-flow frontier starts from a length-two root and
expands its allowed program horizon by exactly one instruction only after a
retention probe. A scalar-verifier-qualified longer file is then promoted as
the next search root. Operator aggregate credit and opaque candidate digests
carry forward; raw verifier rows and controller updates do not.

Across seeds `17–20`, both forward and reversed verifier-state order reached
three successive longer programs at lengths `3`, `4`, and `5`. Every promoted
program scored `1.0000` on held-out initial states, and every earlier external
program remained at `1.0000` after later growth. State and memory reloads were
exact; corrupted growth state and missing evidence were rejected without a
write; shuffled feedback qualified zero rungs; and replayed examples and
optimizer/controller updates were zero.

The matched fresh source-to-length control found the length-3 and length-4
targets but exhausted 600 candidate evaluations at length 5 in all four
seeds. This is a diagnostic curriculum-efficiency signal, not a formal
warm/fresh transfer promotion, because the fresh arm did not cross its final
mastery gate.

Positive arms charged `3,780` verifier bits across `756` candidate lifetimes;
shuffled controls charged `4,800` bits across `960` lifetimes; fresh controls
charged `17,475` bits across `3,495` lifetimes. The combined audit charged
`26,055` bits and `5,211` logical lifetimes, with zero optimizer updates and
zero replay.

This promotes bounded replay-free adaptive-horizon structural growth with
retention. It does not establish efficient arbitrary program synthesis,
unrestricted execution, unrestricted memory growth, or general continual
learning.

The runnable audit is
`experiments/recipe_expressibility/control_flow_adaptive_growth.py`.
