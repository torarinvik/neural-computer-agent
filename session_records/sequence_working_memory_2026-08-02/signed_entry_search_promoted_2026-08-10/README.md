# Promoted signed external-entry search

Three seeds (`9201`, `9202`, `9203`) pass the live-search audit. A signed
external value model is trained only on positive entries, frozen, and attached
to factual beam search. With the same opaque intentions and transition model,
reversing only the external entry assignment reverses the selected intention
on all three seeds. The matched planner without an entry-value model selects
the same first candidate in both regimes.

All seeds pass source mastery, both regime choices, polarity sensitivity,
baseline polarity insensitivity, exact persistence, frozen transition/value
models, zero target updates, and zero replay. This promotes the live
signed-entry search seam, not arbitrary value learning, unrestricted memory
growth, or general continual learning.
