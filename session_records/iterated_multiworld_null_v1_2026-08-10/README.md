# Iterated multi-world null (F124)

Probe 224. Iterating at 64 worlds makes things WORSE than one-shot
(trained programs 0.0077/0.0143 vs 0.1454/0.0679); stranger still
bit-identical to own. F121 fixed the single-world case with the same
change, where no per-world content is needed. Iteration multiplies
whatever the step function knows, including nothing.

First crack: withheld (0.0044) now below own/stranger — the model uses
"an entry exists" but not "which entry".
