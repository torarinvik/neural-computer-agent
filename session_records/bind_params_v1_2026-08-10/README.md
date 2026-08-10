# Bind once (F135)

Probe 235. Decode the entry ONCE into per-piece parameters instead of
re-attending it at every step. Boolean, 256 worlds, depth<=4,
held-out worlds:

  re-attend + oracle (F134): 0.5548 per-bit, 0.0063 exact
  BIND + oracle (F135):      0.9983 per-bit, 0.9872 exact
  BIND + oracle, stranger:   0.5474  (effect is causal)
  BIND + learned reader:     0.5283  (own == stranger)

Conditioned execution at depth is solved. The only remaining gap is
the reader.
