# Promoted online partial-evidence context admission rung

Two seeds (`69601`, `69602`) passed an interleaved-stream pressure test. The
online resolver received one verified opaque transition at a time, kept the
first two observations of each stream unwritten and uncertain, then admitted
each stream on its third observation. A duplicate stream reused its existing
address without growing the store.

A reversal on an already-bound stream produced a conflict with zero writes on
the first contradiction. The second contradiction admitted a new address;
the old address remained factually exact. Wrong-context and corrupted-memory
controls produced non-zero next-state error, fresh memory produced zero hits,
and persistence/controller-immutability checks passed.

This is still bounded memory-side admission, not general continual learning.
The resolver uses an opaque stream binding and fixed evidence thresholds; it
does not yet infer context from raw modalities, learn the thresholds, cluster
arbitrary partial evidence, or compress unbounded history.
