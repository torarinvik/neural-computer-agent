# Promoted learned context-address admission rung

Two seeds (`69501`, `69502`) passed a bounded admission test with no supplied
context labels. A memory-side resolver received opaque transition bundles and
reused an address only when existing stored facts explained every verified
next state; otherwise it allocated a fresh opaque address.

Each seed discovered three addresses for three unique dynamics, reused the
address for a duplicate regime, retained all four regime views including a
reversal, and passed persistence, corruption, shuffled-context, fresh-memory,
and frozen-controller controls. The store grew to 36 factual rows, with zero
target optimizer updates and zero replayed examples.

The direct factual controls are authoritative: wrong contexts produced
non-zero next-state error and fresh memory produced zero factual hits. Some
wrong-context plans reached individual goals by chance, so behavioral mastery
alone is deliberately not used as the context-causality gate.

This is not general continual learning. Admission currently consumes a
complete verified transition bundle, requires exact stored facts, and uses an
opaque allocator for new handles. Learned partial-evidence clustering,
compression, and extrapolation remain open.
