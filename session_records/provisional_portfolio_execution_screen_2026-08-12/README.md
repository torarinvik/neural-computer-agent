# Portfolio-aware provisional execution screen

The online router already staged several opaque model-family hypotheses, but
`provisional_model_at()` exposed the first family statically until promotion.
This screen added factual one-step portfolio selection: the isolated model
currently exposed to the caller is the lowest-error hypothesis, while every
family remains available for held-out promotion. The canonical shadow-bank
probe now also records the selected family before copying its state.

The hard n-back-5 active-discovery rung was rerun for seeds `80–87` with the
same masked context, tight route matching, active probe, and accounting as the
existing baseline. The portfolio boundary reached `5/8` complete gates, with
`146` unique verifier bits and `234` transition rows consumed once. The
controller stayed unchanged, source retention stayed byte-stable in every
run, and replay remained zero. This matches the existing baseline gate count;
it is therefore retained as an interface-correctness improvement, not
promoted as a learning or sample-efficiency gain.

The rejected cumulative-error variant was also tested during development. It
fell to `4/8` complete gates and was not retained.
