# Signed external-entry search audit

This experiment verifies the live architecture seam added after the
factorization audit. A signed external value model is trained only on
positive opaque entries, frozen, and passed into factual beam search. Two
otherwise identical candidate sets receive opposite entry assignments; the
planner must select opposite opaque intentions because of the external
entry, not because the transition model or controller changed.

The matched control uses the same transition model and candidate ordering but
has no entry-value model. It is therefore polarity-insensitive and cannot
follow both external regimes. The verifier is a private scalar outcome
function; no correct action or semantic label enters the planner.

This promotes live signed-entry search, not arbitrary value learning,
unrestricted memory growth, or general continual learning.
