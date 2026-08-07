# Token-level append-router overlay cost boundary promoted (2026-08-07)

This reproduces the promoted 64-candidate no-replay growth mechanism with a
reduced append-router budget. Thirty protected source candidates remain in
three normalized pages; 34 new candidates occupy 17 raw external pages. The
source router/pages stay frozen, while a token-preserving append router learns
from scalar outcomes and is reached only after verifier failure.

Both matched seeds pass strict `1.0000` candidate/page and per-target/per-page
mastery, full page permutation, reward-shuffled null, frozen source/router
state, unchanged controller, verifier fallback, and zero replay. Each run uses
3,072 updates for each append router, 8,768 optimizer updates total, 1,469,952
unique verifier bits, and 1,469,568 logical lifetimes.

This supersedes the 4,096-update promotion as the current cost boundary. It is
still bounded no-replay external page addressing, not arbitrary memory growth,
compression, learned representation selection, or general continual learning.
