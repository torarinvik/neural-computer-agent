# Token-level append-router overlay at 64 candidates promoted (2026-08-07)

This is the promoted no-replay page-growth mechanism. Thirty protected source
candidates remain in three independently trained normalized pages. Thirty-four
new candidates are trained in 17 raw external pages. The source page router and
source pages are frozen; a separate append router retains every normalized
candidate token instead of collapsing a page to a lossy mean. It learns only
from scalar verifier outcomes of attempted append pages. A failed source
attempt gates fallback to the append router.

Both matched seeds pass strict `1.0000` candidate and page accuracy across all
64 candidates and 20 pages, every per-target/per-page mastery gate, source and
full page permutation, reward-shuffled append null, frozen source/router state,
unchanged controller, verifier fallback, and zero replay. Each run uses 10,816
optimizer updates, 1,887,744 unique verifier bits, 1,887,360 logical
lifetimes, and a mean of 1.375 fresh verifier attempts on the mixed audit.

This promotes bounded continual external page addressing without controller
updates or replay. It does not establish arbitrary memory growth, learned
representation selection, compression, or general continual learning. The
4,096-update append-router cost is substantial and is the next efficiency
target.
