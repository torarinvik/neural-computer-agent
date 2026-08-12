# Two-generation token-router growth at 66 candidates promoted (2026-08-07)

This audit tests repeated external growth rather than one append event. Thirty
source candidates live in three normalized pages. Two independent append
generations each add 18 candidates in nine raw pages, for 66 candidates across
21 pages. Each generation owns a separate token-preserving router trained only
on that generation's scalar verifier outcomes. Inference cascades from the
frozen source router to generation one and then generation two only after
verifier failure.

Both matched seeds pass strict `1.0000` candidate/page and per-target/per-page
mastery, full page permutation, generation-local reward-shuffled nulls, frozen
source router/pages, unchanged controller, no unresolved rows, and zero replay.
Each run uses 14,944 optimizer updates, 1,544,448 unique verifier bits, and
1,544,064 logical lifetimes. Mixed-stream mean fresh attempts are `1.8182`:
later generations correctly pay the verifier-gated cascade cost.

This promotes bounded repeated no-replay external growth. It does not establish
unrestricted memory growth, consolidation/compression, representation
selection, or general continual learning.
