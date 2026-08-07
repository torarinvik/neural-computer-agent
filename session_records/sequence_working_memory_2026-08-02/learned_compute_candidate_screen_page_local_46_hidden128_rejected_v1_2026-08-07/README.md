# Page-local 46-candidate hidden-width control rejected (2026-08-07)

This control keeps the replicated 46-candidate page-local configuration but
increases the factorized source scorer hidden width from 64 to 128. Known
source routing worsens to `0.8542/0.8021` on the two seeds, with a `0.0000`
per-target floor on both; all 26 unseen append pages remain perfect.

More router hidden capacity alone is therefore not the remedy. The next test
isolates source competition into independent memory pages behind the existing
scalar failure gate.
