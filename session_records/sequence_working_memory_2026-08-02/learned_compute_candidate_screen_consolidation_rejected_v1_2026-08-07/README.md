# Learned append-screen consolidation rejected (2026-08-07)

This is the first behavior-level audit of the transactional screen compaction
boundary. A twenty-candidate source bank has five isolated two-candidate
stages. A proposed four-candidate replacement for the first two stages is
trained from new scalar verifier outcomes plus opaque route-score distillation
from the source screen. Adoption requires fresh repeated probes for every
logical candidate, aggregate retention, permutation, and known-context
retention; the source and frozen controller must remain unchanged.

The proposal is rejected on both seeds. Seed `69316` has a fully mastered
source bank, but the compact replacement does not clear the per-candidate
retention gate. Seed `69317` already has two source candidates at `0.0000`
per-target retention under the strict audit, so compaction cannot be accepted
without first repairing source mastery. The naïve copied-stage replacement is
rejected on both seeds. No replayed examples are used.

This is a decisive negative result for the current compact replacement
training path, not evidence against the transactional API. The next work is
to solve per-candidate source mastery and behavior-preserving replacement
training before claiming learned consolidation or compression.
