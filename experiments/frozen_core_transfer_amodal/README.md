# Parent-conditioned frozen-core external transfer

This experiment measures whether an already-trained controller makes a new
procedure cheaper to acquire when its core is frozen and only an external
growth slot is trainable. The promoted configuration gives the slot recurrent
temporal state, the frozen controller's learned intention, and a generic
context-conditioned output gate. These are opaque learned tensors, not task
or modality branches.

The inherited arm and a cold-start arm use the same growth architecture,
target stream, verifier seeds, update budget, and held-out prefixes. The
cold-start arm is deliberately generous: all of its parameters remain
trainable, so a positive transfer result must beat a fresh learner with more
plasticity. Both arms receive fresh parent-task rehearsal episodes; no old
examples are replayed.

The strict promotion gates require stable-prefix mastery, retained parent
behavior, unchanged frozen-core digest, a parent-calibrated shuffled-outcome
control, a causal gap over that control, zero replay, and positive transfer
across both seeds. The promoted evidence is archived in
`session_records/sequence_working_memory_2026-08-02/frozen_core_transfer_parent_conditioned_v1_2026-08-06/`.
