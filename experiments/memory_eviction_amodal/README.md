# Outcome-only learned eviction

This trainer qualifies a new memory-side utility beyond write/skip selection.
The controller first acquires a scalar-recall parent and is then frozen. A
separately versioned `ExternalMemoryEvictionPolicy` scores opaque candidate
rows from controller-native write context and memory tensors. When a two-row
bank is full, paired common-random arms force row 0 versus row 1 and train the
ranker only from their scalar recall outcomes.

The controller never receives physical row indices, candidate labels, target
slots, or counterfactual metadata. The backend owns row replacement through
the generic `MemoryCandidates` and `target_index` contract.

## v1 qualification (2026-08-05)

The parent was acquired on fresh randomized opaque event tokens, then frozen
before eviction-policy training. Seed 17 reached held-out balanced recall
`0.916`, target-first `0.903`, and target-last `0.981`; seed 69415 replicated
`0.963`, `0.912`, and `0.999`. Strength-based eviction was only `0.488` and
`0.512` on target-first, while random eviction reached `0.737` and `0.756`.
Both runs passed clear-memory, corruption, persistent reload, checksum
rejection, and recovery controls. The reward-shuffled control stayed at
chance (`0.526` balanced; `0.501` target-first) and never acquired a stable
parent. All runs used zero replayed examples.

This promotes a narrow, replicated learned-utility eviction boundary for a
three-slot/two-row synthetic verifier. It does not establish general
episodic memory, natural-modality transfer, arbitrary new computation, or
general continual learning. Reports and ledgers are archived under
`session_records/sequence_working_memory_2026-08-02/learned_eviction_v1_*`.

## Retention-safe composition boundary (2026-08-05)

The learned ranker can now be composed with the canonical
`CapabilityRetentionLedger`. The ranker still scores which opaque row is
disposable, while the ledger masks any row that has reached stable scalar
mastery. A protected row therefore wins against a high learned eviction score;
if every row is protected, the backend requests growth or verified
consolidation rather than overwriting one. Reversal hysteresis and the
transactional retained-score gate are memory-side state, so this composition
does not add a Brain Workshop/task branch to the frozen controller.

This is an implementation boundary and unit-tested safety result. It is not
yet a promoted Brain Workshop continual-learning result; that requires a
replay-free multi-rung acquisition audit with 1/5/6/7/8-back retention and
reversal controls.
