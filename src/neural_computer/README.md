# `neural_computer`

This is the canonical production package. It owns only versioned neural-IR
contracts and modality-independent runtime composition:

```text
N encoders -> event-token window -> one controller/memory
           -> intention bus -> M decoders
```

Raw modality frontends and protocol backends are independently supplied by the
caller. Historical controller implementations are archived under
`experiments/archive/` and must not be imported by production code.

The public boundary is exposed from `neural_computer.__init__`. Component
checkpoints are loaded into caller-constructed encoders, controller, memory,
and decoders through `load_runtime_components`; checkpoint metadata never
constructs an implicit modality branch.

The controller also emits an execution-plane policy with three operational
states: `WAIT` keeps the intention tentative while transport may provide more
events, `THINK` spends a bounded quiet recurrent tick, and `COMMIT` releases
the current opaque intention to the output bus. A learned, age-gated timeout
residual can make a second decision after `WAIT` when evidence remains absent.
The controller also includes a bounded zero-initialized pairwise event-attention
residual for learned cross-token binding, plus versioned feedback/source-key
interactions for outcome-conditioned evidence binding. Runtime v27 carries the
generic learned source-credit policy: prior event tokens, opaque source keys,
and feedback produce a trust-space credit vector, gated by normalized source
attribution and averaged over present tokens before updating persistent source
trust. Its output bias is neutral at initialization, avoiding an unconditional
source preference and making the update scale independent of encoder count.
This is still one controller; the runtime only enforces the deliberation bound
and does not add a reasoning module.

Runtime v27 exposes a payload-only latest-event memory address with a residual
learned-event identity path that remains stable across recall age and
irrelevant prior events. The write policy receives retained latest-token pair
context plus average and strongest current-to-prior matches, so utility
decisions can condition on bounded event interference without a
modality-specific branch. During outcome-only training, v27 can optionally
sample a Bernoulli write decision
with a straight-through differentiable transaction and expose its opaque
log-probability for policy-gradient credit. This is training infrastructure,
not a deployed protocol. The pre-v22
cue-guided retention diagnostic remains rejected; the corrected query-cue and
latest-address qualification is recorded separately. The v73 outcome-only
retention/transfer rung is promoted for the narrow verifier after three-seed,
four-pair unseen-token, causal, persistence, and positive fresh-transfer
controls. The v76 three-seed v27 outcome-only three-slot/two-row retention
rung also passes balanced-position, unseen-token, causal, persistence, and
checksum controls. These results do not establish general episodic memory or
natural-modality capability.

Memory addresses use one shared learned projection plus a residual learned
event-identity path over the latest learned event payload for both writes and
reads. Transport metadata such as event age,
duration, timestamp presence, and confidence remains available to reasoning
and write utility, but cannot make the same event address differently at
recall time. v23 checkpoints migrate with their transport-augmented address
behavior; v24 checkpoints migrate with the feedback residual disabled, and v25
and v26 checkpoints migrate with their prior address behavior. New checkpoints
are v27.

Memory is a replaceable `MemoryBackend` v1 contract. The default
`ContentAddressedMemory` keeps a bounded content-addressed index in the
runtime, while `PersistentContentAddressedMemory` atomically snapshots and
checksums the same learned keys, values, strengths, timestamps, and version to
disk. Query alignment remains differentiable through read scores and weights;
inside an explicit training transaction, pending values also expose a
differentiable write-strength gate while persisted rows stay detached state.
Query, read, and write-receipt records
are schema-validated, matching keys are upserted instead of duplicated, failed
durable writes restore the prior in-memory state, and runtime checkpoint loads
validate and roll back memory components as well. A narrow scalar outcome-recall
rung now passes clear-memory, corruption, persistent-replacement, four-pair
unseen-token, and matched fresh-transfer controls for the narrow verifier. The
corrected v76 three-slot/two-row rung also passes its balanced-position,
unseen-token, causal, and persistent-memory gate; this remains a narrow
outcome-only claim rather than general episodic memory.
For batched independent trajectories, an optional opaque `memory_scope` selects
one of fixed-capacity isolated banks without entering the learned key/value
content; the legacy single-scope layout and checkpoints remain compatible.

The original v74 three-slot/two-row rung is retained as a superseded harness
record because its duplicated counterfactual arms did not preserve balanced
target positions. The corrected v76 rung now qualifies bounded learned
multi-row retention. The v75 synthetic cross-adapter rung qualified the
two-row neural-IR case. The v77 three-seed rung now also qualifies three-row
outcome-only retrieval with an opaque target cue, cued-row-last presentation,
and persistent reload/recovery; all fresh-token and swapped-slot controls
pass. The three-slot/two-row bounded-interference variant also passes after
separate strict write and learned-IR read-match thresholds were introduced;
natural-modality alignment and broader episodic utility remain open. Evidence is in
`session_records/cross_adapter_memory_amodal_v77_2026-08-04/`.

The v78 cross-adapter follow-up also randomizes the target position within a
three-row sequence while retaining only two memory rows. The generic
counterfactual write-utility trainer stabilizes fresh-reader minima at
`0.991/0.988/0.998` across seeds, with persistent and memory-corruption
controls passing. This is still a bounded synthetic outcome-only result;
cue-conditioned utility, capacity-one compression, natural modalities, and
general episodic memory remain unqualified. Evidence is in
`session_records/cross_adapter_memory_amodal_v78_2026-08-04/`.
