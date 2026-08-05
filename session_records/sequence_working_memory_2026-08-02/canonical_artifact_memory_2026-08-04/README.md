# Canonical executable-artifact memory bridge

This is the first working-memory integration rung through the production
`neural_computer.ExecutableArtifactMemory` boundary. It migrates the existing
span-nine and span-ten learned growth states into a fresh canonical hot/cold
store, reloads the store, routes each learned address, and rehydrates the
artifact on the common frozen parent controller.

Command:

```bash
uv run python -m experiments.working_memory_continuous.audit_canonical_artifact_memory \
  --source-bank artifacts/memory/span_multi_skill_bank_seed49011 \
  --canonical-bank session_records/sequence_working_memory_2026-08-02/canonical_artifact_memory_2026-08-04/canonical_bank \
  --report session_records/sequence_working_memory_2026-08-02/canonical_artifact_memory_2026-08-04/audit.json \
  --count 32 --distractors 2 --seed 49011 --device cpu
```

The sub-minute smoke rung passes all mechanistic gates:

- controller weights unchanged: `true`;
- both routes exact after reload: `true`;
- span-nine direct and rehydrated accuracy: `0.954861`;
- span-ten direct and rehydrated accuracy: `0.912500`;
- zeroing either loaded artifact causes a causal accuracy drop;
- corrupted artifact is rejected by SHA-256 verification;
- the two addresses remain separate despite `0.96677` cosine similarity.

This proves persistence and causal executable-growth-state rehydration, not
cold-start procedure discovery or a population-level continual-learning claim.
The next rung is a matched fresh-process replication with a larger lifetime
count, followed by memory compaction and retention checks on the mastered
lower-order spans.

## Replication seed 49012

The matched 128-lifetime replication also passes all structural and causal
gates. Span-nine direct and rehydrated accuracy is `0.938368`; span-ten is
`0.882031`. Both routes are exact after reload, zeroing either loaded artifact
causes a causal drop, the parent controller remains bit-identical, and the
corruption snapshot is rejected. This replicates the memory-boundary result,
but remains an inherited-artifact result rather than a cold-start learning
promotion.

## Canonical frozen-core loader

The bridge now uses `load_growth_artifact` rather than the historical
rehydration helper. It accepts only `skill_`-prefixed tensors, hashes the
remaining controller state before and after loading, and rejects any artifact
that targets shared state. The same 32-lifetime smoke rung passes with this
enforced loader; its report is `canonical_loader_audit.json` and its bank is
`canonical_loader_bank/`.

The matched retention audit also passes across spans two through eight with a
two-point tolerance. Loading span-nine changes span-eight by `-0.0078` and
loading span-ten changes spans seven and eight by `-0.0045` and `-0.0156`; all
other measured changes are zero in the 32-lifetime audit. The report is
`canonical_retention_audit.json` and the bank is `canonical_retention_bank/`.
