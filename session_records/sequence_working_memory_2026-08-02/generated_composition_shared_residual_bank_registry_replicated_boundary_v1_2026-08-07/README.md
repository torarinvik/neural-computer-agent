# Shared residual capability-bank boundary (2026-08-07)

This record audits a frozen amodal controller with an external neural memory
bank consisting of one shared context encoder plus isolated residual slots and
per-slot opaque decoders.

The first slot is trained on fresh outcomes. Before every later slot is added,
the shared context encoder and all older residuals are frozen. The new slot is
trained only on fresh outcomes for its current opaque procedure plus a fresh
auxiliary stream. The controller receives no procedure IDs, raw modality
formats, correct actions, or verifier-private grammar metadata.

The promoted boundary is replicated for two related registry procedures:

- seeds `69316` and `69317` both master both procedures;
- old-slot behavior remains at `1.0000` during growth and after reload;
- reload is exact and the frozen controller digest is unchanged;
- replayed examples are `0`;
- the shared bank plus decoders use `0.5556` of two independent full
  program-plus-decoder payloads;
- the second procedure reaches a stable threshold at `2,048` verifier bits in
  both runs.

The controls define the claim boundary rather than being hidden:

- the same two-slot bank trained on two genuinely opaque random procedures is
  rejected: the new slot ends at `0.6250` while the old slot remains `1.0000`;
- a third heterogeneous registry procedure is rejected at `0.5430`, after the
  first two remain retained;
- all rejected runs still pass old-slot isolation, exact reload, frozen-core,
  and zero-replay checks.

This establishes genuine shared computation for related procedures, not
arbitrary new computation. The current bottleneck is a common learned basis:
an adapter-only residual cannot create enough sequential computation for a
newly unrelated opaque rule after the shared base is frozen. The next
architecture pressure test should add append-only residual compute capacity or
verified compressed behavioral summaries, while retaining the old basis and
old capabilities immutably.

Run command:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_shared_residual_bank \
  --program-seed 4242 --program-count 3 --primitive-family registry \
  --source-ids 0 1 --parent-updates 64 --slot-updates 128 \
  --batch-size 16 --audit-count 64 --retention-probes 4 \
  --eval-every 32 --torch-threads 1 \
  --report-out /tmp/shared-residual/report.json
```

Evidence files in this directory include both promoted replicas and the
decisive opaque/two-to-three-slot rejection controls.
