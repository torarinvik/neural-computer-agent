# Append-only residual compute for opaque procedures (2026-08-07)

The adapter-only shared residual bank could not learn a genuinely opaque new
procedure after its shared basis was frozen. This audit adds a compact local
recurrent compute encoder to each appended residual slot while retaining one
shared context encoder, immutable older slots, and external recurrent state.

Configuration:

- two runtime-generated `opaque_rule` procedures;
- shared context hidden/width `64/32`;
- local residual compute hidden/width `32/16`;
- 256 fresh-outcome updates per slot;
- four fresh retention probes;
- no replayed examples.

Both seeds `69316` and `69317` promote. The new opaque slot reaches reloaded
behavior `0.8906` and `0.9023`, while the first slot remains `1.0000`. The
shared-base digest and all old-slot digests remain unchanged; exact reload,
deliberate residual corruption detection/recovery, frozen-core equality, and
zero-replay gates pass.

This is a genuine capability gain over the adapter-only control, which ended
at `0.6250` on the same opaque pair. It is still bounded continual-memory
growth: each unrelated procedure may require a local compute slot, and the
two-slot payload is `0.8221` of two independent full programs. The next
pressure test is a third opaque procedure and then capacity-aware reuse of
local compute modules.

Run command:

```bash
PYTHONPATH=src uv run python -m experiments.generated_composition_capability_amodal.train_shared_residual_bank \
  --program-seed 4242 --program-count 3 --primitive-family opaque_rule \
  --source-ids 0 1 --residual-compute \
  --residual-context-hidden 32 --residual-context-width 16 \
  --parent-updates 64 --slot-updates 256 --batch-size 16 \
  --audit-count 64 --retention-probes 4 --eval-every 32 \
  --torch-threads 1 --report-out /tmp/shared-residual-compute/report.json
```
