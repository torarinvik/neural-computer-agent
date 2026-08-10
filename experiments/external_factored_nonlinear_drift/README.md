# Nonlinear external residual learning with gradual drift

This pressure test extends the learned factored residual boundary to a
nonlinear residual relation and a later small drift. A shared nonlinear base
is trained on opaque source transitions and frozen. A target regime receives
only `20/24` current rows; its residual function is learned in an isolated
external random-feature basis and verified on four held-out rows. Six new
drift rows then update the already-bound target slot, with four different
verifier-private drift rows held out and the old target behavior retained as a
separate gate.

Source/target/drift bundles alternate after promotion. A corrupted drift
update must be rejected without changing the committed digest. The controller,
base, and context encoder remain frozen, and target/drift adaptation performs
no old-regime replay. The promoted learner uses a fixed nonlinear feature map
and accumulates only normal-equation sufficient statistics; each new row is
consumed once. A fresh target model is measured using actual held-out loss,
but no positive-transfer claim is made from this fixture.

An earlier MLP variant was rejected: it preserved the old target within the
coarse safety tolerance but did not beat the frozen-base control reliably on
fresh seeds after sparse no-replay drift updates. That failure is why this
fixture uses the replay-free sufficient-statistics backend rather than
silently promoting the optimizer-based variant.

Run one seed with:

```bash
PYTHONPATH=src uv run python experiments/external_factored_nonlinear_drift/train.py \
  --seed 81021 --report-out /tmp/external-factored-nonlinear-drift.json
```

The result is bounded to smooth synthetic nonlinear dynamics, a fixed feature
basis, and finite external capacity; it does not establish arbitrary
computation, unrestricted growth, or general continual learning.

The sparse-identity continuation is archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_nonlinear_drift_sparse_identity_promoted_2026-08-10/`.
