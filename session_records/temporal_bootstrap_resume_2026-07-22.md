# Temporal event-binder bootstrap resume — 2026-07-22

## Current conclusion

The audited event-snapshot diagnostic remains successful (about 95% held-out with true reversal
and shuffled-label controls). The integrated write-path binder is mechanically connected and
trainable, but the first four-epoch supervised-bootstrap stage has not ignited.

Final four-epoch results:

- 32,768 total lifetimes: 16,384 temporal and 16,384 balanced spatial/shape rehearsal.
- 1,024 total optimizer updates, but only 512 temporal updates carried the auxiliary rule loss.
- temporary-head mean accuracy 49.83%, loss 0.6944;
- held-out temporal compact behavior stayed near 50% after demonstrations;
- independent 2,048-lifetime rule probe: raw write 52.69%, compact row 52.05%, recall 51.90% best;
- gradients remained live and residuals bounded; training completed without numerical failure.

This is a bounded pre-ignition result. Prior ignition was near 640 signal-bearing updates for the
direct-colour control and roughly 1,000--1,400 for cached binders. The balanced rehearsal cycle
halved the number of signal-bearing updates, a distinction missed in the original runtime plan.

## Resumable state

Primary checkpoint:

`experiments/forward_transfer_attention/targeted_temporal_event_binding_integration/bootstrap_full.pt`

It contains the model, frozen consolidator, optimizer, temporary auxiliary head, completed epoch
count (4), history, and elapsed training time. Do not restart from a pilot or clean base. Continue
using:

`experiments/forward_transfer_attention/temporal_event_binding_bootstrap_continue_epoch8.supervisor.conf`

That stage resumes through epoch eight, reaching 1,024 temporal auxiliary updates. Run a fresh
write/compact/recall probe there. If still flat with healthy mechanics, continue only to epoch
eleven (1,408 temporal auxiliary updates) before treating the bootstrap as a fair negative.

## Cloud instance

SSH command:

`ssh -i /Users/torarinvikbjarko/.ssh/id_ed25519_vast_ai -p 21421 root@154.9.228.248 -L 8080:localhost:8080`

Repository: `/root/elisa-screenwatch`

Python: `/venv/main/bin/python`

GPU: RTX 5090. Instance cost at measurement: $0.3916/hour including storage.

All training and probe jobs are finished. The instance was intentionally not stopped automatically.

## Important files

- `experiments/forward_transfer_attention/TEMPORAL_BOOTSTRAP_PLAN.md`
- `experiments/forward_transfer_attention/TEMPORAL_INTEGRATION_LADDER.md`
- `experiments/forward_transfer_attention/README.md`
- `experiments/forward_transfer_attention/train_joint_adapter.py`
- `experiments/forward_transfer_attention/train_consolidator.py`
- `experiments/syllogimous_neural_computer/model.py`
- `experiments/forward_transfer_attention/targeted_temporal_event_binding_integration/`

The final report, periodic checkpoints for epochs 2--4, full logs, phase-one checkpoint and probe,
precision-retention audits, joint-reader pilot artifacts, and final independent probe are all in
the integration artifact directory.

## Integrity and methodology

- The model receives only rendered visual/audio streams, never private game state.
- The auxiliary labels are verifier-side and make this branch explicitly supervised-bootstrapped.
- The temporary head is not used for acceptance; all representation gates use newly initialized
  probes on held-out lifetimes.
- Exact no-op initialization, gradient flow, event ordering, logical-lifetime split isolation,
  shuffled labels, and true counterfactual replay are regression-tested/audited.
- Final capability still requires rule decodability, behavioral few-shot gain, reversal causality,
  memory-corruption dependence, high-precision spatial/shape retention, and thin-line graduation.

