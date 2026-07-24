# Syllogimous real-time experiment handoff

Last updated: 2026-07-18

## Local project

Project root: `/Users/torarinvikbjarko/Documents/Coding Projects/Elisa Projects/elisa-screenwatch`

The complete experiment code and checked-in artifacts are under
`experiments/syllogimous_realtime/`. The reproducibility record is
`experiment_manifest.json`; the worktree intentionally contains uncommitted
user changes.

## Deterministic validation

- Five-million max sweep: `challenge_validation_max_5m.json`
- Result: 5,000,000 tasks, 90 families, 0 failures, ~92,800 tasks/sec
- Seed/failure manifest: `challenge_validation_max_5m.seeds.json`
- Native cross-check: 10,000 records, 0 mismatches
- Boundary leak report: `boundary_leak.json`, 256 hidden-field variants, 0 RGB/PCM divergences
- Python suite: 26 tests passing

## Cloud instance

Safe reconnect command:

```sh
ssh -i ~/.ssh/id_ed25519_vast_ai -p 10812 root@87.116.91.146 -L 8080:localhost:8080
```

Remote project: `/root/elisa-screenwatch`
Hardware: NVIDIA H100 NVL, approximately 96 GB VRAM, CUDA 13.x

The instance workspace is not a persistent volume. Important checkpoints and
reports from it have already been copied into the local experiment directory.
Stop the instance when finished to avoid GPU charges.

## Cloud results

- `cloud_model_results.json`: 256M and 500M models timed out; 2.2B reached the conclusion but answered incorrectly in the one-episode tests.
- `adapter_256m_h100_20ep.pt` and `.json`: 20-episode frozen-listener adapter run; 0/20 correct, 20/20 timeouts, approximately one emitted modality per packet.
- `adapter_matrix_h100_256m.json`: dense/fixed/random/learned comparison, all four timed out in the one-episode comparison.

## Resume priorities

1. Improve the visual action protocol/prompt so the model stops advancing once it reaches the conclusion.
2. Run multi-episode cloud evaluations with the 2.2B model and larger deadlines where appropriate.
3. Train/evaluate the streamer adapter against a listener that can answer at least a meaningful fraction of episodes.
4. Add compatible audiovisual checkpoints for the currently unassigned 1M–100M slots, or keep them explicitly unassigned.
