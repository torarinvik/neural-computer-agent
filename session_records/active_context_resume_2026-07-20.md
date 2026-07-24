# Active-context neural computer — resume note

## Verified state

- Local regression suite: **130 passed, 15 subtests passed**.
- `PersistentMemory.select(indices)` copies selected valid latent rows into a compact active store.
- `ActiveContextSelector` consumes only recurrent sensory latents and latent memory keys/values.
- Controller weights remain frozen during selector training.
- Conservative calibration requires query and audit accuracy/loss parity on two held-out splits.
- Three-seed RTX 5090 replication completed using parallel workers with `OMP_NUM_THREADS=32` and
  `MKL_NUM_THREADS=32` per worker.

## Empirical result

The selector sometimes beats a matched random row, but does not reliably identify useful context.
The larger replication therefore selected the safe full eight-row context for all three seeds:
accuracy was preserved exactly, but average context savings were zero. The oracle one-row control
remains much stronger, so the memory-transfer mechanism works and selector representation/credit
assignment is the current bottleneck.

Reports and checkpoints:

- `experiments/syllogimous_neural_computer/targeted_adaptive_context_replication/`
- `experiments/syllogimous_neural_computer/targeted_recurrent_context_parallel/`

## Recommended continuation

Train a variable top-k selector (or jointly improve the sensory query representation) instead of
forcing a one-row decision. Keep the conservative calibrator and full-context fallback so every
change is required to preserve correctness before claiming memory savings.

## Re-run checks

```sh
.venv-vlm/bin/python -m pytest experiments -q
python3 experiments/syllogimous_neural_computer/summarize_adaptive_context.py \
  'experiments/syllogimous_neural_computer/targeted_adaptive_context_replication/seed_*.json'
```
