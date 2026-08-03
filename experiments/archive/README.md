# Archived experiments

This directory contains historical experiment code retained for reproducible
replay and evidence inspection. It is not part of the production agent API or
the canonical test suite.

New runtime code belongs under `src/neural_computer/`. New experiments belong
under `experiments/` and should depend on the production package rather than
the archived controller.

Run the archived suite explicitly with:

```bash
./scripts/test_archived_experiments.sh
```
