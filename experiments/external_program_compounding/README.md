# External program-router compounding

This pressure test checks whether the CPU-like external program router can
continue learning from an already mastered route state. Two executable
programs are learned first. A target stream reuses those routes and introduces
a third opaque program. The warm arm transfers the external route state, uses a
bounded copy-on-write challenger against a fresh state, and continues only the
winner. A matched fresh arm starts with empty route state.

The register interpreter, executable artifacts, and cognitive controller are
frozen. The router receives only opaque event features and terminal scalar
execution outcomes. The verifier's hidden relation is used for evaluation and
accounting, never as learner input. Source rows are not replayed during target
adaptation; target rows may be reused by the challenger and continuation.

Run one seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_compounding/train.py \
  --seed 2401 --report-out /tmp/external-program-compounding-2401.json
```

The experiment is a pressure test for positive transfer, not a claim of
arbitrary program induction or general continual learning.

The promoted three-seed rung reached a stable target threshold in `628`
accounted router updates for warm cells versus `1,000` for matched fresh
cells. The old external source cell retained `95.67–98.67%` accuracy, and
shuffled outcomes failed. Full reports are archived under
`session_records/sequence_working_memory_2026-08-02/external_program_compounding_promoted_2026-08-10/`.
