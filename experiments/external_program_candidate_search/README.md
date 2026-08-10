# Outcome-only executable-program candidate search

This audit closes the next gap after transactional executable-file admission.
The shared register interpreter is pretrained once and frozen. One opaque
instruction file is protected as the source capability; a held-out target is a
two-instruction composition that is not present in external file memory.

`ExternalProgramCandidateSearch` proposes generic structural edits—replace,
insert, delete, swap, and jitter—from an opaque instruction bank. It receives
only ordered scalar verifier scores. Failed candidates remain provisional and
are not admitted; only a candidate with a stable verifier suffix enters the
external file bank. Search statistics retain aggregate reward and acceptance
counts, never raw verifier rows or target programs.

The fresh control starts from a different opaque atom. With the one-edit
search budget, it cannot reach the held-out two-step target, while the
protected source can discover it by insertion. This is deliberate bounded
parent-conditioned synthesis evidence, not a claim of unrestricted program
induction.

Run a seed with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_program_candidate_search/train.py \
  --seed 23001 \
  --report-out /tmp/external-program-candidate-search-23001.json
```

The promotion boundary is bounded outcome-driven structural program search
with frozen shared computation, not arbitrary new computation, universal
program synthesis, Turing-complete learning, or general continual learning.
The next pressure is multi-step beam search and a genuine Brain Workshop
family stream rather than a one-edit held-out composition.
