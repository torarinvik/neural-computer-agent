# Outcome-only external recipe growth — transfer rejected

This audit tests the new external recipe-file bridge. A generic sequence
search receives only scalar exact-match verifier outcomes, while a separate
memory bank admits stable candidates, protects earlier files, and reloads its
checksummed state. Candidate history is scoped by opaque external binding keys.

Both seeds acquired two auxiliary files and an order-sensitive target file.
Source and auxiliary retention, target held-out mastery, reversed-order
rejection, shuffled-feedback rejection, persistence, and zero-replay gates all
passed. The shared scalar edit prior did not improve the target learning curve:
warm target search required 74 versus 28 proposals on seed 17 and 34 versus 23
on seed 18, compared with fresh controls. The transfer claim is therefore
rejected; the scope-isolation and file-admission ABIs are retained.

Run with:

```text
PYTHONPATH=. uv run python experiments/recipe_expressibility/outcome_only_sequence_growth.py --report-out report.json
```

The stored summary is intentionally metadata-only; raw verifier rows are not
retained.
