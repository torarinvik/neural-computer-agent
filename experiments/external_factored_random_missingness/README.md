# Factored factual memory under randomized partial windows

This pressure test keeps the controller, shared transition base, and context
encoder frozen while four opaque nonlinear regimes arrive through seeded random
partial windows. Each regime supplies two disjoint seven-row windows from a
fourteen-row observed set; the random-feature residual learner consumes each
observed row once and is promoted only against four independent held-out rows.

After acquisition, three-row random partial reads must route to the stable
logical slot through factual prediction or the persisted sparse factual
overlap index. A mixed-regime read must remain ambiguous. Persistence,
controller/base/encoder immutability, and zero old-regime replay are required.

This is a bounded randomized-missingness identity result, not arbitrary real
multimodal missingness, learned semantic identity, unrestricted growth, or
general continual learning.

Run one seed with:

```bash
PYTHONPATH=src uv run python -m experiments.external_factored_random_missingness.train \
  --seed 83041 --report-out /tmp/external-factored-random-missingness.json
```

The five-seed promoted reports are archived in
`session_records/sequence_working_memory_2026-08-02/external_factored_random_missingness_promoted_2026-08-10/`.
