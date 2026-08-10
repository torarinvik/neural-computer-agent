# Replay-free partial-stream factored acquisition

This pressure test targets the next bottleneck after lifecycle safety:
admitting a new factual version while evidence arrives one row at a time. A
frozen base and frozen context encoder are paired with the promoted
replay-free nonlinear random-feature residual learner. Four regimes each
arrive through a `14`-row stream; the router must stage after only `7` rows and
promote using four independent held-out rows. Later rows update only the
currently staged external candidate, and no old-regime rows are replayed.

After acquisition, full and one-row partial revisits must route to the stable
logical slots. Mixed-regime partial evidence must remain ambiguous and all
read-only checks must leave the committed digest unchanged.

This promotes bounded partial-stream factual acquisition, not general
continual learning. The context encoder is frozen and the admission schedule
is fixed; open-world context formation, unrestricted growth, and arbitrary new
computation remain unqualified.

Run one seed with:

```bash
PYTHONPATH=src uv run python experiments/external_factored_partial_stream/train.py \
  --seed 81041 --report-out /tmp/external-factored-partial-stream.json
```
