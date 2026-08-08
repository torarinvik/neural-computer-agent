# Probe scripts (F44-F55 line)

Working scripts behind the findings log, moved into the repo from session
scratchpad so they survive machine changes. These are probes, not
promoted code: they exercise `experiments/games_amodal` but are not part
of the production lint/test surface.

- `cotrained.py` — the co-trained self-addressing loop (probe -> fetch ->
  execute). Post-probe scoring per F53 (`earned()`); `--symmetric-plant`
  for the Galashov-style mixture-gradient variant (refuted on seed 69316,
  F-pending); `--ignorance` for the necessity objective.
- `chance_floors.py` — measured no-information floors for every game
  (F52). Source of `CHANCE_FLOORS` in `two_speed_battery.py`.
- `chance_baseline.py` — the twins-only first version (F51).
- `probe_earns_it.py` — the no-agent control that exposed the probe
  artifact (F53). Run something like this against EVERY new gate.
- `gradient_conflict.py` — per-context gradient cosine on the shared
  plant, with/without oracle fragments (F50).
- `fisher_stability.py` — split-half Fisher agreement vs policy entropy
  (F49); `--fisher-temperature` for the tempered estimator.
- `battery_solo.py` — solo-ceiling calibration at the fast budget.
- `BARS.md` — pre-registered pass bars for the runs in flight when it
  was written.

Run from repo root: `PYTHONPATH=.:src python experiments/games_amodal/probes/<script>`
(or `uv run python ...`, which supplies src/ itself).
