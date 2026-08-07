# Promoted: three factorial contexts held at once; fragments are per-rule programs (2026-08-07)

The sharing rung's first promotable result. Three contexts built from two
independent binary rules (`dualAC`, `dualAD`, `dualBC` — every pair
shares exactly one rule except AD|BC, which shares none) are acquired
SIMULTANEOUSLY by one fixed plant with a disjoint-oracle fragment bank,
each at or above its measured solo ceiling — the bank's first positive
transfer at acquisition time (solo conditional ceiling ~0.78; in-bank
0.75-0.83 while also holding two other contexts).

Command (per seed 69316, 69317):

```bash
uv run python -m experiments.games_amodal.fragment_bank \
  --seed <seed> --suite dual --oracle-selection --oracle-map disjoint \
  --warm-updates 0 --updates 2400 --batch-size 32 --steps 48 \
  --fragments 6 --balance-contexts \
  --ignorance-weight 0.5 --ignorance-every 3 --adapt-updates 200
```

Mastery = per-rule accuracy over resolved trials (knowledge-scored; see
F15). Engagement full on every context (20-24 of ~24 trials).

## Result (train / cross-fed per-rule accuracy)

| readout | seed 69316 | seed 69317 |
| --- | --- | --- |
| dualAC / dualAD / dualBC | 0.999 / 0.832 / 0.808 | 0.978 / 0.763 / 0.745 |
| `dualAD<-dualAC` (share takeA) | 1.000 / **0.000** | 0.955 / **0.000** |
| `dualBC<-dualAC` (share takeC) | **0.002** / 1.000 | **0.044** / 1.000 |
| `dualBC<-dualAD` (share none) | 0.000 / 0.485 | 0.008 / 0.467 |

The specification signature, resolved per rule: swap a fragment set and
exactly the rules it contradicts invert to ~0.0 (systematically wrong,
far below chance) while the rules it shares survive at ~1.0. A fragment
set IS a two-rule program; the plant executes whichever it is handed.

## Causal controls

Withheld-bank sits at 0.46-0.62 (residual default; the conditional rules
collapse), decoy-noise similar. The factorial-allocation comparison
(seed 69316, matched budget) also holds all three contexts
(0.992/0.757/0.747) but is uniformly slightly worse, and its ideal
composed fragments do NOT solve the held-out `dualBD` (0.337/0.506 —
one rule inverted, one at chance; no better than adaptation over a
random bank).

## Claim boundary

Promoted: simultaneous multi-context storage with per-rule fragment
specification, at solo-ceiling quality, zero replay, no weight patching.
Not promoted: composition (the F16 negative result — imposed sharing
does not make fragments portable to novel pairings; compositional
practice or R3 consolidation is the open mechanism), learned selection
on this suite (oracle addressing here; the F13 handover recipe is the
known path), and the conditional-context solo ceiling (~0.78, a plant
acquisition limit, tracked separately).
