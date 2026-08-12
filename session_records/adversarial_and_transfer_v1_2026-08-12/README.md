# Adversarial attacks and the 113-domain transfer battery (F209, F210)

## F209 -- attacks on F204-F208

Predictions were written into `adversarial.py` before the run so they
could not be revised. 3 seeds, 200 adversarial worlds per kind.

    ATTACK 1  worlds parallel semantics provably cannot express
              chain:  parallel 0.8541  seq_d1 0.8544  seq_d2 1.0000
              twice:  parallel 0.8695  seq_d1 0.8545  seq_d2 0.9999
              -> CONFIRMED. F204's language claim is about the grid/rule
                 distribution, not about the languages.

              the fix, in the same probe:
              hybrid t=0.95  chain 1.0000 @ 670 candidates, 100% fallback
                             twice 0.9999 @ 363
                             rule families 1.0000 @ 200, 0% fallback
                             grid games 0.8562 @ 27517, 23% fallback

    ATTACK 2  hostile baselines (per-action mode; NN retrieval)
              games   identity .4820  mode .6769  NN .6381  READER .7920  search .8492
              rules   identity .1692  mode  n/a   NN .6067  READER .9418  search 1.0000
              -> survived. The reader is not a lookup table.

    ATTACK 3  slot permutation
              games   reader .7920 -> .6053   search .8492 -> .8492
              rules   reader .9418 -> .5662   search 1.0000 -> 1.0000
              -> CONFIRMED. The reader is NOT amodal; the language and
                 plant are. Sharpest open defect.

    ATTACK 4  starved evidence (reader / search)
              4 rows .6418/.7271   8 .7062/.7898  16 .7469/.8240  32 .7920/.8492
              -> my prediction REFUTED: the reader is worse at every
                 budget and the gap narrows with more data.

    ATTACK 5  maximally coupled programs for the plant
              random 0.9992   coupled 0.9975
              -> my prediction REFUTED. The plant is the most robust part.

## F210 -- 113 domains, transfer matrix, curricula

`battery.py` holds the domains. Distinctness checked: applied to
identical states, 109 of 113 give distinct outputs; the rest are
parameter coincidences at one seed.

**The first version had no test set** and is kept here as
`transfer-NOTESTSET-*` so the mistake stays visible. Its `all_domains`
arm trained on every evaluation domain, and I reported "+0.6181 vs
top-3's +0.2482, breadth dominates". With every fourth domain held out
of both the curricula and the ranking:

    arm                    n     HELD-OUT   eligible
    spread10              10      +0.5414    +0.3001
    top10                 10      +0.3995    +0.3233
    random50              50      +0.3798    +0.6297
    real_domains_only      4      +0.3763    +0.1696
    top3                   3      +0.3702    +0.2054
    all_eligible          85      +0.3486    +0.7057
    random10              10      +0.2096    +0.2862
    bottom3                3      +0.0460    -0.0190
    random3                3      +0.0165    +0.1981

Training on 85 domains is tied with training on 3 (t=-0.26). The
eligible column is the diagnosis: +0.7057 in distribution, +0.3486 out.

Cross-seed selection (each seed re-run with ANOTHER seed's ranking):

    top3 - random3      +0.2743 +- 0.0752  t=+3.65  3/3   SURVIVES
    top3 - shuffled     +0.3657 +- 0.1047  t=+3.49  3/3   SURVIVES
    spread10 - all85    +0.0897 +- 0.0314  t=+2.86  3/3   SURVIVES
    spread10 - random10 +0.2287 +- 0.0778  t=+2.94  3/3   SURVIVES
    spread10 - top10    +0.1369 +- 0.1411  t=+0.97  2/3   NOT ESTABLISHED

Factor structure: 0.253, 0.197, 0.135, 0.070, 0.050, 0.040. No general
factor. Factor 1 separates exactly-expressible from only-approximable;
factor 2 separates real/spatial from permutation/copy.

Rank stability across seeds: Spearman 0.677, top-10 overlap 7/10,
bottom-10 overlap 4/10 -- so the negative-transfer domains named by any
one seed are largely noise and should not be quoted.

Reproduce (single-threaded; `torch.set_num_threads(1)` is pinned):

    python -m experiments.games_amodal.probes.adversarial --seed S
    python -m experiments.games_amodal.probes.transfer_factors --seed S \
        --pool 600 --combo-pool 900 --reader-updates 3000 --eval-worlds 24
    # cross-seed selection:
    ... --ranking-from <another seed's json>
    # different quarter held out:
    ... --split-offset 0|1|2
