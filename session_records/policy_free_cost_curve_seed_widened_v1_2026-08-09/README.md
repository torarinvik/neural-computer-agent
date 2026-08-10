# Policy-free acquisition-cost curve, seed-widened (F70)

Replicates F69 on 5 seeds. The model arm's acquisition cost FALLS across a
three-rung sequence while the policy arm's RISES, on every seed individually.

Probe: `experiments/games_amodal/probes/reacher_ladder.py`
  model  arm: --rung r4 --sparse --updates 400 --eval-batches 2 --targets=r2,r3,r4 --model-search=10
  policy arm: same, with --retrieval-first instead of --model-search

Seeds: 69316 69317 69318 69319 69320

| arm | r2 | r3 | r4 | total |
| --- | ---: | ---: | ---: | ---: |
| policy cost | 100 | 260 | 400* | 760 |
| model cost | 60 | 160 | 45 | 265 |
| policy final reach | 0.944 | 0.850 | 0.444 | |
| model final reach | 1.000 | 0.925 | 0.881 | |

*right-censored at the 400-update budget cap on 5/5 seeds; true policy cost
is >= 400 and the 2.9x total gap is a LOWER bound.

Shape per seed (not just in the mean):
  model  r2->r4 cost: 50->50, 50->25, 50->50, 75->50, 75->50  (falls 5/5)
  policy r2->r4 cost: 100->400 on all five                    (rises 5/5)

Final r4 reach separates with no overlap: model 0.812-0.938, policy
0.234-0.547, no-agent floor 0.219.

SCOPE. Seed-widening is now done; FAMILY-widening is not. r2/r3/r4 nest, so a
model of r4 contains r2. Whether this is compounding or merely nesting is
untested and is the next experiment. Prediction recorded in advance: the model
arm degrades gracefully on disjoint dynamics (incomplete, repairs by
observation) where the policy arm degrades catastrophically (wrong, must
unlearn). If the model arm instead collapses to cold-start cost, F67-F70 are
scoped to nested families only.
