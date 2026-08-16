# Campaign sizing: nothing runs longer than five minutes

Waiting half an hour for one number is the failure this rules out. Every
campaign returns inside a hard cap, scoring whatever landed.

## The constraint, measured

Cost is close to linear in total training steps. On the current box, summed
across workers:

    90 seconds per 1,000 training steps

So the cap is a step budget, and at the cap:

    total training steps <= 300 * workers / 0.090
    six workers  ->  ~20,000 steps
    ten workers  ->  ~33,000 steps

`scripts/plan_campaign.py` turns a design into a projected wall clock and, when
it does not fit, names the cheapest way to make it fit. Run it before launching,
not after waiting.

## What fits

A three-budget, eight-seed, two-arm comparison fits comfortably. A seven-budget,
twelve-seed, six-arm sweep is 2,400 seconds -- eight campaigns' worth -- which is
exactly what the earlier hour-long runs were.

Cut the **grid** before cutting seeds. The top budgets cost the most and say the
least, because that is where both arms have converged; the informative region is
mid-curve, and in the read-path result the whole effect sat at 160 and 256
updates while 768 showed +0.0020.

## Enforcement, not discipline

- `runq.sh` wraps every job in `timeout $MAX_JOB_SECONDS` (default 300), so one
  slow run can never outlast the cap.
- `campaign.sh` stops watching at `CAMPAIGN_MAX_SECONDS` (default 300), pulls,
  and scores what exists. Results stream home every 30 seconds and every scorer
  globs the reports present, so a partial score is a real partial answer.
- Anything slower than the cap is a design error to fix at the grid.

## The tradeoff, stated

Smaller campaigns carry less power each. The answer is not one big sweep but
several capped ones with the same design, pooled: twelve seeds as two campaigns
of six is the same evidence with signal arriving five minutes in rather than
forty. Where a cell is ambiguous, add seeds in another capped campaign instead of
widening the first.

Unmeasured: whether more workers help on a single-GPU box. The GPU reads 99%
busy during a sweep, so worker scaling past six may be flat. Measure it before
assuming the ten-worker column above.

## External proposal routing

The compositional search now has a development-only
`LearnedCompositionProposer`. It shortlists opaque slot/combiner candidates
from learned prediction agreement and retains exhaustive fallback when the
shortlist cannot explain the evidence. The first audit reduced 1,520 candidate
hypotheses to 116 on a held-out pair while preserving the winner; this is CPU
iteration throughput, not a reduction in verifier experience or a promotion.
