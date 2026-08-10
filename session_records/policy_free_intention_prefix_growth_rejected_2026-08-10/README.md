# Rejected: six-regime policy-free intention prefix growth

Date: 2026-08-10
Seeds: `85401`, `85402`
Schema: `neural-computer.policy-free-intention-prefix-growth.v1`

## Question

Can one frozen amodal controller acquire six sequential opaque regimes in an
external routed intention bank while retaining the complete earlier prefix,
without replay?

The run used the richer trajectory-statistics route query from the exported
games session: the opaque controller state plus masked mean/max learned event
tokens. Each successor was a fresh unqualified cell appended to the
accumulated bank and was compared with a matched fresh one-cell learner.

## Verdict

Rejected. Neither seed passed the stable-prefix promotion gate.

The causal controls that did pass were zero replay, frozen controller,
sparse candidate materialization, missing-evidence no-op, delayed physical
credit, and reward-shuffled failure. The failed gates were the important ones:

- all six regimes did not master reliably;
- automatic protection did not activate from noisy acquisition outcomes;
- old content and route probabilities did not remain above the prefix floor;
- the accumulated bank did not provide positive transfer on every successor;
- seed `85402` also failed the corruption probe because the unqualified route
  could still settle on a weak cell.

## Architectural finding

This is a useful rejection, not a lack of infrastructure. It isolates two
remaining bottlenecks:

1. A sampled intention generator is still policy-like. The earlier warm-copy
   formulation copied its weights into a new cell and produced negative
   transfer on contradictory regimes. The exported session's plant/bank
   result therefore applies directly: retain reusable factual computation in
   the frozen plant and represent a new regime as a verified residual/delta
   or a fresh challenger, not as a blind policy copy. The follow-up fresh-cell
   formulation below still failed the stable-prefix gates, so fresh growth is
   necessary but not sufficient.
2. Exploration floors make an appended cell observable, but they do not by
   themselves solve candidate qualification. Retention must be based on a
   held-out verifier prefix rather than noisy exploratory rewards, and the
   winning challenger must be committed copy-on-write only after the old
   prefix is checked.

The trajectory-query seam and unqualified-cell exploration floor remain in
the production API as independently testable mechanisms. This archive does
not promote six-regime growth or general continual learning.

The follow-up implementation now exposes
`ExternalOutcomeIntentionRouter.verify_and_protect` so a future rerun can
qualify cells from a fresh held-out verifier prefix without changing learned
content or route statistics. That gate is not retroactively claimed by this
rejected run.

## Accounting

The JSON reports record unique verifier bits, logical lifetimes, optimizer
updates, replay, search expansions, and wall time. No old examples were
replayed and the controller received zero optimizer updates.

Reports: `seed-85401.json`, `seed-85402.json`.
