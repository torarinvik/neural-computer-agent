# Candidate templates close the proposer gap (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. Nothing
admitted, `AgentBrain.bank` byte identical.

The expressiveness diagnostic found seven sampled rules expressible by
programs the machine already supports, of which search found two. The cause
was not subtle: the searcher tried exactly one prototype — whatever the
acquire rule's reward-weighted average converged to — while the enumeration
tried every template the frontend can form.

`prototype_templates.py` supplies candidates instead. One observation pass
encodes the rendered stream through the learner's own frozen frontend, events
are grouped by distance alone (no alphabet size, no symbol label, no verifier
state), and every subset mean up to size four becomes a candidate. A subset
mean is exactly what acquisition would converge to if it were rewarded on that
subset, so this widens the same hypothesis class rather than adding a new one.
Each template is offered in both polarities, as stated and inverted.

Proposals are **appended**, never inserted. Any rule an earlier proposal
already solved keeps the same winner, so every recorded campaign is unchanged;
the lease and search tests pass untouched.

## Results

| | Before | After | Family ceiling |
| --- | ---: | ---: | ---: |
| Sampled rules solved | 2 / 18 | **7 / 18** | 7 / 18 |
| Mean accuracy | 0.680 | **0.782** | 0.786 |
| Shortfall below own family | +0.106 | **+0.004** | — |

| Rule complexity | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Solved (of 3) | 3 | 2 | 1 | 1 | 0 | 0 |

**The solved set is exactly the expressible set** — every rule the enumeration
said was reachable is now found, and no rule the enumeration said was
unreachable is falsely claimed. The searcher now saturates its own program
family.

Both polarities were needed. Templates alone reached 4/18; the three remaining
rules all wanted `invert prototype`, a template answering the rule flipped,
which the proposer had never offered.

The four hand-written rules are still solved, by the same winners as before
(`and:0`, `and:0`, `invert:0`, `retrieve:0`), which is the check that the
addition changed nothing that already worked.

## Cost

48 programs executed per rule out of 80 offered, plus one observation pass. The
searcher stops at the first proposal that gates, so the cost is paid only until
something works. This is enumeration, not inference: it scales with the
template count, and it is affordable here only because the alphabet is small.
A family with accumulated state will not be enumerable this way, which is what
makes a real proposer necessary later rather than now.

## What it does not fix

Eleven rules remain unsolved, and they are unsolved for a different reason: no
program in this family expresses them at all. Their ceiling is the family's
ceiling, and the family scores `-0.003` against a memoryless policy. Templates
cannot help there. That is the next piece of work.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.rule_baseline --with-templates
```
