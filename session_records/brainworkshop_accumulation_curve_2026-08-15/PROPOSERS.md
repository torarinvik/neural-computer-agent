# Three proposers on the same curriculum (2026-08-15)

Status: **diagnostic**. Development seed 41, already consumed. Every arm grows
a scratch copy of the library; `AgentBrain.bank` is checksummed before and
after each and was never written.

The first curve found a growing library costing 62% more than no library at
all, and localised the cause: proposals are enumerated in a fixed order and
every executable one costs an episode, so each admitted file becomes one more
thing to execute blindly. These arms change the searcher and leave everything
else -- rules, curriculum order, controller, threshold, admission -- fixed.

## Result

| Proposer | growing | control | ratio | solved | reproduced |
| --- | ---: | ---: | ---: | ---: | ---: |
| `enumerate` | 1432 | 886 | 1.616 | 7/18 | 7 |
| `dedup` | 390 | 378 | **1.032** | 7/18 | 7 |
| `feedback` | **125** | **81** | 1.543 | 7/18 | 7 |

**11.5x** fewer verifier episodes end to end, and the same seven rules solved
by all three. That equality is the point: both filters are lossless, so what
they remove is cost and never capability. The `dedup` arm's winning proposal
is byte-identical to the `enumerate` arm's on all eighteen rules.

## `dedup`: two programs that press alike are one experiment

Bottom-up synthesis has collapsed observationally equivalent candidates since
Transit and Escher in 2013. `behaviour_signature.py` is that filter here: an
observation pass records the encoded stimulus stream, every proposal is
replayed against it offline, and proposals with identical presses form one
class of which only the earliest member is executed.

The collapse is **lossless, not approximate**, because signatures are computed
on the very episode a proposal would be scored on. Under a frozen controller
with learning and sampling off, a lifetime's actions are a deterministic
function of the events and the installed program, so equal signatures imply
equal accuracy. `tests/test_behaviour_signature.py` checks that against real
scored runs rather than asserting it.

One exception is load-bearing and was found the hard way, by a first version
of this arm that changed a winner it should not have. A proposal the search
*trains* before scoring -- an un-templated `invent` or `and`, whose prototype
is learned by an acquire lifetime -- does not behave before training the way
it behaves after. Signing it early compares the wrong program. Those are now
exempt from collapsing entirely, which is what makes the winners match.

The measurement that explains the first curve:

| Library size | Proposals offered | Distinct behaviours |
| ---: | ---: | ---: |
| 3 files | 94 | 26 |
| 7 files | 190 | 29 |

The proposal list doubled; the number of distinct things it can do rose by
three. Four admitted files bought three new behaviours and 96 new proposals to
walk through. That is the 1.616.

## `feedback`: the reward already says what the target wanted

Each episode returns a reward at every eligible step -- 448 bits -- and the
enumerating searcher reduces all of it to one accuracy scalar and one
accept/reject decision. It is also self-revealing, because a press is scored
right or wrong:

    target[t] = action[t] if reward[t] else 1 - action[t]

So one episode with an arbitrary program recovers the entire target behaviour.
Nothing reads the rule; this reads the feedback the agent is already given.
`tests/test_feedback_proposer.py` checks the recovered target against the
generating rule and confirms the inversion is exact.

With signatures free, selection becomes offline: rank every candidate by
agreement with the recovered target, and spend episodes only on the ones worth
confirming. Agreement predicts the accuracy an episode would score to within
1e-9, which the tests also check.

Two disciplines keep it honest. The probe runs on one seed and the winner is
confirmed on another, so ranking never fits the episode it is scored on. And a
candidate is skipped only when `lease_discrimination` rules its probe accuracy
out as incompatible with a true rate at the gate, at alpha = 0.01 -- the same
test the leases use to refuse a near-miss, applied before spending instead of
after. It can discard a real winner only with probability alpha.

Per rule:

| | episodes |
| --- | ---: |
| solved | **5** (probe, evaluation, re-derivation, two confirmations) |
| unsolved | 8 to 9 |

against 120 for every unsolved rule under `enumerate`. Nine of the eleven
unsolved rules have **every** signable candidate ruled out by the probe alone:
their best achievable agreement is 0.605 to 0.739, and no episode makes that
clear 0.8.

## Where the library's remaining cost lives, exactly

`feedback` has the worst cost ratio of the three, and the reason is precise
rather than mysterious. Untestable-until-trained proposals, by library size:

| Library size | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proposals that must be trained before they can be judged | 4 | 5 | 6 | 7 | 8 |

Exactly **one per admitted file, plus one**. Every file admitted to the library
spawns an `and:slot` proposal whose prototype is acquired rather than given, so
no amount of offline reasoning can rule it out -- it has to be trained on the
verifier before anyone can tell whether it is any good. The control arm's
library never grows, so it pays 4 forever and spends 81 episodes; the growing
arm reaches 8 and spends 125.

Once the blind walk is gone, this is the entire marginal cost of having a
library, and it is a property of the grammar rather than of the bank. The
grammar already offers templated `and:slot[subset]` proposals built from
observed clusters, and `prototype_templates` argues that a subset mean is what
acquisition converges to anyway. If that argument holds, the un-templated
variants are redundant and dropping them would make every proposal signable
and the library's marginal cost zero. That is a change to the proposal
grammar, so it needs the lease discipline rather than a quiet edit, and it is
not made here.

## What did not change, and it is still the finding that matters

Across all three arms: **zero composes, zero inverts of a learned file, zero
ANDs over a learned file that gated.** Every winner is a `retrieve` of an exact
behavioural duplicate or a fresh `invent`. The library caches; it does not
compose.

Fixing search cut the cost of that caching by an order of magnitude. It did not
turn caching into composition, and nothing in these three arms suggests the
searcher was ever what stood in the way.

## Compression admission

The `feedback` arm admits a file only when no existing slot already presses
that way. Nothing was rejected: all four admitted files are behaviourally
distinct from every bank slot, consistent with the three new distinct
behaviours in the dedup table.

## Reproducing

```bash
PYTHONPATH=src .venv/bin/python -m experiments.brainworkshop_canonical.accumulation_curve --proposer feedback --compression-admission
```

About five minutes per arm. `--proposer` takes `enumerate`, `dedup` or
`feedback`.
