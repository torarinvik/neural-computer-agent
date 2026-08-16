# Learned composition proposer (2026-08-16)

Status: **development proposal-throughput diagnostic; not promoted**.
`AgentBrain.bank` remained unchanged at
`07319eb13c9cac58bbbe94258548e64f13ba1b3512ca2a01fbdc675c49e2e7c9`.

## What changed

The compositional search previously evaluated every stored slot pair under
every boolean combiner before confirmation. `LearnedCompositionProposer` adds
an external, evidence-trained shortlist:

- it sees only learned prediction vectors and scalar verifier outcomes;
- it keeps Beta-smoothed success priors for opaque combiners, updated only
  after verifier-backed confirmation;
- it shortlists the most compatible slots and ranks pair proposals;
- the existing confirmation/admission gate is unchanged;
- when the shortlist cannot explain the evidence, exhaustive search is restored
  and the extra work is recorded.

No controller branch, raw modality, coordinate, rule label, or verifier state
was added. The proposer is an iteration-speed mechanism, not a new capability
claim.

## Result

On a 32-record library and a held-out pair composition:

| search | hypotheses | wall time | result |
| --- | ---: | ---: | --- |
| exhaustive | 1,520 | 65.8 ms | correct `slot 0 and slot 1` |
| learned shortlist | 116 | 36.0 ms | same behavior, no fallback |

The shortlist reduced candidate work by **92.4%** and this small CPU audit's
search wall time by about **45%**. A stranger target triggered exhaustive
fallback and produced no candidate, preserving the no-parts control rather
than claiming a shortcut solved it.

This is not yet evidence of lower verifier experience: proposal scoring is
external arithmetic and the audit does not run a fresh learner, noisy feedback
campaign, or held-out live navigation. The next check is to wire the proposer
through the full compositional transfer stream, compare acquisition curves
against exhaustive search, and keep the fallback/false-recognition controls.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python \
  -m experiments.brainworkshop_canonical.learned_proposer
```

The machine-readable result is in `learned_proposer.json`.
