# Variable deliberation (outcome-only)

This is the first benchmark for the production controller's execution plane.
The same recurrent controller emits an opaque intention and a learned
`WAIT`/`THINK`/`COMMIT` decision. `OBSERVE` is transport: the runtime records
available event tokens and exposes them to the controller.

The verifier renders a high-bit event immediately. Complete, delayed,
think-required, and permanently missing episodes are exposed through balanced
trainer-side curricula;
the learner receives only event tokens and the scalar success outcome after
sampling an opaque four-way action. It is charged a bounded utility cost of
`0.20` for waiting and `0.35` for thinking.

The benchmark is intentionally a sub-minute rung. It is not a claim of a
general learned compute allocator. Promotion requires repeated seeds where
adaptive utility beats immediate commit and fixed-compute controls, while the
missing-evidence and random-action controls remain at their expected levels.

The promoted mixed rung uses an observable transport warmup for the opaque
action path, freezes that path, and trains the execution head from fresh scalar
outcomes. The controller's transport head receives generic event density,
aggregate confidence, and their interaction. Across seeds 17, 18, and 19,
held-out complete, delayed, and think-required audits select the correct
execution state at 100%; the mixed audit reaches 1.0 reward and utility of at
least 0.858 (0.8625 optimal expectation). This promotes a narrow execution
capability; it does not promote general learned compute allocation.

The bounded missing-evidence extension is also promoted, but narrowly. After
the original execution heads are frozen, an age-gated timeout residual is
trained from outcome-only complete/delayed/missing/think-required episodes.
Through the production timestamp buffer, paired mixed audits across seeds 17,
18, and 19 improve utility over immediate commit by `0.0561`, `0.0678`, and
`0.0551`. The timeout policy commits after the bounded quiet tick on all three
seeds. This qualifies termination of a permanently missing partner in this
verifier; broad learned absence handling across modalities remains open.

Run a short rung with:

```bash
PYTHONPATH=src:. .venv/bin/python -m experiments.deliberation_amodal.train --balanced-curriculum --steps 4096 --warmup-steps 8192 --seed 17
```

The output records reward, utility, mean internal think ticks, and the required
sample-efficiency accounting fields.
