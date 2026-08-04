# Mixed WAIT/THINK/COMMIT arbitration

This rung tests one recurrent controller on complete, delayed, and
think-required episodes. The controller receives only learned event tokens and
the scalar verifier outcome. A balanced trainer-side curriculum exposes the
three observable transport states without emitting a curriculum label.

The execution head consumes three generic transport features: event density,
aggregate event confidence, and their interaction. The opaque action path is
stabilized with an observable transport warmup, then frozen while the
execution head is trained from fresh scalar outcomes. `WAIT` costs `0.20`
utility and `THINK` costs `0.35`; `COMMIT` has no execution cost.

Seeds 17, 18, and 19 all pass the causal state gates on held-out episodes:

- complete: `COMMIT` at 100%, reward `1.0`;
- delayed: `WAIT` at 100%, reward `1.0`;
- think-required: `THINK` at 100%, reward `1.0`.

The held-out mixed audit reaches reward `1.0` for every seed, with utilities
`0.8629`, `0.8658`, and `0.8580`. The finite-sample optimum is `0.8625` for
the `.5/.25/.25` mixture under the recorded costs. This promotes a narrow
execution-plane capability, not general multimodal reasoning or language
grounding.
