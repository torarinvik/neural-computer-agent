# Four-step replay-free episodic credit

This is the next pressure test after the promoted two-step isolated-credit
rung. The frozen recurrent context encoder and old route handle families 0/1;
four new capabilities (families 2/3/4/5) are then acquired sequentially.
Each new capability receives an isolated external route extension and credit
head. Earlier extensions remain frozen while later candidates first fail
through every earlier route.

The learner-visible boundary remains ordered learned event tokens, opaque
actions, scalar outcomes, and presence. No task IDs, correct rows, operation
names, or replayed old examples enter the deployed components. The extension
trainer uses fresh paired counterfactual outcomes only.

Across seeds 69316 and 69317, old and new route selection, candidate
permutation accuracy, old-route retention, and isolated credit accuracy were
all `1.000`. Every later capability attempted all prior extensions at rate
`1.000`; disabling the required extension reduced selection to `0.000`; and
reward-shuffled extensions were selected at `0.000`. Both runs used `122,880`
unique verifier bits, `30,976` logical lifetimes, `2,048` optimizer updates,
and zero replay. The four-step chain is promoted as bounded sequential
external growth with isolated credit state.

This does not establish unbounded memory growth, learned consolidation,
arbitrary program induction, or general continual learning. Evidence is in
`report_seed69316.json` and `report_seed69317.json`.
