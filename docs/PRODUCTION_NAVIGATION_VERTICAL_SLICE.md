# Production navigation vertical slice

The first live production boundary for navigation is now explicit in
`src/neural_computer/navigation_runtime.py`.

```text
learned event collection
        -> frozen amodal controller state
        -> external causal identity assignment (or explicit abstention)
        -> external factual transition search
        -> opaque learned intention
        -> replaceable protocol decoder
        -> attributed scalar feedback
```

`PolicyFreeAmodalLiveMachine` is a transport adapter around the existing
`PolicyFreeAmodalRuntime`. It does not reimplement decomposition, identity, or
navigation policy. The controller receives only an `AmodalEventCollection`;
the goal state and candidate intentions are opaque external tensors; the
planner searches an external transition model; and protocol actions return to
the controller only through a caller-supplied feedback encoder.

## What this slice proves

- The live `RECEIVE -> plan -> decode -> EMIT` boundary can run through the
  policy-free runtime.
- Delayed outcomes are attributed to the exact action receipt before becoming
  controller feedback.
- A protocol action cannot be passed directly as controller feedback when its
  width is incompatible; the replaceable feedback encoder is mandatory.
- The production adapter has no access to verifier coordinates, task/rule IDs,
  cluster maps, or scoring oracles. A source-level guard and focused tests
  enforce that boundary.
- A caller-owned `ExternalCausalIdentityAssignment` may select among opaque
  goal-state candidates from learned evidence. A low-margin assignment emits no
  action rather than guessing, and the selected slot is retained only in
  receipt-local credit metadata.

## Deliberate limits

This is a provisional vertical slice, not a capability admission. The
experimental successor-feature, decomposition, and self-identification code is
not silently copied into the curated bank. The current self-model remains
rejected by the adversarial audit, and no navigation artifact is promoted by
this adapter.

The next production step is to supply a temporary external transition/self
artifact through this boundary, run valid pixel-level controls, and compare its
stable learning curve with a matched fresh learner before any bank admission.
The assignment gate is an interface seam, not a promoted identity model; the
collision diagnostic remains synthetic and its artifact is not in the curated
bank.
