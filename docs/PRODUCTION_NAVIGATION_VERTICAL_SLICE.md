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
- The versioned `ExternalCausalIdentityArtifact` can derive that evidence from
  bound learned event histories and opaque action/intention features without
  exposing raw modality formats to the controller. The assignment gate now
  requires both a top-two margin and a minimum absolute evidence score, so a
  weak but high-margin signal still abstains. Its rendered integration remains
  a diagnostic, not a promoted identity learner.
- `PersistentCausalIdentityV2` persists the action-conditioned event dynamics,
  rebinds them to current tracks, and freezes them on contradiction or missing
  evidence until fresh episodes relearn a replacement. Its first closed-loop
  diagnostic quarantined safely but did not beat the episode-local scorer, so
  the reserved integrated holdout remains untouched.
- `PersistentCausalIdentityV3` is the next development schema. It replaces the
  prefix-sensitive global covariance with an action-labelled transition graph
  over learned event states. On a bounded rendered fixture it improved return
  over the episode-local scorer with zero confident identity errors and
  abstained on reversal and ambiguity controls. This remains a development
  signal, not a promotion or holdout result.

## Deliberate limits

This is a provisional vertical slice, not a capability admission. The
experimental successor-feature, decomposition, and self-identification code is
not silently copied into the curated bank. The current self-model remains
rejected by the adversarial audit, and no navigation artifact is promoted by
this adapter.

The next production step is to harden the v3 graph against true crossings,
occlusion, corrupted persistent memory, and near-equivalent causal mimics,
then rerun it on fresh pixel rerenders with matched shuffled-action and
fresh-learner controls before any bank admission. The assignment gate and
artifacts are still an interface seam, not a promoted identity model; the
current diagnostics use bounded two-track fixtures and their artifacts are not
in the curated bank.
