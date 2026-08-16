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
  abstained on reversal and ambiguity controls. The first composition through
  the real rendered relational-navigation loop is now recorded in
  `session_records/brainworkshop_integrated_navigation_v3_2026-08-16`: it
  preserved the low-confident-error behavior, but abstained on 95--100% of
  steps and fell below the episode-local scorer. This is a development failure
  to promote, not a holdout result.
- The belief-state perception diagnostic in
  `session_records/brainworkshop_belief_state_perception_2026-08-16` keeps
  explicit context hypotheses for aliased learned events and improves useful
  prediction coverage under missing evidence. It is still symbolic-fixture
  evidence, not controller or navigation integration.
- The generalization frontier audit in
  `session_records/brainworkshop_generalization_frontier_2026-08-16` transfers
  a verified control-flow operator across changed dynamics while a raw
  source-world successor artifact fails. Cross-frontend and symbol-remapping
  transfer remain open.
- The shared-agent maze audit in
  `session_records/brainworkshop_shared_agent_maze_transfer_2026-08-16` runs
  the existing Neural Workshop agent through a rendered, hidden-goal maze on
  the same controller/event/intention boundary. The Workshop-warm arm beats
  the fresh maze curve in both development replicates, while a stale source
  maze map fails; the fresh arm never reaches the provisional stable threshold,
  so no transfer ratio or promotion is claimed.

## Deliberate limits

This is a provisional vertical slice, not a capability admission. The
experimental successor-feature, decomposition, and self-identification code is
not silently copied into the curated bank. The current self-model remains
rejected by the adversarial audit, and no navigation artifact is promoted by
this adapter.

The next production step is to improve v3 applicability without weakening its
abstention rule: handle common merge/birth histories through explicit missing
evidence and fresh relearning, then harden against true crossings, occlusion,
corrupted persistent memory, and near-equivalent causal mimics. Rerun on fresh
pixel rerenders with matched shuffled-action and fresh-learner controls before
any bank admission. In parallel, carry the belief/mask through the learned
event bus so ambiguity is preserved at the identity seam rather than collapsed
into a slot guess. Then run the explicit cross-frontend and symbol-remapping
cells before claiming broad generalization. For Maze, the next gate is a
measured Workshop planning artifact and more held-out maze replicates, with
stable warm-vs-fresh advantage and no reward-shuffled leakage. The assignment
gate, maze audit, and their artifacts are still development seams, not
promoted identity or planning models; the current diagnostics and their
artifacts are not in the curated bank.
