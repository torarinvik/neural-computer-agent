# External model-based planner initial rung

This archive records the first canonical-repository pressure test inspired by
the exported `Game framework with continual learning` session. It separates
factual transition learning from behavior selection:

`opaque state/intention events -> external transition model -> terminal-goal search -> opaque intentions`

The `AmodalCognitiveController` was instantiated and frozen. The replaceable
`ExternalTransitionModel` received `1,200` self-supervised source updates and
the three target goals received zero optimizer updates and zero replayed
examples. The planner searched over opaque intention tensors and the verifier
decoded them only outside the production boundary.

Both seeds passed every pre-registered gate:

- target mastery: `1.0` on all three target goals;
- retention prefix: `1.0` after the target sequence;
- controller unchanged, exact model persistence, and zero target updates;
- shuffled-goal and shuffled-transition controls: `0.0` mastery;
- fresh-model controls: `0.3333` and `0.0` mastery.

The initial smoke used an invalid intermediate objective: Euclidean distance
between opaque latent states. It failed despite near-perfect transition loss.
The planner was corrected to terminal opaque-goal matching, which is the
appropriate generic contract for this bounded deterministic rung. The failed
smoke remains disposable and is not part of the promotion claim.

This promotes only a narrow interface result: learned external transition
facts can be reused to derive behavior for a sequence of new opaque goals
without modifying the controller or storing a target-specific policy. It does
not establish general continual learning, disjoint-dynamics transfer,
unrestricted model growth, learned goal abstraction, or arbitrary program
induction. The next rung must use genuinely different dynamics and compare
against a matched fresh model with the same search budget.
