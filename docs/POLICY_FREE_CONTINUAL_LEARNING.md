# Policy-free continual learning

## Core distinction

The controller and durable bank should store facts, operators, goals, and
programs—not a task-conditioned action policy.

```text
observation -> factual model/program state -> search or execution -> intention
```

A stored policy commits to behavior that may become stale after a reversal or
context change. A factual model can remain valid while a new goal or search
produces different behavior. Likewise, a stored composite commits to one
arrangement; smaller verified pieces can be recomposed.

## What may learn

- Encoders may learn format-to-event translation and generic perceptual
  reductions.
- External models may learn observed transitions.
- External programs may learn or search for reusable state transformations.
- Goal/value artifacts may learn what outcomes to seek from scalar evidence.
- Memory-side routers and lifecycle policies may learn which opaque artifact
  to read, protect, compact, or replace.
- Decoders may learn intention-to-protocol translation.

The last category of memory policy is operational bookkeeping, not behavioral
policy. It must not receive semantic task identity, correct actions, or hidden
environment state.

## What may not become canonical

- a learned controller action table indexed by task or context;
- a decoder that embeds strategy rather than protocol lowering;
- an encoder that inserts task semantics or correct-action hints;
- a memory eviction learner whose weights are retained despite hurting a
  held-out family;
- replay-dependent claims presented as replay-free continual learning;
- a compatibility fallback that treats unknown memory as a valid prediction.

## Learning transaction

1. Observe an attempted action and trusted scalar outcome.
2. Form a candidate factual row, operator, program, goal/value binding, route,
   or maintenance update outside protected state.
3. Evaluate the candidate on fresh held-out experience.
4. Check causal controls, retained primitives, and matched fresh acquisition.
5. Admit transactionally only after a stable-prefix gate passes.
6. Keep the blueprint but reset candidate weights when inherited weights hurt.

No task is mastered from an isolated threshold crossing. The first threshold
must remain satisfied at every later measured prefix.

## Reward boundary

Reward arrives as an authenticated outcome event through the generic input
bus. It remains ordinary learned input at the controller interface but special
in runtime authority and causal accounting. Programs cannot forge verifier
events, and polling an empty outcome queue cannot manufacture a zero.

## Current evidence

The strongest retained positive result is bounded external n-back-16 to
n-back-32 growth with frozen controller/frontend, frozen source compute file,
and zero replay. The strongest relevant negative result is failed transfer of
an inherited external eviction policy to a held-out compute family. Therefore
the external artifact architecture is retained, while general maintenance
weights are not yet reusable capability.
