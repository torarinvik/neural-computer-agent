# Policy-free transition-model compounding — 2026-08-10

This record imports the strongest architectural result from the exported
games-session into the canonical repository: factual transition models are
adapted externally, while an inference-only planner derives intentions from
the current opaque goal. No controller parameters are updated during target
adaptation and prior model slots remain byte-stable.

Two single-seed smoke/promotional runs were rerun after the architecture note
was added:

- nested compounding, seed `70311`: promoted; warm target updates were
  `24, 24, 17` versus fresh `38, 38, 34`, with all prior regimes retained;
- disjoint compounding, seed `70411`: promoted; warm cumulative target cost
  was `155` versus fresh `158`, with retention, no-agent floor, and planner
  inference-only gates passing.

These are narrow audit rungs, not a general continual-learning claim. The
disjoint run also selected fresh challengers for both targets, which is an
important control: inherited weights are not retained merely because they are
old. See `docs/POLICY_FREE_CONTINUAL_LEARNING.md` for the architectural rule.
