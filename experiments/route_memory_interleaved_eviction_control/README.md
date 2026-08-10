# Interleaved identities with delayed evidence and bounded eviction

This experiment keeps three latent identities active at once. Their first
observations, two high-similarity distractors, and delayed partial observations
arrive in an interleaved schedule. The memory budget is fixed; once full, new
evidence must replace an unprotected distractor. After all delayed evidence is
present, the planner must consolidate each true pair while retaining every
mastered identity from earlier rounds.

Capacity and transaction phase provide a structural action mask, so the
learned policy receives credit only for ranking legal candidates. In a fixed
capacity phase it cannot spend updates selecting `grow`; during consolidation
it cannot select eviction or admission. This is infrastructure, not a learned
task rule.

Promotion requires multi-identity completion, full retention at every round
boundary, learned eviction, delayed consolidation, coordinate reversal,
unseen-pattern transfer, zero false commits, atomic rejection, a frozen
controller, and zero replay. The claim remains bounded: this pressure-tests
continual external-memory routing, not arbitrary semantic identity or general
continual learning. General continual-learning claims must use the
policy-free transition-model/search path documented in
`docs/POLICY_FREE_CONTINUAL_LEARNING.md`.
