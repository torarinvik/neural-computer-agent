# Contextual factual memory with goal-conditioned search

This three-seed composition test combines the strongest current continual-
learning primitives: context-local replay-free reliability state, retention-
verified factual growth, stable-ID eviction, exact persistence, and
inference-time goal-conditioned search over a frozen transition model.

Across all seeds, near-boundary corrupted evidence was vetoed without changing
the factual bank; a verifier reversal released the quarantined evidence;
capacity growth retained prior slots; a third regime was promoted; search
reached a held-out goal without an action-policy target; middle eviction removed
the corresponding contextual reliability state; and the restored system routed
the surviving slot correctly. The controller, shared base, and context encoder
were frozen, and no replay rows were used for reliability calibration.

This promotes a bounded composition of factual external memory and
goal-conditioned behavior synthesis. It does not establish unrestricted memory
growth, arbitrary new computation, learned raw-modality context formation, or
general continual learning.
