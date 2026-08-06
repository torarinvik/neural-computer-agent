# Known weaknesses ledger

One line of truth for what is currently weak, why it matters, and the next
concrete action. Every entry cites its evidence. Update this file whenever a
rung promotes, rejects, or qualifies. Ordered by severity.

## Open

1. **Skill-externalization identity check is not seed-robust** — v5
   promoted fully on seed 69316 (all seven gates: necessity, content, and
   identity causality with fixed computing parameters); seed 69317 leaks
   on one gate (Pong artifact partially plays Snake, 0.4141). Evidence:
   `skill_externalization_qualified_v1_2026-08-06`. Next: per-game null
   gates scaled to game guessability and richer decoy sets — no further
   global-dial escalation (stopping rule).
2. **Consolidation of already-acquired skills into artifacts is unbuilt**
   — externalized skills were acquired externalized from the start; the
   promoted EWC rung's weight-stored skills cannot yet be migrated into
   the bank. Also: skills still live in weights everywhere except the
   externalization harness — the promoted EWC rung stores game skill in core weights
   (recorded transitional violation in
   `docs/DYNAMIC_BRAIN_ARCHITECTURE.md`). Next: once externalization
   promotes, retire the violation by re-running the continual-learning
   ladder with bank-stored skills.
3. **Positive transfer measured but not yet promoted** — the
   EWC-consolidated plastic core beats a fresh core on Pong acquisition on
   both seeds (eval 0.8125/0.9688 vs 0.7051/0.6641; evidence:
   `plastic_core_positive_transfer_v1_2026-08-06`), reversing the
   frozen-core rung's seed-negative result. Cross-run comparison only.
   Next: same-run randomized transfer harness with nulls and more seeds,
   then transfer through fetched bank artifacts (Phase 3).
4. **EWC untested beyond one consolidation step** — one lambda, two games.
   Successive Fisher maps must compose; anchor staleness and diagonal
   blindness untested. Next: three-game ladder (Breakout now available),
   where an enhanced consolidation rule would earn novelty if vanilla
   cracks.
6. **Peripheral skill leakage unquantified** — per-game encoders/decoders
   train jointly with play, so strategy may hide there
   (architecture-doc violation 2). Next: peripheral-swap diagnostic —
   fresh peripherals + frozen core/artifacts; recovery cost measures the
   leak.
7. **Three-game routing qualified, not promoted** — the mechanism extends
   structurally to three slots (permutation 1.0, nulls clean, Snake routes
   1.0), but the Breakout slot plateaus at ~0.54 mastery regardless of
   budget and the Pong/Breakout siblings confuse the router (0.69-0.90).
   Evidence: `three_game_routing_qualified_v1_2026-08-06`. Next: stronger
   slot policies for compound games; key separation trained on sibling
   contrast.
8. **No memory bank in the games runtime** — `memory=None` everywhere;
   games needing recall are unplayable. Next: wire `ContentAddressedMemory`
   into a hidden-state game once externalization stabilizes.
9. **Deliberation loop unused** — all game rungs commit in one forward
   pass; WAIT/THINK/COMMIT exists but is untested on games, a latent
   compute ceiling (see the no-theoretical-ceiling doc section). Next:
   think-budget control on the externalization harness, where artifact
   interpretation plausibly needs iterated thought.
10. **Two-seed replication only** — all promotions rest on seeds
    69316/69317. Next: widen the seed pool on the flagship claims before
    any external write-up.

## Retired

- **One-step truncated BPTT as suspected acquisition bottleneck** —
  retired 2026-08-06 by measurement: detach interval 8 matches interval 1
  on endpoint mastery and curves at matched budgets
  (`bptt_window_diagnostic_v1_2026-08-06`). The shared-controller
  acquisition gap is attributed to the recurrent optimization landscape,
  not truncation. `detach_interval` remains available in the trainers.

- **Catastrophic forgetting in the plastic core** — retired 2026-08-06 by
  `ewc_consolidation_plastic_core_v1` (Fisher consolidation; permuted-null
  causal).
- **Stale-reference protection** — rejected and archived
  (`protected_plasticity_stale_reference_rejected_v1`); superseded by
  Fisher consolidation.
- **Artifact presence-cue and master-key shortcuts** — rejected and
  archived (`skill_externalization_onswitch_rejected_v1`,
  `skill_externalization_master_key_rejected_v2`); closed by decoy and
  cross-artifact ignorance training.
- **No third game for ladder/transfer tests** — retired 2026-08-06:
  `BreakoutVerifier` added with tests; shared-controller agent is now
  game-parametric.
