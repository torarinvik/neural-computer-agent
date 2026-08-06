# Known weaknesses ledger

One line of truth for what is currently weak, why it matters, and the next
concrete action. Every entry cites its evidence. Update this file whenever a
rung promotes, rejects, or qualifies. Ordered by severity.

## Open

1. **Skill externalization incomplete on Pong** — the double dissociation is
   full for Snake, but Pong leaks ~0.2-0.4 mastery through one context null
   per seed (which null varies by seed — boundary noise, not a shortcut).
   Evidence: `skill_externalization_*` archives; v5 (ignorance weight 2.0)
   in flight. Next: if v5 fails, qualify the rung and redesign with
   per-game null gates scaled to game guessability or richer decoy sets.
   No further global-dial escalation (stopping rule set 2026-08-06).
2. **Skills still live in weights everywhere except the externalization
   harness** — the promoted EWC rung stores game skill in core weights
   (recorded transitional violation in
   `docs/DYNAMIC_BRAIN_ARCHITECTURE.md`). Next: once externalization
   promotes, retire the violation by re-running the continual-learning
   ladder with bank-stored skills.
3. **Positive core transfer is unreliable** — frozen-core transfer was
   seed-sensitive and can be strongly negative (qualified
   `shared_controller_two_game_qualified_v1`). Next: measure transfer
   through the EWC-consolidated plastic core, then through fetched bank
   artifacts (Phase 3 compositional-transfer design).
4. **EWC untested beyond one consolidation step** — one lambda, two games.
   Successive Fisher maps must compose; anchor staleness and diagonal
   blindness untested. Next: three-game ladder (Breakout now available),
   where an enhanced consolidation rule would earn novelty if vanilla
   cracks.
5. **One-step truncated gradients through the recurrent core** —
   `state.detached()` per step in all game trainers limits credit over
   time; suspected cause of the shared-controller acquisition gap (0.86 vs
   0.94 standalone). Next: k-step truncation control at matched budget.
6. **Peripheral skill leakage unquantified** — per-game encoders/decoders
   train jointly with play, so strategy may hide there
   (architecture-doc violation 2). Next: peripheral-swap diagnostic —
   fresh peripherals + frozen core/artifacts; recovery cost measures the
   leak.
7. **Routing is first-frame-only over two candidates** — promoted
   `game_routing_native_actions_v1` queries the first observation only.
   Next: mid-lifetime route queries and a three-candidate bank with
   Breakout.
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
