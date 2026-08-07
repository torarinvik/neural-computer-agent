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
2. **Bank-stored ladder with shared drivers is now the required path** —
   sequential weight-learning through the architecture-true shared drivers
   was rejected: the second game stalls at 0.19 through the occupied
   shared decoder at full budget while first-game acquisition and
   whole-plant retention work (evidence:
   `shared_driver_ladder_rejected_v1_2026-08-07`). Skill-as-context
   artifacts are the mechanism that gives the core a per-game decoder
   context switch. Also: consolidation of already-acquired skills into artifacts is unbuilt
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
4. **Arbitrated consolidation promoted; depth and mu-robustness open** —
   the demand-proportional release rule (`a = F/(F + mu*G)`, mu=3) closed
   vanilla's failed acquisition gate and passed all nine ladder gates on
   both seeds while matching retention. Evidence:
   `arbitrated_consolidation_promoted_v1_2026-08-07` (dial probes
   included). Next: deeper ladders, wider seeds, and a genuine
   parameter-conflict task pair.
6. **Peripheral skill leakage quantified: real but bounded** — fresh
   peripherals recover 0.94/0.83 mastery through the trained frozen core
   at half budget (core-resident strategy), while a random-core control
   reaches 0.53/0.67 (peripheral capacity is nonzero). Evidence:
   `peripheral_leak_diagnostic_v1_2026-08-06`. Next: the externalization
   ignorance objective is the squeezing mechanism; re-measure after
   externalized ladders land.
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
10. **Seed pool widened on the flagship only** — the EWC consolidation
    rung now holds on 5/5 seeds (69316-69320; evidence:
    `ewc_consolidation_seed_widening_v1_2026-08-06`). Other promotions
    remain two-seed. Next: widen the ladder and externalization claims
    before any external write-up.

11. **Bank tested on two contexts only, with no shared structure** — the
    self-organizing bank is promoted for contradictory context selection
    but has never been asked to *share* a fragment between related
    contexts, which is the compounding claim. Evidence:
    `self_organizing_fragment_bank_v1_2026-08-07`. *Updated by F16:*
    three factorial contexts now hold simultaneously at or above solo
    ceilings under a disjoint oracle (seed 69316; second seed running),
    but imposed fragment sharing composes nothing — the held-out
    recombination gets one rule inverted and one at chance from its
    ideal fragments. The open problem is now specifically
    **compositional practice**: a curriculum that re-pairs fragments
    across contexts during training so they detach from their birth
    context, plus R3 consolidation merges. Sharing by allocation alone
    is measured and insufficient.
13. **The anti-collapse penalty is also an anti-sharing penalty** — the
    diversity term that closed selector collapse (F12/F13) repels every
    pair of contexts equally, so it forbids the fragment reuse that
    weakness 11 exists to test. Mechanism proposed and implemented:
    swap-test conflict gating (`--conflict-gated`), which weights each
    pair's repulsion by measured cross-feed harm. Unproven until the
    factorial rung reports.
12. **Addressing is per-context logits, not content-addressed** — the
    selector maps a known context label to fragments; it does not
    retrieve from observations. The promoted routing rung
    (`game_routing_native_actions_v1`) does fetch from opaque events, but
    the two have never been composed. Next: replace selection logits with
    the candidate router over event-derived queries.

14. **Battery scale is six games; motor games are excluded** — the
    staggered battery (`staggered_battery_v1_2026-08-07`) holds six
    decision games at 0.72-1.38x solo ceilings, but forage/collect sit
    out (solo ceilings 0.02-0.08 at the fast budget) and seed 69317 is the
    softest of three (all pass the no-collapse gate; super-solo transfer
    replicates 3/3 seeds). Widening calibration shows every remaining
    family component is motor-class at the fast budget. Next: new
    decision games to widen the fast battery, and grown budgets to admit
    the motor games as the complexity ladder.

## Retired

- **Fragment selection collapse and winner-take-all context competition**
  — retired 2026-08-07 by the self-organizing bank rung: diversity
  penalty plus laggard-preferential balancing plus oracle-to-learned
  handover, both seeds at 1.000/1.000 with cross-fed 0.000
  (`self_organizing_fragment_bank_v1_2026-08-07`).

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
