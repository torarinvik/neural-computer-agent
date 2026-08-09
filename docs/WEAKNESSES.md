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

11. *Largely CLOSED by the goal-factored rung — see the update at the end
    of this entry.*
    **Bank tested on two contexts only, with no shared structure** — the
    self-organizing bank is promoted for contradictory context selection
    but has never been asked to *share* a fragment between related
    contexts, which is the compounding claim. Evidence:
    `self_organizing_fragment_bank_v1_2026-08-07`. *Updated by F16:*
    three factorial contexts now hold simultaneously at or above solo
    ceilings under a disjoint oracle (seed 69316; second seed running),
    but imposed fragment sharing composes nothing — the held-out
    recombination gets one rule inverted and one at chance from its
    ideal fragments. The open problem is now specifically
    **compositional practice**. *Updated by F27:* partner rotation is
    now built and measured — it makes fragments interchangeable (combo
    spread 0.042 on one seed) but NOT composable (held-out pairing
    0.27-0.57 vs 0.47 random-bank control, i.e. at chance). Rotating
    fragments is not practicing composition, because the agent never has
    to succeed on an unseen pairing. Next: rotate held-out PAIRINGS
    inside training (train/holdout rotation over rule pairs), which is
    the actual MLC protocol. *Updated 2026-08-09 by the
    composition rung:* the MLC protocol was not needed. Under the
    goal factorisation a fragment is two SLOTS (which side to want per
    cue), so a held-out pairing is ASSEMBLED from slots trained in other
    games with zero gradient steps. Measured on the two non-degenerate
    held-out pairings, two seeds: assembled 0.569-0.706 against trained
    0.698-0.702 and a measured floor of 0.333 — 85% and 111% of trained
    performance on combinations never practised — while the scrambled
    control (same donors, wrong cues) sits at or below floor (0.207-0.271).
    Evidence: `goal_composition_v1_2026-08-09`. What made it work is the
    thing the four failed mechanisms lacked: fragments that name a goal
    in a vocabulary the plant already executes, rather than opaque
    whole-programs. Remaining: c11's swap control is degenerate (both
    cues want the same side) so compose_suite needs a holdout set
    designed for swap controls; and the claim is conditional on a
    competent executor, which at arity 3 is unreliable (weakness 19).
13. **The anti-collapse penalty is also an anti-sharing penalty** — the
    diversity term that closed selector collapse (F12/F13) repels every
    pair of contexts equally, so it forbids the fragment reuse that
    weakness 11 exists to test. Mechanism proposed and implemented:
    swap-test conflict gating (`--conflict-gated`), which weights each
    pair's repulsion by measured cross-feed harm. Unproven until the
    factorial rung reports. *Updated by F50:* the penalty is load-bearing
    and not incidental — with the read path held fixed (disjoint oracle
    fragments) and every anti-collapse mechanism removed, twin gradients
    conflict (cosine mean -0.134, negative at 27/40 checkpoints) and one
    twin takes the plant outright (0.062/1.000, seed 69317). So it cannot
    simply be deleted to permit sharing; it must be *replaced*. Three
    published structural replacements that balance usage without
    forbidding shared fragments are recorded in `docs/LITERATURE_MAP.md`
    S4 — a bias control loop on selection scores, balanced assignment,
    and a shared+private partition with orthogonality on fragment CONTENT
    rather than on context selections. Next: swap the repulsion penalty
    for the shared+private partition and re-run the twins and the
    factorial rung against it.
12. *Necessity closed for the battery by F54; addressing still open.*
    **Addressing is per-context logits, not content-addressed** — the
    selector maps a known context label to fragments; it does not
    retrieve from observations. The promoted routing rung
    (`game_routing_native_actions_v1`) does fetch from opaque events, but
    the two have never been composed. *Addressed and failed by F43:*
    `ContentRouter` queries from the controller's own intention after a
    feedback-gathering probe (the only admissible source, since twins
    render identically). It collapses to identical selections for both
    twins on both seeds -- F12's failure recurring at the addressing
    level once the oracle is removed. Cause: the query is read from a
    representation that is itself still learning, so neither the encoding
    nor the router has a gradient until the other works. Next: stage the
    CONTEXT ENCODING, not the assignment -- a probe phase supervised by
    the sign of the agent's own next reward under a fixed test action. *Updated
    by F53/F54:* that probe phase turned out to CONTAMINATE the gate --
    its fixed test action performs choiceA's task, so the harness scored
    0.961 on that twin with no agent at all, and F48's mastery and decoy
    readings are withdrawn. Separately, F54 re-read the staggered battery
    against measured floors and found BOTH necessity gates (bank withheld
    and norm-matched decoy) passing on every discriminating game across
    three seeds -- so the bank demonstrably carries the skill when
    fetched by oracle or per-context selection. What remains open is
    strictly SELF-addressing: the agent inferring its context unaided.
    Next: read the corrected post-probe-scored re-runs, and if the probe
    must stay, score only what follows it.

17. **Two battery gates cannot discriminate, and every gate lacked a
    measured floor until now** — avoid1 has 0.020 of headroom between its
    measured chance floor (0.902, a constant action) and its calibrated
    ceiling (0.922); dualAD and dualBC have under 0.10. Results reporting
    avoid1 near 0.9 were reporting a degenerate policy as a pass.
    Evidence: F52 (`chance_floors.py`). The load-bearing games
    (choiceA/choiceB, dualAC, forageA, collect1, intercept1) discriminate
    properly, so the battery's claims stand, but raw mastery against a
    ceiling flatters any high-floor game. `CHANCE_FLOORS`, `headroom()`
    and `normalised()` now ship beside `SOLO_CEILINGS`. Next: replace
    avoid1 with an avoidance game whose degenerate policy is not
    near-optimal (e.g. one that requires movement to survive), re-read
    past battery reports on the floor-to-ceiling scale, and state every
    future gate against its measured floor.

18. **Every gate needs a no-agent control, and the addressing line must be
    re-run** — F53 found the co-trained loop's choiceA mastery (0.961) is
    exactly what the harness delivers with NO agent at all: the fixed
    probe action steps onto the positive-plane item, which IS choiceA's
    task, and `(total_reward > 0)` counts it. F48's "both twins mastered"
    and F48/F51's "the decoy gate fails on choiceA" are withdrawn; the
    whole ignorance escalation was chasing an artifact. Fix verified
    (score post-probe steps only; the twin asymmetry vanishes) and the
    re-run is in flight. Four measurement failures this session (F46,
    F49, F52, F53), each a plausible number rather than an exception.
    Next: make the no-agent control mandatory before any gate is
    reported — run the gate with no agent and confirm it FAILS — and
    re-state the addressing line's claims only after the corrected runs
    report on both seeds.

19. **The arity-3 executor is the binding constraint** — composition and
    cued addressing both work conditional on a plant that can follow
    every goal in its vocabulary, and at three goals that plant converges
    unreliably: restart draws of 3, 6 and 5 across three seeds, one seed
    failing all six draws, and uneven per-side competence even on success
    (0.29-0.88). At two goals the same machinery is reliable. Isolation
    always converges; joint training is basin-determined. Evidence: F57,
    `goal_composition_v1_2026-08-09`. Next: budget scaling (running),
    then sequential isolation with the promoted consolidation anchor
    protecting each acquired goal — the program's own continual-learning
    machinery used to build its own executor.

14. **Battery scale is six games; motor games are excluded** — the
    staggered battery (`staggered_battery_v1_2026-08-07`) holds six
    decision games at 0.72-1.38x solo ceilings, but forage/collect sit
    out (solo ceilings 0.02-0.08 at the fast budget) and seed 69317 is the
    softest of three (all pass the no-collapse gate; super-solo transfer
    replicates 3/3 seeds). Widening calibration shows every remaining
    family component is motor-class at the fast budget. Next: new
    decision games to widen the fast battery, and grown budgets to admit
    the motor games as the complexity ladder.

15. **Plant acquisition reliability is the binding constraint** — the
    two-speed assembly (`two_speed_battery`) retains well and keeps the
    fragment specification signature (twin cross-feed 0.000, both
    seeds), but hard motor games acquire on one seed and not the other
    (forageA 0.90/0.07; intercept1 1.65/0.05 across seeds). Evidence:
    F25. This gates further memory work: a bank cannot be evaluated on
    games the plant cannot reliably learn. Progress: the conv driver was
    rejected at the fast budget (F26), but the egocentric CROP lifted
    navigate 0.14 -> 0.33 and collect 0.55 -> 0.78 while costing
    intercept (F28) — encoder choice is per-game, and no single screen
    driver suits every game. Next: per-game encoder selection by
    calibration, wider seed pools on acquisition alone, and optimizer
    work on the recurrent controller.

16. **Cross-game encoder coupling has no admissible fix yet** — F29
    showed a shared screen encoder couples every game's representation;
    F30 showed the obvious fix (one encoder per game) is inadmissible
    because the encoder then absorbs the skill (twin cross-feed 1.000,
    i.e. fragment-blind). Admissible routes: joint calibration of one
    shared view, or task-invariant preprocessing. Next: battery-level
    search over a single shared view, gated on cross-feeding.

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
