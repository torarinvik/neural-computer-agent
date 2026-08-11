# Known weaknesses ledger

One line of truth for what is currently weak, why it matters, and the next
concrete action. Every entry cites its evidence. Update this file whenever a
rung promotes, rejects, or qualifies. Ordered by severity.

## Open

0. **The recipe architecture is the live direction. SEARCH is still
   its bottleneck, but the bottleneck moved and is now better
   characterised (F155, F160-F175).**
   End to end: mean held-out 0.9742 against a 0.5623 identity floor,
   above floor in 21/21 family-seeds, interpreter executing unseen
   programs at 0.9916, no gradient touching any family (F169). Against
   F155's 0.9247/0.5229 the distance to a perfect recipe has roughly
   halved. Evidence: `recipe_search_v1_2026-08-11`.

   Search cost, each against the same frozen control, five seeds where
   noted, `cover` carried as a positive control of known size:

   | mechanism | diverse | related |
   | --- | ---: | ---: |
   | stored programs, with causal null (F161) | 0.929 | 0.772 |
   | coverage filter (F171) | 0.879 | 0.812 |
   | both (F171) | 0.848 | 0.711 |
   | **depth-ordered enumeration (F173)** | **0.425** | **0.406** |

   Only enumeration changes what is COUNTED; the rest shave constant
   factors. `toggle` went from 22,151 candidate evaluations to 328 —
   but that needed three findings composed: the hole was the MODULUS
   and not a missing pair operation (F160), the modulus should be
   OBSERVED rather than searched (F166/F167), and enumeration finds
   the two-instruction recipe immediately once it is (F173).

   What is now known NOT to help, each measured rather than argued:
   (a) storing solved programs adds nearly nothing on top of
       enumeration, 0.421 against 0.425 — reuse was worth a tenth
       against random sampling and about nothing against a systematic
       proposer (F173);
   (b) capping the enumeration is a net loss, because a failed
       enumeration is not wasted work: it raises the best score the
       fallback sampling starts from (F175);
   (c) the equality guard was NOT built — gated families reach 0.9607
       against 0.9781 for plain ones, so the hole it would fill is not
       there (F170).

   Next: a third of families still cost MORE under enumeration than
   under sampling, all of them ones whose recipe is longer than depth
   2. Depth 3 is 27,000 candidates and cannot be enumerated flat, so
   the open question is whether the enumeration can be ORDERED — which
   cannot cut a solution, unlike the cap that was just refuted.

0. **THE READER'S TRAINING SIGNAL — the last piece, and it is now the
   only one.** Everything else in the composition mechanism is
   measured working: a shared per-element step composes and extends to
   unseen arrangements and unseen depths on two unrelated function
   families (F121, F125, F133, all at 1.0000); binding the entry ONCE
   instead of re-attending it per step gives 0.9983 per-bit
   conditioned execution at depth 4 across 256 worlds (F135); and the
   reader can produce entries that drive 0.9723 / 0.9478 on HELD-OUT
   worlds when given a consistent target (F138).
   What does not work is training the reader without a privileged
   target: task loss through a frozen plant gives 0.4973 (F136), joint
   training gives own == stranger (F135). Binding buys execution at
   the cost of a narrow entry target that task loss cannot search for.
   Contrastive signal (either form) beats joint training and beats
   task-loss-through-a-frozen-plant, but the ordering above that is
   NOT established: the auxiliary form's 0.7069 was single-seed and
   its replication gives 0.5405 (two-seed mean 0.6237, inside the
   phase form's range). See the F142 correction.
   RESOLVED for the batch knob: F144 replicates at three seeds
   (0.7993 / 0.8447 / 0.6945, mean 0.7795), the best non-privileged
   result measured — 56.6% of the way from joint training to the
   privileged ceiling, exact match 26x joint. The reader's objective
   obeys F78's diversity law: harder discrimination, finer code.
   Next: the remaining 0.7795 -> 0.9723. (F108's curve was re-checked
   2026-08-11 and my single-seed suspicion was WRONG — two seeds at
   every weight, optimum at 0.5 confirmed and the only weight where
   seeds agree.)

0-carry. **DONE, and it worked (F143).** Binding the games' value
   pathway — entry reduced to one vector added to the state token
   instead of attended over at every rollout step — took pooled
   held-out from +0.0995 to +0.1229 against a +0.1234 oracle-value
   target, on three seeds with no tuning. All three seeds now seek in
   inverted worlds (0.333/0.188/0.375, against 0.417/0.042/0.000),
   closing the polarity asymmetry open since F112. Evidence:
   `bind_value_games_v1_2026-08-10`.
   The games' remaining 31.9% of floor-to-full-oracle is F110's
   search-and-dynamics residual, which no improvement to the entry can
   reach — it needs a better transition model or a better search.

0-lesson. **The ignorance objective is toothless when the model is
   bad (F120).** It penalises being accurate WITHOUT the entry; a
   model that is inaccurate either way satisfies it trivially. It
   fixed F106 precisely because that model was already accurate on
   the twin-average. Precondition now stated: get the model predicting
   before expecting the ignorance term to force reading.

1. **CLOSED by F143** (kept for the trail). **Games polarity: the
   second salience channel is starved by the collection policy
   (F118).** Two-channel salience reached pooled
   held-out +0.0947 (ladder -0.0205 -> +0.0069 -> +0.0816 -> +0.0947
   against the +0.1234 oracle-value target) and normal worlds saturate
   at top=food 1.000 — but in every seed one polarity channel is alive
   at +-1 and the other is dead at ~0, because `--seek` steered
   collection at plane-1 objects only, so plane-2 consumption events
   are nearly absent and inverted worlds have nothing to learn "seek"
   from. Architecture has the slot; the data never filled it. F116
   refuted the F113 hypothesis: the polarity scalar DOES sign-split.
   Evidence: `two_channel_salience_v1_2026-08-10`,
   `polarity_scalar_diagnostic_v1_2026-08-10`.
   Next: balanced-plane seeking (running, 3 seeds). Then the +0.0720
   search/dynamics residual (F110).

2. **Retired metric: "% of headroom" must not be quoted again.** It
   passed 100% (119.5% at F118) because the twin-entry arm falls far
   below the context-free floor — a confident wrong-rule agent seeks
   poison rather than wandering — so the denominator was never valid.
   Report held-out reward against +0.1234 / +0.1954, and entry effect
   separately as an unnormalised causal magnitude. See the measurement
   correction entry in MEMORY_BANK_DESIGN.md.

0-prev-F112. **Polarity asymmetry found (F112): the value head
   suppresses but cannot promote** — top=food 0.219 normal vs 0.021
   inverted, poison avoidance transferring to both; correlation 0.17
   identical across polarity, defect at the top of the ranking.
   Evidence: `value_fidelity_v1_2026-08-10`. Addressed by F113's
   signed pathway (partially — one polarity only).

0-prev-F111. **Value head lands: 45.6%% of headroom, held-out positive
   for the first time (F111); +0.1165 remains to the oracle-value
   target.** Regressing the n-step return (the oracle's interface)
   instead of 3-class bins: held-out -0.0205 -> +0.0069, entry effect
   +0.0577 -> +0.1036, twin penalty deepens on both seeds. Evidence:
   `value_head_v1_2026-08-10`. Ladder: 0.2 -> 22.0 -> 25.4 -> 45.6%%.

0-previous. **DIAGNOSED BY ORACLE SUBSTITUTION (F110): the outcome model is the
   binding constraint — 63%% of the missing headroom — and the split is
   exact.** Ground-truth values through the same learned dynamics and
   beam reach +0.1234 (per-seed +0.1093/+0.1375) against the learned
   arm's -0.0205, i.e. 68.3%% of floor-to-ceiling. Decomposition of the
   whole games gap, measured: entry unread (fixed, +0.0499), object
   hallucination (fixed, +0.0078), OUTCOME MODEL (+0.1439, dominant),
   search+dynamics residual (+0.0720, unaddressed). Evidence:
   `oracle_substitution_v1_2026-08-10`.
   Next: improve the outcome model — 0.4474 balanced accuracy from
   12%%-dense labels through a bank entry. Ordinary candidates: more
   visits per world, an n-step value head instead of 3-class outcomes,
   better collection. Score any attempt against +0.1234 (what perfect
   values buy through this search), on this benchmark, with the twin
   controls and the model gate first.

0-unvicies. **Superseded: remaining ~75%% undiagnosed** — now diagnosed
   (F110). Tested and
   ruled out: the transition model (avatar slots 1.0000 — perfect where
   it matters; the 0.5842 exact figure is the object slots, which are
   stochastic by construction), the search's object rollout (freezing it
   is worth 3.4 points, 22.0%% -> 25.4%%), and the "nearest object only"
   state abstraction (REFUTED and backwards — worlds with 3 item pairs,
   where the abstraction is worst, capture 33.6%% against 19.8%% for the
   1-pair worlds where it is complete). Evidence:
   `search_object_freeze_v1_2026-08-10` (F109).
   The outcome model at 0.4474 balanced accuracy is the obvious
   remaining suspect but nothing shows it is binding. Two wrong
   diagnoses are already on record for this exact question (F101 state,
   F108 transition model), so the next step is a MEASUREMENT that
   discriminates rather than a fourth fix: an oracle-substitution
   ablation — replace the outcome model with ground truth and re-run the
   search. If performance jumps to the oracle, the outcome model is the
   binding constraint; if it does not, the search or the horizon is.
   That isolates it in one run and cannot be argued with.

0-vicies. **Superseded: the ignorance objective breaks the collapse —
   0.2%% to 22.0%% of measured headroom, with the weight curve
   characterised.** F58's phase-1 ignorance objective, applied to the
   bank entry as an entropy term pushing the entry-free prediction
   toward uniform. Model gate moved first as required: twin agreement
   0.9998 -> 0.5343, entry cosine 0.9855 -> 0.7119, outcome accuracy
   unchanged. Then behaviour: entry effect +0.0005 -> +0.0499, held-out
   -0.0466 -> -0.0217 (beating the best context-free policy at -0.0318
   for the first time), and the WRONG entry now actively harms
   (-0.0716 against a withheld -0.0480). Evidence:
   `ignorance_objective_v1_2026-08-10`,
   `ignorance_weight_sweep_v1_2026-08-10`.
   The weight has a threshold (0.1 does nothing — the collapse is stable
   against small pressure), an optimum at 0.5, and DECOUPLING past it:
   at 1.0 the model discriminates most but scores lower; at 2.0 the
   reader emits the most distinct entries while the model's
   discrimination falls back. The quantity to tune is the AGREEMENT
   between reader and model, not the separation of either.
   **Remaining 78%% is undiagnosed**, and the honest candidates are
   ordinary rather than architectural: transition model 0.5842 exact,
   outcome model 0.4474 balanced at its best. Beam search over models
   that inaccurate has a low ceiling however well the entry is read.
   Next: improve the two models directly and re-measure — the entry
   channel is no longer the binding constraint.

0-undevicies. **Superseded: reader and outcome model collapsed onto the
   twin-average (F106)** — diagnosed there, fixed in F107/F108. It is F58's failure, and F58's fix exists and
   was never applied here.** Scoring the models directly instead of
   through behaviour: the outcome model gives **0.9998 label agreement**
   between an entry and its INVERTED TWIN's, with a mean P(food) gap of
   **0.0000** — identical predictions under opposite rules, so no search
   could ever have distinguished them and the search was never the
   defect. The reader is no better: entry cosine with its twin is
   **0.9855**. Evidence: `model_level_diagnostics_v1_2026-08-10`.
   Also found: F103's class-balanced loss inverted F102's degeneracy
   rather than fixing it — per-class recall is cost 0.4672,
   **nothing 0.0000**, food 0.6575. F102's model always said "nothing";
   this one never does.
   Next: apply F58's phase-1 IGNORANCE OBJECTIVE — penalise performing
   well WITHOUT the entry, so reading is the only way to score. Standing
   requirement for that experiment: twin agreement must fall well below
   0.9998 and entry cosine well below 0.9855 BEFORE any behavioural
   claim. The model measurements move first, or the behavioural number
   means nothing.
   Reference points for any future attempt (F105, F106): floor -0.0318,
   ceiling +0.1954, twin agreement 0.9998, entry cosine 0.9855,
   transition exact accuracy 0.5842.

0-duodevicies. **Superseded: the benchmark exists and the stack fails it
   upstream of the bank** — true (F105), now localised precisely (F106). `forage` + `inverted` +
   `recentre_every` + `spawn_radius` gives a multi-step world where
   every inversion-invariant policy loses (idle -0.0499, random -0.0474,
   eat-anything -0.0318, fixed preference -0.0360) against an oracle
   using the hidden bit at +0.1954 — headroom of **+0.2272** reachable
   only by reading context. Validated before use, per F104's rule.
   The mechanism scores -0.0466 held out and **-0.0463 on TRAINED
   worlds**, worse than the best context-free policy and barely above
   idling; entry effect +0.0005, or 0.2%% of the headroom. Evidence:
   `context_required_benchmark_v1_2026-08-10` (F105).
   Because it fails on worlds it was TRAINED on, the defect is upstream
   of the bank. Next, and it is a diagnosis rather than a redesign:
   score the transition and outcome models DIRECTLY (prediction accuracy
   on held-out transitions) instead of only through behaviour. If the
   models are accurate and behaviour is still poor, the beam search is
   the defect; if the models are inaccurate, the 6-slot state or the
   collection policy is. No probe has ever measured them apart.
   The benchmark itself is the durable contribution: floor -0.0318,
   ceiling +0.1954, guarantee that the gap needs context. F100-F104
   spent four findings discovering that they lacked exactly this.

0-septendecies. **Superseded: the multi-step games do not test the
   bank** — true of the component variants (F104), and now moot: a
   multi-step game that DOES test it exists (F105).
   Handing the agent its INVERTED TWIN's entry — same components, same
   rendering, opposite rewards, the only actively wrong entry possible —
   changes behaviour by +0.0011 and +0.0007 on two seeds, with most
   per-variant differences exactly 0.0000. The entry carries none of the
   inversion bit. What density plus training DID buy (seed 69316:
   +0.0236 against an untrained control's +0.0075, large wins on
   `intercept`) is inversion-INVARIANT competence from the transition
   model and search. Evidence: `games_twin_control_v1_2026-08-10`.
   Contrast F99: on `dual` a stranger's entry drove reward to -0.100
   against +0.600. Same mechanism, same reader. `dual` resolves a trial
   every step and has no inversion-invariant policy; the multi-step
   variants leave most reward available without reading context, and
   gradient descent finds that policy first.
   **Test-design rule, now standing:** before using a game to test the
   bank, verify that an inversion-invariant policy scores near floor. If
   it does not, the game measures navigation competence and will report
   the bank as working or failing for reasons unrelated to the bank.
   Also standing: score against SEARCH WITH AN UNTRAINED MODEL, never
   the random-action floor (F103) — persistent directional motion beats
   a random walk for free.
   What remains open is a benchmark question, not a mechanism one:
   the battery has no multi-step game in which context is required.
   Building one means a multi-step task whose reward flips with a hidden
   bit — e.g. a `forage` variant with `recentre_every` and inverted
   twins, which the config already supports and no probe has used.

0-sexdecies. **Superseded: the games' floor was the wrong control** —
   true (F103) and now subsumed by F104. Beam search over a random-but-
   fixed value function produces PERSISTENT DIRECTIONAL motion instead
   of jitter, which covers more ground than a random walk and collects
   scattered food by accident. So "beats a random-action floor" is free
   for anything that moves consistently — and every games number in
   F100-F103 was scored against it. The correct control is SEARCH WITH
   AN UNTRAINED MODEL, which separates what LEARNING contributes from
   what SEARCH contributes. Evidence:
   `games_sparsity_fixes_v1_2026-08-10` (F103).
   Against that control: untrained +0.0075 (10/12), trained +0.0057
   (8/12), same seed. And the entry is still decoration — correct
   +0.0007, withheld +0.0013, stranger +0.0007, a gap of exactly zero.
   The three sparsity fixes DID help (held-out lift -0.0024 -> +0.0007;
   outcome-seeking raised the food class 3.54%% -> 12.18%%, the value
   target barely moved it) but did not close it.
   Next, all ordinary and all untested: 8000 updates over 38 variants
   may simply be far too few (F80 needed 40000 on a much simpler
   distribution); 12%% outcome density may still be too sparse; and a
   searching agent may need on-policy correction rather than one round
   of seeded collection. Re-score everything against the untrained-model
   control, never the random-action floor.

0-quindecies. **Superseded: outcome sparsity is why the games fail** —
   true but incomplete (F102); the fixes helped without closing the gap,
   and the baseline was the larger error (F103). Three formulations were tried and the third — a factored
   multi-object slot state with dynamics, outcome model and beam search
   — scored WORSE than an untrained plant on the same seed and the same
   held-out variants with bit-identical floors (-0.0013 vs +0.0071 lift,
   5/12 vs 10/12 beating floor). Evidence:
   `games_slot_state_v1_2026-08-10` (F102).
   Cause: "always nothing" scores 98.16% on random-play data, so
   cross-entropy has almost no gradient toward the 1.8% that matters;
   what signal remains favours COST 5:1; and beam search over that
   near-flat, cost-biased landscape avoids everything including food —
   consistently wrong where random play is only randomly wrong.
   This explains F99 too: `dual` worked because every step resolves a
   trial, so its outcomes are ~100% non-zero. The mechanism works where
   outcome signal is DENSE and fails at 1.8%.
   Next, and it is ordinary RL practice this line skipped: class-balanced
   or importance-weighted outcome loss, outcome-seeking rather than
   uniform-random data collection, and a VALUE target instead of an
   immediate-outcome target. None has been tried.

0-quaterdecies. **Superseded: the games need a factored multi-object
   state** — F102 built it and the result got worse, refuting the
   state-representation diagnosis (F101) outright. F100 tried greedy one-step reward prediction (no lift;
   a random plant matched 12000 updates). F101 added F67's missing half
   — transition model, value model, breadth-first search over both —
   and search doubled a negligible lift (+0.0016 vs +0.0008) while the
   NULLS collapsed: withholding the entry scored the same, a stranger's
   entry scored slightly better. On the one-step `dual` game the same
   nulls were brutal (stranger entry -0.100 reward, F99), so the
   contrast is diagnostic, not noise.
   The cause: avatar cell + one screen frame is Markov-insufficient.
   `intercept` has objects falling and `avoid` has hazards moving, so a
   single frame carries no velocity or phase and "safe now, lethal in
   two steps" is inexpressible. Evidence:
   `games_multistep_search_v1_2026-08-10` (F101).
   Next: feed the games' objects into the slot interface F71-F98 already
   has — avatar plus each faller and hazard, plus enough frames to
   expose motion. `schema_families.py` handles six slots of eight values
   and a composigrid frame with an avatar and two hazards is that shape.
   This has never been tried and is the single most direct experiment
   left.

0-terdecies. **Superseded: the games need multi-step derivation** —
   first contact with the battery is done (F99, F100,
   `games_rule_reading_v1_2026-08-10`). Reading a game's rule from
   observed outcomes WORKS where the game is a one-step decision: on
   `dual`, trained pairings reach 0.667 choice accuracy, withholding the
   entry drops behaviour to chance (0.241 vs 0.250) and a stranger's
   entry drops it BELOW chance with negative reward (0.083, -0.100) —
   a wrong rule makes the agent eat the wrong item on purpose.
   It does NOT work on multi-step games, and the reason is a probe
   defect rather than a mechanism one: `collect`/`intercept`/`avoid`/
   `navigate` pay out after several moves, while the probe predicts ONE
   action's outcome and acts by greedy argmax. Nearly every single action
   yields zero, so a random plant matches 12000 updates of training.
   Next: the combination F67 prescribed and `reacher_ladder.py` already
   implements — a transition model over an extracted factored state, the
   reward model built here, and SEARCH over both. The bank supplies the
   per-world content for each; the missing piece is the derivation.

0-undecies. **Rule diversity in the battery is capped by construction** —
   `dual` gives 9 rule pairings and the verifier rejects `arity` above 3,
   while F78 measured 64 procedural families producing memorisation and
   4096 needed for reading. Held-out generalisation on `dual` is
   correspondingly weak and seed-unstable (0.49/0.53/0.56 vs
   0.10/0.22/0.17). This is a statement about the benchmark: raising it
   means more cues or more item classes, which changes the games
   themselves.

0-duodecies. **Superseded: the games battery, untouched since F70** Everything in F71-F98 is measured on procedurally
   generated transition families plus the reacher expressed in the slot
   interface. The games have screens, rewards and a verifier-private
   harness; none of this mechanism has met them. That is the project's
   largest untested claim and it should be the next work.
   The bridge that exists: `grid` and `walled` ARE the reacher's r3/r4
   dynamics, and both are now read correctly (1.000 and 0.996). What is
   untested is everything the games add on top — perception, reward,
   addressing under identical-looking worlds.

0-decies. **CLOSED (F97, F98): the exception store works, and the
   approximate version matches the idealised one.** `walled` 0.894 -> 1.000 with 27 stored exceptions, both
   seeds, which is exactly the count of transitions on which `grid` and
   `walled` differ. Plant frozen, entry unchanged, zero gradient steps.
   The degeneracy check passes: ZERO exceptions stored for `grid`,
   `perm` and `line` at every observation budget, so the store grows
   only where rules fail and does not become a lookup table. Evidence:
   `exception_store_v1_2026-08-10`.
   Both caveats are now discharged (F98,
   `approximate_store_and_degeneration_v1_2026-08-10`). A realistic
   similarity-addressed store with capacity matches the exact dict
   (walled 0.996, toggle 1.000) PROVIDED the key preserves slot identity
   — mean-pooling collides and was net-harmful on toggle (0.992 ->
   0.951). And the degeneration case is real: a rule-free family grows
   the store to 249/256 entries. It is separated by violation rate at
   two orders of magnitude (0.0-2.1% where rules hold, 10.3% for
   rule+exceptions, 98.5% for no rule), which is a better guard than a
   capacity cap — capping chaos at 32 entries bounds memory but collapses
   accuracy to 0.127, i.e. fails silently.

0-nonies. **Superseded: the bank stores RULES and cannot store
   EXCEPTIONS** — the reacher's walled grid
   reads 0.894 at every seed and every budget, and `grid`/`walled` agree
   on exactly 229/256 = 0.8945 of transitions. The reader gets every
   non-wall transition right and every wall transition wrong: it reads
   "8x8 grid movement" and ignores the obstacle entirely. Not a budget
   gap (80000 updates repaired every other family and left this one
   unmoved), not a schema gap (the conditional primitive made things
   worse), not capacity (F77/F89). The obstacle is ~121 bits of
   arbitrary, incompressible content and an entry can only name a rule.
   Evidence: `rules_not_exceptions_v1_2026-08-10` (F95, F96).
   Next: wire `ContentAddressedMemory` as an EXCEPTION store holding
   (state, action) -> outcome for states where the rule fails, consulted
   before the rule. Prediction: `walled` 0.894 -> ~1.000 with 27 stored
   exceptions and the rule-bank untouched. This is the first result that
   says precisely what that unused infrastructure is for, and it
   supersedes the vague form of weakness 8 below.

0-octies. **Superseded: position-dependent dynamics are the ceiling** —
   the reacher's open grid is read perfectly at zero gradient steps
   (1.000, cold 50), but its WALLED variant reads 0.894 and costs 438
   updates to finish against a cold 88: five times worse than learning
   it from scratch, the first decisive failure of this mechanism.
   Every op the generator can express is a uniform function of slot
   VALUES; a wall makes an action's effect depend on which STATE you are
   in. Neither `--wide` nor `--balanced` reaches it. Evidence:
   `scaling_to_1024_and_reacher_v1_2026-08-10` (F92). Next: add a
   conditional/masked op primitive to the schema (effect applies only on
   a subset of states) and re-measure `walled`. This is the direct
   analogue of the F84/F91 fix that took `toggle` from 0.096 to 0.917,
   and it is the last identified capability gap.

0-bis. **Verify-step dependence GROWS with bank size** — key-only
   discrimination is effectively gone at N=1024 (a never-seen family
   matches its nearest key at 0.954, gap 0.037), while consequence
   verification still separates at 0.171. Retrieve-then-verify holds
   0.980 at N=1024 on a constant 4 plant passes against a 1024-pass scan
   at 0.853. Evidence: F94. Not urgent, but any design that drops the
   verify step to save 4 forward passes will fail silently at scale.

0-septies. **Superseded: gate passes at N=256** — 256/256
   families mastered, retention drift exactly 0.0, acquisition flat
   across a 4x-larger bank, retrieve-then-verify 0.994 at a constant 4
   plant passes (F87). `toggle`, hardest case since F79, went 0.096 ->
   0.917 by combining op vocabulary with slot-count balancing (F91).
   Evidence: `bank_scaling_256_v1_2026-08-09`,
   `distribution_shaping_v1_2026-08-09`.
   Two things remain, neither urgent: (i) the in-bank/stranger gap
   shrinks with N (0.068 at 256) but its decrements are DECELERATING
   (-0.109, -0.049, -0.039, -0.033, -0.027), so the linear-in-log
   projection to zero was wrong and the curve looks asymptotic —
   re-measure at N=1024 before believing either story; (ii) none of
   F71-F91 has touched the games battery, untouched since F70. That is
   now the largest untested claim in the project: everything here is
   measured on procedurally generated transition families, and whether
   it survives contact with the games is unknown.

0-sexies. **Superseded: discrimination unmeasured beyond N=64** —
   content addressing gives 1.000 retrieval at N=64 with zero plant
   forward passes, and retrieve-then-verify holds 1.000 at a constant 4
   passes, so recognising is now cheaper than relearning (F86,
   `content_addressed_retrieval_v1_2026-08-09`). But the in-bank/stranger
   gap shrinks with N for BOTH mechanisms — keys ~0.066 per doubling
   (0.325 -> 0.128 over N=8..64), consequence ~0.071 (0.571 -> 0.358) —
   and a stranger already matches its nearest key at 0.862. Key-only
   reuse would already fail; only the verify step supplies "none of
   these". Next: measure both gaps at N=128 and N=256. Do not
   extrapolate before that — four points and two seeds, and the
   runner-up margin's decrements are decelerating.

0-quinquies. **Superseded: retrieval is O(N)** — consequence probing
   identifies the right entry among 64 at 0.969 (chance 0.016, both
   seeds), with a working discrimination null (in-bank 1.000 vs stranger
   0.642). But a linear scan of 64 entries costs 64 plant forward passes
   while minting a fresh entry costs 2.7-7.0 update steps: at N=64,
   RECOGNISING a task already costs more than learning it. Evidence:
   `retrieval_by_consequence_v1_2026-08-09` (F85). Next: content-
   addressed keys for sublinear lookup, measured against this linear-scan
   baseline. The infrastructure has existed unused since the start (see
   weakness 8 below) — this is the first time there has been a measured
   reason to wire it in.
   Watch also, but do NOT act on yet: the in-bank/stranger gap shrinks
   ~0.07 per doubling (0.571 -> 0.358 over N=8..64). A linear-in-log
   extrapolation reaches zero in the low thousands, but four points and
   two seeds do not support that, and the runner-up margin's decrements
   are decelerating. Re-measure at N=128 and 256 before treating it as
   real.

0-quater. **Superseded: retrieval is the missing component** — the primary gate's (a) and (b) are met
   (64/64 families mastered, 56-59 of them by reading alone with zero
   gradient steps; retention drift exactly 0.0 across the grown bank).
   Clause (c) passes — acquisition 4.9 vs cold 50.6 over 64 sequential
   families, no positional drift — but it CANNOT FAIL as implemented,
   because entries are independent tensors fitted without seeing each
   other. The cost that would scale with bank size is finding the right
   entry among N, and no probe has ever paid it. Evidence:
   `bank_growth_and_wide_schema_v1_2026-08-09` (F83). Next: wire F57
   cued addressing or F44 consequence probing to a 64-entry bank and
   measure retrieval accuracy and cost against N. This is now the top
   item; everything else below is downstream of it.

0-ter. **Superseded: per-task gate MET (6.9x); LIFETIME gate bounded,
   and the pre-training axis is EXHAUSTED** — acquisition of a novel
   family costs 7.2 updates against 50.0 cold at 40000 pre-training
   updates (per-seed 6.2/8.3 vs 41.7/58.3, no overlap), with every plant
   weight frozen and retention delta 0.0000. Break-even has an interior
   optimum at 936 downstream families: saving is capped at cold's 50,
   40000 already captures 86% of it, and 80000 nearly doubles break-even
   to 1786. Below 20000 there is no saving at all. Evidence:
   `pretraining_budget_and_dissociation_v1_2026-08-09` (F80, F82). Both
   previously-stated next steps were refuted: 20000 was not padding, and
   longer pre-training improves rather than worsens break-even.
   Next: the remaining levers are NOT the budget. They are (a) sample
   efficiency of the reader itself, (b) whether break-even matters at
   all for a system premised on unbounded continual learning — 936 is
   finite and the plant is reused forever with exact retention, so the
   framing question is whether lifetime cost over a bounded task count
   is the right gate. Decide (b) before spending more on (a).

0-bis. **`toggle`-style structure is the one genuine coverage gap** —
   simultaneous multi-slot effects absent from the op vocabulary read at
   0.272 even at 80000 updates, against `perm` (also out-of-support)
   reaching 0.965. Widening the generator lifts it to 0.306 at 20000 but
   un-crosses the cost gate at that budget (acq 81.3 vs cold 57.9).
   Next: `--wide` at 40000+ updates, which is the untested cell.

0a. **Superseded: per-task gate MET at pool 4096; LIFETIME gate unmet** — a
   novel family costs 34.3 updates against 50.0 cold, both seeds
   individually, with every plant weight frozen, retention delta
   0.0000 and all nulls dead. But pre-training is 20000 updates against
   a 15.7-update per-task saving, so break-even needs ~1274 downstream
   families and 16 were measured. Evidence:
   `amortised_diversity_curve_v1_2026-08-09` (F78, F79). Next: drive the
   pre-training cost down (it has never been optimised — 20000 was a
   round number, not a measured minimum) and widen the generator's
   support so out-of-schema families like `toggle` (0.096 at every pool
   size) stop being a hard floor.

0a. **Superseded: the compounding gate is unmet on non-nesting families** — the frozen-plant/banked-content split
   removes forgetting outright (retention delta exactly 0.0000 over 96
   measurements) and shows causal structure transfer (held-out accuracy
   0.973 mean schema-pretrained vs 0.626 scrambled vs 0.083 random
   plant), but fitting a bank entry costs ~2x a cold full-model fit in
   updates (123 vs 62; cheaper in only 2/12 runs). Evidence:
   `frozen_plant_content_bank_v1_2026-08-09` (F75). The remaining cost is
   gradient descent inferring content that a few dozen observed
   transitions already determine. Next: amortised entry inference — an
   encoder mapping (state, action, next state) triples directly to an
   entry, so acquisition is forward passes rather than gradient steps.
   Prediction recorded in advance in `docs/MEMORY_BANK_DESIGN.md`.

0b. **Superseded detail of the above** — F67-F70's
   downward acquisition-cost curve was measured on rungs that share one
   state space and agree on every shared (state, action) pair, so nothing
   was ever contradicted. On four families whose dynamics genuinely
   differ, the policy-free model forgets to the chance floor (`line`
   0.138 against a 0.125 floor) and its sequential saving is smaller than
   a scrambled-dynamics control's. Evidence:
   `schema_family_disjoint_dynamics_v1_2026-08-09` (F71, F72). Next: the
   split F73/F74 forces — structure pre-trained into a FROZEN slot-
   symmetric plant, per-family content held in the bank. Prediction
   recorded in advance: flat retention, cost below the 280-update cold
   total, no negative transfer; if retention still collapses, the bank
   interface is leaking content into weights.

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

19. *Reframed by F58: not arity, memorisability.*
    **The executor learns an unconditional habit instead of reading its
    goal** — composition and
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
