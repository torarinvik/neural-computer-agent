# Goal-factored bank: design note

Recorded 2026-08-08, before implementation, so the gates cannot drift to
fit results. Source: design discussion following F50-F54 and the
literature sweep (`LITERATURE_MAP.md`).

## Motivation, from measurement

- F50: twin conflict is real gradient conflict over the WHOLE plant
  (cosine mean -0.134), and private fragments do not remove it — the
  contradiction pervades the policy because the policy is the unit of
  storage.
- F53/F54: the bank demonstrably carries skill when fetched per-context
  (necessity gates pass on three seeds), but self-addressing remains
  open, and the probe harness contaminated its own gate.
- Composition: four opaque-fragment mechanisms failed at chance, which
  kernel theory predicts (LITERATURE_MAP S3). The two published escapes
  are the MLC study channel and an imposed composition algebra.

## The factorization

Four decisions, each with its rationale:

**1. A micro goal is a single state transition.** The goal vocabulary is
the edge set of the world's state graph — finite, enumerable, and
game-independent. Edge-competence ("make s -> s' happen") is checkable by
the agent itself without the verifier, so it can be trained reward-free
during exploration with zero rule leakage. Credit assignment is one step,
one verdict. Probe evidence arrives per-edge instead of per-episode.

**2. The plant stores the map; the bank stores destinations.** Successor
/ LMDP-style split: dynamics (game-invariant geography) live in the
plant's weights as the legitimate cross-context marginal; each game's
fragment is a small destination/reward vector — the deviation from the
marginal, exactly where the storage rule wants game-specific content.
Edge preference is DERIVED each step ("which reachable neighbour is
closer to good?"), never stored per-edge. A per-edge preference table
would re-create the full policy inside the bank and re-fail composition
by conjunction-memorisation.

**3. Deliberative search is depth-n edge preference.** Following the
value gradient over the map is depth-1 search; the WAIT/THINK/COMMIT
loop (weakness 9, unused) runs the same operation deeper by expanding
imagined transitions in the agent's own map. Backtracking happens in the
learned model ONLY — checkpointing the verifier would hand over
consequence-free access to true reward and dissolve the twins' probe
requirement. `ContentAddressedMemory` (weakness 8, unwired) becomes the
transposition table: episodic cache of evaluated states. Memory roles:
fragments = procedural (what to want), map = semantic (how the world
works), transposition entries = episodic (what I already worked out).
Search cannot resolve twin identity: the map is shared, so imagination is
identical in both twins; the hidden bit exists only in the real reward
channel. Identification stays with the probe, execution deepens with
compute. That separation is a feature — search cannot become a leak path
around the bank.

**4. Cued twins as a parallel rung, not a replacement.** A game-name
banner rendered in the world (a percept through the shared encoder, not
an ID input) is the honest form of context labeling. The uncued twins
remain the graduation exam — they are the only case that forces the
probe. F30's trap applies: with a visible cue the cheapest solution is
name->policy in weights with the bank decorative; the gates below exist
to catch exactly that.

## Deliberate relaxation, recorded per promotion culture

Destination fragments have imposed semantics (they are read as "where
good is"). This is the imposed-algebra route LITERATURE_MAP S3 catalogued,
adopted deliberately after four fully-opaque composition mechanisms
failed at chance (F16/F27/F33/F34 — the archived rejected alternative).
The interface stays as narrow as possible (a small vector between
inference and execution), which is also the bottleneck Kobayashi et al.
found necessary for in-context composition. If the study-channel route
later makes fully opaque fragments compose, this relaxation is to be
revisited.

## Pre-registered gates

Rung A — goal-factored twins (cued first, then uncued):
- Both twins mastered vs measured floors (F52 scale, post-probe scoring
  per F53; no-agent control run FIRST and must fail).
- Cross-feed must invert EXACTLY (destination algebra: w_B = -w_A makes
  inversion a theorem; anything short of full inversion means the
  fragment is not what is being followed).
- Decoy: behaviour must be IDENTICAL across twins (action-distribution
  divergence ~0), not merely at-chance — the sharper criterion F51
  motivated. Score also vs per-context floors.
- Label-swap (cued rung): banner B over world A must swap behaviour
  wholesale.
- Necessity under cue (cued rung): banner visible + noise fragments must
  collapse to floor. Detects F30-style name->weights bypass.
- Cue-absence fallback (mixed curriculum only): occlude banner
  mid-episode -> degrade to probe-driven fetch, not to chance; probe-mode
  performance measured DURING the mix (staging law: cue-fetch and
  probe-fetch are different addressing policies).

Rung B — composition through destinations:
- Held-out pairings from `compose_suite` with destination arithmetic;
  must beat the 0.39-0.47 random-bank/floor band that four opaque
  mechanisms could not.

Rung C — deliberative search:
- New plan-required game (trap corridor: greedy transition dead-ends;
  3-4 plies find the detour). Performance monotone in think-budget on the
  trap game, FLAT on reactive games. Fragments still pass cross-feed +
  decoy through the planner. Cutting transposition capacity degrades deep
  search specifically.

## Sequencing

1. Corrected co-trained re-runs report (in flight) and are judged against
   `BARS.md`.
2. Rung A cued -> uncued.
3. Rung B on A's map.
4. Rung C on A's map.

Every rung: two seeds minimum, no-agent control first, matched
configurations, archives with README + SHA256SUMS on promotion.
