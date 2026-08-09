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

---

# Revision, 2026-08-09: the universal reacher

Recorded after F58. This supersedes the executor half of the design
above; the bank half is unchanged and its composition result stands.

## What F58 forced

The executor was failing and the reason was not what any of our fixes
assumed. Six seeds, phase-1 only: budget 1500 and 3600 both 0/6,
hidden 32 and 64 both 0/6, EWC at 1, 10 and 200 all identical, and two
goals failing exactly as three did. One signature everywhere — one goal
at 1.00, the rest at 0.02.

The plant was learning an unconditional habit and never reading its
goal channel. **With a handful of goals that is a rational solution:
"always do the one thing" scores well. Under isolation it is
*optimal*.** Every curriculum we tried therefore built a habit first and
then asked it to become conditional, which is the one transition none
of them can make. Chan et al. (LITERATURE_MAP S1) describe this exactly:
few, frequent, fixed-meaning classes drive in-weights memorisation;
many varied ones drive in-context reading. Three goals is the
pathological regime by their criterion.

## The formulation

Not "learn goals g1..gN" but:

> **Given any state X, learn the fastest path to make any state Y the
> current state.**

One function, `pi(action | X, Y)`, with a distance `d(X, Y)` underneath.
Every task becomes a query. Tennis is "make the ball-returned state
current"; anti-tennis is "make a state where you are elsewhere current".
The task never enters the machinery — only the target does.

## Why this is a redesign and not a tuning fix

**Forgetting stops being a thing.** Catastrophic forgetting is an
artifact of N separate skills competing for one set of weights. With one
universal function there is no "skill 3" stored anywhere to be
overwritten by skill 4; there is a function that gets refined in one
region of its domain. Local degradation is still possible; "A destroys
B" is not, because A and B were never separate objects. This is a
stronger position than penalising drift, which is what this program has
been doing for weeks — the difference between preventing a collision and
removing the possibility of one.

**The storage rule becomes structural.** A universal reacher is
genuinely task-invariant: it is geography, not skill, so keeping it in
weights is not a violation. The bank holds target states — small,
per-task, composable. Prediction that follows: **adding a game should
require ZERO plant change, only a new target.**

**Conditioning stops needing enforcement.** A goal space too large to
memorise makes reading the instruction the only representable solution.
No update rule polices it; the task design does.

## What it does NOT solve

A perfect reacher tells you how to get anywhere and nothing about where
you should want to go. **The twins survive the reframe untouched**:
identical X, identical reachable Y, opposite desirable Y. Specification
is irreducible and it is exactly the bank's job. The reframe does not
replace the architecture — it cleans the split. Execution becomes
universal and free; wanting stays external and per-context. We have been
muddling this by asking the plant to hold both.

## The compounding claim, and its gate

The learned distance function IS a search heuristic. Perfect d means
greedy descent and no search; approximate d still prunes, because you
can rank candidate next states instead of expanding them. So every task
sharpens the heuristic that makes the NEXT task cheaper. That is the
amortised-planning bargain.

This makes the program's founding claim measurable for the first time.
Every rung so far measured RETENTION — did task 3 survive task 4 —
which is much weaker. The reacher predicts:

> **the cost of acquiring task N falls as N grows.**

Flat curve = we built a library. Downward curve = we built a map. This
is the headline gate for the reacher rung; it is falsifiable in a way
"we did not forget" never was. Note we have already seen its endpoint
without recognising it: the composition rung assembled held-out games
for ZERO learning, which is acquisition cost driven to nothing.

## Why tiny tasks first is the correct curriculum

Previously justified by iteration speed. The better reason: in a small
state space you can learn a nearly exact distance function, and an exact
heuristic in a small space transfers as a good-enough heuristic in a
larger space that shares structure. Bootstrap the map where search is
still cheap, then carry it where naive search would be hopeless.

## What makes it affordable

The (X, Y) pair space is quadratic, and hindsight relabelling is what
makes it tractable: **every trajectory is a correct demonstration of
reaching wherever it actually ended up.** Wander somewhere useless and
you have still produced perfect data for that destination. This is
re-derivation, not recall — nothing is stored — so the no-replay rule is
intact. It is the same property that made the micro-goal self-check
possible: goals are verifiable by the agent without the verifier.

## Honest limits

- **Representation is everything.** "Any state Y" is meaningful only if
  states are representable. Our 8x8 grids are; a rich world needs a
  learned latent space, and a universal reacher over a bad space is a
  universal reacher to the wrong places. This is where real systems
  break and we would inherit it.
- **Not all Y are reachable, and reachability is not symmetric.** The
  function must express "impossible" and "how close can I get", not just
  distance. Irreversibility is the sharp case — death is a state you
  cannot leave, and our intercept games have exactly that structure. A
  shortest-path framing quietly assumes you can retry.
- **Compounding needs shared structure.** Learning NYC does not shrink
  search in Tokyo unless what transferred was "cities are grids". Our
  family shares dynamics by construction; the real world only partly
  does. This is the boundary condition on the whole claim.

## Pre-registered gates for the reacher rung

1. **Reach rate** over random (X, Y) pairs, against a measured no-agent
   floor and a random-goal control.
2. **Conditioning**: same X, different Y must produce different action
   distributions. This is the property every previous executor failed;
   it gets a direct measurement, not an inference from task score.
3. **Held-out targets**: cells never commanded in training must be
   reachable, or the reacher memorised after all.
4. **Zero-plant-change**: a new game is added by supplying a target
   only; the plant is frozen and must not be touched.
5. **Acquisition-cost curve**: updates-to-mastery for game N as a
   function of N, with the no-reacher baseline as control. Downward
   slope is the compounding claim; flat is its refutation.
6. Every gate runs the no-agent control FIRST (weakness 18) and reports
   against measured floors (F52).
