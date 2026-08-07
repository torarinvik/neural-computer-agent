# Memory bank design: fragments, not programs

Grounded in a four-track literature review (2026-08-07): modular neural
representations; program-like skill libraries; context-as-program
conditioning; and the theory of compositional reuse (full agent reports
preserved in the session transcript). This document lays out the option
space, the convergent findings, and the chosen design. It extends
`DYNAMIC_BRAIN_ARCHITECTURE.md`; where they conflict, that document's
storage rule wins.

## The requirement

No "Snake program" and no "Pong program." The bank stores *fragments* —
sub-skill units shared across tasks — so that later abilities are built
from earlier ones (compounding) and overlapping tasks reuse structure
(transfer). The controller stays fixed; the bank grows without bound.

## Option space surveyed

| family | unit | composition | verdict for us |
| --- | --- | --- | --- |
| Modules / MoE | expert subnetwork | router selects | good isolation, weak composition; routers collapse; experts don't align with skills unless imposed |
| Hypernetworks | task embedding | embedding arithmetic (implicit) | best compression + CL story, but skills not individually addressable — conflicts with an auditable bank |
| LoRA-style weight deltas | low-rank delta on a frozen base | addition (task arithmetic) | best composition *theory* (weight disentanglement); but deltas patch the computing plant at fetch time, and post-hoc merging of separately trained skills does NOT share structure (near-orthogonal) |
| Fast weights | outer-product writes | temporal (re)programming | right formalism for a write/read head; not a durable store |
| Symbolic libraries (DreamCoder) | named typed function | application; library grows by MDL compression | the best compounding evidence anywhere; machinery is symbolic, but the *principles* (consolidation pass, compression gate) transfer |
| Options / successor features | sub-policy / psi-vector | sequencing; GPI / value algebra | the reward-only composition math; flat basis, no hierarchy — never consolidates a working composite back into the store |
| Context tokens (ours) | opaque tokens in the sketchpad | concatenation (safe), addition (powerful, needs task-linear geometry) | validated here (double dissociation); ceiling: elicits latent computation only, cannot install new algorithms |
| Discrete codes (VQ / "bytecode") | code from a finite vocabulary | sequencing + search | buys *temporal* composition and search; codebook collapse is chronic; the differentiable-interpreter version is a documented dead end (TerpreT) |

## Convergent findings (independent across all four tracks)

1. **Sharing must be induced at training time, never recovered by merging.**
   Independently trained per-task skills end up near-orthogonal and merely
   coexist. Fragments must be trained jointly over a shared inventory with
   explicit reuse pressure, or carved out by consolidation from solved
   tasks. ("Two maze games share a fragment" is an outcome of joint
   induction, not of post-hoc similarity.)
2. **Composition is itself a skill and must be practiced.** Systems compose
   to novel tasks only when the training loop constantly poses novel
   recombinations of existing fragments (meta-learning for
   compositionality; trained combiners beat hoped-for arithmetic
   everywhere). Corollary: every fragment must be exercised with varied
   partners, or it never detaches from its birth context.
3. **Selection must be discrete and separately learned; execution neural.**
   Learned soft routers are the universal failure point (collapse,
   misroute); differentiable program counters are a documented dead end.
   The winning pattern is a selector trained apart from the repertoire
   (our outcome-trained candidate router), making hard fetch decisions,
   with gradients flowing only inside fragment execution.
4. **Compounding comes from an explicit consolidation pass.** Wake/sleep
   abstraction, verified commits, chord-distillation: every system with
   compounding gains has a step that re-expresses old solutions in terms
   of shared substructure and promotes what compresses the corpus.
   Append-only libraries plateau. The bank needs a consolidation
   operation, with an MDL-flavored promotion gate: a fragment earns
   storage iff (uses x per-use savings) > storage/retrieval cost.
5. **Fragment boundaries come from prediction-error peaks and bottleneck
   states — never task boundaries.** Segmenting at task boundaries is
   exactly what produces per-task monoliths.
6. **Two-speed knowledge is mandatory, not optional.** Context tokens have
   a proven ceiling: they elicit and select computation latent in the
   plant; they cannot install new algorithms. New computation enters
   through the plastic core under (arbitrated) consolidation; fragments
   select and combine it. Breadth in the bank, depth in the weights.
7. **Task-linear geometry — the thing that makes additive composition of
   context vectors work — emerges only past a task-diversity threshold.**
   Three games are far too few. The suite needs a *family* of cheaply
   parameterized game variants sharing components, with every component
   value appearing in some combination (compositional support).

## The chosen design (combination)

**Unit of storage:** opaque fragment = a small set of learned tokens
(skill-as-context, our validated representation), at multiple grains
(primitive fragments and compiled chunks, keeping both a chunk and its
decomposition). Bank rows are ledger-protected, content-addressed by
event-derived keys.

**Induction:** joint training over a shared fragment inventory with a
learned task-fragment allocation (Polytropon-style), a mint-cost prior
(reuse-or-mint: try existing fragments first; mint on persistent failure),
and the ignorance/decoy/cross objectives carried over from the
externalization line so fragments stay content- and identity-causal.

**Composition:** concatenation into the sketchpad as the safe default
(validated; zero forgetting by construction), with the combiner trained by
constantly sampling novel fragment recombinations across game variants
(the MLC lever). Additive/arithmetic composition is adopted only if and
when measured task-linearity earns it.

**Selection:** the promoted `OpaqueCandidateGrowthRouter` making hard
fetches from opaque events, trained on outcomes with the established
null/permutation gates; selector and repertoire never co-trained into
each other.

**Growth and consolidation:** append fast (episodic tier, cheap);
periodically run a consolidation pass that (a) detects recurring
fragment co-activations and compiles chunks, (b) deduplicates
interchangeable fragments (swap test: if two fragments substitute with no
outcome loss, merge them), (c) promotes to the durable tier only what
compresses held-out solutions — all through the existing
promotion/verification gate culture. Chronic gradient conflict between
tasks on a shared fragment is the fragmentation criterion: split the
fragment, don't co-train harder.

**Depth:** the plastic core under arbitrated consolidation remains the
only place new computation is learned; the whole-plant protection rule
covers exactly the fixed components.

## What this predicts and how we falsify it

- Fragment sharing must be *measurable*: the allocation matrix should
  assign overlapping fragments to Pong and Breakout (sibling structure)
  and near-disjoint fragments to Snake. If every game claims a private
  fragment set, induction failed (the Csordás failure).
- Novel-recombination probes: hold out game variants whose component
  combination never appeared; the bank must solve them from existing
  fragments faster than from scratch, and withholding the predicted
  relevant fragments must remove the speedup (causal fetch audit).
- The mint rate must fall over time as the library covers the domain —
  the compounding signature. A flat mint rate means append-only drift.

## Immediate prerequisites

1. A parameterized game family (variants of snake/pong/breakout sharing
   components: interception, navigation, avoidance, collection) to
   provide compositional support cheaply. Without density, every negative
   result is confounded.
2. The shared-driver plant (done) and the externalization objectives
   (done) — this design composes them rather than replacing them.

## Sub-architecture catalog and empirical roadmap (2026-08-07)

Status tags: [proven] literature-backed; [novel] our proposal; [ours]
already validated in this repository. Each numbered strategy below maps to
the ranked list in the session record; sub-items are the candidate
implementations to be tested empirically, with adoption triggers.

### 1. Joint fragment inventory with reuse pressure
- 1a [proven] Learned task-fragment allocation matrix (Polytropon-style);
  task identity used verifier-side during induction only.
- 1b [novel] Attention-allocated inventory: the core's event-window
  attention over all fragments plus a sparsity penalty replaces the
  allocation matrix. Label-free; gate with collapse metrics.
- 1c [proven] Mint-cost nonparametric prior: reuse-or-mint, growth
  penalized; mint only on persistent outcome failure.
- 1d [novel] Ignorance-anchored inventory: decoy/cross ignorance
  objectives at fragment level force each fragment to carry distinct,
  causal content. Our distinctive contribution.

### 2. Compositional practice curriculum
- 2a [proven] Procedural variant generator: games = shared components x
  settings; dense training subset, held-out combinations for probes.
- 2b [novel] Adversarial recombination sampling: choose next variant to
  maximize fragment-reuse under uncertainty.
- 2c [proven] Interleaved schedule via fresh self-play on old variants
  (zero stored data; replay-free in our strict sense).

### 3. MDL consolidation pass
- 3a [proven-adjacent] Co-activation chunk compiler; keep chunk and
  decomposition (multi-grain).
- 3b [novel] Swap-test deduplication: merge fragments iff interchange
  costs nothing on held-out variants.
- 3c [proven] Compression-gated promotion; mint-rate curve is the audit.
- 3d [proven-adjacent] Sleep-phase consolidation on self-generated
  lifetimes.

### 4. Skill-as-context fragments
- 4a [ours] Flat token concatenation (default).
- 4b [novel] Typed slots: role-partitioned sketchpad positions.
- 4c [proven, conditional] Additive fragment arithmetic; adopt only after
  a task-linearity probe passes.
- 4d [proven] Layered injection (workspace as well as event window) if
  the small core under-conditions.

### 5. Discrete fetch, selector apart from repertoire
- 5a [ours] Candidate router extended to top-k fragment-set fetch.
- 5b [novel] Mid-lifetime sequential fetch conditioned on unfolding
  events.
- 5c [proven-adjacent] Deliberative fetch: WAIT/THINK over candidate
  fragment-sets before COMMIT (first deliberation-loop customer).

### 6. Two-tier CLS memory
- 6a [proven, parent repo] ContentAddressedMemory as episodic tier with
  ledger protection.
- 6b [proven] Regularity-gated promotion: only the predictable
  cross-episode core becomes a fragment.
- 6c [proven] Reward-weighted write strength via the controller's
  existing write-strength head.

### 7. Prediction-error segmentation
- 7a [novel-for-us] Self-supervised next-event surprise head proposes
  fragment boundaries.
- 7b [proven] Bottleneck-state detection (defer; heavy).

### Flagged novel combinations
- 1d + 3b: causally verified fragment identity (every fragment provably
  necessary, distinct, content-bearing) — our niche.
- 5c + 3a: reward-only consolidation loop over compiled chunks — the
  open gap named by the skill-library literature.

### Roadmap (empirical order, with decision points)

R1. **Game-family generator (2a).** Parameterized variants of
    snake/pong/breakout over shared components (interception, navigation,
    collection, avoidance) with compositional support. Everything
    upstream depends on task density. Deliverable: generator + holdout
    split + tests.
R2. **Fragment bank v1** = 1a + 1c + 1d + 2a + 2c + 4a + 5a on the
    shared-driver plant. Gates: fragment-overlap structure matches
    component sharing; per-fragment necessity/identity (1d nulls);
    novel-recombination speedup with causal withholding; falling mint
    rate. Decision point: if allocation fails to share, try 1b; if the
    core under-conditions, add 4d.
R3. **First consolidation pass** = 3b + 3c (+ 3a when co-activation
    data exists). Decision point: compounding signature (mint rate falls,
    later variants acquire faster) or diagnose with swap-test data.
R4. **Composition upgrades as earned**: 4c after a task-linearity probe;
    5b when within-episode demand shifts exist; 5c when fetch ambiguity
    is the measured bottleneck.
R5. **Two-tier memory (6a-6c)** once fragments stabilize; then
    declarative/recall games.
Standing rules: fast-iteration probes before full budgets; two seeds
minimum for promotion; every rung keeps the null/permutation/zero-replay
gate culture; rejected sub-architectures are archived with evidence like
everything else.
