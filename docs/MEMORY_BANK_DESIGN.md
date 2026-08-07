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

## The sharing rung: factorial contexts (R2b, 2026-08-07)

The twins rung proved the bank can *distinguish* two contexts. It could
never prove reuse, because contradictory twins share nothing to reuse.
A bank that only distinguishes is a bank of whole programs — the
"Snake program / Pong program" outcome the architecture exists to avoid.

The `dual` world makes reuse the only economical solution. One world runs
two independent binary rules: trials alternate between an A/B pair and a
C/D pair (four marks inside the same three planes), `inverted` says which
of A/B is edible, `inverted2` says which of C/D is edible. The trial kind
is visible; the rule for each kind is not. A context is therefore a
*product of two bits*, and contexts sharing a bit share a sub-rule.

Three contexts train (`dualAC`, `dualAD`, `dualBC`); the fourth,
`dualBD`, is held out — a novel recombination whose every rule was
learned elsewhere but never in this pairing. This yields three graded
predictions no two-context suite can make:

* **Allocation.** Pairs sharing a rule should share a fragment; the pair
  sharing none (`dualAD|dualBC`) should share nothing.
* **Cross-feeding.** A source sharing one rule with the target should
  leave that rule intact and break the other — *partial*, per-rule
  damage, where twins could only show total damage.
* **Composition.** `dualBD` should be servable by two already-trained
  fragments with no new content minted.

Per-rule accuracy (verifier-side scoring, never shown to the learner)
makes the middle prediction measurable: binary mastery cannot separate
"obeys one rule of two" from "obeys neither", since both sit near zero
net reward.

### Conflict-gated diversity (mechanism, this rung)

The anti-collapse diversity penalty of F12/F13 repels *every* pair of
contexts equally. That is correct when contexts contradict and actively
harmful when they overlap: it drives the bank toward one private program
per context, which is the failure this rung exists to detect. The
penalty and the compounding goal are in direct opposition.

The fix keeps the penalty and replaces its blanket constant with
evidence. A **swap test** periodically runs one context on another's
fragments and watches the scalar outcome. Harmless swap ⇒ the pair can
share, so the repulsion between them relaxes toward zero. Costly swap ⇒
they encode incompatible rules, so the repulsion holds. Per-pair weight
is an EMA of the normalized mastery drop.

This uses no privileged rule knowledge — only the reward the verifier
already returns — so it is deployable, not a training-time oracle. It
also doubles as the R3 swap-test instrument: the same estimates say which
fragments are redundant enough to merge.

## Empirical findings log (probe-driven design laws)

Each entry: probe evidence -> design law -> architectural consequence.
This log is the steering record; the roadmap bends to it.

**F1 (probes 1-2, 2026-08-07). Fragments acquire content exactly where
observation underdetermines the objective.** With fully observable
variants (food visible, hazards visible), withheld-bank mastery equaled
with-bank mastery: the bank was redundant *by construction* and stayed
empty regardless of ignorance-objective pressure. Consequence: the memory
bank's value concentrates on hidden objective structure (which twin am I
in, what am I trying to do, what did the instruction say) — the task
suite for bank development must contain ambiguity, and any future claim
that a fragment "stores a skill" must show the withheld-bank collapse.

**F2 (probe 3). Ambiguity is not enough — there must be no passive
escape.** With ambiguous twins where a risk-free policy (avoid
everything) satisfied half the suite, the plant converged to that
attractor and the disambiguation gradient never paid: fragments stayed
empty while inverted twins hit 0.90 via passivity. Consequence:
bank-forcing tasks must make ignorance *costly in every variant*
(passivity scores zero; wrong guesses fatal) — the forage-twin pattern.
More generally: fragment content forms only when the expected value of
disambiguation is materially positive under the current policy, which is
a cold-start condition to design for, not hope for.

**F3 (probe 1 + single-variant diagnostic). The stack works; interleaving
dilutes.** A single crude variant trains to 0.62 in 300 updates through
the full stack (shared plant, fragment context, discrete Plackett-Luce
selection), but six-way interleaving at the same per-variant budget
produced near-zero acquisition, and staged warm-up only partially
recovered it. Consequence: bank training needs either larger budgets than
single-task intuition suggests, curriculum staging, or per-variant
interference control; and metric design must respect heterogeneous reward
structures (survival-mastery for purely negative variants).

**F4 (standing, from the externalization line). Ignorance objectives are
the content-forcing mechanism, but they act on the plant, not the bank.**
They shape the core to be incompetent without context; they cannot create
a *reason* for context when the environment already reveals everything
(F1) or when passivity suffices (F2). All three mechanisms — ignorance
training, ambiguity, and no-passive-escape — are jointly necessary.

**F5 (probe 5). Simultaneous acquisition of contradictory twins deadlocks
— the cold-start theorem of context-dependent skill.** Two variants with
identical observations and opposite required mappings, trained jointly
from scratch, produced flat-zero learning (0.03-0.05 with 700
updates/twin, no competing variants): with uninformative fragments the
conflicting gradients cancel, so competence never forms, so
disambiguation never pays, so fragments never gain content — a fixed
point of nothing. Consequence: contradictory contexts must be acquired
*in stages* — anchor one context to competence with its fragment present,
then introduce the contradiction so the conflicting gradient has a
context difference to route through. Curriculum is not an optimization
nicety for a memory bank; for maximal-conflict content it is the
bootstrap mechanism itself.

**F6 (probe 6 + single-variant diagnostic). Lethal ambiguity has no
gradient: bank-forcing tasks must make wrong choices costly but
survivable.** Staged acquisition did not rescue the twins — and the
diagnosis was that stage A *alone* failed (0.031 after 500 dedicated
updates) while the unambiguous `collect1` reached 0.62 in 300. The
difference: in forage the wrong item was fatal, so random exploration had
negative expected value and the policy's best early strategy was to touch
nothing. Scoring passivity at zero (F2) is not enough when exploring is
worse than zero. Consequence: the third condition on bank-forcing task
design is *survivable error* — wrong choices must cost reward without
ending the episode, so the disambiguating gradient can accumulate across
many within-lifetime trials. Restated as a rule: for a memory bank to
gain content, the agent must be able to *afford to be wrong* often enough
to discover that context predicts which choice is right.

**F7 (800-update budget/ignorance diagnostics). Motor difficulty drowns
the disambiguation signal; the bank test must isolate the choice.**
Survivable-error forage stayed flat at ~0.20 for 800 updates both with
ignorance weight 0.5 and with it disabled (the latter *declined* to 0.11,
the signature of policy collapse under a weak gradient) — so the failure
was neither budget nor mechanism interference. Forage asks the plant to
learn navigation, approach, and avoidance *before* the which-type-is-good
bit can pay anything, and the navigation gradient dominates. Consequence:
bank-forcing tasks must isolate the disambiguation decision from motor
difficulty — the `choice` component places one item of each type adjacent
to the avatar so navigation costs one step and the only learnable content
is which type to take. General rule: when testing whether context can
carry a bit, make everything except that bit trivial; add motor
complexity back only after the bit is demonstrably carried.

**F8 (probe 7). Fragment-blindness: context must be audible, and warm-up
on a single context teaches the plant to ignore context.** With the
choice task learnable (0.31 -> 1.00 in 300 updates), staged twins still
failed — but the diagnosis was not the conflict: `choiceA` scored 1.00
with its fragments, 1.00 with pure-noise decoys, and 1.00 with the *other
twin's* fragments, while `choiceB` never left the floor. Identical scores
across every context condition mean the plant ignored the bank entirely
and simply learned "always take the plane-1 item". Two causes:
(a) **salience** — fragment tokens were initialised at 0.1 scale (norm
~0.8) while screen events are tanh payloads (norm ~4.7), so the bank was
an order of magnitude quieter than the observation and effectively
invisible; (b) **blind warm-up** — anchoring one context with
uninformative fragments trains a fragment-blind policy, after which the
tokens are dead inputs with vanishing gradient.
Consequences: (1) fragment tokens must be initialised at the same scale
as the events they share a window with — a memory bank whose entries are
quieter than perception will never be read; (2) contexts that must be
distinguished should be present from the first update, so the only policy
that earns reward in both is one that routes through the bank. F5's
staging law is hereby narrowed: staging helps when the *task* is
unlearnable, but hurts when it lets the plant reach competence without
consulting context.

**F9 (probe 9). The read path works: fragment content causally specifies
which policy runs.** With the selector replaced by an oracle (each twin
always receives its own distinct, salience-matched fragments) and no
warm-up, one plant mastered BOTH contradictory twins simultaneously —
choiceA 1.000 and choiceB 1.000 in visually identical worlds with
opposite rules. Withholding the bank collapsed both to chance
(0.297/0.313); noise decoys collapsed them (0.164/0.430); and feeding
each twin the OTHER twin's fragments drove mastery to exactly 0.000 —
far below chance, i.e. the agent systematically takes the wrong item.
Below-chance cross-feeding is the decisive signature: the fragment does
not merely enable competence, it *specifies which competence runs*.
Consequences: (1) skill-as-context in the event window is a sufficient
read mechanism for a fixed plant — no FiLM/weight-patching needed at this
scale, so the storage rule survives intact; (2) the remaining bottleneck
is isolated to the WRITE/SELECT path — probe 8 showed the outcome-REINFORCE
selector collapsing to identical picks for both variants, which is the
literature's predicted failure mode and now ours to fix; (3) all future
bank claims must report the cross-fed condition, since withheld-only
audits cannot distinguish "context enables" from "context specifies".

**F10 (probe 9 replication). Joint acquisition of contradictory contexts
is winner-take-all unstable.** The same configuration that mastered both
twins on seed 69316 (1.000/1.000) left one twin at 0.227 on seed 69317
while the other took the plant (1.000). The read mechanism is not at
fault — cross-feeding still collapsed the competent twin to 0.000 — but
two contexts competing for one plastic plant can end with one dominating.
Consequence: the bank training loop needs per-context progress balancing
(e.g. sample the lagging context more often, or normalise each context's
gradient contribution) rather than uniform interleaving. This is the same
interference tax seen at the consolidation level (F3, and the EWC ladder's
acquisition-headroom finding), now appearing at the context level — the
recurring structural theme of this program.

**F11 (probe 10, 2000-update replication). Budget resolves winner-take-all,
but the plant keeps one context as a free default.** At 2000 updates seed
69317 reached 1.000/1.000 with cross-fed collapse to 0.000/0.000 — the
full signature on a second seed, so F10's instability was largely a
budget artifact of joint contradictory acquisition. However the withheld
condition was asymmetric: 0.805 for choiceA versus 0.055 for choiceB. The
plant adopted A's policy as its default and used the fragment only to
*override* into B. Consequence: with a shared plant and contradictory
contexts, externalization is asymmetric by default — the first/easier
context's knowledge stays in the weights and only the deviation is truly
banked. Symmetric externalization requires the ignorance objective to
actually bite (higher weight or every-update application), and the
withheld-per-context spread is the metric that detects the asymmetry.
This does not weaken F9's specification claim (cross-feeding is 0.000 in
both directions on both seeds) but it bounds the storage claim: a bank
over a plastic plant stores *differences from default*, not whole skills,
unless ignorance pressure is strong enough to erase the default.

**F12 (probe 11). Context assignment must be stable before the plant can
learn to read it — the diversity penalty fixes collapse but not
inconsistency.** With learned selection, the anti-collapse penalty worked
exactly as intended (choiceA -> fragments [0,2], choiceB -> [3,4], fully
disjoint), yet neither twin learned and all four audit conditions were
identical: fragment-blind again. Cause: selection logits initialised to
zero give a uniform distribution, so the same context drew a *different*
fragment set almost every update. The plant saw inconsistent context and
correctly learned to ignore it; the selector, receiving no reward signal
in return, settled into disjoint-but-arbitrary sets driven only by the
(task-independent) diversity term. The oracle succeeded precisely because
its assignment was fixed from update one. Consequence: a learned selector
needs *early assignment stability* — distinct peaked logit initialisation,
temperature annealing, or a staged handover from oracle to learned
selection — otherwise the read path never forms and the write path has
nothing to learn from. Generalised: in a bank with both learned content
and learned addressing, one side must be held still long enough for the
other to become informative; simultaneous free optimisation of content,
addressing, and policy is a three-way deadlock (cf. F5).

**F13 (probe 13). Oracle-to-learned handover closes the loop: a
self-organizing bank.** Distinct peaked selection init (F12) broke
fragment-blindness but not the winner-take-all (0.242/0.781). Staging the
*addressing* did: 1200 updates with oracle-driven selection — during
which the learned selector is trained by KL divergence to imitate the
oracle's assignment — followed by 1200 updates with the selector in full
control. Result: choiceA 1.000 and choiceB 1.000, withheld 0.188/0.445,
noise decoy 0.219/0.219, cross-fed **0.000/0.000**, and the learned
selector held a disjoint assignment ([1,2] vs [4,5]) on its own. The F11
default-context asymmetry also largely resolved (withheld A: 0.805 ->
0.188). The imitation term is essential: without it, releasing control
hands the plant a fresh random assignment and destroys what the oracle
phase built. Consequence: the working recipe for a bank with learned
content AND learned addressing is *scaffolded addressing* — hold the
assignment fixed while the read path forms, transfer it into the selector
by imitation, then release. Nothing in the deployed system depends on
privileged information; the oracle is a training schedule, not a runtime
component.

**F14 (probes 14-15). Per-rule cross-feeding reads out a fragment's
content, and it shows the read path is not the bottleneck — acquisition
is.** Three factorial contexts under an oracle selector, once with
private fragments per context and once with one fragment per shared
sub-rule, both failed to acquire: `dualAC` reached 0.930 (per-rule
accuracy 0.96/0.89) while `dualAD` and `dualBC` sat at chance. But the
graded audit says the failure is not a read-path ceiling. Feeding
`dualAC`'s fragments into the other contexts decomposes exactly along
the shared rule:

| fed into | shared rule | per-rule accuracy |
| --- | --- | --- |
| `dualAD` <- `dualAC` | axis 0 (`takeA`) | **0.956** / 0.105 |
| `dualBC` <- `dualAC` | axis 1 (`takeC`) | 0.037 / **0.901** |
| withheld (no bank) | -- | 0.598 / 0.717 |

The rule the source and target agree on is executed near-perfectly; the
rule they disagree on is executed near-perfectly *backwards* (0.037 and
0.105 are far below the 0.5 chance line). Withholding the bank leaves
only a weak residual lean (0.60/0.72), so the sharpening is caused by the
fragment, not by a default. Two consequences:

*Measurement.* Cross-feeding plus per-rule scoring is an instrument that
reads out **what a fragment set encodes, rule by rule**, rather than
merely whether it helps. The F9 below-chance specification signature now
resolves per rule. Every future bank claim should report it; a scalar
mastery column cannot distinguish "obeys one rule of two" from "obeys
neither", and those have opposite architectural implications.

*Diagnosis.* The plant can execute a two-rule program delivered through
the event window, so capacity to *read* is not what fails at three
contexts. What fails is *writing*: the first context to acquire gets a
complete program written into its fragments, and every later context must
both author its own program and overcome the incumbent. This is F5's
cold-start deadlock and F11's default-context asymmetry compounding as
context count grows. Corollary for the roadmap: sharing cannot be tested
until three contexts can be held at all, so the disjoint read path is the
gating rung, not the factorial one.

Secondary observation (one seed, not a law): the factorial allocation was
*worse* than the disjoint one (winner 0.633 vs 0.930). A shared fragment
receives gradients from several contexts at once, and while those
contexts are still failing, those gradients conflict. Sharing may
therefore have to be *earned after* acquisition — consolidated into the
bank by a later merge pass (R3) rather than imposed as the initial
allocation.

**F15 (probes 17-19). A multi-rule world is only as good as its escape
audit: engagement is a mandatory readout, and blind guessing must pay.**
Redesigning the dual world for the sharing rung surfaced three coupled
design failures, each invisible to the metrics that preceded it:

1. *Rule encoding.* Marking the trial kind as item intensity (A/B at 1.0,
   C/D at 0.5) makes each rule a plane-x-intensity conjunction. A single
   unambiguous context, with the whole plant to itself, could not learn
   it (0.88/0.51). Trial kind must be a separate observable — a cue —
   orthogonal to the choice.
2. *Selective refusal.* With a free recentring step, agents mastered one
   trial kind and *declined the other forever* (engagement 1.6-3.6 of ~24
   trials), leaving per-rule accuracy meaningless on the refused kind.
   Scalar mastery could not see this; the new `rule_engagement` readout
   could. This is F2's passive escape resurfacing per-rule in any world
   with more than one reward source.
3. *Variance-aware idling.* Charging idling (-0.1) while a guess stayed a
   symmetric +1/-1 did NOT restore engagement. Under policy gradient a
   zero-mean high-variance action loses to a low-variance idle even when
   idling is worse in expectation. Engagement returned only when blind
   guessing became profitable (+1 right, -0.2 wrong: +0.4 expected).
   Reward asymmetry, not punishment, is what buys exploration — and
   mastery must then be scored on per-rule accuracy, since raw reward
   no longer distinguishes knowing from playing.

With all three fixed: uniform contexts (`dualAC`, `dualBD`) reach
1.000/1.000; conditional contexts engage fully but plateau at one rule
~0.9-1.0 and the other ~0.6 (mean ~0.78, three runs, two seeds, 3x
budget). The residual is not a world flaw — the cue is read (both rules
beat chance under full engagement) — but the plant's known recurrent-
acquisition limit applied to cue-conditional branching. Consequence for
the sharing rung: solo ceilings measured per context are the denominator
for every gate; demanding 1.0 of a conditional context would test the
plant, not the bank.

**F16 (probes 20-21). On a sound world, three contradictory contexts hold
at once — and imposed sharing buys nothing that composition can cash.**
With the F15 world fixes, the disjoint-oracle bank acquired all three
factorial contexts simultaneously at or above their solo ceilings
(dualAC 0.999, dualAD 0.832, dualBC 0.808 vs solo 1.00/0.78/0.78 — the
bank *helps* the conditional contexts, its first measured positive
transfer at acquisition time). The F14 acquisition deadlock is closed:
it was an artifact of the broken world, not a limit of the read path.
The graded specification signature is now fully resolved per rule:

| swap | shared rule | contradicted rule |
| --- | ---: | ---: |
| `dualAD` <- `dualAC` | 1.000 | 0.000 |
| `dualBC` <- `dualAC` | 1.000 | 0.002 |
| `dualBC` <- `dualAD` (share nothing) | 0.000 | 0.485 |

A fragment set is a two-rule program; swap it and exactly the rules it
contradicts invert while the rules it shares survive.

The factorial allocation (one fragment per sub-rule, shared by every
context that obeys it) also held all three contexts (0.992/0.757/0.747)
— refuting F16's precursor worry that shared gradients deadlock — but
was uniformly slightly *worse* than disjoint at matched budget, again.
And the payoff sharing was supposed to buy did not appear: handing the
held-out `dualBD` its two ideal already-trained fragments produced
0.337/0.506 — one rule *inverted*, the other at chance — no better than
the disjoint bank's composed audit and worse than selector-adaptation
over a random bank. A fragment trained inside two contexts does not yet
carry its rule into a novel pairing.

Consequences. (1) The convergent literature finding #2 is now measured
in-house: composition is a skill; fragments detach from their birth
contexts only if training constantly re-pairs them (MLC-style
recombination practice), which the two-phase curriculum never did.
Imposed allocation cannot substitute for compositional practice. (2)
Sharing keeps costing a little at acquisition and paying nothing at
composition, strengthening F14's corollary: reuse should be
*consolidated in after* acquisition (R3 swap-test merges) or *practiced
in* (recombination curriculum), never merely declared in the allocation.
(3) The rung's remaining promotion blocker is seed replication of the
disjoint result, not any new mechanism.

**F17 (probes 23-24). Quantity holds except where ambiguity is total —
and acquisition cost does not amortize yet.** Six calibrated games on one
plant at the fast budget (1800 updates total, ~300 per context vs the
~800 per context the promoted trio enjoyed): avoid1, dualAC, dualAD,
dualBC all hold at 0.90-1.00 of their solo ceilings on both seeds. The
casualties are the pure contradictory twins (choiceA/choiceB): 0.47/0.16
on seed 69316, 1.00/0.44 on 69317, a different loser each seed. The
twins are exactly the contexts with zero observational evidence and full
rule conflict, and they lose first when per-context budget shrinks —
winner-take-all returns at scale precisely where it was hardest to cure
at small scale (F10, F13). Note the choice games also overlap
observationally with the dual family (centre-adjacent pairs, no cue), so
the plant's bank-free default is contested by three families at once;
the withheld audit shows that default flipping between seeds.

Consequences. (1) The battery's per-context budget was ~2.5x below what
the trio needed; quantity-first exposes that the bank currently makes
storage cheap but not *acquisition* — each new ambiguous context still
pays nearly full price. The compounding claim predicts later contexts
get cheaper; measured: not yet. (2) Scheduling by recent mastery is not
enough at scale; the laggard signal dilutes across six contexts. The
next levers, in evidence order: budget accounting per context (equalize
effective updates, not sampling probability), the F13 oracle-anchored
staging applied to the twins inside the battery, and freeze-plant entry
for late ambiguous contexts so incumbents cannot contest them.

**F18 (probes 25-26). Sequential admission, not compute, is what scales
quantity — and staged contexts show the first super-solo transfer.** The
two levers for the F17 twin collapse, at matched total budget (3600
updates, seed 69316): doubling budget under laggard-balancing recovered
choiceA (1.00) but left choiceB at 0.42 — more compute does not resolve
a six-way contest for the plant's default. Staggered admission (one new
context every 300 updates, each contradiction arriving against an
already-anchored read path) put EVERY game at or above its solo ceiling:
choiceA 1.00, choiceB 1.00, dualAC 1.00, dualAD 0.80 (1.16x solo),
dualBC 0.76 (1.05x solo), avoid1 1.00, with the twin cross-feed still
0.0 both ways. The conditional dual contexts ending ABOVE their solo
ceilings is the first measured super-solo transfer: earlier contexts
made later ones easier, the compounding direction, visible only under
staging. F5's cold-start law is therefore not a small-scale artifact but
the scaling law for ambiguous contexts: simultaneous introduction is the
pathology, arrival order is the cure, and — conveniently — one-at-a-time
arrival is the natural continual-learning setting the architecture
targets anyway. Consequence: the battery's default protocol is staggered
admission; simultaneous introduction is demoted to a stress test.

**F19 (probe 29). The complexity ladder is blocked by plant acquisition,
not by the bank.** Grown-budget solo calibration (5x updates, steps 32,
seed 69316): collect1 0.47, navigate1 0.13, forageA 0.03, intercept1
0.03. Only collect responds to budget at all; the rest are flat — the
motor games are acquisition-limited, not budget-limited, which is the
shared controller's known recurrent-optimization weakness now measured
as the binding constraint on game complexity. Consequence: the next
complexity step is not "more updates" but either (a) motor curriculum
bridges (e.g. forage items spawning adjacent first — the choice trial —
then progressively farther, so the mastered decision skill seeds the
navigation skill), or (b) plant acquisition work directly. The bank
itself is not the bottleneck anywhere on the current map.

**F20 (probe 30). Spawn distance is not the bridge — trial structure is.**
The staged spawn-radius curriculum (r=1 -> 2 -> 4 -> unrestricted, two
seeds) failed at its FIRST stage: radius-1 forage reaches 0.125, nowhere
near the choice trial's 1.0, and later stages inherit the rubble (final
unrestricted 0.02-0.125, vs 0.03 cold). Radius-1 forage is not the
choice trial: choice FORCES a decision every step (avatar recentred,
both options adjacent), while forage merely starts items nearby — the
avatar wanders, the items stay behind, and the engagement collapse of
F15 returns in motor form. What made choice learnable was never
proximity; it was that every step is a trial. Consequence: the motor
bridge must relax the *forcing structure* gradually (recentre every
step -> every k steps -> never), not the spawn distance. Curricula must
preserve the property that made the anchor learnable, and that property
must be identified by ablation (F15's probes), not assumed.

**F21 (probe 31). Forcing transfers only if the learner performs the act
being taught — a teleport bridge teaches waiting, not navigating.** The
recentre-relaxation curriculum (forced trial every k steps, k=1 -> 3 ->
6 -> 12 -> never, two seeds) mastered every forced stage (0.94-1.00 at
k=12) and collapsed at k=0 (0.09-0.13, indistinguishable from cold).
Cause: recentring TELEPORTS the avatar to the decision point, so forced
trials alone pay positive reward and the skill under test — returning to
the decision — is never exercised; between forcings, idling is free (F2
in yet another costume). Together with F20 this closes the diagnosis
from both sides: F20 removed the forcing and lost the *decision*
engagement; F21 kept the forcing and lost the *navigation*. The bridge
that remains untested is the one where the AGENT does the returning:
no teleport — items re-dealt at Chebyshev radius r of the avatar's
current position, r growing with mastery, plus a per-step idle cost so
approaching strictly pays (the F15 economics applied to distance).
Curriculum design law: each stage must make the learner produce the
target behaviour, not merely reward states that the scaffold reaches on
the learner's behalf.

**F22 (probes 32-33). The motor wall is translation invariance, not
curriculum — egocentric rendering breaks it in one move.** The honest
no-teleport bridge (items re-dealt near the avatar, idle cost, staged
radius) failed at its FIRST stage (0.08 both seeds): even adjacent items
are not learned when the avatar sits at arbitrary screen positions. The
comparison that closes the case: the same decision at fixed geometry
(choice) is 1.00; at arbitrary position, 0.08. The flat screen encoder
gives the plant no translation equivariance, so every avatar position is
a separate policy to be learned from +-1 rewards — THAT is why every
motor game (forage 0.03, intercept 0.03, navigate 0.13 at 5x budget)
has been stuck while every fixed-geometry decision game promotes.
Egocentric rendering — the observation rolled so the avatar is always
centred, an encoder-side choice that touches neither the verifier's
privacy nor the amodal core — takes solo forage from 0.03 to 0.48/0.30
(seeds 69316/69317) with no curriculum at all. Consequences: (1) the
complexity ladder runs through the ENCODER, not the trainer — an
egocentric (or convolutional) screen driver is the next architectural
step, and the F20/F21 curriculum knobs become refinements on top of it;
(2) three probes spent on curricula located a representational failure —
the ablation discipline (change one property, compare to a mastered
anchor) is what caught it; (3) peripheral upgrades being decisive while
the fixed core stays untouched is exactly the division of labour the
architecture prescribes.

**F23 (probes 34-35). Laggard-preferential scheduling is catastrophic in
mixed-difficulty batteries: hopeless laggards capture the schedule.** Ten
games (six decision + four motor, egocentric, staggered, 6000 updates)
collapsed on both seeds: every motor game flat (0.03-0.17 of ceiling)
AND the early-admitted choice twins decayed after mastery (choiceA 1.00
-> 0.19 on seed 69316) — a retention failure the six-game battery never
showed. Cause: balancing samples contexts by softmax(-mastery/T), so
contexts stuck near zero receive ~e^4 times the sampling of mastered
ones. When laggards are merely slow this is the correct triage (F10);
when they are hard, they monopolize the schedule, the shared plant
thrashes on barely-learnable gradients, and mastered contexts starve
and drift. Difficulty diversity converts the cure for winner-take-all
into a new pathology. Consequence: the scheduler needs a bounded-share
guarantee — mix the laggard softmax with a uniform floor so every
mastered context retains a maintenance ration and no context's share
can exceed a cap. Deeper consequence for the architecture: retention
under a plastic plant currently depends on continued rehearsal through
the scheduler; the freeze-plant/consolidation path (F14, EWC line) is
what removes that dependence, and mixed-difficulty batteries are where
it becomes mandatory rather than optional.

**F24 (probe 37). The uniform floor cures retention; acquisition and
scheduling are separate axes.** With the floor genuinely wired in (the
probe-36 no-op is documented in the git log), the F23 decay pathology
vanished: choiceA 0.19 -> 1.00, dualAC 0.82 -> 0.97, avoid 1.00 — every
mastered context held under ten-game load. What the floor did not do:
motor games stayed flat (0.05-0.17 of ceiling) and choiceB was squeezed
(0.38). Scheduling and acquisition factorize exactly as the ledger
predicted. The motor residual is not a scheduler problem: ~600 shared
updates cannot do what 500 dedicated updates barely did (F22
calibration), and convergent finding #6 says fragments cannot install
the new computation navigation requires — only the plastic core can.
Consequence: mixed-difficulty batteries need the architecture's full
two-speed design — acquire hard games in a protected/solo phase through
the consolidating core (the promoted EWC/arbitrated line), then bank
their context — rather than asking joint bank training to do both jobs.
The battery harness as written co-trains everything; wiring the
consolidation line into it is the next build.

**F25 (probes 40-45). The two-speed assembly works, and the binding
constraint is now plant acquisition reliability, not memory.** Running
the program's two promoted lines as one system — protected acquisition
through the consolidating core, context stored in the bank, no replay —
produced, across three iterations on the same harness:

1. *Audit-path bug.* Acquisition fetched fragments by oracle index while
   the audit fell back to untrained selection logits, scoring every game
   against fragments it never trained with. Caught because the FIRST
   game, alone on a fully plastic plant, scored 0.23 against a 1.00 solo
   ceiling — an impossible number, not a disappointing one.
2. *Ignorance pressure is required in a consolidating loop.* Without it
   the plant keeps the first context's rule as a weight-level default
   (F11); consolidation then locks the default in and later contexts
   inherit it. Adding it moved forageA 0.07 -> 0.90 and intercept1 0.05
   -> 1.65 (seed-dependent), and lowered worst-case forgetting.
3. *A phase must be a conflict GROUP, not a game.* Sequencing suits
   contexts that differ in what they SHOW (F18) and fails for contexts
   that differ only in what they MEAN: consolidating twin A asks twin B
   to invert a rule the penalty is holding still. Measured with twins
   sequential: choiceA 1.00 / choiceB 0.17-0.20, forageA 0.90 /
   forageB 0.00. Grouping twins into one joint phase (balanced sampling,
   uniform floor, one consolidation per member against the shared
   anchor): choiceB 0.39 and 0.88, with `choiceA<-choiceB` cross-feed at
   **0.000 on both seeds** — the specification signature survives the
   assembly, and seed 69317 held choiceA 1.00 *and* choiceB 1.00 at
   acquisition.

What remains is not a memory failure. Retention is good (worst
forgetting 0.20-0.33, usually 0.00-0.10), the decision games sit at
0.77-1.45 of solo ceilings, and cross-feeding still inverts behaviour.
The residual is that hard motor games acquire on one seed and not the
other (forageA 0.90 on 69317 / 0.07 on 69316; intercept1 1.65 on one
run / 0.05 on the next) — a per-seed acquisition lottery in the plant,
the same recurrent-optimization weakness the ledger has tracked since
the BPTT diagnostic. Consequence: further memory-architecture work is
now blocked behind plant acquisition reliability, which is the honest
next front (wider seeds, optimizer/architecture work on the controller,
or a convolutional screen driver — F22 showed encoder structure moves
motor acquisition more than any trainer change).

**F26 (probe 46). The convolutional screen driver is rejected at the
fast budget: equivariance does not pay for its own optimization cost.**
Predicted from F22 (the motor wall is translation invariance) and F25
(encoder structure moves motor acquisition more than trainer changes), a
two-layer conv frontend should have beaten both the linear driver and
the egocentric roll. Measured at matched budget (500 updates, seed
69316), solo ceilings:

| game | linear | egocentric linear | conv |
| --- | ---: | ---: | ---: |
| forageA | 0.031 | **0.453** | 0.031 |
| collect1 | 0.469 | **0.547** | 0.031 |
| intercept1 | 0.031 | **0.313** | 0.016 |
| navigate1 | 0.125 | **0.141** | 0.016 |

Conv is worst everywhere — below even the plain linear driver it was
meant to replace. The mechanism is plausible: the conv stack multiplies
the frontend's parameter count (16 channels x 64 cells -> event width,
versus 3 x 64 -> event width) and adds depth, and REINFORCE from scalar
outcomes is a weak enough signal that the extra optimization burden
outweighs the structural prior at this budget. The egocentric roll gets
the same invariance for free because it moves no parameters at all.

Standing lesson, consistent with F20/F21: a structural prior that is
CORRECT can still lose to a cheaper trick when the learning signal is
too weak to pay for it. Architecture claims must be settled at the
budget the program actually runs at, not at the budget where the theory
is prettiest. The conv driver stays in the codebase behind
`conv_screen=False` for a future test at larger budgets; the egocentric
roll remains the promoted motor fix.

**F27 (probe 47). Compositional practice makes fragments interchangeable
without making them composable — partner-swapping is necessary but not
sufficient.** The MLC lever, implemented as `practice_map`: two
interchangeable fragments per sub-rule with a fresh partner combination
drawn every update, so no fragment can co-adapt with a habitual partner.
Training held (dualAC 0.96/1.00, dualAD 0.79/0.84, dualBC 0.83/0.76 on
seeds 69316/69317), which is itself the first result: fragments trained
against *rotating* partners still support their contexts, so the bank
tolerates interchangeability.

But the held-out recombination did not compose. Every combination of
`dualBD`'s two rule-fragments scored 0.27-0.57 (seed means 0.55 and
0.40) against a random-bank control of 0.47 — at chance, indeed
indistinguishable from feeding the plant a randomly initialised bank.
The one thing practice did buy is visible on seed 69316: the spread
across partner combinations fell to 0.042, i.e. the four combinations
became genuinely interchangeable. Seed 69317 kept a 0.298 spread, so
even that is not seed-robust.

Reading. Interchangeability and composability are different properties,
and only the first is bought by partner rotation. A fragment can be
robust to *which* partner it appears with while still encoding "the rule
that applies in the contexts I was trained in" rather than a portable
"take type B". Under the taxonomy of convergent finding 2, rotating
partners within the training set is not the same as practicing
composition, because the model never once had to *succeed* on a pairing
it had not seen — practice varied the fragments, not the tasks.

Consequence, and the strongest architectural statement the composition
line has produced: the missing ingredient is held-out pairings inside
TRAINING. The curriculum must repeatedly withhold a combination, require
the agent to solve it from existing fragments, and score it — which is
exactly meta-learning for compositionality, and is a curriculum change
(a train/holdout rotation over rule pairings) rather than a bank
mechanism. Three attempts (imposed sharing F16, partner rotation F27,
and the never-run additive route) now converge on the same answer:
composition must be a training objective, never an emergent hope.

**F28 (probe 48). Egocentric CROP beats egocentric ROLL where geometry
is real, and loses where it is not — the encoder must respect what the
game means by an edge.** Replacing the toroidal roll with a zero-filled
shift (same centring, no wraparound) at matched budget, seed 69316:

| game | linear | egocentric roll | egocentric crop |
| --- | ---: | ---: | ---: |
| navigate1 | 0.125 | 0.141 | **0.328** |
| collect1 | 0.469 | 0.547 | **0.781** |
| forageA | 0.031 | **0.453** | 0.312 |
| intercept1 | 0.031 | **0.313** | 0.016 |

navigate more than doubles and collect gains 43%: both are games whose
meaning depends on boundaries — walls, and a goal that may sit behind
them — and the roll was manufacturing walls by wrapping distant content
into view. intercept collapses under the crop, which is equally
sensible: a faller's meaning is its distance to the FLOOR, so cropping
the boundary away destroys the signal the game is about, while the roll
keeps the floor visible (wrapped, but present).

Design law: an egocentric transform trades absolute position for
relative position, and that trade is only free when the game's rules are
translation-invariant. Games anchored to a boundary (interception) want
the boundary; games anchored to local geometry (navigation, collection)
want the crop. Consequence for the architecture: the screen driver is
not one choice but a per-game encoder decision — exactly what the N
encoders of the amodal design are FOR. This is the first empirical
demand for encoder heterogeneity in the program, and it is a peripheral
decision, so it costs the fixed core nothing. Recommended defaults:
crop for navigate/collect/forage-class games, roll for intercept-class,
both available behind flags and both settled by calibration rather than
assumption.

**F29 (probe 49). Per-game encoder views raise the floor but couple
through the shared plant, so encoder choices cannot be calibrated
independently.** Giving each battery game the view its solo calibration
preferred (crop for collect, roll for forage/intercept, none for the
centred decision games) improved the worst-case ratio on both seeds
(0.07 -> 0.17 and 0.05 -> 0.10) and lifted collect sharply (0.29 -> 0.66,
0.43 -> 0.80). It also *cost* games whose own view never changed:
choiceB 0.88 -> 0.22 and dualAD 1.45 -> 0.69 on seed 69317.

The mechanism matters more than the numbers. Every game trains the same
plant, so changing how forage and collect are rendered changes the
representation that choice and dual inherit. A per-game encoder decision
is therefore NOT a local decision, and solo calibration — the tool this
program has leaned on since the battery began — measures each view in
exactly the condition that does not hold when games share a plant. The
twin cross-feed also drifted off zero (0.000 -> 0.188/0.203), the first
softening of the specification signature in the battery line.

Consequences. (1) Encoder heterogeneity is real (F28) but must be chosen
jointly, by battery-level search or by giving each game its own encoder
*parameters* rather than only its own transform — the latter is what the
amodal design actually prescribes, and the current shared screen driver
violates it for convenience. (2) Solo ceilings remain the right yardstick
for whether a game is learnable at all, and are NOT a reliable predictor
of a configuration's value inside a shared-plant battery. (3) The
regression in cross-feeding says the specification property must be
re-gated after any encoder change, not assumed to carry over.

**F30 (probe 50). Per-game encoders are REJECTED: they buy their gains by
absorbing the skill, and the cross-feed audit catches it.** Giving each
game its own screen encoder — which the amodal design's "N encoders"
appears to license, and which F29 motivated as the fix for cross-game
coupling — produced the best battery numbers the program has recorded:

| readout | shared encoder | N encoders |
| --- | ---: | ---: |
| choiceA / choiceB (69316) | 0.72 / 0.31 | **1.00 / 1.00** |
| choiceA / choiceB (69317) | 0.78 / 0.22 | **1.00 / 1.00** |
| worst forgetting | 0.203 / 0.219 | **0.048 / 0.062** |
| mean solo ratio (69317) | 0.61 | **0.84** |
| **twin cross-feed** | 0.19 / 0.20 | **1.000 / 1.000** |

The last row voids the rest. Cross-feeding `choiceB`'s fragments into
`choiceA` should inverte behaviour toward 0.000 (the specification
signature, held on every promoted rung). It scores 1.000: the fragments
no longer matter at all, because each game's ENCODER now carries its
rule. The twins were "solved" by giving each twin a private program —
precisely the per-game-model failure this program exists to avoid, and
precisely what was ruled out at the outset ("let the frozen core carry
small trainable side-state per game" — rejected as defeating the
research focus). The measurement now confirms that instinct with
evidence rather than principle.

Standing law, the sharpest the program has: **any per-game trainable
component will absorb the skill if allowed to, and performance gains are
not evidence of architectural progress — cross-feeding is.** The N
encoders of the amodal design mean one encoder per MODALITY (a screen, a
sound, a text stream), never one per task; a per-task encoder is
weight-stored skill wearing the architecture's vocabulary. F29's
coupling problem is therefore real but must be solved without per-task
frontends: joint calibration of a shared view, or a single encoder with
a task-invariant preprocessing, are the admissible routes.

Consequence for methodology: every future performance improvement must
report the cross-feed audit in the same table. This rung improved four
metrics and destroyed the one that mattered, and only the audit would
have revealed it.

**F31 (probe 51). Joint view calibration: the shared roll is the best
admissible configuration, and it is the only one that keeps the
specification signature intact.** With per-task frontends ruled out
(F30), the view must be one choice for every game. All four admissible
configurations, seed 69316, two-speed battery:

| config | mean ratio | worst | worst forgetting | twin cross-feed |
| --- | ---: | ---: | ---: | ---: |
| **roll (shared)** | **0.69** | 0.07 | 0.328 | **0.000** |
| per-game views | 0.71 | 0.17 | 0.203 | 0.188 |
| crop (shared) | 0.45 | 0.05 | 0.125 | 0.078 |
| none | 0.53 | 0.00 | 0.078 | 0.188 |

Per-game views edge the shared roll on mean (0.71 vs 0.69) and worst
(0.17 vs 0.07) but are inadmissible for the F30 reason and, tellingly,
already show the signature softening (0.188). Among genuinely shared
views the roll wins on mean by a wide margin (0.69 vs 0.53 and 0.45) and
is the ONLY configuration in the program's history with a twin cross-feed
of exactly 0.000.

The apparent trade is instructive: the configurations with the best
retention (`none` 0.078, `crop` 0.125) are the ones that learned least,
so their retention is cheap. Forgetting must always be read against how
much competence existed to lose — a lesson the ledger should apply
retroactively to any retention claim made without an acquisition column
beside it.

Note also that solo calibration mispredicted the battery outcome again
(F29): crop won solo on collect and navigate but is the worst shared
choice overall, and `none` beats crop despite losing to it on every solo
motor game. Component-wise calibration does not compose on a shared
plant. Decision: the shared roll stays the battery default; encoder work
is closed as a lever until a task-invariant preprocessing (not a
per-task frontend) is designed.

**F32 (probe 52). The pressure hypothesis is untested, because six
arity-3 contexts do not reliably acquire — acquisition gates every
memory question downstream of it.** The compose suite put nine rule
pairings over six rules on one plant, the first setting where
factorising is cheaper than memorising. Result:

| | seed 69316 | seed 69317 |
| --- | --- | --- |
| training pairings | 0.43-0.72 | **0.32-0.35 (all at chance)** |
| held-out `c02` / `c11` / `c20` composed | 0.52 / 0.36 / 0.47 | 0.33 / 0.35 / 0.34 |
| random-bank control | 0.40 / 0.34 / 0.28 | 0.33 / 0.31 / 0.32 |

Seed 69317 failed to acquire anything (chance is 0.333 at arity 3), so
its composition numbers measure nothing. Seed 69316 acquired partially
and shows composed scores above the random-bank control on two of three
held-out pairings (0.52 vs 0.40, 0.47 vs 0.28) — a weak signal, one
seed, inside noise. No claim is available either way.

The methodological error is worth more than the result: solo ceilings
were calibrated for individual arity-3 games (1.00 uniform, 0.44-0.50
conditional) and that was treated as evidence the SUITE was trainable.
It is not the same measurement. F29 already recorded that per-component
calibration does not compose on a shared plant; this rung shows the same
mistake made in the other direction, and the standing rule is now:
calibrate the workload that will actually run, not its parts.

Architectural consequence, and the honest summary of the program's
current state: the binding constraint is plant acquisition reliability
(F25), and it now blocks the composition question specifically. Adding
contexts to force factorisation adds acquisition load at the same time,
so pressure and difficulty cannot be varied independently on this plant.
Either acquisition gets reliable enough to carry nine pairings, or the
composition test needs a design where pressure rises without the
workload rising — e.g. holding the context count fixed while increasing
how many pairings each fragment must serve.

**F33 (probe 53). Composition tested properly and it does not happen:
held-out pairings sit at chance, indistinguishable from a random bank.**
F32's "blocked" verdict was premature — the compose suite had been run
without a screen view, and with the roll view both seeds acquire (mean
0.42 and 0.57 against solo ceilings of ~0.44-1.00, all six pairings
above chance on seed 69317). With training working, the composition
readout is finally meaningful, and it is negative:

| held-out pairing | composed (69316 / 69317) | random-bank control |
| --- | --- | --- |
| c02 | 0.316 / 0.357 | 0.350 / 0.350 |
| c11 | 0.378 / 0.351 | 0.332 / 0.364 |
| c20 | 0.340 / 0.410 | 0.333 / 0.337 |

Chance is 0.333. Every composed score lands between 0.32 and 0.41, and
the random-bank control lands in the same band: handing the plant the
two ideal already-trained fragments is worth no more than handing it
noise. Six seed-pairing measurements, no separation.

This is the cleanest negative the composition line has produced, and it
retires the pressure hypothesis (F27/F32): making factorisation the
economical solution — nine pairings over six rules, each rule exercised
in at least two contexts, all six training pairings acquired — does not
make it the learned solution. Together with F16 (imposed sharing does
not compose) and F27 (partner rotation buys interchangeability, not
composability), three independent mechanisms have now failed, and the
common factor is that none of them ever made composition itself the
thing being optimised.

What remains, stated as the sharpest form of the open problem:
concatenating two fragments into the event window is not an operation
the plant was ever trained to perform. Convergent finding 2 said trained
combiners beat hoped-for arithmetic; every attempt so far has hoped. The
remaining admissible design is an explicit combiner — a trained
operation over fetched fragments, optimised on held-out pairings — which
is a change to what the controller DOES with the bank, not to what the
bank stores. That is the next architectural rung, and it is a large one.

**F34 (probe 54). The trained combiner does not compose either — and it
costs acquisition. The composition line is exhausted at this scale.**
The remaining admissible design from F33: one shared learned function
from a fetched fragment SET to the controller's context tokens, so a
novel pairing is merely another application of a familiar operation.
Measured against concatenation on the compose suite, both seeds:

| | training mean | c02 | c11 | c20 |
| --- | ---: | --- | --- | --- |
| concat 69316 | 0.42 | 0.316 v 0.350 | 0.378 v 0.332 | 0.340 v 0.333 |
| concat 69317 | 0.57 | 0.357 v 0.350 | 0.351 v 0.364 | 0.410 v 0.337 |
| combiner 69316 | 0.33 | 0.340 v 0.348 | 0.327 v 0.332 | 0.332 v 0.337 |
| combiner 69317 | 0.41 | 0.341 v 0.323 | 0.393 v 0.366 | 0.322 v 0.338 |

(composed vs random-bank control; chance 0.333.) Twelve held-out
measurements across two mechanisms and two seeds, every one at chance
and every one matching its control. The combiner additionally LOWERS
training (0.42 -> 0.33, 0.57 -> 0.41): pooling the fetched set through a
bottleneck discards information the raw concatenation preserved, and
buys nothing back.

Four mechanisms have now failed — imposed sharing (F16), partner
rotation (F27), economic pressure (F33), trained combiner (F34) — and
the honest conclusion is structural rather than a fifth mechanism.
Composition requires the composed pairing to be *supervised* somewhere:
every design here trained only on seen pairings and hoped an unseen one
would fall out, including the combiner, which learned a function fitted
to six pairings with no term rewarding generalisation to a seventh. The
literature's protocol (MLC) supplies exactly that missing term through
in-context study examples and query episodes, and this architecture has
no study-phase mechanism: the event window holds fetched fragments, not
worked examples.

Standing conclusion for the roadmap. Compounding through composition is
NOT available from the current architecture, and no amount of bank-side
engineering will supply it. It requires either (a) a study-phase channel
so query episodes can be posed and scored during training, or (b)
abandoning compositional transfer as the compounding mechanism in favour
of the one the program HAS demonstrated: super-solo transfer under
staggered admission (F18, 3/3 seeds), where earlier games measurably
speed later ones without any fragment recombination. (b) is the honest
current claim; (a) is the next architecture, not the next probe.

**F35 (probes 55-57). Acquisition does not respond to standard RL
tuning, and it gets WORSE with capacity — the bottleneck is the
optimization landscape, not variance or expressiveness.** Four
independent interventions on the motor games that gate the program
(F25), all at matched budget, seed 69316 unless noted:

| intervention | forageA | intercept1 | collect1 | c01 |
| --- | ---: | ---: | ---: | ---: |
| baseline (h=32) | **0.453** | **0.312** | **0.547** | **1.000** |
| per-timestep baseline | 0.484 | 0.062 | — | 1.000 |
| + entropy bonus 0.01 | 0.406 | 0.188 | — | 1.000 |
| normalized advantage | 0.047 / 0.203 | 0.109 / 0.172 | 0.719 / 0.078 | 0.990 |
| hidden 64 | 0.031 | 0.078 | — | — |
| hidden 128 | 0.031 | 0.031 | — | — |

Nothing helps and most things hurt. Advantage normalisation is the
clearest failure (forage 0.45 -> 0.05): dividing by the deviation
amplifies pure noise into full-size gradients while the policy is still
near-random, which is exactly when these games have no signal yet.
Capacity is the most informative failure: quadrupling the controller
collapses both motor games, so the constraint is not expressiveness. A
bigger recurrent policy trained by REINFORCE from scalar outcomes simply
has a harder landscape to descend.

Taken with the earlier BPTT diagnostic (truncation is not the cause) and
F22/F28 (encoder structure moves these games more than any trainer
change), the acquisition constraint is now localised: it is the
recurrent policy-gradient landscape itself, and it is not reachable by
the standard levers — baselines, entropy, normalisation, or width. The
levers that HAVE moved it are all representational (egocentric roll
+0.42 on forage, crop +0.19 on navigate).

Consequence for the roadmap: further acquisition work should be
representational or algorithmic (a critic/actor-critic, or supervised
bootstrapping from a scripted policy), not another variance trick. All
four options remain in the code behind flags, defaulting off, with this
table as the reason.
