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

**F36 (probe 58). A learned critic is the first intervention to improve
motor acquisition on both seeds — partially.** F35 exhausted the
state-INDEPENDENT levers; the one remaining variance reduction was a
proper critic, since a scalar or per-timestep baseline cannot separate
"this state was bad" from "that action was bad". A `ValueHead` over the
controller's opaque intention (amodal side of the boundary, shared
infrastructure, training-time only, emits no actions) gives:

| game | baseline | critic 69316 | critic 69317 |
| --- | ---: | ---: | ---: |
| collect1 | 0.547 | **0.812** | **0.859** |
| intercept1 | 0.312 | 0.453 | 0.141 |
| forageA | 0.453 | 0.469 | 0.281 |

collect improves ~55% on BOTH seeds — the first robust acquisition gain
in the program, and the first intervention of six that is not uniformly
negative. forage and intercept stay a seed lottery, so the constraint is
narrowed rather than removed: credit assignment was genuinely part of
the problem for the game with the longest reward horizon (collect
requires reaching a distant object repeatedly), and is not the whole
problem for games that also demand timing (intercept) or discrimination
under a moving reward source (forage).

This also retires the F35 reading in one respect: acquisition is not
wholly beyond standard machinery, it was beyond the *state-independent*
subset of it. The remaining gap is concentrated in games whose failure
mode is exploratory rather than credit-assignment-shaped, which points
at the untried lever — supervised bootstrapping from a scripted policy,
where exploration is supplied rather than discovered.

**F37 (probe 59). Novelty-driven exploration fails here because novelty
and engagement are in direct conflict.** Random network distillation was
the admissible exploration lever for the games the critic did not fix
(F36): intrinsic reward from the agent's own observation stream, no game
rules injected, bonus shaping learning only while mastery stays scored
on the verifier's reward. Measured with the critic already in place:

| game | baseline | critic | critic + RND |
| --- | ---: | ---: | ---: |
| forageA | 0.453 | 0.469 / 0.281 | 0.500 / 0.109 |
| intercept1 | 0.312 | 0.453 / 0.141 | **0.000** / 0.141 |

No gain, and intercept collapses to zero on one seed. The mechanism is
specific and was foreseeable from this program's own findings: in these
worlds the most novel observations are produced by MOVING — the avatar
plane changes every step — while reward requires COMMITTING to an item
and resolving a trial. Novelty therefore pays exactly the wandering that
F15 and F20 identified as the engagement-collapse failure mode. The
bonus does not supply the missing first success; it subsidises the
behaviour that prevents it.

General law: an intrinsic objective must be checked against the task's
known failure mode, because a generic curiosity signal can be
anti-correlated with engagement in worlds where the agent's own motion
dominates observation change. Undirected novelty is the wrong exploration
prior for trial-structured tasks; what they need is novelty over
OUTCOMES (which trials have been resolved, and how) rather than over
observations — a bonus this architecture cannot currently compute,
because the learner never sees trial structure.

Acquisition status after seven interventions: credit assignment is
partly solved (critic, F36), exploration is not, and the two admissible
routes left are both out of reach without new machinery — outcome-space
novelty needs verifier structure the learner is denied, and scripted
bootstrapping injects the rules the discipline withholds. This is a
genuine architectural limit, not a tuning gap.

**F38 (probe 60). Outcome novelty WAS computable — F37 was wrong about
that — and it fails too, completing a unifying account of acquisition.**
F37 claimed novelty over outcomes needed verifier structure the learner
is denied. That was an error: the reward scalar is already the learner's
legitimate input, so scarcity over the reward stream is computable from
it alone. `OutcomeNovelty` does exactly that — count-based bonus over
reward buckets, signed by the reward so a rare SUCCESS is amplified and
a rare failure is not, counts persisting across training so scarcity is
measured over the run. It behaves as designed (63x weight on a rare +1
versus a common 0) and it does not work:

| game | baseline | critic | critic + outcome novelty |
| --- | ---: | ---: | ---: |
| forageA | 0.453 | 0.469 / 0.281 | 0.375 / 0.031 |
| intercept1 | 0.312 | 0.453 / 0.141 | 0.031 / 0.203 |

Eight interventions now have a single consistent explanation. Sort them
by their effect on gradient VARIANCE:

* variance-reducing: learned critic — the only win (collect 0.55 ->
  0.81/0.86, both seeds).
* variance-neutral: per-timestep baseline, entropy — no effect.
* variance-increasing: advantage normalisation (amplifies noise while
  the policy is random), RND (amplifies a signal anti-correlated with
  engagement), outcome novelty (amplifies rare events into gradient
  spikes), extra width (a harder landscape) — all harmful, and the more
  aggressively they amplify, the worse.

The law: on this plant, acquisition is variance-limited, and any
intervention that adds gradient variance loses more than its signal
gains — including interventions whose signal is exactly the right one.
Outcome novelty pays the correct event and still fails, because paying
it as a large rare bonus is itself the problem. The admissible direction
is therefore not a better exploration BONUS but a lower-variance
estimator: the critic generalised (per-action baselines, advantage
estimation over multiple steps), or off-policy reuse of the rare
successes once they occur, which extracts more signal per success
instead of shouting louder about it.

**F39 (probes 61-62). Generalised advantage estimation does not rescue
the exploration-limited games, and the variance account needs
qualifying: the critic's win is a CREDIT-ASSIGNMENT win, not a variance
win.** F38 predicted that lower-variance estimators were the admissible
direction. GAE is the standard one, and a lambda sweep with the critic
in place gives:

| game | critic (MC) | GAE .95 | GAE .5 | GAE 0 |
| --- | --- | --- | --- | --- |
| forageA | 0.47 / 0.28 | 0.41 / 0.05 | 0.45 / 0.31 | 0.12 / 0.17 |
| intercept1 | 0.45 / 0.14 | 0.02 / 0.16 | 0.08 / 0.12 | 0.02 / 0.39 |
| collect1 | 0.81 / 0.86 | 0.70 / 0.88 | — | — |

Lowering lambda monotonically reduces estimator variance, so F38's law
predicts monotone improvement. It does not appear: lambda 0.5 roughly
matches Monte Carlo, and lambda 0 (maximum variance reduction, maximum
bias) is the worst setting on both games. The bias introduced by
bootstrapping through a critic that is itself poorly fit on a task the
policy cannot yet perform costs more than the variance it removes —
a critic can only explain away states it has seen succeed.

This qualifies F38 rather than overturning it. The correct statement is
narrower: the critic helped `collect` because collect's failure was
long-horizon CREDIT ASSIGNMENT, and a state-dependent baseline addresses
that directly. It was never a general variance result, and the eight-way
sort by variance was an over-generalisation from one win. Variance
reduction per se buys nothing on forage and intercept, whose failure is
that the policy rarely produces a success at all — and no estimator,
however well conditioned, can lower the variance of a signal that has
not occurred.

Standing conclusion for acquisition, after ten interventions. There are
two distinct failures wearing one name. Credit assignment is solved
where the reward is dense enough to fit a critic (collect: 0.55 ->
0.81/0.86, both seeds, shipped). Sparse-first-success exploration is
unsolved and is not reachable from the estimator side at all; it needs
either a source of successful trajectories the agent did not have to
discover (inadmissible here) or an environment whose first success is
not rare (a curriculum, which F20/F21 showed must preserve the property
that makes the anchor learnable). The honest recommendation is the
second: build the curriculum, and stop paying for probes on the
estimator side.

**F40 (probe 63). A density curriculum solves sparse-first-success where
errors are survivable — and the game it fails on completes the design
law.** F39 established that exploration is unreachable from the
estimator side and that the remedy must be an environment whose first
success is common, subject to F20/F21 (the learner must perform the
target act; nothing may be granted on its behalf). Item density is that
knob: three pairs on the grid make blundering into one likely, one pair
is the target task, and every stage requires the identical act.
Scored always on the TARGET task (level 1), never on the easy stage:

| game | cold (critic) | after L3 | after L2 | after L1 |
| --- | --- | ---: | ---: | ---: |
| forage 69316 | 0.469 | 0.422 | 0.406 | **0.562** |
| forage 69317 | 0.281 | 0.469 | 0.516 | **0.594** |
| intercept 69316 | 0.453 | 0.016 | 0.031 | 0.016 |
| intercept 69317 | 0.141 | 0.000 | 0.219 | 0.469 |

Forage improves on both seeds and — the part that matters — the SEED
SPREAD COLLAPSES, from 0.469/0.281 to 0.562/0.594. The seed lottery that
has gated this program since F25 is, for this game, gone: the
curriculum does not merely raise the mean, it removes the dependence on
getting lucky early. This is the first solution to sparse-first-success
in the program.

Intercept gets worse, and the reason completes the law. Three
simultaneous fallers give three chances to catch AND three chances to
miss — and a miss is fatal (-1 and episode end), while a wrong forage
item is survivable (-1, keep playing). Density therefore raises success
opportunity and failure opportunity together wherever errors are fatal,
and the easy stage is harder than the target.

Design law, joining F6 (survivable error) to F20/F21 (curricula must
preserve what makes the anchor learnable): **a curriculum stage may
multiply opportunities only where errors are survivable; where failure
is terminal, density is not an easing knob but a difficulty knob.** For
fatal-error games the admissible easing axis is the one that lengthens
the time to commit a fatal mistake — slower dynamics, larger margins —
not more objects. Intercept's curriculum should vary faller SPEED, not
faller count, and that is the next probe rather than a rebuttal.

**F41 (probe 64). The speed curriculum fails, falsifying F40's
prediction — and the reason unifies it with F21 into a stronger law.**
F40 predicted that fatal-error games need an easing axis that lengthens
the time to commit a fatal mistake, and named faller speed. Built
(`faller_period`, verified by test to slow descent without adding
danger) and run on intercept, scored always on the target (period 1):

| seed | after p4 | after p2 | after p1 | cold critic | density curriculum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 69316 | 0.000 | 0.000 | 0.094 | 0.453 | 0.016 |
| 69317 | 0.219 | 0.125 | 0.469 | 0.141 | 0.469 |

Worse than cold on one seed, identical on the other. The prediction is
falsified, and the diagnostic detail is in the intermediate column:
after training at the SLOWEST setting, target mastery is 0.000 and
0.219 — the easy stage transfers nothing.

That is the F21 failure in new clothing. Slowing the dynamics does not
make the same task easier, it makes a DIFFERENT task: at period 4 the
agent can watch a faller descend and stroll under it; at period 1 it
must already be positioned when the faller appears. Those are different
policies, so the curriculum teaches a skill the target does not want,
exactly as the teleport bridge taught waiting instead of navigating.

Unified law, replacing F40's second clause and subsuming F20/F21:
**an easing axis must preserve the POLICY the target task requires, not
merely its reward rate.** Density passes (approach-and-choose is the
same act at any density; only the encounter rate changes). Speed fails
(leisurely positioning is not anticipation). Spawn distance failed for
the same reason, and so did teleport-forcing. The test for a proposed
curriculum is therefore not "is the easy stage easier?" but "would a
policy that is optimal at the easy stage still be optimal at the
target?" — and it can be checked cheaply, before spending a run, by
asking whether the optimal action at each stage is the same function of
what the agent can see.

Acquisition final status: credit assignment solved where reward is
dense (critic, F36); sparse-first-success solved where errors are
survivable AND a policy-preserving easing axis exists (density, F40);
unsolved for fatal-error games, where no policy-preserving easing axis
has been found and the two tried (count, speed) fail for opposite
reasons.

**F42 (probe 65). The spread curriculum fails too — and it passed F41's
pre-flight test, so the test itself was too weak.** F41 required an
easing axis to preserve the policy the target needs. Faller spawn spread
was designed to pass exactly that: "move toward the faller's column" is
the same function of the same observation at every spread, differing
only in the distance it must be applied. It fails:

| axis | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| cold (critic, no curriculum) | **0.453** | 0.141 |
| count (density) | 0.016 | **0.469** |
| speed (period) | 0.094 | **0.469** |
| spread (distance) | 0.188 | 0.219 |

Three axes, none better than cold on both seeds. The diagnosis for
spread is specific and it corrects F41: at spread 1 the faller lands
within one column of the avatar, so the optimal action is almost always
"stay". The POLICY is unchanged in form — move toward the column — but
its typical OUTPUT is constant, so what the learner actually fits is the
constant. The easy stage was solvable without ever exercising the
behaviour the target needs.

Strengthened law (final form): **an easing axis must preserve the target
policy AND exercise the full range of its outputs.** Checking the first
alone admits degenerate stages where the policy is technically the same
function but is only ever evaluated at one point. All four curriculum
successes and failures in this program now follow:

* density on forage — same act, full range of approach distances still
  required. WORKS (F40, seed lottery removed).
* spawn radius on forage — collapses the approach distance to ~0. FAILS
  (F20).
* teleport forcing — supplies the return the agent should perform.
  FAILS (F21).
* faller speed — leisurely positioning is a different policy. FAILS (F41).
* faller spread — same policy, degenerate output range. FAILS (F42).

Intercept remains unsolved, and honestly so: no axis is known that eases
a fatal-error, timing-critical task while exercising the full range of
the anticipatory policy it requires. The two properties are in tension
here — anything that gives the agent more slack removes the need to
anticipate, which is the skill. That tension may be intrinsic to
timing-critical tasks rather than a gap in the search, and saying so is
more useful than a sixth axis.

**F43 (probe 66). Self-addressing from feedback fails by router collapse
— and the failure is F12's, recurring at the level the oracle was
hiding.** Weakness 12: every bank rung addressed fragments by a
per-context label the deployed system would not have. `ContentRouter`
closes that in the only way the twins permit — the worlds render
identically, so the context is knowable only from the CONSEQUENCES of
acting. Protocol: probe the world briefly with no fragments, read the
controller's own intention (which has integrated the reward feedback),
query the router, fetch discretely, then commit. Nothing outside the
agent names the context at any point.

| seed | selection A / B | mastery A / B | cross-fed |
| --- | --- | ---: | ---: |
| 69316 | [5,0] / **[5,0]** | 0.164 / 0.250 | 0.281 / 0.094 |
| 69317 | [1,5] / **[1,5]** | 0.203 / 0.133 | 0.281 / 0.156 |

The router selects the SAME fragments for both twins on both seeds. It
has collapsed, so the bank is fragment-blind, so neither twin learns and
cross-feeding is meaningless. This is exactly F12 — the selector
collapse that the oracle-to-learned handover (F13) was built to cure —
returning once the oracle is removed at the addressing level rather than
the assignment level.

The cure does not transfer, and the reason is instructive. F13 worked by
holding the ASSIGNMENT fixed while the read path formed, then imitating
it. Here there is no assignment to hold: the query is computed from a
recurrent state that is itself still learning what the feedback means.
Two things must be true before a feedback-derived query can be
informative — the controller must encode "which world am I in" in its
intention, and the router must map that encoding to fragments — and
neither has a gradient until the other works. It is F5's cold-start
deadlock at one remove, with the plant's own representation as the thing
that must be scaffolded.

Consequence: self-addressing needs its own staging, and the staged
quantity is not the fragment assignment but the CONTEXT ENCODING. The
admissible design is a probe phase trained to predict something the
agent already receives — the sign of its own next reward under a fixed
test action — which supplies a dense supervised signal for "which world
is this" without naming the world. That is buildable within the
discipline and is the concrete next rung; the router itself is sound and
stays in the code.

**F44 (probes 67-69). Self-addressing splits cleanly into two problems:
the ENCODING is solved, the ROUTING is not — and the split is the
result.** F43 left both entangled. Three probes separate them.

*The encoding was absent, then solved.* A probe reading the controller's
state at step ZERO predicts identically for both twins (0.39/0.39,
0.70/0.70): the worlds render alike and no feedback has arrived, so the
context is not hard to encode there, it is ABSENT, and the probe can
only learn the mean. Reading the state after one test action and its
outcome fixes it completely — the probe then predicts **+1.00 for
choiceA and -1.00 for choiceB on both seeds**. The agent's own state
carries the context exactly, obtained from nothing but acting and
observing the consequence.

Standing architectural statement: **addressing for observationally
identical contexts is necessarily ACT-THEN-FETCH.** A memory system that
must fetch before acting cannot serve contexts that look alike, however
good its router. The event window fetch must therefore be permitted to
happen mid-episode, which the architecture already allows and no rung
had previously required.

*The routing did not follow.* With a demonstrably perfect encoding
available, the router still collapses to the same fragments for both
twins (seed 69316 [4,3]/[4,3], seed 69317 [0,5]/[0,5]), and adding the
F13 diversity pressure only partially separates it (69316 [4,3]/[1,3] —
one fragment differs; 69317 still identical), with mastery at chance in
every condition. So the residual failure is isolated: not the plant, not
the encoding, not the bank, but the discrete selector's own learning
signal, which is a single scalar per rollout arriving through REINFORCE
while the read path has not yet formed.

That is exactly the F5/F12 deadlock, and F13's cure applies but cannot
be run here as written: it staged the ASSIGNMENT with an oracle. The
equivalent for a content router is to stage the MAPPING — supervise the
router to reproduce a known-good assignment while the read path forms,
then release it, exactly as F13 did one level down. Since the encoding
is now solved and readable at +-1.00, that supervision is available
without an oracle over contexts: the probe's own prediction sign is a
legitimate training target for which fragment to fetch. That is the next
rung, and unlike the previous three it has no missing ingredient.

**F45 (probe 70). Mapping handover does not rescue the content router
either: three staged designs, three collapses. The learned router is
this program's confirmed universal failure point.** F44 specified the
rung with no missing ingredient — stage the MAPPING as F13 staged the
assignment, supervised by the probe's own sign (positive means "this
world rewards the positive-plane item"), with two disjoint fragment sets
and nothing naming which set means which world. Built, run with 900
staging updates plus diversity pressure, both seeds:

| seed | probe | selection A / B | mastery A / B |
| --- | --- | --- | ---: |
| 69316 | +1.00 / -1.00 | [5,0] / **[5,0]** | 0.188 / 0.156 |
| 69317 | +1.00 / -1.00 | [2,5] / **[2,5]** | 0.234 / 0.258 |

The encoding is perfect, the staging assigns different fragments per
sign during training, the imitation term is present, diversity pressure
is on — and after release the router still returns one selection for
both worlds. Three designs now collapse identically: outcome-REINFORCE
alone (F43), plus diversity (F44), plus staged mapping and imitation
(F45).

This confirms, in our own system, the sharpest warning of the opening
literature review (convergent finding 3): *learned soft routers are the
universal failure point*. The review's recommended alternative was a
selector trained apart from the repertoire on outcomes — which is what
the promoted per-context selector IS, and it works (F13, 1.000/1.000).
The difference is not the training signal but what the selector is
indexed BY: a per-context table has one parameter set per context and
cannot merge them, while a content router computes selections through a
shared query map that can, and under a single scalar reward per rollout
it always does.

Consequence, and the honest architectural position: content-addressing
over a shared query map is not reachable with this learning signal. The
admissible routes are (a) keep per-context tables and accept that the
context index must come from somewhere — which the F44 probe now
supplies from the agent's own measurement rather than an oracle, making
the table legitimately self-indexed; or (b) give the router a denser
signal than one scalar per rollout, which means supervising fetch
decisions directly against measured per-fragment outcomes, i.e. the swap
test (F23's conflict machinery) repurposed as router supervision. (a) is
available now and is the smaller claim; (b) is the next real rung.

**F46 (probes 71-72). The addressing phase was destroying the encoding
it depends on. Freezing the plant fixes the addressing and exposes the
real residue — the read path was never trained to use a probe-indexed
fetch.** A bug, and it retroactively weakens F43-F45.

The encoding probe reads +1.00 / -1.00 for the twins BEFORE the
addressing phase and collapses to one sign AFTER it, on both seeds. The
controller sat in that phase's optimiser, trained by an RL objective
with no reason to preserve a context signal, so every router in F43-F45
was keyed on a quantity decaying to a constant underneath it. A router
keyed on a constant must collapse; those three collapses are therefore
not evidence about routers. The claim "the learned router is this
program's universal failure point" (F45) is withdrawn as unsupported by
its own experiment — it may still be true, but nothing here shows it.

With the plant frozen for the addressing phase, the encoding survives
and addressing WORKS: `probe_sign` is 1 for choiceA and 0 for choiceB on
both seeds, and the table returns genuinely different fragments per
world ([3,2] vs [1,5]; [5,2] vs [3,0]). The agent identifies which of
two identical-looking worlds it is in, from nothing but acting and
observing its own reward, and fetches accordingly. No oracle, no context
label. That is weakness 12's actual requirement, met.

Mastery does not follow (0.21/0.24 and 0.17/0.31, at chance), and the
reason is now clean rather than confounded: the read path was formed in
an earlier regime and the plant is frozen, so nothing ever learned to
EXECUTE the fetched fragments under probe-indexed addressing. The three
pieces each work in isolation — encoding (+-1.00), addressing (correct
per world), read path (F13, 1.000/1.000) — and have never been trained
together, because freezing to protect the encoding also forbids the read
path from adapting.

That is precisely what the promoted consolidation line exists for, and
the next rung is now specific: run the addressing phase with the plant
PLASTIC but under arbitrated consolidation anchored on the encoding's
Fisher, so the context signal is protected while the read path adapts.
Freezing was the diagnostic; consolidation is the architecture-true fix,
and it is a mechanism this program already promoted on 5/5 seeds.

Standing methodological rule, learned the hard way twice (cf. F31):
a quantity measured before a training phase is not evidence about its
value after that phase. Any claim resting on a measured signal must
re-measure it at the point of use.

**F47 (probe 73). Encoding-anchored consolidation preserves the
addressing and does not restore mastery: the self-addressing loop has
three working parts and one that will not transfer.** Running the
addressing phase with the plant plastic under an arbitrated penalty
anchored on the encoding's own Fisher, both seeds:

| seed | probe signs A/B | fragments A/B | mastery A/B | frozen-plant control |
| --- | --- | --- | --- | --- |
| 69316 | 1 / 0 | [3,2] / [1,5] | 0.211 / 0.547 | 0.211 / 0.242 |
| 69317 | 1 / 0 | [5,2] / [3,0] | 0.172 / 0.250 | 0.172 / 0.305 |

The consolidation does its job: the encoding survives a plastic phase
(+-1.00 retained, signs still split) where an unprotected phase destroyed
it (F46). That is the promoted EWC line generalising to a competence it
was never built for — protecting a context ENCODING rather than a game
skill — and it is the fourth setting in which that mechanism has held.

Mastery does not follow. One cell moved (choiceB 0.242 -> 0.547 on seed
69316) and nothing else did; seed 69317 is flat or slightly worse. With
addressing correct and the encoding intact, the residue is isolated to
the last place it can be: the read path was formed under ORACLE
addressing, where the fetched fragments are fixed per context from
update one, and it does not transfer to probe addressing, where the
fetch depends on a measurement the agent makes afresh each episode.

That is a real and specific finding, not a tuning gap. Under oracle
addressing the plant may condition on a constant; under probe addressing
it must condition on the CONSEQUENCES of its own probing action, which
is a different computation over a different input. Staging cannot bridge
them because each stage teaches the wrong one: this is F41's law —
an easing stage must exercise the policy the target requires — applied
to addressing rather than to a game.

Consequence, and the honest end state of this line: the three parts must
be CO-TRAINED from the start (probe, fetch, and execute in one loop with
one objective), not assembled from separately validated stages. Every
staged design in this program worked when the stages shared a policy
(F13, F18, F40) and failed when they did not (F20, F21, F41, F47). That
is now a five-times-confirmed law and it predicts the co-training
requirement rather than merely permitting it.

**F48 (probes 74-76). Co-trained self-addressing masters both twins,
inverts under cross-feed, and FAILS the decoy gate: the skill is in the
recurrent state, not the bank.** The co-trained loop (probe, fetch,
execute under one objective from update zero) was F47's prediction and it
does close on mastery. The full gate set, seed 69317:

| gate | choiceA | choiceB | verdict |
| --- | ---: | ---: | --- |
| own fragments | 1.000 | 1.000 | pass |
| cross-fed (other twin's) | 0.094 | 0.000 | pass — inverts |
| **norm-matched decoy** | **0.812** | **0.781** | **FAIL** |

Cross-feeding inverts behaviour, so a fragment can still override the
policy. But replacing the fetched fragment with noise of matched norm
costs almost nothing: the agent scores 0.78-0.81 without any real
content. The bank is therefore SUFFICIENT to override and NOT NECESSARY
to perform — and the two together identify the mechanism exactly. The
co-trained agent probes, holds the outcome of its own probing action in
the controller's recurrent state, and plays the rest of the episode from
that. The fetched fragment is decorative; a contradictory one still
misleads, which is why cross-feed inverts and decoy does not.

Seed 69316 is the same finding without the ambiguity. There the loop
hits winner-take-all (0.961 / 0.008), and for the twin that DID learn,
neither audit bites: `choiceA` scores 0.906 on the other twin's
fragments and 0.969 on pure noise. It is not merely that the content is
unnecessary — the bank is ignored outright, and the cross-feed signature
that looked decisive on seed 69317 does not even appear here. Across
both seeds the decoy gate fails; the cross-feed gate passes on one and
fails on the other, which is exactly the pattern expected when a
recurrent state is doing the work and a fetched vector sometimes
happens to perturb it.

This is the architecture's storage rule violated in a new place. F30
caught skill migrating into per-game ENCODER weights; here it migrates
into WORKING MEMORY within an episode. Both were found by an audit that
mastery alone could not have distinguished, and both produced the
program's best numbers on the way to being rejected.

Two consequences. (1) The gate set is incomplete without the decoy: for
several rungs the cross-feed signature alone was treated as sufficient
evidence of specification, and this shows it is not — a cue that
overrides can coexist with content that is unnecessary. Every prior
promotion resting on cross-feed alone should be re-audited with a
norm-matched decoy. (2) The co-trained loop omits the ignorance
objective (withheld and decoy rollouts pushed toward uniform) that the
promoted externalization line uses precisely to make the bank necessary.
Adding it is the obvious next run, and its absence — not the addressing
design — is what this failure is about.

Standing correction to the addressing claim: the loop demonstrates that
an agent can identify which of two identical-looking worlds it is in
from its own actions and fetch accordingly (signs split, selections
distinct, both seeds). It does NOT yet demonstrate that the fetched
content carries the skill, which is the whole point of a memory bank.

**F49 (probes 77-78). The consolidation anchor survives mastery, but its
reliability is gated on policy entropy, not on mastery.** A literature
sweep (`docs/LITERATURE_MAP.md`) surfaced a predicted failure in shipped
promoted code: our diagonal Fisher is built from score-function gradients
of sampled actions, which vanish as a policy saturates. Recent EWC
analyses report that this under-estimates importance for exactly the
best-learned skills. We normalise each game's Fisher to unit mean, which
divides out the magnitude collapse — but normalisation cannot restore
signal-to-noise, so the sharper worry is that a mastered game's anchor is
noise rescaled to look confident, protecting arbitrary directions.

Tested directly: train choiceA, and at checkpoints estimate the Fisher
TWICE from independent rollout seeds, then ask whether the two agree.
Pearson over the full per-parameter vector, plus overlap of the top 1%
(the parameters the penalty actually acts on), against an entropy
estimate from the same policy.

| pooled, both seeds | n | correlation min/mean/max |
| --- | ---: | --- |
| pre-mastery | 4 | 0.671 / 0.793 / 0.913 |
| mastered | 14 | 0.091 / 0.796 / 0.990 |
| mastered, entropy < 0.01 | 4 | 0.091 / 0.612 / 0.920 |
| mastered, entropy >= 0.01 | 10 | 0.489 / 0.870 / 0.990 |

**The simple prediction is false.** Mean agreement after mastery (0.796)
is identical to before it (0.793); mastery per se does not degrade the
estimator. What changes is the VARIANCE: the pre-mastery range spans
0.24, the mastered range spans 0.90. The estimator does not decay, it
becomes unreliable — sometimes excellent (0.990), occasionally
near-worthless (0.091).

The controlling variable is entropy, not mastery. Every bad checkpoint
sits at entropy < 0.01, and the two seeds separate on exactly this:
69316 collapses to entropy 0.001 at three checkpoints and produces the
0.091, 0.489 and 0.647 correlations; 69317 never falls below 0.003, sits
mostly at 0.02-0.10, and returns 0.798-0.985 throughout. Whether the
anchor is trustworthy is therefore a seed-dependent property of how far
the policy saturates, which no check on the penalty's magnitude could
detect.

**But the anchor is not noise, and the promoted rule stands.** The top-1%
overlap never falls below 0.393 against a chance baseline of 0.01 — even
at the worst checkpoint the most-protected parameters agree ~39x better
than chance. Full-vector Pearson is the harsher measure here because it
is dominated by the near-zero bulk; the penalty is carried by the head of
the distribution, which holds up. F47's encoding-anchored consolidation
and the arbitrated rule are not invalidated by this.

**The entropy floor works, but as a variance reducer, not an improver
(probes 79-80).** `--fisher-temperature 2.0` tempers the sampling policy
for the Fisher pass only. Control first: entropy and mastery trajectories
are bit-for-bit identical to the untempered runs on both seeds, so the
knob provably did not leak into training.

| mastered checkpoints, both seeds | n | correlation min/mean/max | top-1% min/mean |
| --- | ---: | --- | --- |
| untempered | 14 | 0.091 / 0.796 / 0.990 | 0.393 / 0.718 |
| tempered (T=2.0) | 14 | 0.652 / 0.790 / 0.964 | 0.615 / 0.700 |

**The mean is unchanged (0.796 -> 0.790). What collapses is the spread:
[0.091, 0.990] becomes [0.652, 0.964].** It buys a floor by giving up the
ceiling, and the two seeds show each side of that trade separately. On
69316, the seed whose entropy collapsed, every bad checkpoint is repaired
(worst 0.091 -> 0.652, mean 0.671 -> 0.814). On 69317, which was already
healthy, agreement *falls* at nearly every checkpoint (0.985 -> 0.730,
0.977 -> 0.855, 0.945 -> 0.662). Tempering costs real precision when the
untempered estimate was fine — it is smoothing, and smoothing a good
estimate makes it worse.

So the honest reading is: this removes the catastrophic tail, does not
improve the typical case, and is a net win only if the low-correlation
events actually cause retention failures. **That is not yet measured, so
the flag stays off by default.** The promotion bar is a matched battery
comparison — retention with and without tempering, two seeds — and until
that runs, turning it on would be trading a measured cost for a
hypothesised benefit.

Two further consequences. (1) The variance itself is the finding worth
carrying: our anchor's reliability is a seed-dependent property of how
far the policy saturates, and no check on the penalty's magnitude can
see it. Estimator health belongs in the battery report alongside the
retention numbers. (2) Methodological: this is the second time a quantity
we relied on was measured in a regime where it was not valid (F46 was the
first — re-measure any signal at the point of use). The rule generalises:
a statistic estimated from a policy's own samples inherits that policy's
degeneracy, so estimator health must be reported alongside the estimate.

Recorded as a qualification, not a rejection. Probes 77-80 are
`fisher_stability.py`, both seeds, choiceA, 600 updates.

**F50 (probes 81-84). Disjoint oracle fragments do NOT resolve the twin
contradiction. The promoted anti-collapse machinery is what does.** The
routing literature (`docs/LITERATURE_MAP.md` S4) says to check the
cheapest thing first: whether twin winner-take-all is genuine gradient
conflict at all. PCGrad's tragic triad holds that joint training is
harmed when task gradients CONFLICT (negative cosine) and DOMINATE
(imbalanced norms); if the cosine is positive, no routing surgery helps
and the failure is acquisition instead. Measured directly: per
checkpoint, take each twin's REINFORCE gradient on the shared plant
separately and report their cosine, under bare alternating training with
no diversity penalty, no laggard balancing, and no handover.

| pooled, both seeds, 40 checkpoints | cosine mean | negative | best-twin mastery |
| --- | ---: | ---: | ---: |
| no bank | -0.113 | 24/40 | 0.341 |
| each twin's own oracle fragments | -0.134 | 27/40 | 0.478 |

**The conflict is real and fragments do not remove it** — with private
fragments the cosine is if anything slightly more negative. What
fragments change is the outcome: bank-free, both seeds end at chance
(0.250/0.250 and 0.250/0.312), which is correct and not a bug, since
twins render identically and demand opposite actions, so a plant with no
context signal cannot beat chance. With fragments, seed 69317 ends at
**0.062 / 1.000** — textbook winner-take-all, one twin taking the plant
outright and the other pushed below chance — while 69316 ends at
0.250/0.375, neither learned.

So private context breaks the symmetry enough for a winner to emerge but
not enough for both twins to coexist. That is exactly the gap the
promoted rung fills: the diversity penalty, laggard-preferential
balancing and oracle-to-learned handover are not incidental tuning, they
are what converts "one twin wins" into 1.000/1.000 on both seeds. This
is the first measurement isolating what that machinery buys, because it
holds the read path fixed (oracle fragments, disjoint by construction)
and removes every anti-collapse mechanism at once.

Two consequences. (1) The routing/balancing line is on-topic — an
earlier reading of a truncated 50-update run suggested a positive cosine
and was wrong; the full runs invert it. Truncated trends are not
evidence, which is F32's rule (calibrate the workload that will run) in
a new place. (2) Gradient surgery specifically is still the wrong tool
here, and the negative cosine does not recommend it: PCGrad and CAGrad
seek a compromise direction, but for contexts that contradict on
identical observations no compromise exists — the compromise IS chance,
which is precisely the bank-free result above. Contradiction is resolved
by context, not by averaging. The admissible readings of the literature
for us are therefore the structural ones (shared+private partition,
orthogonality on fragment CONTENT rather than on selections), not the
gradient-space ones.

Probes 81-84 are `gradient_conflict.py`, both seeds, 500 updates,
with and without `--bank`.

**F51 (probes 85-86). The decoy gate's floor is 0.35, not 0 — and read
against it, the twins fail in OPPOSITE directions.** Every decoy gate so
far has been stated as "collapses to chance" and read as though chance
were 0. Chance was never measured. It is now: drive each twin with
uniformly random actions and no agent at all.

| twin | uniform random (mean / max) | best fixed action |
| --- | --- | ---: |
| choiceA | 0.371 / 0.500 | 0.289 |
| choiceB | 0.336 / 0.469 | 0.293 |

Chance is **symmetric** across the twins (0.371 vs 0.336), which kills
the tempting explanation that choiceA is simply the twin random play
solves by accident. Read against the measured floor, the co-trained
loop's decoy numbers say something sharper than "one twin fails":

| decoy, seed 69316 / 69317 | score | vs chance 0.35 |
| --- | --- | --- |
| choiceA | 0.969 / 0.906 | far ABOVE — plays a real policy on noise |
| choiceB | 0.031 / 0.062 | far BELOW — actively does the wrong thing |

This is the default-context asymmetry (F11) demonstrated positively
rather than inferred. Handed noise, the plant does not become ignorant in
either context: it plays choiceA's policy regardless of which twin it is
actually in. On choiceA that scores 0.91-0.97; on choiceB the same
behaviour scores *below random*, because choiceB rewards the opposite
action. One fixed default explains both numbers at once, and nothing
else does.

Consequences. (1) The gate must be stated against the measured floor per
context, not against 0, and "below chance" must be recognised as its own
failure mode — it is evidence of a wrong policy being executed
confidently, which a "collapses toward 0" reading would have scored as a
PASS. choiceB's 0.031 has been reported as passing since F48; it is
better described as failing in the other direction. (2) The ignorance
objective's target is wrong in the same way: it pushes the decoy policy
toward uniform, but uniform is 0.35, and what we actually require is that
the decoy policy carry no information about which twin is present.

**Methodological, third instance (F46, F49, now this).** The first
version of this diagnostic reported decoy entropy 1.3859 against
ln(4) = 1.38629 — numerically exact uniformity — for a policy that scored
0.875 when sampled, which is impossible when chance is 0.371. The
measurement was averaging over post-death steps, where the episode keeps
stepping and the logits drift to uniform. Masked by `alive`, the same
quantity is being re-measured now. The rule has now failed us three
times in three different guises: a signal measured before a phase that
destroys it (F46), a statistic inheriting its policy's degeneracy (F49),
and an average taken over steps that did not determine behaviour. Report
the support a statistic was computed over, every time.

Probes 85-86 are `chance_baseline.py`; the entropy re-measurement is
`cotrained.py` with the masked diagnostic.

**F52 (probes 87-88). Two battery gates cannot tell a learner from a
constant action.** F51 measured the floor for the twins; this measures it
for every game. The floor is the max of uniform-random play and the best
single constant action, since a flat-logit argmax collapses to the
latter. Headroom is ceiling minus floor — how much a game can actually
discriminate.

| game | floor | ceiling | headroom |
| --- | ---: | ---: | ---: |
| avoid1 | 0.902 | 0.922 | **0.020** |
| dualBC | 0.626 | 0.720 | **0.094** |
| dualAD | 0.588 | 0.686 | **0.098** |
| forageA | 0.168 | 0.453 | 0.285 |
| dualAC | 0.609 | 1.000 | 0.391 |
| choiceA | 0.371 | 1.000 | 0.629 |
| choiceB | 0.336 | 1.000 | 0.664 |
| intercept1 | 0.039 | 0.313 | 0.274 |
| collect1 | 0.324 | 0.547 | 0.223 |

**avoid1 is not a gate.** A constant action scores 0.902 against a
calibrated ceiling of 0.922, so the entire measurable range is 0.020. It
is an avoidance game, and standing still or pushing into a wall survives
trivially — the degenerate policy is nearly optimal by construction.
Every battery result reporting avoid1 near 0.9 has been reporting a
constant action as a pass. dualAD and dualBC are weak for the same reason
at under 0.1 of headroom.

This does not overturn the battery — choiceA/choiceB (0.63/0.66),
dualAC (0.39), forageA (0.29), collect1 (0.22) and intercept1 (0.27) all
discriminate properly, and they carry the load-bearing claims. But raw
mastery against the ceiling alone flatters any game with a high floor, so
`normalised()` (floor-to-ceiling scale) now ships beside `SOLO_CEILINGS`,
and `CHANCE_FLOORS` is pinned by test.

**Independent confirmation for F27.** The composition games measure
0.385-0.416. F27 read the held-out pairing (0.27-0.57 against a 0.47
random-bank control) as at-chance; the floor measured here from an
entirely different direction agrees with that control. The composition
negative stands on two independent baselines now.

**Second metric-misuse caught in one session.** The first run of this
probe reported dual floors of 0.98 and composition floors of 0.96 —
above dualAD's own solo ceiling, which is what made it obviously wrong.
Cause: `mastery` scores dual games by per-rule accuracy but only when
`rule_accuracy` is present, and silently falls through to the reward
branch otherwise, which credits an agent that engages every trial and
knows neither rule. The harness had not supplied the verifier-side
fields. Fixed by supplying them exactly as `rollout_family` does. Note
the failure mode: not an exception, a plausible number. The only thing
that caught it was a floor exceeding a ceiling.

Probes 87-88 are `chance_floors.py` over the battery, twins, composition
suite and extras.

**F53 (probes 89-90). The co-trained loop's choiceA mastery is the PROBE,
not the agent. F48 and F51's readings of the decoy gate are withdrawn.**
Chasing the contradiction F51 left — a decoy policy at entropy 1.3852
against a maximum of 1.3863 (numerically almost uniform) that still
scored 0.9375 when *sampled*, where uniform random scores 0.371 — the
first explanation (averaging over post-death steps) was wrong:
`decoy_live_fraction` is 1.000, nothing dies, and the masked and unmasked
entropies are identical.

The real cause is the harness. Every episode opens with `probe_steps`
steps of a fixed, hand-coded `test_action` that deliberately steps onto
the positive-plane item so the agent can read the reward sign and infer
which twin it is in. But **choiceA's rule *is* "take the positive-plane
item"** — the probe action performs choiceA's task — and mastery for a
choice game is `(total_reward > 0)`, which the probe alone can satisfy.
On choiceB the identical probe action is exactly wrong, which is why that
twin scored below chance.

Measured with **no agent at all** — probe prefix, then random or frozen
actions:

| scoring | choiceA | choiceB |
| --- | ---: | ---: |
| no probe + random | 0.356 | 0.344 |
| probe + random | 0.832 | 0.090 |
| probe + frozen | **0.961** | **0.004** |

Against the co-trained loop's own numbers (seed 69316):

| gate | loop | no-agent probe artifact |
| --- | ---: | ---: |
| choiceA train | 0.961 | 0.961 |
| choiceA decoy (greedy) | 0.969 | 0.961 |
| choiceA decoy (sampled) | 0.938 | 0.832 |
| choiceB decoy (greedy) | 0.000 | 0.004 |
| choiceB decoy (sampled) | 0.094 | 0.090 |

**choiceA's entire mastery is what the probe delivers for free.** There is
no evidence the agent learned anything on that twin. choiceB's 1.000
against a 0.004 floor is real learning; choiceA's 0.961 is the harness
scoring itself.

Withdrawn as a result:
- **F48's "both twins mastered"** — only choiceB was.
- **F48/F51's "the decoy gate fails on choiceA"** — you cannot fail a gate
  the harness passes for you. The decoy number never measured the bank.
- **F51's attribution of the asymmetry to F11's default context** — the
  asymmetry is manufactured by a probe action that performs one twin's
  task and anti-performs the other's. F11 may still hold elsewhere; it is
  not what these numbers showed.
- **The entire ignorance escalation (0.5 -> 2.0, every-3 -> every-1)** was
  chasing an artifact. That the 4x increase "moved nothing" is exactly
  what an artifact predicts.

Not affected: F49, F50 and F52 are separate experiments, and the probe
harness exists only in the addressing scripts — the committed battery,
bank and consolidation code call `rollout_family`, which has no probe
phase. The promoted rungs do not inherit this.

Still real: cross-feeding drives choiceA to 0.000, far *below* its 0.961
probe floor, so a wrong fragment actively destroys reward the harness
would otherwise hand over. That is a genuine effect, but it is
"cross-feed is worse than doing nothing", not "cross-feed inverts a
learned skill".

**Fix, verified.** Score only what happens after the probe. With
post-probe-only scoring the artifact disappears and the twins become
symmetric again: probe+frozen falls to 0.238 / 0.180 and probe+random to
0.383 / 0.289, both back at the no-probe floor. The addressing line must
be re-run under this scoring before any of its claims are restated.

**Methodological, fourth instance.** F46 (a signal measured before the
phase that destroys it), F49 (a statistic inheriting its policy's
degeneracy), F52 (a metric silently taking its wrong branch), and now a
score collecting reward the agent did not earn. Each was a plausible
number, not an error. The generalisation worth keeping: **an experiment
must be able to fail.** Before trusting any gate, run it with no agent at
all and check that it fails — the no-agent control would have caught all
four.

Probes 89-90 are `probe_earns_it.py`.

**F54 (probe 91, re-analysis). The battery's bank IS necessary — both
necessity gates pass against measured floors on three seeds.** F48 left a
standing action: re-audit every promotion that rested on cross-feed alone
with a norm-matched decoy. The audit turns out to be cheap, because the
staggered battery archive already recorded BOTH necessity gates. What it
lacked was a floor to read them against. With F52's measured floors:

| game | trained | bank withheld | norm-matched decoy | floor |
| --- | ---: | ---: | ---: | ---: |
| choiceA | 1.000 / 1.000 / 1.000 | 0.141 / 0.156 / 0.391 | 0.328 / 0.234 / 0.266 | 0.371 |
| choiceB | 1.000 / 0.719 / 0.875 | 0.406 / 0.281 / 0.188 | 0.047 / 0.219 / 0.172 | 0.336 |
| dualAC | 1.000 / 0.860 / 1.000 | 0.434 / 0.403 / 0.526 | 0.609 / 0.505 / 0.530 | 0.609 |
| dualAD | 0.799 / 0.947 / 0.953 | 0.458 / 0.648 / 0.462 | 0.567 / 0.501 / 0.460 | 0.588 |
| dualBC | 0.757 / 0.607 / 0.630 | 0.542 / 0.353 / 0.531 | 0.439 / 0.499 / 0.540 | 0.626 |

(seeds 69316 / 69317 / 69318.)

**Every discriminating game clears both gates on all three seeds.** Remove
the bank and performance falls to or below the measured floor; replace
the fetched fragments with norm-matched noise and it does the same. The
one game that fails is avoid1, whose decoy sits at 0.922 — but F52 showed
avoid1 has 0.020 of headroom and cannot discriminate a learner from a
constant action, so it is uninformative in both directions.

This is the necessity evidence weakness 12 has wanted, and it was sitting
in the archive unreadable for want of a floor. The bank is not merely
sufficient to override: with the fragments gone or replaced, the plant
cannot perform.

**It also corroborates F53 from the opposite direction.** The battery
harness has no probe phase, and there the decoy collapses correctly. The
co-trained addressing harness has one, and there choiceA's decoy sat at
0.97. Same architecture, same decoy construction, opposite outcomes — the
difference is the probe, exactly as F53 diagnosed. Two independent routes
now say the addressing failure was the harness rather than the bank.

**Scope, stated precisely.** This closes necessity for the battery, where
fragments are fetched by oracle or learned per-context selection. It says
nothing about SELF-addressing, where the agent must infer the context
from its own actions — that remains open and is what the corrected re-runs
are testing. The claim is "the bank carries the skill", not "the agent can
find the right page unaided".

**Noted in passing:** dualAD scores 0.95 against a calibrated solo ceiling
of 0.686 (normalised 3.7). Partly F18's super-solo transfer, but a
normalised score that far above 1.0 suggests the dualAD ceiling is
mis-calibrated low and should be re-run.

Probe 91 is a re-analysis of `staggered_battery_v1_2026-08-07`; no new
training.

**F55 (probes 91-92, 16 seeds). Corrected self-addressing passes the full
gate set at a 5/16 rate; acquisition, not addressing, is the binding
constraint. The symmetric-plant variant is rejected.** First use of the
remote 192-core lab: the corrected co-trained loop (post-probe scoring,
F53) on seeds 69316-69331 simultaneously, judged against the
pre-registered bar (both twins >=0.9 vs floors 0.383/0.289, cross-feed
<=0.1, sampled decoy at floor).

| gate | rate |
| --- | ---: |
| full bar | **5/16** |
| both twins mastered | 8/16 |
| cross-feed inverts | 10/16 |
| decoy at floor | 11/16 |

Read conditionally, the story is sharp: **among the 8 seeds that master
both twins, 7 invert under cross-feed and 5 clear everything.** When the
plant acquires both contexts, the addressing machinery — probe, sign
encoding, fetch, and the bank's necessity — works most of the time. The
failures are dominated by acquisition (winner-take-all or double
failure: 0.47/0.91, 0.26/1.00, 0.21/0.27...), which is weakness 15's
constraint surfacing in the twin setting, not an addressing defect.
Self-addressing WORKS at some rate; making it reliable is now an
acquisition-stability problem, which is where the literature's levers
(small policy-head init, return normalisation, larger batches, TSCL-style
learning-progress sampling — LITERATURE_MAP S4/S5) point next.

The symmetric-plant variant (mixture gradient only, Galashov-style) is
**rejected**: local seeds fail twice in two different ways (69316
acquisition 0.281; 69317 decoy 0.500), remote partials split 1/2, and
its mechanism removes the balancing that F50 showed is load-bearing.
Four runs, three failures, no configuration in which it beats the
baseline. The remaining 14 remote symmetric runs were abandoned with the
instance — confirming a refutation at n=16 was not worth the rental.

**Platform nondeterminism, recorded as a rule.** The same seed does not
reproduce across torch 2.12/Linux and local torch/macOS: local 69316
passed the full bar, remote 69316 failed cross-feed outright. Per-seed
identities are platform-bound; **only rates transfer across machines.**
Seed-widening claims must therefore be stated as rates with the platform
named, and a promotion that leans on a specific seed's pass is evidence
about one platform only.

Evidence: `cotrained_seed_widening_v1_2026-08-08` (16 remote + 4 local
runs, README, SHA256SUMS).

**F56 (probes 93-96). The generic on-policy reliability levers do not fix
twin acquisition; the pre-registered consequence is a pivot to the
goal-factored design.** Mastery-only screen, 4 fresh seeds x 4 configs:
control 0/4 both-mastered, small policy-head init 0/4 (several total
failures — with early entropy that high, 3000 updates is too few to
converge), advantage normalisation 1/4, combined 0/4. The bar stated
before the run — beat control's rate or the levers are the wrong tool —
was not met by any lever.

This is the second time a mechanism transplanted from the literature
without its ecosystem has failed here (the symmetric plant was the
first), and it sharpens F55: the twins' acquisition instability is not
generic policy-gradient variance, it is the specific difficulty of
holding two anti-correlated policies in one small plant while a third
process learns to route between them. The literature levers treat the
symptom class, not this cause. Per the pre-registration in the screen
README, the goal-factored redesign (`docs/GOAL_FACTORED_DESIGN.md`) —
which shrinks the game-specific surface from a whole policy to a
destination vector — is now the MAIN path for the self-addressing line.

Confound recorded rather than hidden: this cohort differs from F55's in
platform, seeds, AND ignorance-off simultaneously, so the screen's 0/4
control against F55's 8/16 is not evidence that ignorance helps
acquisition — but that hypothesis is now live and cheap to test (one
matched pair), and F32 forbids treating the cross-cohort comparison as
an answer.

Evidence: `acquisition_levers_screen_2026-08-08`.

**F57 (probes 97-108). The goal-factored cued rung: skill in the bank,
executor in the weights, 3/6 seeds through every gate.** The design
(`docs/GOAL_FACTORED_DESIGN.md`) splits the plant into a game-INVARIANT
goal-following executor (trained verifier-free on self-checkable micro
goals, then frozen) and a bank of per-game destination fragments, with
the game's name rendered in the world as the cue. Per-game gradients
touch only the fragments and a 2->2 cue-reader, so the information
asymmetry F55/F56 showed penalties cannot enforce is enforced by
construction.

| seed | draws | mastery A/B | cross-feed | decoy | label-swap |
| ---: | ---: | --- | --- | --- | --- |
| 69316 | 1 | 1.00 / 1.00 | 0.05 / 0.00 | 0.29 / **0.49** | 0.00 / 0.00 |
| 69317 | 1 | 1.00 / 1.00 | 0.00 / 0.00 | 0.31 / 0.33 | 0.00 / 0.00 |
| 69318 | 1 | 0.82 / 0.77 | 0.06 / 0.00 | 0.18 / **0.43** | 0.01 / 0.00 |
| 69319 | 1 | 0.28 / 0.48 | 0.20 / 0.01 | 0.30 / 0.35 | 0.13 / 0.18 |
| 69320 | 1 | 1.00 / 1.00 | 0.00 / 0.00 | 0.31 / 0.30 | 0.00 / 0.00 |
| 69321 | 1 | 1.00 / 1.00 | 0.02 / 0.00 | 0.24 / 0.25 | 0.00 / 0.00 |

Floors 0.344 / 0.320. **Full bar 3/6; both twins mastered 4/6;
cross-feed inverts below floor 6/6; label-swap collapses 5/6.** Every
full-pass run masters at exactly 1.00/1.00 — this architecture does not
half-work.

**What the causal gates establish.** Swapping the two fragments inverts
behaviour to below chance on essentially every run; rendering the wrong
game's NAME carries behaviour with it, wholesale. So the banner drives
fetch and the fragment drives behaviour, with a plant that never saw a
verifier reward. That is the program's core claim demonstrated end to
end: an amodal executor of fixed size, and skill that lives outside it.

**Comparison to the line it replaces.** The monolithic co-trained loop
measured 5/16 on its bar with acquisition binding (F55), and generic
reliability levers did nothing (F56). Here the causal gates almost never
fail; what varies is acquisition. Removing the shared-policy competition
by construction changed the failure profile, which is what the design
predicted.

**Eleven measured iterations produced three mechanisms**, each from a
failure the previous fix exposed: engagement economics in phase 1 (the
plant satisfied "consume plane B" by IDLING on 1493 of 1536 steps until
idleness was priced -- the family's own DUAL_IDLE_COST lesson in
self-supervised form); destinations expressed in the plant's own goal
vocabulary; and no reward feedback into the controller, because a plant
taught to self-correct from reward re-derives the twin from consequences
and makes the bank decorative (F48's leak, fourth setting).

**Two harness bugs are worth remembering more than the mechanisms.**
`artifact_events` broadcasts dim 0 as tokens, so per-row commands became
32 events flooding a capacity-8 window and NO agent ever saw its own
command -- three "findings" about vocabulary readability were readings of
severed-channel noise, all retracted. And the decoy gate used a single
fixed noise vector: a sample size of one, where a direction that happens
to point plane-B-ward reads as a failed necessity gate. Both produced
plausible numbers rather than errors.

Open: decoy on ONE twin at 0.43-0.49 on two seeds (the default-response
residue), and phase-2 acquisition on two seeds. Evidence:
`goal_factored_cued_v1_2026-08-09`.

**F57 (probes 97-100, 4 arms x 8 seeds). Ignorance trades against
acquisition; gating it on measured competence recovers both.** The
goal-factored cued rung's two residual failures were decoy-on-one-twin
and phase-2 acquisition. A matched four-arm sweep, same seeds
(69316-69323), same everything else:

| arm | mastery both | FULL bar | worst-twin decoy |
| --- | ---: | ---: | ---: |
| ignorance 0.0 | 6/8 | 2/8 | 0.551 |
| ignorance 2.0, ungated | 4/8 | 4/8 | 0.323 |
| **ignorance 2.0, gated at 0.7** | **6/8** | **5/8** | **0.355** |
| ignorance 3.0, gated at 0.85 | 5/8 | 3/8 | 0.391 |

Ignorance does what it was built for — worst-twin decoy falls from 0.551
to 0.323 against a measured floor of ~0.33, so a noise fragment stops
buying performance — but it costs acquisition, 6/8 down to 4/8. This is
the retention/acquisition tension (F24, Continual World) appearing in a
third place: pressure that protects one property degrades the other.

**Gating on measured competence gets both.** Ignorance switches on only
once the laggard command score clears 0.7, so the executor acquires
unimpeded and necessity is trained in afterwards. That recovers mastery
to 6/8 while holding decoy at 0.355, and is the best full-bar rate the
program has produced on this problem: 5/8. Note the ordering is the
promoted consolidation line's own (acquire, then protect), arrived at
here from measurement rather than by analogy.

Harder is worse: ignorance 3.0 gated at 0.85 is the weakest gated arm
(3/8), so this is not a dial to turn up.

**Retraction.** I hypothesised from a 6-seed run that ignorance
regularises phase-1 and collapses the restart count to 1. It does not:
across 8 seeds per arm the draw means are 1.75 (off) and 1.38 (on), max
3 in both, and the gated arms sit at 2.12. The all-ones run was
small-sample luck.

Evidence: `goal_factored_cued_v1_2026-08-09` plus the four-arm sweep.

**F58 (probes 101-108). The executor does not fail from goal COUNT; it
fails because a small goal set is memorisable. Four mechanisms
eliminated, and the arity framing withdrawn.** The arity-3 executor was
logged as weakness 19 on the theory that three goals are harder than
two. Measured, on six fresh seeds, phase-1 only, first draw:

| arm | converged | note |
| --- | ---: | --- |
| mixed, 1500 updates | 0/5 | every side 0.01-0.03 |
| mixed, 3600 updates | 0/6 | budget is not the constraint |
| mixed, hidden 64 | 0/6 | capacity is not the constraint |
| sequential isolation | 0/6 | but the ONLY arm producing a learned side |
| consolidated (EWC 1 / 10 / 200) | 0/6 each | goal0 1.00, later goals 0.02 |
| **two goals only** | **0/4** | **so it is not the arity** |

Three things fall out. (1) **Consolidation strength is irrelevant** —
EWC at 1, 10 and 200 give identical results, so my earlier "over-
protection / intransigence" reading is withdrawn: the penalty was never
the blocker. (2) **Two goals fail on this game too**, so the arity-3
framing in weakness 19 is wrong. (3) The signature across every arm is
the same: **one goal reaches 1.00 and the rest sit at 0.02** — the plant
learns an unconditional habit and never reads the goal channel at all.

The mechanism is now clear and is a property of the TASK DESIGN, not the
optimiser. With a handful of goals, "always do the one thing" is a
competitive policy, so nothing ever forces the plant to condition on its
instruction. Isolation is worst of all: with a single goal commanded,
ignoring the goal channel is *optimal*. Every curriculum we tried
therefore trains a habit and then asks it to become conditional, which
is the one transition none of them can make.

This is the literature's own result arriving from our side: Chan et al.
(LITERATURE_MAP S1) find that few, frequent, fixed-meaning classes drive
in-WEIGHTS memorisation while many varied ones drive in-CONTEXT reading.
Three goals is the pathological regime by that criterion.

**Consequence, and it is a redesign rather than a fix.** The remedy is
not a better update rule but a goal space too large to memorise: a
UNIVERSAL reacher, "from any state X reach any state Y", with hindsight
relabelling supplying free dense data (every trajectory is a correct
demonstration of reaching wherever it ended up — re-derivation, so the
no-replay rule is intact). Conditioning stops being something to
enforce and becomes the only representable solution. It also predicts
that adding a game should require ZERO plant change, only a new target —
the continual-learning claim by construction rather than by measurement.
Recorded as the executor's new direction; the bank keeps its job
unchanged, since a universal reacher tells you how to get anywhere and
nothing about where to want to go (the twins survive the reframe
untouched).

Probes 101-108 are `goal_composition.py` with `--curriculum`,
`--hidden`, `--ewc`, and `--phase1-sides`.

**F59 (probes 109-120). The goal-conditioning machinery works; the GRID
reacher does not, and five interventions failed to fix it.** The
universal-reacher reframe (GOAL_FACTORED_DESIGN revision) was built in
two modalities. Same controller, same goal channel, same decoder.

| testbed | reach | floor | path ratio | budget |
| --- | ---: | ---: | ---: | ---: |
| **numeric line** | **0.938** | 0.344 | **1.010** | 250 updates |
| grid (best arm) | 0.375 | 0.180 | — | 2500 updates |

**The number line settles the central question.** With state = an
integer, the agent reaches 94% of commanded targets by a path within 1%
of the provably optimal `|Y - X|`, from a tenth of the budget. So the
controller, the goal channel and the decoder are sound: F58's "the plant
cannot condition on a goal" is refuted. It is also the first time this
program has verified OPTIMALITY rather than inferring competence from
score, and the first test of the amodal claim in a modality with no
spatial structure.

**The grid stays at ~2x floor through five interventions**, each
measured on two seeds:

| intervention | result |
| --- | --- |
| egocentric crop/roll, absolute goal | WORSE (0.207/0.258 vs 0.316) |
| relative goals | held-out 0.273 -> 0.461, trained flat |
| egocentric crop, relative goal | worse again; most habitual arm (0.532) |
| self-supervised localisation head | WORSE (0.375 -> 0.305) |
| true shortest-path (BFS) reward | 0.285 vs Manhattan's 0.375 |

Only relative goals earned their place, and for generalisation rather
than competence: offsets are translation-invariant, so a reserved
quadrant is not novel in offset space. That is also the property that
makes a goal TREE work, since a subgoal must mean the same thing
wherever it appears.

**Two retractions.** (1) A linear probe reads avatar position out of the
screen encoder at 0.699/0.672 per axis against 0.125 chance — I inferred
that the ~30% error was the binding constraint, and the localisation
intervention refuted it. A measured correlate is not a cause; the
intervention is what settles it. (2) I predicted the BFS reward would
fix the stall, since the map has a solid wall column and 21 of 64 cells
need a detour that a Manhattan reward pays negatively. It did not:
Manhattan is right for the other 43 cells and gives a smooth dense
gradient, while true distance adds plateaus and long detours a 24-step
budget often cannot finish. Correcting the reward also made the task
harder.

**Measurement faults in this rung alone: three** — Manhattan distance in
a walled map, the unreachable-target sentinel leaking into the metric
(one seed collapsed to conditioning agreement 1.000), and the
localisation red herring. Session total: six. The mechanism has been
changed far less often than the things measuring it, and every "the
learner cannot do this" has so far resolved into "the task or the metric
was wrong" — except this one, which is still open.

**Standing:** the reacher is validated in the modality where perception
is free and unvalidated where it is not. The grid gap is a perception or
horizon problem, not a goal-conditioning one, and it is the honest
blocker for the reacher rung. Numeric remains the control surface where
every part is known to work.

Probes 109-120 are `numeric_reacher.py` and `universal_reacher.py`.

**F60 (probes 121-124). Perception is NOT the grid reacher's bottleneck.
Seven interventions refuted; the 2D task itself is the difficulty.** The
decisive test: replace the screen encoding with a perfect one-hot of the
avatar's own cell -- exactly the clean state the numeric reacher gets.

| condition | trained reach |
| --- | ---: |
| numeric line | **0.938** |
| grid, ORACLE state (perfect perception) | 0.379 |
| grid, screen encoder + one-hot goal | 0.316 |
| grid floor | ~0.20 |

Flawless self-knowledge recovers almost none of the gap. Every
perception-side hypothesis is therefore dead, and with it the story I
had been telling since the linear probe:

| intervention | verdict |
| --- | --- |
| egocentric roll / crop (absolute goal) | worse |
| egocentric crop (relative goal) | worse, most habitual arm |
| self-supervised localisation head | worse |
| true shortest-path (BFS) reward | worse than Manhattan |
| one-hot goal encoding | +0.03, marginal |
| relative goals | held-out only (0.273 -> 0.461) |
| **oracle state** | **+0.06, marginal** |

What remains is the task. Numeric: two actions, one dimension, no
obstacles, and a path that is one sustained direction. Grid: four
actions, a wall pierced by a single gap, and paths needing turn
sequences. The difficulty is SEQUENTIAL DECISION-MAKING under sparse
credit, not seeing.

**Consequence: build the ladder from the working end.** Rather than
attacking the grid again, vary one axis at a time between the two --
1D line, 1D with a blocked cell, 2D open, 2D walled, 2D from pixels --
and find exactly which step breaks competence. Every rung is tiny, so
this also supplies many points for the acquisition-cost curve, and it
converts the games from a target into an INSTRUMENT: the headline gate
becomes whether pre-training on lower rungs makes higher rungs cheaper
to acquire, measured against from-scratch controls. Transfer must be
measured, not assumed -- a flat curve is a real result and would say the
shared controller carries nothing across these steps.

Probes 121-124 are `universal_reacher.py --oracle-state`.

**F61 (probes 129-132). Freezing the plant does NOT rescue cross-domain
transfer. The plant absorbs domain-specific POLICY, not just habits, and
a goal adapter cannot undo it.** The sharpest test of the storage rule
we have posed, and it fails.

| arm | r4 (walled grid) reach | mastered |
| --- | ---: | --- |
| cold start | **0.996** | 150 updates |
| warm from r1 (line), trainable plant | 0.277 | 0/2 seeds |
| warm from r1, **frozen plant + 4160-param goal adapter** | **0.211** | 0/2 seeds |

The warm rung itself reaches 1.000, so the plant did master the line
before transferring. Pre-training on a structurally unrelated domain
costs roughly 0.78 of final performance, and freezing recovers none of
it — it is marginally worse.

**Why the adapter cannot work, stated precisely.** The adapter re-maps
what the goal MEANS; it cannot change how the plant PURSUES goals. A
plant trained on a line has learned "pick a direction and hold it" as
its pursuit policy. That policy is wrong in 2D with an obstacle, and no
re-encoding of the goal vector repairs a wrong policy. So the failure
localises: after training on one domain the plant holds domain-specific
CONTROL, not merely a stylistic habit.

**This is the storage rule violated at the deepest level yet.** F11 put
a default context in the weights, F50 let one twin take the plant, F58
produced an unconditional habit instead of goal-reading — all content
that should have been external. Here the thing in the weights is the
pursuit policy itself, which the architecture has always assumed was the
legitimately shared part. The assumption is now measured false for
structurally unrelated domains.

**Two admissible routes, both untested.**

1. **Never let the plant specialise.** Train it on diverse domains
   concurrently from the start, so no single domain's control policy can
   become the prior. This is the Chan et al. argument (LITERATURE_MAP S1)
   applied one level up: few domains and it memorises one, many and only
   the common machinery survives. Cheap to test with the ladder — train
   r1 and r3 interleaved, then measure r4.
2. **Give the plant no policy to be wrong.** Keep only the transition
   model in weights and DERIVE the policy by search over that model.
   There is then nothing domain-specific to transfer, because there is no
   learned controller — only a map plus a planner. This is the
   deliberative-search rung (GOAL_FACTORED_DESIGN rung C), promoted from
   "future work" to "the architecture's answer to F61".

Route 2 is the stronger claim and the better fit to the reacher
formulation: a learned d(X, Y) IS a search heuristic, and a planner over
a model has no habits to carry.

Probes 129-132 are `reacher_ladder.py --warm-start r1 [--freeze-plant]`.

**F62 (probes 133-138). Concurrent multi-domain warm-start converts
catastrophic negative transfer into mild negative transfer — but does
NOT produce positive transfer. And the test cannot currently detect it.**
Route 1 of F61, three arms, two seeds, target = r4 (walled grid):

| arm | r4 reach | mastered at | path ratio |
| --- | ---: | ---: | ---: |
| cold start | **1.000** | **200** | 1.03 |
| warm from r1 alone | 0.277 | never (0/2) | 1.23 |
| warm from r1+r3 concurrently | 0.824 | 400 (1/2) | 1.10 |

The mechanism is confirmed: a plant trained on ONE domain cannot
separate "how to pursue a goal" from "how to pursue a goal in a line" —
both explain its data, and the specific one is cheaper, so it stores
that. Two structurally different domains at once make those hypotheses
disagree, and only the shared machinery survives contact with both.
Diversity is not regularisation here; it is what makes the general layer
IDENTIFIABLE. That recovers 0.277 -> 0.824.

**But the objective is not met.** Stated properly (and this is the
formulation to adopt): *produce a program such that having learned task
A makes a novel task B faster to learn than from scratch.* Cold masters
at 200 updates; the best warm arm needs 400 and only on one seed. Prior
training still costs. The honest verdict is that we have reduced harm,
not created benefit.

**A methodological flaw in the test itself, recorded before it misleads
us.** r4 is too EASY cold — 200 updates to 1.000. There is almost no
headroom for a prior to save anything, and any prior at all is a
handicap on a task that is cheap to learn fresh. **A transfer test needs
a target that is expensive from scratch**, or positive transfer is
undetectable by construction. Every transfer measurement in this project
so far has used cheap targets, so the absence of positive transfer is
weaker evidence than it looks.

Next: re-run the same three arms against a target that is genuinely
expensive cold (a larger grid, the composition suite, or the twins),
where a prior has room to pay for itself. Until then "no positive
transfer" should be read as "not yet measurable", not as "does not
exist".

Probes 133-138 are `reacher_ladder.py --warm-mix`.

**F63 (probes 139-144). Positive transfer is NOT demonstrated, on a
target expensive enough to show it. Diversity buys neutrality, not
benefit.** F62 found our transfer tests used targets that master cold in
200 updates, leaving no headroom. Sparse reward fixes that -- arrival
must be discovered, and cold start never masters. Three arms, two seeds:

| arm | reach | mastered | path ratio |
| --- | ---: | --- | ---: |
| cold | **0.625** | 0/2 | 1.02 |
| warm from r1 alone | 0.176 | 0/2 | 1.04 |
| warm from r1+r3 concurrently | 0.613 | 0/2 | 1.10 |

Multi-domain warm-start is NEUTRAL (0.613 vs 0.625). Single-domain is
harmful (0.176), reproducing F61/F62 a third time. So across cheap
targets (F62) and expensive ones (here), the pattern is stable:

> **Diversity converts harmful priors into neutral ones. Nothing so far
> converts them into helpful ones.**

Stated against the adopted objective -- *produce a program such that
having learned A makes a NOVEL B faster to learn than from scratch* --
the project's founding claim has **no supporting measurement**. Prior
learning has never once made a novel task cheaper here. That is the
honest headline and it should not be softened: every positive result in
this program (bank necessity F54, composition, cued addressing F57) is
about STORING and REUSING skills within a task family, not about
compounding across novel ones.

Caveat recorded rather than used as an excuse: no arm mastered (0/2
everywhere), so 800 updates is short for this target. The reach numbers
still separate the arms cleanly, but a budget where some arm succeeds
would measure the curve rather than a single point.

**Two candidate explanations, both untested.**

1. **Nothing pressures reuse.** There is no cost term in the objective
   and no retrieval-before-learning loop (ARCHITECTURE.md §5.2, §5.3).
   A system with no incentive to consult its prior will relearn, and a
   neutral prior is exactly what "carried but unused" looks like. This
   is the cheaper hypothesis and it is directly testable: add a cost
   term, add a try-the-bank-first step, re-run these three arms.
2. **The plant is the wrong home for a policy at all.** F61 showed a
   frozen plant plus a goal adapter cannot repair a wrong pursuit
   policy. Route 2 keeps only a transition model in weights and DERIVES
   the policy by search, so there is no learned controller to carry or
   to be stale. This is the larger rebuild and the stronger claim.

Probes 139-144 are `reacher_ladder.py --sparse`.

**F64 (probes 145-150). Transfer of CAPABILITY is demonstrated. Transfer
of LEARNING SPEED is not. The distinction is the finding.** Two things
F63 never measured: what the warm plant scores on the target BEFORE any
target training, and what happens when the plant is frozen after DIVERSE
domains rather than one. Sparse r4, two seeds:

| arm | zero-shot | final | budget |
| --- | ---: | ---: | ---: |
| cold | — | **0.625** | 800 |
| multi-domain warm + frozen plant | **0.520** | 0.543 | 200 |
| multi-domain warm + frozen plant | 0.520 | 0.613 | 800 |

**Zero-shot reach is 0.520 against a 0.172 floor** — three times chance,
with zero gradient steps on the target. A plant trained on a line and an
open grid carries genuine, substantial competence to a walled grid it
has never seen. That is the first direct evidence for the founding claim
in this project, and it was invisible until now because every previous
transfer number confounded what CARRIED with what was RELEARNED.

But it does not become speed. Cold start still finishes ahead at matched
budget (0.625 vs 0.613), and adapting the frozen prior moves it only
0.520 -> 0.613 in 800 updates. So:

> **Prior learning transfers competence but not learnability. The
> architecture stores and reuses; it does not yet compound.**

That is a more precise statement than F63's flat "no positive transfer",
and it relocates the problem: the bottleneck is not what the plant knows
but the channel through which a new task can exploit it. A 4160-param
linear adapter on the goal payload is almost certainly too weak a
channel -- it can re-map what a goal MEANS but cannot change what the
frozen policy DOES with it (F61's mechanism, now confirmed from the
other side).

**Correction to F61.** F61 concluded "freezing does not rescue transfer"
from a single-domain warm start -- a plant whose policy was already
measured wrong. Freezing preserved something worthless, so the test said
nothing about freezing. The correct claim was "freezing a bad policy
does not help". I generalised from one condition to the mechanism, and
that wrong generalisation shaped F62 and F63: I pursued curriculum fixes
for two rungs while never testing the obvious combination of the two
ideas already in hand. Each mechanism was tested in isolation, each
found insufficient, and the pair never tried.

**Also recorded:** I reported a smoke run (one seed, 0.688 beating cold)
as a positive signal. It did not replicate. Single-seed smoke numbers
are for checking that code runs, not for claims.

Next, in order of cost: a richer adaptation channel than a linear goal
adapter (the frozen prior clearly has more to give than 0.613 extracts);
then the cost term and retrieval-before-learning, which remain untested
and are the only mechanisms that would make reuse preferred over
relearning.

Probes 145-150 are `reacher_ladder.py --sparse --warm-mix --freeze-plant`.

**F65 (probes 151-156). Widening the adaptation channel does nothing.
The bottleneck is not capacity.** F64 located the problem as the channel
through which a new task exploits a frozen prior — 4160 params that
re-map goal meaning but cannot change action selection. Adding the
output head (5348 params) should have loosened exactly that.

| arm | zero-shot | final |
| --- | ---: | ---: |
| cold | — | 0.625 |
| goal adapter only | 0.520 | 0.613 |
| goal adapter + output head | 0.520 | **0.613** |

Identical to three decimals. So the frozen prior's competence is not
being withheld by a too-narrow adapter, and F64's diagnosis is wrong.
Something else prevents 0.520 of transferred capability from becoming
better-than-cold learning, and adding parameters to the adaptation path
is not it.

The two mechanisms that have never been tested remain the live
candidates, and both are about INCENTIVE rather than capacity: there is
no cost term making reuse preferable to relearning, and no
retrieval-before-learning step that consults the bank at all. A neutral
prior is exactly what "carried but never consulted" looks like.

**Process failure recorded.** The amortisation arm of the same sweep did
not run the configuration I specified: my shell wrapper referenced `$3`,
so every flag after the first was silently dropped and the "warm" arm
ran without freezing, without the head, and without multiple targets —
its `cost_per_target 800, lifetime 800` (implying one target, not three)
is what exposed it. Fixed by `"$@"`. This is the same species as the
argparse no-op earlier today and the runs I reported as training while
they were dead: **the harness lying about what it ran.** Third instance
in one session, and the reason every sweep now prints back the
configuration it actually received.

Probes 151-156 are `reacher_ladder.py --adapt-decoder`.

**F66 (probes 157-160). Freezing the plant moves forgetting into the
adapter. The founding claim remains unsupported.** First properly
instrumented compounding test: one warm-start, three sequential novel
targets, cost measured as work actually done (early-stop on mastery, and
a target the prior already solves costs zero), against a cold-sequential
control on the identical sequence.

| arm | lifetime cost | r2 zero-shot | r3 zero-shot | r4 zero-shot | r4 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold sequential | **800** | 0.422 | 0.176 | **0.363** | **0.441** |
| warm + frozen + head | 1300 | 0.531 | 0.184 | **0.129** | 0.133 |

The warm arm costs 60% more and ends worse. But the diagnostic number is
zero-shot ACROSS the sequence: **cold stays roughly flat (0.422 ->
0.363) while warm-frozen DECLINES (0.531 -> 0.129).** It begins with the
better prior and ends with the worse one.

**Mechanism.** With the plant frozen, every task's content must go into
the 5348-param adapter, and each successive target overwrites the last.
We removed forgetting from the plant by freezing it and reintroduced it
in the adapter — smaller, unprotected, and now carrying all task
content. The cold arm, free to update everything, accumulates better
than the "protected" one.

This is the storage rule biting for the sixth distinct time. F11 kept a
default context in weights, F50 let one twin take the plant, F58 stored
a habit, F61 stored a pursuit policy, F64/F65 showed the prior is
carried but unused — and here the bank-substitute itself becomes the
overwritten store. **Freezing does not solve the problem; it relocates
it to whatever is still plastic.**

**Honest note on circularity, stated before the run.** Retrieval-first
makes a warm arm's cost lower almost by construction, so total spend was
never the evidence — zero-shot per target was, and it went the wrong
way. The mechanism flattered itself on cost and still lost.

**Status of the founding claim after 66 findings.** *Produce a program
such that having learned A makes a novel B faster to learn than from
scratch* — no supporting measurement. What IS supported: storage and
reuse within a family (F54 necessity, composition at 85-111% with zero
learning, F57 addressing), and single-target capability transfer (F64,
0.520 zero-shot vs 0.172 floor). What is not: any reduction in the cost
of acquiring a novel task.

The untested mechanism list is now down to one that is genuinely
untried: a plant that holds NO policy at all, with behaviour derived by
search over a learned transition model (GOAL_FACTORED_DESIGN rung C).
Everything else — diversity, freezing, wider channels, cost accounting,
retrieval-first — has been built and measured.

Probes 157-160 are `reacher_ladder.py --retrieval-first --targets`.

**F67 (probes 161-169). Policy-free beats every policy-storing variant,
and transfers. Six failures were one failure.** F11, F50, F58, F61,
F64/65 and F66 are the same event: a policy gets stored, and it is wrong
for the next task. Freezing relocated the staleness; diversity only made
the stored policy less wrong; a wider channel did nothing. So remove the
policy. The plant learns a transition model — (state, action) -> next
state, self-supervised from random play, no reward and no goal — and
behaviour is DERIVED by breadth-first search in that model toward
whatever goal the bank supplies.

Three seeds, r4 (walled grid), sparse:

| arm | reach | target training |
| --- | ---: | --- |
| floor | 0.172 | — |
| warm + frozen policy (F66) | 0.133 | 800 updates |
| cold-trained policy | 0.441 | 800 updates |
| **policy-free, model from r1+r3** | **0.573** | **NONE** |
| **policy-free, model from r4 itself** | **0.969** | none (dynamics only) |

**The founding claim is satisfied for the first time.** A model learned
on other domains, with zero gradient steps on the target, beats a policy
trained on the target directly (0.573 vs 0.441). Prior learning made a
novel task cheaper — in fact free.

**And on its own domain, model+search more than doubles a learned
policy** (0.969 vs 0.441) while needing no policy training at all. The
model reaches accuracy 1.000 in every arm: dynamics are easy to learn;
it was always the policy that was hard.

**Depth buys competence**: 0.531 at depth 6, 0.573 at depth 12. First
evidence in this program for the deliberation property — more thinking,
better behaviour, no additional weights — which weakness 9 has flagged
as untested since the beginning.

**Why a model transfers where a policy cannot.** A model is FACTUAL; a
policy is PREFERENTIAL. A model learned on a line and an open grid is
not *wrong* about a walled grid, only *incomplete* — it has never seen a
wall, so it mispredicts exactly there, which is why transferred search
gets 0.573 rather than 0.969. Incompleteness is repaired by observing
new dynamics, cheaply and additively. A wrong policy must first be
unlearned. That asymmetry is the whole result.

**Architectural consequence.** The plant should hold the map, the bank
should hold destinations, and behaviour should be computed rather than
stored. That is what `GOAL_FACTORED_DESIGN` specified and what was never
actually built — every rung until now trained a policy into the plant
while claiming the plant held only general machinery.

Probes 161-169 are `reacher_ladder.py --model-search`.
