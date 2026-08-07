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
