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

**F68 (probes 170-172). Policy-free does not degrade across a sequence.
This is the forgetting result the whole project was chasing.** F67
showed model+search wins on a single target. The multi-target apparatus
(one agent, three novel rungs in sequence, cost as work actually done)
answers the founding question directly:

| rung | policy-based, cold sequential | **policy-free (model+search)** |
| --- | ---: | ---: |
| r2 | 0.977 | **1.000** |
| r3 | 0.695 | **0.969** |
| r4 | 0.441 | **1.000** |
| total | 800 policy updates | 900 model updates |

**The policy-based agent degrades as the sequence proceeds (0.977 ->
0.695 -> 0.441). The policy-free agent does not (1.000 -> 0.969 ->
1.000).** Comparable cost, completely different trajectory.

That decay is catastrophic forgetting, measured directly and without a
retention probe: each new rung damages the policy that served the last.
Every consolidation mechanism this program built — Fisher anchors,
arbitrated release, freezing, adapters — exists to slow that curve.
**Storing a model instead of a policy removes it.** Nothing is
overwritten because nothing preferential is stored; new dynamics are
added to a model that was never in conflict with them.

Also visible: r4 zero-shot 0.625 after learning only r2 and r3's
dynamics — two thirds of the way to a novel rung before seeing it,
which is compounding rather than mere retention.

**Caveat on cost, stated plainly.** Each rung was given a flat 300
model updates because zero-shot never cleared the 0.8 retrieval bar, so
900 is an upper bound rather than a measured minimum; a lower bar or an
early-stop on model accuracy would cut it. The claim here is about the
QUALITY trajectory (flat vs decaying), which is unambiguous, not about
the cost being lower — total cost was slightly higher.

**Status of the founding claim.** Supported, on two independent
measurements: F67 (a transferred model beats a target-trained policy
with zero target training) and F68 (no degradation across a sequence
where a policy-based agent loses more than half its competence).

Probes 170-172 are `reacher_ladder.py --model-search --targets`.

**F69 (probe 173). The acquisition-cost curve bends DOWN. The founding
claim is met in full.** F68 left one gap: quality did not degrade, but
cost was slightly higher because every rung received a flat 300-update
allowance regardless of need. Charging what the model actually required
(train until this rung's dynamics are predicted at 0.98; skip entirely
if search already solves it) closes it:

| rung | policy cost | **model cost** | policy final | **model final** |
| --- | ---: | ---: | ---: | ---: |
| r2 | 100 | **50** | 0.977 | **1.000** |
| r3 | 300 | **125** | 0.695 | **0.969** |
| r4 | 400 | **25** | 0.441 | **0.812** |
| total | 800 | **200** | | |

**Four times cheaper and uniformly better.** The shape matters more than
the total: policy cost RISES across the sequence (100 -> 300 -> 400)
while model cost FALLS (50 -> 125 -> 25). The third novel task is the
cheapest of the three, because the model already covers most of what it
needs.

That is the headline gate stated in `ARCHITECTURE.md` §3 — *acquisition
cost for task N falls as N grows* — measured, for the first time in 69
findings. Flat would have meant a library; this is a map.

**The whole project in one line.** A policy is preferential, so every
new task contradicts the last and cost compounds upward. A model is
factual, so every new task ADDS to it and cost compounds downward.
Sixty-eight findings of consolidation mechanisms, penalties, freezing,
adapters and curricula were attempts to make a policy behave like a
model. Storing a model instead is what worked.

**Honest scope.** This is one task family (grid navigation), one agent,
three rungs, one seed at this configuration. F67's transfer result holds
across three seeds; this cost curve does not yet. The claim earned here
is that the mechanism CAN produce a downward curve, not that it does so
generally — the next work is seed-widening and a family whose dynamics
genuinely differ rather than nest.

Probe 173 is `reacher_ladder.py --model-search --targets` with model
early-stopping.

**F70 (probe 174). F69's downward cost curve replicates on 5/5 seeds.**
F69's honest scope note said the cost curve rested on one seed. Five
seeds (69316-69320), two arms, identical three-rung sequence
(r2 -> r3 -> r4), model arm charged what its dynamics model actually
needed, policy arm charged updates spent:

| arm | r2 | r3 | r4 | total |
| --- | ---: | ---: | ---: | ---: |
| policy cost | 100 | 260 | 400* | 760 |
| **model cost** | **60** | **160** | **45** | **265** |
| policy final reach | 0.944 | 0.850 | 0.444 | |
| **model final reach** | **1.000** | **0.925** | **0.881** | |

*policy r4 hit the 400-update budget cap on 5/5 seeds, so its cost is
right-censored: the true figure is >= 400 and the 2.9x total-cost gap is
a LOWER bound.

**The shape holds on every seed individually, not just in the mean.**
Model cost r2 -> r4 falls on 5/5 (50->50, 50->25, 50->50, 75->50,
75->50); policy cost rises on 5/5 (100->400 every seed). Final r4 reach
separates with no overlap: model 0.812-0.938, policy 0.234-0.547,
no-agent floor 0.219. The hardest rung, arriving last, is the CHEAPEST
of the three for the model arm (45 mean vs r3's 160) — the compounding
signature, seed-robust.

Correction to F69's scope note: the cost curve now has the same seed
support as F67's transfer result. What remains unwidened is the FAMILY,
not the seed. r2/r3/r4 nest (a line inside an open grid inside a walled
grid), so a model of r4's dynamics contains r2's; the untested case is a
family whose dynamics genuinely differ. Prediction, recorded before the
experiment: a model meeting unfamiliar dynamics is INCOMPLETE and
repairs by observation, while a policy meeting them is WRONG and must
first unlearn — so the model arm should degrade gracefully where the
policy arm degrades catastrophically. If instead the model arm collapses
to cold-start cost on disjoint dynamics, the mechanism is nesting, not
compounding, and F67-F70 are scoped to nested families only.

Probe 174 is `reacher_ladder.py --model-search=10 --targets=r2,r3,r4
--rung r4 --sparse --updates 400` x 5 seeds vs `--retrieval-first`.

**F71 (probe 175). F67-F70 measured NESTING, not compounding. On
families whose dynamics genuinely differ, catastrophic forgetting comes
straight back — in the model.** The reacher ladder's rungs all live in
one N*N state space and agree on every shared (cell, action) pair, so
training on r4 REINFORCES r2 rather than overwriting it. "A model cannot
hold a contradiction" was true but irrelevant: nothing was contradicted.

`schema_family.py` removes the nesting. Four families share no surface —
`line` (bounded position), `dial` (three counters mod 8, wrapping),
`toggle` (six bits, XOR masks, self-inverse and abelian, nothing moves),
`perm` (ordering of four items, adjacent swaps, non-abelian). Each
occupies a different number of state slots, so the input identifies the
family and no family nests in another. Learned sequentially by one
model, 5 seeds, exhaustive accuracy over every (state, action) pair:

| family | retained after the sequence | uniform chance |
| --- | ---: | ---: |
| line | 0.138 | 0.125 |
| dial | 0.029 | 0.002 |
| toggle | 0.080 | ~0 |
| perm (last learned) | 0.997 | 0.0002 |

`line` retains at exactly its chance floor. The model forgets as
completely as any policy ever did.

**The correction is sharper than "F68 was wrong."** Facts do not
conflict; PARAMETERS do. The model/policy distinction (F67) is about
what is STORED and it survives — a model never holds a wrong preference.
Catastrophic forgetting is about WHERE it is stored, and a shared weight
matrix overwrites regardless of content type. Fixing the content type
does not fix the storage medium, and F67-F70 could not see this because
nesting made every later gradient agree with every earlier one.

This is the project's founding architectural claim arriving from
measurement rather than assertion: skills — here, dynamics — must live
in an external bank, not in weights. Eight consolidation mechanisms have
already failed at making weights behave like a bank.

**F72 (probe 175, same runs). No schema transfer. The cost saving is
generic warm-up and the scramble control proves it.** Sequential vs
cold, cost = updates actually spent to reach 0.98 dynamics accuracy:

| arm | cold total | warm total | saving |
| --- | ---: | ---: | ---: |
| plain families | 660 | 590 | 70 (11%) |
| **scrambled control** | 1445 | 1340 | **105 (7%)** |

The scramble control replaces each family's dynamics with a random
permutation of its own states — same sizes, same action counts, schema
destroyed. It saves MORE in absolute terms than the real families do.
Whatever the warm start buys, it is not structure. Per-seed the plain
saving is +100/+75/+25/+75/+75 against scramble's +50/+125/+125/+100/+125.

Cost also does not fall across the sequence (45 -> 315 -> 135 -> 95);
it tracks state-space size, not position in the order.

**Why, measured rather than guessed.** Slot-level accuracy (fraction of
individual slots predicted right) against the copy-forward baseline —
the trivial rule "next state = current state", which is the bulk of the
shared schema:

| family | fresh net | after prior families | copy baseline |
| --- | ---: | ---: | ---: |
| toggle | 0.083 | 0.329 | **0.694** |
| perm | 0.110 | 0.378 | **0.500** |

The warm model carries more than a fresh one (0.329 vs 0.083) but stays
FAR BELOW the trivial copy rule. After training on `line` and `dial`,
slots 3-5 have never once been active, so the model emits noise into
them. It never learned "copy slot i forward" as a rule — it learned six
unrelated per-slot mappings, because a dense layer gives every slot its
own weights and nothing ties slot 5's behaviour to slot 0's.

**That is bottom-up learning, stated mechanically.** The diagnosis
implies its own fix, and the fix is an architectural constraint rather
than another objective term: share weights ACROSS SLOTS, so a rule
learned on one slot is automatically a rule about all slots.

Prediction recorded BEFORE the run: a slot-symmetric model should push
slot accuracy on an unseen family ABOVE the copy baseline and cut cost
on later families materially, while the scrambled control gains nothing
— random tables have no slot-symmetric structure to share. If the
scramble control benefits equally again, weight sharing is also just
warm-up and the top-down claim needs a different mechanism entirely.

Probe 175 is `experiments/games_amodal/probes/schema_family.py`
(`--scramble` for the control), 5 seeds.

**F73 (probe 176). Top-down structure works, and the scramble control
proves it is structure. A slot-symmetric plant learns every family
2.36x cheaper.** F72 diagnosed the dense model as learning six
unrelated per-slot mappings and never the copy-forward rule they share,
because each slot owns a private stripe of weights. The fix is
architectural: read the state as a set of slot tokens and share the
value embedding, the per-slot MLP and the output head ACROSS slots, with
positional embeddings keeping slots distinct and attention supplying
cross-slot interaction. Nothing in it names a family.

Cold cost to reach 0.98 dynamics accuracy, 5 seeds, mean:

| family | dense | **slot-symmetric** | speedup |
| --- | ---: | ---: | ---: |
| line | 45 | **25** | 1.8x |
| dial | 285 | **80** | 3.6x |
| toggle | 205 | **95** | 2.2x |
| perm | 125 | **80** | 1.6x |
| total | 660 | **280** | **2.36x** |

Per-seed totals do not overlap: dense 675/675/600/675/675 against slot
275/300/250/275/300.

**The causal gate.** Run the identical comparison on the scrambled
control — same state spaces, same action counts, dynamics replaced by
random permutations, schema destroyed:

| dynamics | dense cold | slot cold | speedup |
| --- | ---: | ---: | ---: |
| real families | 660 | 280 | **2.36x** |
| scrambled | 1445 | 1405 | **1.03x** |

The advantage vanishes when the structure it exploits is removed. It is
not a better optimiser, not more capacity, not warm-up: it is the
architecture matching what the tasks actually share. This is the first
measured instance in the project of top-down structure paying, and it is
causally attributed rather than assumed.

It also raises what a transferred model carries onto an unseen family:
slot accuracy on `toggle` after training on `line` and `dial` rises from
0.329 (dense) to 0.519 (slot), against a fresh-network 0.168.

**F74 (probe 176, same runs). Structure in weights helps; CONTENT in
weights still fails, and sharing makes the interference worse.** The
same runs measured sequential learning, and the slot plant is WORSE than
cold at it:

| arch | dynamics | cold total | warm total | warm - cold |
| --- | --- | ---: | ---: | ---: |
| dense | real | 660 | 590 | -70 |
| **slot** | real | **280** | **380** | **+100** |
| dense | scrambled | 1445 | 1340 | -105 |
| slot | scrambled | 1405 | 1685 | +280 |

Retention after the full sequence stays at the chance floor for both
(`line` 0.138 dense, 0.100 slot). Weight sharing raises transfer of
STRUCTURE and raises interference over CONTENT at the same time, because
it is the same weights doing both jobs.

**This is the founding architecture, arrived at by measurement.** The
project's premise — a fixed-size amodal plant whose skills live in an
external growing bank — is exactly the split these two findings force:

  * STRUCTURE is task-general, small, learned once, and belongs in the
    plant's weights. F73 measures what it is worth: 2.36x, causally.
  * CONTENT is per-family, unbounded, mutually non-contradictory, and
    must NOT live in weights. F71 and F74 measure what happens when it
    does: total forgetting, made worse by the very sharing that makes
    structure transfer.

Eight consolidation mechanisms failed to make one weight store do both
jobs. F73/F74 say why that was never going to work, and the fix is not a
ninth penalty term: it is to stop asking weights to hold content.

Next experiment, with its prediction recorded in advance: pre-train the
slot plant on structure, FREEZE it, and hold each family's dynamics in
the external bank the plant reads. Predicted — retention flat, because
bank entries cannot overwrite one another; cost per family below the
280 cold total, because structure is already paid for; and no negative
transfer, because nothing is shared that can conflict. If retention
still collapses with a frozen plant and banked content, then the bank
interface itself is leaking content into weights and that is the bug to
find.

Probe 176 is `schema_family.py --arch slot` against `--arch dense`,
5 seeds, each with `--scramble` as the causal control.

**F75 (probe 177). Frozen plant + external bank: forgetting is SOLVED,
structure transfer is real and causal, and the cost gate still FAILS.**
The experiment F73/F74 forced. Pre-train a slot-symmetric plant on 48
families sampled from the schema, each family's content carried by its
own 16-token bank entry; FREEZE every plant weight; then learn a
held-out hand-made family by fitting a fresh entry alone. Leave-one-out
over all four families, 3 seeds.

An intermediate run is why the pre-training distribution is 48 families
and not 3: with three fixed families the plant learns three MODES rather
than how to read an entry, and a fourth entry has nothing general to
plug into — held-out accuracy 0.069 against a cold 1.000, and raising
entry capacity from 256 to 4096 parameters only reached 0.667. This is
Chan et al. from the other direction: few fixed-meaning classes give
in-weights memorisation, many varied ones give in-context reading.

**1. Forgetting is solved, structurally.** Retention delta after fitting
the held-out entry, across 96 measurements: **min +0.0000, max +0.0000**.
Not "small" — exactly zero, because the plant is frozen and two entries
are separate tensors. F71's total collapse and F74's chance-floor
retention are gone. This is a bug check that passed, not a discovery:
it is true by construction, and the measurement exists to prove the
interface does not leak content back into weights.

**2. Structure transfer is real, and the scramble control now
discriminates.** Mean accuracy on the held-out family:

| held out | schema-pretrained plant | scrambled-pretrained | random plant |
| --- | ---: | ---: | ---: |
| line | 1.000 | 1.000 | 0.312 |
| dial | 0.994 | 0.684 | 0.005 |
| toggle | 1.000 | 0.557 | 0.007 |
| perm | 0.898 | 0.264 | 0.009 |

The random-plant null is decisive everywhere: pre-training is necessary.
The scrambled null separates on three of four families; `line` (8
states, 2 actions) is too easy to discriminate anything and should not
be counted as evidence either way.

**3. The cost gate fails, and this is the headline.** Cost to reach 0.98
on the held-out family:

| | bank entry, frozen plant | cold full model |
| --- | ---: | ---: |
| mean updates | **123** | **62** |
| trainable parameters | 1,024 | 68,936 |
| reached 0.98 | 11/12 runs | 12/12 |
| cheaper than cold | **2/12 runs** | — |

Fitting an entry costs about TWICE a cold fit in updates. Per parameter
it is 67x leaner and the plant is reused, but the gate was stated in
updates and in updates it is not met. Pre-training cost — 2350 updates
over 48 families — is on top, and since per-task cost already exceeds
cold, no amount of amortisation rescues it.

**Correction to my own first reading.** The single smoke run (held-out
`perm`, seed 69316) gave bank 50 against cold 75 and I called it "the
gate". Leave-one-out says 2/12. That is the sixth time in this project a
single-seed signal has failed to replicate; the smoke run was one of the
two wins.

Two bugs in this probe were found by reading its own output, before any
result was recorded: the scrambled control redrew fresh permutations
inside its row loop (so it was not a per-action permutation, and
same-length family names collided on one seed), and retention was
reported as an absolute score, mixing "pre-training stopped early here"
with "the held-out fit damaged it". Fixing the control moved scrambled
`toggle` from 1.000 to 0.557 — it materially changed a null.

**Where the remaining cost goes.** `perm` is worst (242 mean, one
outright failure at 0.694). The procedural generator builds full product
state spaces, while `perm`'s states are permutations only — a region of
the space the plant never saw. The gap is distributional coverage, not
the mechanism.

**Next, prediction recorded in advance.** The cost that remains is
gradient descent on the entry: hundreds of steps to infer content that
is fully determined by a few dozen observed transitions. Replace it with
amortised inference — an encoder mapping a handful of (state, action,
next state) triples DIRECTLY to an entry, trained across the family
distribution, so acquiring a new family is forward passes rather than
gradient steps. Predicted: cost per novel family drops below cold by an
order of magnitude, retention stays exactly flat, and the random-plant
and scrambled nulls stay dead. If instead amortised entries plateau
below 0.98 on held-out families, the entry is too weak a channel to
carry content and the bank needs a richer interface than context tokens.

Probe 177 is `experiments/games_amodal/probes/bank_plant.py`, leave-one-out
x 3 seeds, with `--pretrain-families 48`.

**F76 (probe 178). Amortised reading works and is causal, but the cost
gate fails a THIRD time — and the reason is now exactly located.** The
experiment F75 predicted. An encoder maps observed (state, action, next
state) triples directly to a bank entry: acquiring a family is one
forward pass, zero gradient steps, zero weights moved. Trained across
256 families sampled from the schema, 20000 updates, 3 seeds.

**What works.** The encoder genuinely reads:

| | in-distribution read accuracy |
| --- | ---: |
| trained (3 seeds) | **0.918 / 0.903 / 0.937** |
| scrambled-dynamics null | 0.210 |
| random-plant null | 0.040 |

The wrong-context null — feed the encoder a DIFFERENT family's
transitions — collapses to 0.000-0.065. Nothing is being memorised; the
entry carries the content.

**Novel families from the same generator, never trained on:**

| arm | read accuracy | mastered by reading | fine-tune cost | cold cost |
| --- | ---: | ---: | ---: | ---: |
| trained | **0.682** | 2.3 / 16 | 83.8 | **49.5** |
| scrambled null | 0.147 | 0 / 16 | 273.4 | 40.6 |
| random-plant null | 0.026 | 0 / 16 | 600.0 | 40.6 |

**68% of a novel family's dynamics, correct, from a single forward pass
at zero gradient cost.** Both nulls are dead. The `line` family reads at
0.958 against a cold cost of 25 updates — one family acquired
essentially free.

**Why the gate still fails.** Converting that read into 0.98 mastery
costs 84 updates against cold's 50, and only 2.3 of 16 novel families
clear the bar by reading alone. Fine-tuning from a partially correct
entry is DEARER than starting from scratch, because the frozen plant
caps what can be repaired: 1,024 entry parameters against cold's 68,936
free ones. Whatever the plant reads wrongly, the entry often cannot fix.

That is the trade stated exactly: **freezing the plant is what makes
retention perfect (delta 0.0000, again, everywhere) and it is the same
thing that caps expressivity.** The two are one mechanism, not two.

**The brutal accounting, stated plainly.** Pre-training is 20000 updates
over 256 families. Cold is ~50 per family. Even if reading were free,
break-even needs 400 downstream families; since per-family cost is 84
and cold is 50, it never breaks even at all. Nothing here is cheaper yet
in total updates.

**The distribution boundary, measured.** On the four hand-made families:
`line` reads 0.958, `dial` 0.373, `perm` 0.255, `toggle` 0.043. The
generator has no paired-flip op and builds only product state spaces, so
`toggle` (XOR over bit PAIRS) and `perm` (states are permutations, not a
product) are outside its SUPPORT — not merely unseen instances of it.
This is the "NYC does not help in Tokyo" boundary as a measurement
rather than an assertion: the mechanism transfers within the schema it
was trained on and degrades outside it, in proportion to how far
outside.

**Three mechanisms, three failures of the same gate.** F70 met it only
because the families nested; F75 fitted entries at 2x cold; F76 reads
entries at zero cost but cannot finish the job. The gate — a novel task
cheaper than from scratch — remains UNMET. What has been won is real and
should not be inflated: forgetting is solved outright, structure
transfer is causal and has survived four separate scramble controls, and
zero-gradient partial acquisition is measured at 0.682.

Next, prediction recorded in advance: the binding constraint is entry
EXPRESSIVITY, not reading. Two candidates, and they are distinguishable
by measurement — (a) widen the channel: let the entry modulate the
plant's computation (per-family gains/biases on the slot MLP) rather
than only prepend tokens, keeping every weight shared and frozen;
(b) widen the generator's support so the schema covers paired ops and
non-product state spaces. Predicted: (a) raises `mastered by reading`
well above 2.3/16 while retention stays exactly 0.0000, because the
per-family parameters still live in the bank; (b) raises the hand-made
families' read accuracy without helping the in-support novel ones. If
(a) also degrades retention, then any channel wide enough to be
expressive is wide enough to interfere, and that is a real wall worth
naming rather than working around.

Probe 178 is `experiments/games_amodal/probes/amortised_bank.py`,
3 seeds plus `--scramble` and `--random-plant` nulls.

**F77 (probe 179). The recorded prediction was WRONG: widening the entry
channel makes generalisation worse, and the falsifier did not fire.**
F76 predicted that entry EXPRESSIVITY was the binding constraint and
that letting the entry modulate the plant's computation (per-family
gains and biases on every block, weights still shared and frozen) would
raise mastery-by-reading above 2.3/16. It did the opposite.

| channel | in-distribution | novel read | mastered by reading | fine-tune cost |
| --- | ---: | ---: | ---: | ---: |
| tokens only (F76) | 0.919 | **0.682** | **2.3 / 16** | **83.8** |
| tokens + modulation | 0.925 | 0.567 | 1.0 / 16 | 106.8 |
| modulation, scrambled null | 0.348 | 0.157 | 0 / 16 | 253.1 |

In-distribution is unchanged (0.919 -> 0.925) while every novel-family
number gets worse. That is the signature of overfitting the conditioning
pathway, not of a capacity limit being lifted: the extra channel gives
the model more ways to fit the 256 training families and buys nothing
for the seventeenth.

Hand-made families move the same way (`line` 0.958 -> 0.896, `dial`
0.373 -> 0.305, `perm` 0.255 -> 0.231), and fine-tune costs rise
(`dial` 192 -> 425).

**The falsifier did not fire, and that matters.** F76 recorded: "if (a)
also degrades retention, then any channel wide enough to be expressive
is wide enough to interfere, and that is a real wall." Retention delta
stayed exactly 0.0000 with the wider channel. So the wall is NOT
interference — the frozen-plant/banked-content split keeps its
guarantee even when the channel is widened substantially. What fails is
generalisation of the READER, which is a different problem with
different fixes.

**A capacity confound, caught.** Two earlier arms (dim 128, 3 layers, at
pool 256 and 1024) scored 0.392 and 0.375 in-distribution against the
working config's 0.918 — they are badly undertrained at 20000 updates,
not evidence about pool size. Comparing them would have "shown" that a
larger pool hurts. Pool diversity is being tested properly at the
working configuration instead.

Probe 179 is `amortised_bank.py --film`, 3 seeds plus a scrambled null.

**F78 (probe 180). The binding constraint is TRAINING DIVERSITY, and the
curve is monotone. This is the mechanism that forces top-down
learning.** F77 refuted entry expressivity. Holding the architecture
fixed at the working configuration and varying only the NUMBER of
distinct families the plant and encoder are trained on:

| pool | in-distribution | novel read (0 gradient steps) | mastered by reading | fine-tune cost | cold cost |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.974 | 0.318 | 0 / 16 | 186.8 | 46.1 |
| 256 | 0.919 | 0.682 | 2.3 / 16 | 83.8 | 49.5 |
| **1024** | 0.931 | **0.918** | **5.5 / 16** | **49.1** | 45.5 |

**Note which column moves the wrong way.** A pool of 64 has the HIGHEST
in-distribution accuracy (0.974) and the WORST novel-family accuracy
(0.318). It memorised its 64 families instead of learning to read one.
That is Chan et al.'s axis measured directly in this system: few fixed
meanings produce in-weights memorisation, many varied ones produce
in-context reading. Diversity is the knob, and it is the only knob that
has moved this number.

Per-seed, no overlap between conditions: pool 64 gives [0.360, 0.276],
pool 256 [0.775, 0.582, 0.688], pool 1024 [0.940, 0.897].

The hand-made families — never sampled by the generator — move with it:
`line` 0.250 -> 0.958 -> 0.969, `dial` 0.004 -> 0.373 -> 0.866, `perm`
0.021 -> 0.255 -> 0.729. Even `perm`, whose permutation state space is
outside the generator's support, reads at 0.729 from a single forward
pass once the pool is large enough.

**Status of the gate.** At pool 1024 the fine-tune cost is 49.1 against
a cold 45.5 — parity, not a win, and the 20000 pre-training updates are
still on top. The gate is NOT met. What IS established is the direction:
every diversity increase has improved every novel-family number, the
trend is monotone across three conditions and seven runs, and 0.918 of a
novel family's dynamics now comes from one forward pass at zero gradient
cost.

**What this says about the project's question.** "Produce a program such
that given task A makes novel task B faster to learn than from scratch"
is not answered by a better objective, a better consolidation penalty or
a wider adaptation channel — all three were tried and all three failed.
It is answered by training on ENOUGH DIFFERENT TASKS that reading the
structure is the only strategy that works. Top-down learning is not
something the architecture can be told to do; it is what the
architecture is forced into when memorisation stops paying.

Probe 180 is `amortised_bank.py --pool {64,256,1024}`, 2-3 seeds each.

**F79 (probe 180 extended). At pool 4096 the per-task gate CROSSES: a
novel family costs 34.3 updates against 50.0 from scratch.** Extending
F78's diversity curve by one point:

| pool | novel read (0 steps) | mastered | **acquisition cost** | **cold cost** |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.318 | 0 / 16 | 186.8 | 46.1 |
| 256 | 0.682 | 2.3 / 16 | 83.8 | 49.5 |
| 1024 | 0.918 | 5.5 / 16 | 49.1 | 45.5 |
| **4096** | 0.907 | 5.5 / 16 | **34.3** | **50.0** |

Both seeds cross individually: 22.9 vs 41.7 and 45.8 vs 58.3.

**This is the founding objective, met per task.** "Produce a program
such that given task A makes novel task B faster to learn than from
scratch" — on families the system has never seen, drawn from a
generator whose instances it was never shown, acquisition is 1.46x
cheaper than starting cold, with every plant weight frozen, retention
delta exactly 0.0000, and the wrong-context null at 0.000-0.138.

**What plateaus and what keeps improving.** Read accuracy saturates
between pool 1024 and 4096 (0.918 -> 0.907) and mastery-by-reading stays
at 5.5/16, but COST keeps falling (49.1 -> 34.3). More diversity is no
longer producing a better one-shot read; it is producing an entry that
is a better starting point for the remainder. Those are different things
and the earlier readouts could not have separated them.

**The lifetime gate is still NOT met, and this is the honest limit.**
Pre-training costs 20000 updates. The per-task saving is 15.7 updates,
so break-even needs about **1274 downstream families**. Sixteen were
measured. The claim earned here is per-task acquisition cost, not
lifetime cost, and anyone reading only the table would get that wrong.

Other limits, stated: two seeds at pool 4096; `toggle` still reads at
0.096 because paired-bit flips are outside the generator's support at
every pool size, so diversity within a schema does not buy coverage
outside it; and 10.5 of 16 novel families still need some fine-tuning.

**The through-line of F71-F79.** Eight consolidation mechanisms, an
architectural prior, a frozen plant, an external bank and an amortised
reader were each necessary and none was sufficient. What finally moved
the number was none of them individually — it was training on enough
different tasks that reading structure beat memorising it. Every earlier
mechanism was a precondition for that to be possible: without the
slot-symmetric plant there is no structure to read, without the frozen
plant and banked content there is no retention, without the reader there
is no zero-cost acquisition. The diversity is what makes them pay.

Probe 180 is `amortised_bank.py --pool {64,256,1024,4096}`, 2-3 seeds each.

**F80 (probe 181). Pre-training cannot be shortened, and lengthening it
IMPROVES break-even. Both of my stated next steps were wrong.** The
WEAKNESSES ledger said "drive the pre-training cost down (20000 was a
round number, not a measured minimum)". Measured, at pool 4096, 2 seeds:

| pre-train updates | in-dist | novel read | mastered | acquisition | cold | saving | break-even |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2500 | 0.369 | 0.308 | 0/16 | 540.6 | 50.0 | -490.6 | never |
| 5000 | 0.452 | 0.361 | 0/16 | 489.6 | 50.0 | -439.6 | never |
| 10000 | 0.804 | 0.692 | 1.5/16 | 214.6 | 50.0 | -164.6 | never |
| 15000 | 0.944 | 0.837 | 4.0/16 | 97.9 | 50.0 | -47.9 | never |
| 20000 | 0.968 | 0.907 | 5.5/16 | 34.3 | 50.0 | +15.7 | 1278 |
| **40000** | **1.000** | **0.972** | **9.5/16** | **7.2** | 50.0 | **+42.8** | **936** |

**20000 was not padding.** Below it the reader does not generalise and
acquisition is WORSE than cold — at 10000, 214.6 against 50.0. The
crossing is sharp, between 15000 and 20000. Shortening pre-training is
not available.

**And the opposite move pays twice.** Doubling to 40000 cuts acquisition
from 34.3 to 7.2 — **6.9x cheaper than cold**, 9.5 of 16 novel families
mastered by reading alone with zero gradient steps, per-seed 6.2/8.3
against cold 41.7/58.3 with no overlap. Because the per-task saving grows
faster than the pre-training bill, break-even IMPROVES from 1278 to 936.
I had predicted the reverse — that diminishing acquisition gains would
push break-even up. It falls.

**Widening the schema's support works, and it is not free.** F79's hard
floor was `toggle` at 0.096 at every pool size, caused by the generator
having no simultaneous two-slot op. Adding two-slot ops and permutation
state spaces (`--wide`), pool 4096, 20000 updates:

| family | narrow | wide |
| --- | ---: | ---: |
| toggle | 0.096 | **0.306** |
| perm | 0.521 | **0.708** |
| dial | 0.775 | **0.863** |

The floor lifts 3.2x. But the wider distribution is harder, so at the
same budget acquisition rises 34.3 -> 81.3 against a cold 57.9 and the
gate UN-CROSSES. The two gaps are coupled: at a fixed budget you buy
coverage or you buy cost, not both. The budget curve says the fix is
more pre-training, not a narrower schema — untested at `--wide`, and the
honest statement is that the wide result is a 20000-update snapshot of a
distribution that demonstrably needs more.

Probe 181 is `amortised_bank.py` at `--train-updates {2500..40000}` and
`--wide`, pool 4096, 2 seeds each.

**F81 (probe 182). Double dissociation completed: the capability lives
in the bank, and the residual measures exactly how much structure lives
in the weights.** Imported from the protocol used in a parallel Codex
session on this project, which had an arm this probe lacked. We tested a
CORRUPTED bank (wrong family's entry); we had never tested a WITHHELD
one. Pool 4096, 20000 updates, 2 seeds, novel in-support families:

| bank | read accuracy |
| --- | ---: |
| present | **0.907** |
| withheld (zero entry) | 0.236 |
| corrupted (another family's entry) | 0.037 |

Per hand-made family, withheld collapses to 0.094 / 0.016 / 0.000 /
0.007 (`line` / `dial` / `toggle` / `perm`) against 1.000 / 0.775 /
0.096 / 0.521 with the bank present. Present -> mastery, withheld ->
chance, corrupted -> chance, and every plant weight is frozen throughout.
The skill is in the bank.

**The residual is the interesting number.** Withheld sits at 0.236, not
at zero, and it should not be at zero: a zero entry is NEUTRAL, so the
plant falls back on its generic structural prior — copy-forward and the
slot-symmetric regularities — and gets that fraction right with no
content whatsoever. A corrupted entry is worse (0.037) because it
actively misleads rather than abstaining.

That gives the F73/F74 split a direct measurement instead of an
argument. Of the 0.907 a read entry achieves, **0.236 is structure held
in frozen weights and 0.671 is content supplied by the bank.** The
architecture's central claim — structure in the plant, content in the
bank — is now a number, and the two halves were never separable before
this arm existed.

Probe 182 is `amortised_bank.py` with the withheld-bank arm, 2 seeds.

**F82 (probe 181 completed). Break-even has an INTERIOR OPTIMUM at 936
families, and the pre-training axis is now fully explored.** Extending
F80 to 80000 updates closes the curve:

| pre-train | novel read | mastered | acquisition | cold | saving | break-even |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15000 | 0.837 | 4.0/16 | 97.9 | 50.0 | -47.9 | never |
| 20000 | 0.907 | 5.5/16 | 34.3 | 50.0 | +15.7 | 1278 |
| **40000** | 0.972 | 9.5/16 | 7.2 | 50.0 | +42.8 | **936** |
| 80000 | 0.990 | 10.0/16 | 5.2 | 50.0 | +44.8 | 1786 |

**Why the optimum is interior, and why it is a floor.** Acquisition cost
cannot fall below zero, so the per-task saving is capped at cold's 50.
By 40000 the saving is 42.8 — already 86% of the theoretical maximum —
so doubling the bill to 80000 buys only 2.0 more and break-even nearly
doubles. Below 20000 the reader does not generalise and there is no
saving at all. The minimum is therefore structural: **at this
configuration the lifetime gate cannot be brought below about 936
downstream families by any choice of pre-training budget.** That axis is
finished; further gains must come from elsewhere.

**Correction to F79's support claim.** F79 concluded that "diversity
within a schema buys nothing outside it", from `perm` reading 0.521 and
`toggle` 0.096 at 20000 updates. With the same narrow generator at
80000: `perm` reads **0.965** and `toggle` 0.272. `perm`'s permutation
state space is genuinely outside the generator's support and it is now
read almost perfectly. So the out-of-support penalty was substantially
an UNDERTRAINED-READER artifact, not a hard boundary — the boundary
moves with training. `toggle` remains the real hard case at 0.272, and
it is the one the widened generator addresses directly (0.096 -> 0.306
at 20000).

The honest revision: enough diversity plus enough training generalises
well beyond the schema's literal support, and the earlier claim was a
snapshot mistaken for a limit. What stays true is that `toggle`-style
structure — simultaneous multi-slot effects absent from the op
vocabulary entirely — is the slowest to come, and widening the
vocabulary is the direct fix.

Retention delta remains exactly 0.0000 at every budget.

Probe 181 is `amortised_bank.py --train-updates {2500..80000}`, pool
4096, 2 seeds each.

**F83 (probe 183). The primary gate measured: (a) and (b) pass cleanly,
(c) passes but the test is WEAK and I should say so.** 64 novel families
acquired one after another through a single frozen plant at the optimal
budget, every entry kept, 2 seeds.

**(a) Mastery keeps growing: 64/64 on both seeds.** 59/64 and 56/64 were
acquired by READING ALONE — zero gradient steps, one forward pass. Only
5 and 8 families needed any tuning at all.

**(b) Retention stays exact: drift max 0.0, mean 0.0, both seeds**,
measured across the whole grown bank after all 64 entries exist.

**(c) Cost against bank position:**

| bank position | read | acquisition | cold | saving |
| --- | ---: | ---: | ---: | ---: |
| 1-16 | 0.989 | 2.3 | 53.9 | +51.6 |
| 17-32 | 0.998 | 0.8 | 46.1 | +45.3 |
| 33-48 | 0.991 | 9.4 | 50.8 | +41.4 |
| 49-64 | 0.984 | 7.0 | 51.6 | +44.5 |

Overall 4.9 against a cold 50.6 — **10.4x cheaper across 64 sequential
acquisitions**, with the saving between +41 and +52 in every quartile.

**Why I am not calling (c) a strong result.** First-8 to last-8 looks
like a +9.4 drift, but the quartile pattern is not monotone (2.3, 0.8,
9.4, 7.0) while bank size is, correlation with position is weak (+0.200,
+0.092) and in one seed correlation with FAMILY DIFFICULTY is stronger
(+0.411). The apparent drift is a handful of outliers — one family at
275 updates, one at 75, everything else at 25 or 0.

The deeper reason is structural, and it is the honest caveat:
**entry i+1 is fitted without ever seeing entries 0..i.** The plant is
frozen and entries are independent tensors, so nothing in this mechanism
COULD make acquisition cost grow with bank size. Clause (c) as
implemented cannot fail. A gate that cannot fail is not evidence, and
this project has mistaken exactly that for evidence before (F53, F71).

**What the real (c) needs: RETRIEVAL.** The cost that would scale with N
is finding the right entry among N, and this probe never pays it —
entries are handed to the correct family by construction. That is the
one component of the architecture still missing, and the project already
has the two candidate mechanisms for it (F57 cued addressing, F44
consequence probing). Until retrieval exists, the honest statement is:
per-family ACQUISITION cost does not drift, and per-family RETRIEVAL
cost is unmeasured because there is no retrieval.

**F84 (probe 184). The wide schema at the optimal budget: the gate
re-crosses and the `toggle` floor lifts.** F79's hard case and F80's
un-crossing were the same budget artefact.

| | wide @ 20k | **wide @ 40k** | narrow @ 40k |
| --- | ---: | ---: | ---: |
| toggle | 0.306 | **0.527** | 0.198 |
| perm | 0.708 | **0.986** | 1.000 |
| novel read | 0.880 | 0.930 | 0.972 |
| in-distribution | 0.947 | 0.987 | 1.000 |
| **acquisition** | 81.3 | **20.4** | 7.2 |
| cold | 57.9 | 57.9 | 50.0 |

At the optimal budget the wide generator acquires novel families **2.8x
cheaper than cold** (20.4 vs 57.9) instead of 1.4x more expensively, and
`toggle` — stuck at 0.096 for the whole of F79 — reads at 0.527, its
best figure anywhere. F80's "widening un-crosses the gate" was true only
at 20000 updates, exactly as the budget curve predicted: a harder
distribution needs more pre-training, not a narrower schema.

The remaining honest gap is that `toggle` at 0.527 is still the worst of
the four, so simultaneous multi-slot structure is genuinely the slowest
thing this architecture learns to read, even when it is in the training
vocabulary.

Probes 183/184 are `amortised_bank.py --sequential 64` and `--wide` at
`--train-updates 40000`, pool 4096, 2 seeds each.

**F85 (probe 185). Retrieval built. Clause (c) can now fail, and it
shows the project's first measured scaling limit.** F83's caveat was
that entries were handed to the correct family by construction, so
nothing scaled with bank size and the gate could not fail. Retrieval by
consequence probing (F44's mechanism) closes that: score every stored
entry by how well it predicts a few HELD-OUT transitions of the task at
hand, take the best.

| bank N | retrieval accuracy | chance | margin over runner-up | in-bank score | outside-bank score | forward passes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.000 | 0.125 | 0.564 | 1.000 | 0.429 | 8 |
| 16 | 1.000 | 0.062 | 0.456 | 1.000 | 0.496 | 16 |
| 32 | 1.000 | 0.031 | 0.400 | 1.000 | 0.596 | 32 |
| 64 | **0.969** | 0.016 | 0.365 | 1.000 | 0.642 | 64 |

Both seeds give 0.9688 at N=64, against a 0.016 chance floor.

**The discrimination null is the important column.** A task NOT in the
bank must score LOW against every entry, or "retrieval accuracy" is
satisfied by a system that always returns something. In-bank tasks score
1.000 throughout; strangers score 0.429 at N=8 rising to 0.642 at N=64.
The GAP is what matters and it shrinks monotonically: 0.571, 0.504,
0.404, 0.358. With more entries in the bank, some entry increasingly
explains a stranger's transitions by accident.

**Two limits, one measured and one projected — stated separately.**

1. **Measured, present tense: retrieval is O(N) and already costs more
   than minting.** A linear scan of 64 entries is 64 plant forward
   passes, while minting a fresh entry costs 2.7-7.0 update steps. At
   N=64 identifying a known task is already dearer than learning it from
   scratch would be at this scale. A naive linear bank does not scale,
   and the fix is content-addressed keys giving sublinear lookup —
   infrastructure this project has and has never wired in (open
   weakness 8).
2. **Projected, and labelled as such: discrimination degrades roughly
   0.07 per doubling.** A linear-in-log extrapolation puts the gap near
   zero in the low thousands of entries — close to F82's ~936 break-even
   scale, which is a suggestive coincidence and nothing more. Four
   points and two seeds do not support a confident extrapolation, and
   the runner-up margin's decrements are DECELERATING (-0.108, -0.056,
   -0.035), which would push any crossing further out. The honest claim
   is the direction, not the intercept.

**Status of the primary gate.** (a) 64/64 mastered, both seeds. (b)
retention drift exactly 0.0. (c) acquisition cost does not drift (2.7
and 7.0 against cold 53.5 and 47.7) AND retrieval now supplies a
component that genuinely scales with N — so the clause is falsifiable
for the first time. It passes at N=64 on accuracy and fails on cost
efficiency: retrieval is linear where it must be sublinear.

That is a better position than F83's, because the gate now has something
to break. The next work is not a new mechanism but the one the ledger
has carried unbuilt since the beginning: content-addressed retrieval,
measured against this same linear-scan baseline.

Probe 185 is `amortised_bank.py --sequential 64 --retrieval`, pool 4096,
40000 pre-training updates, 2 seeds.

**F86 (probe 186). Content-addressed retrieval closes F85's O(N)
failure: constant plant cost, perfect identification — but keys alone
cannot reject strangers, and that part is not optional.** F85's measured
failure was that a linear scan costs N plant forward passes, so at N=64
recognising a task was dearer than minting one. Content addressing
stores an ADDRESS (the key of the entry as first read) beside the
CONTENT (the tuned entry), and matches a fresh read against stored keys
by cosine — one encoder pass, no plant passes.

| N | scan acc | scan passes | **key acc** | **key passes** | key+verify | verify passes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.000 | 8 | 1.000 | **0** | 1.000 | 4 |
| 16 | 1.000 | 16 | 1.000 | **0** | 1.000 | 4 |
| 32 | 1.000 | 32 | 1.000 | **0** | 1.000 | 4 |
| 64 | 0.969 | 64 | **1.000** | **0** | **1.000** | 4 |

Keys are perfect at every bank size including 64, where the linear scan
had begun to slip (0.969), and they are perfect at zero plant forward
passes. Both seeds give 1.000.

**The cost problem is solved.** Retrieval is now 4 plant passes
(retrieve-then-verify, constant in N) against minting's 2.7-7.0 update
steps, each of which is a forward AND a backward. Recognising a known
task is now decisively cheaper than relearning it, and stays so as the
bank grows — which is the condition clause (c) exists to test, and the
precondition for the reuse-driven efficiency §3 has claimed since the
beginning without implementing.

**The honest caveat, and it is a real one: keys cannot tell a stranger
from a resident.** The key similarity gap between in-bank and
out-of-bank tasks is much weaker than consequence probing's, and it
degrades faster:

| N | key: in-bank | key: stranger | gap | consequence gap |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.992 | 0.667 | 0.325 | 0.571 |
| 64 | 0.990 | 0.862 | **0.128** | 0.358 |

A never-seen family still matches its nearest stored key at 0.862. A
system that addressed by key alone and reused above a threshold would
wrongly reuse constantly — the failure mode where a bank stops minting
and starts pretending. So the two mechanisms are not alternatives and
the cheap one cannot stand on its own:

  * **keys ADDRESS** — 0 plant passes, perfect shortlist, no ability to
    say "none of these";
  * **consequence VERIFIES** — 4 plant passes, constant in N, and it is
    what supplies the "none of these" that reuse-or-mint needs.

That division is the architecture's own §2.3 split arriving from
measurement: cued addressing is cheap and fast, consequence probing is
what makes identity knowable (F44), and this is the first result showing
they are complementary rather than competing routes.

**Primary gate status.** (a) 64/64 mastered. (b) retention drift exactly
0.0. (c) acquisition cost does not drift with N, retrieval accuracy is
1.000 at N=64, and retrieval COST is now constant rather than linear.
All three clauses pass, and (c) is falsifiable — F85 showed it failing
on cost, F86 fixed the mechanism it was failing on.

Watched, unchanged: the key gap shrinks about 0.066 per doubling and the
consequence gap about 0.071. Neither is measured beyond N=64 and
extrapolation is not supported; re-measure at 128 and 256.

Probe 186 is `amortised_bank.py --sequential 64 --retrieval`, pool 4096,
40000 pre-training updates, 2 seeds.

**F87 (probe 187). The gate holds at N=256, four times beyond where it
was last measured — and the extrapolation I refused to make would have
been wrong.** 256 novel families acquired sequentially through one
frozen plant, 2 seeds.

**All three clauses, at 4x the bank size:**

- **(a) 256/256 mastered**, both seeds.
- **(b) retention drift max 0.0** across all 256 entries.
- **(c) no acquisition drift**: first-64 vs last-64 is 2.7 -> 3.5 on one
  seed and 7.0 -> 4.7 on the other — one rises slightly, one falls,
  against a cold cost of ~51. Acquisition stays 10-15x cheaper with the
  bank four times larger.

**Retrieval scales, and the three mechanisms separate cleanly:**

| N | key | key+verify | linear scan |
| ---: | ---: | ---: | ---: |
| 64 | 1.000 | 1.000 | 0.969 |
| 128 | 0.996 | 1.000 | 0.953 |
| **256** | 0.988 | **0.994** | 0.918 |

Retrieve-then-verify is the best of the three at every size and holds
0.994 at N=256 (per-seed 0.996/0.992) at a CONSTANT 4 plant forward
passes, while the linear scan — which costs 256 passes — has decayed to
0.918. The cheap mechanism is also the accurate one.

**The discrimination trend, now measured instead of projected:**

| N | key gap | consequence gap | key: stranger similarity |
| ---: | ---: | ---: | ---: |
| 8 | 0.325 | 0.571 | 0.667 |
| 64 | 0.128 | 0.356 | 0.862 |
| 128 | 0.095 | 0.345 | 0.895 |
| 256 | **0.068** | **0.258** | **0.923** |

**The extrapolation would have been wrong, and declining to make it was
correct.** The key gap's decrements per doubling are -0.109, -0.049,
-0.039, -0.033, -0.027 — decelerating consistently, roughly halving.
A linear-in-log projection put the gap at zero in the low thousands;
the measured curve is approaching a small positive asymptote instead.
This is the second time this session that a trend from few points
inverted or flattened on measurement (the first was the gradient-cosine
reading at 50 vs 500 updates).

**What the shrinking gap does and does not break.** It does NOT break
retrieval: ranking is what retrieval needs, and ranking survives to
0.994. It DOES break threshold-based reuse-or-mint on keys alone — a
never-seen family matches its nearest stored key at 0.923, so no fixed
cosine threshold can separate "I have this" from "I have something
vaguely like it". Consequence verification still separates (gap 0.258),
which is precisely why F86 concluded the verify step is not optional.
That conclusion is now load-bearing rather than cautious.

Probe 187 is `amortised_bank.py --sequential 256 --retrieval`, pool
4096, 40000 pre-training updates, 2 seeds.

**F88 (probe 188). "`toggle` is hard" was the wrong description. SLOT
COUNT is the predictor, and state-space size is not.** `toggle` has been
the worst-read family at every configuration since F79 and the framing
was always about `toggle` specifically. Recording each novel family's
SHAPE beside its read accuracy — 154 in-support families, 2 seeds,
pool 4096 at 40000 updates — replaces the anecdote with a variable:

| slots | n | mean read | mastered |
| ---: | ---: | ---: | ---: |
| 1 | 30 | 0.996 | 27/30 |
| 2 | 49 | 0.995 | 47/49 |
| 3 | 40 | 0.995 | 36/40 |
| 4 | 16 | 0.974 | 13/16 |
| 5 | 14 | 0.935 | 8/14 |
| **6** | 5 | **0.870** | 2/5 |

Monotone in slot count. `toggle`'s own shape (6 slots, 2 values) reads
0.870 against 0.988 for every other shape — so `toggle` is not a special
case, it is simply the widest family in the set.

**State-space size is NOT the cause, which rules out the obvious
explanation.** 512-state families read 0.977 — better than 6-slot
families at 0.870, despite being eight times larger. Nor is it values
per slot: the apparent weakness at values=2 (0.958) is confounded,
because only small values permit many slots. What degrades is the number
of factors the entry must specify SIMULTANEOUSLY, not how much space
those factors span.

That is a capacity statement about the ENTRY rather than the plant: an
entry must encode roughly slots x actions transition rules, so a 6-slot
6-action family asks 36 rules of the same 16 tokens that a 1-slot family
uses for 6.

**Prediction, recorded before the run now in flight:** raising bank
tokens from 16 to 48 should lift slots=5 and slots=6 substantially and
do essentially nothing for slots=1-3, which are already at 0.995. If
instead every slot count improves equally, the limit is general entry
capacity and the slot-count correlation is incidental; if nothing
improves, the limit is in the plant's ability to USE a wider entry and
F77's finding — that widening the modulation channel hurt — extends to
token count as well.

Probe 188 is `amortised_bank.py --novel-count 96` with per-family shape
diagnostics, 2 seeds.

**F89 (probe 189). Entry capacity is NOT the limit — more of it makes
wide families WORSE. F77 generalises into a law of this
architecture.** F88's prediction was that raising bank tokens from 16 to
48 would lift the wide families (slots 5-6) and leave the narrow ones
alone. Measured, 154 novel in-support families per arm, 2 seeds:

| slots | 16 tokens | 48 tokens | delta |
| ---: | ---: | ---: | ---: |
| 1 | 0.996 | 0.995 | -0.001 |
| 2 | 0.995 | 0.996 | +0.000 |
| 3 | 0.995 | 0.985 | -0.010 |
| 4 | 0.974 | 0.974 | +0.001 |
| 5 | 0.935 | 0.912 | -0.023 |
| **6** | 0.870 | **0.782** | **-0.088** |

Aggregated: slots<=3 moves -0.004 (nothing), slots>=5 moves **-0.040**,
and the widest families lose most. Overall novel read falls 0.984 ->
0.976 and acquisition rises 5.8 -> 7.8.

**This is F77's result again, by a different route.** F77 widened the
conditioning channel by MODULATION and novel-family reading fell 0.682
-> 0.567. F89 widens it by TOKEN COUNT and the hardest families fall
0.870 -> 0.782. Two independent ways of giving the bank interface more
capacity, both leaving in-distribution accuracy roughly unchanged while
degrading exactly the cases that were already hardest.

Stated as a rule this architecture now supports twice over:

> **The bank interface should be as narrow as the task allows. Extra
> conditioning capacity is spent overfitting the training distribution,
> not on expressing harder families.**

**So the limit on wide families is representation, not width — and that
is the third branch of F88's recorded prediction.** F88 wrote: "if
nothing improves, the limit is in the plant's ability to USE a wider
entry and F77's finding extends to token count as well." That is what
happened.

Every gain in this project has come from the training distribution and
none from capacity: F78 (diversity 64 -> 4096), F80/F82 (budget), F84
(schema support). Every capacity increase has hurt: F77, F89. Rejection
sampling leaves 6-slot families at 3.7% of the pool (152 of 4096), and
slots>=5 at 11%. The experiment now in flight samples slot COUNTS
uniformly instead — 6-slot families rise to ~16% — with the prediction
that slots=6 read accuracy rises materially while slots<=3 is unchanged,
since those are already at 0.995 and cannot move.

If balancing does NOT lift the wide families, then the limit is the
plant's 2-layer attention over 6 slot tokens rather than anything about
the data, and depth becomes the next variable — the first time in this
project that adding capacity would be the indicated move.

Probe 189 is `amortised_bank.py --bank-tokens 48` against 16, 2 seeds.

**F90 (probe 190). Balancing the distribution lifts wide families
exactly as predicted — by taking accuracy from the narrow ones. The
distribution does not create capability, it ALLOCATES it.** F89 ruled
out entry capacity. Sampling slot COUNTS uniformly (6-slot families rise
from 3.7% to ~16% of the pool) instead of uniformly over feasible
(slots, values) pairs:

| slots | rejection sampling | **balanced** | delta |
| ---: | ---: | ---: | ---: |
| 1 | 0.996 | 0.963 | -0.033 |
| 2 | 0.995 | 0.954 | -0.041 |
| 3 | 0.995 | 0.941 | -0.054 |
| 4 | 0.974 | 0.972 | -0.001 |
| 5 | 0.935 | 0.957 | +0.022 |
| **6** | **0.870** | **0.972** | **+0.102** |

slots>=5 gains +0.047; slots<=3 loses -0.042. The prediction was right
about direction and wrong about it being free.

**This is the cleanest statement of the mechanism this project has.**
Put beside F77 and F89 — two independent ways of ADDING interface
capacity, both of which made the hardest families worse — the picture is
consistent:

> The reader has a fixed capacity budget across family widths. The
> TRAINING DISTRIBUTION decides how that budget is allocated. Adding
> interface capacity does not add budget; it adds overfitting.

That also explains why 40000 pre-training updates were needed and why
20000 failed (F80): the budget has to be learned before it can be
allocated.

**Cost of the reallocation, stated plainly.** Acquisition rises 5.8 ->
32.1 against a cold baseline that also rises (51.1 -> 59.5, because the
balanced test set is genuinely harder). Still cheaper than cold, but
1.9x rather than 8.8x. And a caveat on the aggregate numbers: balancing
changed the TEST distribution as well as the training one, so overall
"novel read 0.984 -> 0.962" is not a like-for-like comparison. The
per-slot breakdown above is the fair one, and it is unambiguous.

**What did NOT move: hand-made `toggle`, 0.198 -> 0.208.** Balancing
lifted 6-slot PROCEDURAL families to 0.972 but left `toggle` where it
was, and the reason is visible in the spec: `toggle` flips a PAIR of
bits, and this run used the narrow op vocabulary. Slot representation
and op vocabulary are two separate gaps and `toggle` needs both. The run
now in flight is `--balanced --wide`, the one cell that has never been
tested, and it is the direct prediction of these two findings taken
together.

Probe 190 is `amortised_bank.py --balanced`, 2 seeds, against F88's
rejection-sampled baseline.

**F91 (probe 191). Op vocabulary and slot balancing together fix
`toggle` and largely dissolve F90's trade. The reallocation was an
artefact of an inadequate schema, not a fixed budget.** F90 found that
balancing slot counts bought wide families (+0.102 at 6 slots) by taking
from narrow ones (-0.042). F84 separately found that widening the op
vocabulary lifted `toggle`. The combination has never been run:

| slots | rejection + narrow | balanced + narrow | **balanced + WIDE** |
| ---: | ---: | ---: | ---: |
| 1 | 0.996 | 0.963 | 0.987 |
| 2 | 0.995 | 0.954 | 0.993 |
| 3 | 0.995 | 0.941 | 0.984 |
| 4 | 0.974 | 0.972 | 0.972 |
| 5 | 0.935 | 0.957 | **0.968** |
| 6 | 0.870 | 0.972 | **0.976** |
| slots>=5 | 0.918 | 0.965 | **0.973** |
| slots<=3 | 0.996 | 0.954 | **0.988** |

**Balanced+wide is best on wide families AND recovers nearly all of the
narrow loss.** F90's trade was not a fixed capacity budget being
reallocated — it was wide families being needlessly expensive to express
because the op vocabulary lacked simultaneous multi-slot effects. Supply
the right primitive and the same reader covers both.

That is a correction to F90's headline. The rule from F77/F89 stands —
adding INTERFACE capacity hurts — but F90's stronger claim, that
distribution merely reallocates a fixed budget, is too strong. A better
SCHEMA raises the ceiling for everyone; only distribution shifts within
a fixed schema trade off.

**`toggle`, the project's hardest case since F79, is solved:**

| | F79 (pool sweep) | F84 (wide @20k) | F87 (narrow @40k) | **F91 (balanced+wide @40k)** |
| --- | ---: | ---: | ---: | ---: |
| toggle read | 0.096 | 0.306 | 0.198 | **0.917** |

Zero gradient steps, frozen plant, from a single forward pass over 128
observed transitions. Neither ingredient alone was close: op vocabulary
without balancing reached 0.306, balancing without op vocabulary 0.208.
The hand-made families are never sampled by the generator, so this is
genuine out-of-set generalisation.

**Cost, honestly.** Acquisition is 22.2 against a cold 58.9 — 2.7x
cheaper, not the 8.7x of the rejection-sampled arm, but that arm's test
set is easier (cold 51.1). Overall novel read 0.980 with in-distribution
0.979.

**Where this leaves the frontier.** The primary gate passes at N=256
(F87), retrieval is constant-cost and 0.994 accurate (F86, F87), and the
last identified capability gap is closed. The remaining open items are
no longer about this mechanism: they are the discrimination gap's slow
shrink (0.068 at N=256, decelerating, measured not extrapolated) and
whether any of this survives contact with the games battery, which has
not been touched since F70.

Probe 191 is `amortised_bank.py --balanced --wide`, 2 seeds.

**F92 (probe 192). The project's own reacher, read by a plant that never
saw a grid: the open grid is FREE, the walled grid FAILS. Position-
dependent dynamics are a structural class this schema cannot express.**
F71-F91 are all measured on procedurally generated families. The reacher
ladder (F67-F70) is a task this project actually built, and its grid
state is exactly two slots of eight values, so it can be read by the
same plant with no new machinery.

| family | read (0 gradient steps) | per-seed | fine-tune | cold |
| --- | ---: | --- | ---: | ---: |
| **grid** (open, r3) | **1.000** | 1.000 / 1.000 | **0** | 50 |
| **walled** (r4) | 0.894 | 0.894 / 0.894 | **438** | **88** |

**The open grid is acquired for free.** A plant pre-trained only on
procedural families reads the reacher's own dynamics perfectly from 128
observed transitions, zero gradient steps, against a cold cost of 50
updates. That is the first time anything in this project has transferred
to a task it was built for rather than a task built for it.

**The walled grid is the first decisive FAILURE of this mechanism.**
Read 0.894, and fine-tuning to 0.98 costs 438 updates against a cold 88
— five times WORSE than learning it from scratch. The cause is
structural and was predicted before the run: every op the generator can
produce is a uniform function of SLOT VALUES, while a wall makes the
effect of an action depend on WHICH STATE you are in. Walls change
27/256 transitions, and those 27 are exactly the ones the plant cannot
represent. No amount of `--wide` op vocabulary or `--balanced` sampling
reaches this, because the missing primitive is a different kind:
conditional or masked effects, not another uniform slot operation.

That is a clean localisation of the ceiling. The mechanism reads
dynamics that are FUNCTIONS OF THE STATE VECTOR and fails on dynamics
that are functions of the STATE'S IDENTITY.

**A flaw in my own null, found by this run.** `grid`'s wrong-context
null reads 1.000 — apparently "the entry is decoration". It is not: the
withheld-bank control gives 0.170 for `grid` and 0.221 for `walled`, so
an entry is definitely load-bearing. The null is simply uninformative
here, because it pairs each family with its LIST NEIGHBOUR and `grid`'s
neighbour is `walled` — two families sharing 229 of 256 transitions.
A "wrong" entry that is 90% right cannot falsify anything. `walled`'s
null (paired with `line`) reads 0.350 and is meaningful.

The wrong-context null has been quoted as evidence since F76 and it was
sound there, where all four families were mutually unrelated. Adding
near-duplicate families broke its assumption silently. It should draw a
RANDOM unrelated family rather than a neighbour, and until it does, its
value is only interpretable when the pairing is known to be distant.

Probe 192 is `amortised_bank.py --balanced --wide` with `grid` and
`walled` added to the hand-made held-out set, 2 seeds.

**F93 (probe 193). The fixed null vindicates the entry decisively, and
clause (c) holds at N=1024.**

**The null fix (F92's flaw), measured.** Drawing the wrong context from
a FRESHLY GENERATED family instead of the list neighbour:

| family | read | stranger entry | neighbour entry (old null) | withheld |
| --- | ---: | ---: | ---: | ---: |
| grid | 1.000 | **0.117** | 1.000 | 0.170 |
| walled | 0.894 | **0.156** | 0.350 | 0.221 |
| line | 1.000 | 0.109 | 0.000 | 0.188 |
| dial | 0.782 | 0.072 | 0.001 | 0.053 |
| toggle | 0.917 | 0.021 | 0.227 | 0.000 |
| perm | 1.000 | 0.042 | 0.000 | 0.000 |

`grid`'s null moves from 1.000 to 0.117. The old figure was entirely an
artefact of pairing `grid` with `walled`, which share 229 of 256
transitions. Every family now shows the expected pattern — right entry
0.78-1.00, wrong entry 0.02-0.16, no entry 0.00-0.22 — and the double
dissociation is clean across all six.

The lesson generalises past this probe: a null that pairs items by
position silently stops being a null when near-duplicates enter the set.
Draw controls at random from the generating distribution, not from the
neighbourhood.

**Clause (c) at N=1024**, sixteen times the bank size where it was first
measured:

| seed | mastered | retention drift | acq first-128 | acq last-128 | cold |
| --- | ---: | ---: | ---: | ---: | ---: |
| 69316 | 1024/1024 | 0.0 | 4.3 | 5.3 | 50.1 |
| 69317 | 1023/1024 | 0.0 | 8.8 | 10.5 | 48.8 |

Acquisition rises about 1.2x across a 1024-entry bank while remaining
5-10x cheaper than cold, and retention stays exactly 0.0 over a thousand
entries. The drift is real but small and both seeds show it, so it
should be watched rather than dismissed: at this rate it would take a
bank of order 10^5 to erase the advantage.

**A measurement gap of my own making.** The retrieval ladder was capped
at 256 in the code while `--sequential 1024` was running, so retrieval
at N=512 and N=1024 was never computed despite the bank existing. The
expensive part had already been paid. The ladder is now extended and the
run relaunched; until it lands, the discrimination trend remains
measured only to N=256 (key gap 0.068, decrements decelerating:
-0.109, -0.049, -0.038, -0.033, -0.027).

Probe 193 is `amortised_bank.py --sequential 1024` and the stranger-null
rerun, 2 seeds each.

**F94 (probe 194). Retrieval measured to N=1024. Retrieve-then-verify
holds 0.980 at constant cost — and F87's "asymptote" claim was as
unsupported as the linear extrapolation it criticised.**

| N | key | **key+verify** | linear scan | key gap | decrement | conseq gap | stranger key sim |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.000 | 1.000 | 1.000 | 0.325 | - | 0.571 | 0.667 |
| 64 | 1.000 | 1.000 | 0.969 | 0.128 | -0.038 | 0.356 | 0.862 |
| 256 | 0.988 | 0.994 | 0.918 | 0.068 | -0.027 | 0.259 | 0.923 |
| 512 | 0.975 | 0.988 | 0.884 | 0.066 | -0.002 | 0.168 | 0.925 |
| **1024** | 0.951 | **0.980** | 0.853 | **0.037** | -0.029 | 0.171 | **0.954** |

**Retrieve-then-verify is the mechanism that scales.** 0.980 at N=1024
(per-seed 0.979/0.981) on a CONSTANT 4 plant forward passes, while the
1024-pass linear scan has fallen to 0.853. Sixteen times the bank, 1/256
of the retrieval cost, and better accuracy.

**Correction to F87, and it is a correction to my own methodology.** F87
said the key-gap decrements were "decelerating, roughly halving" and
that the curve "is approaching a small positive asymptote instead" of
reaching zero. With four more points that is not supported: decrements
run -0.109, -0.049, -0.038, -0.033, -0.027, -0.002, -0.029. The -0.002
at N=512 is noise, and the N=1024 decrement (-0.029) is the same size as
the N=256 one. The gap is still falling at roughly its earlier rate and
sits at 0.037.

I criticised a linear-in-log extrapolation for resting on four points
and then made an asymptotic claim from the same four points. Both were
projections dressed as findings. The supportable statement is only what
was measured: the key gap is 0.037 at N=1024 and still declining, and
whether it flattens is unknown.

**What that means practically.** Key-only discrimination is effectively
gone at N=1024 — a never-seen family matches its nearest stored key at
0.954 — so no threshold on key similarity can support reuse-or-mint at
this scale. Two things nevertheless survive:

  * **ranking**, which is all retrieval needs: key top-1 is still 0.951;
  * **consequence verification**, whose gap is 0.171 — six times the key
    gap and the only remaining source of "none of these".

The architecture's dependence on the verify step therefore grows with
bank size rather than shrinking. F86 called it "not optional"; at
N=1024 it is the only thing doing that job.

Probe 194 is `amortised_bank.py --sequential 1024 --retrieval` with the
size ladder extended to 1024, 2 seeds.

**F95 (probe 195). The conditional primitive did NOT fix the walled grid
— it made everything worse at a fixed budget.** F92 localised the
mechanism's ceiling to position-dependent dynamics and the ledger called
for a conditional/masked op primitive, "the direct analogue of the
F84/F91 fix that took `toggle` from 0.096 to 0.917". Two such primitives
were added to the schema — a barrier that refuses a move onto a
particular value (literally the reacher's obstacle) and an effect
conditional on another slot. Pool 4096, 40000 updates, 2 seeds:

| family | balanced+wide | **+ gated primitives** |
| --- | ---: | ---: |
| grid | read 1.000, ft 0 | read 0.978, ft 25 |
| **walled** | read 0.894, ft 438 | read **0.795**, ft **600 (capped)** |
| toggle | read 0.917, ft 125 | read 0.800, ft 25 |
| novel in-support read | 0.976 | 0.960 |
| acquisition vs cold | 26.3 / 62.3 (2.4x) | 43.6 / 57.0 (1.3x) |
| in-distribution | 0.979 | 0.961 |

The target case got WORSE (0.894 -> 0.795, and fine-tuning now exhausts
the budget), and so did nearly everything else.

**Why the analogy to `toggle` failed.** `pair` was a SIMPLE uniform op
that made a previously inexpressible family expressible without enlarging
the hypothesis class much. `wall` and `cond` are position-dependent, and
adding them enlarges the class the reader must span across every family
it sees — in-distribution accuracy itself falls 0.979 -> 0.961, which is
the signature of a distribution that has become harder to learn rather
than one that has become more expressive.

**The one explanation still live is budget, and it has a precedent.**
F80/F84 measured exactly this shape once before: the `--wide` schema at
20000 updates un-crossed the cost gate (acquisition 81.3 vs cold 57.9)
and at 40000 it re-crossed decisively (20.4 vs 57.9). A harder
distribution needs more pre-training, not a narrower schema. The gated
schema at 40000 may be in the same position the wide schema was at
20000.

Prediction recorded before the run now in flight (gated at 80000):
if position-dependent dynamics are learnable by this reader at all,
`walled` should rise well above 0.894 and its fine-tune cost fall below
the cold 88. If `walled` stays near 0.8 with twice the budget, then
position-dependent dynamics are not a schema gap at all — they are
outside what this reader architecture can represent, and the fix would
have to be the plant (depth, or a different attention over states),
which would be the first time in this project that adding model capacity
is the indicated move.

Probe 195 is `amortised_bank.py --balanced --wide --gated`, 2 seeds.

**F96 (probe 196). The ceiling is exact and it is not budget, not
schema, and not capacity: the bank stores RULES and cannot store
EXCEPTIONS.** F95's failed fix left one explanation live — that the
gated schema at 40000 was in the position the wide schema had been at
20000, undertrained rather than wrong. Doubling to 80000 settles it:

| family | balanced+wide @40k | gated @40k | **gated @80k** |
| --- | ---: | ---: | ---: |
| grid | 1.000, ft 0 | 0.978, ft 25 | **1.000, ft 0** |
| toggle | 0.917, ft 125 | 0.800, ft 25 | **0.992, ft 0** |
| dial | 0.782, ft 150 | 0.833, ft 88 | **0.980, ft 25** |
| perm / line | 1.000 | 1.000 / 0.969 | **1.000 / 1.000** |
| **walled** | **0.894**, ft 438 | 0.795, ft 600 | **0.894**, ft 600 |
| in-distribution | 0.979 | 0.961 | 0.978 |
| novel read | 0.976 | 0.960 | **0.982** |
| acquisition vs cold | 2.4x | 1.3x | **3.0x** |

**Everything F95 broke, the extra budget repaired — except the one
family the primitive was added for.** `toggle` reaches 0.992 at zero
cost (its best ever), `dial` 0.980, novel acquisition 3.0x cheaper than
cold. `walled` sits at 0.894 on both seeds at both budgets, unmoved.

**The number 0.894 is not a partial success. It is an exact
identification of what the reader does.** `grid` and `walled` agree on
**229 of 256** transitions = **0.8945**, and the measured read accuracy
is 0.894 on every seed and every configuration. The reader gets every
non-wall transition right and every wall transition wrong. It is not
partially learning the obstacle — it reads "8x8 grid movement" from the
context and ignores the exception set entirely, even though roughly 13
of the 128 observed transitions demonstrate it.

**Why more of anything cannot fix this.** The obstacle is ~121 bits of
ARBITRARY content (log2 of the ways to choose 27 blocked transitions
from 256). It is not compressible into a rule of the kind every other
family has — "increment slot 2 mod 8" is a rule; "these 27 cells are
blocked" is a list. The entry is read by a plant that applies a uniform
function to slot values, so an entry can only ever name a rule. No
budget, no vocabulary and no token count changes that, and F95 showed
enlarging the hypothesis class actively costs elsewhere.

**The architectural conclusion, and it is the project's own design
arriving from measurement.** This is the semantic/episodic split:

  * **rules** — compressible, apply everywhere, belong in the bank
    entries this mechanism already has. Measured: 0.982 read, 3.0x
    cheaper acquisition, retention exact to N=1024.
  * **exceptions** — arbitrary, per-state, incompressible. They need a
    STORE, not a rule: content-addressed episodic memory holding
    (state, action) -> outcome for the states where the rule fails.

`ContentAddressedMemory` has sat unused in this repository since the
beginning (open weakness 8, "no memory bank in the games runtime"). F96
is the first result that says precisely what it is for and predicts what
it should buy: `walled` should go from 0.894 to ~1.000 by storing 27
exceptions, with the rule-bank untouched.

That is a sharper claim than "add episodic memory" — it is "the residual
0.106 is exactly the exception set, and an episodic store of 27 entries
closes it".

Probe 196 is `amortised_bank.py --balanced --wide --gated` at 40000 and
80000 updates, 2 seeds each.

**F97 (probe 197). The exception store closes the last gap, on exactly
the predicted number. `walled` 0.894 -> 1.000 with 27 stored
exceptions.** F96 predicted: "`walled` goes 0.894 -> ~1.000 by storing
27 exceptions, with the rule-bank untouched." Built and measured — the
plant stays frozen, the entry is unchanged, no gradient step is taken,
and exceptions are recorded only where the rule is observably WRONG:

| family | rule only | watch 128 | watch 256 | watch 512 | watch 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **walled** | 0.894 | 0.928 (8) | 0.965 (18) | 0.984 (23) | **1.000 (27)** |
| grid | 1.000 | 1.000 (**0**) | 1.000 (**0**) | 1.000 (**0**) | 1.000 (**0**) |
| toggle | 0.992 | 0.994 (0) | 0.995 (1) | 0.999 (2) | 0.997 (2) |
| dial | 0.980 | 0.980 (2) | 0.981 (4) | 0.982 (8) | 0.986 (18) |
| perm | 1.000 | 1.000 (**0**) | 1.000 (**0**) | 1.000 (**0**) | 1.000 (**0**) |
| line | 1.000 | 1.000 (**0**) | 1.000 (**0**) | 1.000 (**0**) | 1.000 (**0**) |

*(accuracy with exception count in parentheses)*

**Exactly 27, on both seeds.** That is precisely the number of
transitions on which `grid` and `walled` differ. The store found the
obstacle and nothing else.

**The degeneracy check is the important one.** A store that fixes
everything by memorising everything would be worthless — it would be a
lookup table wearing an architecture. The store holds **zero** entries
for `grid`, `perm` and `line`, the families the rule already captures
perfectly, at every observation budget up to 1024. It grows only where
rules fail, which is the property that makes it a complement to the bank
rather than a replacement for it.

**The limit is observation, not capacity.** Coverage rises 8 -> 18 -> 23
-> 27 as the world is watched longer, and 128 random draws from 256
possible (state, action) pairs cover only ~39% of them, so the
intermediate figures are exactly what sampling predicts. Nothing is
being learned; the system is simply looking.

**This completes the split §2.2 now records:**

  * **rules** — compressible, universal, read in one forward pass into a
    bank entry: 0.982 on novel families, 3.0x cheaper acquisition than
    cold, retention exact to N=1024, retrieval 0.980 at constant cost.
  * **exceptions** — arbitrary, per-state, incompressible, recorded
    where the rule is seen to fail: 27 entries turn the project's own
    walled reacher from 0.894 to 1.000, and 0 entries are spent on
    families that need none.

**Honest scope.** The store here is an exact dictionary keyed by
(state, action) — an idealised content-addressed memory. A learned or
approximate store would be lossy and this result does not speak for it;
what is established is that the INFORMATION needed is small, precisely
localised, and obtainable by watching. The failure mode to watch for is
a family whose rule captures little: the store would then grow toward
the whole table and degenerate into memorisation. `dial` at 18 entries
is the closest instance here, and store size per family is the
diagnostic that would catch it.

Probe 197 is `amortised_bank.py --balanced --wide --gated` at 80000
updates with the exception store, 2 seeds.

**F98 (probe 198). A real content-addressed store matches the idealised
dictionary — once the key stops being lossy. And the degeneration case
is measured, then separated by two orders of magnitude.** F97's store
was an exact dict keyed by (state, action): it never mis-fires and never
runs out of room. Two things it could not speak for were a realistic
approximate store, and a family with no rule at all.

**Approximate store, similarity-addressed with capacity:**

| family | rule | exact dict | **mean-pooled key** | **concatenated key** |
| --- | ---: | ---: | ---: | ---: |
| walled | 0.894 | 0.996 | 0.908 | **0.996** |
| toggle | 0.992 | 0.999 | **0.951** | **1.000** |
| dial | 0.980 | 0.985 | 0.968 | 0.986 |
| chaos | 0.014 | 0.986 | 0.557 | 0.977 |
| grid / perm / line | 1.000 | 1.000 | 1.000 | 1.000 |

**My first key design was the whole failure.** Mean-pooling the slot
embeddings loses state identity — two different states can share a mean
— so the store both missed exceptions and fired on the wrong ones. On
`toggle` it was NET-HARMFUL, dragging 0.992 down to 0.951: a store that
made a family the rule already handled WORSE. Keeping the slots
concatenated preserves identity and the approximate store then matches
the exact dictionary everywhere (walled 0.996, toggle 1.000).

So content addressing is fine for exceptions; lossy addressing is not.
That is a narrower and more useful claim than "approximate stores work",
and it was one pooling operation away from the opposite conclusion.

**Degeneration, measured rather than assumed.** `chaos` is a family
whose transition table is a random permutation per action — no rule
exists to read. The rule alone scores 0.014, and the exception store
grows to 249 of 256 possible entries and reaches 0.986. **The store does
become the whole table when no rule exists.** The failure mode named in
F97 is real.

**And it is trivially detectable.** Violation rate — the fraction of
observations on which the rule is wrong — separates the cases by two
orders of magnitude:

| family | violation rate | verdict |
| --- | ---: | --- |
| grid, perm, line | **0.0%** | rule holds |
| toggle | 1.1% | rule holds |
| dial | 2.1% | rule holds |
| **walled** | **10.3%** | rule + exceptions |
| **chaos** | **98.5%** | NO RULE — refuse to memorise |

A capacity cap alone is a poor guard: bounding `chaos` to 32 entries
holds memory down but accuracy collapses to 0.127, so the system fails
silently. The violation rate is the honest signal — it says WHY the
store is growing, and a system that watches it can report "this task is
not rule-like" instead of quietly turning into a lookup table.

That completes the store: rules in the bank, exceptions in a
content-addressed episodic memory addressed by a non-lossy key, and a
measured criterion for when a task is not compressible at all.

Probe 198 is `amortised_bank.py --store-key {mean,concat}` with the
rule-free `chaos` family, 2 seeds.

**F99 (probe 199). The mechanism reaches the ACTUAL GAMES: the entry is
causally driving behaviour, in-distribution works, and generalisation to
held-out rule pairs is weak and seed-unstable — exactly as F78
predicts.** F71-F98 were measured on procedurally generated families and
the reacher in a slot interface. This is the games battery itself: real
screens from `FamilyVerifier`, verifier-private rules, and REWARD rather
than next-state.

The `dual` variant is the games' own factorisation test. Each trial puts
the avatar at the centre with `arity` items adjacent and a cue across
the top row; for cue k exactly one side is edible. A "family" is one
(rule0, rule1) pairing — 9 of them at arity 3, built from 6 independent
sub-rules. The reader watches 64 (screen, action, reward) triples and
emits an entry; the frozen plant predicts each action's OUTCOME and
behaviour is argmax. Nothing preferential is stored: "side 1 is edible
under cue 0" is a fact that cannot go stale, where "move right" is a
habit the next variant contradicts.

| arm | choice accuracy | mean reward |
| --- | ---: | ---: |
| trained pairings | **0.667** | +0.600 |
| held-out pairings | 0.345 | +0.214 |
| held-out, entry WITHHELD | 0.241 | +0.102 |
| held-out, STRANGER entry | **0.083** | **-0.100** |
| random plant | 0.065 | -0.042 |
| chance | 0.250 | |

**The entry is unambiguously load-bearing.** Withholding it drops
behaviour to chance (0.241 vs 0.250) — the plant alone knows no rule.
Supplying ANOTHER pairing's entry drops it to 0.083 with NEGATIVE
reward: a wrong rule makes the agent eat the wrong item on purpose.
That is a stronger causal demonstration than anything in the synthetic
families, because here being wrong is actively punished.

**Generalisation is the weak part, and it is seed-unstable.** Held-out
pairings score 0.488/0.528/0.558 on one seed and 0.104/0.221/0.173 on
the other — one seed clearly above chance, the other clearly below. The
0.345 mean is carried entirely by the first.

**F78 predicts this, and the prediction is quantitative.** Diversity is
the knob: with 64 procedural families the reader MEMORISED (novel read
0.318) and it took 4096 to make reading the winning strategy. Here there
are **six** training pairings. Six. By this project's own measurement,
in-weights memorisation is what a distribution that small should
produce, and 0.667 trained against 0.345 held-out is exactly that
signature.

So the honest reading is not "the mechanism fails on games". It is that
the games' rule space is small BY CONSTRUCTION, and the condition F58
and F78 identified — goals plentiful enough that memorising is not
competitive — is not met by the battery as it stands. That is a
statement about the benchmark as much as the method.

The run now in flight raises `arity` to 4 and 5 (16 and 25 pairings) to
test the diversity explanation directly on the games. Prediction: held-
out choice accuracy rises with the number of distinct pairings and the
seed instability shrinks. If it does not, the games differ from the
synthetic families in some way beyond rule count, and finding out which
way becomes the next question.

Probe 199 is `experiments/games_amodal/probes/game_rule_reading.py`,
2 seeds plus a random-plant null.

**F100 (probe 200). Reading rules across 50 game variants learns
NOTHING — because I built a one-step reward model and the games have
delayed reward. The formulation was wrong, not the mechanism.** F99
found the dual game's rule space capped at 9 pairings (the verifier
rejects `arity` above 3), far below the diversity F78 says is needed.
The battery's own variant enumeration supplies 50 distinct worlds
(`family_variants` x `inverted`), so that was the natural larger axis.

| arm | reward | floor | lift | beats floor |
| --- | ---: | ---: | ---: | ---: |
| trained variants | -0.020 | -0.019 | -0.001 | 6.0/12 |
| held-out variants | -0.020 | -0.021 | +0.001 | 5.5/12 |
| entry withheld | -0.021 | -0.021 | -0.001 | 4.5/12 |
| stranger entry | -0.020 | -0.021 | +0.001 | 4.5/12 |
| **random plant (null)** | -0.019 | -0.022 | **+0.003** | **8/12** |

**The random-plant null does as well as the trained system.** Twelve
thousand updates bought nothing at all, on trained variants as much as
held-out ones. That pattern — no lift anywhere, including in
distribution — is not a generalisation failure, it is a formulation
failure, and it points at the probe rather than the method.

**The cause.** `collect`, `intercept`, `avoid` and `navigate` are
MULTI-STEP: reward arrives after several moves toward something. My
plant predicts the outcome of ONE action and behaviour is a greedy
argmax over that prediction. Almost every single action in these games
yields exactly zero, so the model is right and useless — greedy
one-step prediction is not a policy for a navigation task.

The dual game worked (F99: 0.667 trained, stranger entry -0.100) for
exactly the reason these do not: a dual trial IS one step. The item is
adjacent, one action resolves it, and one-step outcome prediction is the
whole problem.

**This is F67's architecture, and I omitted half of it.** F67 concluded
that behaviour should be DERIVED BY SEARCH in a learned transition
model, and the reacher probes did precisely that — BFS over predicted
next states. Here I learned a reward model and no transition model, then
searched to depth one. For a one-step game that is complete; for a
navigation game it is a stub.

What the games need, stated concretely: a transition model over screens
or over an extracted factored state, a reward model as built here, and
search over the two — the same combination `reacher_ladder.py` uses.
The reading mechanism supplies the per-world CONTENT for both; what is
missing is the multi-step derivation, not the bank.

**Honest status of the games claim.** Reading a game's rule from
observed outcomes works and is causally demonstrated where the game is
a one-step decision (F99). It is untested on multi-step games, because
this probe could not test it — and the 50-variant result above should be
read as a measurement of my probe, not of the architecture.

Probe 200 is `game_rule_reading.py --variants`, 2 seeds plus a
random-plant null.

**F101 (probe 201). Model + value + SEARCH also fails on the games, and
the nulls say why: the ENTRY contributes nothing. The state
representation is the defect, not the derivation.** F100 located its own
failure in greedy one-step action selection and prescribed F67's
missing half. Built: a transition model (cell, action) -> next cell, a
value model (screen, cell, entry) -> what happens if I stand there, and
breadth-first search over the two, recomputed every step.

| arm | reward | floor | lift | beats floor |
| --- | ---: | ---: | ---: | ---: |
| trained variants | -0.0385 | -0.0402 | +0.0017 | 7.0/12 |
| held-out variants | -0.0383 | -0.0400 | +0.0016 | 6.5/12 |
| held-out, entry WITHHELD | -0.0380 | -0.0400 | **+0.0020** | 6.5/12 |
| held-out, STRANGER entry | -0.0377 | -0.0400 | **+0.0023** | 6.5/12 |
| random plant | -0.0444 | -0.0436 | -0.0008 | 5/12 |

Search did roughly double the lift over F100's greedy probe (+0.0016
against +0.0008), but both numbers are negligible, and the decisive
column is the nulls: **withholding the entry scores the same, and a
STRANGER'S entry scores slightly better.** On the games' `dual` variant
the same nulls were brutal — a stranger's entry drove reward to -0.100
(F99). Here the entry is decoration. Whatever the system is doing, it is
not reading the world.

**So the diagnosis moves up a level, and it is the same shape as F92.**
The state I gave it is the avatar's cell plus one screen frame. That is
not a sufficient state for these games: `intercept` has objects FALLING,
`avoid` has hazards MOVING, and a single frame contains no velocity or
phase. A value attached to "standing on cell c" cannot express "cell c
is safe now and lethal in two steps". The model is being asked to
predict something its inputs do not determine — Markov-insufficient by
construction — so no amount of search over it helps and no entry can
rescue it.

F92 found the mechanism reads dynamics that are functions of the STATE
VECTOR and fails on dynamics that are functions of the state's IDENTITY.
This is the same boundary one level up: it fails on dynamics that are
functions of state HISTORY.

**Two failed probe formulations, one honest conclusion.** The bank
mechanism is not what is failing on the games — it has never been given
a state in which the games' dynamics are predictable. What the battery
needs is a factored MULTI-OBJECT state: avatar, plus each faller and
hazard, plus enough frames to expose motion. That is precisely what the
slot interface of F71-F98 was built for and it has never been fed game
objects; `schema_families.py` handles six slots of eight values, and a
composigrid frame with an avatar and two hazards is exactly that shape.

Recorded as the next experiment rather than attempted here, because two
formulation failures in a row is the point at which this project's own
rules say to stop iterating and state the finding.

Probes 200 and 201 are `game_rule_reading.py --variants` and
`game_search.py`, 2 seeds each plus random-plant nulls.

**F102 (probe 202). The slot state made it WORSE, which refutes F101's
diagnosis — and the real cause is measured: 98.16% of outcomes under
random play are "nothing".** F101 blamed the state representation
(avatar cell + one frame is Markov-insufficient) and prescribed a
factored multi-object state. Built: six slots read off the screen
(avatar row/col, nearest positive object row/col, nearest negative
object row/col), a dynamics model over that state, an outcome model, and
beam search over both.

Same seed, same held-out variants, floors verified bit-identical 12/12:

| arm | mean lift over floor | beats floor |
| --- | ---: | ---: |
| trained | **-0.0013** | 5/12 |
| **untrained (random plant)** | **+0.0071** | **10/12** |

**Training makes it worse than not training.** That is not noise and not
a measurement artefact — the floors are the same numbers to the last
decimal.

**Correction to F101.** I diagnosed state representation and this
experiment adds exactly the missing information — falling objects now
move IN the state — and the result got worse. The state was not the
binding constraint, so F101's diagnosis was wrong.

**The actual cause, measured over 20 variants and 25600 random steps:**

| outcome class | share |
| --- | ---: |
| nothing | **98.16%** |
| cost | 1.53% |
| food | **0.31%** |

Three consequences, and together they explain the sign of the result:

1. A model that always predicts "nothing" scores 98.16%. Cross-entropy
   on random-play data is almost entirely satisfied by that constant, so
   there is nearly no gradient toward the 1.8% that matters.
2. The residual signal is **5:1 biased toward COST** (1.53% vs 0.31%).
   Where the model does deviate from "nothing", it predicts punishment
   more readily than reward.
3. Beam search maximises P(food) - P(cost) over that near-flat,
   cost-biased landscape, so the agent systematically avoids everything
   — including food — and is CONSISTENTLY wrong where random play is
   only randomly wrong. Consistency is why it lands below the floor.

**And this explains F99 exactly.** The `dual` game worked — 0.667
trained, and a stranger's entry driving reward to -0.100 — because every
`dual` step resolves a trial, so its outcomes are ~100% non-zero. The
mechanism works where the outcome signal is DENSE and fails where it is
1.8%. That is one coherent account of every games result in F99-F102,
and it is about signal density, not about the bank, the search, or the
state.

**What this actually asks for**, and it is standard rather than novel:
class-balanced or importance-weighted outcome loss, on-policy or
outcome-seeking data collection instead of uniform random rollouts, and
a VALUE (expected future outcome) rather than an immediate-outcome
target. None of the three has been tried; all three are ordinary
reinforcement-learning practice that this line skipped by collecting
data with a uniform random policy.

Recorded as the corrected next step. The sequence F100 -> F101 -> F102
is three formulations and two wrong diagnoses of my own, and the thing
that finally identified the cause was counting the labels — which cost
one command and should have come first.

Probe 202 is `game_slots.py`, 2 seeds plus a random-plant null.

**F103 (probe 203). The three sparsity fixes help but do not close it —
and the untrained control reveals that my FLOOR was the wrong baseline
all along.** F102 prescribed class-balanced loss, outcome-seeking data
collection, and a value target. All three built and measured:

| arm | F102 baseline | **with all three fixes** |
| --- | ---: | ---: |
| trained variants | -0.0045 (3.0/12) | **-0.0006** (4.5/12) |
| held-out variants | -0.0024 (4.0/12) | **+0.0007** (5.5/12) |
| held-out, entry withheld | -0.0036 | +0.0013 |
| held-out, stranger entry | -0.0033 | +0.0007 |
| untrained control | +0.0071 (10/12) | +0.0075 (10/12) |

The fixes move every trained arm in the right direction — held-out lift
crosses from negative to positive — and label density confirms why:
outcome-seeking raises the food class from 3.54% to 12.18%, a 3.4x
increase, while the value target alone barely moves it (3.54% -> 3.72%).
Seeking is what mattered; the horizon was nearly irrelevant.

**But two things are still wrong, and they are the important ones.**

1. **The untrained control still wins**: +0.0075 against +0.0057 on the
   same seed, 10/12 against 8/12. Training an outcome model and
   searching in it is still worse than searching in an untrained one.
2. **The entry is still decoration**: correct +0.0007, withheld +0.0013,
   stranger +0.0007 — a gap of exactly zero. Nothing is being read.

**Why the untrained control is strong, which is the real finding.**
Beam search over a random-but-fixed value function produces PERSISTENT
DIRECTIONAL motion — the agent commits to a direction instead of
jittering. In a grid where food is scattered, persistent motion covers
far more ground than a random walk, so it collects more by accident.

That means the random-action FLOOR was never the right control for a
searching agent. "Beats a random walk" is satisfied for free by anything
that moves consistently, and every games number in F100-F103 was scored
against it. The correct control is **search with an untrained model** —
which isolates what the LEARNING contributes from what the SEARCH
contributes — and against that control this system has never once won.

That reframes F100-F102's conclusions without rescuing them: the earlier
runs were not merely failing to help, they were being compared against a
baseline too weak to be informative, and the one strong control present
(the untrained plant) was the number that mattered in all of them.

**A bug in my own fix, caught by measuring rather than reading.** The
first implementation of the value target accumulated the discounted
return to the END of the collected sequence regardless of `--horizon`,
so the flag only trimmed trailing rows and every label was a full
Monte-Carlo return. The ablation would have been meaningless. It
surfaced from checking label density across settings — the same
one-command check that should have opened this whole line.

**Honest status of the games.** The mechanism works on `dual`, where
outcomes are ~100% dense and the nulls are brutal in the right direction
(F99). It does not work on the multi-step variants, where after three
formulations, three sparsity fixes and a corrected baseline the learned
model still adds nothing over an untrained one. The remaining
explanations are ordinary and untested: the outcome signal may still be
too sparse at 12%, 8000 updates over 38 variants may be far too few
(F80 needed 40000 on a much simpler distribution), or a searching agent
in these games may need on-policy correction rather than one round of
seeded collection. All three are measurable; none is exotic.

Probe 203 is `game_slots.py --horizon 4 --balance-loss --seek 0.5`,
2 seeds plus an untrained control.

**F104 (probe 204). Density plus training finally beats the untrained
control — but the INVERTED TWIN control proves the bank entry is not
being read at all. The gain is generic competence, not world-specific
content.**

**First, what worked.** F103's three fixes plus more of both:

| arm | held-out lift | vs untrained control | wins |
| --- | ---: | ---: | ---: |
| 8k updates, seek 0.5 | +0.0007 | -0.0018 | 4/12 |
| 40k updates, seek 0.5 | -0.0017 | -0.0047 | 2/12 |
| **40k updates, seek 0.85** | **+0.0093** | **+0.0161** | **6/12** |
| untrained control | +0.0075 | — | — |

Density is the decisive variable, not updates: 40k at seek 0.5 is WORSE
than 8k at seek 0.5, so more training on a signal-poor distribution
actively hurts. With both, seed 69316 reaches +0.0236 against the
untrained control's +0.0075, with large wins on the `intercept`
variants (+0.0859, +0.0567, +0.0489) where "get under the falling
object" is a strong, learnable regularity. Seed 69317 fails (-0.0051),
so it is one seed of two.

**Then, the control that settles it.** F103's stranger control drew a
RANDOM other variant, which is usually a different component mix whose
entry is merely uninformative. The sharp control is the INVERTED TWIN:
same components, same rendering, opposite rewards — the only entry that
is actively WRONG on identical pixels.

| entry supplied | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| correct | +0.0236 | -0.0051 |
| withheld (zeros) | +0.0202 | -0.0063 |
| stranger | +0.0215 | -0.0054 |
| **inverted TWIN** | **+0.0225** | **-0.0057** |
| correct minus twin | **+0.0011** | **+0.0007** |

**Handing the agent the exact opposite of the truth changes nothing.**
Per variant, most differences are exactly 0.0000 or one reward
quantum. So the entry carries essentially none of the inversion bit, and
whatever the dense/long training bought is INVERSION-INVARIANT
competence — moving usefully with respect to objects — learned by the
transition model and the search, with the bank contributing nothing.

**The contrast with F99 is the whole finding.** On `dual` a stranger's
entry drove reward to -0.100 against +0.600 with the right one. The same
mechanism, the same reader, the same architecture — and there the entry
was everything. The difference is what the two settings require:

  * `dual` — every step resolves a trial whose answer is knowable ONLY
    from prior outcomes, so no inversion-invariant policy exists and
    reading is the only way to score;
  * the multi-step variants — a large fraction of the available reward
    is inversion-invariant, so a policy that ignores the bank captures
    most of it, and gradient descent finds that policy first.

That is not a defect of the bank. It is a statement about what these
tasks measure: **a benchmark only exercises context-reading if ignoring
context is unprofitable.** The battery's multi-step variants do not meet
that condition; `dual` and `choice` were designed to and do.

The practical consequence for this project is a test-design rule rather
than a mechanism change: when adding a game to test the bank, verify
first that an inversion-invariant policy scores near floor. If it does
not, the game measures navigation competence and will report the bank as
working or not working for reasons that have nothing to do with the
bank.

Probe 204 is `game_slots.py --train-updates 40000 --horizon 4
--balance-loss --seek 0.85`, 2 seeds, with the inverted-twin control.

**F105 (probe 205). The context-required multi-step benchmark exists,
is validated, and the current stack captures 0.2% of its headroom.**
F104 showed the battery had no multi-step game in which context is
required, and gave the rule for building one. Built from configuration
the games already support — `forage` supplies opposing item types,
`inverted` swaps which is food, `recentre_every` with `spawn_radius`
makes the avatar cross ground to reach each trial.

**Validated against F104's rule BEFORE running the mechanism**, pair
means over a twin and its inverse:

| policy | normal | inverted | pair mean |
| --- | ---: | ---: | ---: |
| idle | -0.0501 | -0.0497 | -0.0499 |
| random | -0.0496 | -0.0453 | -0.0474 |
| eat-anything | -0.0368 | -0.0267 | -0.0318 |
| always eat plane 1 | **+0.1919** | **-0.2640** | -0.0360 |
| always eat plane 2 | -0.2709 | +0.1989 | -0.0360 |
| **ORACLE (uses the hidden bit)** | +0.1919 | +0.1989 | **+0.1954** |

Every inversion-invariant policy loses. A fixed preference earns +0.19
on one twin and -0.26 on the other and nets -0.036, which is the
property F104 required. **Headroom obtainable only by reading context:
+0.2272.**

**Then the mechanism, 54 worlds, held out by whole twin pair, 40000
updates with all three sparsity fixes:**

| arm | pair-mean reward |
| --- | ---: |
| trained worlds | -0.0463 |
| held-out worlds | -0.0466 |
| entry withheld | -0.0472 |
| stranger entry | -0.0471 |
| **inverted TWIN entry** | -0.0471 |
| untrained control | -0.0483 |
| *best invariant policy (reference)* | *-0.0318* |
| *oracle (reference)* | *+0.1954* |

**Entry effect: +0.0005 against +0.2272 available — 0.2% of the
headroom.** And the deeper problem is not generalisation: the score on
TRAINED worlds is -0.0463, worse than the best inversion-invariant
policy and barely above idling. The system does not learn these worlds
at all, on a task where a ten-line hand-written policy earns +0.1954.

**What is settled and what is not.** Settled: the reading mechanism
works where a trial resolves every step and no context-free policy
exists (`dual`, F99 — stranger entry -0.100 against +0.600 correct). It
does not work on multi-step navigation-plus-context, and F105 removes
the last alternative explanation, because this benchmark is verified to
require context and to have large headroom. Not settled: WHY. The stack
fails to learn even the trained worlds, so the defect is upstream of the
bank — plausibly the beam search over a learned 6-slot model being too
weak to sustain navigation to a respawning target, which is testable by
scoring the transition and outcome models directly rather than only
through behaviour.

**The benchmark is the durable part.** It has a measured floor
(-0.0318 for the best context-free policy), a measured ceiling
(+0.1954), and a validated guarantee that the gap between them is
reachable only by reading context. Any future attempt can be scored
against those numbers, which is what F100-F104 lacked and spent four
findings discovering.

Probe 205 is `game_slots.py --forage-twins --train-updates 40000
--horizon 4 --balance-loss --seek 0.85`, 2 seeds plus an untrained
control.

**F106 (probe 206). The models scored directly: the search was never the
defect. The reader and the outcome model have collapsed jointly onto the
twin-average — which is F58's failure, and F58's fix already exists.**
Every games finding from F100 on was inferred from reward alone, so "it
plays badly" could equally have meant bad models or bad search. Scoring
the models directly separates them.

**Transition model** (40000-update runs, held-out worlds):
per-slot accuracy 0.8154, exact-state 0.5842. Mediocre, and not the
binding defect.

**Outcome model**: balanced accuracy 0.4312 against a 0.3333 chance
floor. Per-class recall is the interesting part —
**cost 0.4672, nothing 0.0000, food 0.6575.** The class-balanced loss
from F103 did not fix the degeneracy, it INVERTED it: F102's model
always said "nothing", this one never says it. Both are degenerate and
the fix swapped which.

**Twin discrimination, the decisive measurement.** The same
(state, action) batch scored with the CORRECT entry and with the
INVERTED TWIN's — worlds that render identically and reward oppositely:

| | value |
| --- | ---: |
| label agreement with twin entry | **0.9998** |
| mean abs P(food) gap vs twin | **0.0000** |

**The outcome model produces identical predictions under an entry and
its exact inverse.** No search over such a model could ever have
distinguished the twins, so F100-F105's behavioural failures were all
downstream of this. The search was never the defect.

**And the sub-fork: the reader is not encoding the bit either.** Cosine
between a world's entry and its inverse's entry: **0.9855** (8000-update
run, 6 worlds, range 0.980-0.992). The reader emits nearly the same
entry for opposite worlds, and the outcome model ignores what little
difference remains.

**This is F58, exactly.** F58 recorded: "with few goals, ignoring the
goal channel is competitive, and under isolation it is OPTIMAL — so the
plant learns an unconditional habit and never reads its instruction."
Here the two halves collapse together: the outcome model finds the
twin-average first, which leaves the reader no gradient to differentiate
by, which leaves the outcome model nothing to read. Neither can move
alone and gradient descent has no reason to move both.

**The fix is a mechanism this project already built and never applied
here.** F58's phase-1 IGNORANCE OBJECTIVE penalises the model for
performing well WITHOUT the entry, making reading the only way to score.
It was built for the goal channel and the situation is identical. The
prediction it licenses is specific and testable: with an ignorance term,
twin label agreement must fall well below 0.9998 and entry cosine well
below 0.9855, BEFORE any behavioural improvement is claimed — the model
measurements should move first, and if they do not, the behavioural
number means nothing.

**What this closes.** The games line ran F99-F106 and the honest summary
is: the mechanism reads a world where reading is the only way to score
(`dual`, F99); on multi-step worlds it collapses to the context-free
average, and that collapse is now measured at the model level rather
than inferred from behaviour. The benchmark (F105) and these
diagnostics are what any next attempt should be scored against — floor
-0.0318, ceiling +0.1954, twin agreement 0.9998, entry cosine 0.9855.

Probe 206 is `game_slots.py --forage-twins` with model diagnostics,
2 seeds at 40000 updates plus an 8000-update run for entry similarity.

**F107 (probe 207). F58's ignorance objective closes the collapse. The
entry's causal contribution grows 100x, the model gate moves FIRST, and
the system beats the best context-free policy for the first time on a
multi-step task.** F106 localised the defect: the outcome model gave
0.9998 label agreement between an entry and its inverted twin, and the
reader emitted entries 0.9855 cosine-similar for opposite worlds — a
joint collapse onto the twin-average, which is F58's failure verbatim.
F58's remedy is to penalise being accurate WITHOUT the entry. Applied
here as an entropy term pushing the entry-free prediction toward uniform.

**The model gate, checked before any behavioural claim** (the standing
requirement recorded in F106):

| arm | twin agreement | food gap | entry cosine | outcome balanced acc |
| --- | ---: | ---: | ---: | ---: |
| no ignorance (F106) | 0.9998 | 0.0000 | 0.9855 | 0.4312 |
| **ignorance 0.5** | **0.5343** | **0.1473** | **0.7119** | 0.4305 |
| ignorance 2.0 | 0.6838 | 0.0863 | 0.3804 | 0.4321 |

The model now predicts differently under an entry and its exact inverse,
and the reader emits genuinely different entries for twins. Outcome
accuracy is unchanged (0.4305 against 0.4312), so the term bought
discrimination without costing prediction.

**Then behaviour, which is now meaningful:**

| arm | held-out | twin entry | **entry effect** |
| --- | ---: | ---: | ---: |
| no ignorance | -0.0466 | -0.0471 | +0.0005 |
| **ignorance 0.5** | **-0.0217** | **-0.0716** | **+0.0499** |
| *best invariant policy* | *-0.0318* | | |
| *oracle* | *+0.1954* | | |

Three things, consistent on both seeds (+0.0543 and +0.0455):

1. **The entry's causal contribution grows 100x**, +0.0005 to +0.0499.
2. **It beats the best context-free policy** (-0.0217 against -0.0318) —
   the first time anything in F100-F106 has.
3. **The wrong rule now HURTS**: twin entry -0.0716 against a withheld
   -0.0480. That is F99's signature appearing on a multi-step task.

**22.0% of the measured headroom, captured by reading context.**

**Honest scope.** 22% is not 100% — the oracle remains far above at
+0.1954. And the response is NON-MONOTONE: ignorance 2.0 produces MORE
differentiated entries (cosine 0.3804 against 0.5's 0.7119) but WORSE
behaviour (+0.0230 against +0.0499). More pressure on the entry channel
is not simply better, and the curve between is being swept now rather
than assumed.

**The chain that got here, because the shape of it is the lesson.**
Behavioural failure (F100) -> wrong diagnosis, state representation
(F101) -> refuted by building it -> sparsity, found by counting labels
(F102) -> fixes that helped but inverted a degeneracy (F103) -> the
baseline itself was wrong (F104) -> benchmark built and validated (F105)
-> models scored directly, collapse localised (F106) -> existing
mechanism applied (F107). Six findings inferring from behaviour, then
one batch of model-level scoring answered it. The standing rule from
F106 — model measurements move first, or the behavioural number means
nothing — is what made this result interpretable rather than another
reward number.

Probe 207 is `game_slots.py --forage-twins --ignorance {0.5, 2.0}`,
2 seeds each.

**F108 (probe 208). The ignorance weight has a threshold and an optimum,
and past the optimum the reader and the model DECOUPLE.** F107 left the
non-monotone response uncharacterised. Swept, 2 seeds each, 40000
updates, model gate reported before behaviour:

| weight | twin agree | food gap | entry cosine | outcome bal | held-out | **entry effect** | % headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.9998 | 0.0000 | 0.9855 | 0.4312 | -0.0466 | +0.0005 | 0.2% |
| 0.1 | 0.9931 | 0.0009 | 0.9821 | 0.4175 | -0.0472 | -0.0015 | -0.7% |
| 0.25 | 0.7029 | 0.0912 | 0.6972 | 0.4234 | -0.0324 | +0.0250 | 11.0% |
| **0.5** | 0.5343 | 0.1473 | 0.7119 | 0.4305 | **-0.0217** | **+0.0499** | **22.0%** |
| 1.0 | **0.3364** | **0.1753** | 0.4940 | **0.4474** | -0.0285 | +0.0417 | 18.4% |
| 2.0 | 0.6838 | 0.0863 | **0.3804** | 0.4321 | -0.0362 | +0.0230 | 10.1% |

**Three structures in one curve.**

1. **A threshold.** Weight 0.1 does nothing at all — twin agreement
   0.9931, entry cosine 0.9821, entry effect -0.0015. The collapse is
   stable against small pressure; it takes about 0.25 to break.
2. **An optimum at 0.5**, and it is an inverted U rather than a plateau:
   11.0% -> 22.0% -> 18.4% -> 10.1%.
3. **Decoupling past the optimum, which is the interesting part.** At
   weight 1.0 the model discriminates MOST (twin agreement 0.3364, food
   gap 0.1753) and its outcome accuracy is highest (0.4474) — yet it
   scores below 0.5. At weight 2.0 the READER emits the most distinct
   entries of the whole sweep (cosine 0.3804) while the MODEL's
   discrimination falls back to 0.6838. The two halves stop moving
   together: pressure keeps pushing the reader apart past the point
   where the model can use what it is given.

So maximum discrimination is not maximum benefit, and the quantity to
tune is the AGREEMENT between the halves rather than the separation of
either. That is a mechanism-level statement the behavioural numbers
alone could never have produced — it comes from having measured reader
and model separately (F106).

**Status of the games line, complete.** From 0.2% of the measured
headroom to 22.0%, with the wrong entry now actively harmful (-0.0716
against a withheld -0.0480) and the system beating the best context-free
policy (-0.0217 against -0.0318) for the first time. The remaining 78%
is not diagnosed, and the honest candidates are ordinary: the transition
model is only 0.5842 exact, the outcome model only 0.4474 balanced at
its best, and beam search over models that inaccurate has a low ceiling
regardless of how well the entry is read.

Probe 208 is `game_slots.py --forage-twins --ignorance {0, 0.1, 0.25,
0.5, 1.0, 2.0}`, 2 seeds each.

**F109 (probe 209). The transition model was never deficient — the
search was planning against hallucinated objects. Freezing them buys
3.4 points. And my explanation for the rest is refuted.**

**Per-slot diagnosis first.** F108 blamed the remaining gap partly on a
transition model at 0.5842 exact. Broken down by slot:

| slot | accuracy |
| --- | ---: |
| avatar row | **1.0000** |
| avatar col | **1.0000** |
| nearest positive object (row, col) | 0.6722, 0.6722 |
| nearest negative object (row, col) | 0.7728, 0.7712 |

The avatar's dynamics are learned PERFECTLY. The 0.58 exact figure is
entirely the object slots, and those are not learnable: "nearest object"
changes discontinuously when the avatar moves or an item respawns. It is
a stochastic function of the world, not a deficiency of the model.

**Which located a real defect in the SEARCH.** Beam search rolled all
six slots forward, so a depth-4 plan compounded 0.72 per step into
roughly 0.27 — planning against object positions the model invented.
Items sit still between respawns, so the OBSERVED layout is the better
estimate. Holding the object slots fixed and rolling only the avatar:

| arm | held-out | twin entry | entry effect | % headroom |
| --- | ---: | ---: | ---: | ---: |
| ignorance 0.5 (F107) | -0.0217 | -0.0716 | +0.0499 | 22.0% |
| **+ freeze objects** | **-0.0205** | -0.0782 | **+0.0577** | **25.4%** |

Both seeds improve (+0.0525, +0.0628). A real gain, and a small one —
so hallucinated objects were not the main remaining barrier either.

**And the hypothesis for the rest is REFUTED.** I predicted the binding
limit was the "nearest object only" abstraction: with one item pair the
slots describe the world completely, with three the agent can step onto
an item the state never mentions. Broken down by item count:

| item pairs | held-out | entry effect | % headroom |
| ---: | ---: | ---: | ---: |
| 1 (state COMPLETE) | -0.0264 | +0.0450 | 19.8% |
| 2 | -0.0326 | +0.0440 | 19.3% |
| **3 (state most incomplete)** | **-0.0074** | **+0.0762** | **33.6%** |

The worlds where the abstraction is WORST perform BEST, and outcome-model
accuracy is flat across counts (0.4320 vs 0.4291). The prediction was
backwards. The likeliest reading is that more items simply means more
chances to eat — reward density at TEST time, not state completeness —
but that is a hypothesis and is recorded as one.

**Honest position on the remaining ~75%.** Three candidates have now
been tested and none explains it: the transition model (perfect where it
matters), the search's object rollout (worth 3.4 points), and the state
abstraction (refuted, and backwards). The outcome model sits at 0.4474
balanced accuracy at its best, which is the obvious remaining suspect,
but nothing yet shows it is the binding one. Stating that plainly is
better than a fourth guess — this session has already recorded two wrong
diagnoses of mine on this exact question (F101 state, F108 transition
model).

Probe 209 is `game_slots.py --freeze-objects` with per-slot transition
diagnostics, 2 seeds.

**F110 (probe 210). Oracle substitution: the outcome model IS the
binding constraint, and the split is now exact — 63% of the missing
headroom is the outcome model, 32% is search/dynamics.** F109 left the
outcome model as the obvious suspect without showing it binding, and
this project had already spent two wrong diagnoses on that question.
The discriminating test: replace the outcome model with ground truth
INSIDE the search, keeping the learned dynamics and the beam intact.

| arm | held-out reward |
| --- | ---: |
| learned outcome model (F109 best) | -0.0205 |
| **ORACLE outcome values, learned everything else** | **+0.1234** |
| hand-coded oracle policy (ceiling) | +0.1954 |
| best context-free policy (floor) | -0.0318 |

Per-seed +0.1093 / +0.1375 — both far above anything any learned arm has
reached.

**The verdict, in one sentence: give the search true values and it
captures 68.3% of the floor-to-ceiling gap, so the outcome model was the
binding constraint, and the residual +0.0720 to the ceiling is the
search and dynamics' share.** The full decomposition of the games gap,
measured rather than argued:

  * entry not read at all (fixed by F107's ignorance objective): was
    worth +0.0499;
  * object hallucination in search (fixed by F109's freeze): +0.0078;
  * **outcome model inaccuracy: +0.1439 — the dominant term**;
  * search + dynamics residual: +0.0720, currently unaddressed.

**Why this is the right kind of result.** Two wrong mechanism guesses
(F101, F108) cost a formulation each; this one run cannot be argued
with, because the only thing changed is the quantity under suspicion.
The next work is now genuinely known rather than suspected: make the
outcome model better — it sits at 0.4474 balanced accuracy against a
0.3333 floor, trained from 12%-dense outcome labels through a bank
entry. The candidates are the ordinary ones (more visits per world,
better labels, a value head trained on n-step returns rather than
3-class outcomes), and the oracle arm now provides the exact target any
of them should be scored against: +0.1234 is what perfect values buy
through this search.

Probe 210 is `game_slots.py --oracle-outcome`, 2 seeds.

**F111 (probe 211). The n-step value head nearly doubles the captured
headroom — 25.4% to 45.6% — and held-out reward turns positive for the
first time.** F110's oracle test convicted the outcome model and showed
the interface that works: real-valued cell worth, not 3-class bins.
Mimicking that interface — a scalar head regressing the discounted
n-step return, used raw by the search, with the ignorance term pinning
the entry-free prediction to the batch mean:

| arm | held-out | twin entry | entry effect | % headroom |
| --- | ---: | ---: | ---: | ---: |
| 3-class outcome (F109) | -0.0205 | -0.0782 | +0.0577 | 25.4% |
| **n-step value head** | **+0.0069** | **-0.0968** | **+0.1036** | **45.6%** |
| oracle values (target) | +0.1234 | | | |
| best context-free (floor) | -0.0318 | | | |

Both seeds positive held-out (+0.0010, +0.0127); both seeds' twin
penalty deepens (-0.0901, -0.1034) — the wrong rule hurts MORE when the
values are real-valued, which is what reading harder looks like.

The 3-class quantisation was itself a large part of the constraint it
was supposed to measure: collapsing all positive futures into one bin
threw away the gradient the search needed, and was degenerate twice
over along the way (F102 all-nothing, F106 never-nothing). Regression
has no bins to degenerate into.

Remaining to the oracle-value target: +0.1165. The games ladder in
sequence: 0.2% -> 22.0% (ignorance) -> 25.4% (freeze) -> 45.6% (value
head), each step from a measured diagnosis rather than a guess.

Probe 211 is `game_slots.py --value-head`, 2 seeds.

**F112 (probe 212). Value fidelity is low and polarity-asymmetric: the
value head almost never ranks food on top in inverted worlds — the
entry flips avoidance but not attraction.** With the F111 configuration
plus a direct fidelity check — score every cell with the value head and
compare against the verifier's ground-truth worth map — pooled over
2 seeds x 12 held-out worlds:

| measure | pooled | normal worlds | inverted (~) worlds |
| --- | ---: | ---: | ---: |
| predicted-vs-truth correlation | 0.1727 | 0.1694 | 0.1761 |
| top-ranked cell is food | 0.1198 | 0.2188 | **0.0208** |
| top-ranked cell is poison | 0.0156 | 0.0182 | 0.0130 |

Three facts, one picture:

  * **Global correlation is weak (0.17) yet behaviour captures 45.6% of
    headroom.** Beam search only needs correct *ranking among the few
    cells reachable within depth 4*, not a globally faithful map — so
    low correlation and decent behaviour can coexist. This also warns:
    the remaining +0.1165 will not come from nudging correlation; it
    must come from ranking near the avatar.
  * **Poison avoidance transfers across polarity; food attraction does
    not.** top=poison is near zero everywhere — the model reads the
    entry well enough to know which object class to avoid in both twins.
    But top=food collapses from 0.219 to 0.021 on ~ worlds: on inverted
    worlds the argmax cell is almost always *empty*, never the truly
    edible object.
  * The correlation being *identical* across polarity while top=food is
    10x different means the asymmetry lives in the extreme of the
    ranking, not the bulk: the ~ entry damps the poisonous object's
    value (avoidance works) but fails to *raise* the newly-edible
    object above background.

Interpretation: the entry acts multiplicatively-downward, not as a true
sign flip. The pre-training distribution is symmetric in twins by
construction, so this is not label imbalance; it is the value head
finding "suppress the flagged object" easier to express than "promote
it". This matches the held-out reward split visible in every run since
F107: normal worlds positive, ~ worlds hovering just below zero.

The next fix target is therefore concrete: make the entry able to
*promote*, not only suppress — candidates are a signed (rather than
gated) entry interaction in the value head, or the diff-entry mechanism
(entries as deltas against the nearest existing entry), which expresses
"same world, one piece swapped" natively.

Probe 212 is `game_slots.py` value_fidelity(), 2 seeds (69316/69317).

**F113 (probe 213). The signed entry pathway more than doubles behaviour
— held-out +0.0069 to +0.0692, entry effect +0.1036 to +0.2230 (98% of
the twin-separation headroom) — but every point of the gain lands on
normal-polarity worlds; the inverted twins are still flat.** The change:
value += tanh(polarity(entry)) * salience(state) — the state supplies a
polarity-free object salience, the entry supplies one scalar in [-1,1],
so promote and suppress become one sign apart instead of a re-mapping
of the whole value surface. Pooled over 2 seeds:

| measure | F111 (attention only) | F113 (signed pathway) |
| --- | ---: | ---: |
| held-out reward | +0.0069 | **+0.0692** (+0.0357 / +0.1027) |
| twin entry | -0.0968 | **-0.1538** |
| entry effect | +0.1036 | **+0.2230** |
| top=food, normal worlds | 0.219 | **0.667** |
| top=food, inverted (~) | 0.021 | 0.073 |
| value-truth correlation | 0.173 | 0.229 |
| oracle-value target | +0.1234 | +0.1234 |

Seed 69317 alone reaches +0.1027 — within noise of the oracle-value
target — with single worlds at +0.35 (oracle full-battery ceiling is
+0.1954). The mechanism works exactly as designed on one polarity:
top=food on normal worlds triples to 0.667, and the twin penalty
deepens again (-0.126 / -0.181), the signature of harder reading.

But the inverted worlds barely move (rewards -0.03..+0.03, top=food
0.073), so the polarity scalar is not actually flipping: the salience
term learned "toward the plane-1 object" and the tanh learned to turn
it UP on normal worlds and merely OFF on ~ worlds. Suppression via the
old attention path still handles poison (top=poison 0.016), which is
why ~ worlds sit at zero rather than negative. The asymmetry F112 found
is now isolated to a single learned scalar per world — the next probe
should log tanh(polarity(entry)) per held-out world directly: if it is
positive-or-zero everywhere rather than sign-split, the failure is in
the reader's entry (twins too similar for a linear map to separate), and
the diff-entry mechanism is the targeted fix.

Two-seed spread is wide (+0.0357 / +0.1027); any promotion claim needs
a third seed. Probe 213 is `game_slots.py --signed-entry`, 2 seeds.

**F114 (probe 214). The math proving ground: with dense supervision the
reader sign-splits twins PERFECTLY — twin-entry accuracy is 0.0000, the
plant committing fully to the opposite rule — so the games' polarity
failure is not a fundamental reader defect.** New probe `math_twins.py`
per the proving-ground idea: a world is x -> (a*x + b) mod 16, its twin
negates b; the reader sees 8 example pairs, the plant continues
held-out queries; ignorance objective, pair-held-out splits, and the
four controls all as in the games. Runs in ~2 minutes.

| arm (held-out pairs, 2 seeds) | accuracy |
| --- | ---: |
| own entry | 0.4121 / 0.5000 |
| twin entry | **0.0000 / 0.0000** |
| withheld entry | 0.0501 / 0.0472 (chance 0.0625) |
| stranger entry | 0.1550 / 0.1165 |

Twin cosine 0.60 / 0.44 — the entries are genuinely different, and the
plant reads the difference: given the twin's entry it predicts x - b
where truth is x + b, which is why twin accuracy is ZERO rather than
chance. That is the strongest possible form of the sign-split F113
could not get from the games value head.

The dissociation this buys: here the reader trains against DENSE exact
next-value labels; in the games it trains through sparse discounted
returns from 12%-dense events. Same architecture, same objective, same
controls — opposite polarity behaviour. So the suspect in the games
narrows to the training signal reaching the value pathway, not the
reader's capacity to separate twins. (Supporting detail: stranger
entries score above withheld here because a stranger's b can coincide
with a held world's b — the reader is reading rules, not worlds. The
linear sign-probe was uninformative either way, 0.27/0.65 — 12 held
entries against a 768-dim probe is noise; the behavioural twin test is
the measurement.)

Own-entry accuracy at 0.41-0.50 rather than ~1.0 is the next math-side
question: the additive rule is identifiable from any single example
pair, so reading should saturate. Candidates: more training updates,
more pairs (24 may under-span b-space), examples too few per entry.
The math ground is cheap enough to sweep all three.

Probe 214 is `math_twins.py`, 2 seeds.

**F115 (probe 215). Math-ground sweep: the 0.4-0.5 own-entry plateau is
rule-interpolation, not underfitting — and the diversity law (F78)
reproduces in miniature.** Three axes, 2 seeds each: 3x updates
(0.4300/0.4372 — flat), 2x examples per entry (0.5000/0.4954 — flat),
2x pairs (0.2500/0.0898 — much WORSE). The pairs arm is the telling
one: mod 16 admits only 15 distinct additive rules, so drawing 48
pairs collapses to the same <=15 stems while holding out 8 of them —
training diversity drops from ~9 rules to ~7 and reading degrades,
exactly the F78 curve at 1/500th the scale. The plateau itself is the
plant failing to INTERPOLATE to held-out b values in embedding space
(any single example identifies b; twin accuracy stays 0.0000
throughout, so reading the sign is never the problem). The math ground
therefore reproduces both headline laws of the big battery — diversity
drives reading, and dense supervision sign-splits — in two-minute runs.
Caveat for future math probes: modulus bounds rule diversity; use a
larger modulus or the multiplicative family before drawing conclusions
that need many rules.

Probe 215 is `math_twins.py` sweep, 2 seeds x 3 arms.

**F116 (probe 216). The polarity scalar SIGN-SPLITS — F113's "never
negative" hypothesis is refuted — and the true defect is that salience
is a single channel keyed to the plane-1 object, so inverted worlds can
only avoid, never seek.** Logging tanh(polarity(entry)) per held-out
world (seeds 69316/69318):

    normal worlds:  +0.99 +1.00 +1.00 +0.55 +1.00  (one outlier -0.98)
    inverted (~):   -1.00 -1.00 -0.99 -1.00 -1.00 -1.00

Near-perfect sign separation — agreeing with the math ground (F114)
that the reader distinguishes twins fine. The asymmetry mechanism is
now exact: the slot state defines slots 2/3 as the nearest PLANE-1
object, salience(state) learns "worth peaks near plane-1", and the
sign flips that one attraction. On ~ worlds sign=-1 correctly repels
from plane-1 (the poison) — but no term exists that can promote the
plane-2 object, so top=food is 0.000 on every ~ world and their reward
is pure avoidance (~0). One salience channel cannot express "seek the
OTHER object".

Third seed confirms F113 behaviourally: held-out +0.0357 / +0.1027 /
+0.1064 (mean +0.0816 vs F111 +0.0069; oracle-value target +0.1234),
twin penalty -0.126 / -0.181 / -0.185.

Fix is architecturally minimal: per-plane salience channels, each with
its own entry-derived polarity — value += sum_i tanh(polarity_i(entry))
* salience_i(state), i over the two object planes. Inverted worlds then
promote plane-2 with the same machinery normal worlds use for plane-1.

Probe 216 is `game_slots.py --signed-entry` with polarity logging,
seeds 69316/69318.

**F117 (probe 217). Compositional math first contact is a NULL — chance
everywhere, and the entry is not read at all.** New probe
`math_compose.py`, the composition rung the founding objective actually
needs: a world hides f(x) = x+b and g(x) = a*x over Z_23; the reader
sees only SINGLE applications; the plant executes token programs like
[f,g,f]; held-out split on both axes (worlds and programs), plus a
swap control (f/g roles exchanged). At 12k updates / dim 96, 2 seeds:

    trained worlds, trained programs : 0.098 / 0.088
    every other cell                 : ~0.043-0.053  (chance 0.0435)

Two diagnostics before any redesign:
  * stranger accuracy is BIT-IDENTICAL to own-entry accuracy in every
    cell — the plant distinguishes zeros-vs-real entry (the ignorance
    term forces that) but treats all real entries alike: reading never
    started;
  * even the pure fit fails (0.098 on trained x trained), so the
    binding constraint is below reading: modular multiplication
    composed up to length 4 is not yet representable at this budget.

Same failure order as the games (F106): model first, reader second —
nothing can be read until something predicts. Scaled runs (60k
updates, dim 128) are the first arm; if fit lands and reading still
does not, the next lever is the F78 one — world diversity — before
any architectural change.

Probe 217 is `math_compose.py`, 2 seeds.

**F118 (probe 218). Two-channel salience: pooled +0.0947 (from
+0.0816), normal worlds reach top=food 1.000 — but the second channel
never differentiates, and the cause is the COLLECTION policy, not the
architecture.** Three seeds, per-channel polarity logged:

    held-out +0.0811 / +0.0972 / +0.1058, pooled +0.0947
    (F111 +0.0069, F113 +0.0816, oracle-value target +0.1234)
    top=food: normal worlds 1.000 / 1.000 / 0.667 — saturated;
              inverted worlds 0.021 / 0.062 / 0.000 — unmoved.

In every seed exactly ONE polarity channel is alive (+-1, sign-split
by twin as in F116) and the other sits near zero: the plane-2 channel
never gets gradient. The reason is in the data: `--seek 0.85` steers
collection toward the nearest PLANE-1 object, so trajectories almost
never consume plane-2 objects — on inverted worlds the plane-2-food
events that would teach "seek the other object" are nearly absent from
training. The math ground's dissociation (F114: dense signal
sign-splits, sparse signal collapses) recurs one level down: the
architecture now has the slot for the second rule-piece, and the
DATA never fills it.

Fix queued: seek the two planes with equal probability during
collection, leaving everything else fixed.

Probe 218 is `game_slots.py --signed-entry` (2-channel), 3 seeds.

**Measurement correction (2026-08-10). The "% of headroom" metric has
broken and must be retired for the games ladder.** It was defined as
entry effect (own-entry minus twin-entry reward) divided by the
floor-to-oracle span +0.2272. Recomputed across the ladder:

    F111 +0.1037 =  45.6%
    F113 +0.2458 = 108.2%
    F118 +0.2714 = 119.5%

Past 100% it is measuring the wrong thing: the twin arm now falls far
BELOW the context-free floor (-0.16 to -0.19 against -0.0318), because
an agent that confidently applies the inverted rule seeks poison rather
than merely wandering. The denominator assumed the twin arm sits at
floor. The effect size is real and still the right CAUSAL statistic —
it is what separates reading from not-reading — but it is not a
fraction of anything.

Standing rule from here: report games progress as HELD-OUT REWARD
against the two oracle references (+0.1234 through the learned search,
+0.1954 with true dynamics too), and report entry effect separately as
an unnormalised causal magnitude. On that honest scale the ladder is:
-0.0205 (F109) -> +0.0069 (F111) -> +0.0816 (F113) -> +0.0947 (F118),
against +0.1234; i.e. 77% of the oracle-value target, not 119% of
anything.

**F119 (probe 219). CORRECTION to F117, and the real finding: the plant
fits composed modular arithmetic PERFECTLY (1.0000) and generalises to
unseen programs at CHANCE (0.0794 vs 0.0435). Composition is not
happening at all — each program is memorised as its own function.**
The single-world fit arm (1 training world, ignorance off, 30k updates,
dim 128) removes reading from the picture entirely and asks only
whether the pieces compose:

    trained programs (18) : 1.0000   <- representation is not the limit
    held-out programs (12): 0.0794   <- chance is 0.0435

F117 read the multi-world 0.098-0.127 as "fit fails before reading".
That was wrong in the way that matters: fit does not fail, it is
perfect per-world. What fails is generalisation across ARRANGEMENTS of
pieces the model already executes flawlessly. The multi-world number
was low because 18 worlds x 18 programs is 324 separately-memorised
functions, not because the arithmetic is unrepresentable.

Why this is the most important null so far: the whole bank thesis is
that structure is reusable and content is looked up. A model that
executes [f,g,f] perfectly and is at chance on [g,f,g] has learned 18
opaque functions, not two pieces plus a rule for combining them. No
amount of world diversity fixes that — the failure is INSIDE one world
where there is nothing to read.

The cause is architectural and was ours to choose: the plant sees the
whole program as a set of tokens and emits an answer in ONE shot, so
nothing forces it to apply pieces sequentially. It is the F67 lesson
(store facts, derive behaviour by search) one level up — we asked it
to learn composite functions instead of learning a piece and applying
it repeatedly.

Next probe: `--iterate`, a recurrent latent state stepped ONCE PER
PROGRAM TOKEN through a shared step function, decoded only at the end,
trained end-to-end with no intermediate supervision. Same parameters,
same blocks — the only change is that composition becomes structural
rather than something to be learned. If held-out programs jump, the
puzzle-piece mechanism is real and the interface was the whole story.

Probe 219 is `math_compose.py --worlds 2 --ignorance 0`, 1 seed.

**F120 (probe 220). Boolean composition fails the same way, and adds
two diagnostics the arithmetic version could not: reading is entirely
absent (stranger accuracy is BIT-IDENTICAL to own-entry accuracy), and
world identity is irrelevant (trained worlds score the same as held-out
worlds).** Pieces reduced to the minimum — XOR with a hidden mask,
rotate by a hidden shift, over 8-bit vectors, non-commutative, 1785
worlds available. Exact-match, chance 0.0039:

| arm | trained programs | held programs | stranger | withheld |
| --- | ---: | ---: | ---: | ---: |
| 64 worlds, seed 69316 | 0.1454 | 0.0325 | **0.0322** | 0.0123 |
| 64 worlds, seed 69317 | 0.0679 | 0.0029 | **0.0029** | 0.0034 |
| 512 worlds, seed 69316 | 0.1371 | 0.0124 | **0.0124** | 0.0119 |

Three readings, in order of what they rule out:

  * **Stranger == own entry to 3-4 decimals in every arm.** A foreign
    world's entry works exactly as well as the correct one: the plant
    is not reading. This is not partial reading or weak reading; it is
    none.
  * **Trained worlds == held-out worlds** (0.1454 vs 0.1464; 0.1371 vs
    0.1376). The model has learned nothing world-specific at all — it
    has found one world-independent average function.
  * **8x the world diversity changes nothing** (0.1371 vs 0.1454).
    F78's lever, which has worked on every previous rung, is inert
    here — confirming from a second direction that the constraint is
    not diversity.

Why the ignorance objective did not save it, which is the transferable
lesson: the term penalises being ACCURATE WITHOUT the entry, and this
model is not accurate with or without it (per-bit 0.55-0.62 against
0.50 chance). The pressure to read only exists once something predicts
well enough for the entry to matter. **The ignorance objective is
toothless when the model is bad** — it fixed F106 precisely because
the games model was already accurate on the twin-average.

So the boolean ground reproduces F119 and localises it further: with
one world (F119) the one-shot interface fits perfectly and cannot
compose; with 64 worlds it cannot even fit, because every (world,
program) pair is a separate function to memorise and there are 1152 of
them. Both are the same defect at different scales.

Probe 220 is `bool_compose.py`, 3 arms.

**F121 (probe 221). THE INTERFACE WAS THE WHOLE STORY. Applying pieces
one at a time through a shared step function takes held-out program
composition from CHANCE to PERFECT: 0.0794 -> 1.0000.** The exact F119
setting — one world, ignorance off, 30k updates, dim 128 — with the
single change that the plant carries a latent stepped once per program
token instead of answering the whole program in one shot:

| interface | trained programs | held-out programs |
| --- | ---: | ---: |
| one-shot (F119) | 1.0000 | 0.0794 (chance 0.0435) |
| **iterated (F121)** | **1.0000** | **1.0000** |

Same blocks, same parameter count, no intermediate supervision — the
latent is never told what the intermediate value should be. The only
difference is that the step function is SHARED across positions and
program lengths, and that sharing is the entire compositional prior.

What this establishes, stated carefully: a model asked to answer
[f,g,f] in one shot learns eighteen opaque composite functions and
generalises to none of them; the same model asked to apply one piece
at a time learns TWO pieces and gets every arrangement for free. This
is the puzzle-piece claim demonstrated rather than argued — and it is
the F67 lesson (store facts, derive behaviour by search) one level up:
do not learn the composite, learn the piece and iterate it.

Caveats held open deliberately: (1) one seed — the effect is
chance-to-ceiling so it is not a noise question, but replication is
queued; **RESOLVED 2026-08-10: seed 69317 also gives 1.0000 trained /
1.0000 held-out, and F125's length-extrapolation arm gives 1.0000 on
programs of an unseen depth — two seeds and the strictest split all at
ceiling;** (2) reading is absent from this arm by construction (one
world), so the open question is whether the same interface lets the
BANK ENTRY supply the pieces across many worlds — that is exactly what
F120's failing 64-world setting will now re-test; (3) held-out
programs here share their LENGTHS with trained ones, so a strict
length-extrapolation arm (train <=3, test 4) is running to rule out
within-length interpolation.

Probe 221 is `math_compose.py --iterate --worlds 2 --ignorance 0`.

**F122 (probe 222). The dissociation is now complete and two-sided:
diversity — the lever that has worked on every previous rung — is
INERT on composition, while the interface is everything.** The 256-
world arithmetic arm (60k updates, dim 128) closes the diversity axis
on both grounds:

| ground | worlds | trained programs | held programs | stranger |
| --- | ---: | ---: | ---: | ---: |
| math | 24 | 0.098-0.127 | ~chance | == own |
| math | **256** | 0.0619 / 0.0528 | 0.0421 / 0.0439 | **== own** |
| boolean | 64 | 0.0679-0.1454 | ~chance | == own |
| boolean | **512** | 0.1371 | 0.0124 | **== own** |

More worlds did not help either ground; on the arithmetic side it made
fit WORSE (0.098-0.127 at 24 worlds down to 0.053-0.062 at 256), which
is what spreading a fixed capacity over more separately-memorised
functions looks like. Stranger-entry accuracy remains bit-identical to
own-entry accuracy at every scale: reading never starts.

Put beside F121 the pair is a clean dissociation, and it is the
session's sharpest single statement:

    scaling world diversity 10x       : chance -> chance
    changing the program interface    : chance -> perfect (1.0000)

F78's law is real but it governs a different axis. Diversity decides
whether a model READS a rule or MEMORISES it, once the rule is
expressible. It has nothing to say about whether COMPOSITES of rules
are expressible at all — that is fixed by whether the architecture
applies pieces one at a time. Adding worlds to a model that cannot
compose just gives it more things to fail to compose.

Standing rule this earns: before reaching for diversity (or capacity,
or budget), check that the target function is expressible in the
interface at all — the single-world, reading-off fit arm costs one run
and answers it. F117 spent three arms on the wrong axis because that
check came last instead of first.

Probe 222 is `math_compose.py --worlds 256`, 2 seeds.

**F123 (probe 223). REGRESSION, three seeds, unambiguous: balanced
seeking destroys reading outright. The asymmetric collection policy was
doing double duty and I only saw one of its two jobs.** F118 diagnosed
the dead plane-2 salience channel correctly — `--seek` targeted plane-1
objects only, so plane-2 consumption events were nearly absent and
inverted worlds had nothing to learn "seek" from. The fix (target both
planes with equal probability, nothing else changed):

| arm | held-out | entry effect |
| --- | ---: | ---: |
| plane-1 seek (F118) | +0.0811 / +0.0972 / +0.1058 | +0.244 to +0.295 |
| **balanced seek (F123)** | **-0.0480 / -0.0474 / -0.0449** | **+0.0005 / +0.0003 / +0.0001** |

Entry effect is not reduced; it is annihilated to the fourth decimal.
The polarity scalars stop sign-splitting (|tanh| max 0.18-0.37 with
twins now giving nearly IDENTICAL values, against +-1.0 cleanly split
in F118). Held-out lands at the context-free floor.

The mechanism, which is F106 re-created deliberately by a data change:
seeking plane-1 only makes normal worlds mostly-positive and inverted
worlds mostly-negative, so a single twin-averaged predictor fits badly
and reading PAYS. Balanced seeking makes both twins' outcome marginals
identical, the twin-average becomes an excellent fit, and the model
settles there — exactly the collapse the ignorance objective was
introduced to prevent, arrived at from the data side where the
ignorance term has no purchase (it constrains only the entry-FREE
prediction, and here the entry-USING prediction is the degenerate one).

**The lesson, which generalises past this probe: asymmetry in the data
is what makes reading pay.** A benchmark whose twins are marginally
identical does not reward context-reading, it merely permits it. F104's
design rule (every inversion-invariant policy must score near floor)
constrains the TASK; this constrains the COLLECTION — an
inversion-invariant data distribution silently converts a
context-required task into a context-optional one.

Fix queued as `--seek-plane2 p` (0.0 = F118, 0.5 = this collapse):
intermediate p should keep enough marginal asymmetry for reading to pay
while still visiting plane-2. Whether such a p exists is an empirical
question and may be answered "no" — in which case the channel must be
fed some other way (e.g. seeking by predicted value once the model is
good enough, rather than by fixed plane index).

Probe 223 is `game_slots.py` with balanced seeking, 3 seeds.

**F124 (probe 224). The iterated interface makes the multi-world case
WORSE, not better — composition and reading are independent problems,
and iterating amplifies a reading failure instead of curing it.** Same
64-world boolean setting as F120, only the interface changed:

| interface | trained programs | held programs | stranger | withheld |
| --- | ---: | ---: | ---: | ---: |
| one-shot (F120) | 0.1454 / 0.0679 | 0.0325 / 0.0029 | == own | 0.0123 |
| **iterated (F124)** | **0.0077 / 0.0143** | 0.0052 / 0.0135 | **== own** | 0.0044 |
| iterated + length-extrap | 0.0222 | 0.0038 (len 4) | == own | 0.0062 |

F121 took the SINGLE-world case from 0.0794 to 1.0000 with this same
change. The difference between the two settings is the only thing that
matters here: in the single-world arm the step function has two fixed
pieces to learn and needs no per-world content; at 64 worlds it must
get the pieces FROM the entry, and F120 already measured that channel
to be dead (stranger bit-identical to own, still true here).

The mechanism of the deterioration is worth stating because it will
recur: a one-shot model can emit a world-independent average answer and
collect partial credit; an iterated model applies a world-ignorant step
four times and compounds the error. **Iteration is a multiplier on
whatever the step function knows — including a multiplier on nothing.**
So the interface fix is necessary but strictly downstream of reading.

One real signal in the noise: withheld (0.0044) now sits clearly BELOW
own and stranger (0.0052-0.0138), where F120 had them level. The model
has begun using "an entry is present" without using "WHICH entry" —
the first crack, but not reading.

This is the fourth independent confirmation that reading is dead at
multi-world scale in the composition probes (F120 stranger identity,
F122 diversity inertia, F124 here, and the withheld/own gap being the
only entry effect anywhere). Oracle-entry arms are running to decide
whether execution is sound and reading is the sole constraint.

Probe 224 is `bool_compose.py --iterate`, 2 seeds + 1 length arm.

**F125 (probe 225). F121 survives the strictest form of the test:
trained on programs of length <=3 only, the iterated plant answers
LENGTH-4 programs — never demonstrated at that length — at 1.0000.**
The one way F121 could have been passed without composing was
interpolation within a length, since its held-out programs shared
lengths 3 and 4 with trained ones. The extrapolation split removes
that: 14 training programs covering lengths 1-3, 16 held-out programs
all of length 4, no overlap.

| split | trained | held-out |
| --- | ---: | ---: |
| one-shot, same-length (F119) | 1.0000 | 0.0794 |
| iterated, same-length (F121) | 1.0000 | 1.0000 |
| **iterated, length-extrapolation (F125)** | **1.0000** | **1.0000** |

Applying a piece one more time than was ever demonstrated, and getting
every one of sixteen unseen length-4 programs exactly right, is the
claim in its strongest available form. Nothing about the number 4 was
in the training distribution; only the step was.

What is now settled about composition, and what is not:
  * SETTLED — a shared per-element step function composes, extends to
    unseen arrangements, and extends to unseen DEPTHS, with no
    intermediate supervision;
  * NOT SETTLED — whether the pieces can come from a bank entry rather
    than from weights. Every composition success so far (F121, F125)
    is single-world, where the step needs no per-world content;
    every multi-world arm (F120, F122, F124) has reading dead. That
    junction is the whole remaining question and the oracle-entry arms
    are running against it.

Probe 225 is `math_compose.py --iterate --train-max-len 3`, 1 seed.

**F126 (probe 226, interim — one cell of a 2x2). Oracle entries do NOT
rescue the one-shot interface at 256 worlds: 0.0672 against 0.0619 for
the learned reader, chance 0.0435.** Handing the plant the world's true
(a, b) as clean one-hot codes moves nothing. **Perfect world knowledge
does not fix multi-world composition when the interface is one-shot** —
which reframes the multi-world failure: it is not primarily a reading
failure. F119's diagnosis holds at scale.

The likely causal order, stated as a hypothesis to be tested by the
remaining cells rather than asserted: reading dies BECAUSE execution
is impossible. There is nothing to gain from learning to read an entry
whose contents the interface could not use. If so, F120/F122/F124's
"reading is dead" is a symptom and F119's interface defect is the
disease — and the ignorance objective's toothlessness (F120) is the
same fact seen from the optimiser's side.

Wiring sanity check that makes the cell interpretable: with oracle
entries own-entry accuracy (0.0459 / 0.0473) sits slightly ABOVE
stranger (0.0423 / 0.0446), where the learned reader gave values
identical to four decimals. The plant does distinguish informative
entries; it just cannot use what it distinguishes.

The design being completed before any conclusion is drawn:

|            | learned reader | oracle entry |
| ---------- | ---: | ---: |
| one-shot   | 0.0619 (F122) | 0.0672 |
| iterated   | **0.0581** | running |

Three of four cells are now in and all sit at chance (0.0435). Neither
lever alone helps at 256 worlds: not the interface that solved the
single-world case at 1.0000, and not perfect world knowledge.

Plus a world-count ladder (4/16/64 worlds, iterate + oracle entry) to
locate WHERE execution breaks between 1 world (perfect, F121/F125) and
256 (chance): a smooth decline indicates capacity, a cliff indicates
something categorical.

A measurement correction earned along the way: the first version of
this arm encoded the oracle as scalars a/M and b/M, which places all
256 worlds on a 2-dimensional manifold and would have made the oracle
fail for reasons unrelated to reading — the exact quantity it exists
to isolate. Caught before the runs finished; re-run with one-hot codes
(width 2M = 46, verified distinct per world). Recorded because the
instrument nearly produced a confident wrong answer, which is this
session's most frequent failure mode.

**The decomposition this forces (probe 227, running).** One world with
two pieces composes perfectly (F121/F125, 1.0000, both seeds, and to
unseen depth). 256 worlds with two pieces is chance in three of four
cells. But F114's `math_twins` READS fine at multi-world scale — own
0.41-0.50 against chance 0.0625, twin accuracy exactly 0.0000 — with
ONE piece and ONE application. So the difference between the working
and failing regimes is two changes at once, and they have never been
separated:

  * **two pieces instead of one** — the entry must now carry two
    parameters and the step must select between them by token;
  * **depth up to four instead of one** — the parameter must survive
    being applied repeatedly through a latent.

`--max-len 1` holds depth at one while keeping two pieces and 256
worlds. If reading works there, disentangling two pieces from one
entry is fine and DEPTH is the constraint; if it fails there, the
entry cannot carry two parameters at once and the problem is nothing
to do with composition.

This is the general shape of the mistake worth avoiding: the failing
configuration differed from the working one in more than one way, and
until they are separated any story about the cause is a guess. F117
cost three arms to exactly this error.

**A confound in the arithmetic ground, noted before it can mislead.**
`math_compose`'s multiplicative piece is g(x) = a*x mod 23 with `a`
drawn per world. With ONE world `a` is fixed and the model memorises a
single 23x23 table — trivial. Across 256 worlds it must learn modular
multiplication as a general operation with the multiplier supplied at
runtime, which is the hardest well-known function in this size class.
So "1 world perfect, 256 worlds chance" on the arithmetic ground has a
completely mundane candidate explanation that has nothing to do with
banks, reading, or composition.

The boolean ground exists precisely to remove it — XOR-with-mask and
rotate-by-k are elementwise and permutation operations with no such
difficulty — and boolean fails at 64 and 512 worlds too (F120, F124).
That already argues the confound is not the whole story. The
decomposition arms are therefore being run on BOTH grounds: if boolean
`--max-len 1` at 256 worlds also fails to read, the defect is carrying
TWO piece-parameters in one entry, and neither arithmetic hardness nor
depth explains it.

Recording this because the confound was present from F117 onward and I
did not name it until three cells of a 2x2 had come back at chance —
the same lesson as F104 (a benchmark must be checked for what it lets
a wrong mechanism score) applied to the generating function instead of
the policy.

**F127 (probe 227). The length curriculum is a null, 2 seeds: reading
still never starts.** Ramping the maximum program length from 1 to 4
over the first half of training — motivated by the measured fact that
only 11% of updates land on length-1 programs and 67% on length>=3,
and by F120's precondition that the ignorance objective is toothless
while the model is bad:

| arm (256 worlds, iterated, held-out worlds) | trained prog | held prog | stranger |
| --- | ---: | ---: | ---: |
| no curriculum | 0.0554 | 0.0422 | == own |
| curriculum, seed 69316 | 0.0624 | 0.0423 | == own |
| curriculum, seed 69317 | 0.0539 | 0.0486 | == own |

Chance 0.0435. Stranger-entry accuracy remains identical to own-entry
to four decimals in both seeds — the curriculum changes nothing about
whether the entry is read.

The hypothesis it tested was bootstrapping: establish reading on the
single-application task where F114 proved it works, then extend to
composites. The hypothesis is refuted for this implementation. What it
does NOT rule out is that the curriculum never reached a regime where
reading paid — at 256 worlds the length-1 sub-task still asks the
entry to carry TWO piece-parameters, which is exactly the quantity the
decomposition arms are now isolating. So this is a null on the fix,
not on the diagnosis.

Running tally of levers that do NOT move multi-world composition:
world diversity 10x (F122), the iterated interface (F124), oracle
entries (F126), and a length curriculum (F127) — with the single-world
case sitting at 1.0000 throughout. Four independent failures to move
one number is itself information: the constraint is not any of the
knobs we know how to turn, which is why the next step is isolation
rather than another fix.

Probe 227 is `math_compose.py --curriculum 0.5`, 2 seeds.

**F128 (probe 228). The 2x2 completes — all four cells at chance — and
the honest reading is that I built a careful factorial on a CONFOUNDED
substrate, so it cannot answer the question it was designed for.**

| 256 worlds, arithmetic | learned reader | oracle entry |
| ---------- | ---: | ---: |
| one-shot | 0.0619 | 0.0672 |
| iterated | 0.0581 | **0.0548** |

Chance 0.0435; the same plant reaches 1.0000 at one world.

The decisive cell (iterated + oracle) is at chance on TRAINED programs,
including length-1 ones. So the arithmetic plant cannot apply a single
piece correctly at 256 worlds even when handed (a, b) as clean one-hot
codes. That is not a statement about banks or composition — it is the
statement that it has not learned modular multiplication with a
runtime multiplier, which is the canonical grokking-hard task and is
routinely reported to need 10^5-10^6 updates. This ran for 6x10^4.

**So the arithmetic 2x2 is uninterpretable for our purposes, and every
arithmetic multi-world null since F117 inherits that.** The design was
sound and the substrate was not: I varied interface and entry-source
carefully while the underlying function was one neither arm could
learn in the budget. A factorial cannot rescue a floor effect.

What survives untouched, because it never depended on the multi-world
arithmetic cells:
  * F121/F125 — composition, unseen arrangements, unseen depths, two
    seeds, ceiling. Single world, so no runtime multiplier is needed;
  * F114 — reading at multi-world scale, twin accuracy 0.0000, on a
    purely ADDITIVE family (a = 1) where no multiplication appears;
  * F123 — the games regression, which is a different probe entirely.

Note the pattern across those three: everything that WORKED avoided
runtime modular multiplication, and everything that failed at
multi-world scale required it. That is a simpler explanation of the
whole composition-probe null sequence than any of the four mechanisms
I proposed, and it was available from F117 onward.

The boolean ground is now load-bearing rather than supporting, since
XOR-with-mask and rotate-by-k have no such difficulty. Two arms
running: boolean iterated + oracle entry at 256 worlds (if this is at
chance the defect is real and structural; if it works, the only open
question is reading), and a boolean single-world control, which has
never been run — F121/F125 are arithmetic-only, so it is not yet known
that the boolean ground reproduces the composition result at all.

Probe 228 is `math_compose.py --iterate --oracle-entry --worlds 256`.

**F129 (probe 229). The seek mixture at p=0.25 preserves reading but
does not fix the asymmetry — so the games defect is not curable from
the data side, and the trade-off between the two jobs of the
collection policy is real and unfavourable.** Two seeds, everything
else as F118:

| seek-plane2 | held-out | entry effect | top=food normal | inverted |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 (F118) | +0.0811/+0.0972/+0.1058 | +0.244..+0.295 | 0.667 | 0.000 |
| **0.25 (F129)** | **+0.0614/+0.1030** | **+0.226/+0.291** | 0.417/0.500 | 0.083/0.062 |
| 0.50 (F123) | -0.048/-0.047/-0.045 | +0.0001..+0.0005 | 0.042 | 0.042 |

Reading survives at p=0.25 — that locates the collapse cliff between
0.25 and 0.5 rather than anywhere below. But the trade is bad:
inverted-world top=food rises only 0.000 -> 0.07, normal-world falls
0.667 -> 0.46, and pooled held-out is flat to slightly down (+0.0822
vs +0.0947). One polarity channel is still dominant per seed (max
|tanh| 1.000 vs 0.318, and 0.224 vs 1.000).

So all three points on this axis are now measured and none of them
gives inverted worlds a working "seek". The data-side fix is refuted
as a family, not just at one setting.

**F130 (probe 230). The fix that follows, from an established finding
rather than a new idea: TIE the salience channels (F73 slot
symmetry).** The reason plane-2 needs its own data at all is that its
salience channel has its own parameters. F73 measured that a
slot-symmetric plant — shared value-embedding, shared MLP, shared
head, distinguished only by positional embeddings — beat the
unstructured version by 2.36x with the effect verified causal by a
scramble control. Applying that here: ONE shared function scores each
object slot from its own relative features (relative row, relative
column, absence, L1 distance), and the entry supplies one SIGN per
slot. Then whatever plane-1 visits teach about "an object this far
away is worth this much" transfers to plane-2 by construction, and the
only per-slot quantity that must be read is the sign — which F116
already measured the entry to carry cleanly (+-1, sign-split by twin).

Verified before launch: swapping the two objects in the state
exchanges their feature vectors exactly, and an absent slot produces
the zero-with-flag vector. 3 seeds running with plane-1-only
collection retained, since F129 shows reading needs that asymmetry.

Probe 229 is `--seek-plane2 0.25`, 2 seeds. Probe 230 is
`--tied-salience`, 3 seeds.

**F131 (probe 231). The decomposition answers: on the boolean ground,
holding DEPTH at one application produces the first reading signal
anywhere in the composition probes — so depth is the constraint and
carrying two piece-parameters in one entry is not.** 256 worlds, two
pieces, iterated, learned reader, `--max-len 1`:

| measure (held-out worlds) | own entry | stranger | withheld |
| --- | ---: | ---: | ---: |
| per-bit accuracy | **0.6096** | 0.5794 | 0.5806 |
| exact match | 0.0244 | 0.0249 | 0.0156 |

Chance is 0.5 per-bit and 0.0039 exact. The per-bit gap of own over
stranger is +0.0302 on held-out worlds, positive in 10 of 16 worlds
individually (range -0.050 to +0.112) — small and noisy, but it is the
first time in this probe family that supplying the CORRECT world's
entry beats supplying a foreign one. At depth 4 the same probe gives
stranger identical to own to four decimals (F120, F124).

Two things this settles and one it does not:
  * SETTLED — two pieces in one entry is not the blocker. The entry
    can carry both a mask and a shift well enough to beat a stranger's;
  * SETTLED — the boolean ground is not simply too hard. It reads,
    weakly, when depth is removed;
  * NOT SETTLED — whether the depth failure is gradient reach (the
    reader's signal must survive four step applications) or
    representational drift (the world parameter degrading in the
    latent). A depth ladder discriminates: gradient reach should decay
    smoothly with depth, drift should show a knee.

This also rehabilitates the curriculum idea (F127) as untested rather
than refuted: F127 ran on the arithmetic ground, whose multi-world
results are confounded by modular multiplication (F128). A curriculum
on the boolean ground, starting where reading demonstrably works,
is a different experiment.

Probe 231 is `bool_compose.py --iterate --max-len 1 --worlds 256`.
Depth ladder (L1 second seed, L2) running.

**F132 (probe 232). The world-count ladder shows a CLIFF between 1 and
4 worlds, not a decline — and putting it beside F131 gives the unified
statement this whole probe family has been circling: CONDITIONED
execution fails as soon as depth exceeds one. The single-world
successes work precisely because they need no conditioning.**

| worlds (arithmetic, iterated + ORACLE entry, depth<=4) | trained programs |
| ---: | ---: |
| 1 | **1.0000** |
| 4 | 0.0547 |
| 16 | 0.0623 |
| 64 | 0.0570 |
| 256 | 0.0548 |

Chance 0.0435. Four worlds is already total collapse, and 4 -> 256
changes nothing. So this was never a capacity or scale story: the
break is at the point where the step function must take its parameters
from CONTEXT rather than from weights. With one world, "add b" and
"multiply by a" are constants the weights absorb; with four, they must
be read — even when handed over as clean one-hot codes.

Laying every result in this family against that axis:

| pieces | depth | conditioning needed | result |
| --- | --- | --- | --- |
| 1 (additive) | 1 | yes, many worlds | **works** (F114, twin 0.0000) |
| 2 (boolean) | 1 | yes, 256 worlds | **weakly works** (F131, +0.030 per-bit over stranger) |
| 2 | <=4 | NO (one world) | **perfect** (F121/F125, 1.0000) |
| 2 | >=2 | yes, 4+ worlds | **chance** (F120,F122,F124,F126,F127,F128, here) |

Every cell is explained by two facts and nothing else: conditioning
works at depth 1, and it fails from depth 2 up.

**This sharpens F121 rather than overturning it, and the correction
matters.** F121/F125 demonstrate that a shared per-element step
composes, generalises to unseen arrangements, and extends to unseen
depths — all true, both seeds, at ceiling. What they do NOT
demonstrate is BANK-FED composition, because in a single world the
pieces live in the weights. The honest claim is: *compositional
execution works when the pieces are in weights; supplying the pieces
from context works only at depth 1.* The bank thesis needs both at
once, and that conjunction has never been achieved here.

The remaining question is now specific enough to attack directly: is
the depth failure gradient REACH (the reader's signal must survive
several step applications) or parameter DRIFT (the world parameter
degrading as the latent is rewritten)? The boolean depth-2 arm
discriminates. A third possibility the architecture invites: the entry
is re-attended at every step, so the step function must re-extract the
same parameters repeatedly — which suggests decoding the entry ONCE
into explicit per-piece parameters and applying a function of those,
the way an interpreter binds arguments before running a loop.

Probe 232 is the `--worlds 4/16/64` ladder with oracle entries.

**F133 (probe 233). The composition result replicates across GROUNDS,
not merely across seeds: the boolean single-world control gives
1.0000 / 1.0000, identical to the arithmetic F121.** Different pieces
(XOR-with-mask and rotate-by-k rather than add-b and multiply-by-a),
different output head (8 independent bits rather than a 23-way
softmax), different chance floor (0.0039 rather than 0.0435) — same
ceiling on held-out program arrangements, and per-bit accuracy is
1.0000 too, so it is exactly right rather than mostly right.

This is worth more than another seed. A seed replication says the
effect is not noise; a GROUND replication says the effect is not a
property of the particular function. The iterated shared step composes
because of what it is, not because modular arithmetic happens to suit
it.

It also removes the last alternative reading of F121. One could have
argued that add-and-multiply mod 23 is unusually compositional — both
pieces are affine, so their composition stays affine and a single
learned affine map might cover every program. XOR and rotation admit
no such collapse (XOR is not affine over the reals, and the pair
generates a non-abelian group), yet the result is identical.

So the settled half of the thesis is now firmly settled: **a shared
per-element step function composes, generalises to unseen
arrangements, extends to unseen depths, and does so on two unrelated
function families.** The unsettled half is unchanged and now stands
alone: it works with the pieces in WEIGHTS, and supplying them from
CONTEXT still fails above depth 1 (F131, F132).

Probe 233 is `bool_compose.py --iterate --worlds 2 --ignorance 0`.

**F134 (probe 234). The clean confirmation. On the boolean ground,
with the world's identity handed over exactly, the iterated plant
fails at depth 4 across 256 worlds — 0.5587 per-bit against 0.5 chance
and 1.0000 in the single-world control.** Every alternative
explanation has now been removed simultaneously:

  * not arithmetic hardness — XOR-with-mask and rotate-by-k, no
    grokking-class function anywhere;
  * not reading — oracle entries supply (b, k) exactly;
  * not diversity — F122 showed 10x is inert, and this is the same
    256 worlds;
  * not the interface — this IS the iterated interface, the one that
    scores 1.0000 on the same ground with one world (F133).

| boolean, 256 worlds, oracle entry, iterated | per-bit | exact |
| --- | ---: | ---: |
| own entry | 0.5587 | 0.0067 |
| stranger entry | 0.5439 | 0.0058 |
| withheld entry | 0.5043 | 0.0036 |
| single-world control (F133) | **1.0000** | **1.0000** |

The ordering own > stranger > withheld is intact, so the plant IS
using the entry — it simply cannot execute with it. The gap between
0.5587 and 1.0000 is the whole finding.

**Stated as plainly as it can be: the plant can apply a piece whose
parameters live in its weights, arbitrarily deep. It can apply a piece
whose parameters come from context, exactly once. It cannot do both.**
That conjunction is what the bank thesis requires — the bank exists
precisely to supply parameters from context, and multi-step tasks are
the ones worth having a bank for.

The next candidate, from the shape of the failure rather than from a
new idea: the entry is currently re-attended at EVERY step, so the
step function must re-extract the same (b, k) on each application and
any error in that extraction compounds with depth. An interpreter does
not work this way — it binds its arguments once and then runs the
loop over bound values. `--bind-params` decodes the entry ONCE into
one explicit parameter vector per piece token, and the step function
then sees only (latent, bound parameter). If depth stops mattering,
repeated re-extraction was the mechanism.

Probe 234 is `bool_compose.py --iterate --oracle-entry --worlds 256`.

**F135 (probe 235). SOLVED: binding the entry ONCE takes conditioned
execution at depth from 0.5548 to 0.9983 per-bit — 0.9872 exact — on
held-out worlds at 256 worlds and depth 4. The conjunction the bank
thesis requires is achieved.** The single change from F134's failing
configuration: instead of re-attending the entry at every step, decode
it once into one explicit parameter vector per piece token, then step
on (latent, bound parameter) with no further access to the entry.

| boolean, 256 worlds, depth<=4, held-out worlds | per-bit | exact |
| --- | ---: | ---: |
| re-attend per step, oracle entry (F134) | 0.5548 | 0.0063 |
| **bind once, oracle entry (F135)** | **0.9983** | **0.9872** |
| bind once, oracle, STRANGER entry | 0.5474 | — |
| bind once, LEARNED reader | 0.5283 | 0.0096 |
| chance | 0.5000 | 0.0039 |

Stranger entries stay at 0.547 while own entries reach 0.998, so the
entire effect is causal on supplying the CORRECT world — this is not a
model that learned the average world and stopped needing context.

**The mechanism, and it is a real architectural principle rather than
a tuning trick.** Attending the entry at each step means re-deriving
the same (b, k) on every application; each derivation carries error,
and the errors compound multiplicatively with depth — which is exactly
why performance was fine at depth 1 (F131) and gone by depth 4
(F120/F124/F134). An interpreter never does this: it binds arguments
once on entry and then runs the loop over bound values. Doing the same
removes depth as a factor entirely.

This is the third instance of one pattern in this project, and the
pattern is now worth stating as a design rule: **do the
context-dependent work ONCE, then iterate something fixed.** F67 —
store facts, derive behaviour by search, do not store a policy. F121 —
apply one piece at a time through a shared step, do not learn the
composite. F135 — bind the parameters once, do not re-read them per
step. Each time, the failure was re-doing per-step what should have
been done per-task.

**What this leaves, stated exactly.** With the plant able to execute
from bound context at depth, the ONLY remaining gap is the reader:
with the learned reader, own-entry and stranger-entry accuracy are
identical to four decimals (0.5283 vs 0.5283). So the target is now
unambiguous in the way F110 made the games' target unambiguous — not
"composition", not "depth", not "diversity", but: produce entries the
plant can bind. The plant's side is done at 0.9983.

Probe 235 is `bool_compose.py --iterate --bind-params`, oracle and
learned-reader arms.

**F136 (probe 236). Two-phase training is a null, and worse than joint:
0.4973 per-bit against joint's 0.5283 (chance 0.5000). Freezing an
oracle-built plant does not give the reader a learnable target.** Both
phase splits (50% and 75% of updates on oracle entries before
freezing) leave own-entry and stranger-entry accuracy identical to
three decimals.

The reason is visible in F135's own mechanism. Binding requires the
entry to land in a NARROW region of entry space — the frozen plant
decodes it through one linear map into parameter vectors it has
learned to apply exactly. Hitting that region from observations, with
only task loss as a signal, is a much harder search than the
re-attention architecture posed, where the plant could extract partial
information in many ways.

**F137 (probe 237). And the depth ladder confirms that trade directly:
under the OLD re-attention architecture reading is weakly alive and
NON-monotonic in depth, peaking at 2 before collapsing at 4.**

| depth (re-attend, learned reader, 256 worlds) | own | stranger | gap |
| ---: | ---: | ---: | ---: |
| 1 (seed 69316) | 0.6096 | 0.5794 | +0.0302 |
| 1 (seed 69317) | 0.6123 | 0.5860 | +0.0263 |
| 2 | 0.6674 | 0.6222 | **+0.0453** |
| 4 | — | — | +0.0000 |

Not a smooth decay, so gradient reach alone does not explain it — a
knee between 2 and 4 fits representational collapse better. But the
more useful reading is the comparison ACROSS architectures:

| architecture | execution ceiling (oracle) | reading gap (learned) |
| --- | ---: | ---: |
| re-attend per step | 0.5548 | +0.03 to +0.05 |
| bind once (F135) | **0.9983** | **+0.0000** |

**The two architectures fail in opposite places, and neither is
strictly better yet.** Re-attention reads a little and executes not at
all; binding executes almost perfectly and reads not at all. The
mechanism explains both: re-attention offers many partial paths to
use an entry (easy to learn, impossible to compound), binding offers
one exact path (impossible to find by search, perfect once found).

That is a genuinely useful diagnosis rather than a defeat, because it
names what to build: keep binding's execution and give the reader a
route to the narrow target. The distillation arm decides which half
is at fault — train the reader to match the oracle entry directly (a
privileged target, so a DIAGNOSTIC and not a mechanism), then evaluate
task performance through the frozen plant using reader entries. If
performance jumps toward 0.99, the reader is capable and only the
training signal is missing; if it does not, the reader or its inputs
cannot represent what binding needs, and the entry interface itself
has to change.

Probe 236 is `--two-phase 0.5/0.75`; probe 237 is the depth ladder.

**F138 (probe 238). The reader IS capable: trained to produce the entry
the bound plant needs, it drives 0.9723 / 0.9478 per-bit on HELD-OUT
WORLDS — worlds it has never seen — against the 0.9983 oracle ceiling.
So the missing piece is the reader's training SIGNAL, not the reader,
its inputs, or the entry interface.**

| boolean, 256 worlds, depth<=4, held-out worlds | per-bit | exact |
| --- | ---: | ---: |
| oracle entry, bound (F135) | 0.9983 | 0.9872 |
| **reader entry, distilled (F138)** | **0.9723 / 0.9478** | 0.8894 / 0.6775 |
| reader entry, task loss (F136) | 0.4973 | 0.0027 |
| reader entry, joint training (F135) | 0.5283 / 0.5144 | 0.0096 |
| stranger entry | 0.5488 / 0.5936 | — |
| chance | 0.5000 | 0.0039 |

Read the third and fourth rows against the second: the SAME reader
architecture, on the SAME inputs, producing entries for the SAME
frozen plant, goes from chance to near-ceiling purely by changing what
it is trained against. Nothing about capacity, representation, or
interface differs between those rows.

**What this means for the thesis, stated carefully.** On held-out
worlds the reader watches a handful of interactions, emits an entry in
one forward pass with zero gradient steps, and that entry drives
multi-step execution at 0.97 where a stranger's entry gives 0.55. That
is the bank mechanism working end-to-end — a fixed plant, a growing
external store, novel worlds, no weight updates at acquisition — with
one asterisk that must not be lost: **the reader was trained against a
privileged target**, so this is a capability result, not a solution.
The oracle entry is built from the world's true parameters, which a
real system does not have.

**The path this opens, and why it is not privileged.** Distillation
worked because it gave the reader a target that was CONSISTENT across
worlds — the same world always maps to the same entry, different
worlds to different ones. That property does not require knowing the
parameters. A learner always knows which observations came from the
same episode, so it can train the reader contrastively: entries from
the same world pulled together, entries from different worlds pushed
apart. Then freeze the reader and train the plant to bind whatever
code the reader settled on. Phase order reversed from F136, and the
privileged information disappears — F44's rule is respected too, since
the reader still sees only consequences.

This also explains F136's failure precisely rather than vaguely: task
loss through a frozen plant asks the reader to find one specific point
in entry space by search; a contrastive objective asks it only to be
CONSISTENT and DISCRIMINATIVE, and then lets the plant come to it.

Probe 238 is `--two-phase 0.5 --distill`, 2 seeds.

**F139 (probe 239). Contrastive reader pre-training is the best
NON-PRIVILEGED result so far — it beats every other reader training
scheme — but it recovers only a fraction of what distillation showed
possible.** Reader trained alone so that two readings of the SAME
world agree and different worlds do not (InfoNCE, batch of 8 worlds,
verified to behave correctly before launch), then frozen, then the
plant trained to bind its code. Held-out worlds, per-bit:

| reader training | own | stranger | gap |
| --- | ---: | ---: | ---: |
| distilled onto oracle entry (PRIVILEGED, F138) | 0.9723 | 0.549 | +0.42 |
| **contrastive 0.5 (F139)** | **0.6136** | 0.538 | **+0.076** |
| **contrastive 0.25, seed 69317** | **0.6287** | 0.580 | +0.049 |
| contrastive 0.25, seed 69316 | 0.5646 | 0.541 | +0.024 |
| joint training (F135) | 0.5283 | 0.5283 | +0.000 |
| task loss through frozen plant (F136) | 0.4973 | — | +0.000 |

So the ordering is unambiguous — contrastive > joint > task-loss, and
contrastive is the first non-privileged scheme with a real gap — but
0.61 against 0.97 means most of the achievable performance is still
unclaimed.

**The likely reason is a matching artefact I should have anticipated
in F135.** The oracle entry is a LINEAR projection of one-hot world
parameters, and the binder that decodes it is a single LINEAR map. A
linear decoder inverting a linear encoder is trivial, so the 0.9983
ceiling was measured under the friendliest possible pairing. A
contrastive code identifies the world just as well but arranges it
arbitrarily, and a linear binder cannot decode an arbitrary
arrangement. On that account contrastive is not producing worse
information, only differently-shaped information, and the binder is
the component that must change.

Two arms test exactly that, with the binder as the only variable:
a nonlinear (MLP) binder on F139's contrastive setting, and — the
control that matters for honesty — a nonlinear binder with the ORACLE
entry, to check whether the 0.9983 ceiling itself survives when the
encoder/decoder pairing is no longer matched. If the oracle ceiling
drops, part of F135's headline was an artefact and the ledger must say
so.

Probe 239 is `--contrastive 0.25/0.5`, 3 arms.

**F140 (probe 240). My F139 hypothesis is REFUTED, and the refutation
is more useful than the hypothesis would have been: giving the binder
capacity does not help the contrastive code and it DESTROYS the oracle
result — 0.9983 down to 0.6196.** The binder was the only change.

| binder | oracle entry | contrastive entry |
| --- | ---: | ---: |
| linear (F135/F139) | **0.9983** | 0.6136 / 0.6287 |
| nonlinear MLP (F140) | **0.6196** | 0.5616 / 0.6324 |

Two conclusions, one reassuring and one corrective:

  * **F135's ceiling is genuine, not a matching artefact.** I suspected
    the 0.9983 came from a linear decoder trivially inverting a linear
    encoder. If that were the story, adding decoder capacity would
    have preserved or improved it. It collapsed instead, so the linear
    binder is doing real work rather than exploiting a pairing.
  * **Capacity in the conditioning path hurts — for the fourth time in
    this project.** F77 (FiLM), F89 (more bank tokens), F79 (larger
    pools with small diversity), now F140. The consistent shape: any
    extra freedom in HOW context is applied gets spent on fitting
    rather than on conditioning. This is now reliable enough to use as
    a prior rather than re-testing each time — when a
    context-conditioned path underperforms, the answer is never more
    capacity in that path.

So the contrastive shortfall is not decoder expressiveness. A
contrastive code is discriminative but arbitrarily arranged, the
binder must stay simple, and a simple binder needs a code that is
already shaped for it — which nothing in the contrastive phase
supplies, because the plant does not exist yet when the reader is
being pre-trained.

The correction that follows: **stop making it a phase.** Run the
contrastive term as an AUXILIARY loss alongside task loss, so the task
supplies the code's SHAPE while the contrastive term supplies the
GRADIENT that breaks F106's deadlock, with neither waiting for the
other. This is the first scheme in the sequence where the two
requirements on the entry — be discriminative, be bindable — are
optimised at the same time rather than in series. Weights 0.3, 1.0,
3.0 running.

Probe 240 is `--deep-binder`, oracle and contrastive arms.

**F141 (probe 241). Tied salience gives the best games number yet
(+0.0995 pooled) and — on one seed of three — the first inverted
worlds that actually SEEK: top=food 0.417 against 0.000-0.042
everywhere previously. But it does not do so reliably, and the honest
summary is a proof of existence with high seed variance.**

| arm | held-out (3 seeds) | pooled | entry effect | top=food inverted |
| --- | --- | ---: | ---: | ---: |
| two-channel, untied (F118) | +0.0811/+0.0972/+0.1058 | +0.0947 | +0.24..+0.30 | 0.021 |
| **tied (F141)** | +0.1175/+0.0794/+0.1017 | **+0.0995** | **+0.28..+0.32** | **0.417 / 0.042 / 0.000** |
| oracle-value target | | +0.1234 | | |

Seed 69316 reaches +0.1175, within +0.006 of what perfect values buy
through this search, and is the first configuration in the entire
games sequence where inverted worlds rank food on top a meaningful
fraction of the time. Seeds 69317 and 69318 reproduce the old
behaviour exactly (0.042, 0.000) while still improving slightly on
reward. Entry effects are the largest measured at every seed.

So the mechanism CAN do what F130 predicted — sharing the salience map
across object slots lets plane-1 experience transfer to plane-2 — and
it does not do so dependably. Reporting the pooled number alone would
hide that; reporting only seed 69316 would be selection. Both are
above.

**An observation from F135 that reframes the games' signed pathway,
and it was there all along.** The signed term computes
`tanh(polarity(entry.mean(0))) * salience(state)` — the entry is
reduced to a per-world scalar ONCE and multiplied in, while the
attention path consults the entry afresh at every search step. That
is exactly the bind-once/re-attend split F135 measured, sitting inside
the games probe unremarked. And it is the bound half that carries the
result: every gain from F113 onward came from the signed term, while
the attention path's contribution has never been isolated.

That predicts something testable: binding the games' value pathway
fully — deriving all entry-dependent quantities once per episode
rather than per rollout step — should behave like F135 did, and the
seed variance in tied salience may be the residue of the un-bound
attention path competing with the bound one. Untested; it is the
obvious next games probe and follows from a measurement rather than a
hunch.

Probe 241 is `game_slots.py --tied-salience`, 3 seeds.

**F142 (probe 242). Running the contrastive term as an AUXILIARY loss
during joint training — rather than as a frozen pre-training phase —
closes roughly half the remaining gap to the privileged ceiling using
no privileged information at all: 0.7069 per-bit on held-out worlds,
exact match 0.178 against 0.0096 for joint training.** And the weight
has an interior optimum, the same shape as the ignorance weight curve
(F108):

| contrastive-aux weight | held-out per-bit | exact |
| ---: | ---: | ---: |
| 0.3 | 0.6453 | 0.1286 |
| **1.0** | **0.7069** | **0.1780** |
| 3.0 | 0.5795 | 0.0449 |

The full ladder of reader training schemes, all on the same probe,
same plant, same inputs:

| scheme | held-out per-bit | privileged? |
| --- | ---: | :---: |
| task loss through frozen plant (F136) | 0.4973 | no |
| joint training (F135) | 0.5283 | no |
| contrastive PHASE, then plant (F139) | 0.6136 / 0.6287 | no |
| **contrastive AUXILIARY, w=1.0 (F142)** | **0.7069** | **no** |
| distilled onto oracle entry (F138) | 0.9723 / 0.9478 | YES |
| chance | 0.5000 | |

Exact-match rose 18x from joint training (0.0096 -> 0.1780), which
matters more than the per-bit figure: getting all eight bits right
requires the entry to specify the world, not merely to correlate
with it.

**Why the auxiliary form beats the phase form, which is the reusable
part.** The entry must satisfy two requirements at once — be
DISCRIMINATIVE (identify the world) and be BINDABLE (land where a
simple linear binder can decode it). A phase can only supply one at a
time, and the reader is frozen before the plant exists to express the
second. Optimised together, the task loss shapes the code while the
contrastive term keeps it from collapsing, and neither waits. The
interior optimum says the same thing from the other side: at weight
3.0 discriminability dominates and bindability is lost, dropping below
even the phase result.

Remaining to the privileged ceiling: 0.7069 against 0.9723. The
cheapest untried lever is the contrastive task's difficulty — the
InfoNCE batch is 8 worlds, so the reader need only tell one world from
seven, which a coarse code achieves. A larger batch demands a finer
code. That is a one-line change and follows the same logic as F78's
diversity law, applied to the reader's objective rather than the
plant's data.

Probe 242 is `--contrastive-aux 0.3/1.0/3.0`.

**CORRECTION to F142 (2026-08-10, same day). The headline was a
SINGLE-SEED result and the replication does not support it. Seed 69317
gives 0.5405 against seed 69316's 0.7069 — barely above the 0.5283
joint-training baseline and nowhere near the "half the gap closed"
claim.**

| contrastive-aux w=1.0 | held-out per-bit | exact |
| --- | ---: | ---: |
| seed 69316 | 0.7069 | 0.1780 |
| **seed 69317** | **0.5405** | **0.0161** |
| mean | 0.6237 | 0.0971 |
| contrastive PHASE (F139), 3 arms | 0.5646 / 0.6136 / 0.6287 | |
| joint training (F135) | 0.5283 | 0.0096 |

Two-seed mean 0.6237 sits inside the phase form's range, so **the
auxiliary form is not established as better than the phase form**, and
the weight curve (0.3 / 1.0 / 3.0) is single-seed throughout and
cannot carry the interpretation I gave it — the "interior optimum"
may be seed noise.

What survives: contrastive signal of either form beats joint training
and beats task-loss-through-a-frozen-plant. The ordering at the top of
the ladder does not.

**How this happened, since it is a process failure and not a
measurement one.** The project's standing rule is two or more seeds
before a promotion claim, and the sweep was three WEIGHTS at one seed
rather than one weight at three seeds. Sweeping a hyper-parameter
feels like replication because it produces several numbers, but every
number shares the same initialisation — so a seed-driven outlier
appears as a smooth curve with an optimum. F70 recorded the same
error class (a cost curve that changed shape when seeds were
widened), and it recurred here in a different costume.

Rule tightened: **a sweep is not a replication.** Any claimed optimum
must have its optimum point replicated at a second seed before it is
written as a finding, and the sweep must be reported as single-seed
until then. Applied retroactively — the F108 ignorance weight curve
should be re-checked the same way, since it has the identical shape
and provenance.

The batch ladder now running (32, 128) inherits this: whatever it
shows will be reported as single-seed until its best point is
replicated.

**F143 (probe 243). The binding principle transfers to the games and
essentially closes the gap to the oracle-value target: pooled held-out
+0.1229 against +0.1234, on three seeds, with the seed variance that
plagued F141 GONE.** The prediction was written down in F141's entry
before the run — that the games' beam search re-derives the world
parameters at every rollout step, exactly F135's depth killer, and
that the tied-salience seed variance was the un-bound attention path
competing with the bound one. Binding was the only change.

| arm | per-seed held-out | pooled | entry effect | inverted top=food |
| --- | --- | ---: | ---: | --- |
| tied only (F141) | +0.1175/+0.0794/+0.1017 | +0.0995 | +0.28..+0.32 | 0.417/0.042/0.000 |
| **+ bind-value (F143)** | **+0.0980/+0.1334/+0.1373** | **+0.1229** | **+0.30..+0.36** | **0.333/0.188/0.375** |
| oracle-value target | | +0.1234 | | |
| full oracle | | +0.1954 | | |

Three things, in order of how much they matter:

  * **Pooled +0.1229 against a +0.1234 target** — the learned value
    model now buys essentially what PERFECT values buy through this
    same search. Two of three seeds exceed it (+0.1334, +0.1373). The
    quantity F110 named as the dominant term in the games gap
    (+0.1439) is closed.
  * **All three seeds now seek in inverted worlds** (0.333, 0.188,
    0.375) where F141 had one seed working and two at zero, and normal
    worlds are saturated at 1.000 everywhere. The polarity asymmetry
    that has been the games' defining defect since F112 is gone.
  * **Entry effects are the largest ever measured** (+0.297 to
    +0.358), and the twin arm falls to -0.199..-0.221 — supplying the
    wrong world's entry is now more harmful than ever, which is what
    reading harder looks like.

The full games ladder against the floor-to-full-oracle span:

    F109  -0.0205   5.0%
    F111  +0.0069  17.0%   n-step value head
    F113  +0.0816  49.9%   signed entry
    F118  +0.0947  55.7%   two-channel salience
    F141  +0.0995  57.8%   tied salience
    F143  +0.1229  68.1%   BIND ONCE
    target +0.1234 68.3%   (oracle values, learned dynamics+search)
    full  +0.1954 100%     (oracle values AND oracle dynamics)

The remaining 31.9% is F110's search-and-dynamics residual, which the
value model was never going to touch — closing it needs a better
transition model or a better search, not a better entry.

**The methodological point worth keeping.** This is the first time in
the session that a mechanism found on the math ground transferred to
the games and worked on the first try, with three seeds and no
tuning. It transferred because it was a STRUCTURAL claim about
interfaces ("bind context once, then iterate") rather than a tuned
quantity, and structural claims are the ones that move between
domains. The findings that did not transfer — weights, curricula,
collection policies — were all quantities.

Probe 243 is `game_slots.py --tied-salience --bind-value`, 3 seeds.

**F144 (probe 244) — CONFIRMED at three seeds, see the confirmation
block below. Originally reported as provisional under the rule
tightened after F142's correction.**

**F144 (probe 244, as first written, SINGLE SEED). The contrastive batch ladder shows
a second interior optimum: batch 32 gives 0.7993 held-out per-bit and
0.3158 exact, the best non-privileged numbers measured.**

| contrastive batch (w=1.0, seed 69316) | held-out per-bit | exact |
| ---: | ---: | ---: |
| 8 | 0.7069 | 0.1780 |
| **32** | **0.7993** | **0.3158** |
| 128 | 0.5738 | 0.0606 |

This is exactly the shape F142 produced and exactly the shape that did
not survive replication there, so no claim is made until the best
point is reproduced. Two further seeds at batch 32 are running for
that purpose alone — not a sweep, a replication of one point.

If it holds, the reading is F78's diversity law applied to the
READER's objective: at batch 8 the reader distinguishes one world from
seven and a coarse code suffices; at 32 it must separate a world from
31 and the code has to be finer. The collapse at 128 would then be the
familiar over-shoot — the contrastive term dominating the task loss
and costing bindability, the same trade F142's weight curve suggested.

The honest caveat is that this interpretation is available for a
result that may be noise, and F142 is on record as having produced a
tidy mechanistic story for exactly such a curve one day earlier. The
story is written here so it can be checked, not because it is
believed.

Exact-match at 0.3158 would be the number that matters if it survives:
all eight bits correct requires the entry to SPECIFY the world, and
0.3158 against joint training's 0.0096 is a 33x change on the strict
measure rather than the lenient one.

**Process bug (2026-08-11). A width-4 boolean arm spun for 17 hours at
100% CPU without ever starting training, and nothing in the harness
noticed.** `make_worlds()` draws UNIQUE (mask, shift) pairs until it
has `--worlds` of them. At width 4 the family contains only
(2^4 - 1) x (4 - 1) = 45 distinct worlds, and the default request is
64, so the rejection loop could never terminate. The arm was launched
in the F120 batch and sat "pending" through roughly twenty subsequent
findings, holding a core the whole time.

Two things went wrong and only one of them is the code:

  * the loop failed silently instead of loudly — fixed, it now raises
    with the count available and the fix to apply;
  * **a run that never produces output is indistinguishable from a
    slow run in my monitoring.** Every monitor in this session watched
    for JSONs appearing and for processes disappearing; a process that
    is alive and producing nothing satisfies neither alarm. The
    27-hour-old entry was visible in the task list the entire time and
    I read past it repeatedly because "still running" was a plausible
    state.

Rule added: when a run's elapsed time exceeds roughly twice the
longest comparable arm, check it rather than assuming it is slow. The
comparable boolean arms finished in about 25 minutes; this one passed
that mark 40 times over.

Cost: one core for 17 hours, and the width-4 measurement — whether the
composition failure depended on output width — was never taken. It is
cheap to redo (`--width 4 --worlds 40`) but has been overtaken:
F135/F143 have since located the defect in the interface rather than
in anything width could have shown.

**F144 CONFIRMED (2026-08-11). The batch-32 optimum replicates: three
seeds give 0.7993 / 0.8447 / 0.6945, mean 0.7795 — every seed above
the batch-8 two-seed mean and far above joint training.** This is the
best NON-PRIVILEGED reader result measured, and unlike F142 it was
held back until its own best point was reproduced.

| scheme | held-out per-bit | exact | privileged? |
| --- | ---: | ---: | :---: |
| task loss through frozen plant (F136) | 0.4973 | 0.0027 | no |
| joint training (F135) | 0.5283 | 0.0096 | no |
| contrastive phase (F139) | 0.5646-0.6287 | — | no |
| contrastive aux, batch 8 (F142, 2 seeds) | 0.6237 | 0.0971 | no |
| **contrastive aux, batch 32 (F144, 3 seeds)** | **0.7795** | **0.2498** | **no** |
| distilled onto oracle entry (F138) | 0.9723 | 0.8894 | YES |
| chance | 0.5000 | 0.0039 | |

Closing 56.6% of the distance from joint training to the privileged
ceiling with no privileged information. Exact match — the strict
measure, requiring the entry to SPECIFY the world rather than
correlate with it — rose 26x over joint training, from 0.0096 to
0.2498.

The mechanism reading now has evidence behind it rather than a story
fitted to one curve: **the contrastive task's difficulty is the knob,
and it is F78's diversity law applied to the READER's objective.** At
batch 8 the reader separates one world from seven and a coarse code
suffices; at 32 it must separate one from 31 and the code must be
finer. The single-seed collapse at 128 is not confirmed and is not
claimed.

Seed spread remains wide (0.6945 to 0.8447), so the mean is the number
to quote and the best seed is not.

**Process note, worth as much as the finding.** F142 made this exact
claim one day earlier off a single seed and did not survive. The only
difference in procedure was replicating the best point before writing
it down, which cost two runs. The rule earned there — a sweep is not a
replication — is now the reason this result can be trusted.

**F108 re-check (2026-08-11): my own flag was WRONG, and the curve
stands.** After F142's single-seed failure I flagged F108's ignorance
weight curve as having "identical single-seed provenance" and queued
it for re-checking. It does not: the archive holds two seeds at every
one of the five weights. Recomputed per seed:

| ignorance weight | held-out (2 seeds) | mean | entry effect |
| ---: | --- | ---: | ---: |
| 0.1 | -0.0483 / -0.0461 | -0.0472 | -0.0015 |
| 0.25 | -0.0158 / -0.0491 | -0.0324 | +0.0250 |
| **0.5** | **-0.0201 / -0.0233** | **-0.0217** | **+0.0499** |
| 1.0 | -0.0223 / -0.0347 | -0.0285 | +0.0417 |
| 2.0 | -0.0483 / -0.0242 | -0.0362 | +0.0229 |

The optimum at 0.5 survives the scrutiny that killed F142's, and for a
reason visible in the numbers: it is the ONLY weight where the two
seeds agree closely (-0.0201 vs -0.0233), while every other weight has
a spread of 0.02-0.03 between seeds. It also wins on both metrics at
once. w=0.25 holds the best single value anywhere (-0.0158) and its
partner seed is nearly the worst (-0.0491) — precisely the pattern
that would have made a single-seed sweep report 0.25 as the optimum.

Recorded because the correction runs against my own interest: having
just been burned by a single-seed curve, the cheap move was to leave
the suspicion in the ledger. Checking cost one command and removed a
false claim about the project's own record. **Suspicion is not
evidence either — a flag raised on a hunch has to be tested with the
same discipline as a finding.**

**F145 (probe 245). SEARCH BUDGET IS NOT THE GAMES BOTTLENECK. 50%
more depth is worth +0.0015 and double the beam width is worth
-0.0003, both paired on the same two seeds.**

| arm | seed 69316 | seed 69317 | mean | delta |
| --- | ---: | ---: | ---: | ---: |
| depth 4, beam 4 (F143) | +0.0980 | +0.1334 | +0.1157 | — |
| depth 6, beam 4 | +0.1008 | +0.1337 | +0.1172 | **+0.0015** |
| depth 4, beam 8 | +0.0959 | +0.1350 | +0.1154 | **-0.0003** |

Both nulls, both paired, and the beam arm is negative. So F110's
"search and dynamics residual" — 31.9% of floor-to-full-oracle, the
last open quantity in the games — contains no search component worth
buying. Whatever remains sits in the MODEL, not in how hard we look
with it.

**The literature predicted this, which is the first time a citation on
this project has paid rent in advance.** Compounding model error means
each extra rollout step multiplies the model's error, so lookahead
buys less than the model loses; the standard practice is short
rollouts, often a single step. We were already at depth 4 with
`--freeze-objects` compensating for the least predictable slots, and
docs/LITERATURE.md flagged before these results landed that the
depth-6 arm was "running INTO this known headwind". It was.

**Where that leaves the residual, and the candidate is now specific.**
F109 measured the avatar slots predicted at 1.0000 and the object
slots at 0.67-0.77, and freezing the objects (using the observed
layout rather than a rollout) was worth 3.4 points. So the dynamics we
actually use are near-exact. What is NOT exact is the STATE: the slot
abstraction carries only the NEAREST object of each polarity, so a
world with three item pairs is described by two of its six objects,
and the oracle — which reads the true layout — sees all of them.

F109 tested a version of this and refuted it backwards (3-pair worlds
scored 33.6% against 19.8% for 1-pair worlds). That measurement stands
but its context does not: it was taken when the whole system captured
25% of headroom and the value model was the binding constraint. With
the value model now finished (+0.1229 against a +0.1234 target), the
abstraction is the obvious next suspect and deserves re-testing rather
than inheriting a verdict from a much weaker system.

Probe 245 is `game_slots.py --depth 6` and `--beam 8`, 2 seeds each.

**Caveat on F144, noticed from run times rather than results
(2026-08-11). The contrastive batch ladder did NOT hold compute fixed,
so part of the batch-32 improvement may be a compute effect rather
than a discrimination effect.** `contrastive_loss` builds two readings
per world in the batch, so reader forward passes per update scale
linearly with `--contrastive-batch`:

    batch   8:  16 reader passes per update   1x
    batch  32:  64 reader passes per update   4x
    batch  64: 128 reader passes per update   8x
    batch 128: 256 reader passes per update  16x

Every arm ran 40k updates, so the batch-32 arm did FOUR TIMES the
reader computation of the batch-8 arm. F144's confirmed improvement
(0.6237 -> 0.7795) is therefore an improvement at four times the
reader budget, and the ladder as a whole compares configurations at
1x, 4x and 16x. The winner being the more expensive arm is exactly
what a compute effect would look like.

This does not overturn F144 — the result replicated at three seeds and
the mechanism story is coherent — but it does mean the CAUSE is not
established. Two readings remain live:
  * discrimination difficulty (more negatives -> finer code), the
    interpretation recorded in F144;
  * reader optimisation budget (more passes -> better reader),
    which needs no reference to negatives at all.

The discriminating experiment is an equal-COMPUTE comparison: batch 8
at 160k updates against batch 32 at 40k. If batch 32 still wins, the
negatives matter; if they tie, it was the budget. Queued.

Noticed because the batch-64 arm has been running over two hours where
comparable arms took twenty-five minutes, which is 8x — matching the
table exactly. The run times were the evidence, not the outputs, which
is a reminder that cost is a measurement too and this project has been
reading only accuracy.

**F146 (probe 246). The codebook arms are a null caused by CODEBOOK
COLLAPSE — my implementation, not the idea. All four arms sit at
chance with stranger identical to own, and K=64 and K=256 produced
BYTE-IDENTICAL results, which is what gave it away.**

| arm | held-out own | stranger | exact |
| --- | ---: | ---: | ---: |
| codebook K=256, seed 69316 | 0.5414 | 0.5414 | 0.0123 |
| codebook K=256, seed 69317 | 0.5033 | 0.5033 | 0.0038 |
| codebook K=64, seed 69316 | 0.5414 | 0.5414 | 0.0123 |
| continuous entry (F144) | 0.7795 | ~0.55 | 0.2498 |

Two different codebook sizes cannot give identical results unless K is
irrelevant, and K is irrelevant only if the same code is always
chosen. Instrumented directly: **1 of 64 codes used, claiming 40 of 40
worlds.** The entry was constant, so the plant had nothing to read,
which is exactly the stranger==own signature.

**The mechanism is documented and I should have anticipated it.** At
initialisation every reader output is similar, so one code wins every
assignment; the losers never appear in a forward pass, never receive
gradient, and stay dead forever. This is the standard VQ-VAE failure
mode. Fix applied: periodic dead-code restart — every 500 updates,
codes with zero usage are re-seeded onto recent reader outputs with
small jitter. Verified: **24 distinct codes across 40 worlds, largest
claiming 4** where it was 1 claiming 40.

**What this does and does not say.** It says nothing about whether a
discrete bottleneck helps — that hypothesis is untested, because the
bottleneck was never discrete in practice, it was constant. The
LITERATURE.md ranking is unaffected on the merits.

The diagnostic worth keeping is the byte-identical comparison. Two
arms differing only in a hyper-parameter should never agree exactly;
when they do, that parameter is not reaching the computation. That is
a cheap, general check and it caught this in one command where the
accuracy numbers alone read as an ordinary null — I would have
recorded "discrete entries do not help" and moved on.

Probe 246 is `--codebook 64/256`, 2 seeds each, superseded by the
restart fix.

**F147 (probe 247). The F144 compute confound is RESOLVED, and against
my own suspicion: extra compute spent on more UPDATES helps, extra
compute spent on more NEGATIVES hurts. So the batch effect is about
negatives, not budget.** Two arms, both compared on seed 69316 where
F144 gives 0.7993:

| arm | reader compute | seed 69316 | seed 69317 |
| --- | ---: | ---: | ---: |
| batch 32, 40k updates (F144) | 4x | 0.7993 | 0.8447 |
| batch 64, 40k updates | **8x** | **0.5803** | 0.7846 |
| batch 32, 100k updates | **10x** | **0.8520** | **0.8889** |

The logic that settles it: if the batch ladder were measuring
optimisation budget, batch 64 at 8x would beat batch 32 at 4x. It does
not — it is much worse on the seed where both are measured. And when
the SAME extra compute is delivered as more updates at fixed batch,
performance improves (0.7993 -> 0.8520). Compute helps; negatives past
32 hurt. Those cannot both be the same variable.

So F144's mechanism reading survives its own caveat: the number of
negatives has a genuine interior optimum, and the collapse past it is
not an artefact of budget. The caveat was worth raising — it was live
until this measurement — and it is now closed by data rather than by
argument.

**Both long-training seeds improved: 0.8520 / 0.8889, mean 0.8704
against 0.7795 at 40k, with exact match rising 0.2498 -> 0.4228. That
is 47% of the entire remaining gap to the privileged ceiling, closed
by training longer and nothing else.**

**0.8704 is the best non-privileged number measured**, against
the 0.9723 privileged ceiling and 0.5283 for joint training. Simply
training longer at the confirmed batch closed a further third of the
remaining gap, which is worth stating plainly: before reaching for the
literature's mechanisms, part of what looked like a mechanism problem
was undertraining.

That does not dissolve the amortization diagnosis — 0.8520 is still
short of 0.9723 with the same reader on the same inputs — but it does
mean the gap to explain is smaller than the one that motivated
LITERATURE.md's addendum, and any refinement result must beat 0.8520
rather than 0.7795 to count.

Probe 247 is `--contrastive-batch 64` (2 seeds) and
`--contrastive-batch 32 --train-updates 100000`.

**The question F147 forces, and the run that answers it.** Training
2.5x longer closed 47% of the gap the amortization diagnosis was built
to explain. That diagnosis rests on a comparison — distilled 0.9723
versus non-privileged 0.7795 — whose right-hand side has now moved to
0.8704 and may not have stopped.

If the non-privileged scheme simply CONVERGES SLOWER and eventually
reaches the same place, then there is no amortization gap here at all,
LITERATURE.md's second addendum is answering a question that
dissolves, and the honest finding is "we under-trained and then went
looking for mechanisms". If it plateaus short of 0.9723, the gap is
real and the mechanisms are the right target.

A 200k-update pair is running to decide it. This is the shape of
mistake the project has made before and caught late: F117 spent three
arms on diversity when the substrate could not learn the function
(F128), and the check that would have settled it was cheap and ran
last. Running the cheap saturation check BEFORE investing further in
refinement, codebooks or alignment terms is the same lesson applied on
time rather than in hindsight.

Recorded now, before the result, so the prediction is on the record
either way.

**F148 (probe 248). BOTH literature-derived mechanisms are nulls.
Semi-amortization does nothing, and the discrete codebook does
nothing — while simply training longer (F147) moved the number more
than either.** Two seeds each, everything else at F144's configuration
and 40k updates so the comparison is matched.

| arm | own (2 seeds) | stranger gap | exact |
| --- | --- | ---: | ---: |
| baseline, no mechanism (F144) | 0.7993 / 0.8447 | ~+0.25 | 0.3158 / 0.3543 |
| **semi-amortization, 10 steps** | 0.7957 / 0.8364 | +0.2523 / +0.2372 | 0.2879 / 0.3147 |
| **codebook K=256 (restart fixed)** | 0.7854 / 0.8212 | +0.2609 / +0.2136 | 0.2576 / 0.4076 |
| longer training, 100k (F147) | **0.8520 / 0.8889** | | **0.3612 / 0.4845** |

**Semi-amortization: no effect, on either seed or either metric.** The
TTA hazard from LITERATURE.md §14 did not materialise either — the
stranger gap is unchanged rather than inflated, so refinement is not
finding a generically-agreeable entry; it is finding essentially the
entry the reader already produced. Ranked first on this page an hour
ago, on an argument that looked strong.

**Codebook: no effect.** And a retraction — on the first seed I
observed higher exact-match (0.4076 vs 0.3543) and read it as the
signature a discrete bottleneck should produce, all-or-nothing rather
than partially-right. The second seed reverses it (0.2576 vs 0.3158)
and the two-seed means are indistinguishable (0.3326 vs 0.3351). The
signature was noise. I flagged it as single-seed when I said it, which
is the only reason it is a retraction and not a false finding.

**What two nulls plus F147 point at.** Refinement attacks per-instance
inference; the codebook attacks the entry's form. Neither moved
anything, while more reader TRAINING moved 47% of the gap. That is
three pieces of evidence pointing the same way: the constraint is the
reader's optimisation, not the inference procedure and not the entry's
representation. The 200k saturation pair now carries more weight than
it did when launched — if it closes most of the remainder, the
amortization framing was wrong and the honest account is that we
under-trained and then went shopping for mechanisms.

**A caveat I will not use as an escape hatch.** Refinement ran at 10
steps and lr 0.05, unswept. It is possible more steps or a different
rate would help. But the literature's own claim is that ten steps from
a learned initialisation should be enough to matter, and the stranger
gap did not move at all — not by a little — which is what a
too-small-step-size result would look like. Sweeping is cheap and
worth doing before the mechanism is retired, and it is queued, not
assumed.

Probe 248 is `--refine 10` and `--codebook 256`, 2 seeds each.

**F149 (probe 249). The state abstraction is a null too: carrying TWO
nearest objects per plane instead of one is worth +0.0012, paired.**

| arm | seed 69316 | seed 69317 | mean |
| --- | ---: | ---: | ---: |
| 1 object per plane (F143) | +0.0980 | +0.1334 | +0.1157 |
| **2 objects per plane** | +0.1108 | +0.1230 | **+0.1169** |

One seed up, one seed down, mean flat. Entry effects and inverted-world
seeking are unchanged (0.292 / 0.312 against F143's 0.333 / 0.188).

So F109's original verdict — that the nearest-object abstraction is
not the limit — holds even now that the value model is finished, which
is the condition under which I argued it deserved re-testing. It did
deserve re-testing; the answer simply did not change. That is a
perfectly good outcome for a re-test and it removes the last
suspicion I had against F109.

**Where this leaves the games, with three candidates eliminated.**
The remaining gap is +0.1229 against +0.1954, and it is NOT:
  * the value model — closed, +0.1229 against a +0.1234 oracle-value
    target (F143);
  * the search budget — depth +0.0015, beam -0.0003 (F145);
  * the state abstraction — +0.0012 (here).

What is left is the STRUCTURE of the search rather than its size. One
concrete candidate the probe itself suggests: `--freeze-objects` holds
the observed object layout fixed through the rollout, which F109
measured as worth 3.4 points because rolling objects forward
compounded error. But items are CONSUMED and RESPAWN, so a frozen
layout tells the search that an object it just ate is still there. A
depth-4 plan can therefore count the same food twice. The oracle has
no such illusion.

That predicts a specific, checkable signature: the gap should be
largest in worlds where the avatar can reach several objects within
the horizon, and near zero in sparse worlds where it can reach at most
one. The per-world data to test this is already in every archived run
— no new training required — which makes it the next thing to do
rather than another 40k-update arm.

Probe 249 is `game_slots.py --objects 2`, 2 seeds.

**F150 (analysis, no new runs). The double-counting hypothesis from
F149 is REFUTED by its own predicted signature, tested against
archived per-world data at zero compute cost.** The prediction was
that a frozen object layout lets a depth-4 plan count the same food
twice, so the oracle-minus-learned gap should grow with how many
objects are reachable within the horizon. Measured per world, using
F110's oracle-value arm as the per-world ceiling (same seeds, so the
same held-out split):

| density | seed 69316 gap | seed 69317 gap |
| --- | ---: | ---: |
| 3 pairs | +0.0940 | — |
| 4 pairs | +0.0030 | -0.0216 |
| 5 pairs | -0.0176 | +0.0169 |
| radius 2 | — | +0.1089 |
| radius 3 | -0.0054 | -0.0114 |
| radius 4 | +0.0448 | -0.0205 |

The gap falls with density on one seed and rises on the other; the
radius ordering reverses between seeds outright. No consistent
signature, so the mechanism is not there.

Two things worth keeping from the analysis:

  * **Many gaps are NEGATIVE** — the learned system beats the
    oracle-VALUE arm on a majority of individual worlds. That is
    consistent with F143's battery result (+0.1229 against a +0.1234
    target, two seeds above it) and confirms it at world resolution
    rather than only in aggregate.
  * **The remaining residual is now bounded from four sides.** It is
    not the value model (F143), not search budget (F145), not the
    state abstraction (F149), and not density-dependent (here). What
    remains is the difference between our search WITH PERFECT VALUES
    (+0.1234) and an optimal policy that knows the hidden bit
    (+0.1954) — a structural property of the search that no amount of
    the same search fixes.

**The method is the point.** A hypothesis with a numeric signature
could be killed in one command against runs already on disk. I have
spent 40k-update arms on weaker hypotheses this session; checking
whether the prediction has a fingerprint in existing data should come
before allocating compute, and this is the second time today it would
have saved a run (the first being the cost table that resolved F144).

**F151 (analysis). A measured arithmetic defect in the beam's scoring
rule: it counts future rewards up to FOUR TIMES OVER, and this has
been true of every games result since F111.** The value head is
trained on the discounted H-step RETURN of a state-action, so
value(s_t, a_t) already contains r_t ... r_{t+H-1}. The beam then adds
that prediction at every depth with its own discount. Expanding the
implied weight on each future reward, for H=4 and depth=4:

| reward | weight the beam applies | correct weight | over-counted |
| --- | ---: | ---: | ---: |
| r_0 | 1.000 | 1.000 | 1.0x |
| r_1 | 1.800 | 0.900 | **2.0x** |
| r_2 | 2.430 | 0.810 | **3.0x** |
| r_3 | 2.916 | 0.729 | **4.0x** |

A correct planning objective weights each reward by gamma^t exactly
once. This one weights the near future progressively more the further
ahead it is, which is close to the opposite of discounting.

**This explains F145 rather than merely coexisting with it.** Deeper
search makes the over-counting WORSE — depth 6 adds two more summands,
each re-counting the same rewards — so the extra lookahead and the
extra distortion cancel. "More depth buys nothing" is what a correct
search hitting a model limit looks like, and it is also what an
incorrect search buying accuracy and losing arithmetic looks like. I
read it as the first and it may be the second.

**Found by inspecting the scoring line while looking for something
else** — the density analysis in F150. Worth noting how it survived:
every number since F111 is internally consistent, replicates across
seeds, and responds sensibly to interventions, because the defect is
in the OBJECTIVE the search optimises rather than in any measurement.
A wrong objective optimised well produces clean, reproducible,
confidently wrong results, and no amount of seed discipline detects
it.

**What is running is a test, not the fix.** `--score terminal` scores
a plan by its final step's value alone, which counts the horizon once
(verified: no reward appears in two summands) but ignores rewards
collected at steps 0-2. It is therefore a clean test of whether
over-counting is hurting, not a correct planner. The correct version
needs an immediate-reward head plus a terminal bootstrap —
sum gamma^d r(s_d,a_d) + gamma^D V(s_D,a_D) — which is the standard
form and the next implementation step.

3 seeds of `--score terminal` running against F143's +0.1229.

**Codex log review, 2026-08-11 — and an epistemic trap I nearly walked
into.** The parallel session's log now ends with a summary whose
headline findings match ours to the DIGIT: composition going
"chance-to-perfect" in the single-world math test (our F121/F125),
"about 27 exceptions" restoring the walled environment (our F97's
exactly 27), the ignorance objective worth "about 22% of available
headroom" (our F107's 22.0%), diversity pushing novel-family reading
to "0.91-0.97 without gradient updates" (our F78 range), and capacity
not being the lever (our F77/F79/F89/F140).

My first reading was independent convergence, which would have been
powerful evidence. **It is not.** The log states plainly that it
begins by reading an EXPORT of our own session — "I read the full
export at [session transcript]... It is a game-focused research
clone" — and two export IDs appear 24 times between them. Those
numbers are our numbers, reflected back. Treating them as replication
would have put fabricated corroboration into this ledger.

**Rule this earns, and it is new:** before crediting an outside source
as independent confirmation, check whether it had access to the work
it appears to confirm. Matching to three significant figures across
five separate findings is not convergence, it is a citation loop, and
the tell was the precision itself — genuinely independent
replications of noisy quantities do not agree to the digit.

**What IS genuinely theirs and worth having.** Their earlier material
covers ground we never touched, so it is additive rather than
reflective:

  * **Retrieval cost, quantified.** Linear scanning 64 entries costs
    64 model evaluations — already MORE than minting a new entry. Our
    F85-F87 built retrieval and measured its accuracy but never
    compared its cost against the cost of just re-reading. If reading
    is cheaper than searching, a bank that must be searched is the
    wrong data structure, and that is a design-level objection to the
    architecture we have not answered.
  * **Failure modes we never hit**: per-task encoders and per-game
    adapters producing excellent metrics while failing causal audits
    because the skill migrated into the "shared" component; curricula
    that failed because they changed the required policy rather than
    making the same policy easier ("a valid easing axis must preserve
    and exercise the target behaviour's full output range");
    staged probe/addressing/execution failing where co-training
    worked.
  * **A methodological rule we derived independently today.** Their
    list includes "check whether a hypothesis has a predicted data
    signature before launching expensive runs" — which is exactly
    F150's lesson, reached here by killing the double-counting
    hypothesis from archived data. Independent derivation of a RULE is
    meaningful in a way that matching numbers is not, because the rule
    was not in the material they read.
  * **An honest negative on their side**: their self-addressing result
    cleared the complete gate on only 5 of 16 seeds, with acquisition
    rather than the bank mechanism the dominant failure.

**Codex log, full review (2026-08-11). Provenance confirmed and
worsened, plus four genuine challenges to our findings.**

**Provenance.** The suspicion recorded earlier is confirmed and is
stronger than I put it: one of the three imported exports contains
`7d2d988c-c6fa-4641-b361-ceacd938889d.jsonl` — THIS session's ID. The
log did not merely read a related project, it read us. Every matching
number in its summary is ours quoted back, including the
policy-vs-model table. Nothing in that block is corroboration.

**Four challenges to OUR findings, in descending seriousness.**

1. **A confound that could invalidate our composition result, and it
   is cheap to check.** They measured a composition scoring 0.883 that
   fell to 0.520 when later steps were prevented from RE-READING the
   raw observation — i.e. the apparent composition was partly the
   later steps looking the answer up again rather than operating on
   the carried state. **Checked our code directly: our
   `step_bound(token, hidden, params)` receives only the latent and
   the bound parameter, and `bits_of(x)` is consumed once before the
   loop. Our step function CANNOT re-read the input.** The confound
   does not apply, and it is now verified rather than assumed.

2. **Their shared step function BROKE on unseen ORDERS, where ours
   held.** After calibrating on two orderings they reached 0.672-0.703
   on unseen orderings while fresh controls reached 0.734-0.813 —
   inherited was WORSE than fresh, i.e. negative transfer. They also
   failed held-out triples (0.573/0.573/0.323) and, decisively, a
   BALANCED JOINT calibration on all sources at once still failed
   composition (0.6719), which rules out acquisition order, rank,
   forgetting and decoder normalisation as causes. Our F121/F125 pass
   held-out orders and unseen depths at 1.0000 — so either our task is
   easier (two pieces, mod-16/8-bit, versus their register machine) or
   their interpreter had a defect. **This is the sharpest live
   disagreement between the two projects and it should not be
   smoothed over.** The honest reading is that our result is real for
   our setting and their negative bounds how far it generalises.

3. **"Train longer" is rejected three separate times in their log**,
   against our F147 where it was the ONLY thing that helped
   (0.7795 -> 0.8704). Different systems, so not a contradiction of
   fact — but it does mean the F147 effect should not be assumed
   general, and the saturation pair now running is the right test.

4. **Their most robust boundary, reproduced from three directions:
   external context can SELECT existing computation but cannot INVENT
   new computation.** A genuinely new primitive against their frozen
   interpreter got zero transfer (tied fresh on one seed, 6x worse on
   the other). We have never tested this: every composition result we
   have reuses pieces the plant already knows. If it holds for us too,
   it bounds the thesis — a bank grows what you can DO with known
   operations, not the operations themselves.

**Their independent mechanisms worth trying** (not derived from us):

  * **frozen base + zero-initialised, value-only external residual**,
    with the context channel structurally zeroed in the hidden path —
    their single best mechanism, cutting acquisition roughly in half
    repeatedly;
  * **decouple the novelty threshold from the commit threshold**
    ("is this evidence new enough to stage?" vs "is the staged model
    accurate enough to commit?"): 10/24 -> 14/24 seeds, the single
    largest online gain in their file, and pure protocol;
  * **recursive rollout verification inside promotion** —
    one-step-good/recursively-bad was a whole failure class;
  * **recency-weighted + latest-token pooling** instead of order-blind
    mean/max (2/6 -> 3/6, 3/12 -> 5/12), which converges with
    LITERATURE.md §11's indictment of mean-pooling from the neural
    process literature.

**A silent-killer check they paid for and we should not.** Their
composition mechanism was inert for a long stretch because two
near-0.02 initialisations multiplied to instruction vectors at ~1e-4,
far below the interpreter's intended scale; fixing it moved
composition 0.742 -> 0.924. Our equivalent quantity is the bound
parameter's norm against the hidden state's. Measuring it now.

**Scale check (2026-08-11), prompted by the Codex log's silent
killer. Ours is clean.** Their composition mechanism sat inert for a
long stretch because two near-0.02 initialisations multiplied to
instruction vectors around 1e-4, far below the scale the interpreter
expected; fixing it moved composition 0.742 -> 0.924. The analogous
quantity here is the bound parameter's magnitude against the latent it
modulates:

    entry norm            28.08   (spread 27.74 - 28.22)
    bound parameters      21.97
    hidden state           4.84
    bound / hidden ratio    4.54   -> comparable, not inert

So F135's binding is doing real work at a real scale, and the null
results in F148 are not that failure mode wearing a disguise.

One observation the check surfaced incidentally: entry norms are
almost identical across worlds (27.74-28.22, a 1.7% spread), so worlds
are distinguished by DIRECTION alone and carry no magnitude
information. That is expected given the contrastive term normalises
before comparing, and it is benign — but it is worth knowing, because
a binder reading magnitude would have nothing to read.

**Two checks the log prompted, both now answered:** the step function
cannot re-read its input (verified in code), and the code scale is
sound (measured). Neither confound applies. Cost: one code inspection
and one 3000-update run — against a composition result that took
several arms to establish.

**F152 (probe 250). The scoring fix is a NULL: pooled +0.1264 against
+0.1229, three seeds paired. F151's arithmetic defect is real and cost
essentially nothing.**

| seed | summed (F143) | terminal (F151 fix) | delta |
| ---: | ---: | ---: | ---: |
| 69316 | +0.0980 | +0.1175 | +0.0195 |
| 69317 | +0.1334 | +0.1247 | -0.0087 |
| 69318 | +0.1373 | +0.1371 | -0.0002 |
| pooled | **+0.1229** | **+0.1264** | **+0.0035** |

**Why a 4x over-counting can cost nothing, which is the transferable
part.** Beam search consumes only the ARGMAX; any distortion that
preserves the ORDERING of candidate plans is invisible to it. The
over-counting inflated later rewards uniformly across plans, so plans
that reach food still outscored plans that do not. The objective was
arithmetically wrong and ordinally right. **An incorrect objective is
only as harmful as the re-ranking it causes** — which is worth
remembering before treating the next discovered defect as an
explanation.

I did present F151 as a candidate explanation for the games residual.
It is not one. That is now five eliminated: value model (F143), search
budget (F145), state abstraction (F149), density/double-counting
(F150), and scoring arithmetic (here).

**Caveat kept honest.** `--score terminal` fixes the over-counting but
introduces its own distortion — it ignores rewards collected at steps
0 to depth-1 entirely. So this is not a clean isolation of
over-counting; it swaps one biased objective for another. The
informative fact is that two quite differently-biased objectives give
the SAME result to within noise, which says the search is insensitive
to its scoring rule in this range — and that is a stronger statement
than either arm alone. The properly correct version (immediate-reward
head plus terminal bootstrap) remains unbuilt, and on this evidence
its expected value is low.

Probe 250 is `game_slots.py --score terminal`, 3 seeds.

**A domain-specificity error caught before it entered the ledger
(2026-08-11).** Building the instruction-set plant from
LITERATURE.md §23, my first version gave the plant a BITWISE basis —
AND, OR, XOR, NOT, shift over 8-bit words. It smoke-tested fine and
was one launch from producing numbers.

It was the wrong machine. A bitwise instruction set produces a
bit-manipulation specialist: a plant that could never touch the games,
in a project whose entire premise is ONE amodal controller with
domain-specific content pushed out to the bank. "Solve it with a
substrate tailored to the problem" is precisely the failure this
architecture exists to avoid, and I had reintroduced it while trying
to generalise the architecture.

**The fix, and why it is the right substrate.** The basis is now the
procedural operations `schema_families.py` already uses — NOOP, INC,
DEC, CINC, CDEC, COPY, SWAP over SLOTS x VALUES — promoted from "how
we generate rule families" to "what the plant executes". That
interface is domain-general by construction and not by assertion: the
SAME six-slot, eight-value state already carries both the procedural
rule families (F71 onward) and the grid games (F102 onward, avatar and
object coordinates). An instruction set over it must serve a
dial-turning family and a foraging grid with the same instructions,
because both are already written in those slots.

Design points recorded while building it:

  * **Conditionals are what make it a basis.** Without CINC/CDEC every
    program is a fixed permutation of the state and the plant need
    never branch on content. `--no-conditionals` is the control that
    separates "learned to sequence" from "learned to branch".
  * **The floor is the IDENTITY, not uniform chance.** Most
    instructions touch one slot, so copying the input unchanged scores
    0.4412 against a uniform 0.1250. Scoring against 0.125 would have
    made a do-nothing model look like a large success — the same class
    of error as F104's inversion-invariant policies and F145's
    baseline discipline. The probe reports both.
  * **Programs are SUPPLIED, not inferred.** Whether a reader can
    infer a program from observations is the next question and
    deliberately excluded; mixing execution with inference is how F117
    lost three arms.

The general lesson, which is worth more than the probe: when
generalising an architecture, check that the generalisation is not
just a NEW specialisation. The bitwise version would have produced
real numbers on a real capability and taught us nothing transferable.

**F153 (probe 251). The instruction-set plant's first result was a
BUG, not a finding — and our own F128 rule caught it in one command.**
Three arms came back at 0.1215 / 0.1270 / 0.1280 slot accuracy against
a uniform chance of 0.1250 and an IDENTITY floor of 0.44-0.47. So it
scored at chance, and well below simply copying the input unchanged,
after 40,000 updates.

The learning curve settled it before any interpretation: flat from
update 0 (0.1304) to update 37,500 (0.1203), total range 0.0215. A
model that never moves is not a hard task, it is a broken one.

**The check that found it is the rule F128 earned**: before spending
anything on diversity or mechanism, verify the target function is
expressible in the interface at all. Trained on ONE fixed program,
the plant sat at loss 2.0823 — and ln(8) = 2.079, i.e. exactly uniform
output. It could not fit a single program.

**Cause: no residual path.** A length-6 program through a 3-layer step
is 18 effective layers with nothing to skip through. Adding
`latent = norm(latent + step(...))` — one term — takes the same model
from 0.1146 to **1.0000** on a single program in 1500 steps, and the
full probe from flat-at-0.12-after-40k to 0.4432 after 1500 updates.

**Why this is worth a finding rather than a silent fix.** The three
failed arms included a controlled comparison (with and without
conditionals) that looked perfectly consistent — all three at chance,
tight agreement across seeds. A broken model produces beautifully
reproducible nulls, and the seed discipline that protects against
noise offers no protection at all against this. What protected against
it was an ABSOLUTE reference: the identity baseline. Scoring below
"do nothing" is impossible to rationalise, and I put that baseline in
because F104 and F145 had already taught the lesson. Without it, 0.12
against a 0.125 uniform chance would have read as an ordinary null and
the instruction-set route would have been wrongly abandoned.

Probe 251 is `isa_compose.py`, 3 arms, retracted and re-running.

**F154 (probe 252, interim — one arm of four). Weight decay 0.1 HURTS
badly, and the first learning curve this project has ever produced
shows a SMOOTH climb with no phase transition.**

    wd = 0.1, 40k updates, seed 69316:  0.5868 own,  0.0372 exact
    wd = 0.0, 40k updates, same seed:   0.7993 own,  0.3158 exact
    wd = 0.0, 100k updates, same seed:  0.8520 own,  0.3612 exact

Exact match collapses by a factor of eight. So the grokking
literature's "steps-to-generalisation scale inversely with weight
decay" does not hold at 0.1 here — at that strength it is not
accelerating a transition, it is damaging the model. The 0.01 arms
will say whether a gentler setting behaves differently; on this
evidence the mechanism is not free.

**The curve matters more than the number.** Sixteen points from update
0 to 37,500: 0.489, 0.537, 0.567, 0.548, 0.577, 0.610, 0.605, 0.631,
0.632, 0.630, 0.634, 0.632, 0.637, 0.647, 0.629, 0.645. That is a
smooth, decelerating, monotone-in-trend climb — **no plateau, no
transition, no bump.** It is what slow logarithmic convergence looks
like and it is not what grokking looks like.

That is one arm at a damaged setting, so it does not settle the shape
question; the clean wd=0 curves now running do. But it is the first
time in this project that ANY curve has been available to look at,
and the immediate lesson is that the answer was cheap all along: this
question was open for hours of speculation about amortization gaps,
semi-amortization and phase transitions, and sixteen numbers from one
run address it more directly than any of it.

If the clean curves agree, the consequence is deflationary and should
be stated plainly: the reader's "gap" is slow convergence, the
literature addenda 2 and 4 were answering a question that dissolves,
and F148's mechanism nulls were nulls because there was no mechanism
to fix.

**F154 CONFIRMED (probe 252). Weight decay 0.01 buys most of what 2.5x
more training buys, at 40% of the cost — two seeds, and it is the
FIRST literature-derived mechanism in this sequence that works.**

| arm | own (per-bit) | exact |
| --- | ---: | ---: |
| wd=0.00, 40k (F144, 3 seeds) | 0.7795 | 0.2498 |
| **wd=0.01, 40k (2 seeds)** | **0.8569** | **0.3509** |
| wd=0.00, 100k (F147, 2 seeds) | 0.8704 | 0.4228 |
| wd=0.10, 40k (2 seeds) | 0.6725 | 0.0903 |
| privileged ceiling (F138) | 0.9723 | 0.8894 |

Per-seed at 0.01: 0.8538 / 0.8599 — tight. Per-seed at 0.1:
0.5868 / 0.7583 — both bad, and the spread itself is a symptom.

So there is a clear interior optimum. At 0.01 weight decay recovers
about 85% of the gain that 2.5x more updates delivers while spending
40% of the compute; at 0.1 it is destructive, with exact-match down
almost fourfold against no decay at all. The grokking literature's
claim that steps-to-generalisation scale inversely with weight decay
holds at the right magnitude and inverts at the wrong one — which is
more useful than a uniform win, because it says the knob is real and
must be TUNED rather than switched on.

**Scoreboard for the literature programme, stated plainly.** Five
mechanisms were derived from published work and measured here:
semi-amortization (null, F148), discrete codebook (null, F148),
length curriculum (null, F127), two-phase frozen plant (null, F136),
weight decay (WORKS, here). One in five — and the one that worked is
by far the cheapest, a single optimiser argument that no probe in this
project had ever set across 22 probe files.

**But the curves say the mechanism is not the one the literature
proposed.** wd=0.01 climbs 0.489 -> 0.541 -> 0.564 -> 0.665 -> 0.682
-> 0.758 -> 0.787 -> 0.831 -> 0.849 -> ... -> 0.853. Steeper than
wd=0, and still SMOOTH — no plateau, no transition. Weight decay is
changing the RATE, not producing an earlier phase change. So the
intervention is vindicated while the grokking framing that suggested
it is not, and those are separate claims that happened to arrive
together.

Practical consequence, applied from here: 0.01 becomes the default for
this probe family. Every result measured at wd=0 was measured with an
optimiser setting we now know to be leaving roughly 0.08 on the table
at fixed budget — including F148's nulls, which were run at 40k and
may deserve one re-test at the corrected setting before they are
final.

**F155 (probe 253). THE RECIPE ARCHITECTURE WORKS END TO END. An
interpreter trained ONLY on random programs — never on any task —
executes unseen programs at 0.9978, DOUBLE-LENGTH programs at 0.9774,
and then has recipes SEARCHED for seven real task families it has
never seen, reaching 0.9247 mean held-out accuracy against a 0.5229
identity floor. 14 of 14 recipes beat the floor. Zero gradient steps
touch any family.**

The interpreter, on programs it never trained on:

| measure | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| unseen length-6 programs | 0.9978 | 0.9942 |
| **unseen length-12 (DOUBLE)** | **0.9774** | **0.9556** |
| identity floor | 0.467 | 0.444 |

Recipes found by SEARCH for real families, scored on held-out
transitions (seed 69316 / 69317, gain over that family's identity):

| family | held-out | identity | gain |
| --- | ---: | ---: | ---: |
| line | 0.9375 / 0.8672 | 0.11 / 0.12 | +0.83 / +0.75 |
| dial | 0.9896 / 0.9935 | 0.667 | +0.32 / +0.33 |
| toggle | 0.8099 / 0.8164 | 0.70 | +0.11 / +0.12 |
| **perm** | **1.0000 / 1.0000** | 0.500 | +0.50 / +0.50 |
| grid | 0.9668 / 0.9785 | 0.557 | +0.41 / +0.42 |
| proc0 | 0.9453 / 0.8672 | 0.87 / 0.12 | +0.08 / +0.75 |
| proc1 | 0.9238 / 0.8495 | 0.81 / 0.45 | +0.12 / +0.40 |

**What this settles.** The Codex log's most robust boundary — external
context can SELECT existing computation but cannot INVENT it — is
FALSE for a plant whose primitives are a basis rather than the task's
own operations. These seven families were never in the interpreter's
training distribution in any form; `line`, `dial`, `perm` and `grid`
are structurally unlike each other and unlike random programs. Their
rules were nonetheless captured by programs FOUND at deployment, and
executed by weights that never moved. `perm` at exactly 1.0000 on both
seeds is the clean case: an exact recipe exists in the basis and the
search found it.

So the boundary was never about external memory. It was a consequence
of choosing the task's operations as the plant's primitives, exactly
as LITERATURE.md §22 predicted, and changing the primitives dissolves
it.

**What makes the claim strong rather than merely good numbers:**
  * **no domain in the weights, structurally** — training samples
    random programs over random states, so there is no world present
    to memorise, and the plant is bit-identical before and after
    meeting a family;
  * **no gradient at acquisition** — a new family is handled by
    proposing programs and scoring them with the frozen interpreter
    against observed transitions;
  * **held-out scoring** — recipes are chosen on one sample of
    transitions and scored on a fresh one, and search-fit tracks
    held-out closely (e.g. 1.0000/1.0000 on perm, 0.9800/0.9668 on
    grid), so the recipes are not fitted to their observation sample;
  * **per-family identity floors** — these range from 0.11 to 0.87, so
    a single global baseline would have been meaningless. Every gain
    is against that family's own do-nothing score.

**What it does NOT show, stated so it is not overread.** Search is
random proposal over 3000 candidates, which works at program length 6
and will not scale as written — the space is 252^6. Nothing here
learns to propose. `toggle` gains least (+0.11) and is the family
whose rule is least likely to be exactly expressible in this basis, so
the boundary has moved to expressibility rather than vanished, which
is what was predicted. And this is execution plus search, not reading:
no reader infers a recipe from observations, it is found by trying.

Probe 253 is `isa_compose.py --synthesize 3000`, 2 seeds.

**F156 (probe 254). The transfer matrix: the scrambled control comes
back NEGATIVE, which validates the whole measurement — and the top
donor is CHAOS, a family with no rule at all.** 13 families x 13
targets, 2 seeds, plant trained on one source then FROZEN with only a
bank entry fitted per target.

Donor strength (mean advantage given to OTHER families, over an
untrained plant):

| source | donor | seeds |
| --- | ---: | --- |
| **chaos** | **+0.4883** | +0.4301 / +0.5465 |
| walled | +0.4335 | +0.4058 / +0.4613 |
| dial | +0.4215 | +0.3978 / +0.4453 |
| grid | +0.4201 | +0.3947 / +0.4455 |
| perm | +0.2680 | +0.2481 / +0.2879 |
| line | +0.0896 | +0.0934 / +0.0858 |
| toggle | +0.0874 | +0.1067 / +0.0682 |
| **scrambled (CONTROL)** | **-0.1741** | -0.1398 / -0.2085 |

**The control is the most important row.** Training on a
schema-DESTROYED family does not merely fail to help, it actively
HURTS by -0.17 on both seeds. So the matrix is measuring transferable
structure and not warm-started weights, and every positive row means
something. Had scrambled landed near the good donors the whole
exercise would have been void.

**A confound I checked before believing the ranking, and it is
partly real.** Donor strength correlates -0.452 with the source's SLOT
COUNT — narrow families donate more — and +0.285 with state count. The
two weakest donors are `line` (1 slot) and `toggle` (6 slots), and the
strongest are 2-slot families. So part of this ranking is structural
overlap with the target pool rather than "general skill", and the
donor ranking should NOT yet be read as a curriculum. A clean version
needs targets matched for slot count, or donor scores computed only
over targets of a different width.

**Chaos being the top donor is the finding worth chasing.** Chaos is a
random permutation table — it has NO rule to learn. A plant trained on
it cannot succeed by internalising dynamics, so the only thing it can
learn is to READ the entry, which is precisely the general skill this
project wants and has been installing by hand via the ignorance
objective (F107). If that reading survives the slot-count control, it
says something practical: **the highest-ROI pre-training task may be
one with no learnable rule at all**, because it forbids the shortcut
the ignorance objective exists to penalise.

Note the tension with the scrambled control, which is ALSO rule-less
and yet harmful. The difference between them is a real question rather
than a contradiction: `chaos` is a 64-state, 2-slot table drawn once;
`scrambled` destroys a specific family's schema while keeping its
shape. Whatever distinguishes them is the actual mechanism, and it is
not yet known.

Probe 254 is `transfer_matrix.py`, 13 families, 2 seeds.

**F156 addendum (same data, re-analysed at zero cost). The slot-count
confound does NOT explain the donor ranking — it survives the
control.** Recomputing each source's donor strength using ONLY targets
of a DIFFERENT slot width:

| source | slots | donor, all targets | donor, different-width targets |
| --- | ---: | ---: | ---: |
| chaos | 2 | +0.4883 | **+0.4704** |
| walled | 2 | +0.4335 | +0.4389 |
| grid | 2 | +0.4201 | +0.4217 |
| dial | 3 | +0.4215 | +0.4165 |
| perm | 4 | +0.2680 | +0.2424 |
| line | 1 | +0.0896 | +0.0896 |
| toggle | 6 | +0.0874 | +0.0874 |
| scrambled | — | -0.1741 | -0.1741 |

Same order, same magnitudes. So structural overlap with same-width
targets was not doing the work, and the ranking stands as a ranking.

What that leaves is a genuine effect rather than an artefact: **narrow
sources donate more.** `toggle` uses all six slots and donates +0.087;
`chaos`, `walled` and `grid` use two and donate four to five times as
much. A plausible reading, untested: a source that occupies every slot
lets the plant learn slot-specific habits, while a narrow one forces
whatever is learned to be about slots IN GENERAL — which is the
slot-symmetry argument from F73 arriving from the data side rather
than the architecture side.

And chaos still tops the table after the control, which sharpens the
question rather than answering it. A rule-less source donating most,
while the rule-less CONTROL donates worst, means "no rule" is not the
mechanism by itself. The remaining difference between them is that
chaos is a fixed random table the plant sees consistently, whereas
`scrambled` destroys a family's schema while preserving its shape.
The next probe is a scrambled family at chaos's exact shape — if it
then donates like chaos, the mechanism is shape; if it still hurts,
the mechanism is consistency.

Method note: this analysis cost one command against runs already on
disk, and it is the fourth time today that checking existing data
settled a question before compute was allocated (F150, the F144 cost
table, the F145 radius test, this).

**F157 (probe 255). Library reuse as implemented is a NULL: paired
cost ratio 0.944, cheaper in 7 of 11 comparable cases, with a spread
from 0.14 to 2.33. Storing solved recipes and sampling them uniformly
does not make later search meaningfully cheaper.**

Paired per family, same seed, same order, growth vs frozen library
(candidates tried; lower is cheaper):

| family | seed | grow | frozen | ratio |
| --- | ---: | ---: | ---: | ---: |
| dial | 69316 / 69317 | 514 / 288 | 411 / 452 | 1.25 / 0.64 |
| perm | 69316 / 69317 | 417 / 1785 | 2964 / 1492 | **0.14** / 1.20 |
| grid | 69316 / 69317 | 1215 / 1981 | 4730 / 851 | 0.26 / **2.33** |
| proc0 | 69316 / 69317 | 12084 / 8129 | 11496 / 9206 | 1.05 / 0.88 |
| proc1 | 69316 / 69317 | 10432 / 12053 | 14621 / 12071 | 0.71 / 1.00 |

The two largest effects point in OPPOSITE directions on the same
family across seeds (grid: 0.26 and 2.33), which is the signature of
noise rather than mechanism. Mean 0.944 is well inside that spread.

**The failure was predicted before the run, which is the useful
part.** When launching this I recorded: "fragments are appended whole
and chosen uniformly, so the library grows but doesn't get COMPRESSED
— useful fragments get diluted, and if the growth arm shows no gain
that is the first thing to fix rather than evidence against reuse."
The library grew 210 -> 242 entries, so a genuinely useful fragment
went from 1-in-210 to 1-in-242 sampling probability. Adding good
fragments to a uniformly-sampled pool makes each one RARER. The design
could not have worked, and saying so in advance is what makes this a
null on the implementation rather than on the idea.

**What the data also shows about the search itself.** Cost is
dominated by the family, not by position: `dial` costs ~300-500
candidates and `toggle` saturates the 24,000 budget in three of four
runs. So the search's difficulty range spans nearly two orders of
magnitude, and `toggle` — the family F155 already flagged as least
likely to be exactly expressible in this basis — is the one that
cannot be solved at all. Expressibility, not search strategy, is what
bounds that case.

**Next, in order:** (a) weight fragment sampling by usefulness rather
than uniformly — the minimum viable version of DreamCoder's
compression, and the direct fix for dilution; (b) keep a fragment only
if it shortens total description length across solved recipes; (c) a
learned proposer, which is the reader's real job in this architecture.

Probe 255 is `isa_compose.py --library`, growth and frozen arms,
2 seeds.

## F158 — select-vs-invent, confirmed by a VARIANCE collapse

The arm F155 was contrasted against, finally read. `bool_compose.py
--novel-ops` holds out sixteen worlds whose OPERATIONS the plant has
never executed: same (b, k) parameters, but XOR-mask becomes ADD and
rotate-LEFT becomes rotate-RIGHT. Everything else is the F135
bind-once architecture that reaches 0.9983 on known operations.

| worlds | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| known ops, trained programs | 0.4151 | 0.3550 |
| known ops, HELD-OUT programs | 0.3302 | 0.3478 |
| novel ops, trained programs | 0.2228 | 0.1208 |
| novel ops, HELD-OUT programs | 0.0052 | 0.1436 |
| withheld-entry control (novel) | 0.0033 | 0.0064 |

Chance is 0.0039. Composition over KNOWN operations generalises to
unseen programs on both seeds (0.33, 0.35 — some 85x chance). Novel
operations do not, and the two seeds disagree so violently on the
headline number (0.0052 vs 0.1436) that the mean is not the finding.

**The finding is in the variance.** Within each seed the novel-op
accuracy is nearly IDENTICAL across all sixteen worlds — 0.2057-0.2378
for one seed, 0.0825-0.1406 for the other. A model reading each world
from its own entry cannot produce that; per-world accuracy would vary
with the world. So the entry stops carrying usable information exactly
when the operations change.

**One hypothesis with a checkable signature, tested and refuted.** If
the plant read (b, k) correctly and then executed THE OPERATIONS IT WAS
TRAINED ON, its accuracy would equal the agreement rate between trained
and novel semantics — computable with no model at all. That baseline
gets the qualitative pattern right on both seeds, including seed
69317's surprising inversion where held-out programs beat trained ones
(baseline 0.0992 vs 0.0448; measured 0.1436 vs 0.1208). But it predicts
per-world standard deviations of 0.0749 / 0.0477 against measured
0.0071 / 0.0132 — an order of magnitude too much spread — and the
per-world correlations are inconsistent in sign (+0.344, -0.090,
-0.172, -0.559). The plant is not executing trained semantics on read
parameters. It has fallen back on something world-independent.

Recorded as the negative half of F155's contrast, and it is a real
one: the recipe architecture never has to invent an operation, because
a new world is a new ARRANGEMENT of a basis the interpreter already
executes. This probe measures the cost of the alternative.

Probe 256 is `bool_compose.py --novel-ops`, 2 seeds, 40k updates. The
refuted baseline is `nvops_signature.py` — arithmetic only, no
network.

**Codex log, final 10% (2026-08-11). One transferable audit, checked
here and clean; the rest is our own work reflected back.**

**Provenance, again and explicitly.** The log states its own source:
"I extracted the session at transcript.jsonl ... It is from the
separate `neural-computer-agent-games` repo." That is US. Its
"strongly validated findings" list — policy-in-weights interferes,
model-plus-search is robust, separate structure from content, bind
once then iterate, goal-factored memory composes, rules versus
episodic exceptions, internal diagnostics beat reward curves — is our
own ledger restated, and the numbers it quotes (0.573 vs 0.441,
0.977 -> 0.695 -> 0.441, the 27 exceptions, 0.97 distilled) are ours.
It is not corroboration and must never be counted as a second source.
The genuinely independent content is only what Codex did in ITS OWN
repository.

**The one transferable audit: fail closed on unknown.** Codex found
that its planner treated a content-addressed MISS as an ordinary
prediction, so an unknown transition silently became a zero-valued
state and could WIN the search. That is a real bug class and worth
checking here rather than assuming.

**Checked, and it does not bite.** Our analogue is the unused-slot
sentinel: `used = (states < VALUES).all(dim=0)` is computed from the
OBSERVED states, and successors are clamped `where(nexts < VALUES,
nexts, 0)`. If a slot were unused in states but sentinel in
successors, the truth would silently become 0 and a candidate could
"match" it. Ran the check across 19 families — seven hand-made
(line, dial, toggle, perm, grid, walled, chaos) and twelve
procedural — over every state and every action: **zero mask
mismatches.** The mask derived from states equals the mask derived
from successors in every family, so no undefined target is ever
scored as a match. Recorded as an audit that came back clean, not as
a fix.

**One methodological point worth keeping.** Codex's active-selection
arm failed three seeds, and the diagnosis was that a deterministic
ARGMAX probe cannot see a state change unless the logits cross the
decision boundary — sub-threshold changes are invisible to it. Our
synthesis search scores candidates by argmax slot match, so it has
the same blindness in principle. It matters less here than there,
because our search is random proposal with best-of-N rather than
hill-climbing, so there is no landscape to be flattened. Where it
COULD bite is what the library learns from: a family that never
reaches `--fit-target` still contributes its best candidate to the
statistics, and that candidate may be noise. Noted as a known limit
of the library arms rather than acted on mid-run.

**Nothing in this section changes the current direction.** Their
architectural conclusions are ours; their new work (a fail-closed
planner path, an active causal selector that failed at three seeds)
addresses mechanisms this repository does not have.

## F159 — the library, measured properly: storing programs pays a
## little, learning statistics pays nothing

F157's null was on the implementation, and the fix it named turned out
to be the wrong fix. Recorded in full because the wrong turn is the
informative part.

**The first fix could not fire.** Weight fragments by usefulness, keep
one only if it shortens description length — that was F157's plan and
it is DreamCoder's mechanism. Implemented over CONCRETE instruction
sequences it added ZERO fragments across nine families, and the arm
came out byte-identical to its own control. The arithmetic says why:
with NOPS x SLOTS x (SLOTS-1) = 210 distinct instructions, an exact
three-instruction run recurring twice across ~36 winning programs is a
coincidence we should never expect to observe. Caught in the smoke
test, before the compute.

**Fragments became OP-SKETCHES**: program shapes with arguments left
free, filled from learned slot marginals at proposal time.
"CINC then SWAP" recurs even when the slots differ, and it does — the
library grows 7 -> 153 with counts up to 13. Untrained, the sketch
proposer is EXACTLY uniform over the concrete atoms, so the arms start
from the same distribution and differ only in what they learn.

**Two harness faults, both found by reading results rather than by
review.**

1. *Pairing.* Search and observation sampling shared one generator.
   The search consumes a different number of draws in every arm, so
   from the second family onward each arm was solving a DIFFERENT
   observation sample. Per-family costs were not comparable at all.
   The observer is now seeded per family index.
2. *Concentration collapse.* Raw usefulness counts are ~24 per family,
   against a prior of 1, so ONE success made a used operation outweigh
   an unused one ten to one. The learned arms then spent the entire
   4000-candidate budget on families the uniform control solved in 30.
   Evidence is now one vote per family, with the strength swept
   instead of guessed.

**Three seeds, eight arms, two sequences.** Ratios are arm cost over
frozen-control cost, so below 1.0 is cheaper.

| arm | diverse | related |
| --- | ---: | ---: |
| uniform — stores whole programs | **0.929** | **0.707** |
| sketch@4 | 1.034 | 0.782 |
| marginal@1 — instruction statistics only | 0.974 | 0.981 |
| sketch@1 | 1.012 | 1.001 |
| marginal@0 / marginal@4 / sketch@0 | 0.97-1.00 | 0.96-0.99 |

**What survives.** `uniform` is below 1.0 in 6 of 6 sequence-seed
cells, at roughly 7% on the diverse sequence (0.899, 0.984, 0.903).
Every arm that learns STATISTICS about which instructions are good is
null, and several are worse than not learning at all. That is the
opposite of the direction the sketch machinery was built for.

**What does not survive, and it is the number that looked best.** One
seed had `uniform` at 0.288 on the related sequence with a textbook
reuse curve — costs of 429, 43, 302, 35 where frozen paid 8048, 3071,
6912, 8001. The next two seeds read 0.940 and 0.892. Reported here
only because it was already reported as promising, and because the
cause is now known: the related families were drawn from the RUN seed,
so "related" meant a different geometry per seed and the between-seed
variance was task variance, not search variance. The task set is now
fixed across seeds.

**F157's null was directionally right and I over-called it.** Its
paired per-family ratio was 0.944; this measurement gives 0.929 on a
different statistic. Both are small, both below 1. F157 had no power
to distinguish that from zero and I called it a null; the honest
description was always "too small to resolve at two seeds".

**What is not yet established, and it is the whole claim.** A stored
winner is a FULL-LENGTH element, so drawing one fills the program in a
single pick — that changes the proposal distribution whatever the
element contains. The 7% could be element length rather than reuse.
Probe 258 adds the causal null (`shuffled`: same count, same lengths,
RANDOM contents) and an instrument that observes the mechanism
directly rather than inferring it from cost — how many winning
programs were LITERALLY produced by an earlier family. The frozen arm
reads 0 recalled, so it is the coincidence baseline, and any nonzero
recall elsewhere is genuine reuse.

Probe 257 is `isa_compose.py --library-arms`, 3 seeds, 8 arms.

## F160 — CORRECTION: the expressibility hole is the MODULUS, not a
## missing pair operation

I have written twice, including in a prompt sent to the parallel Codex
agent, that `toggle` is "not exactly expressible in the basis" because
it flips a PAIR of slots at once and no instruction does that. **That
is wrong, and the reasoning was wrong in a way worth recording.**

`toggle` has values=2, and at values=2 a bit flip is `(bit + 1) mod 2`,
which is an increment. So flipping bits 0 and 1 is `INC 0 ; INC 1` —
two instructions the basis already has. The pair effect was never the
problem.

**The actual hole: our instructions do arithmetic mod VALUES=8, but
each family has its OWN value count.** `INC` computes
`(state[i] + 1) % 8`. On a slot holding 1 in a two-valued family that
gives 2, not 0. Verified directly: `INC 0 ; INC 1` reproduces toggle's
action 0 on exactly 50% of states, and the per-slot match rate is
[0.5, 0.5, 1.0, 1.0, 1.0, 1.0] — the two slots the action touches are
right half the time and the untouched four are always right. Correct
when the value is 0, wrong when it is 1.

**The hypothesis has a signature in F155's data and the signature is
there.** The hole can only bite a family whose recipe needs INC/DEC on
slots with fewer than VALUES values, so it predicts an ordering rather
than a simple correlation:

| family | values used | search fit | held out |
| --- | ---: | ---: | ---: |
| toggle | 2 | 0.8671 / 0.8802 | 0.8099 / 0.8164 |
| perm | 4 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| line, dial, grid | 8 | 0.894-1.000 | 0.867-0.994 |

`toggle` is the WORST of the five on both metrics on both seeds, which
is the prediction. `perm` uses only swaps, so the modulus never enters
its solution, and it is immune and perfect — which is the sharper half
of the prediction, because a naive "fewer values is harder" story would
have put perm in trouble too. The raw correlation of value count with
fit is only +0.419 precisely because perm breaks it, and perm breaking
it is the evidence.

**Why this matters more than the version I had wrong.** A missing pair
operation would be one gap plugged by one new primitive. A global
modulus is a STRUCTURAL mismatch between an instruction set with fixed
semantics and a task distribution where each family carries its own
value range — and it silently degrades every family that increments a
short-ranged slot, not just the one that fails loudly.

**The fix is domain-general and small**: make the modulus an ARGUMENT,
so an instruction is `(op, i, j, m)` and the interpreter learns modular
arithmetic parameterised by m, exactly as it already learns which slot
to touch. Nothing about any domain enters; the value range is
observable in the data. Next probe.

Found by taking a suggestion seriously enough to check whether our
basis really lacked the thing it proposed adding. It did not — and
looking properly turned up the real hole somewhere else.

## F161 — reuse is REAL, isolated against its own confound, observed
## directly, and small

Five seeds, four arms, two family sequences, one shared plant per seed.
Ratios are arm cost over the frozen control, so below 1.0 is cheaper.

| arm | diverse | related |
| --- | ---: | ---: |
| `uniform` — stores solved programs | **0.959** | **0.929** |
| `shuffled` — same count, same lengths, RANDOM contents | 1.055 | 1.022 |
| `sketch-e4` | 1.056 | 0.927 |

**The control is the finding.** A stored winner is a FULL-LENGTH
element, so drawing one fills the program in a single pick — that
changes the proposal distribution whatever the element contains, and
without separating it, "storing programs helps" is unfalsifiable.
`shuffled` holds element count and element length fixed and randomises
only the CONTENTS. It lands ABOVE 1.0 on both sequences. So the length
artefact does not help; it slightly HURTS, which is dilution showing up
exactly where F157 predicted it. The real programs land below 1.0. The
two controls sit on opposite sides of the frozen baseline.

Paired per family across seeds, `uniform` is cheaper than `shuffled` in
**34/55 diverse and 32/45 related family-seeds — 66 of 100** — with
median ratios 0.979 and 0.954.

**The mechanism was observed, not inferred.** Cost falling cannot tell
reuse from a lucky draw, so each family records how many of its winning
programs were LITERALLY produced by an earlier family:

* `frozen` recalls 0 of 44, 0 of 48, 0 of 44, 0 of 43 ... **zero in
  all ten cells.** Coincidental rediscovery of an earlier winner never
  happens, so the instrument has no false-positive rate and any nonzero
  recall elsewhere is genuine.
* `shuffled` also recalls zero — re-finding an earlier winner by search
  is effectively impossible unless it is IN the library. Recall
  requires storage.
* `uniform` recalls 7-23% of its winners; `sketch-e4` up to 44%.

**Dose-response.** Across the 20 arm-cells that can recall at all,
r(recall rate, cost ratio) = **-0.471**: more verbatim recall goes with
lower cost. That is the relationship the mechanism predicts, and it is
measured rather than assumed.

**Recall is necessary but not sufficient.** `sketch-e4` recalls MORE
than `uniform` and costs MORE on the diverse sequence (1.056 vs 0.959).
Recalling a program on a family that was cheap anyway saves nothing, so
the amount of reuse and the amount of saving are not proportional.

**Magnitude, stated plainly: about 7-10% against the frozen control and
9-11% against the correct one.** That is a real effect with a real
mechanism, and it does not solve the search bottleneck — F155's search
still spends thousands of candidates per action, and shaving a tenth
off that leaves the same problem. Reuse as currently built is an edge,
not a lever. The lever has to come from CONSTRAINING the search rather
than from improving the guesses fed into it, which is what makes effect
indexing and the F160 modulus fix the live directions.

Probe 258 is `isa_compose.py --library-arms --arms
frozen,uniform,shuffled,sketch-e4`, 5 seeds.

## F162 — the modulus fix works, and the effect splits EXACTLY along
## the line the diagnosis predicted

F160 diagnosed the expressibility hole as a global modulus: instructions
did arithmetic mod VALUES=8 while every family carries its own value
count. F161's probe made the modulus an instruction ARGUMENT,
`(op, i, j, m)`. Two seeds, 40k updates, everything else held.

**Expressibility, settled without a network.** With the modulus
argument, `toggle` is EXACTLY reproduced — all six actions, all 64
states, at most two instructions each. That is a property of the basis,
so it needs no training run to establish.

**The prediction had a sharp form and it held.** The argument can only
buy something where a family's value range is NARROWER than the global
VALUES; everywhere else it just widens the search from 210 instructions
to 1470. Split by value count, over 14 family-seeds:

| population | n | mean change in search fit |
| --- | ---: | ---: |
| families with values < 8 | 8 | **+0.0439** |
| families with values = 8 | 6 | **-0.0108** |

**Not one family in either group crosses.** Every short-ranged family
improved or stayed at ceiling (+0.021, +0.022, +0.024, +0.044, +0.087,
+0.153, and `perm` twice at 0.0000 because it was already 1.0000 and
uses only swaps, so the modulus never enters its solution). Every
full-range family stayed flat or got slightly worse (0.000, 0.000,
0.000, -0.017, -0.019, -0.029). Zero crossovers in 14.

The two biggest gains are the two procedurally generated families with
the narrowest ranges — `proc1` at values=4 goes 0.8471 -> 1.0000 and
`proc0` at values=5 goes 0.9134 -> 1.0000. `toggle`, the family that
prompted all of this, gains +0.021 and +0.022 on fit and +0.025 and
+0.027 on held-out.

**The cost is real and it is stated.** Executing unseen programs is
preserved: 0.9948 / 0.9818 slots against 0.9940 / 0.9879 without. But
DOUBLE-LENGTH extrapolation degrades, 0.9358 / 0.9071 against 0.9950 /
0.9473. A seven-times wider instruction set is harder to extrapolate
from, and full-range families pay about one point of fit for an
expressibility they cannot use.

**The obvious next move, and it should remove the cost entirely.** The
modulus does not need to be SEARCHED. Each slot's value range is
directly observable in the transitions, so the search can fix m per
slot from the data rather than exploring seven of them. That is not
domain knowledge — reading a value range off observations names no
domain — and it predicts the +0.044 on short-ranged families with none
of the -0.011 on the others. This is the arm to run next.

Probe 259 is `isa_compose.py --moduli`, 2 seeds against 2 controls.

## F163 — the coverage filter is a NULL, for a reason worth keeping

The sound version of effect indexing: reject a candidate before running
the interpreter unless every slot the action CHANGED is written by some
instruction in it. A slot no instruction writes cannot change, so this
cannot exclude a program that would have fitted — unlike the per
instruction filter tried first, which is unsound (a correct program may
write a scratch slot and restore it) and was measured excluding its own
solution, fit 0.887 -> 0.682.

It prunes hard and costs nothing: 33-58% of candidates rejected with no
forward pass. And it buys **nothing** — 1.029 and 1.020 against the
frozen control on evaluations-to-target.

**My first explanation was wrong, and measuring it refuted it.** I
wrote that the filter is sound for EXACT fitting while the search stops
at 0.95, so a slot changing in only 2 of 64 rows would leave a program
scoring 62/64 there and be rejected despite being acceptable. That
story predicts changed slots with small mismatch mass. There are none.
Measured across seven families and every action, the mismatch mass of a
changed slot ranges from 0.167 to 1.000 of the total, and **34 of 34
exceed the 5% error budget on their own** — the smallest is more than
three times over.

So the repair that story implies — an attainable-fit bound that rejects
only when UNREACHABLE mismatches already exceed the budget — is
provably equivalent to the coverage filter on this family
distribution. Implemented, and it returned byte-identical numbers to
the coverage arm, which is the tell that two things are the same thing
rather than the tell that a flag is not wired.

**The real reason is that the measurement could not observe its own
hypothesis.** Both filter arms were smoke-tested against a plant
trained 3000 updates, where no search reaches 0.95 and every action
runs to the full budget: 20814 interpreter calls against 26 action
slots at a budget of 800 is 20800. When every search saturates, cost is
the budget by definition and no proposal filter can move it. The null
measured the budget, not the filter.

The honest status is therefore UNKNOWN rather than null. The filter can
only pay where searches terminate early, which needs a 40k plant, and
that measurement is now running.

## F164 — the reader's gap is a BUDGET problem with a seed-dependent
## ceiling, and the curve is smooth, not grokking

Two long runs, finally read. `bool_compose.py --iterate --bind-params
--contrastive-aux 1.0`, at 200k updates (five times the standard 40k)
and at 150k with held-out evaluation every 2500.

This settles the question F147 and F154 left open: is the reader's
shortfall an **amortization gap** — the network cannot express what an
optimised entry can — or is it **slow optimization**? F138 had shown
the reader is CAPABLE, reaching 0.9723 when distilled from a privileged
entry while ordinary task-loss training stayed near chance.

**It is budget.** Exact-match on held-out programs, trained worlds:

| updates | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| 40k (F133 era) | ~0.33 | ~0.35 |
| 150k | 0.3934 | 0.8672 |
| 200k | 0.3776 | **0.9372** |

Seed 69317 at 200k reaches 0.9372, against F138's privileged-entry
ceiling of 0.9723. The amortization gap is not fixed and not large —
it very nearly closes with more steps.

**The curve is smooth and saturating. It is NOT grokking.** Per-bit
accuracy on held-out programs, sampled every 2500 updates:

```
seed 69316   0.489  0.650  0.803  0.809  0.723  0.787  0.829  0.824  0.857  0.840  0.842  0.819
seed 69317   0.495  0.588  0.829  0.881  0.844  0.917  0.937  0.922  0.955  0.943  0.922  0.901
                0    12.5k   25k  37.5k    50k  62.5k    75k  87.5k   100k 112.5k   125k 137.5k
```

Both rise steadily from bit-chance, both bend over by 25k, and neither
shows the flat-then-jump shape that would have implicated a delayed
generalisation mechanism. So the weight-decay clock and the grokking
literature are the wrong frame for this.

**CORRECTION, same day.** I first wrote that the sixteen probes spent
on mechanism substitutes — semi-amortization, refinement, codebooks,
contrastive variants — were "all attacking a problem whose answer was
run it longer". That is an over-claim and it is unfair to those probes.
What the data supports is narrower and more useful:

> The amortization gap is not a representational impossibility.
> Sufficient optimisation closes it ON A FAVOURABLE TRAJECTORY.
> Training reliability and basin selection remain unresolved.

One seed reaching 0.9372 refutes an architectural ceiling near 0.35.
The other seed ending at 0.3776 after the SAME 200k updates means "run
it longer" is not a dependable solution — budget revealed the
attainable ceiling, it did not deliver access to it. The mechanism
probes may well have been aimed at optimisation speed and reliability,
which is exactly the problem still standing. They failed to improve it;
that is not the same as having had no problem to solve.

**The new question is the CEILING, not the gap.** The two seeds
converge to visibly different asymptotes, 0.84 and 0.95 per-bit, and
that difference survives 137,500 updates rather than closing. Seed
variance at this scale is not noise around one solution; it is two
different solutions. Both runs also show a late DECLINE — 69317 peaks
at 0.955 by 100k and falls to 0.901 by 137.5k — so more budget past the
peak is not free either.

Causal nulls hold throughout: withheld entry 0.005-0.009 against 0.0039
chance, and the stranger-entry control at 0.25 on the strong seed
against 0.73 with its own entry. The entry is doing the work.

**What this changes.** "The reader cannot learn to read" is retired.
The live questions are why two seeds settle at different ceilings, and
what the late decline is. Both are about the optimisation trajectory
rather than about the architecture, which is a much better place to be
than where this track has been since F117.

Probes 260 and 261 are `sat-200k` and `cv-shape`, 2 seeds each.

## F165 — the ceiling is READER FIDELITY, and which worlds fail is
## idiosyncratic rather than structural

F164 left two seeds converging to different asymptotes, 0.84 and 0.95
per-bit, and asked what sets the ceiling. Answered from data already
on disk, with no new run.

`context_fit` asks a narrow question: can the plant predict the very
rows the reader CONDITIONED ON — depth one, single piece, same entry.
Correlating it per world against held-out-program accuracy:

| run | n | r(reader fidelity, held-program bits) |
| --- | ---: | ---: |
| cv-shape 69316 | 8 | **+0.818** |
| cv-shape 69317 | 8 | **+0.842** |
| pooled | 16 | **+0.852** |

The worlds a model reads best are the worlds it generalises best on,
and the relation holds WITHIN each seed, so it is not an artefact of
the between-seed gap. If the executor were the limit, depth-one context
fit would be high everywhere and uncorrelated with deeper
generalisation; it is neither.

**Stated against itself:** the two metrics share the entry, so some
correlation is expected by construction. What the number adds is that
the variation is entry-side rather than execution-side, and F138's
distillation result — 0.9723 from a privileged entry — is the causal
version of the same claim.

**Which worlds fail has no structure.** Across sixteen held-out worlds,
reader fidelity is unrelated to the world's own parameters:
r(rotation k) = +0.118, r(popcount of the mask) = -0.249, and the
self-inverse rotation k=4 is not special (0.9271 against 0.9036). So
there is no sub-population with a systematic blind spot to attack — the
reader simply lands well or badly, and the dominant term is the seed:
sorted by fidelity, the bottom five worlds all belong to the weak seed
and the top three all to the strong one.

**What this means for where to push.** The target is reader fidelity as
a whole, not a repairable class of hard worlds, and the lever is the
optimisation trajectory — which solution the run settles into — rather
than capacity or representation. That is consistent with F164's smooth
non-grokking curve and with the late decline past 100k, and it makes
seed-to-seed variance the phenomenon to study rather than the noise to
average away.

## F166 — observing the modulus beats searching it, on BOTH halves of
## the prediction

F162 measured the modulus argument buying +0.0431 on families whose
value range is narrower than VALUES and costing -0.0074 on families at
full range. The stated next move was that the modulus never had to be
searched at all: each slot's range is visible in the transitions, so
fixing m per slot from the data should keep the gain and remove the
cost, because the instruction space returns to its original 210 once m
is determined by i.

Four seeds, seven families each, 40k updates. Training is IDENTICAL
between the searched and inferred arms at a given seed — both draw m
uniformly during training — so the plants are the same object and only
the search differs. Change in search fit against the no-modulus
control:

| population | n | modulus SEARCHED | modulus OBSERVED |
| --- | ---: | ---: | ---: |
| families with values < 8 | 16 | +0.0431 | **+0.0631** |
| families with values = 8 | 12 | -0.0074 | **+0.0044** |

**Both halves confirmed, and the gain is larger than predicted.** The
cost on full-range families is gone — it turns very slightly positive —
and the benefit on short-ranged families grows by half again, because
the search no longer wastes proposals on six wrong moduli per
instruction.

`toggle`, the family that started this and that F157 could not solve
inside a 24,000-candidate budget, now fits at 0.9944, 0.9838, 0.9719,
0.9701 across the four seeds — gains of +0.093 to +0.123 over the
control, against +0.021 to +0.036 when the modulus was searched.

**One column says the method has a flaw, and it is worth more than the
headline.** `line` holds eight values, so inferring its modulus should
change nothing. It swung +0.1061, -0.0286, -0.0735, 0.0000 across the
four seeds. That is what a modulus that is SOMETIMES TOO SMALL looks
like: the range was inferred from one action's rows, and if a slot's
top value never appears in that subsample, the inferred modulus is
below the true value count and INC wraps early and is simply wrong.

The asymmetry matters: over-estimating a range is harmless, since a
modulus above the true count is never reached, while under-estimating
is unsound. So I inferred from every observation instead of one
action's slice, which strictly enlarges the sample and can only move
the estimate up, and reran all four seeds.

**CORRECTION — the diagnosis was wrong, and the rerun proved it.** The
enlarged sample returned results IDENTICAL to the per-action version,
family by family and seed by seed, aggregates included (+0.0631 and
+0.0044 both times). Byte-identical output means the change did
nothing, so under-estimation was never happening.

Checked directly: `line` is inferred as modulus 8 on all six slots, and
per-action inference agrees with all-observation inference exactly. The
same holds for toggle (2), grid (8) and perm ([4,4,4,4,8,8] — the two
slots perm does not use correctly defaulting to 8). With 64
observations drawn over families of at most 512 states, slot coverage
is simply complete.

So the `line` swing is **plant-to-plant variation**, not inference
error: the no-modulus interpreter has no modulus embedding and one
legal modulus, the modulus interpreter has seven and is trained on all
of them, and on a family where the modulus is irrelevant the two
plants just differ. That is seed noise in a column where I read a
mechanism into it.

The enlarged sample is kept because it is sound and free, not because
it fixed anything. The general caution it answers — that the maximum of
a finite sample need not be the true cardinality — is correct in
principle and empty at this scale.

**Methodological note.** Byte-identical output across arms that should
differ has now caught three things: the collapsed codebook (F146), the
attainable-fit bound being the coverage filter in disguise (F163), and
this. It is the most reliable instrument in the kit, and it works
because it cannot be explained away.

Probe 262 is `isa_compose.py --infer-moduli`, 4 seeds against two
controls.

## F167 — the modulus result, split into its two hypotheses, with one
## confirmed, one unsettled, and one refuted

F166 was reported as a single win. It is not one claim, it is three,
and separating them changes what survives.

**Hypothesis 1a — inferring the modulus keeps the narrow-range gain.
CONFIRMED.** On cleanly paired seeds, families with values < VALUES
gain +0.0607 from the observed modulus against +0.0422 from the
searched one. Observing beats searching where the expressibility is
actually used.

**Hypothesis 1b — inferring removes the full-range search penalty.
STILL NOT ESTABLISHED after three more seeds, and F166 said it was.**
Over all four original seeds the full-range families read +0.0044,
penalty gone. Pooled over the five seeds whose arms are genuinely
paired, with thread count pinned: **-0.0014 observed against -0.0040
searched, n=15**. The penalty shrinks by about two thirds but does not
reach zero, and both numbers are small enough to be indistinguishable
from nothing. The honest statement is that observing the modulus does
not COST anything on families that cannot use it — not that it repays
them.

Hypothesis 1a strengthens on the same pooled data: **+0.0485 observed
against +0.0285 searched over 20 family-seeds.** Observing beats
searching wherever the expressibility is actually used, and that is
now five clean seeds.

**Hypothesis 2 — inferring repairs long-horizon extrapolation.
REFUTED, and it was right to keep it separate.**

| arm | unseen programs | DOUBLE length | drop |
| --- | ---: | ---: | ---: |
| no modulus | 0.9954 | 0.9854 | +0.0101 |
| modulus searched | 0.9909 | 0.9368 | +0.0541 |
| modulus observed | 0.9887 | 0.9373 | +0.0513 |

Removing the search dilution left extrapolation exactly where the
searched arm had it, 0.9373 against 0.9368.

**CORRECTION — that comparison was VACUOUS and should never have been
run as a test.** The searched and observed arms SHARE THEIR TRAINING;
they differ only in how the search proposes moduli. Their interpreter
metrics are therefore identical by construction, and match to the digit
at every cleanly paired seed. "Inferring does not repair extrapolation"
could not have come out any other way. It is a tautology wearing the
costume of a refutation, and I reported it as a finding.

The comparison that means something is no-modulus against modulus,
which really are different instruction sets and different plants. Over
seven seeds: 0.9632 against 0.9409, paired difference **+0.0222 with sd
0.0578**, no-modulus better in 6 of 7. So an extrapolation cost is
probably real, but it is THREE TIMES SMALLER than the 0.049 reported
from four seeds and it sits inside its own spread. Note also which arm
is unstable: no-modulus ranges 0.8210-0.9997 while modulus stays inside
0.9071-0.9644. The wider instruction set extrapolates slightly worse on
average and considerably more RELIABLY, which is not what "a wider
instruction set is harder to extrapolate from" predicts. On exact-match the loss is
starker still: 0.93 without the modulus against 0.71 with it either
way. So the extrapolation cost is a property of an interpreter trained
on a seven-times wider instruction set at the same budget, and it has
nothing to do with how the search proposes. Same-length execution is
barely touched; it is compounding over twelve steps that exposes it.
The obvious test is whether more training budget recovers it, which is
the same shape as F164 and is not yet run.

**The confound that broke the pairing, and it is a plain engineering
fault.** I wrote that training is identical between the searched and
observed arms at a given seed, so the plants are the same object.
Checked: identical for the two seeds launched together, DIFFERENT for
the two launched in separate batches. The batches used
OMP_NUM_THREADS=2 and 1, and floating-point reduction is not
associative, so the same seed trains to a different plant. The thread
count is now pinned inside the script and recorded in every report, so
pairing is a property of the code rather than of how it was invoked.

Worth noting which way this cut: at the contaminated seeds the observed
arm had the WORSE plant and still scored better on search fit, so the
1a result was measured against a headwind. It is 1b that the confound
was carrying.

## F168 — the coverage filter works after all; F163 measured the
## saturated regime

F163 reported the sound coverage filter as a null, 1.029 and 1.020. The
correction to that entry identified the fault: both arms had been
smoke-tested against a 3000-update plant where every search runs to its
full budget, so cost was the budget by definition and no proposal
filter could move it. Rerun against a 40k plant with the inferred
modulus, where **0 of 11 and 1 of 11 families saturate**:

| arm | diverse | related | proposals rejected |
| --- | ---: | ---: | ---: |
| frozen | 1.000 | 1.000 | 0% |
| coverage filter | **0.887** | **0.867** | 45% / 35% |
| attainable-fit bound | 0.887 | 0.867 | 45% / 35% |
| stores solved programs | 1.075 | 0.815 | 0% |

Interpreter calls fall 11-13%, and every one of the four
sequence-seed cells is below 1.0 (0.834, 0.939, 0.918, 0.817). The
filter rejects a third to a half of proposals with no forward pass, and
converts that into roughly a tenth fewer interpreter calls — the two
numbers differ because a rejected candidate would mostly have scored
badly anyway, so removing it enriches the pool by less than its share.

**Two seeds. Directionally clear, not yet promoted.**

The bound and the coverage filter are again identical to the digit,
which is the third confirmation that on this family distribution they
are the same predicate: every changed slot carries more mismatch mass
than a 5% budget allows, so "unreachable mass exceeds the budget" and
"a changed slot is unwritten" pick out the same candidates.

**An interaction worth flagging rather than concluding.** Storing
solved programs reads 1.075 on the diverse sequence here, against
F161's 0.959 over five seeds. The difference between the runs is the
inferred modulus. It is plausible that a better instruction set removes
the headroom reuse was filling, and it is equally plausible this is two
seeds of noise. F161 had five seeds and a causal null; this has two and
neither. Not a retraction — a flag for the next run, which should carry
`cover+store` so the combination is measured rather than assumed
additive.

**What this says about the method, and it is the more durable part.**
The same filter has now been reported as a null and as a 12% win from
the same code. Nothing about the mechanism changed; what changed is
whether the measurement could observe it. The saturated regime made
every arm cost exactly the budget. Before trusting any search-cost
result, check what fraction of searches terminate early — a result
gathered where nothing terminates is a measurement of the budget
wearing the costume of a measurement of the search.

## F169 — where the recipe architecture stands, with the modulus in

The end-to-end claim, restated so it can be checked: the plant is
trained ONLY on random programs over random states, so no task ever
touches its weights, and every family is handled by SEARCHING for a
program that explains it against those frozen weights. Three cleanly
paired seeds, inferred modulus.

| family | held-out fit | identity floor | margin |
| --- | ---: | ---: | ---: |
| line | 0.9518 | 0.1315 | +0.8203 |
| dial | 0.9900 | 0.6667 | +0.3233 |
| toggle | 0.9434 | 0.6958 | +0.2476 |
| perm | 1.0000 | 0.5000 | +0.5000 |
| grid | 0.9538 | 0.5710 | +0.3828 |
| proc0 | 1.0000 | 0.6960 | +0.3040 |
| proc1 | 0.9802 | 0.6749 | +0.3053 |

Mean held-out **0.9742** against a mean identity floor of **0.5623**,
above the floor in **21 of 21** family-seeds. The interpreter executes
programs it has never seen at 0.9916 per slot and 0.9512 exact.

Against F155, which established the architecture: recipes reached
0.9247 against a 0.5229 floor, above floor in 14 of 14. The distance
to a perfect recipe has roughly halved, 0.0753 to 0.0258, and the
family that could not be solved at all inside a 24,000-candidate
budget now sits at 0.9434.

**The floor is the part worth keeping honest.** Identity — copy the
input unchanged — scores 0.5623 on average and 0.6960 on one family.
Chance is 0.125. Any report of these numbers against chance would be
inflating them by roughly four times, which is why the floor is in the
table rather than in a footnote.

**What this does not show.** These are recipes found by SEARCH against
a frozen interpreter, not recipes inferred by a reader. The reader
remains the open half of the architecture, and F164's correction
stands: sufficient optimisation closes the amortization gap on a
favourable trajectory, while reliability and basin selection are
unresolved. A system that must search thousands of candidates per
action is also not yet a system that has LEARNED anything from having
solved a family before — F161 measured that transfer honestly at about
a tenth of the search cost.

## F170 — the equality-guard hole does NOT exist, so the guard should
## not be built

The next expressiveness extension under discussion was an equality
guard, `CINC_EQ i,j,v`, on the reasoning that our conditionals gate
only on "slot j is non-zero" and so cannot express an effect that fires
when a slot holds a PARTICULAR value. That is true of the basis. The
question is whether it costs anything, and the rule this project has
earned is to find the failure signature BEFORE extending — which is
exactly why the modulus was worth adding, since `toggle` failed loudly
first.

Three seeds, inferred modulus, with `walled` and two procedurally gated
families added as synthesis targets:

| family | held-out | identity floor | margin | per seed |
| --- | ---: | ---: | ---: | --- |
| walled | 0.9160 | 0.6107 | +0.3053 | 0.914, 0.926, 0.908 |
| gate0 | 0.9796 | 0.6799 | +0.2997 | 0.949, 0.990, 1.000 |
| gate1 | 0.9863 | 0.4501 | +0.5363 | 0.973, 1.000, 0.986 |
| the seven plain families | 0.9781 | 0.5525 | +0.4256 | — |

**Gated 0.9607 against plain 0.9781.** Two of the three gated families
fit as well as anything else in the set — `gate1` at 0.9863 is above
five of the seven plain families. There is no gated failure to fix.

`walled` is the weakest of all ten families, and reproducibly so:
0.914, 0.926, 0.908 is far too tight to be noise. But it is weakest by
about four points against a family it clears its own identity floor by
thirty, and one family four points down does not justify a new
instruction class whose cost would be a wider search on all ten. F92's
"decisive failure" on walled was measured in a different probe with a
learned policy; it does not reproduce as an expressibility failure
here.

**Recorded as a negative that prevented a build.** The guard was the
third priority on an outside recommendation and it looked well
motivated. Measuring first cost three runs; building it first would
have cost an instruction-set change, a retrain, and the same
extrapolation debate the modulus is still having — for a hole that is
not there.

The rule generalises and is worth stating plainly: **an extension needs
a failure signature, not an argument.** The modulus had one — toggle at
exactly 50% on the slots it touched, arithmetic, no network. The guard
has an argument and no signature.

## F171 — the filter replicates at five seeds, and it COMPOSES with
## program reuse

Five seeds, inferred modulus, searches terminating early throughout
(0 or 1 of 11 families saturate, so this is a measurement of the search
and not of the budget — the check F168 made standing). Ratios are
interpreter calls against the frozen control.

| arm | diverse | sd | related | sd |
| --- | ---: | ---: | ---: | ---: |
| frozen | 1.000 | — | 1.000 | — |
| coverage filter | 0.879 | 0.046 | 0.812 | 0.074 |
| stores solved programs | 0.929 | 0.124 | 0.772 | 0.044 |
| **both** | **0.848** | **0.033** | **0.711** | **0.062** |

**The filter replicates.** 0.879 and 0.812, below 1.0 in all ten
sequence-seed cells, at a quarter the spread of the reuse arm. F168's
two-seed reading of 0.887/0.867 holds up.

**They compose, and neither is redundant.** Together they reach 0.848
and 0.711, better than either alone on both sequences. The gain is
SUB-multiplicative — 0.879 x 0.929 would predict 0.817 and 0.812 x
0.772 would predict 0.627 — so the two mechanisms overlap partially,
which is what one would expect of a filter that removes hopeless
candidates and a store that supplies good ones. They are not the same
saving twice, and they are not independent either.

The combination is also the most RELIABLE arm, sd 0.033 against 0.124
for reuse alone on the diverse sequence. Adding a sound filter to an
unreliable mechanism steadies it.

**F168's flag resolves as noise, in the direction I guessed but for a
reason I should record.** F168 read reuse on the diverse sequence at
1.075 against F161's 0.959 and I flagged it as possibly an interaction
with the inferred modulus. At five seeds it reads 0.929, consistent
with F161. The per-seed values are 1.089, 1.061, 0.814, 0.889, 0.790 —
F168's two seeds were the only two above 1.0 in the set. A two-seed
sample of a mechanism whose sd is 0.124 was never going to settle
anything, and the flag was right to be a flag.

**What this adds up to for the search bottleneck.** F161 measured reuse
honestly at about a tenth. Two cheap, sound additions now take the
combination to 15% on unrelated families and 29% on related ones, with
the filter costing no interpreter calls at all and the modulus already
paid for. That is real and it is still not the order of magnitude the
search needs: thousands of candidates per action becomes hundreds of
candidates fewer, not hundreds of candidates total.

## F172 — a positive control for search-cost measurements, and a
## retro-audit of every result that needed one

F168 established that a search-cost measurement taken where searches
SATURATE their budget measures the budget, not the search. That was
stated as a check to run. It is now an instrument, and the past results
have been audited against it.

**The instrument: `cover` as a positive control.** The coverage filter
is established at 0.879 and 0.812 over five seeds with low spread
(F171). So it is a mechanism of known size. Run it alongside anything
new, and if it does not reproduce, the regime cannot detect anything
and no other arm's number should be read.

This is not hypothetical. A 4000-update smoke of the enumeration arm
read every arm at ~1.000 — including `cover` at 0.994 and 1.000, with
4 of 7 and 2 of 4 families saturating. A mechanism known to be worth
12% reads as nothing there. Without the control I would have had four
arms all at 1.000 and no way to tell "the new mechanism does nothing"
from "this measurement can see nothing".

**The retro-audit.** Every search-cost result in the ledger, by
fraction of families whose search ran to the full budget:

| result | saturated |
| --- | ---: |
| F159, library arms at 3 seeds | 4/60 = 6.7% |
| F161, reuse with the causal null at 5 seeds | 9/100 = 9.0% |
| F168, the filter at 2 seeds | 1/40 = 2.5% |
| F171, filter and reuse composed at 5 seeds | 3/100 = 3.0% |

All four are in a regime that can detect an effect, against the 36%
and 50% that voided the smoke tests. **No past cost claim needs
withdrawing on these grounds.** Worth having checked rather than
assumed — F157 explicitly noted `toggle` saturating three of four runs,
which is what made the whole question live.

**The general form, since this is the third instrument of its kind.**
Byte-identical output catches a parameter that is not reaching the
code, or two things that are secretly one thing. A positive control of
known size catches a regime that cannot measure. Both work because they
fail loudly and cannot be argued with. An experiment needs at least one
quantity whose expected value is known in advance, or a null result
from it is uninterpretable.

## F173 — enumeration cuts search cost by 2.4x, and it changes what is
## being counted rather than shaving a factor

Four seeds, inferred modulus, regime 2.5% saturated. **The positive
control reproduces**: `cover` reads 0.893 and 0.840 here against its
established 0.879 and 0.812, so the measurement can detect an effect
and the other arms can be read.

| arm | diverse | sd | related | sd |
| --- | ---: | ---: | ---: | ---: |
| frozen | 1.000 | — | 1.000 | — |
| coverage filter | 0.893 | 0.039 | 0.840 | 0.056 |
| **enumeration** | **0.425** | 0.183 | **0.406** | 0.034 |
| enumeration + stored programs | 0.421 | 0.173 | 0.376 | 0.063 |

Every mechanism before this one — reuse, the filter, the modulus —
changes which candidates get drawn or discarded, so each shaves a
constant factor off an exponential. Enumerating by increasing depth
over instructions that write a changed slot changes what is counted: a
family whose recipe is one instruction pays the size of the instruction
set.

**The prediction was recorded before the run and it holds, including
the failure half.** Big win where the recipe is short, wasteful where
it is not:

| family | enum/frozen | frozen calls | enum calls |
| --- | ---: | ---: | ---: |
| dial | 0.010 | 2538 | 21 |
| rel3 | 0.013 | 5042 | 64 |
| toggle | 0.015 | 22151 | 328 |
| perm | 0.042 | 4032 | 156 |
| ... | | | |
| grid | 1.114 | 5856 | 6505 |
| proc0 | 1.829 | 2389 | 3169 |
| line | 1.876 | 3660 | 4292 |
| extra3 | 2.209 | 1405 | 1371 |

**14 families cheaper, 6 more expensive, none in between.** The wins
run to 100x and the losses are bounded at 2.2x, because a failed
enumeration costs at most the enumerated set before falling back to
sampling. That asymmetry is why the aggregate is strongly positive
despite a third of families being worse.

**`toggle` is the headline.** F157 could not solve it inside a 24,000
candidate budget and I recorded it as probably inexpressible. F160
found the real hole was the modulus. It is now the third CHEAPEST
family in the set: 22,151 calls down to 328, a 67-fold reduction, and
its recipe is exactly `INC i mod 2 ; INC j mod 2`, which a depth-2
enumeration finds immediately once the modulus comes from the slot.
Three findings had to compose to get there and none of them would have
worked alone.

**An honest deflation of F161.** Stored programs add almost nothing on
top of enumeration — 0.421 against 0.425, and 0.376 against 0.406.
Reuse was worth about a tenth against random sampling and is worth
about nothing against a systematic proposer, because both are answering
"find a short program that explains this" and enumeration answers it
better. F161's measurement stands; its IMPORTANCE does not survive a
better baseline. That is the ordinary fate of a mechanism measured
against a weak alternative, and it is worth recording as such rather
than leaving two live results that quietly contradict each other about
where the saving comes from.

**The obvious next move.** The loss is entirely "paid for a failed
enumeration, then sampled anyway". Interleaving enumerated and sampled
candidates bounds the worst case at 2x while keeping most of the win.

## F174 — the enumeration cap was sized from the wrong half of the data

F173 ended with an obvious improvement: the whole loss was "paid for a
failed enumeration, then sampled anyway", so cap the enumeration. I set
the cap at a quarter of the per-action budget, reasoning that every WIN
on record arrived within 328 calls while the enumeration can run to
3600.

Four seeds later the capped and uncapped arms were **identical to the
digit**, arm by arm and family by family. The byte-identical tell
again, and this time it was not a flag failing to reach the code — the
cap is there and correct. It simply never binds.

**Why, and it is the exact inverse of the intended effect.** The
enumeration size per action is `(NOPS-1) x |changed slots| x (SLOTS-1)`,
plus its square at depth 2. So:

| |changed| | enumeration per action | against a cap of 1000 |
| ---: | ---: | --- |
| 1 | 930 | never binds |
| 2 | 3660 | binds |

Every family that LOSES — line, grid, proc0, proc1's cheap actions —
changes exactly ONE slot per action, so its enumeration is 930 and
slips under the cap. Every family where the cap binds — toggle, perm —
is one that WINS. **The cap was cutting the winners and sparing the
losers.**

**The mistake is worth naming precisely.** I chose the parameter from
the distribution of WIN costs (all ≤328) and never looked at the
distribution of LOSS costs (930 per action). Those two are unrelated
quantities: the first is where a successful enumeration terminates, the
second is how large the enumeration is when it fails. Sizing a
threshold from one and applying it to the other has no reason to work,
and the arithmetic was available before the run — 6 x 1 x 5 = 30, and
30 + 900 = 930, against a cap of 1000.

A cap of 0.1 (400 calls) binds on the 930 while still clearing every
win on record at 328. Predicted: the six losing families lose roughly
half their wasted enumeration, and the aggregate improves from 0.425.

**What saved this was the instrument, not the reasoning.** Byte-
identical output has now caught four things. A result that had merely
been *disappointing* rather than *impossible* would have been written
up as "capping does not help", which is true and useless, instead of
"the cap was set above the quantity it was meant to bound".

**Pre-registered prediction for the 0.1 cap, written before reading the
run.** F174's lesson was that the arithmetic was available in advance
and I did not do it, so it is done here first.

The cap binds only where the enumeration FAILS and burns its full pool.
At 400 instead of 930 per action, three diverse families should recover
roughly 530 calls per action and nothing else should move:

1. `line`, `grid` and `proc0` — the three that cost more than the
   frozen control at 1.17, 1.11 and 1.33 — should all fall BELOW 1.0,
   to approximately 0.88, 0.75 and 0.38.
2. Every winning family should be UNCHANGED to within noise, because
   the most expensive successful enumeration on record terminated at
   328 calls and the cap is 400. `dial` at 0.008, `toggle` at 0.015 and
   `perm` at 0.039 must not move.
3. The aggregate must improve.

If (2) fails — if the winners degrade — the cap is again cutting the
wrong thing and 400 is still above nothing useful. If (1) fails, the
losing families are not spending what the pool-size arithmetic says
they are, and the model of where the cost goes is wrong.

Note on statistics: these per-family figures are ratio-of-means, while
the headline 0.425 is a mean of per-seed ratios. They are different
estimators and are not comparable to each other; the predictions above
are stated in the ratio-of-means the table uses.

## F175 — the enumeration cap is REFUTED, and a failed enumeration is
## not wasted

The prediction was pre-registered with what each failure would mean.
Two of three failed. Regime valid throughout (2.5% saturated, control
`cover` reproducing at 0.893/0.840).

**Prediction 2 — winners unchanged — CONFIRMED exactly.** `dial`
0.008 -> 0.008, `toggle` 0.015 -> 0.015, `perm` 0.039 -> 0.039, to the
digit. The cap at 400 clears their termination at 328 and does not
touch them, exactly as the arithmetic said.

**Prediction 1 — the three losing families fall below 1.0 — FAILED,
one of three.**

| family | uncapped | capped | predicted |
| --- | ---: | ---: | ---: |
| line | 1.173 | 1.099 | 0.883 |
| grid | 1.111 | **1.154** | 0.749 |
| proc0 | 1.326 | 0.756 | 0.383 |

`grid` got WORSE under a cap meant to help it.

**Prediction 3 — the aggregate improves — SPLIT, and it is a net
loss.** Diverse 0.425 -> 0.374, related 0.406 -> **0.503**. Averaged
across both sequences the cap costs 0.4155 -> 0.4385.

**Why, and it is the parenthesis I wrote and dismissed.** When I set
the cap I noted that a smaller enumeration leaves a lower best score
for the sampling to start from, and called it minor. It is the whole
effect. **A failed enumeration is not wasted work** — it raises the
best score, and the sampling that follows stops at `--fit-target`
relative to that. Cutting the enumeration saves calls in one place and
spends them in the other. On `grid` it spends more than it saves.

On the related sequence the loss is larger still, which points at a
second mechanism worth noting rather than guessing about: those
families plausibly find their recipes by enumeration at a position
BETWEEN 400 and 930, so the cap converts wins into fallbacks. That is a
prediction, not a measurement, and the way to settle it is to record
where a successful enumeration terminates rather than reason about it.

**Default restored to uncapped.** F173's 0.425/0.406 stands as the
best configuration measured.

**Three obvious improvements have now failed in this session** — the
coverage filter as first framed, the attainable-fit bound, and this
cap. All three were sound arguments about where cost goes, and all
three were wrong about a system whose parts interact. The pattern is
specific enough to name: **each assumed a component's cost could be
removed without changing what the rest of the system then does.**

## F176 — my explanation for F175 is refuted, and the search cost is
## not distributed the way every previous finding assumed

F175 guessed that capping hurt the related sequence because those
recipes are found BETWEEN 400 and 930 calls, so the cap converted wins
into fallbacks. Recording the distribution rather than reasoning about
it, over four seeds with the `cover` control reproducing at
0.893/0.840:

| sequence | successes | below 400 | 400-930 | above 930 |
| --- | ---: | ---: | ---: | ---: |
| diverse | 152 | 146 (96%) | 6 (4%) | 0 |
| related | 136 | 128 (94%) | 8 (6%) | 0 |

**Refuted.** At most 6% of successes fall in the band the cap would
have cut — far too few to explain a 0.406 -> 0.503 degradation. That
leaves the mechanism F175 already established as the sole explanation:
a failed enumeration raises the best score the fallback sampling starts
from, so cutting it lengthens what follows.

**The number that matters is not the one I went looking for.**

| | actions | mean cost | share of total cost |
| --- | ---: | ---: | ---: |
| enumeration SUCCEEDS | 288 (84.5%) | 57.6 | 9.2% |
| enumeration FAILS | 53 (15.5%) | 3090.6 | **90.8%** |

**Ninety-one percent of all search cost sits in fifteen percent of
actions, and a failing action costs 54 times a succeeding one.** The
median successful enumeration terminates at the SIXTH candidate. Half
of every action in the benchmark is solved by the sixth thing tried.

**This reframes every search result in the ledger.** F161, F171 and
F173 all measured mean cost, and a mean over a distribution this skewed
is a measurement of the tail wearing the name of the average. Reuse at
0.929, the filter at 0.879, enumeration at 0.425 — all of these are
almost entirely statements about the 15% of actions where enumeration
fails, because the other 85% contribute a twelfth of the total. None of
those numbers is wrong; what is wrong is reading them as "search got
cheaper" when what they say is "the tail got cheaper".

**And it says what to work on.** There is no general search problem
here to solve. There is a specific, small, identifiable set of actions
whose recipes are longer than depth 2, and everything else is already
essentially free. Making the common case faster cannot buy more than
9% however clever it is. The whole remaining budget lives in a set that
can be ENUMERATED and inspected rather than reasoned about — which is
the next thing to do, and the instrument for it already exists, since
an action that produces no `enum_hits` entry is exactly a member.

## F177 — the expensive tail is SATURATING arithmetic, and the ISA has
## none. The names collided; the semantics did not.

F176 located 91% of search cost in 15% of actions and said to inspect
those cases rather than theorise. Inspected.

**The two worst families give it away.** `line_step` is
`min(7, max(0, state[0] + delta))` — it CLIPS at the boundary.
`grid_step` refuses a move that would leave the grid — also clipping.
`dial_step` is `(out[which] + delta) % 8` — it WRAPS. Failure rates:
line 75%, grid 44%, dial 0%.

**The correlation, per seed, each seed matched to its OWN procedural
specs** (the first version of this analysis matched one seed's specs
against four-seed pooled rates, which is the same mismatch that has
burned this session twice):

| population | n | mean enumeration failure rate |
| --- | ---: | ---: |
| families with NO clipped op | 15 | **0.0%** |
| families with half or more clipped ops | 18 | **44.5%** |

r = 0.729 over 44 family-seeds, and the separation at the extremes is
total: **not one family without a clipped operation ever failed.**

**The cause, and it is embarrassing in an instructive way.** The ISA
was built by promoting `schema_families`' procedural vocabulary to
executable instructions. But it took the NAMES and not the SEMANTICS:

| name | in `schema_families` | in our ISA |
| --- | --- | --- |
| `inc` / INC | `(x + 1) % values` | `(x + 1) % m` — agrees |
| `cinc` / CINC | `min(values - 1, x + 1)` — SATURATING | `(x+1) % m` gated on slot j — CONDITIONAL |
| `cdec` / CDEC | `max(0, x - 1)` — SATURATING | `(x-1) % m` gated on slot j — CONDITIONAL |

The `c` was read as "conditional" and it meant "clipped". So the ISA
acquired a gating mechanism nothing in the task distribution needs —
which is exactly why F170 found no gated hole, gated families fitting
at 0.9607 — and never acquired the saturating arithmetic that half the
task distribution is built from.

**This is the same class of error as F160** and it is now the second
time: the ISA's arithmetic did not match the families' arithmetic, the
mismatch was invisible in aggregate accuracy, and it surfaced only when
a cost distribution was broken out by family. F160 was the modulus,
this is saturation. Both were mis-taken from the same source file.

**The fix is small, domain-general, and has the signature F170
requires**: add `SINC i` = `min(m-1, x+1)` and `SDEC i` = `max(0, x-1)`,
with m the slot's own inferred range. Two operations, NOPS 7 -> 9, and
they are slot operations exactly like the ones already there. Building
it now — unlike the equality guard, this one has 44.5% against 0.0% to
justify it.

**Expressibility settled first, by arithmetic, before any run.** With
SINC/SDEC every action of the two worst families becomes a SINGLE
instruction, exact on every state:

| family | previous failure rate | with saturating ops |
| --- | ---: | --- |
| line | 75% | `SDEC 0` and `SINC 0`, both exact |
| grid | 44% | `SDEC 0`, `SINC 1`, `SINC 0`, `SDEC 1`, all exact |
| walled | — | same four, 0.945-0.953 — the wall is the residue |

`walled` is the interesting one: a single saturating instruction gets
it to 0.945-0.953 against a `--fit-target` of 0.95, so it lands exactly
on the threshold. That is the wall itself showing up as the 5% the
basis cannot express, which is a much more precise statement of F92's
old complaint than "walled is hard".

**Pre-registered predictions for the saturating run.**

1. `line` and `grid` failure rates go to approximately 0, and both
   become depth-1 hits. Together they hold 42,006 of the ~164,000
   fallback calls, so the tail should lose about a quarter on those two
   families alone.
2. Families with NO clipped op — dial, toggle, perm — must be
   UNCHANGED. They already fail 0% of the time and cannot improve; if
   they get worse, the cost below has swamped the benefit.
3. The cost is real and one-sided: NOPS goes 7 -> 9, so the
   enumeration pool grows by a third at depth 1 and by 78% at depth 2.
   Every action that still fails pays that with nothing in return.
4. Net: the aggregate should improve, but by less than the tail
   arithmetic suggests, because (3) is charged against every one of the
   53 failing actions while (1) removes only some of them.

If (2) fails the extension is not free and the trade needs stating. If
(1) fails the expressibility arithmetic above is wrong, which would be
surprising since it is exact and needs no network.

## F178 — saturating arithmetic closes the tail completely: search cost
## falls 13x, and the search stops failing

Four seeds, control `cover` reproducing at 0.842/0.727, regime valid.

| measure | before | after |
| --- | ---: | ---: |
| enumeration success rate | 288/341 = 84.5% | **341/341 = 100.0%** |
| mean candidates per action | 528 | **22.9** |
| enum cost, diverse | 0.425 | **0.031** |
| enum cost, related | 0.406 | **0.016** |
| mean search fit, diverse | 0.9877 | 0.9973 |
| worst search fit, diverse | 0.9239 | 0.9816 |
| interpreter, unseen programs | 0.9887 | 0.9887 |
| interpreter, double length | 0.9373 | 0.9444 |

**Every action in the benchmark is now solved by enumeration.** The
53-action tail that held 91% of all search cost (F176) is gone — not
reduced, eliminated. Per-family failure rates went 75%, 44%, 35%, 21%,
19%, 19%, 14%, 7% to zero across the board.

**And it is cheaper AND better.** Mean search fit rose 0.9877 to
0.9973 and the worst family rose 0.9239 to 0.9816, so this is not a
search that got faster by settling for less. The interpreter is
unchanged on unseen programs at 0.9887 despite carrying nine
instructions instead of seven, and double-length extrapolation is
slightly BETTER, 0.9373 to 0.9444.

**Prediction audit, since all four were on the record.**

1. `line` and `grid` go to ~0% — CONFIRMED, both exactly 0%.
2. Families with no clipped op unchanged — CONFIRMED, dial, toggle and
   perm all 0% before and after.
3. The wider pool is charged to every action that still fails —
   VACUOUS, because no action still fails. The cost was real and had
   nothing to be charged against.
4. "The aggregate improves, but by LESS than the tail arithmetic
   suggests" — **REFUTED, and in the optimistic direction.** It
   improved by far more. I sized the prediction from `line` and `grid`
   holding 42,006 of ~164,000 fallback calls and forgot that the
   procedural families' failures were ALSO clipped-op failures — the
   diagnosis applied to eight families, not two, and I counted two.

**What the whole arc actually was.** F155 built the recipe
architecture and called search its bottleneck. Everything from F157 to
F175 tried to make the search cleverer: reuse, compression, learned
proposal distributions, sketches, filters, bounds, caps. The best of
those got to 0.848. None of it was the problem. The problem was that
the instruction set could not express half the task distribution, and
the search was spending thousands of candidates hunting for programs
that did not exist. Two operations — twelve lines — did 13x what all
of that did.

The lesson is not "extend the basis instead of improving search". It is
that **a search failing expensively looks identical to a search that is
too slow**, and only a per-family cost distribution distinguishes them.
F176 was the finding; F177 and F178 are its consequences.

**Pre-registered for the end-to-end re-derivation.** F169's headline —
mean held-out 0.9742 against a 0.5623 identity floor, 21/21 above
floor — was measured on the seven-instruction basis and is stale.
Re-running it paired against the old basis at four seeds, with `walled`
and the gated families included.

1. `line` (0.9518) and `grid` (0.9538) should rise most, since both are
   now exactly expressible as single instructions.
2. `dial`, `toggle`, `perm` should be unchanged — all three were
   already at or near ceiling and use no saturating arithmetic.
3. `walled` should rise but STOP SHORT of the others, because a single
   saturating instruction reaches only 0.945-0.953 on it and the
   remainder is the wall, which no operation in this basis expresses.
   If walled reaches the others, the wall is expressible after all and
   the F92 complaint has a different cause than I have assumed twice.
4. The mean should rise; the identity floor must NOT move, since it
   depends only on the families.

(4) is the control that matters: the identity floor is computed from
the task and not from the basis, so if it shifts, something is wrong
with the comparison rather than with the basis.

## F179 — expressibility bought COST, not accuracy; and cheap search
## converges to its own stopping threshold

The end-to-end re-derivation, four seeds, paired old basis against
saturating basis, `walled` and gated families included.

| family | old held-out | saturating | delta | prediction |
| --- | ---: | ---: | ---: | --- |
| line | 0.9717 | **1.0000** | +0.0283 | P1 |
| grid | 0.9668 | 0.9922 | +0.0254 | P1 |
| walled | 0.9141 | 0.9395 | +0.0254 | P3 |
| gate0 | 0.9847 | 1.0000 | +0.0153 | |
| dial | 0.9954 | 0.9925 | -0.0029 | P2 |
| perm | 1.0000 | 1.0000 | 0.0000 | P2 |
| **toggle** | 0.9551 | **0.9128** | **-0.0423** | P2 |
| MEAN | 0.9732 | 0.9790 | +0.0058 | |

**P4, the control, is exact.** The identity floor moved by 0.000000 —
it depends on the families and not on the basis, so the comparison is
sound. 40/40 above floor in both arms.

**P1 confirmed.** `line` reaches exactly 1.0000 and `grid` 0.9922, the
two families whose every action became a single exact instruction.

**P3 confirmed, and it pins F92.** `walled` rises by the same +0.0254
as `grid` and stops at 0.9395 while the rest sit near 0.98. It gets
the movement and not the wall. The wall is the residue, and it is the
only thing in this benchmark the basis genuinely cannot express.

**P2 FAILED on `toggle`, consistently: -0.0423 across all four seeds.**
I recorded in advance that a P2 failure would mean the wider
instruction set had swamped the benefit. That reading is wrong, and the
data says why:

| | seed 1 | seed 2 | seed 3 | seed 4 |
| --- | ---: | ---: | ---: | ---: |
| toggle search fit, old basis | 0.9802 | 0.9732 | 0.9929 | 0.9833 |
| toggle search fit, saturating | 0.9546 | 0.9572 | 0.9564 | 0.9516 |

Every saturating value sits in 0.9516-0.9572, just above the
`--fit-target` of 0.95, which looks exactly like the search stopping
the moment it clears its threshold.

**CORRECTION — that diagnosis is wrong, and the check that caught it
was a sweep that came back byte-identical.** Sweeping `--fit-target`
over 0.95, 0.99 and 1.0 produced IDENTICAL numbers on every family and
every seed. The reason: `synthesise` has no early stop at all. It runs
the full `--synthesize` budget per action unconditionally, and
`--fit-target` enters that path only through the optional
`--fit-bound` filter, which is off. There is no threshold for the
search to settle at, so it cannot have settled at one. The clustering
near 0.95 is a coincidence I read a mechanism into.

**My ORIGINAL pre-registered reading was right and the correction was
wrong.** P2's stated failure meaning was "the wider instruction set has
swamped the benefit", and that is what happened: `synthesise` proposes
RANDOM programs, so going from 7 operations to 9 dilutes every draw by
29%, and toggle's recipe needs two specific instructions. Families that
gained expressibility (line, grid) more than paid for the dilution;
toggle, which was already expressible, paid it for nothing.

**And this exposes a real inconsistency in the ledger.** The COST
results (F173, F178) come from `search_with_library`, which enumerates.
The QUALITY results (F169, F179) come from `synthesise`, which samples
at random and always spends its whole budget. Those are two different
search algorithms, and I have been reporting them as one system. The
end-to-end quality figure does not describe the configuration the cost
figure describes. Fixing that — wiring enumeration into `synthesise` —
is the next change, and until it lands the honest statement is that
0.9790 is the quality of RANDOM search on the saturating basis, not of
the system F178 measured.

**So the fit-target is now the binding constraint on quality, and it
was not before.** That is a consequence of success and it is invisible
in the aggregate: mean held-out still rose. It also means the headline
comparison in F178 — "cheaper AND better" — is only true on average.
Families that used to overshoot now settle.

**The distinction worth keeping.** Adding saturating arithmetic bought
a 13x reduction in search cost (F178) and +0.0058 in accuracy. Those
are not the same size and it would be easy to report the first and
imply the second. The basis was never badly wrong about WHAT these
families do — the old one approximated them to 0.973 — it was wrong
about how expensive it is to find out.

Fix is one line and one measurement: raise `--fit-target`, or spend the
recovered budget continuing to improve after the threshold rather than
stopping. Cheap search makes a stricter target affordable for the first
time, which is the more interesting version.

**Pre-registered for the fit-target sweep.** F179 showed the search now
stops the moment it clears 0.95, so quality is bound by the threshold
rather than by the basis or the budget. Sweeping `--fit-target` over
0.95, 0.99 and 1.0 on the saturating basis, four seeds.

1. `toggle` recovers. Its saturating search fits cluster at
   0.9516-0.9572, so a target of 0.99 should push it back toward the
   0.98 the old basis reached by accident, and its held-out should
   recover most of the -0.0423.
2. Families already at 1.0000 — `perm`, and `line` on the saturating
   basis — must not move. There is nothing above 1.0 to find.
3. Cost rises, but from 22.9 candidates per action. Even a large
   multiple stays far under the old basis's 528, which is the whole
   point: a stricter target is affordable now and was not before.
4. At `--fit-target 1.0` some families will fail to reach it and burn
   the full budget, so cost should jump sharply there while quality
   gains little over 0.99. If 1.0 is BOTH cheap and better, the recipes
   are exact more often than F178's 100% enumeration success implies,
   and the fit-target was never doing anything useful.

## F180 — where the founding objective actually stands, and an honest
## re-reading of what "transfer" now means here

The objective, in its standing wording: *produce a program such that
given task A makes novel task B faster to learn than chance or starting
from scratch.*

**What the current system does.** A never-before-seen task family is
handled by enumerating about 23 candidate programs against a frozen
interpreter, reaching 0.979 held-out against a 0.549 identity floor,
with the plant's weights bit-identical before and after. No gradient
step, no fitting, no per-family parameters outside the recipe.

**The uncomfortable reading, which should be said first.** That is NOT
"having solved family A made family B cheaper". F161 measured that
mechanism directly, with a causal null and verbatim-recall
instrumentation, and got about a tenth. F173 then showed even that
tenth mostly evaporates against a systematic proposer — stored programs
add 0.421 against 0.425. **Sequential accumulation across real families
buys almost nothing here.** A bank that grows as families are solved is
not, on this evidence, what makes the next family cheap.

**The reading that is actually supported, and it is stronger.** What
makes family B cheap is that the plant was trained on RANDOM PROGRAMS
over random states — a task A that contains no family at all — and that
training transfers to every family expressible in the basis. The
interpreter executes programs it has never seen at 0.9887, and that one
competence covers line, dial, toggle, perm, grid, walled, gated and
procedurally generated families alike, none of which touched its
weights.

So the objective is met, with A = "execute arbitrary programs over an
amodal slot basis" and B = any family in that basis. That is a harder
version of the claim than the one originally intended, because A is
synthetic and universal rather than a previously solved real task, and
because there is no risk of B's answer having leaked into A.

**What this costs the bank story.** The bank still holds each family's
recipe and still gives exact retention — a stored program is a program,
it does not drift. But the bank is not currently doing COMPOUNDING
work: solving family 20 is no cheaper than solving family 2. The
project's premise that a growing memory buys growing capability is, on
the recipe track, currently supported only in the weak sense that the
bank stores what was found, not the strong sense that it accelerates
what comes next.

**And the honest limit.** Everything above is inside one basis. F179
pinned the boundary precisely: `walled` reaches 0.9395 while everything
else sits near 0.98, and the residue is the wall — a state-dependent
refusal no operation here expresses. Two basis holes have already been
found by accident (the modulus, saturation) and each was worth more
than every search improvement combined. There is no reason to think the
third does not exist, and the instrument that would find it is a
per-family cost distribution, which is now permanently wired.

**F180 addendum — checking the "no compounding" claim, and rejecting
the obvious test for it.** F180 asserts that solving family 20 is no
cheaper than solving family 2. The obvious check is whether cost falls
with POSITION in the sequence:

| basis | sequence | r(position, cost per action) | first half | second half |
| --- | --- | ---: | ---: | ---: |
| old | diverse | -0.160 | 776.5 | 544.6 |
| old | related | +0.211 | 272.7 | 852.4 |
| saturating | diverse | -0.288 | 34.9 | 20.9 |
| saturating | related | +0.127 | 18.8 | 18.4 |

**That test is confounded and should not be used.** The families appear
in a FIXED order — line, dial, toggle, perm, grid, proc0, proc1, then
the extras — so position is entangled with family identity. The diverse
sequence puts `toggle` (the most expensive family on the old basis)
fifth and the cheap procedural extras last, which manufactures a
negative correlation out of ordering alone. The signs disagree between
sequences, which is what a confounded measurement looks like.

**The sound test is the arm comparison and it was already run.**
`enum` against `enum+store` is identical in every respect except
whether solved programs are stored and proposed: same families, same
order, same observations, same plant. It reads 0.425 against 0.421 on
the diverse sequence and 0.406 against 0.376 on the related one. That
is the paired, unconfounded measurement, and it is what F180's claim
rests on — not the position analysis.

Recorded because the position table looks like evidence and is not.
The fix, if the question is ever worth more compute, is to shuffle the
family order per seed so position and identity are independent.

**Pre-registered for the unified end-to-end run.** `--enumerate-synthesis`
puts the quality measurement on the same algorithm as the cost
measurement. Four seeds, saturating basis, inferred modulus, gated
targets, against the random-sampling numbers already in hand.

1. `toggle` RECOVERS. Its regression to 0.9128 was dilution of random
   proposal by two extra operations (the corrected F179 reading).
   Enumeration does not dilute — it finds `INC i mod 2 ; INC j mod 2`
   at depth 2 regardless of how many operations exist — so toggle
   should return to at least the 0.9551 the old basis reached.
2. Families already at 1.0000 stay there.
3. The mean rises above 0.9790, because the families that lost to
   dilution get it back while the ones that gained expressibility keep
   their gain.
4. Cost per action collapses toward F178's 22.9.

If (1) fails, the corrected F179 reading is also wrong and toggle's
regression is neither early stopping nor dilution, which would leave no
candidate explanation standing and make it the thing to chase.

## Pre-registered: the continual-learning measurement (probe 268)

`continual.py` puts both architectures on ONE family sequence with one
evaluation. Nine families, four seeds, and a replay control.

  WEIGHTS  a slot model trained by gradient descent on each family as
           it arrives, carrying only its own weights forward.
  BANK     a frozen instruction interpreter trained ONLY on random
           programs, plus one searched recipe per family held outside
           the weights.

**Predictions, in the order they matter.**

1. **Bank forgetting is EXACTLY 0.0000 on every seed.** Not "low", not
   "better" — bit-identical, because re-scoring family 1 after family 9
   runs the same frozen weights over the same stored integers. This is
   a control, not a result: a claim that cannot degrade gracefully
   fails loudly if the harness is wrong, and the smoke run already
   returned 0.0 against the weights arm's 0.4226.
2. **Weights forgetting is large without replay.** The smoke run put it
   at 0.4226, with `dial` falling to 0.4616 against an identity floor
   of 0.6667 — forgetting past the point of copying the input.
3. **Replay substantially closes it.** This is the control that keeps
   the comparison honest. Replay is how the field actually prevents
   forgetting, and a bank that only beats a no-replay baseline has
   beaten a straw man. If replay closes the gap entirely, the bank's
   advantage is "needs no buffer of old data", which is a real but much
   narrower claim than "does not forget".
4. **The bank is WORSE at learning time.** Gradient descent fits each
   family to 1.0000 as it arrives; the bank is capped by interpreter
   quality near 0.99 and by whether the family is expressible at all.
   Zero forgetting of a slightly worse model is the actual trade, and
   stating it in advance stops the headline from hiding it.

The interesting quantity is the size of (2) minus (3), because that is
what the bank is worth over the standard remedy.

## F180 — the unified end-to-end: 15x cheaper, `toggle` exact, and the
## mechanism I invented and retracted is now genuinely present

Four seeds, saturating basis, inferred modulus, quality and cost
finally measured on the SAME algorithm.

| family | random search | enumeration | delta | cost rnd | cost enum |
| --- | ---: | ---: | ---: | ---: | ---: |
| toggle | 0.9128 | **1.0000** | +0.0872 | 24000 | 431 |
| proc1 | 0.9834 | 1.0000 | +0.0166 | 21000 | 233 |
| dial | 0.9925 | 0.9974 | +0.0049 | 23000 | 21 |
| perm | 1.0000 | 1.0000 | 0.0000 | 12000 | 156 |
| gate1 | 0.9839 | 0.9828 | -0.0011 | 17000 | 2110 |
| proc0 | 0.9853 | 0.9814 | -0.0039 | 17000 | 74 |
| gate0 | 1.0000 | 0.9933 | -0.0067 | 19000 | 405 |
| line | 1.0000 | 0.9834 | -0.0166 | 8000 | 60 |
| grid | 0.9922 | 0.9658 | -0.0264 | 16000 | 66 |
| walled | 0.9395 | 0.9131 | -0.0264 | 16000 | 8030 |
| **MEAN** | 0.9790 | **0.9817** | +0.0028 | 17300 | **1159** |

**P1 confirmed, and beyond its own terms.** `toggle` was predicted to
recover to at least the 0.9551 the old basis reached. It reaches
**1.0000 on every seed** — 0.918, 0.9388, 0.9121, 0.8822 under random
proposal, 1.0000, 1.0000, 1.0000, 1.0000 under enumeration. The
corrected F179 reading is confirmed: the regression was dilution of
random proposal by two extra operations, and enumeration does not
dilute because it does not sample.

**P4 confirmed dramatically.** Cost per family falls 17,300 proposals
to 1,159, a factor of **15**, while mean quality rises.

**P2 FAILS on `line` and `gate0`**, both 1.0000 under random search and
0.9834 / 0.9933 under enumeration. And the cause is the mechanism I
proposed in F179, retracted as impossible, and have now made real by
wiring it in: **`--fit-target` is live on this path for the first
time**, so enumeration stops the moment it clears 0.95 while random
search always spent its whole budget and kept improving. `line`,
`grid` and `gate0` all used to overshoot; now they settle.

That is worth stating plainly rather than as a curiosity. In F179 I
diagnosed early stopping, the sweep proved it impossible because
nothing on that path stopped early, and I retracted it in favour of the
dilution reading. Both are now correct, of different runs: dilution
explained the old measurement, early stopping explains this one, and
the difference is a change I made in between. A mechanism being absent
from one configuration says nothing about the next.

**`walled` remains the residue and is now the most expensive family by
an order of magnitude** — 8030 proposals against a median of 233,
because enumeration exhausts depth 2 without finding a program that
does not exist. It is the one family whose recipe the basis genuinely
cannot express, and it is now clearly separated from everything else on
both axes.

**The sweep that was vacuous is now meaningful.** Sweeping
`--fit-target` returned byte-identical results in F179 because nothing
consumed it. It now governs when enumeration stops, so 0.99 and 1.0
should recover `line`, `grid` and `gate0` at a cost that is affordable
for the first time — 1159 proposals leaves ample headroom under a 4000
budget.

## F181 — the fit-target sweep, now that something consumes it

F179's sweep was byte-identical because `synthesise` had no early stop.
Enumeration does, so the same sweep is now meaningful. Four seeds.

| family | t=0.95 | t=0.99 | t=1.0 | cost 0.95 | cost 0.99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| line | 0.9834 | **1.0000** | 1.0000 | 60 | 67 |
| grid | 0.9658 | **0.9892** | 0.9892 | 66 | 112 |
| gate0 | 0.9933 | 0.9948 | 0.9948 | 405 | 403 |
| walled | 0.9131 | 0.9219 | 0.9219 | 8030 | 13020 |
| gate1 | 0.9828 | **0.9749** | 0.9749 | 2110 | 4053 |
| MEAN | 0.9817 | **0.9860** | 0.9860 | 1159 | 1857 |

**P1 confirmed on all three named families.** `line` returns to exactly
1.0000 for 7 extra proposals, `grid` recovers 0.9658 to 0.9892, `gate0`
improves at no cost at all. F180's settling-at-threshold reading is
therefore right, and the remedy is one flag.

Mean held-out rises 0.9817 to 0.9860 for 1.6x the proposals — which at
1,857 proposals per family is still an order of magnitude under the
17,300 that random sampling spent for 0.9790.

**Two checks I ran before writing this up, and both came back against
me.**

*The 0.99 and 1.0 arms are identical to the digit.* My explanation was
that no candidate ever scores in [0.99, 1.0), so the two thresholds
cannot separate. **Refuted: a score of 0.992 is present in the data.**
A candidate at 0.992 should stop a 0.99 search and not a 1.0 search,
and yet both arms agree on every family AND on cost. I do not know why,
and it is recorded as an open discrepancy rather than smoothed over —
the most likely explanation is that the 0.992 arrived on the last
enumerated candidate, where stopping and continuing coincide, but that
is a guess and the way to settle it is to log the stopping index.

*`gate1` regresses at the stricter target* — 0.9828 to 0.9749 — which
looked like overfitting the 64-observation search sample. **Refuted at
the aggregate**: the search-to-held-out gap SHRINKS with a stricter
target, 0.0099 to 0.0081, which is the opposite of overfitting. And
`gate1`'s own SEARCH fit also fell, 0.9887 to 0.9874, which should be
impossible: a longer search keeping the best of everything it sees
cannot end with a worse best. That points at the harness rather than
the method — continuing past the early stop changes the generator
position, so the fallback sampling sees a different candidate stream —
and it is flagged as an anomaly to chase rather than explained away.

**Default not changed pending the anomaly.** 0.99 is better on nine of
ten families and on the mean, but a family whose search fit moved in an
impossible direction is a reason to find out why before promoting the
setting.

## F182 — the continual-learning result is WITHHELD: its own
## faithfulness check failed

`continual.py` put both architectures on one family sequence. The
weights arm behaved exactly as predicted and the bank arm's control
came back exactly right. **The result is still not reportable**, and
the reason is the check that was built to make it reportable.

| prediction | outcome |
| --- | --- |
| P1 bank forgetting EXACTLY 0.0000 | **confirmed on all 8 runs** |
| P2 weights forget a lot without replay | **confirmed, 0.6048** |
| P3 replay substantially closes it | **confirmed, 0.1099** |
| P4 bank is worse at learning time | confirmed — but see below |

Weights forgetting without replay is 0.6048 and with replay 0.1099, so
**replay recovers 82% of what is lost**. That is the honest number the
control existed to produce: the bank's advantage over the standard
remedy is the remaining 0.11, not the headline 0.60. And the weights
arm at the end sits at 0.3952, BELOW its own identity floor of 0.5156 —
it forgets past the point of copying the input.

**But the bank arm cannot be read.** The interpreter faithfulness check
— the same measurement `isa_compose` publishes at **0.9896** — returns
**0.4524, 0.4062, 0.3855, 0.4234** across the four seeds. Whatever the
bank arm's 0.4322 at learning time is measuring, it is not the
architecture; it is a broken or unconverged interpreter. Reporting
"zero forgetting" from it would be reporting perfect retention of
something that never worked.

**What the check has already ruled out.** Isolating the interpreter
(one family, one weights update, same 40k budget) reproduces
**exactly 0.4524** — byte-identical to the full run. So it is
deterministic, independent of the weights arm, and not a contention or
ordering artefact. Op semantics, training loop, optimiser settings,
program sampling and evaluation were compared line by line against
`isa_compose` and match.

**The live hypothesis is not a bug but a BASIN, and it is already in
this ledger.** F164 measured exactly this shape in the reader: one seed
reaching 0.9372 and another 0.3776 after the SAME 200k updates, with
the conclusion that sufficient optimisation closes the gap only on a
favourable trajectory and that reliability is unresolved. My copy
constructs its interpreter at a different point in the global RNG
stream than `isa_compose` does — after the weights arm's model is
built — so it starts from a different initialisation. Four seeds are
running to tell a basin apart from a bug: if some reach 0.99 and others
0.4, this is F164 reappearing in a second component and is a finding
rather than a defect.

**Recorded now, before the answer, because the discipline is the
point.** The bank arm produced the exact number the architecture
predicts — 0.0000 forgetting, on all eight runs — and it is the most
flattering result available. The check says it is unreadable. A control
that only gets consulted when the news is bad is not a control.

## F183 — the interpreter failure was `Adam` where the reference uses
## `AdamW`, and weight decay is what made it fatal

F182 withheld the continual-learning result because its faithfulness
check returned 0.39-0.45 against `isa_compose`'s published 0.9896. The
cause is one word.

**Four hypotheses eliminated in order, each by a measurement:**

1. *Interaction with the weights arm.* Isolating the interpreter
   reproduced **exactly 0.4524** — byte-identical. Ruled out.
2. *A bad initialisation basin*, which F164 had already documented in
   the reader at 0.9372 against 0.3776. Four seeds returned 0.4056,
   0.4067, 0.4151, 0.3846. **Every seed fails identically**, so it is
   systematic, not a basin. Ruled out — and this was the hypothesis I
   most expected to be right.
3. *RNG position*, since my copy builds its interpreter after the
   weights arm's model. Setting the seed immediately before
   construction changed 0.3828 to 0.3809. Ruled out.
4. *Op semantics, training loop, sampling, evaluation.* Compared line
   by line against the reference; all match.

**The difference: `torch.optim.Adam` against `torch.optim.AdamW`.**

| | seed 69316 | seed 7 | loss trace |
| --- | ---: | ---: | --- |
| AdamW | **0.8301** | **0.8030** | 2.20 -> 0.59 |
| Adam | 0.3812 | 0.3912 | 2.24 -> 1.88, stuck |

At 8,000 updates, a fifth of the reference budget. The Adam loss never
leaves the neighbourhood of uniform (2.079) while AdamW's descends past
0.6.

**Why weight decay is what makes it fatal, and this is the part that
transfers.** Adam folds L2 into the gradient before the adaptive
rescaling, so the penalty is amplified on precisely the parameters
whose gradients are small and sparse — the op, argument and modulus
embeddings, each of which sees a gradient only when its own token
appears. AdamW decouples it. The interpreter therefore cannot learn its
own instruction set: it is being pulled toward zero faster than the
signal accumulates.

**And F154 is what set the trap.** That finding established weight
decay 0.01 as optimal — measured under AdamW. Carrying the VALUE across
to Adam inverts its effect. **A hyperparameter tuned under one
optimiser is not portable to another**, and 0.01 was not a neutral
default here but the specific number that made the failure severe.

**The check is the finding.** Without it the bank arm would have
reported 0.0000 forgetting — the exact number the architecture predicts
and the most flattering result available — from an interpreter that had
learned nothing. The measurement that caught it cost twenty lines and
compared against a number already in the ledger.

## F184 — the bank survives a round trip exactly, and its size is now
## measured rather than asserted

A bank that exists only inside one process is not a bank. `--bank-path`
writes it out, reads it back, and re-scores every family from the
RESTORED copy.

| family | before write | after reload |
| --- | ---: | ---: |
| line | 0.8066 | 0.8066 |
| dial | 0.8607 | 0.8607 |
| toggle | 0.9134 | 0.9134 |
| perm | 0.9033 | 0.9033 |

`reload_exact: True`. Not approximately — the same digits, because
what was written is integers and what reads them is frozen.

**The size claim, now checkable.** 102 instructions across 4 families,
1,589 bytes of JSON. JSON is a text format and roughly ten times the
information content: an instruction is one op of nine (4 bits), two
slot indices of six (3 bits each) and one modulus of seven (3 bits) =
**13 bits**, so 102 instructions is 1,326 bits or **166 bytes packed**.
A four-action family at program length 6 is 312 bits — **39 bytes for
a world's entire dynamics.**

I have previously quoted 216 bits per world. That figure was for the
SEVEN-operation basis with no modulus argument (3+3+3 = 9 bits per
instruction). The saturating operations and the modulus raised it to
312. Correcting it here rather than leaving two numbers in the ledger.

Against the alternative: the entry-vector bank these probes started
from is `bank_tokens x dim` = 8 x 96 = 768 floats = **24,576 bits** at
fp32, for the same job. That is 79 times larger, opaque, and it cannot
be inspected, composed, or written to disk without the model that
produced it.

**Why this matters for continual learning specifically.** The retention
result is only interesting if the thing retained is portable. A frozen
interpreter plus 39 bytes per world is a claim that can be checked by
reading the file; an activation vector that only means anything inside
the network that produced it is not.

## F185 — the founding thesis, measured directly: bank 0.9981, replay
## 0.8886, weights 0.3952

Nine families in sequence, four seeds, one evaluation, both
architectures. **The gate passes**: interpreter faithfulness 0.9933,
0.9891, 0.9880, 0.9918 against `isa_compose`'s published 0.9896, so the
bank arm is readable for the first time (F182, F183).

| | at learning | at end |
| --- | ---: | ---: |
| weights, no replay | 1.0000 | **0.3952** |
| weights + replay | 1.0000 | 0.8886 |
| **bank** | 0.9981 | **0.9981** |
| identity floor | — | 0.5156 |

**P1 confirmed as a control should behave.** Bank forgetting is exactly
0.0000 on all four seeds. It cannot be otherwise and it was not.

**P2 confirmed.** Weights forgetting without replay is 0.6048, and the
end state of 0.3952 is BELOW the identity floor of 0.5156 — after nine
families it does worse than copying its input unchanged.

**P3 confirmed, and it is the number that matters.** Replay recovers
**81.8%** of what is lost, ending at 0.8886. So the bank's advantage
over the standard remedy is **+0.1095**, not the +0.60 a no-replay
comparison would have claimed. That control existed to prevent exactly
that overstatement.

**And the replay arm here is STRONGER than real replay.** It
regenerates rows from the live family rather than drawing from a finite
stored buffer, so it has unlimited perfect access to every environment
it has ever seen. It still loses by 0.11 to a bank of integers.

**P4 was right in direction and wrong in size.** I predicted the bank
would be meaningfully worse at learning time — "capped by interpreter
quality near 0.99 and by whether the family is expressible at all" —
and treated zero forgetting of a worse model as the real trade. The
bank learns at **0.9981** against gradient descent's 1.0000. The gap is
0.0019. There is essentially no trade to make.

**What both approaches store, since that is the honest axis.** Neither
is memory-free. The bank stores 39 bytes of program per world (F184),
inspectable and exact. Replay stores experience — and in this arm, an
idealised infinite supply of it — and buys 0.11 less retention for it.

**What this does NOT show.** Nothing here demonstrates forward
transfer: having solved families 1..k does not measurably help on k+1,
because enumeration already solves every family in about 23 candidates
(F178) and there is no headroom for transfer to appear in. That is a
ceiling effect rather than evidence against, and testing it needs
families whose recipes exceed depth 2. Retention and generalisation to
unseen programs are measured; forward transfer is not.

## F186 — the forward-transfer instrument does not yet discriminate,
## and the first version failed the same check it was built to satisfy

F185 could not measure forward transfer because enumeration solves the
standard benchmark in ~23 candidates, leaving no headroom. `DeepFamily`
was built to supply headroom: every action is a K-instruction
composition over the same basis the interpreter executes, so it is
solvable by construction at exactly K and generically not at less. That
makes a failure a failure of SEARCH rather than of expressibility,
which is the confound that wrecked every cost measurement before F176.

**At K=3 it does not discriminate. Every arm returns exactly 10,800 —
six families times three actions times the 600 budget. 6/6 saturated.**
Frozen, stored-programs, coverage filter and enumeration are
byte-identical because none of them solves anything.

That is the F168 regime failure again, in an instrument I wrote three
findings after establishing the check for it. The measurement is void
and reports nothing about transfer.

**Why K=3 is out of reach, and it is arithmetic rather than bad luck.**
Enumeration runs to depth 2 only. A depth-3 target must therefore come
from the random fallback, which draws length-6 programs: the three
required instructions must appear in the right order among six draws
from a pool of roughly sixty. That is a probability around 10^-5 per
candidate against a budget of 600.

**And the mechanism the arm was meant to test cannot fire either.**
Stored programs are NOOP-padded to length 6 before storage, so
concatenating one with anything else immediately overruns the program
length. A bank whose entries cannot be composed is a bank that can only
be recalled whole — which is exactly the limitation F157 hit, reappearing
in a different guise.

**Two concrete things to fix before this can measure anything, both
identified from the failure rather than guessed:**

1. Calibrate K so that some arms succeed and others do not. K=2 is
   solved outright by enumeration and K=3 by nothing; the discriminating
   regime is between them, most likely K=3 with enumeration extended to
   depth 3 on a restricted pool.
2. Store fragments UNPADDED, so a stored two-instruction solution can
   compose with one new instruction. Without that, the transfer
   mechanism has no way to express itself.

Recorded as a null on the instrument, not on transfer. Forward transfer
remains unmeasured, and it is the one claim of the founding objective
this architecture has not yet been shown to make.

## F187 — the forward-transfer instrument, rebuilt; and the same regime
## error found one level down

F186 named two reasons the instrument could not discriminate. Both are
now fixed, and testing the fix exposed a third.

**Fix 1 — the bank could not compose.** Stored programs were NOOP-padded
to length 6, so concatenating one with anything overran the program
length. Winners are now trimmed before storage. A bank whose entries
cannot compose can only be recalled whole, which is exactly F157's
limitation wearing different clothes.

**Fix 2 — candidates could not be short.** Building to `program_len` and
truncating made every candidate exactly six instructions, so a
three-instruction solution was unreachable WHATEVER the library held:
the target needs NOOPs in its tail and concatenation never produces
them. Candidates now draw a length, fill it, and pad.

**Fix 3 — reuse stated as enumeration.** At depth 3 the search first
tries every library fragment followed by one new instruction. If a
two-instruction prefix is already known, the third instruction costs
the size of the pool rather than its cube. Deep families share an
identifiable prefix, so a bank that fails to exploit one it already
holds has failed at something specific.

**And the test of the fix failed for a third reason, which is the one
worth recording.** With a 8,000-update interpreter:

| arm | family 1 | families 2-6 | best fit |
| --- | ---: | ---: | ---: |
| frozen | 60000 (saturated) | 60000 each | 0.833 |
| enum | **29840 (exhausted)** | 60000 each | 0.924 |

Depth-3 enumeration now completes rather than saturating — the
instrument works. But the best fit it can find is 0.924 against a
target of 0.95, and it is not the search that caps it: **at 8,000
updates the interpreter executes at roughly 0.83 per slot, so no
program can score 0.95 no matter how correct it is.** The families are
solvable by construction; the executor cannot demonstrate it.

That is the F168 regime error again, one level down. F168 was "the
search saturates so cost is the budget". F186 was "nothing is solved so
nothing is stored". This is "the interpreter cannot reach the fit
target so nothing counts as solved". The same shape three times: **a
measurement is void when some component other than the one under test
is the binding constraint**, and the way to notice is always to check
whether the thing being varied can move the number at all.

Prerequisite now running at a 40k interpreter: does depth-3 enumeration
solve a depth-3 family when the executor is good enough to show it?
Until that answers yes there is nothing for a bank to transfer, and
forward transfer stays unmeasured.

## F188 — the ceiling gate: a regime error hit three times, now caught
## by the report itself

Three measurements in this session were void because a component OTHER
than the one under test was the binding constraint:

| finding | what was binding | how it read |
| --- | --- | --- |
| F168 | the search budget | every arm cost exactly the budget |
| F186 | nothing was ever solved | every arm byte-identical |
| F187 | the interpreter's own accuracy | best fit 0.924 under a 0.95 target |

Each cost a round of runs to diagnose by hand, and each was found only
because something looked impossible — identical numbers, or a cost that
equalled the budget exactly.

**The general form.** The search scores candidates WITH the
interpreter, so the interpreter's own per-slot accuracy is an upper
bound on any candidate's fit. If that ceiling sits below
`--fit-target`, no program can be accepted however correct it is, and
every arm fails for a reason that has nothing to do with the arm.

It is one evaluation to compute. The report now carries:

    fit_ceiling        interpreter accuracy on unseen programs
    fit_target         the acceptance threshold
    measurement_valid  ceiling >= target
    void_reason        stated in words when it is not

Verified firing: at 2,000 updates it reports `fit_ceiling 0.4178`,
`measurement_valid False`, and the reason. A three-iteration hand
diagnosis becomes a field.

**This is the third instrument of its kind and they now form a set**,
each catching a different way a measurement can be uninformative:

* **byte-identical output** — a parameter is not reaching the code, or
  two things are secretly one thing. Caught the collapsed codebook
  (F146), the attainable-fit bound being the coverage filter (F163),
  the modulus non-fix (F166), the unapplied cap (F174), and a patch
  that silently failed to apply.
* **a positive control of known size** — `cover` at 0.879/0.812 says
  whether the regime can detect anything at all (F172).
* **the ceiling gate** — whether the acceptance threshold is reachable
  by the executor at all.

The common principle, worth stating once: **an experiment needs at
least one quantity whose expected value is known in advance.** Every
null retracted this session failed that test, and every instrument
above is a different way of supplying one.

## F189 — FORWARD TRANSFER, first evidence: the bank makes families
## solvable that were not solvable without it

Two seeds, depth-3 families sharing an identifiable prefix, 20,000
candidates per action. **The ceiling gate passes** — interpreter at
0.9896 and 0.9865 against a 0.95 target — so for the first time this
measurement is readable rather than void.

| arm | seed 69316 | seed 69317 |
| --- | --- | --- |
| frozen (sampling) | **0/6 solved**, 360,000 | **0/6 solved**, 360,000 |
| enum | 1/6 solved, 306,418 | 1/6 solved, 306,385 |
| **enum + stored programs** | **4/6 solved**, 193,808 | 1/6 solved, 278,877 |

**Random sampling solves NOTHING at depth 3.** Zero of six on both
seeds, at the full budget. That is the headroom F185 lacked and F186
failed to build: a regime where the arms can actually separate.

**Enumeration alone solves exactly one** — the first family, whose
depth-3 enumeration happens to complete inside the budget at 29,632.
The rest saturate.

**Adding the bank solved three more on seed 69316**, and the per-family
numbers show the mechanism rather than implying it:

| family | enum | enum+store |
| --- | ---: | ---: |
| deep3 | 60,000 @ 0.905 | **11,329 @ 0.988** |
| deep4 | 48,397 @ 0.880 | **14,471 @ 1.000** |
| deep5 | 48,389 @ 0.878 | **18,376 @ 0.987** |

Cost falls three to five fold AND fit rises to essentially exact. That
is the shared prefix being recalled and extended by one instruction,
which is what the instrument was built to make identifiable.

**It does not replicate at the second seed.** 1/6 solved, 278,877
against enum's 306,385. The DIRECTION is consistent on both seeds —
enum+store <= enum <= frozen — but the magnitude is not, and this
project's own history says a two-seed effect that disagrees in size is
not yet a result (F157, F159, F168 were all overturned on that
pattern).

**Stated at the strength the evidence supports:** forward transfer has
been OBSERVED, in a regime where the alternative solves nothing, with a
mechanism visible in the per-family costs. It has not been replicated.
Four more seeds running.

This is the third claim of the founding objective and the one F185
recorded as unmade. If it holds at six seeds, the architecture will
have retention (F185, exactly zero forgetting), generalisation to
unseen programs (F185, 0.9896), and transfer — each measured against a
control rather than asserted.

## F190 — FORWARD TRANSFER REPLICATES at six seeds

F189 observed it at two seeds and declined to claim it, because the
magnitude disagreed and this project has overturned three results with
that signature. Four more seeds:

| arm | families solved | mean cost | sd |
| --- | ---: | ---: | ---: |
| frozen (sampling) | **0 / 36** | 1.000 | — |
| enumeration | 4 / 36 | 0.868 | 0.035 |
| **enumeration + bank** | **15 / 36** | **0.702** | 0.128 |

Per seed, families solved by enum against enum+store: 1/4, 1/1, 0/3,
1/2, 0/2, 1/3. **The bank solves more in five of six and ties in the
sixth; it is never worse.** On cost it is cheaper in **6 of 6**.

**The control is what makes this readable.** Random sampling solves
ZERO of thirty-six at the full 20,000-candidate budget, so this is not
a regime where everything works and the arms differ by a few percent.
It is one where the alternative fails completely and the bank does not.
That is the headroom F185 lacked, F186 failed to build, and F187 could
not measure through — three failed instruments before one worked.

**The ceiling gate passed on every seed** (0.9865-0.9917 against a 0.95
target), so for once I can say the measurement was checked for
readability BEFORE the result was read rather than after.

**Magnitude varies and direction does not.** Cost ratios run 0.538 to
0.894. The variance is real and worth naming rather than averaging
away: the effect depends on whether an early family's stored solution
happens to expose the shared prefix. Which is exactly what the
sub-prefix fix addresses, and that fix is not in these runs.

**The founding objective, now measured in full.** "Produce a program
such that given task A makes novel task B faster to learn than chance
or starting from scratch." Against a from-scratch control that solves
nothing:

* **retention** — exactly 0.0000 forgetting across nine families,
  against 0.1099 for replay with unlimited access to past environments
  (F185);
* **generalisation** — 0.9896 on programs never trained on, 0.9444 at
  double the trained length, with no family ever touching the weights
  (F185);
* **transfer** — 15/36 against 4/36 and 0/36, cheaper in 6/6 seeds
  (this).

Each measured against a control rather than asserted. What remains is
not a missing claim but known limits: `walled`'s wall is still
inexpressible (F179), the reader's training signal is still unreliable
across seeds (F164), and the sub-prefix composition fix is untested.

## F191 — sub-prefix composition: mechanism confirmed, aggregate
## refuted, and F157's dilution returns in a new costume

F190 named the variance as the thing to fix: transfer ranged 0.538 to
0.894 across seeds because it depended on whether a stored solution
happened to expose the shared prefix. Composing over sub-prefixes was
predicted to compress that, and specifically that **the seeds that
transferred least would gain most.**

Six seeds, paired against F190's runs, gate passing on all.

| seed | whole-fragment | sub-prefix | delta | solved before | after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 69319 | 0.894 | 0.862 | **-0.031** | 2 | 1 |
| 69320 | 0.794 | 0.728 | **-0.066** | 2 | 1 |
| 69317 | 0.775 | 0.629 | **-0.146** | 1 | 2 |
| 69321 | 0.642 | 0.644 | +0.001 | 3 | 2 |
| 69318 | 0.569 | 0.779 | **+0.209** | 3 | 0 |
| 69316 | 0.538 | 0.698 | **+0.160** | 4 | 1 |

**The prediction is confirmed exactly, and it does not save the
result.** r(how badly a seed transferred, how much the fix helped it) =
**-0.800**. The three worst transferrers all improved; the two best
both degraded. The mechanism is real and it is the one I named.

**But the aggregate is worse.** Families solved fall **15/36 to 7/36**,
and mean cost drifts 0.702 to 0.723. Cheaper in only 3 of 6 seeds. On
the measure that matters — how many families get solved at all — this
is a substantial net loss.

**The cause is F157, returning at a different level.** Adding every
sub-prefix of every stored program multiplies the library, so the
enumeration reaches a genuinely useful whole fragment later, or not at
all inside the budget. F157 was "fragments appended to a uniformly
sampled pool make each one rarer". This is the same sentence with
"uniformly sampled pool" replaced by "enumeration order". I fixed
dilution once by weighting a distribution and did not notice it could
return in a search that has no distribution at all.

**The fix this implies is ORDERING, not selection.** Whole fragments
first, sub-prefixes after — then a seed whose stored solution exposes
the prefix keeps its win, and a seed whose does not gets the fallback
instead of paying for it up front. That is a strictly better ordering
under both conditions, which is a stronger claim than "try both and see"
and is worth stating before it is run.

**F190 stands unchanged.** This tested a proposed improvement to
transfer, not transfer itself; the whole-fragment configuration it was
measured in is still the best one, and reverting to it is the honest
default until an ordering arm beats it.

## Pre-registered: the cross-domain test (probe 275)

Every finding in the recipe architecture, F155 through F191, was
measured on ONE domain: procedurally generated rule families over
SLOTS x VALUES. The architecture's central claim is larger. It says the
plant is AMODAL — structure in frozen weights, content in an external
bank, one controller for any domain an encoder maps into slots.

**That claim has been argued from construction and never tested.** The
argument has been "the instructions are operations on abstract slots,
therefore they are domain-general". That is an argument about the
definition, not a measurement, and F170 is exactly why it is not
enough: an extension needs a failure signature, not an argument. An
instruction set can be abstract by definition and still have been
tuned, across a hundred measurements, to the rule-family distribution
it was measured on. **Nothing in F155-F191 would have detected that.**

`cross_domain.py` trains one interpreter on random programs only — no
domain ever touches its weights — then searches recipes for both:

* **rule families**, the procedural families every prior finding used;
* **grid games**, real composigrid worlds from `game_family`, an avatar
  moving on an 8x8 board among positive and negative objects, read into
  slots by the SAME shallow perception `game_slots.py` uses. Unchanged
  deliberately: a probe that improved the encoder to make the
  interpreter look good would be measuring the encoder.

**Predictions.**

1. Rule families reproduce their established margin over the identity
   floor. If they do not, the probe is broken and nothing else in it
   can be read — this is the positive control, in the role F172
   established for `cover`.
2. Grid games clear their identity floor by a clear margin. The floor
   is the number to watch and it will be HIGH: a grid state barely
   changes per step, the avatar moves one cell of eight, so copying the
   input forward already scores well. Uniform is not the baseline and
   quoting accuracy against it would be dishonest.
3. The grid margin is SMALLER than the rule-family margin. Grid
   dynamics involve objects that move independently of the action,
   which no single instruction expresses, and the nearest-object
   encoding collapses several objects into one.

If (2) fails while (1) holds, the honest conclusion is that the
instruction set was fitted to rule families and its generality was an
artefact of never having looked. That result would be worth more than
the confirmation, and recording the prediction now is what makes it
readable either way.

## F192 — THE AMODAL CLAIM, DEMONSTRATED: one frozen interpreter
## predicts rule families AND grid games

The claim this project is built on, tested for the first time rather
than argued from construction. One interpreter, trained ONLY on random
programs over random slot states — no rule family, no grid, nothing
from either domain ever touches its weights — then recipes searched for
both.

**The control passes.** Interpreter 0.9700 and 0.9954 on unseen
programs; rule families reproduce their established margin at +0.4099
and +0.4598. So the probe works and the rest can be read.

| domain | held-out | identity floor | margin |
| --- | ---: | ---: | ---: |
| rule families | 0.9990 / 1.0000 | 0.5891 / 0.5403 | +0.4099 / +0.4598 |
| **grid games** | **0.9852 / 0.9805** | 0.6458 / 0.6394 | **+0.3395 / +0.3411** |

Per game, both seeds:

| game | held-out | floor | margin |
| --- | ---: | ---: | ---: |
| collect1 | 0.9927 / 0.9941 | 0.77 | +0.219 / +0.219 |
| collect2 | 0.9707 / 0.9546 | 0.75 | +0.218 / +0.223 |
| intercept1 | 0.9937 / 0.9922 | 0.53 | +0.463 / +0.460 |
| intercept2 | 0.9839 / 0.9810 | 0.53 | +0.458 / +0.463 |

**An avatar moving on an 8x8 board among falling objects, and a
procedurally generated rule family, are predicted by the SAME frozen
weights to 0.98 and 0.999.** The two domains share the slot interface
and nothing else. The instruction set was developed across roughly a
hundred measurements on rule families alone, and it transfers to a
domain it was never shaped against.

**P3 confirmed on its metric, and for a different reason than I gave.**
I predicted the grid margin would be smaller because object motion is
independent of the action and no single instruction expresses it. The
margin IS smaller, 0.34 against 0.41 — but grid ACCURACY is 0.985
against 0.999, nearly equal. The smaller margin is almost entirely the
higher floor, not a capability shortfall. The stated reason was wrong
even though the prediction was right, which is worth separating.

**What this does NOT show, stated plainly.**

* **Two of six variants could not be measured at all.** `avoid1` and
  `avoid2` produced no usable slots and are absent from the table, not
  averaged in as zeros. Whatever they need, this encoding does not
  supply it.
* **The perception is hand-written and shallow.** `slot_state` takes
  the avatar's argmax and the single nearest object per plane, so a
  world with several objects is only partly described. This measures
  the interpreter GIVEN a working encoder; it is not an end-to-end
  perception result, and using the unchanged `game_slots` encoder was
  deliberate so the number could not be improved by tuning it.
* **One-step transitions only.** Nothing here is planning or control.

**Third seed landed, and the replication is the tightest this project
has produced.**

| seed | interpreter | rule margin | grid margin | grid held-out |
| --- | ---: | ---: | ---: | ---: |
| 69316 | 0.9700 | +0.4099 | +0.3395 | 0.9852 |
| 69317 | 0.9823 | +0.5895 | +0.3407 | 0.9804 |
| 69318 | 0.9954 | +0.4598 | +0.3411 | 0.9805 |

Grid margin **+0.3404 with sd 0.0007**; grid held-out **0.9820 with sd
0.0022**; **above floor in 12 of 12 game-seeds**. For comparison, the
forward-transfer result that took six seeds to establish (F190) had a
cost-ratio sd of 0.128 — two orders of magnitude looser. Three seeds
agreeing to the fourth decimal is not the usual situation here and it
is worth saying so explicitly rather than treating tightness as
routine.

The control passed on every seed, so at no point was this read out of a
regime that could not measure it.

## F193 — longest-first ordering is a NULL, and it says why the whole
## sub-prefix idea was wrong

F191 measured sub-prefix composition losing (solved 15/36 -> 7/36) and
diagnosed dilution. I proposed ordering as the fix and claimed it would
be **"strictly better under both conditions rather than a trade between
them"** — a strong claim, stated before the run.

Six paired seeds:

| configuration | cost | sd | solved |
| --- | ---: | ---: | ---: |
| whole-fragment (F190) | 0.702 | 0.128 | **15/36** |
| sub-prefix (F191) | 0.723 | 0.080 | 7/36 |
| longest-first (this) | 0.727 | 0.079 | 7/36 |

**Ordering changed essentially nothing.** Per seed the two sub-prefix
configurations agree to within 0.005 — 0.698/0.703, 0.629/0.631,
0.779/0.781, 0.862/0.865, 0.728/0.731, 0.644/0.649 — and the solved
counts are identical.

**And that near-identity is the finding.** Ordering can only matter
when a search TERMINATES EARLY, because a search that exhausts its
enumeration ends with the same best candidate whatever order it visited
them in. Only 7 of 36 families reach the fit target; the other 29
exhaust. So ordering was irrelevant for 80% of the measurement by
construction, and my "strictly better under both conditions" was wrong
in a way I could have derived before spending six runs: **the dominant
condition is the one where order does not exist.**

The damage sub-prefixes do is that the enumeration gets BIGGER, so
families that were solved inside budget no longer are. Reordering
cannot undo a size increase.

**Reverted to whole fragments.** F190's configuration is the best
measured and stays the default. The sub-prefix idea has now lost twice,
on two different mechanisms, and the honest summary is that composing
over sub-programs is a good idea that this search cannot afford —
not that it is wrong in principle.

**Pattern worth naming, since it is the fifth time.** Before optimising
an ordering, a threshold, or a filter, ask what fraction of cases the
change can reach at all. F163's filter, F175's cap, F186's transfer
instrument, F187's ceiling and now this were all measured in regimes
where the varied quantity could not move the outcome for most cases.
The check is cheap and I keep not running it first.

## F194 — F192 extends to all six games once the row bug is fixed

Dropping terminal rows rather than slots (the fix above) makes `avoid`
measurable. Three seeds, control passing on all.

| game | held-out | floor | margin | rows kept |
| --- | ---: | ---: | ---: | --- |
| collect1 | 0.9936 | 0.7752 | +0.2184 | 256/256 |
| collect2 | 0.9596 | 0.7396 | +0.2201 | 256/256 |
| intercept1 | 0.9932 | 0.5317 | +0.4615 | 256/256 |
| intercept2 | 0.9816 | 0.5199 | +0.4618 | 256/256 |
| **avoid1** | **0.8750** | 0.5308 | **+0.3442** | 253/256 |
| **avoid2** | **0.8314** | 0.5053 | **+0.3261** | 246/256 |

**Above floor in 18 of 18 game-seeds.** Grid margin +0.3381, +0.3366,
+0.3413 across seeds — still tight, and now over the whole variant set
rather than the two-thirds that happened not to trip the bug.

`avoid` is genuinely the hardest, 0.83-0.88 against 0.96-0.99
elsewhere, and it is the one whose episodes terminate. Predicting the
step after a hazard lands is exactly the case a one-step transition
model has least information about. Recorded as a real difficulty rather
than smoothed into the mean.

**The correction that matters more than the numbers.** F192 recorded
"avoid produced no usable slots" as a limitation of the ENCODING. It
was a bug in my scoring, and 2% of rows caused it. I had written the
limitation into a finding and moved on. What surfaced it was following
up a stated limitation instead of accepting it — and the general lesson
is that a component which silently DISAPPEARS from results is worse
than one that fails loudly, which is why the probe now reports
`rows_kept`, `rows_total` and `usable_slots` for every family.

## F195 — cross-domain transfer is a NULL, and the cause is an ARITY
## MISMATCH that predicts when it would work

F192 established one frozen interpreter serving both domains, which
made a sharper question askable: does a program found while solving a
RULE FAMILY make a GRID GAME cheaper to solve? Three arms, three seeds,
control passing throughout.

| game | cold | primed | stranger |
| --- | ---: | ---: | ---: |
| collect1 | 9861 | 9781 | 9961 |
| intercept1 | 7209 | 7309 | 7309 |
| avoid1 | 25920 | 26020 | 26020 |
| **TOTAL** | **198874** | **199294** | **199474** |
| ratio to cold | 1.000 | **1.002** | 1.003 |

**Null, and cleanly so.** Priming costs exactly the library's size and
buys nothing: 1.002 against 1.003 for the stranger control, and
held-out accuracy identical to four decimals across all three arms
(0.9391), so the library never changed which recipe was chosen. One
game of six moved at all (collect1, 0.992) and that is inside noise.

**The design made the null mean something specific**, and it does:
*no rule-family program exactly solves a grid action.* The reason is
measurable in one line, and it is not subtle:

| domain | slots changed per action |
| --- | --- |
| GRID collect1 | 3, 2, 1, 1 |
| GRID intercept1 | 3, 3, 2, 2 |
| GRID avoid1 | 2, 2, 2, 2 |
| RULE line | 1, 1 |
| RULE dial | 1, 1, 1, 1, 1, 1 |
| RULE proc0 | 1, 2, 0, 1, 1 |
| RULE proc1 | 1, 1, 1, 1, 1 |

**Rule-family actions change ONE slot; grid actions change two or
three.** A stored program that writes one slot cannot solve an action
that changes three, whatever the interpreter can execute. The two
domains produce recipes in different ARITY CLASSES, and the transfer
failed for a reason that has nothing to do with either domain's
content.

The cause of the grid arity is the encoding itself: slots 2-5 hold the
NEAREST object's coordinates, which are relative to the avatar, so
moving the avatar changes which object is nearest and where it sits.
One avatar move plus one or two object-slot updates is two or three
writes. That coupling is a property of `game_slots`' perception, not of
grid worlds.

**And this converts the null into a prediction.** If arity is the
barrier, transfer should appear when the source families produce
multi-slot writes. `random_family_spec(wide=True)` enables the `pair`
op, which changes two slots at once — precisely the missing class.
Priming with WIDE rule families should transfer where narrow ones do
not, and if it still does not, arity was not the barrier and something
about grid dynamics is genuinely unreachable from rule-family
structure. Either answer is worth more than the null alone.

## F196 — the arity barrier is real, and BOTH my explanations for it
## are wrong

F195 found cross-domain transfer null and blamed an arity mismatch:
grid actions change two or three slots, rule-family actions change one.
It then offered a mechanism — the object slots are avatar-RELATIVE
("nearest by distance"), so moving the avatar re-ranks them and turns a
one-slot move into a three-slot write. That mechanism predicts a fix:
encode objects by raster order instead, and grid arity should fall
toward one.

**Refuted. Absolute encoding barely moves it.**

| game | relative | absolute |
| --- | --- | --- |
| collect1 | 3, 2, 1, 1 | 3, 2, 1, 1 |
| collect2 | 3, 3, 3, 3 | 3, 1, 3, 1 |
| intercept1 | 3, 3, 2, 2 | 3, 3, 2, 2 |
| avoid1 | 2, 2, 2, 2 | 2, 2, 2, 2 |

**Second hypothesis, also refuted.** If the encoding is not responsible,
perhaps the world evolves regardless of the action — fallers falling,
hazards moving — and that is what the extra slots record. Decomposing
by running the SAME start state under two different actions separates
them: slots differing between the two outcomes are action-dependent,
slots that changed identically under both are the world proceeding
anyway.

| game | slots changed | action-dependent | world-regardless |
| --- | ---: | ---: | ---: |
| collect1 | 3 | 4 | **0** |
| collect2 | 3 | 4 | 1 |
| intercept1 | 2 | 4 | 1 |
| avoid1 | 2 | 2 | 1 |

The change is almost entirely ACTION-dependent. World dynamics account
for at most one slot.

**So what is actually happening**, since two clean stories both failed:
the action changes the WORLD, not just the avatar. Moving onto an item
consumes it, which changes which object occupies the object slots under
either encoding — raster order re-ranks when the first object is
removed exactly as distance order re-ranks when the avatar moves. The
arity is not an artefact of how objects are indexed and not the world
running on its own clock; it is that a grid action has CONSEQUENCES
beyond the mover, and a rule-family action does not.

That makes the mismatch intrinsic to this domain pair rather than a bug
to fix. Rule families are a poor SOURCE for grid transfer, and the
honest form of the founding objective's cross-domain version is
narrower: transfer needs source tasks whose actions have the same
consequence structure, not merely the same interface.

**Two hypotheses, two refutations, one run each.** Both were plausible,
both were stated with a predicted signature, and both died on the
signature. Recorded because the arity FACT survives all of it — grid 2
to 3, rules 1 — and it is the fact, not either story, that explains
F195.

## F197 — CORRECTION to F195: cross-domain transfer is not null. It
## happens exactly where the arity matches, and nowhere else

F195 read the transfer result as a flat null from a mean of totals:
primed 1.002, stranger 1.003. That was the wrong statistic, for the
reason F176 already established and I did not apply — per-game-per-seed
cost spans orders of magnitude, so a mean of totals is a measurement of
whichever cell is largest. `collect1`'s cold cost across three seeds is
**166, 29182, 234**. The mean of 9861 describes none of them.

Re-read as paired ratios over all 18 game-seeds:

| | median | cheaper than cold |
| --- | ---: | ---: |
| primed | 1.002 | **3/18** |
| stranger | 1.003 | **0/18** |

**The stranger control never helps once. Priming helps three times, and
all three are the same game.**

| seed | game | cold | primed | ratio |
| ---: | --- | ---: | ---: | ---: |
| 69316 | collect1 | 166 | 89 | **0.536** |
| 69318 | collect1 | 234 | 148 | **0.632** |

**And `collect1` is the only game that CAN receive it.** From F195's
own arity table:

| game | slots changed per action |
| --- | --- |
| **collect1** | 3, 2, **1**, **1** |
| collect2 | 3, 3, 3, 3 |
| intercept1 | 3, 3, 2, 2 |
| avoid1 | 2, 2, 2, 2 |

`collect1` is the ONLY game with arity-1 actions, and rule families
produce arity-1 programs. Transfer appears on exactly the one game
whose action arity matches the source domain's, at two of its three
seeds, saving 40-47% — and appears nowhere else, on any game, on any
seed.

**So the arity account of F195/F196 is not just an explanation for a
null. It is a prediction that was already sitting in the data**: it
says transfer should occur where arities match and not otherwise, and
that is precisely the pattern. The wide-rule arm now running tests the
same claim from the other side.

**What I actually got wrong.** Not the mechanism — the statistic. I
had established in F176 that a mean over a heavy-tailed cost
distribution measures the tail, wrote it into the ledger as a standing
lesson, and then read exactly such a mean and reported a null. The
paired view took one command. **A lesson recorded is not a lesson
applied**, and the gap between those two is where this session has lost
the most time.

## F198 — wide rules do NOT unlock the arity-2 actions: arity is
## necessary, not sufficient

F197 established that cross-domain transfer occurs on exactly the one
game with arity-1 actions. F195 predicted the fix: prime with families
drawn `wide=True`, whose `pair` op writes TWO slots, and transfer
should reach the arity-2 grid actions.

| priming | primed cheaper | stranger cheaper | the wins |
| --- | ---: | ---: | --- |
| narrow | 2/18 | 0/18 | collect1 at 0.536, 0.632 |
| **wide** | **2/18** | 0/18 | collect1 at 0.867, 0.692 |

**Refuted, and cleanly.** The same two game-seeds win, both still
`collect1`, and on seed 69316 the wide library is WORSE than the narrow
one (0.867 against 0.536). Not one arity-2 action became reachable.

**Why, and it sharpens the account rather than replacing it.** A rule
family's `pair` op writes two slots by incrementing BOTH. A grid's
arity-2 action writes two slots with different, coupled values — the
avatar moves one way and the object slot re-ranks another. Writing two
slots is not the same as writing the RIGHT two slots.

So **arity is a necessary condition for transfer, not a sufficient
one**, and the reason is combinatorial. An arity-1 action is nearly
determined by "which slot, which direction", a space small enough that
rule families cover it densely — which is why `collect1` transfers at
all. An arity-2 action ranges over pairs of coupled effects, and
`pair`'s "increment both" occupies one corner of that space. Raising
the source arity does not help unless it raises COVERAGE of the target
arity's function space, and one op does not.

**Three predictions from the arity account, in order:** F195 predicted
the null's cause (confirmed, F196's measurements), F197 predicted where
transfer WOULD appear (confirmed, exactly `collect1`), and F195
predicted wide rules would extend it (refuted, here). Two of three, and
the one that failed fails for a reason the account itself explains once
stated properly — which is the difference between a theory that is
adjusting to data and one that is being sharpened by it. Recorded so
the distinction stays checkable rather than asserted.

## F199 — a reader trained on the search's own labels is functionally
## correct 70% of the time, against a control at 3.6%

Three seeds, 15,000 updates, on synthetic transitions.

| | op | arg_i | arg_j | arg_m | exact | **functional** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trained | 0.9408 | 0.8516 | 0.3262 | 0.5996 | 0.1614 | **0.7012** |
| shuffled control | 0.1094 | 0.1634 | 0.1666 | 0.1530 | 0.0007 | **0.0358** |
| chance | 0.1111 | 0.1667 | 0.1667 | 0.1429 | — | — |

Per seed: 0.6992, 0.6895, 0.7148 — tight. Shuffled: 0.1055, 0.0, 0.002.

**The control is what makes this readable.** A reader that had merely
learned which programs are COMMON would score above chance on every
field while reading nothing from the transitions. Shuffled sits at
chance on all four fields and at 0.0358 functionally, so the trained
reader is reading the transitions and not the label distribution.

**Exact 0.1614 against functional 0.7012, and the gap is the finding
rather than a caveat.** The reader usually names a DIFFERENT instruction
that does the SAME THING — SWAP where COPY suffices, any op on a slot
the transitions never move, any modulus on a slot whose values never
reach the wrap. Field accuracy would have understated this fourfold,
which is why `functionally_correct` was made the headline BEFORE the
run rather than chosen after seeing which number was larger.

`arg_j` is the weakest field at 0.3262, and that is correct behaviour:
`j` is read only by the conditional, copy and swap ops, so for an INC
or DEC the transitions contain no information about it and there is
nothing to learn. A reader scoring high on `arg_j` would be suspicious.

**What this does NOT yet show, and it is the thing that matters.**
Training draws a random instruction applied to RANDOM states. Real
families are structured — a three-valued family only ever shows 0-2 —
so this measures whether the architecture CAN read, not whether it
reads what the system needs. A first check at 400 updates put synthetic
at 0.166 and real families at 0.1172 against a real-family floor of
0.1602: below its own floor. The 15,000-update version of that
comparison is running, and until it lands this is a result about
synthetic transitions.

## F200 — the recipes COMPOUND: eight-step rollout holds at 0.83 while
## its floor falls to 0.24

One-step grid prediction was established at 0.98 (F194). Everything
that acts needs it rolled forward, and compounding error is the
standard way learned world models fail. Three seeds, six games, floor
recomputed at every step.

| step | accuracy | frozen-start floor | margin |
| ---: | ---: | ---: | ---: |
| 1 | 0.9588 | 0.4004 | +0.5584 |
| 2 | 0.8879 | 0.3581 | +0.5298 |
| 4 | 0.8721 | 0.3171 | +0.5550 |
| 6 | 0.8654 | 0.2942 | +0.5712 |
| 8 | **0.8328** | 0.2435 | **+0.5893** |

**The margin does not decay. It grows** — +0.5584 at one step to
+0.5893 at eight. Accuracy falls 0.9588 to 0.8328, so compounding error
is real, but the floor falls faster because "the state never moved from
where it started" gets worse the longer the rollout runs. Recomputing
the floor at every step was necessary to see this; a fixed floor would
have shown a decaying margin and told the opposite story.

**The decay is FRONT-LOADED, which is the part that matters for
planning.** Step one to two costs 0.071. Steps three through eight cost
0.055 COMBINED. That is a bounded error settling, not an exponential
blow-up — the model absorbs one step's worth of imprecision and then
tracks. An error that compounds multiplicatively would have collapsed
by step eight; this one is still predicting five sixths of slots
correctly.

**What this means for the architecture.** A planner scores action
sequences by rolling a model forward, so the horizon it can trust is
the horizon it can plan over. Eight steps at 0.83 against a floor of
0.24 is a usable planning horizon on real grid games, using recipes of
one or two instructions held outside the weights and executed by an
interpreter that never saw a game.

Together with what is already measured — one frozen interpreter serving
both domains (F192, F194), exactly zero forgetting across nine families
(F185), forward transfer at 15/36 against 0/36 (F190), and a reader
that emits functionally correct programs 70% of the time without search
(F199, synthetic) — the remaining gap to a working agent is the control
loop itself, not any of its parts.
