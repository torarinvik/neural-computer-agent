# Literature map

Prior work mapped onto this project's open problems. Organised by *our*
problem, not by paper, and every entry is labelled with what it does to us:

- **CONFIRMS** — independent support for a mechanism we already promoted.
- **PREDICTS** — the literature predicts a failure we measured, which
  reclassifies that failure from "bug" to "expected outcome".
- **CONTRADICTS** — the literature says something we believed is wrong.
- **MECHANISM** — a specific untried technique with a route into our code.

Searched 2026-08-08. Papers are cited for their mechanism; none of this is
evidence about *our* system until we run it. Nothing here is promoted.

---

## 1. Making the bank NECESSARY (weakness 12, F48, current work)

Our problem: cross-feed inverts (bank can override) but a norm-matched decoy
costs nothing for one twin (bank is not necessary). The plant keeps choiceA as
a bank-free default; the bank stores only choiceB's deviation. Raising the
ignorance weight 4x moved nothing (0.969/0.906 decoy on the default twin).

**PREDICTS — output-space objectives cannot fix this.** Entropy and
KL-to-uniform penalties are monotone transformations of the logit gaps: they
shrink gaps but never reorder them at any finite weight. Greedy evaluation
extracts the surviving order. This is the structural reason our ignorance
objective failed, and it is the same phenomenon the machine-unlearning
literature calls suppression-not-removal — "Failure Modes of Zero-Shot Machine
Unlearning in RL and Robotics" (2025) shows forgotten behaviour returns under
re-elicitation because it was suppressed at the output, not removed from the
weights. Our greedy decoy gate *is* a re-elicitation probe, which is why it
keeps finding the default.

**PREDICTS — two contexts is the wrong number.** Chan et al., "Data
Distributional Properties Drive Emergent In-Context Learning" (NeurIPS 2022)
and its companion on in-context vs in-weights generalisation: context-reliance
and weight-reliance trade off, and which wins is set by the training
distribution. Few classes, each frequent, each with a *fixed meaning* drives
in-weights absorption. Many classes, bursty, with **dynamic item meanings**
drives in-context reliance. Our twin protocol is the worst case of the first
regime. Their follow-up shows in-context reliance can *decay* with prolonged
training as weights absorb the mapping — a reason to re-measure our decoy gate
late rather than at convergence.

**CONFIRMS/REFRAMES — delta coding is a feature, not our bug.** Residual policy
learning (Silver et al. 2018; Johannink et al. ICRA 2019), task arithmetic
(Ilharco et al. ICLR 2023), the Kanerva Machine (Wu et al. ICLR 2018), and
predictive-coding accounts of memory (Barron et al. 2020) all store
deviations-from-a-prior deliberately, and treat it as optimal coding. Biology
does the same: schema-consistent content is absorbed into the model, deviant
content gets episodic encoding. The literature's verdict is that storing
differences is correct *provided the reference is the cross-context marginal
rather than one context's policy*.

This gives us a better statement of our own bar. Not "the bank stores
everything", which fights an optimal coding scheme, but:

> the plant's bank-free policy is at chance on **every** context, and each
> context's policy is a bank-supplied deviation of comparable magnitude.

That is what F11's asymmetry actually violates, and it is testable directly
(measure bank-free score per context) rather than only through the decoy.

**MECHANISM — information asymmetry in the gradient path.** Galashov et al.,
"Information Asymmetry in KL-Regularized RL" (ICLR 2019) and Tirumala et al.,
"Behavior Priors" (JMLR 2022): what a default policy absorbs is controlled by
*what it can see and how much capacity it has*, not by penalty strength. Their
default policy is denied task-identifying inputs, so it converges to the
marginal by construction. Our plant is the default-policy analogue, so the
transfer is: allow per-context gradients into fragments only, and update plant
weights on the mixture objective — the plant then *cannot* specialise to
choiceA because it never receives a choiceA-specific gradient. This is
architecturally enforceable, unlike a penalty.

**MECHANISM — penalise task information in the weights.** Yin et al.,
"Meta-Learning without Memorization" (ICLR 2020) is the closest published
formalisation of our exact failure: with non-mutually-exclusive tasks the
meta-learner stores task solutions in the meta-parameters and ignores the
context channel. Their fix is an information bottleneck on the *weights*
(penalise I(weights; task)), making the context pathway the only channel with
capacity. Parameter-space, so argmax cannot defeat it.

**MECHANISM — sever the bypass architecturally.** Taguchi & Tsuruoka (2018)
found NTM/DNC controllers solve memory tasks in the recurrent controller and
bypass external memory; their fix was to restrict the recurrent pathway so
persistence has no route except through memory. Our F48 working-memory leak is
the same pathology. Our version: make the policy head degenerate without
fragments — e.g. multiplicative gating by fragment features rather than
additive context tokens — so a bank-free forward pass has no expressible
policy.

**MECHANISM — task-vector negation.** Ilharco et al. (ICLR 2023): fine-tune
bank-free on choiceA, take the weight delta, subtract it. Operates below the
softmax, so it cannot be defeated by argmax-preserving flattening. A direct
attack on "erase the default".

**MECHANISM — environment-side necessity.** Memory Gym (Pleines et al., ICLR
2023) argues partial observability is insufficient and designs for *strong
memory dependency*: tasks unsolvable without recall. By their criterion our
task pair fails, because one context is near-solvable bank-free. The fix is
structural: >=2 contexts arranged so that no fixed policy beats chance in
expectation over contexts — i.e. every context is some other context's
anti-policy, so there is no free default to keep.

**Instruments we lack.** Per-state KL(pi_with-bank || pi_bank-free) from
InfoBot (Goyal et al. ICLR 2019) localises *where* the bank matters and would
show choiceA contributing ~zero deviation directly. Distillability of the bank
into the plant (how few steps until bank-free matches with-bank) is a
continuous necessity margin, sharper than our binary decoy.

**Field-position note.** Retrieval-augmented RL (Goyal et al. ICML 2022)
evaluates retrieval as *helpful* (positive delta), not *necessary* (ablation to
chance). We have not found a necessity protocol stricter than ours in this
literature. Independently, the RAG community measured our decoy result: adding
*random* documents to context can improve QA accuracy because the model falls
back on parametric knowledge (Cuconasu et al. SIGIR 2024) — noise plus a strong
default survives, in a completely different modality.

---

## 2. Probe -> infer -> fetch -> execute (F43-F47)

Our result: context is knowable only from consequences of acting; a learned
router queried from the agent's own intention collapses to identical selections
for both twins unless the encoding is staged/supervised first (F44/F46), and
the encoding needs protecting through later plastic phases (F47).

**CONFIRMS — this chicken-and-egg is a named, solved-by-staging problem.**
DREAM (Liu, Raghunathan, Liang, Finn; ICML 2021) names it exactly: learning to
explore requires good exploitation to gauge the exploration's utility, and
exploiting requires information that exploration gathers. Their solution is
ours: train the encoding first with privileged supervision (an information
bottleneck on the ground-truth task ID), then train exploration to make that
encoding recoverable, then execute conditioned on the recovered code. They
demonstrate that end-to-end joint learning gets stuck in the identical-selection
local optimum — which is F43 to the letter.

**CONFIRMS — nobody learns a reward-only task encoding from policy gradient
alone.** PEARL (Rakelly et al. ICML 2019) trains the task latent through the
critic's Bellman error; VariBAD (Zintgraf et al. ICLR 2020) trains the belief
through a reward/transition reconstruction ELBO decoupled from the RL loss;
DREAM uses privileged IDs. All three ground the representation outside the
policy gradient. Our probe-supervised encoding is the same move, arrived at
independently.

**CONFIRMS — scaled retrieval systems freeze the key space.** R2A (Goyal et al.
ICML 2022) and Humphreys et al. (NeurIPS 2022) both query into *pretrained,
frozen* embedding spaces; the network learns to use retrievals, not to define
the key space. Our staged-then-handover design is the norm in this literature,
not a workaround. NTM/DNC-era work reports the same: learning content-based
addressing jointly with the content it addresses is notoriously unstable and is
usually fixed with curriculum or careful initialisation.

**CONFIRMS — the probe phase has precedent.** Denil et al., "Learning to
Perform Physics Experiments" (ICLR 2017): an interaction phase where objects of
different mass *look identical* (our twin property exactly), followed by a
commitment action. It trains end-to-end — but the "fetch" there is a single
discrete answer, not a learned retrieval, which supports our finding that the
retrieval query is the hard part, not the probe.

**MECHANISM — VariBAD's reward decoder as an alternative to our probe
supervision.** A decoder trained to predict future rewards from the belief
forces the hidden state to encode reward-sign evidence whether or not the
policy has learned to use it. This is a self-contained fix for the
encoding-destruction failure of F46 that does not require our staged handover.

**MECHANISM — intrinsic information gain to un-fix the probe.** Our probe
action is hand-fixed. MetaCURE (Zhang et al. ICML 2021) rewards the explorer
for mutual information with task identity; HyperX (Zintgraf et al. ICML 2021)
adds novelty bonuses in (state, belief) space specifically to break the
encoder-side bootstrap deadlock. Either would let the probe be learned without
collapsing, when we want to remove the hand-fixed action.

**Formal name for our twins.** Hidden-Parameter MDPs (Doshi-Velez & Konidaris,
IJCAI 2016): a family indexed by a latent parameter fixed within an episode.
Our twins are the degenerate case where the parameter is invisible in the
transition function and visible only in reward.

**Gap.** No paper combines (i) observationally identical reward-mirrored
contexts, (ii) an explicit probe protocol, and (iii) mid-episode retrieval of
*executable skill programs* from an external bank. DREAM has (i) and a
task-code but no program bank; Ritter et al.'s episodic LSTM (ICML 2018)
reinstates stored computational state mid-episode but keys on observable cues,
which our twins remove.

---

## 3. Composition (weakness 11, F16/F27/F33/F34)

Our four failed mechanisms: imposed factorial sharing, partner rotation, a
permutation-invariant combiner, cross-context fine-tuning. All at chance on
held-out pairings.

**PREDICTS — chance was the correct prediction, not a bug.** Lippl &
Stachenfeld, "When Does Compositional Structure Yield Compositional
Generalization? A Kernel Theory" (ICLR 2025): kernel-regime models are
restricted to *conjunction-wise additivity* — they can only assign values to
component conjunctions actually seen in training and sum them. Full pairwise
coverage is necessary but not sufficient. Fu et al. (2024) prove a **No Free
Lunch theorem for compositional generalisation**: no assumption-free solution
exists; composition must be paid for with either matching inductive bias or
matching training-distribution structure. Wiedemer et al. (NeurIPS 2023) give
sufficient conditions, one of which is that the *model architecture mirrors the
compositional structure* — which an opaque-fragment plant does not.

Our F33/F34 negatives are therefore promotable as confirmations of theory, and
our own diagnosis ("the agent never has to succeed on an unseen pairing") is
the literature's conclusion.

**MECHANISM — the study-phase channel is the MLC protocol.** Lake & Baroni
(Nature 2023): every episode carries study examples of a *freshly resampled*
grammar and is graded on queries requiring recombinations absent from that
episode's study set. Memorising any single mapping is useless across episodes,
so in-context induction is the only strategy that reduces loss. Ported to us:
resample fragment semantics per episode, expose (A+B) and (C+D) as study
rollouts, reward only on (A+D). Compositional-ARC (2025) shows this transfers
to non-linguistic grid transformations at 5.7M parameters, so scale is not the
blocker. AdA (Bauer et al. ICML 2023) is the RL-native existence proof:
in-context adaptation to held-out tasks, from meta-RL over a vast task space
plus attention memory plus an automatic curriculum at the capability frontier.

**MECHANISM — a bottleneck between fragment-inference and execution.**
Kobayashi et al. (2024) is the most on-point paper for our failed combiner:
transformers meta-trained on a subset of module combinations *fail* to compose
in-context and solve seen combinations non-compositionally — until an
information bottleneck separating task inference from task execution is
inserted, which flips the result. Our FragmentCombiner had no such bottleneck.

**MECHANISM — manufacture the recombination pressure if the environment won't
supply it.** Akyürek et al., "Learning to Recombine and Resample" (ICLR 2021)
synthesises examples realising unseen combinations and resamples training
toward them; Lee & Chung (NeurIPS 2021) train on imagined mixtures of latent
task dynamics; Spilsbury et al. (EMNLP 2024) generate support demonstrations
when no relevant ones exist, beating oracle retrieval of real ones. All three
are portable as a "dream phase" splicing recorded rollouts into held-out-style
pairings.

**CONTRADICTS (mildly) — our conclusion that four exhausted mechanisms means
the problem needs a new channel.** Two escape routes exist that we have not
tried, and one of them (imposed composition algebra — successor features and
GPI (Barreto et al.), the Boolean task algebra (Nangue Tasse et al. NeurIPS
2020), skill machines (ICLR 2024)) buys *provable* zero-shot composition. We
should record explicitly that we are declining it because it requires giving
fragments fixed semantics, which violates the opaque-fragment principle — not
because it doesn't work. It is the fallback if the study channel fails.

**Also relevant to our "interchangeable but not composable" result (F27).**
Jarvis et al. (ICLR 2023) prove modular architectures are necessary but not
sufficient for specialisation: modules trained jointly encode the *conjunction*
unless coupling is made costly. Devin et al. (ICRA 2017) got mix-and-match
zero-shot transfer with an explicitly narrow, regularised module interface;
Béna & Goodman (Nat. Comms 2024) show specialisation emerges only under
resource constraints. Our fragment interface may simply be wide enough to smuggle
partner information — a cheap thing to test by shrinking fragment width.

**Evaluation caveat.** Staged-emergence work (2025) reports compositional
ability appearing long after in-distribution loss saturates. Our failed
mechanisms should be re-probed at longer horizons before we call them closed.

---

## 4. Selector collapse, winner-take-all, and the anti-sharing penalty
(weaknesses 11/13, F12/F13, F23/F24)

**PREDICTS — the seed lottery is a symmetry break, so "balancing relocated the
winner" is the expected signature.** Routing collapse is a rich-get-richer
feedback loop needing no asymmetry in the data: a slightly preferred expert
gets more gradient, improves, and becomes more preferred; starved experts get
noisier gradient and degrade. Statistical-mechanics analyses of MoE describe a
symmetric (unspecialised) phase and a symmetry-broken phase past a critical
sample size, with the realised partition selected by fluctuation — i.e. by
seed. Rosenbaum et al. (2019), the canonical catalogue of routing pathologies,
names **module collapse** and the **module-router co-adaptation problem**
(router locks in an assignment before modules are good enough for it to be
right). Our measured result — balancing relocates rather than removes the
winner — is what this predicts. Removing the lottery requires changing the
dynamical system, not reweighting the objective.

**PREDICTS — marginal balance can never prevent our collapse.** A
load-balancing loss constrains only the *marginal* usage per fragment. Two
contexts following identical selection rules satisfy it perfectly. Our
uniform-floor balancing is a marginal constraint; F12-style collapse is a
*conditional* failure. This is a clean explanation of why our floor helped
retention (F24) but not collapse.

**CONTRADICTS — our diversity penalty is the wrong instrument, and there are
three better ones.** We built a pairwise repulsion penalty, then recorded as
weakness 13 that it also forbids legitimate sharing. The literature treats that
tension as solved:
- **Loss-free balancing** (Wang et al. 2024, used in DeepSeek-V3): a per-expert
  additive *bias* on selection scores, updated by a control loop on usage
  error, not by gradient. Under-used fragments get boosted until chosen. It
  never penalises two contexts for choosing the same fragment, and injects zero
  interference gradient into the RL objective.
- **BASE layers** (Lewis et al. ICML 2021) and **expert-choice routing** (Zhou
  et al. NeurIPS 2022): balance by *assignment* rather than penalty. With two
  contexts and a handful of fragments, the assignment problem is tiny and a
  Sinkhorn/Hungarian solve is nearly free.
- **Shared + private partition** (DeepSeekMoE 2024; Domain Separation Networks,
  NeurIPS 2016; MOORE, ICLR 2024): put orthogonality on fragment *content*,
  never on context *selections*. MOORE orthogonalises the basis via
  Gram-Schmidt while contexts freely mix over it — diversity and sharing stop
  competing. This is the structural fix for weakness 13.

**MECHANISM — routing geometry may be the actual cause of F12/F43 collapse.**
X-MoE (Chi et al. NeurIPS 2022) argues routing *causes* representation
collapse: when routing scores are dot products between hidden states and expert
keys in the same space the policy uses, the routing gradient pulls
representations toward the winning key, clustering them. Fix: compute selection
scores in a low-dimensional routing-only space, L2-normalised, with a learnable
temperature. If our selection logits share the intention space, this is a
plausible root cause of both F12 and F43 and is cheap to test. ST-MoE's router
z-loss (penalising logit magnitude) is a second cheap regulariser keeping
selection soft and reversible.

**MECHANISM — check reward scale before anything else.** Hessel et al.'s PopArt
(AAAI 2019) argues the dominant cause of one task swamping others in multi-task
RL is reward scale and density, not representational conflict. Our contexts
differ in reward economics. Per-context return normalisation is a two-line
change that may dissolve our winner-take-all outright, and should be ruled out
before more routing machinery.

**MECHANISM — better balancing signal and better metric.** Teacher-Student
Curriculum Learning (Matiisen et al. 2017) samples by *learning progress*
(slope), not by lowest score — importantly it deprioritises a context stuck
flat at zero, where our laggard-preferential sampler would pour samples in, and
its forgetting term resamples any context that starts regressing. And the right
headline number for our battery is the **worst-context** return under a minimax
objective, not the mean: "balancing relocated the winner" is invisible in a
mean and obvious in a worst-case. Agarwal et al. (NeurIPS 2021) add the
statistical machinery — IQM and performance profiles over seeds, where a
bimodal profile is the fingerprint of a lottery.

**MECHANISM — freeze the selector after burn-in.** StableMoE (ACL 2022) names
*routing fluctuation* as distinct from imbalance and fixes it by distilling then
freezing the router. For us this is a sharp diagnostic: if freezing selection
after burn-in removes the seed lottery, the lottery lives in the selector, not
the plant.

**Diagnostics we should run before more tuning.** TAG inter-task affinity
(Fifty et al. NeurIPS 2021) measures whether a gradient step on context i
reduces context j's loss — turning "should these contexts share?" from a
hyperparameter into a measurement. PCGrad's cosine similarity between
per-context gradients (Yu et al. NeurIPS 2020) tells us whether our
winner-take-all is genuine gradient conflict at all; if the cosine is positive,
no routing surgery will help and the problem is acquisition.

**Note for RL specifically.** Obando-Ceron et al. (ICML 2024) find that in deep
RL the *soft* (fully differentiable, no discrete routing, no balancing loss)
MoE variant is what unlocks parameter scaling. Soft modularisation (Yang et al.
NeurIPS 2020) makes sharing continuous rather than a discrete tie to arbitrate.
If our selection can be made soft with a temperature schedule, the RL-side
evidence favours it.

---

## 5. Continual learning without replay (weaknesses 4, 15; F24, F40-F42, F47)

**RISK — our arbitrated release rule may open the gate on exactly the
most-mastered skills.** Recent analyses of EWC ("Elastic Weight Consolidation
Done Right", 2026) show the empirical Fisher *collapses* when a network is
confident and correct, because log-likelihood gradients vanish — importance is
systematically under-estimated for well-learned skills. Our plant is a
REINFORCE policy: a mastered game yields a near-deterministic policy, tiny
score-function gradients, and therefore tiny F. Our rule `a = F/(F + mu*G)`
then releases protection *most* where mastery is *highest*.

*Checked against our code (`two_speed_battery.family_fisher`).* We compute the
empirical Fisher from the score function of sampled actions, so the vanishing
mechanism is present — but we then normalise each game's Fisher to **unit
mean**, which divides the absolute scale out. The cross-game under-protection
the paper describes is therefore already mitigated: a mastered game's Fisher is
rescaled to the same mean as any other game's.

That leaves a sharper, unmitigated version of the risk. Normalisation rescales
the *magnitude* but not the *signal-to-noise ratio*. As a policy saturates, the
per-parameter Fisher estimate is increasingly dominated by sampling noise —
and unit-mean normalisation then amplifies that noise to look like a confident
protection pattern. The predicted failure is not "mastered games are
unprotected" but "mastered games are protected in **arbitrary directions**",
which is worse, because it is invisible to any check that only looks at the
penalty's magnitude. Testable directly: estimate F twice with independent seeds
and correlate, as a function of policy entropy. High correlation means the
pattern is real; correlation decaying toward zero as entropy falls means we are
protecting noise. (Related: "Fishers for Free?" 2507.18807 notes Adam's
squared-gradient accumulator is a serviceable free Fisher, giving F and G from
one piece of machinery.)

**NOVELTY — no published rule has our form.** The nearest relatives all differ
in a specific way: online EWC (Schwarz et al. ICML 2018) decays old protection
*uniformly and unconditionally* by gamma; UPGD (Elsayed & Mahmood, ICLR 2024)
gates the step by per-weight utility but has **no new-task demand term**; AFEC
(NeurIPS 2021) releases where old knowledge conflicts with the new task but
implements it by temporary parameter expansion; TRGP (ICLR 2022) relaxes
gradient orthogonality where the new task overlaps old subspaces. Per-parameter,
demand-conditioned release as a closed-form ratio appears to be ours. A
related-work paragraph against those four covers the claim.

**CONFIRMS — F24's retention/acquisition split is the field's central
tradeoff.** Continual World (Wołczyk et al. NeurIPS 2021) reports that the
methods best at preventing forgetting *lose forward transfer*, with scratch
training often acquiring faster than continual learners; they measure
forgetting and transfer as separate metrics. RWalk (ECCV 2018) supplies the
published vocabulary: **forgetting vs intransigence**. Our F24 ("the floor
cures retention; acquisition is a separate axis") is an independent
rediscovery, and our arbitration rule is a candidate resolution.

**MECHANISM — a policy-preserving easing axis for the timing games.** This is
the most directly useful result for weakness 15. Reverse Curriculum Generation
(Florensa et al. CoRL 2017) eases along the **start state**, not the dynamics:
sample starts near the goal and expand outward, keeping success in a band.
Start-state easing leaves the optimal policy on visited states unchanged *by
construction* — so it satisfies our curriculum law where our speed (F41) and
spread (F42) axes provably did not, because slowing the ball changes the
optimal action-timing function itself. For intercept: initialise episodes close
to the interception moment at true dynamics, and expand the initial distance
outward. A 2025 table-tennis-catching paper reaches the same practical
conclusion — ease the *context distribution*, never the dynamics.

**CONFIRMS + NOVELTY — our curriculum law is half-formalised in the
literature.** Ng, Harada & Russell (ICML 1999) proved potential-based shaping is
the *only* reward modification that preserves the optimal policy — the formal
ancestor of our law's first clause, but for the reward axis only. Self-paced
deep RL (Klink et al. NeurIPS 2020) formalises the *pace* along an axis via a
KL budget, and is benchmarked on ball-catching, but presumes the axis is sound.
No published criterion for environment-easing-axis *validity* was found, and
our second clause (must exercise the full range of the target policy's outputs)
has no antecedent at all. That makes F40-F42 a genuine contribution rather than
a local lesson.

**MECHANISM — episode-level exploration for fatal-timing acquisition.** Our
intercept failure may be an exploration-unit mismatch: per-step entropy
dithering never executes a precisely timed sequence by chance, whereas the
credit unit is a whole timed attempt. The published remedy is temporally
consistent exploration — Bootstrapped DQN's per-episode head sampling (Osband
et al. NeurIPS 2016), or parameter-space noise (Plappert et al. ICLR 2018),
which ports to REINFORCE directly: sample a policy-head perturbation once per
episode rather than dithering per step.

**MECHANISM — plasticity, and which remedies suit an on-policy plant.** Dohare
et al. (Nature 2024) establish that plain backprop loses the ability to learn at
all over long non-stationary sequences, remedied only by continually injecting
variability. But Moalla et al. (NeurIPS 2024) study plasticity loss
specifically in *on-policy* RL and find many off-policy remedies fail there,
while **regenerative regularisation** is the class that consistently works —
i.e. L2 toward the *initial* parameters (Kumar et al., ICLR 2025), one
hyperparameter, no resets. That matters for us because reset/recycle methods
(ReDo, continual backprop) actively fight consolidation: they could recycle a
unit our Fisher regards as load-bearing. There is an elegant synthesis
available: L2-Init anchors at theta_0, EWC anchors at theta*, and **our `a`
already computes the per-parameter blend** — protected parameters anchor at
theta*, released parameters anchor at theta_0. That would make one dial cure
both forgetting and plasticity loss. Cheap, consolidation-compatible, and
untested. LayerNorm inside the recurrent cell is a second low-friction
intervention (Lyle et al. ICML 2023).

**MECHANISM — acquisition reliability has published levers we have not
pulled.** Andrychowicz et al. (ICLR 2021), ~250k trained agents on on-policy
design choices, find that small final-layer policy init (near-zero initial
action mean, which keeps early entropy high), observation/return normalisation,
and larger batch sizes are the strongest variance reducers. These are the
closest thing to a recipe for making single-task on-policy acquisition reliable
rather than a lottery, and all three are trivial to apply to our plant.

**CAUTION — a third explanation for our seed-dependent acquisition failures.**
"Prevalence of Negative Transfer in Continual RL" (ICLR 2025) shows prior
learning frequently *hurts* later acquisition, distinct from forgetting and
from plasticity loss. Our forageA/intercept1 failures could be negative
transfer from the consolidated prior rather than an intrinsic difficulty of the
game — distinguishable by a from-scratch control on the failing seed, which we
have not run.

**Reporting.** Henderson et al. (AAAI 2018) is the canonical citation that our
one-seed-acquires/one-doesn't observation is systemic to policy-gradient
training. Colas et al. (2018) give power analysis — with bimodal outcomes we
should model acquisition as Bernoulli per seed and power on *acquisition rate*,
not mean return. Agarwal et al. (NeurIPS 2021) give performance profiles and
IQM, which surface exactly the lottery that a battery mean hides.

---

## What this changes

1. **The ignorance objective is a dead end as an output-space penalty** (S1).
   Our F49 candidate should record the structural reason, not just the number.
2. **Our necessity bar is better stated as "bank-free is at chance on every
   context"** (S1), which is measurable directly and does not fight optimal
   delta coding.
3. **Two contexts is probably the root cause of the default asymmetry** (S1),
   and the fix is more contexts with re-randomised bindings, not more pressure.
4. **Our staged addressing (F44-F47) is the field-standard solution** to a named
   problem (S2) — worth stating in those terms.
5. **Our composition negatives are theory-confirmed** (S3), and the study-phase
   channel we proposed is a published protocol with an RL existence proof.
6. **Weakness 13 has three published structural fixes** (S4) and our diversity
   penalty should be retired in favour of one of them.
7. **Cheap diagnostics we have never run** (S4): per-context return
   normalisation, gradient cosine, inter-task affinity, worst-context metric.
8. **Our promoted consolidation rule has a predicted failure mode** (S5): a
   vanishing Fisher on mastered policies means `a` releases protection where
   mastery is highest. This is a live risk in shipped code, not a future
   concern, and it is testable.
9. **Intercept has a policy-preserving axis we never tried** (S5): start-state
   easing at true dynamics, which satisfies our own curriculum law where speed
   and spread did not. Weakness 15 is not blocked on an intrinsic tension.
10. **Three of our results are candidate original contributions** (S1/S3/S5):
    the demand-conditioned release rule, the curriculum-axis validity law
    (especially the full-output-range clause), and the decoy-plus-cross-feed
    necessity protocol — retrieval work in this literature tests whether memory
    *helps*, not whether it is *necessary*.

---

# Second sweep: eight papers against the navigation/object/goal stack (2026-08-16)

User-supplied list, read against the caveats standing after the successor
transfer, learned decomposition and object identity records. Same labels, same
caution: mechanism until it runs here.

## S6. The probe policy is uniform and unexamined (successor_transfer caveat)

**MECHANISM — prediction error as the exploration signal.** Burda et al.,
"Large-Scale Study of Curiosity-Driven Learning" (arXiv:1808.04355): agents
rewarded purely by forward-model prediction error explore 54 environments with
no extrinsic reward at all, and in many games the curiosity objective aligns
with the score. Our tabular analogue is nearly free: the world-model already
knows which `(place, action)` cells are unvisited and which tables still
disagree, so "go where the model is wrong" is a `w` vector — visit-count or
disagreement cumulants fed to the same GPI machinery we just built, no new
learning rule. That would replace uniform wandering in `explore()` and is
measurable as coverage per step against the uniform arm.

**PREDICTS — the noisy-TV failure is our `random_walk` distractor.** Their
headline limitation: prediction-error curiosity is captured by any source of
irreducible stochasticity (the "TV"). We already built the discriminator the
fix needs — the matched contrast in `TrackEvidence.controllability` scores a
random walker at 0.0 precisely because its surprise never resolves. A
curiosity bonus here must be *controllability-gated* (seek states where the
model is wrong AND the action matters), or the distractor becomes the most
interesting thing in the scene. This is a prediction we can test directly:
un-gated curiosity should chase the distractor; gated should not.

**MECHANISM — Agent57's split, in tabular form.** Badia et al.
(arXiv:2003.13350) decompose Q into separate extrinsic and intrinsic heads
combined as `Q^e + beta*Q^i`, keep a *family* of (beta, gamma) policies from
exploratory to exploitative, and let a sliding-window UCB bandit pick which to
run. All three pieces have exact tabular counterparts: two psi stores (reward
occupancy and novelty occupancy) instead of one; beta as a weight in the dot
product rather than a network parameter; and the bandit is the same binomial
machinery `integrated_agent` already uses for admission. Their measured
warning transfers too — a single mixed value function was "on par with a
random policy" where the split was near-optimal, so if we fold novelty into
the reward cumulant instead of keeping a second psi, we should expect the
same collapse.

**CONFIRMS — episodic-vs-lifelong novelty is a distinction we already hit.**
NGU's episodic memory resets per episode; RND novelty decays over the
lifetime. Our seed-ledger discipline (never re-spend experience) and the
per-episode intersection rule in `identify_goal` are the two timescales in
primitive form.

## S7. The model is value-blind and the planner is exact — is that the right
corner? (world_model.py, successor_features.py)

**CONFIRMS — models need not predict observations.** MuZero (Schrittwieser et
al., arXiv:1911.08265): the learned model predicts only reward, value and
policy — the quantities planning consumes — and matches AlphaZero without ever
reconstructing a frame. Our psi *is* this claim in closed form: successor
features are a value-equivalent abstraction (occupancies suffice for any `w`),
and the learned-decomposition record's criterion — keep the cut that makes
dynamics cheap to write down, not the one that reconstructs pixels — is the
same doctrine applied to perception. Worth stating in the records: we did not
skip the MONet decoder out of poverty; the value-equivalence literature says
observation reconstruction is not what a model is *for*.

**MECHANISM — plan with the model without trusting it.** I2A (Weber et al.,
arXiv:1707.06203): imagined rollouts are handed to the policy as *context*,
not executed as plans, so a wrong model degrades performance gracefully
instead of catastrophically. Our `plan_to` already refuses to route through
unknown cells, which is the crude version. The finer version for us: when the
frontend is noisy (the 0.10 noise ceiling from environment widening), a plan's
value should be discounted by the evidence count of the cells it crosses —
counts we already store in `WorldModel.counts` and currently reduce to argmax.

## S8. Goals are vectors over hand-chosen features (successor_transfer caveat:
"cumulants are hand-chosen")

**MECHANISM — relations as features, without learning the encoder.** Relation
Networks (Santoro et al., arXiv:1706.01427): relational generalisation comes
from one shared function applied to every *pair* of objects, summed. The
architecture is a network; the inductive bias is not. We now have real object
slots with symbols, so pairwise cumulants are a table, not a module:
phi(pair) = indicator over (symbol_i, symbol_j) relations — same-place,
adjacent-column, left-of. A `w` over pair-features makes "be next to the
marker" or "put A beside B" expressible in exactly the machinery
`successor_features` already has. This is the cheapest route out of the
"goals are places" ceiling and the natural next rung after `avoid`.

**PREDICTS — where our generalisation will break.** PGM (Barrett et al.,
arXiv:1807.04225): networks that interpolate competently fail sharply when a
held-out *attribute* or attribute-combination appears at test — and auxiliary
symbolic explanation targets markedly improve generalisation. Two transfers:
our held-out-goal splits are their "interpolation" regime, the easy one, so a
claim of compositionality should also show the "held-out relation" regime
(train on same-place goals, test on adjacency goals). And their
explanation-target result is structurally our decode-audit/parts-recovery
discipline — demanding the system expose *which* relation it used, not only
the answer, is measured there to be worth generalisation, not just hygiene.

## S9. One controller, many tasks (the standing integration question)

**CONTRADICTS (in emphasis) — Gato is the opposite bet, and its cost is
visible from here.** Reed et al. (arXiv:2205.06175) serialise every modality
into one token stream and train one transformer on hundreds of tasks; task
identity arrives by prompt. It is the strongest existing claim for "one
network, many tasks" — and it buys generality with exactly what AGENTS.md
forbids us: gradient updates over pooled task data, task identity as
privileged context, and no per-task verification gate. The useful import is
not the architecture but the interface lesson: *a task told by context* is
what our `w` vector and shown-goal scene already are, done compositionally
rather than by imitation. Gato is the control condition our approach should
eventually be measured against, not a design to borrow.

**MECHANISM — one compressed event space, if we ever train an encoder.** VAE
(Kingma & Welling, arXiv:1312.6114) is the standard answer if the frozen
frontend ever becomes the binding constraint: an encoder trained offline by
ELBO, then frozen and curated like any other checkpoint, keeps the agent path
gradient-free. The ELBO's reconstruction term is, however, exactly what MuZero
and our own decomposition criterion argue is the wrong target — if we train
an encoder at all, the dynamics-compression objective from
`learned_decomposition` is the principled one; a VAE is the fallback that
needs no actions.

## What this changes (second sweep)

1. **Exploration is the nearest actionable item** (S6): novelty cumulants +
   GPI + a two-store split is a complete tabular Agent57 skeleton, every part
   of which exists in the repo today.
2. **Un-gated curiosity has a predicted failure we can already reproduce**
   (S6): the `random_walk` distractor is the noisy TV. Gate novelty by the
   matched-contrast controllability we shipped in `slot_alignment`.
3. **Pairwise cumulants are the exit from "goals are places"** (S8): a
   relation table over slot pairs, then `w` over relations — no new learning
   machinery.
4. **Our next generalisation claim needs a held-out-relation regime** (S8):
   PGM says interpolation splits flatter; ours are interpolation splits.
5. **The value-equivalence doctrine retroactively names two of our choices**
   (S7): psi instead of an observation model, and dynamics-compression
   instead of reconstruction for the cut. Cite it in both records when they
   are next touched.
6. **Plans should be weighted by the evidence under them** (S7): I2A's
   graceful degradation, in tabular form, from counts we already store.

## Standing caution

Every claim above is about *other people's systems*. The value of this document
is hypotheses and instruments, not evidence. Nothing here promotes or retires
anything in `MEMORY_BANK_DESIGN.md`; a mechanism enters that log only after it
runs here, on two seeds, through its own causal gates.
