# Literature map

What this project measured, and what the published work calls it.

Written 2026-08-11, after F144. The purpose is narrow: several of our
findings were arrived at empirically and turn out to be known problems
with named solutions we have not tried. Where that is true, the named
solution is a cheap next probe. Where our result DISAGREES with the
literature, that is worth more than agreement and is flagged.

Nothing here is evidence. A citation is a hypothesis with a track
record, and it enters the ledger only after it measures on our probes.

---

## 1. The deadlock (F106) is POSTERIOR COLLAPSE

**What we measured.** The outcome model predicts a world and its
inverted twin identically (0.9998 label agreement). It has found the
twin-average, which is adequate for both, so the reader receives no
gradient, so there is nothing for the model to read. Neither half can
move alone.

**What it is called.** Posterior collapse: the latent's posterior
q(z|x) matches the prior, the latent carries no information about the
input, and the generative network ignores it. The standard diagnosis
is that a sufficiently expressive decoder models the observation
without the latent — which is exactly a plant that predicts the
twin-average without the entry.

Our conditional setting matches the sharper version: in conditional
VAEs the model tends to ignore the latent when the conditioning signal
is strong and the decoder is expressive enough to produce a plausible
output from the condition alone.

**What we already do, and its published name.** The ignorance
objective — penalising accuracy WITHOUT the entry — is decoder
weakening: deliberately handicapping the entry-free path so the latent
becomes the only route to a good score. The published variants weaken
the decoder by masking its inputs; ours weakens it by penalising its
entry-free confidence directly. Same mechanism, different lever.

**What we have NOT tried, in rough order of cheapness.**

- **Free bits / KL floor.** Guarantee a minimum information rate per
  latent dimension rather than penalising the entry-free path. Our
  ignorance term is a soft global pressure; free bits is a hard
  per-dimension one, and F120 measured that our soft version is
  toothless when the model is bad. A hard floor does not have that
  failure mode.
- **Latent skip connections.** Feed the entry in at every layer, not
  only at the input. NOTE: this points the OPPOSITE way from F135,
  where feeding the entry in at every STEP was the defect. The two are
  not the same axis (layers within one application vs. applications
  within a program) but the tension is real and worth resolving by
  measurement rather than assumption.
- **Auxiliary reconstruction decoders.** Force the latent to
  reconstruct the observations it was read from, in addition to
  serving the task. This is close to our contrastive term but supplies
  a denser signal.

Sources:
[posterior-collapse technique list](https://github.com/sajadn/posterior-collapse-list),
[Don't Blame the ELBO (NeurIPS 2019)](http://papers.neurips.cc/paper/9138-dont-blame-the-elbo-a-linear-vae-perspective-on-posterior-collapse.pdf),
[Mitigating Posterior Collapse in Strongly Conditioned VAEs](https://openreview.net/forum?id=rJlHea4Kvr),
[splitting decoders in variational recurrent autoencoders](https://www.sciencedirect.com/science/article/abs/pii/S0925231223012262)

---

## 2. Our two-phase failure (F136) is the LAGGING INFERENCE NETWORK
   problem — and we chose the wrong extreme

**What we measured.** Training the plant on oracle entries, freezing
it, then training the reader through it gives 0.4973 — worse than
joint training's 0.5283. The frozen plant demands entries in a narrow
region and task loss cannot search for it.

**What it is called.** He et al. (ICLR 2019) diagnose posterior
collapse as an optimisation-schedule problem: the true posterior is a
moving target and the inference network cannot keep up, so it gives up
and collapses to the prior. Their fix is to train the encoder
AGGRESSIVELY — many encoder steps per decoder step — until the mutual
information stops improving.

**Why this matters to us specifically.** Their solution and our F136
sit on the same axis at opposite ends. They interleave, updating the
encoder more often while the decoder still moves. We froze the decoder
completely, which is the limit of that idea, and it failed. The
literature's reading is that the encoder must be given time to catch
up with a target that is still moving — not chased against a target
that has stopped.

**Concrete probe this suggests.** Replace `--two-phase` with a
step-ratio: N reader updates per plant update, N in {2, 5, 10}, both
still learning. This is a schedule change, not an architecture change,
and it is the cheapest untried item on this page.

Sources:
[Lagging Inference Networks and Posterior Collapse (ICLR 2019)](https://arxiv.org/abs/1901.05534),
[reference implementation](https://github.com/jxhe/vae-lagging-encoder)

---

## 3. Bind-once (F135, F143) is HYPERNETWORK-style conditioning

**What we measured.** Decoding the entry ONCE into explicit per-piece
parameters, instead of attending over it at every step, takes
conditioned execution at depth from 0.5548 to 0.9983, and transferred
to the games first try for +0.0995 -> +0.1229.

**What it is called.** A hypernetwork: a network that takes a task
embedding and emits the weights (or parameters) of the network that
does the work, as against FiLM or attention, which modulate activations
during the forward pass. Reviews report hypernetwork conditioning
beating FiLM and concatenation in several domains, particularly with
recurrent architectures — which is our setting exactly, since the
failure appeared only once the computation became iterative.

**Where we AGREE and where we ADD.** The literature's comparisons are
mostly about final accuracy. Our F131/F132/F134 sequence isolates the
mechanism: re-conditioning per step re-derives the same parameters on
every application and the errors compound with DEPTH, which is why
depth-1 was fine and depth-4 was dead. That is a sharper claim than
"hypernetworks tend to work better", and it predicts where FiLM-style
conditioning should be safe (shallow) and where it should not (deep or
iterated).

**Where we DISAGREE with the obvious extension.** F140: giving the
binder capacity (linear -> MLP) destroyed the result, 0.9983 -> 0.6196.
Much hypernetwork work is concerned with generating LARGE parameter
sets through expressive mappings. Our measurement says the mapping
from embedding to parameters wants to be as simple as possible — the
fourth such result in this project (F77, F79, F89, F140).

Sources:
[A brief review of hypernetworks in deep learning](https://link.springer.com/article/10.1007/s10462-024-10862-8),
[arXiv version](https://arxiv.org/pdf/2306.06955),
[hypernetwork-based conditioning overview](https://www.emergentmind.com/topics/hypernetwork-based-conditioning),
[parameter complexity of hypernetworks vs embeddings](https://www.researchgate.net/publication/339471407_Comparing_the_Parameter_Complexity_of_Hypernetworks_and_the_Embedding-Based_Alternative)

---

## 4. The reader (F138/F142/F144) is CONTEXT-BASED META-RL task inference

**What we measured.** A reader that emits an entry from a handful of
observations, with zero gradient steps at acquisition, driving 0.9723
when distilled and 0.7795 when trained contrastively.

**What it is called.** Context-based meta-RL. PEARL encodes task
context as probabilistic latent variables by variational inference and
conditions the policy on samples from that posterior; VariBAD casts it
as a Bayes-Adaptive MDP where the context variable is a belief state
inferred from past transitions. Our reader is the same object; our
bank entry is their z.

**Our contrastive term has a direct analogue.** CORRO and related
offline meta-RL work robustify the task encoder with a contrastive
objective, and adding a contrastive objective to VariBAD is reported to
improve generalisation. F144's confirmed result — contrastive
auxiliary beats task loss alone — is consistent with that literature
rather than novel.

**What IS ours, and is worth checking against the literature.** The
batch-size effect (F144: batch 8 -> 0.6237, batch 32 -> 0.7795) frames
the contrastive task's difficulty as the knob, which is F78's
diversity law applied to the reader's objective. Contrastive learning
generally is known to benefit from more negatives; whether anyone has
reported it as the binding constraint on TASK-inference quality
specifically is worth a literature check before we claim it.

Sources:
[PEARL: Efficient Off-Policy Meta-RL via Probabilistic Context Variables](https://arxiv.org/pdf/1903.08254),
[Robust Task Representations for Offline Meta-RL (CORRO)](https://z0ngqing.github.io/paper/corro-haoqi.pdf),
[Entropy Regularized Task Representation Learning for Offline Meta-RL](https://arxiv.org/pdf/2412.14834),
[Meta-RL based on Self-Supervised Task Representation](https://ojs.aaai.org/index.php/AAAI/article/view/26210/25982)

---

## 5. The remaining reader gap: DISCRETE bottlenecks (untried, best fit)

**The gap.** 0.7795 contrastive against 0.9723 distilled. The entry
must be discriminative AND bindable, and our continuous entry achieves
the first more easily than the second.

**What the literature offers.** VQ-VAE replaces the continuous latent
with a lookup into a learned codebook. The reported reason it does not
suffer posterior collapse is structural rather than a tuned penalty:
the hard discrete assignment means the decoder cannot bypass the
bottleneck, because no averaging or relaxation is available. That is
precisely the failure mode we have been fighting since F106 — our
plant's escape route has always been to average.

**Why this fits our architecture unusually well**, beyond fixing the
symptom:

- a BANK is naturally a finite set of entries; a codebook IS a bank,
  and this would make the data structure and the mechanism the same
  object rather than two things we bolt together;
- bindability becomes trivial. A simple binder must map each of K
  codes to a parameter set, which is a lookup rather than a regression
  — and F140 says the binder must stay simple, so making its job
  easier is exactly the right direction;
- discreteness gives exact retention for free. F98 already found that
  approximate, lossy addressing breaks the exception store while exact
  keys match a real dictionary.

**The risk to measure, not assume.** A codebook caps the number of
distinguishable worlds at K, which collides with the diversity law
that has driven every reading result (F78, F144). The probe must sweep
K against world count, not fix it.

Sources:
[Neural Discrete Representation Learning (VQ-VAE)](https://vitalab.github.io/article/2019/09/26/VQ-VAE.html),
[Discretized Bottleneck in VAE: Posterior-Collapse-Free Seq2Seq](https://www.researchgate.net/publication/340859542_Discretized_Bottleneck_in_VAE_Posterior-Collapse-Free_Sequence-to-Sequence_Learning),
[Improve VAE for Text Generation with Discrete Latent Bottleneck](https://arxiv.org/pdf/2004.10603),
[Discrete Key-Value Bottleneck for continual learning](https://arxiv.org/pdf/2412.08528)

---

## 6. The games residual (F110) is COMPOUNDING MODEL ERROR

**What we measured.** With the value model finished (+0.1229 against a
+0.1234 oracle-value target), 31.9% of floor-to-full-oracle remains,
and it is reachable only through the search and the transition model.

**What it is called.** Compounding model error: small per-step
prediction errors accumulate when trajectories are built by
bootstrapping successive model predictions, so modelled rollouts drift
from the true trajectory even from the same start and actions.

**What the literature does about it.**

- **Short rollouts.** The standard practice, often taken to one step.
  We are at depth 4 with `--freeze-objects` already compensating for
  the least predictable slots — so our depth-6 arm is running INTO
  this known headwind and should be read with that in mind.
- **Ensembles with uncertainty.** Heteroskedastic ensemble dynamics
  models, so the search can distrust states the model is unsure about.
  Untried here and directly applicable: our object slots are
  stochastic by construction (F109) and the search currently treats
  every prediction as equally reliable.
- **Multi-step models.** Predict k steps ahead directly instead of
  iterating a one-step model, with bounds on value error in terms of
  model error. This is the same shape as F135's bind-once — do the
  work once rather than compounding — which is mildly encouraging.
- **Value-equivalent / latent models** (MuZero, Dreamer): do not
  predict observations at all, only quantities that affect the value.
  Note the caveat in the recent literature: the MuZero loss has
  constructed counterexamples showing failure in stochastic
  environments and exponential sample complexity in some deterministic
  ones, so this is not a free win.

**Predicted from our own results.** F109 measured the avatar slots
predicted at 1.0000 and the object slots at 0.67-0.77, and freezing
the objects was worth 3.4 points. That is an uncertainty-weighting
result obtained by hand. An ensemble would generalise it, and the
prediction is that it helps the object slots and does nothing for the
avatar — a falsifiable claim we can check per-slot.

Sources:
[Combating the Compounding-Error Problem with a Multi-step Model](https://arxiv.org/pdf/1905.13320),
[Learning to Combat Compounding-Error in Model-Based RL](https://arxiv.org/pdf/1912.11206),
[Diminishing Return of Value Expansion Methods](https://arxiv.org/html/2412.20537),
[A Note on Loss Functions and Error Compounding in Model-based RL](https://arxiv.org/pdf/2404.09946),
[Model-based RL with Multi-step Plan Value Estimation](https://arxiv.org/pdf/2209.05530)

---

## Ranked next probes, from this map

1. **Discrete codebook entry** (§5) — best fit to the measured gap,
   and it makes the bank and the mechanism the same object. Sweep K
   against world count.
2. **Reader step-ratio instead of freezing** (§2) — cheapest item
   here, a schedule change, and it corrects an error we can now name:
   F136 took the aggressive-encoder idea to an extreme the literature
   does not endorse.
3. **Dynamics ensemble with uncertainty-weighted search** (§6) —
   generalises F109's hand-made freeze, with a per-slot falsifiable
   prediction.
4. **Free bits / hard information floor** (§1) — targets exactly
   F120's finding that our soft ignorance term is toothless when the
   model is bad.

---

# Addendum, 2026-08-11: the CURRENT bottleneck, named precisely

The first map treated "the reader's training signal" as one problem.
F138 split it, and the split has a name in the literature that matches
our measurements exactly.

## 7. Our gap is the AMORTIZATION GAP, not the approximation gap

**What we measured.** The reader's architecture CAN produce the entry
the bound plant needs — distilled onto a consistent target it reaches
0.9723 per-bit on held-out worlds (F138). Trained by any
non-privileged objective it reaches at most 0.7795 (F144). Same
network, same inputs, same frozen plant.

**What it is called.** Cremer, Li and Duvenaud (ICML 2018) decompose
the inference gap into two parts:

- the **approximation gap** — the variational family cannot express
  the right posterior at all;
- the **amortization gap** — the family can, but the amortised
  recognition network fails to produce the right parameters for each
  datapoint, because one network must serve every input in one pass.

Their headline finding is that divergence from the true posterior is
usually caused by imperfect recognition networks rather than by the
limited complexity of the approximating family — and that the
amortization gap is large on complex datasets even with a powerful
inference network.

**This is our situation stated exactly.** F138 IS an approximation-gap
measurement: it shows the family suffices. Everything since is
amortization. That reframes the whole reader effort — every scheme we
have tried (task loss, two-phase, contrastive phase, contrastive
auxiliary, batch size) has attacked the amortization gap by improving
the OBJECTIVE, and the literature's most effective answer is not a
better objective at all.

## 8. The literature's answer: SEMI-AMORTIZATION

**The method.** Kim et al. (ICML 2018): use the amortised encoder to
produce an INITIALISATION, then run a small number of gradient steps
on the variational parameters for that specific datapoint. Reported
result: 10 refinement steps from a learned initialisation beat 80
steps of pure per-instance inference — the initialisation is doing
most of the work, and the refinement closes the residual.

**Why this fits us unusually well, and the objection to answer first.**
Our standing claim is "zero gradient steps at acquisition", which
semi-amortization appears to violate. It does not, and the distinction
matters: **the gradient steps refine the ENTRY, not the plant.** The
entry is data in an external store; the plant's weights never move. A
bank whose entries are polished on arrival is still a bank, and this
is much closer to the architecture's spirit than it first sounds —
F44 already established that world identity is only knowable from
consequences, and refining an entry against observed consequences is
that principle applied at acquisition rather than abandoned.

It also makes a sharp prediction we can falsify: if the gap is
amortization, a handful of refinement steps from the reader's own
output should recover most of the 0.7795 -> 0.9723 distance. If it
does not, the gap is something neither term covers and both F138 and
this analysis are wrong.

**Cost note, given the F144 confound.** Refinement is per-world and
cheap; unlike the contrastive batch, its cost does not scale with the
number of worlds per update. Any comparison must still be run at
matched compute.

## 9. Why contrastive stalls: ALIGNMENT vs UNIFORMITY

**What we measured.** Contrastive objectives beat task loss but
plateau well below what distillation shows possible, and F140 showed
the shortfall is not decoder expressiveness.

**What it is called.** Wang and Isola (ICML 2020) decompose contrastive
learning into two asymptotic properties: **alignment** (positive pairs
close together) and **uniformity** (features spread over the
hypersphere). Both are optimised by InfoNCE, and the balance between
them predicts downstream performance — with the specific observation
that linear downstream decoders benefit from TIGHT CLUSTERS, i.e. from
alignment, while pushing uniformity harder can cost within-class
alignment.

**Why that is our exact trade.** Our two requirements have been
"discriminative" and "bindable", and they map onto uniformity and
alignment respectively: our binder is LINEAR by necessity (F140), so
it wants tight clusters, while more negatives buy separation. F144's
batch-128 collapse is what over-weighted uniformity looks like, and
F142's weight-3.0 collapse is the same thing from the weight axis.
Both were read as "over-shoot" without a mechanism; this supplies one.

**What it suggests.** Wang and Isola report that optimising alignment
and uniformity as SEPARATE terms performs comparably or better than
InfoNCE itself. That gives an explicit knob for the balance our probes
keep rediscovering as an interior optimum, instead of controlling it
indirectly through batch size and loss weight — and it predicts that
at fixed compute, raising alignment should help more than raising
uniformity, because the binder is linear.

## 10. For the games' new bottleneck: TASK-SUFFICIENT state

F145 relocated the games residual from the search to the state
abstraction. The relevant literature is bisimulation-based
representation learning (Zhang et al., *Learning Invariant
Representations for RL without Reconstruction*): learn an encoder
whose latent distances equal behavioural distances, keeping exactly
what affects outcomes and discarding the rest, without reconstructing
observations.

Our `--objects N` widening is the crude version of this — more of the
state kept, chosen by hand. The bisimulation framing suggests the
principled version: keep what changes VALUE. Worth noting the caveat
in the recent literature that bisimulation methods have known pitfalls
and failure modes, so this is a direction rather than a recipe.

Sources:
[Inference Suboptimality in VAEs (ICML 2018)](https://arxiv.org/abs/1801.03558),
[Semi-Amortized VAEs (ICML 2018)](https://arxiv.org/pdf/1802.02550),
[Reducing the Amortization Gap: Bayesian Random Function](https://arxiv.org/abs/2102.03151),
[Amortized Inference Regularization](https://arxiv.org/pdf/1805.08913),
[Alignment and Uniformity on the Hypersphere (ICML 2020)](https://arxiv.org/abs/2005.10242),
[Learning Invariant Representations for RL without Reconstruction](https://openreview.net/forum?id=-2FCwDKRREu),
[Pitfalls of Bisimulation-based representations (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/5a1667459d0cdeb2fe6b2f0dffc5cb9d-Paper-Conference.pdf)

---

## Revised ranking after the addendum

1. **Entry refinement at acquisition** (§8) — semi-amortization.
   Directly targets the gap we have actually measured, makes a
   falsifiable prediction, and refines DATA rather than weights so the
   architecture survives intact. Displaces the codebook from first
   place because it attacks the measured quantity rather than a
   plausible cause.
2. **Explicit alignment/uniformity terms** (§9) — replaces two
   indirect knobs (batch size, loss weight) with the axis they were
   both proxying, and predicts which direction helps given a linear
   binder.
3. **Discrete codebook** (§5) — still strong, still running, and
   §9 gives it a second rationale: a codebook is maximal alignment
   (every world in a cluster of radius zero).
4. **Reader step-ratio** (§2) — cheap, unresolved.
5. **Dynamics ensemble** (§6) — DEPRIORITISED by F145: search budget
   buys nothing, so uncertainty-weighted search has little to act on.
   The state abstraction (§10) replaces it.

---

# Addendum 2, 2026-08-11: deeper on the amortization gap

Four more bodies of work bear directly on it. The first is the closest
match to our architecture found so far.

## 11. Our reader IS a Neural Process, and NP UNDERFITTING is our
   symptom exactly

**The correspondence.** A Neural Process maps a context set of
input-output pairs to a latent, then predicts at new inputs. Our
reader maps a set of (piece, x, piece(x)) rows to an entry, and the
plant predicts at new queries. Same object, same permutation-invariant
context set, same amortised single-pass inference.

**The known failure.** NPs UNDERFIT the context set — they give
inaccurate predictions even at the inputs they conditioned on. The
diagnosed cause is mean-aggregation in the encoder: averaging over
context points weights them equally and prevents the decoder from
identifying which ones matter.

**This is a specific, checkable prediction about our reader.** Our
`bind()` takes `entry.mean(dim=0)` over the bank tokens, and the games
signed pathway takes `entry.mean(dim=0)` as well. Mean-pooling is
exactly the operation the NP literature indicts — and F98 already
measured, independently and in a different part of this project, that
**mean-pooled keys are net-harmful while concatenated keys match an
exact dictionary.** We found the same defect once and did not
generalise it.

**Immediate cheap probe, sharper than anything else on this page:**
measure whether the reader can predict the rows it CONDITIONED ON. If
it underfits its own context, the diagnosis is confirmed without any
new mechanism, and the fix is attention over context rows (ANP) or
dropping the mean-pool for concatenation, both of which we have the
parts for.

Sources:
[Attentive Neural Processes (ICLR 2019)](https://arxiv.org/abs/1901.05761),
[OpenReview version](https://openreview.net/pdf?id=SkE6PjC9KX)

## 12. Task vectors: the field independently found our two-way split

In-context learning research extracts a **task vector** — a single
activation-space vector distilled from a few demonstrations, which can
be injected elsewhere to induce the same function, even in zero-shot
contexts with no task information present. That is our bank entry
described in someone else's vocabulary, including the injection
property (our stranger-entry control is the same manipulation used
adversarially).

The convergence worth noting: sparse-autoencoder work on in-context
learning reports **two classes of feature — one that RECOGNISES the
task from the examples, and another that ENCODES it for later
EXECUTION.** We arrived at the same split empirically and called it
"discriminative" versus "bindable" (F139-F142). Two independent routes
to the same decomposition is mild evidence the distinction is real
rather than an artefact of our probes.

Sources:
[Task vectors overview](https://www.emergentmind.com/topics/task-vectors-tvs),
[Task Vector Geometry Underlies Dual Modes of Task Inference](https://arxiv.org/pdf/2605.03780),
[Extracting SAE task features for in-context learning](https://www.alignmentforum.org/posts/5FGXmJ3wqgGRcbyH7/extracting-sae-task-features-for-in-context-learning),
[Distributional Alignment as a Criterion for Designing Task Vectors](https://arxiv.org/pdf/2605.20730)

## 13. If refinement works, TRAIN FOR IT (LEO / learned initialisations)

Our `--refine` bolts refinement onto a reader trained without it, so
the reader has no incentive to produce a good STARTING POINT — only a
good final answer in one pass. The meta-learning literature closes
that loop:

- **LEO** (Meta-Learning with Latent Embedding Optimization) performs
  the MAML inner loop in a LATENT space produced by an encoder rather
  than in weight space — structurally identical to refining our entry,
  but with the encoder trained through the refinement.
- **Learned initialisations for coordinate networks** (Tancik et al.,
  CVPR 2021) report the practical shape: meta-learn the starting
  point, and at test time optimise for many more steps than the inner
  loop used, with benefits continuing past the training horizon.

**The staged plan this implies**, and the order matters: first measure
whether refinement helps at all with the reader as-is (running now).
Only if it does is it worth paying for backpropagation through the
refinement loop, which is expensive and easy to get subtly wrong.

Sources:
[Meta-Learning with Latent Embedding Optimization](https://meta-learn.github.io/2018/papers/metalearn2018_paper34.pdf),
[Learned Initializations for Coordinate-Based Representations (CVPR 2021)](https://arxiv.org/pdf/2012.02189),
[MAML](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf)

## 14. Test-time training: the same idea, and its known hazard

Refining at acquisition is test-time training under another name —
adapting using self-supervised signal available at inference, with no
labels. The literature is active and reports robust gains across
vision, speech and time-series.

**The hazard is the part we should take seriously**: TTA methods are
known to COLLAPSE when the adaptation signal is degenerate, which is
why recent work is explicitly concerned with collapse-free entropy
optimisation. Our refinement objective is the plant's own likelihood
on the reader's rows, so a degenerate entry that makes the plant
confidently wrong on everything is a reachable optimum.

**Guard to add before trusting any refinement result:** check that the
refined entry still beats a STRANGER's refined entry. If refinement
merely finds a generically-agreeable entry, both will improve
together, and the gap — not the level — is the evidence. Our probes
already compute the stranger arm, so this costs nothing.

Sources:
[Test-Time Adaptation survey topic](https://www.emergentmind.com/topics/test-time-adaptation-tta),
[Test-Time Training: Adaptive Inference](https://www.emergentmind.com/topics/test-time-training),
[ZeroSiam: entropy optimisation without collapse](https://arxiv.org/pdf/2509.23183),
[Open-World Test-Time Training](https://arxiv.org/pdf/2409.09591)

---

**Baseline correction (F147).** The gap these mechanisms must beat is
now 0.8520, not 0.7795: training longer at the confirmed batch closed
a third of the remaining distance to 0.9723 on its own. Part of what
this map treated as a mechanism problem was undertraining. The
amortization diagnosis survives — same reader, same inputs, still
short of the distilled ceiling — but every entry below should be read
against the corrected baseline.

## Ranking after addendum 2

1. **Context-reconstruction check** (§11) — can the reader predict the
   rows it conditioned on? One evaluation, no training, and it either
   confirms NP underfitting as the mechanism or eliminates it.
2. **Entry refinement** (§8, running) — with the stranger-refined
   control from §14 as the acceptance test, not the raw level.
3. **Drop the mean-pool** (§11) — F98 already measured mean-pooling
   harmful elsewhere in this project; `bind()` and the games signed
   pathway both still use it.
4. **Explicit alignment/uniformity terms** (§9).
5. **Discrete codebook** (§5, re-running after the F146 collapse fix).

---

# Addendum 3, 2026-08-11: after three mechanism nulls

Semi-amortization (§8), the discrete codebook (§5) and the widened
state (§10) all measured null. Only longer training moved anything.
That changes what is worth reading.

## 15. The games' planning objective was simply WRONG (F151)

Not a bottleneck needing literature so much as a bug the literature
names the correct form of. TD-MPC and infinite-horizon MPPI state the
standard decision-time planning objective explicitly: **estimate
short-term rewards from model rollouts, and long-term return with a
TERMINAL value function** — sum gamma^d r(s_d,a_d) over the horizon,
then add gamma^H V(s_H). The value enters ONCE, at the end.

Ours summed an H-step-return prediction at every depth, giving r_3 a
weight of 2.916 where 0.729 is correct (F151). The literature's form
is exactly the fix, and it names the missing component precisely: an
immediate-reward head, which we do not have because F111 replaced the
3-class outcome head with a return head and nothing kept the one-step
quantity.

This also reframes F110's "search and dynamics residual" as possibly
neither: a residual attributed to search QUALITY may be search
ARITHMETIC. The `--score terminal` arms running now are the cheap test
and the reward-plus-bootstrap head is the real fix.

Sources:
[TD-MPC: Temporal Difference Learning for Model Predictive Control](https://proceedings.mlr.press/v162/hansen22a/hansen22a.pdf),
[MPC and RL: A Unified Framework Based on Dynamic Programming](https://arxiv.org/html/2406.00592v3),
[MPC with Differentiable World Models for Offline RL](https://arxiv.org/html/2603.22430v1)

## 16. The reader's "only training helps" pattern may be GROKKING

**What we measured.** 40k updates 0.7795, 100k updates 0.8704, with
exact-match rising 0.2498 -> 0.4228. Three mechanism interventions did
nothing; time did a lot.

**What the literature describes.** Grokking: a model first memorises,
then sits on a plateau where aggregate loss barely moves, then
transitions to generalisation — described as being pushed away from
memorisation toward simpler, more general representations. Originally
found on ALGORITHMIC tasks, which is what our probes are.

**Why this is a hypothesis and not yet a finding.** We have three
points on a curve (40k, 100k, and 200k running) measured only at the
END of training. Grokking is a claim about the SHAPE of a curve — a
plateau followed by a transition — and we have never plotted held-out
performance against training step even once. Two very different
stories fit our three points equally well:
  * smooth logarithmic improvement, in which case the ceiling is
    approached slowly and the "amortization gap" is just budget;
  * a phase transition, in which case there is a specific step count
    where reading appears and everything before it is wasted.

They imply opposite things about what to do next, and distinguishing
them costs one instrumented run rather than another mechanism.

**Concrete gap in our instrumentation, which this exposes.** Every
probe in this project reports only final numbers. We have run
hundreds of arms and never once looked at a learning curve. That is
why "it needed more training" was invisible until a run happened to be
longer, and why the shape question is currently unanswerable from
anything on disk.

Sources:
[Towards Understanding Grokking: An Effective Theory of Representation Learning (NeurIPS 2022)](https://papers.neurips.cc/paper_files/paper/2022/file/dfc310e81992d2e4cedc09ac47eff13e-Paper-Conference.pdf),
[Two Speeds of Learning: Representation-Readout Decomposition of Grokking](https://arxiv.org/html/2605.27078),
[A Two-Phase Perspective on Deep Learning Dynamics](https://arxiv.org/pdf/2504.12700),
[Grokking overview](https://www.emergentmind.com/topics/grokking)

---

## Ranking after addendum 3

1. **Instrument the learning curve** (§16) — periodic held-out
   evaluation during training. Answers the grokking-vs-slow-convergence
   question, makes every future run more informative, and is the
   cheapest item ever added to this page. It should have existed from
   the start.
2. **Reward head + terminal bootstrap** (§15) — the literature's
   standard planning objective, replacing a measured 4x over-counting.
3. **Context-reconstruction check** (§11) — implemented, awaiting a
   run that includes it.
4. Retired for now: semi-amortization, codebook, widened state — all
   measured null (F148, F149). Alignment/uniformity terms (§9) remain
   untested but drop below the two items above.
