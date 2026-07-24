# Deep research prompt: preventing predictive-core drift and producing compounding learning

Conduct an exhaustive, evidence-weighted review of the relevant research
literature and turn it into a concrete experiment strategy for our tiny neural
computer. Search broadly across primary papers, recent follow-up work,
negative results, replications, benchmarks, and theoretically relevant
literature. Do not merely list papers: reconcile them with our actual results
and derive conclusions.

## The project

We are building a small real-time neural computer that learns reusable
cognitive capabilities from sensory experience. The long-term objective is:

> Each acquired capability should reduce the verified experience, reward bits,
> compute, and time required to acquire later genuinely different
> capabilities—without catastrophic forgetting or human-coded semantics.

Accuracy is primary. Latency and resource use are secondary rewards. We aim
for emergent amodal latent concepts, learned external memory, and eventually
superhuman fluid reasoning speed.

Hardware is one RTX 5090 with 32 GB VRAM. Current experiments run in roughly
4–43 seconds per seed. We use a strict ladder: under one minute, then roughly
three minutes only after a gate passes, then ten minutes only after replicated
evidence.

## Non-negotiable philosophy and information boundary

The deployed learner may observe only:

- rendered visual, auditory, or textual sensory streams;
- its recurrent/latent state and learned external memory;
- its own attempted actions;
- exact logging propensities when actions are sampled;
- observed scalar outcomes from a deterministic verifier;
- task-agnostic predictive, consistency, compression, retention, and causal
  retrieval-utility signals derived from its own experience.

The deployed learner must never receive:

- semantic concept labels, human class names, or English reasoning traces;
- task IDs, game state, generator parameters, private rules, correct actions,
  or solution traces;
- a target for an action it did not attempt, even when the complement is
  logically inferable;
- fixed symbolic planners, hand-written DSLs, semantic module routers, or
  manually assigned meanings for latent coordinates.

Verifier metadata may be used only by discarded offline probes, deterministic
reward generation, dataset balancing, and causal audits. Probe weights never
enter the agent. Probe performance is not agent capability. The external
verifier remains sovereign.

We follow the bitter lesson: useful abstractions, routing, memory formats,
plasticity, and compositional procedures must emerge because they improve
verified behavior and learning speed.

## Accounting and evidence standards

Every comparison reports separately:

- unique verifier interactions;
- unique logical lifetimes;
- unique observed reward bits;
- optimizer updates;
- examples processed, including replay;
- predictive pretraining experiences;
- GPU-seconds and wall time;
- inference latency;
- memory reads, writes, capacity, and bytes.

Replay may establish interaction efficiency only when compute is disclosed. It
is never “free sample efficiency.”

Direct deterministic held-out rerendering is the behavioral source of truth.
OPE is secondary unless direct execution is impossible or online policy shift
requires it.

All load-bearing causal interventions use valid rerendered sensory streams.
Impossible hidden-state swaps are not behavioral counterfactuals. One seed is
provisional; three seeds support an exploratory milestone; at least five seeds
and uncertainty intervals are required for a load-bearing claim. Fixed-budget
negatives are bounded claims.

## Established empirical results

### 1. Predictive latent plus REINFORCE was insufficient

A tiny visual encoder and GRU trained by paired latent-delta prediction formed
a relation-bearing recurrent state. A discarded supervised MLP probe could
decode the temporal relation at about 80%, including under true sensory
reversal. Reward-only REINFORCE behavior remained around chance.

Conclusion: actionable information existed, but the sparse policy-gradient
readout did not exploit it efficiently.

### 2. Attempted-action-only success prediction worked

A success model consumed:

`(frozen recurrent state, attempted action, logging propensity, observed scalar reward)`

BCE was applied only to the attempted action. No target was created for an
unattempted action.

On the repaired support-only temporal primitive, across seeds 211/257/313:

- final accuracy: 78.13% / 82.03% / 80.73%;
- mean reward AULC above 50%: 0.2236;
- shuffled-future representation: 55.73% final and 0.0434 AULC;
- equally optimized IPS learner: 0.1780 AULC;
- true support reversal: 78.82% relabeled accuracy and 59.11% flips.

Action-shuffled and reward-shuffled controls failed. Every reward-bit prefix
used 200 updates and 6,000 replayed examples.

### 3. Actuator/interface transfer was extremely strong

An eight-dimensional learned intention was frozen, its old two-action decoder
discarded, and a fresh four-command protocol adapter learned from attempted
commands and scalar reward.

Across three seeds:

- experienced intention reached 75% after 32 new reward bits every time;
- fresh systems required 510 / 256 / 256 bits;
- median reward-bit transfer ratio: 8×;
- mean ratio: 10.65×;
- mean true reversal accuracy: 79.69%;
- opposite-rule stale intention: 19.70%;
- swapped protocol without recalibration: 19.70%.

Conclusion: same-concept interface reuse is real. This is not cross-primitive
compounding.

### 4. Task-specific temporal intention did not accelerate spatial learning

A new balanced spatial primitive displayed two objects simultaneously and then
showed only the selected identity. The verifier asked whether it had been left
or right. A new four-command protocol prevented old action-ID reuse.

At first, temporal-intention and fresh-intention arms had nearly identical
learning curves. The temporal intention did not reliably improve AULC or
reward-bit thresholds.

Conclusion: old task-specific output weights were not the reusable asset.

### 5. The structured predictive core did transfer across primitives

We added the missing factorial controls:

- correctly paired temporal predictive core + fresh spatial intention;
- same pixels and equal predictive compute with future targets shuffled;
- fully fresh encoder/GRU/intention;
- retained temporal intention.

Across seeds 211/257/313:

| Metric | Paired predictive core | Shuffled-future core | Fully fresh core |
|---|---:|---:|---:|
| Mean final spatial accuracy | 78.56% | 58.25% | 50.00% |
| Mean spatial AULC | 0.2231 | 0.0849 | 0.0000 |
| 75% reward threshold | 256 bits on all seeds | Never by 510 | Never by 510 |

Paired-core mirror accuracy averaged 80.38%; missing feedback returned to
50.69%; opposite-rule stale state collapsed to 21.44%.

The retained temporal intention had 0.2234 mean AULC, essentially identical to
0.2231 for a fresh intention over the paired core.

Conclusion: correctly paired predictive experience created a reusable
visual/recurrent representation that accelerated a genuinely different
primitive. The reusable asset was the structured predictive core, not the old
intention head.

### 6. Naively adding a second predictive primitive did not compound

We tested a third delayed same/different identity primitive. Prior-experience
arms were:

1. temporal predictive training only;
2. temporal, then paired spatial prediction;
3. temporal, then the same spatial pixels with future targets shuffled;
4. temporal, then equal additional temporal prediction;
5. fully fresh.

Seed-211 result:

| Prior experience | Final | AULC | Reward bits to 75% |
|---|---:|---:|---:|
| Temporal + paired spatial | 80.47% | 0.2422 | 256 |
| Temporal + spatial shuffled | 81.77% | 0.2578 | 128 |
| Temporal + extra temporal | 82.03% | 0.2568 | 128 |
| Temporal only | 82.03% | 0.2406 | 128 |
| Fully fresh | 50.00% | 0.0000 | Never |

The task itself was causal:

- change only the second identity: 84.64% relabeled accuracy;
- prediction flips: 65.10%;
- remove first identity: 49.22%;
- remove second identity: 50.26%;
- opposite-rule stale state: 19.53%;
- protocol swap: 19.53%.

The candidate failed its pre-registered gate and was not replicated.

Conclusion: the predictive core transfers once, but naive sequential
spatial-only predictive training did not compound. It is consistent with
representational drift, forgetting, task-irrelevant specialization, objective
interference, or the additional primitive being redundant for the third task.

## Central research problem

What is the smallest task-agnostic mechanism that lets a predictive core
accumulate useful experience without drifting, so that each additional
primitive improves later learning speed?

We need the literature to distinguish and test these competing explanations:

1. catastrophic forgetting of temporal structure during spatial prediction;
2. loss of broadly useful features despite improved current predictive loss;
3. gradient interference between temporal and spatial predictive objectives;
4. the spatial primitive adds no information useful for same/different;
5. shuffled training acts as beneficial regularization;
6. additional temporal data simply matches the third task better;
7. fixed-capacity saturation;
8. representation collapse in only a subset of dimensions;
9. EMA-target or delta-prediction instability across sequential domains;
10. success-head variance obscuring a small real transfer effect.

## Research mandate

Search the primary literature systematically across:

- continual self-supervised representation learning;
- continual predictive learning and world models;
- catastrophic forgetting without labels;
- experience replay and generative replay;
- reservoir sampling, coresets, and online rehearsal;
- elastic weight consolidation, SI, MAS, LwF, and functional regularization;
- orthogonal-gradient and gradient-projection methods such as GEM/A-GEM;
- parameter isolation, progressive networks, adapters, and modular networks;
- dynamic sparse expansion and capacity allocation;
- learned plasticity, differentiable plasticity, fast weights, and meta-RL;
- recurrent predictive-state representations;
- successor features, GVFs, and universal value representations;
- contrastive continual learning, VICReg/Barlow-style continual learning,
  BYOL/data2vec-style continual targets;
- representation rank, covariance, feature diversity, and collapse;
- information bottleneck and minimal sufficient state;
- external episodic memory, DNC/NTM/MERLIN/NEC;
- retrieval-augmented agents and causal memory utility;
- curriculum learning, learning-progress signals, and competence progress;
- forward transfer, backward transfer, intransigence, plasticity loss, and
  stability–plasticity trade-offs;
- lottery tickets, dormant neurons, plasticity injection, and loss of
  plasticity in deep RL;
- closed-loop latent dynamics, RSSM, Dreamer, PlaNet, MuZero-style value
  prediction, PSRs, and action-conditioned multi-horizon prediction;
- small-data contextual bandits and partial-label action learning;
- neuroscience evidence on complementary learning systems, replay,
  consolidation, working memory, and synaptic stabilization, only where it
  yields operational ML hypotheses.

Prioritize work that is relevant to:

- tiny models;
- online or continual learning;
- visual/recurrent predictive representations;
- few hundred unique lifetimes;
- strict compute budgets;
- no task IDs at deployment;
- no semantic labels;
- deterministic verifiers;
- transferable learning speed rather than merely final accuracy.

Do not overweight large ImageNet/class-incremental supervised benchmarks unless
the mechanism plausibly transfers to our setting.

## Questions to answer

### A. Diagnose the third-primitive failure

1. What measurements most cheaply distinguish forgetting, interference,
   redundancy, saturation, and readout noise?
2. Should we evaluate old-task behavioral retention, old predictive loss,
   representation similarity, effective rank, CKA, subspace overlap,
   gradient cosine, Fisher overlap, or linear/MLP probes? Rank them by causal
   usefulness, not popularity.
3. How can we test whether spatial paired prediction destroyed old temporal
   features even though the same/different task still reached 80%?
4. Could the shuffled-future spatial stage have regularized the core through
   noise or anti-collapse pressure? Give decisive controls.
5. What would demonstrate that same/different simply benefits more from extra
   temporal experience than spatial experience?

### B. Pick the single next sub-minute experiment

6. Choose exactly one next experiment. The leading candidate is a
   retention-aware mixed predictive update, but challenge that assumption.
7. Compare:
   - temporal/spatial rehearsal mixtures;
   - reservoir replay;
   - functional distillation on old sensory streams;
   - EWC/SI/MAS-style weight constraints;
   - gradient projection;
   - frozen backbone plus adapters;
   - expandable slots/modules;
   - periodic rollback based on verified retention;
   - no update when future-learning utility does not improve.
8. Specify the smallest fair factorial that separates “more data,” “rehearsal,”
   “regularization,” and “new useful structure.”
9. Give exact pass/fail gates, learner-visible information, optimizer/example
   accounting, and failure interpretations.

### C. Rehearsal without task IDs or semantics

10. How can an online agent choose rehearsal items without knowing task
    boundaries or task identities?
11. Compare uniform reservoir sampling, surprise, gradient diversity,
    prediction-error reduction, coverage in latent space, and causal later-use
    advantage.
12. How do we prevent surprise-based replay from storing noisy-TV events?
13. Can verified retention and future-learning speed choose memory contents
    without leaking semantic task metadata?
14. Should rehearsal occur in pixel space, recurrent-state space, latent-delta
    space, memory rows, or compressed generative replay?
15. What is the minimum memory budget worth testing first?

### D. Plasticity and expansion

16. Does the literature suggest our 64-dimensional recurrent core is suffering
    fixed-capacity saturation or loss of plasticity after only 80 predictive
    updates?
17. Which diagnostics detect dormant features, feature-rank loss, gradient
    starvation, or target-network lock-in?
18. Compare adding capacity, resetting selected features, ReDo-style
    rejuvenation, orthogonal subspaces, adapters, and learned fast weights.
19. Which mechanism best respects our bitter-lesson requirement instead of
    hard-coding one module per human task?
20. Can a generic learned write/admission gate allocate new slots based only on
    verified future utility?

### E. External memory as the compounding mechanism

21. Design a causal retrieval-advantage experiment that cannot profit from
    storing surprise or noise.
22. The task should include an early useful cue, a long stream of surprising
    distractors, a fixed write budget, and a deterministic final verifier.
23. Compare no memory, random writes, surprise writes, reservoir writes,
    learned writes, shuffled contents, garbage contents, random retrieval, and
    equal-volume noise memory.
24. Credit a write only when an intervention enabling that memory improves
    later verified success relative to masking or shuffling it.
25. Can memory store reusable latent procedures or predictive subspaces rather
    than surface episodes? Give an operational test.
26. How should working memory in VRAM/RAM interact with compressed long-term
    disk memory while preserving the zero-semantic constraint?

### F. Compounding measurement

27. Define the six-primitive curriculum most likely to reveal a genuine
    decreasing reward-bit curve. Include at minimum:
    - reaction/inhibition;
    - temporal selection;
    - spatial selection;
    - delayed same/different;
    - a composition task;
    - a closed-loop control task.
28. Should curriculum order be fixed, randomized, or Latin-square balanced?
29. How do we analyze tasks that never reach threshold? Specify a
    right-censored survival or accelerated-failure-time analysis.
30. Define forward transfer, backward transfer, retention, intransigence,
    plasticity, and total prior-experience cost for this project.
31. What evidence separates a genuine compounding mechanism from simply
    accumulating more relevant pretraining data?
32. How many seeds and curriculum orders are needed before claiming a negative
    slope in log reward bits to threshold?

### G. Success prediction and online policy

33. We already established attempted-action-only success prediction. Do not
    recommend rerunning that foundational comparison unless a new control
    changes the scientific question.
34. Specify a four-command conservative exploration schedule based on minimum
    per-command evidence and exact propensities.
35. Distinguish calibration needed for ranking from calibration needed for
    abstention, extra thought, or retrieval.
36. Design the passive-to-active ladder:
    - passive success prediction;
    - conservative action ranking;
    - extra thought gating;
    - retrieval gating;
    - final answer influence.
37. Explain how induced-policy recalibration avoids self-reinforcing
    confidence collapse.

### H. Closed-loop escalation

38. Should we repair continual predictive learning first, or move immediately
    to micro-intercept because actions will create a more useful predictive
    objective?
39. Design the smallest micro-intercept task where actions causally alter later
    pixels.
40. Compare passive prediction, action-conditioned multi-horizon prediction,
    shuffled actions, and fixed no-action behavior.
41. What result justifies RSSM/PSR machinery? What result rules it out at this
    model scale?
42. Could action-conditioned experience itself solve the representation-drift
    problem by grounding features in controllable consequences?

## Required deliverables

1. A concise executive verdict naming the single next sub-minute experiment.
2. A causal diagnosis tree for the failed third-primitive compounding attempt.
3. An evidence table covering at least 40 directly relevant primary papers:
   - paper and year;
   - setting and model scale;
   - learning signals;
   - task-boundary/task-ID assumptions;
   - memory/compute cost;
   - reported forward transfer and retention;
   - known negative results or limitations;
   - applicability to our exact regime.
4. A ranked list of at least 25 candidate experiments by expected information
   gain per GPU-minute.
5. For the top five experiments:
   - exact learner-visible inputs;
   - architecture changes;
   - losses or rewards;
   - pseudocode;
   - matched controls;
   - reward-bit and compute accounting;
   - causal audits;
   - pre-registered pass bars;
   - what every possible outcome would mean.
6. A no-task-ID rehearsal and retention mechanism.
7. A causal retrieval-advantage memory experiment.
8. A six-primitive, multi-order compounding curriculum and statistical
   analysis plan.
9. A four-command online exploration/calibration schedule.
10. A micro-intercept design and RSSM/PSR escalation gate.
11. A “do not adopt yet” section.
12. A bibliography grouped by evidential role, emphasizing primary sources.

## Literature-review quality requirements

- Search for disconfirming papers and negative transfer results, not only
  methods claiming success.
- Distinguish class-incremental supervised learning from task-free continual
  self-supervised learning.
- State when a method depends on task boundaries, replay labels, semantic
  classes, or large pretrained backbones that we do not have.
- Separate final retention from forward sample-efficiency transfer.
- Separate representation diagnostics from verified behavior.
- Separate theoretical guarantees from empirical heuristics.
- Identify mechanisms that fail under distribution shift or small memory.
- Prefer primary papers and official implementations over secondary summaries.
- Quote no result without its experimental regime and scale.
- Mark every recommendation as established, inferred, or speculative.
- Do not recommend a large architecture before a cheaper measurement
  distinguishes the competing explanations.
- Do not repeat experiments we have already completed:
  attempted-action success versus REINFORCE, fresh-core localization,
  shuffled-future core transfer, actuator remapping, temporal intention
  warm-start, or the third same/different factorial.

## Desired closing decision

End with one of these explicit recommendations, supported by the literature:

1. **Repair continual predictive learning first**, naming the exact
   rehearsal/regularization mechanism and sub-minute experiment;
2. **Make external causal-retrieval memory the next compounding mechanism**,
   naming the exact task and gate;
3. **Move to closed-loop micro-intercept now**, explaining why controllable
   consequences are more likely to yield reusable state than more passive
   prediction;
4. or a better alternative that strictly respects all constraints and is
   cheaper or more decisive.

The answer must explain why the chosen path has higher expected information
gain per GPU-minute than the other three.
