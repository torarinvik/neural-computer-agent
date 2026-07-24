# Deep-research prompt: the smallest honest closed-loop test of compounding neural cognition

You are the research scientist advising an active experimental project. Search
the primary literature deeply and produce a decision-oriented report, not a
generic survey. We need the most relevant evidence for the *next tiny
experiments* in a zero-semantic-label neural computer whose central objective is:

> Maximize verified improvement in reusable capability per unit of experience,
> so that each learned cognitive primitive makes later novel primitives faster
> to acquire.

## Non-negotiable scientific rules

The deployed learner may receive only:

- rendered visual, auditory, or text-like sensory streams;
- its own opaque actions;
- its own learned latent working and long-term memory;
- exact logging propensities when actions are sampled;
- scalar outcomes from deterministic external verifiers;
- elapsed time and generic resource costs.

The deployed learner may **not** receive:

- game state, coordinates, velocity, object IDs, task IDs, rule IDs, correct
  actions, counterfactual outcomes for unattempted actions, semantic concepts,
  English chain of thought, symbolic DSLs, hand-written planners, or privileged
  verifier metadata;
- hand-labeled auxiliary targets;
- labels inferred for actions it did not attempt.

Discarded offline probes and audits may use private generator metadata to
measure where information exists. Their weights must never enter the agent.
Direct deterministic rerendered execution is the source of truth. A causal
counterfactual must be generated as a valid new sensory trajectory; swapping
history-bearing hidden states is not a valid behavioral counterfactual.

Accuracy is primary. Latency is a small secondary reward only after correctness.
Always account separately for:

- unique verifier interactions / reward bits;
- unique logical lifetimes;
- optimizer updates;
- examples processed including replay;
- GPU-seconds and wall time;
- inference latency;
- memory reads and writes.

Our experimental ladder is strict: first a sub-one-minute run; promote to about
three minutes only after a pre-registered mechanistic or behavioral gate passes;
promote to ten minutes only after replicated causal evidence.

## Architecture and long-term goal

The agent is a tiny recurrent neural computer with:

- modality-specific streaming encoders;
- a learned modality-independent latent or “amodal concept” space;
- recurrent working memory;
- external long-term memory with learned read, write, consolidation, and
  compression;
- learned latent action intentions translated by replaceable actuator adapters;
- no native dependence on words such as `LEFT`, `UP`, or `BIRD`.

Skills and internal representations must emerge from experience. We ultimately
want perception, attention, relation binding, prediction, action selection,
memory utility, confidence, and learning strategy to become reusable primitives.

## What has already worked

Do not recommend repeating these merely to rediscover them:

1. An attempted-action-only binary success model over frozen recurrent latents
   is our strongest zero-label readout. Across three seeds on a repaired temporal
   primitive it achieved 78.13%, 82.03%, and 80.73% final accuracy. Shuffled
   future controls averaged about 55.7%; matched IPS policy learning was worse.
   True pixel-space reversal passed.

2. A learned eight-dimensional latent intention transferred to a new opaque
   four-command actuator. The frozen intention plus a fresh actuator adapter
   reached 75% with 32 new reward bits on all three seeds. A fully fresh system
   needed 256–510 bits. Median transfer advantage was 8x. Protocol swap and
   stale/opposite controls collapsed to about 20%.

3. A paired predictive temporal core transferred to a new spatial selection
   primitive. It reached 75% with 256 reward bits on all three seeds; shuffled
   and fully fresh controls did not reach threshold by 510. Mirror, missing
   feedback, and stale-state audits behaved causally.

4. Sequential predictive training retained the old temporal behavior within our
   pre-registered three-point gate. The paired spatial addition reduced temporal
   accuracy by about 2.6 points, so retention is provisionally acceptable.

5. Representation probing and causal auditing have repeatedly found errors that
   endpoint accuracy hid. Important permanent lessons include:
   - probe both sides of a boundary before repairing it;
   - fit training data before interpreting held-out failure;
   - fixed-budget negatives mean only “no ignition within this budget”;
   - unique logical diversity mattered more than render augmentation at scale,
     although augmentation accelerated early ignition;
   - valid rerendered reversal is mandatory for recurrent models.

## What has not worked or remains unproven

1. REINFORCE repeatedly converted reward bits into behavior less efficiently
   than attempted-action success prediction.

2. Five reader/router repairs failed because the relevant rule had never been
   stored. Later work localized separate sensory, event-retention, binding, and
   write-path failures.

3. Adding another predictive primitive did **not** produce compounding gains on
   a delayed same/different task. Several pretrained variants reached 75% in
   128 bits, but none beat the simplest temporal-only predecessor robustly.
   This is transfer, not compounding.

4. A supervised diagnostic event-snapshot pairwise binder reached about 95%
   held-out and passed true counterfactual reversal plus shuffled-label audits.
   This proves the information and architecture can support binding, but it does
   not prove zero-label reward training can discover it efficiently.

5. Memory dependence and compression are causally real on associative tasks, but
   we have not yet demonstrated a learned write policy that selects memory by
   later causal utility, nor compounding across a broad primitive curriculum.

## Newest experiment and the live fork

We implemented a pixels-only “micro-intercept” admission preflight:

- two frames reveal a target's hidden velocity;
- one opaque command has an unknown seed-specific effect in `{-1, 0, +1}`;
- the attempted command moves a cursor and changes the next rendered frame;
- only the attempted command's scalar success is given;
- the predictive core sees RGB frames and its own action;
- its decision latent contains the recurrent state plus learned predicted latent
  consequences for all three opaque candidate commands;
- controls are passive prediction, shuffled-action prediction, a fixed no-action
  stream, a fully fresh core, action-shuffled replay, and reward-shuffled replay;
- audits include true velocity reversal, a geometrically valid mirror,
  missing-evidence frames, Brier score, confidence, and frozen-cursor baseline.

The first GPU tier used 252 predictive lifetimes, 40 predictive updates, 270
reward bits, 68 replay updates per prefix, 192 held-out episodes, and finished
in about 20.5 seconds on an RTX 5090.

Results:

| Arm | AULC above 1/3 | Final accuracy |
|---|---:|---:|
| action-conditioned | 0.0169 | 34.38% |
| passive | 0.0391 | 34.38% |
| shuffled action | 0.0117 | 32.29% |
| fixed no-action | 0.0234 | 32.81% |
| fully fresh | ~0 | 33.33% |
| action-shuffled replay | 0.0039 | 30.73% |
| reward-shuffled replay | 0.0078 | 34.38% |

No arm reached 60%. The action-conditioned arm lost to passive by 0.0221 AULC.
Reversal was 32.81% with only 5.47% moving prediction flips. Mirror was 35.42%
with 53.13% flips. Missing-evidence accuracy was near chance, but confidence did
not fall relative to normal. The pre-registered gate failed.

Interpret this only as a bounded negative at this tiny budget. It is also not yet
a genuinely extended closed-loop control task: it has one chosen transition and
one terminal outcome. We suspect the task can be solved as a contextual bandit
from exogenous target velocity, so action-conditioned dynamics may provide no
useful advantage even though the action changes the last image.

## The central research question

What is the **smallest, fastest, scientifically honest closed-loop experiment**
in which:

1. actions causally change later observations over several decisions;
2. success genuinely requires learning action-conditioned consequences, rather
   than merely recognizing an exogenous state and selecting an action;
3. an action-conditioned predictive core has a principled opportunity to beat a
   passive predictor and shuffled-action control;
4. training uses no semantic or counterfactual labels;
5. attempted-action scalar outcomes remain the only behavioral supervision;
6. valid causal rerenders can distinguish learned control from shortcuts;
7. the entire first tier can run in under one minute on one RTX 5090;
8. the learned primitive is plausibly reusable on a later novel task.

## Required literature search

Search at least 40 relevant **primary papers**, including foundational and recent
work through July 2026 where available. Prefer original papers and official
implementations. Clearly distinguish direct evidence, analogy, and speculation.
Cover:

1. minimal partially observable control benchmarks and diagnostic micro-POMDPs;
2. action-conditioned representation learning and world models;
3. latent forward models, inverse dynamics, controllable representations,
   successor features, predictive-state representations, and value-equivalent
   models;
4. when passive video prediction fails to learn controllable factors;
5. multi-step overshooting and whether it helps at tiny scale;
6. contextual-bandit versus sequential-RL boundaries;
7. attempted-action-only / partial-label learning for sequential decision
   problems;
8. replay ratio and phase transitions in low-data learning;
9. system identification of unknown actuator protocols;
10. active experiment design and information-gain exploration without semantic
    labels;
11. confidence calibration under policy-induced distribution shift;
12. causal representation learning for controllable factors;
13. memory writes selected by later causal retrieval advantage;
14. forward transfer and measurable compounding across primitive curricula;
15. failure modes: noisy-TV prediction, shortcut policies, latent collapse,
    action leakage, simulator leakage, inadequate action coverage, and invalid
    counterfactual audits.

Specifically compare the relevance and computational burden of:

- a simple recurrent action-conditioned latent-delta model;
- contrastive predictive models with actions;
- inverse-dynamics auxiliaries;
- successor features;
- PSRs;
- RSSM/PlaNet/Dreamer-style models;
- value-equivalent models;
- small model-based controllers that evaluate candidate latent consequences;
- recurrent-free versus recurrent controllers;
- direct success prediction versus actor-critic in short horizons.

Do not recommend a large architecture merely because it is fashionable. Identify
the lowest-complexity method that can answer the scientific question.

## Required deliverables

### 1. Executive verdict

State the single best next sub-minute experiment and why it has the highest
expected information gain per GPU-second. Say explicitly whether to:

- repair and rerun the existing one-transition preflight;
- replace it with a 6–10 decision closed-loop task;
- or perform an even cheaper representation/causality probe first.

### 2. Diagnosis of the newest negative

Rank the plausible explanations for the result:

- insufficient predictive updates;
- insufficient reward replay;
- task solvable without action-conditioned dynamics;
- predicted-consequence interface not decision-useful;
- passive objective already captures everything required;
- action-conditioned objective is dominated by nuisance/background variation;
- optimization or normalization flaw;
- protocol/action coverage problem;
- confidence metric or audit flaw.

For each explanation, give the cheapest discriminating measurement and a
pre-registered interpretation.

### 3. Exact environment specification

Design a deterministic tiny closed-loop environment with:

- precise frames, action timing, horizon, hidden variables, and verifier;
- an opaque seed-specific actuator mapping;
- nuisance randomization;
- no privileged learner input;
- guaranteed balanced action/state coverage;
- a meaningful need for action-conditioned dynamics;
- direct causal rerenders;
- a way to test transfer of the learned primitive to a new actuator or task.

Avoid any environment where the optimal action can be read directly from the
current image without using learned consequences.

### 4. Exact experimental arms

Specify the smallest decisive matched comparison. At minimum consider:

- action-conditioned predictive core;
- passive predictor;
- shuffled-action predictor;
- fixed/no-effect environment;
- fully fresh core;
- attempted-action-only success readout;
- action-shuffled and reward-shuffled replay;
- a matched simple model-free baseline.

State which model weights are frozen, which are trained, and exactly what each
learner sees.

### 5. Losses without semantic leakage

Provide implementation-ready equations or pseudocode for the best losses. Every
loss must be classified as:

- legal deployed-learning signal;
- discarded diagnostic-only supervision;
- or prohibited semantic/counterfactual supervision.

Explain how to prevent predictive learning from spending capacity on irrelevant
background changes.

### 6. Sub-minute → three-minute → ten-minute ladder

For each tier provide:

- exact interaction budget;
- unique logical lifetimes;
- optimizer steps;
- replay ratio;
- approximate RTX 5090 runtime;
- primary metrics;
- mechanistic leading indicators;
- promotion gate;
- stop condition;
- interpretation of a flat pre-ignition valley.

Only escalate when the earlier tier gives solid evidence of high ROI.

### 7. Adversarial audit suite

Pre-register valid pixel-space audits for:

- reversed dynamics while holding the actuator mapping fixed;
- remapped actuator protocol;
- removed decisive evidence;
- frozen/no-effect actuator;
- irrelevant moving distractors;
- palette and geometry changes;
- action-sequence permutation through a valid rerender, not hidden-state swaps;
- action/reward shuffles;
- memory corruption if memory is used.

For each, state the expected accuracy, prediction-flip, and confidence response
under a genuinely causal solution.

### 8. Sample-efficiency and compounding ledger

Define how this experiment should measure:

- bits-to-threshold;
- AULC over unique reward bits;
- predictive utility;
- calibration;
- compute efficiency;
- transfer to a new actuator;
- transfer to a novel related primitive;
- retention of earlier primitives;
- and eventual negative slope in
  `log(reward_bits_to_threshold)` across at least six primitives.

Distinguish transfer from true compounding.

### 9. Ranked next 20 experiments

Rank 20 experiments by expected information gained per GPU-minute. Every row must
include hypothesis, inputs, legal loss, controls, metric, audit, pass threshold,
failure interpretation, and runtime tier. The first several must be sub-minute.

### 10. Do-not-adopt-yet list

Explicitly identify seductive but premature ideas—large world models, planners,
semantic concept bottlenecks, inferred unattempted-action targets, massive
meta-RL, long training runs, or anything else not justified by current evidence.

## Standard of reasoning

The report must actively try to falsify its recommendation. Extraordinary gains
must face shuffled-label/action controls and true causal rerenders. Treat all
fixed-budget failures as bounded. Do not confuse replay depth with interaction
efficiency, transfer with compounding, probe decodability with behavioral use, or
one post-action frame with genuine sequential control.

End with:

1. the exact first command/configuration we should run;
2. the exact gate that authorizes the three-minute tier;
3. the most likely finding that would make you abandon your preferred approach.
