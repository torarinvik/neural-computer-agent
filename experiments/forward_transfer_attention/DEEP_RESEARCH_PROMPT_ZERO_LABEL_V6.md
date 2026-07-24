# Deep research prompt: from actuator transfer to compounding cognitive transfer

We are building a tiny real-time neural computer that must learn reusable
abstract capabilities from experience. We need a rigorous, implementation-ready
research review and experiment plan for the next fork.

## North star

Maximize:

> verified reusable capability gained per unique experience, reward bit,
> optimizer/example budget, GPU-second, wall-clock second, and action latency.

Accuracy is primary. We want each learned primitive to reduce the evidence
needed for later genuinely different primitives. Eventually we want a negative
slope in log(reward bits to threshold) across a growing curriculum while old
capabilities remain intact.

## Non-negotiable information regime

The deployed learner may receive only:

- rendered visual/audio/text sensory streams;
- its recurrent/latent state and external memory;
- its own attempted actions and known logging propensities;
- observed scalar outcomes from a deterministic verifier;
- task-agnostic predictive, consistency, compression, retrieval-utility, and
  verified-retention signals derived from its own experience.

The deployed learner must not receive:

- semantic concept labels or human-authored class names;
- game state, task IDs, generator parameters, private rules, correct actions,
  or solution traces;
- targets for actions it did not attempt, even when a human could infer them;
- fixed symbolic planners, DSLs, English chain of thought, semantic routers, or
  manually assigned meanings for latent coordinates.

Verifier metadata may train discarded offline diagnostic probes and construct
offline causal audits. Probe weights never enter the agent, and probe accuracy
is not capability. The verifier remains sovereign.

We want emergent amodal latents and intentions. A learned intention should be
usable through new keystroke, JSON, bit-flag, speech, or device adapters without
making those protocols the reasoner's internal ontology.

## Experimental discipline

- Start under one minute.
- Advance to roughly three minutes only after a pre-registered gate passes.
- Advance to ten minutes only after replication and causal audits.
- Treat one seed as provisional; use three seeds for an exploratory milestone.
- Use at least five seeds and uncertainty intervals for a load-bearing claim.
- Count unique interactions, unique logical lifetimes, reward bits, optimizer
  updates, examples processed, GPU-seconds, wall time, and latency separately.
- Never call replay “free” sample efficiency: fixed reward bits plus more
  optimizer work is interaction efficiency only when compute is disclosed.
- Direct deterministic held-out rerendering is the source of truth. Use OPE
  only when direct execution is unavailable or adaptive logging demands it.
- Valid counterfactuals are rerendered sensory streams, not impossible hidden
  state swaps.
- A fixed-budget negative means only “no learning within this budget.”

Hardware is one RTX 5090 with 32 GB VRAM. Current full microexperiments take
about 28 seconds per seed.

## Established result 1: zero-label temporal behavior

A small visual encoder + GRU was trained by paired latent-delta prediction with
anti-collapse variance/correlation terms. A downstream action-conditioned
success model learned only from logged tuples:

`(frozen recurrent state, attempted action, propensity, observed scalar reward)`

BCE was applied only to the attempted action. No unattempted-action target was
created.

After catching and removing a generator shortcut, the support-only temporal
task had one valid route:

`object A → object B → selected-object feedback → infer first versus last`

Across seeds 211/257/313:

- final verified accuracy: 78.13% / 82.03% / 80.73%;
- mean reward AULC above 50%: 0.2236;
- shuffled-future representation mean final: 55.73%;
- shuffled-future representation mean AULC: 0.0434;
- equally optimized IPS mean AULC: 0.1780;
- true support-order reversal: 78.82% mean relabeled accuracy;
- reversal prediction flips: 59.11% mean.

Action-shuffled, reward-shuffled, and fresh controls remained weak. Every curve
point used a fixed reward-bit prefix, 200 optimizer updates, and 6,000 replayed
examples.

This establishes one zero-semantic temporal primitive, not compounding.

## Established result 2: strong actuator/interface transfer

Phase A trained an eight-dimensional intention bottleneck and a two-action
success decoder on the temporal primitive using attempted-action outcomes only.

Phase B froze the learned intention, discarded the old decoder, and trained a
fresh four-command device adapter. A seed-specific protocol mapped the two
intentions onto two of four command IDs; two commands were distractors.
The adapter learned the unknown protocol only through attempted commands and
scalar outcomes.

Three-seed result:

| Metric | Seed 211 | Seed 257 | Seed 313 | Mean |
|---|---:|---:|---:|---:|
| Experienced final accuracy | 77.60% | 81.51% | 81.77% | 80.30% |
| Fresh final accuracy | 77.08% | 79.17% | 78.13% | 78.13% |
| Experienced AULC above 50% | 0.2828 | 0.3073 | 0.3151 | 0.3017 |
| Fresh AULC above 50% | 0.1750 | 0.2115 | 0.2010 | 0.1958 |
| Reward bits to 75%, experienced | 32 | 32 | 32 | 32 |
| Reward bits to 75%, fresh | 510 | 256 | 256 | 340.7 |
| True reversal accuracy | 77.08% | 80.47% | 81.51% | 79.69% |
| Opposite-rule stale accuracy | 22.40% | 18.49% | 18.23% | 19.70% |
| Swapped-protocol accuracy | 22.40% | 18.49% | 18.23% | 19.70% |

The reward-bit transfer ratio to 75% was 15.94× / 8× / 8× (median 8×,
mean 10.65×) at equal 200-update/6,000-example phase-B compute.

This is clean zero-label actuator/interface transfer. It demonstrates reuse of
the same intention through a new protocol, not transfer to a different
cognitive primitive.

## New result: temporal intention did not accelerate a spatial primitive

We created a balanced simultaneous spatial-selection task:

1. one RGB frame shows two colored objects at left and right;
2. one feedback frame shows only the selected object's identity;
3. the private verifier asks whether that identity had been on the left or
   right;
4. training and test use disjoint color-pair sets;
5. output again uses a new seed-specific four-command protocol.

The private left/right rule is exactly balanced. A horizontal mirror reverses
the two visible positions while preserving the feedback identity and flips the
correct protocol command.

Phase-B comparison at seed 211:

- temporal intention, trainable: final 78.39%, AULC 0.2427;
- temporal intention, frozen: final 75.26%, AULC 0.2281;
- fresh intention: final 79.43%, AULC 0.2396;
- trainable temporal advantage over fresh: only +0.0031 AULC;
- all three reached the same 55/65/75% thresholds at 32/128/256 reward bits.

Causal audits passed:

- mirrored relabeled accuracy 81.77%;
- mirrored prediction flips 60.16%;
- missing-feedback accuracy 48.70%;
- opposite-rule stale accuracy 21.61%;
- swapped-protocol accuracy 21.61%;
- action/reward shuffled arms crossed no threshold.

Therefore the spatial capability is real, but the temporal intention weights
did not make it meaningfully faster. We stopped after one seed under the
pre-registered rule.

### Critical interpretation detail

The “fresh intention” arm shared the same delta-predictively pretrained visual
encoder and GRU as the temporal arm. Thus this result cleanly rejects transfer
through the task-specific intention initialization, but it does **not** yet
measure whether transfer already occurred in the shared predictive core.

A fully fresh-core/fresh-intention arm is the obvious missing factorial cell.
Please assess whether running this cell is the highest-ROI immediate
experiment, and specify the fairest initialization and compute matching.

## Central research question

Why did same-intention actuator transfer produce an 8× median reward-bit gain,
while temporal-intention initialization produced essentially zero gain on a
different spatial primitive?

What is the smallest task-agnostic mechanism likely to turn learned primitives
into faster acquisition of later different primitives under our strict
zero-label information regime?

Do not answer with “use a larger model” or a generic survey. We need a ranked,
falsifiable plan driven by this exact dissociation.

## Questions to answer

### A. Localize where reuse can live

1. Design the minimum factorial matrix separating:
   - experienced versus fresh visual encoder;
   - experienced versus fresh recurrent core;
   - experienced versus fresh intention module;
   - retained versus reset success head;
   - retained versus reset/remapped actuator adapter.
2. Which cells distinguish:
   - generic perceptual transfer;
   - recurrent relational transfer;
   - intention warm-start;
   - protocol reuse;
   - mere optimizer conditioning?
3. Should previous modules be frozen, fine-tuned, gated, or made available as a
   library? Give a cheapest-first causal sequence rather than one large run.
4. How should we detect negative transfer and catastrophic interference at
   every boundary?

### B. Mechanisms for emergent primitive reuse

Rank the following—or better alternatives—by expected information gain per
GPU-minute in this exact tiny regime:

- shared recurrent predictive core alone;
- fast weights or a differentiable within-lifetime plasticity rule;
- task-agnostic external memory storing event/intention latents;
- learned retrieval based only on later verified advantage;
- a set/slot workspace that can preserve multiple primitive states;
- hypernetwork or learned optimizer driven only by experience and reward;
- successor features or generalized value features;
- contrastive/predictive objectives over successful experience;
- latent action-effect representations;
- modular routing learned without task IDs or semantic module labels;
- recurrent meta-RL / RL²-style adaptation;
- predictive-state representations;
- compression/consolidation that preserves future learning speed.

For each, state what information it consumes, why it does or does not violate
our philosophy, its likely failure mode, and the cheapest decisive control.

### C. Design the next cross-primitive task

5. Was temporal first/last → spatial left/right too easy, too unrelated, or
   already solved by the shared core? Explain how each hypothesis predicts the
   observed equal thresholds.
6. Propose the single best next primitive that is different enough to count,
   but shares a reusable computational operation. Candidates include:
   - same/different identity;
   - delayed match/non-match;
   - count one/two/three events;
   - odd-one-out;
   - conjunction of order and identity;
   - spatial/temporal composition;
   - reversal/inhibition;
   - visual search with distractors;
   - line counting or path following;
   - simple mental rotation.
7. Specify a deterministic visual generator, private verifier, chance/majority
   floor, reward-bit prefixes, and valid causal interventions.
8. Make the task impossible to solve through old action IDs, palette
   correlations, support/query anti-correlations, feedback identity alone, or
   fixed renderer artifacts.

### D. Compounding architecture and curriculum

9. Propose a six-primitive curriculum where each new task can measure reuse of
   one or more earlier operations without hand-declaring semantic composition.
10. How can a learned curriculum reward verified learning progress without
    abandoning tasks during the 600–1,400-update pre-ignition valleys observed
    in our project?
11. How should novelty be computed from verifier-side generator metadata
    without exposing it to the learner or rewarding noisy unpredictability?
12. Define the correct compounding statistic when some agents never cross a
    threshold. Compare log bits-to-threshold slopes, survival analysis,
    transfer matrices, forward transfer, backward transfer, and retention.

### E. Memory that earns its existence

13. Design a fixed-write-budget retrieval-advantage microexperiment where:
    - early information is needed after a distractor interval;
    - surprise/noise is abundant but useless;
    - real retrieval, no retrieval, random retrieval, shuffled memory, garbage
      memory, and equal-volume noise storage are compared;
    - a write/retrieval gets credit only if it improves later verified success;
    - memory compression is rewarded only when capability and future learning
      speed do not degrade.
14. Could learned memories store reusable “program-like” latent procedures
    without imposing a symbolic DSL? Give an operational test rather than a
    philosophical answer.

### F. Online action selection and calibration

15. We currently use balanced uniform logs and direct held-out execution. When
    should we move to an induced online policy with a conservative exploration
    floor?
16. Specify a four-command exploration schedule based on minimum evidence per
    command, not a copied two-action epsilon.
17. Which calibration metrics matter for action ranking versus abstention or
    extra thought? Distinguish ranking quality from probability calibration.
18. Could a success predictor guide extra thought/retrieval without generating
    its own reward or Goodharting? Give a staged passive-to-active ladder.

### G. Closed-loop escalation

19. When should we introduce the micro-intercept task where actions alter future
    pixels?
20. Design a sub-three-minute comparison of passive prediction,
    action-conditioned multi-horizon prediction, and shuffled-action control.
21. State what behavioral result would justify RSSM/PSR machinery and what
    negative would rule it out at this scale.

## Required deliverables

1. An executive verdict naming the **single next sub-minute experiment**.
2. A precise interpretation of the actuator-success / cross-primitive-failure
   dissociation.
3. The missing fully fresh-core factorial design, including expected outcomes.
4. A ranked table of at least 20 experiments by expected information gain per
   GPU-minute.
5. For the top five: pseudocode, learner-visible inputs, losses, controls,
   accounting, causal audits, pass bars, and failure interpretations.
6. A concrete six-primitive curriculum and compounding-analysis protocol.
7. A retrieval-advantage memory experiment that cannot profit from noise.
8. An online calibration/exploration plan for four commands.
9. A closed-loop micro-intercept escalation criterion.
10. A “do not adopt yet” section.
11. Primary-source citations, prioritizing small-data, online, continual,
    meta-learning, external-memory, causal-representation, contextual-bandit,
    predictive-state, and learned-plasticity work.

## Epistemic requirements

- Separate established facts, inferences, and speculative hypotheses.
- Do not treat probe decodability as deployed capability.
- Do not infer labels for unattempted actions.
- Do not call the one-seed temporal-to-spatial negative universal.
- Do not call actuator transfer cross-primitive compounding.
- Do not hide replay compute behind reward-bit efficiency.
- Do not recommend semantic concepts, hand-written modules, or task IDs.
- Do not use OPE as the main truth when deterministic rerendering is available.
- Treat any fixed-budget negative as bounded.
- Prefer the smallest experiment that distinguishes competing explanations.
