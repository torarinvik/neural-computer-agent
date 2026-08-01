# Syllogimous Bitter Lesson experiment

> **Historical experiment:** This folder records the architecture actually
> tested at the time; it does not redefine the project target. See the
> [canonical amodal N-to-M specification](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md).

This folder is deliberately separate from `syllogimous_latent_agent`. The
hand-coded neural closure agent remains a diagnostic oracle; this experiment
tests whether a generic neural policy can learn its own internal procedure.

## Contract

- Inference inputs: raw RGB frames, raw PCM samples, and batching mask only.
- Outputs: public UI actions and a learned halt decision.
- No parsed propositions, entity IDs, relation IDs, graphs, closure, search, or
  semantic auxiliary targets enter the model.
- The verifier supplies outcome reward. Correctness dominates. To prevent fast
  random guessing from exploiting binary tasks, the latency objective is zero at
  chance and ramps in only after sustained accuracy exceeds 55%. A bonus of at
  most 0.05 is then paid only for a correct answer and decreases with thinking
  time.
- Memory writing, repeated reasoning, and halting are learned.

The curriculum changes problem length only. This controls experience without
specifying a solution algorithm.

## First H100 runs

Run from the repository root:

```bash
python -m experiments.syllogimous_bitter_lesson.train_rl \
  --scale 1m --overfit-fixed --train-samples 128 --eval-samples 128 \
  --train-premises 2 --eval-premises 2 --epochs 100 --batch-size 128 \
  --speed-bonus 0 --entropy-weight 0 \
  --checkpoint experiments/syllogimous_bitter_lesson/h100_overfit_128.pt \
  --report experiments/syllogimous_bitter_lesson/h100_overfit_128.json
```

This fixed-set diagnostic must reach at least 95% before larger generalization
runs are scientifically useful.

The matched Q-learning diagnostic replaces only the generic credit-assignment
rule and fixes the computation budget during this sanity check:

```bash
python -m experiments.syllogimous_bitter_lesson.train_rl \
  --scale 1m --learning-signal q_learning --q-epsilon 0.2 \
  --overfit-fixed --train-samples 128 --eval-samples 128 \
  --train-premises 2 --eval-premises 2 --epochs 100 --batch-size 128 \
  --speed-bonus 0 --checkpoint experiments/syllogimous_bitter_lesson/h100_q_overfit_128.pt \
  --report experiments/syllogimous_bitter_lesson/h100_q_overfit_128.json
```

`--learning-signal verifier` is a diagnostic control. It uses the verifier's
correct final UI action directly but still provides no semantic labels or proof
steps. If this control cannot overfit 128 episodes, the architecture itself is
broken; if it can while RL cannot, credit assignment is the isolated failure.

```bash
python -m experiments.syllogimous_bitter_lesson.train_rl \
  --scale 1m --train-samples 50000 --epochs 8 --batch-size 128 \
  --checkpoint experiments/syllogimous_bitter_lesson/h100_1m.pt \
  --report experiments/syllogimous_bitter_lesson/h100_1m.json

python -m experiments.syllogimous_bitter_lesson.train_rl \
  --scale 5m --train-samples 100000 --epochs 10 --batch-size 128 \
  --checkpoint experiments/syllogimous_bitter_lesson/h100_5m.pt \
  --report experiments/syllogimous_bitter_lesson/h100_5m.json
```

Treat chance-level or unstable learning as a real result. Do not repair it by
inserting syllogism structure. Improve generic optimization, curriculum,
memory, or scale and preserve all failed runs.

## Preserved first result

`h100_1m.json` and `h100_1m.pt` preserve the first outcome-only run before the
latency gate was corrected. After 400,000 generated training episodes it scored
49.45% on 2,000 held-out episodes (49.65%, 44.06%, 50.70%, 50.18%, 50.53%,
50.00%, and 51.05% at 2, 4, 8, 16, 24, 32, and 64 premises). It also halted after
one thought step. This is evidence of two failures, not learned reasoning:

1. Sparse binary policy gradients provided inadequate credit assignment.
2. A correct-only speed bonus still rewards fast guessing in expectation when
   chance accuracy is 50%.

The result is retained as the `v1` baseline rather than overwritten.

## Fixed-set diagnostic results

All runs below used the same 1,066,020-parameter model, the same 128 fixed
two-premise RGB/PCM episodes, 100 optimizer updates, and no latency reward.

| Signal/path | Greedy training-set accuracy | Conclusion |
|---|---:|---|
| REINFORCE through memory | 50.78% | failed |
| Sampled immediate Q-learning through memory | 49.22% | failed |
| Full verifier action through memory | 50.78% | failed |
| Full verifier action, direct sensory bypass | 74.22% | learned, below 95% target |

The full-information verifier control rules out sparse reward as the only
problem. The direct sensory bypass establishes that the pixels can produce a
learnable signal, while the present soft-write memory plus repeated reasoning
path destroys or obscures it. The next small experiments should replace that
path with simpler residual recurrent memory and add generic sensory
self-supervision; scaling is not yet justified.

### Generic memory and computation sweep

Matched runs retained the 128 fixed episodes and full verifier action control:

| Generic memory path | Thought steps | Accuracy after 100 updates |
|---|---:|---:|
| Soft slots with sensory residual | 6 | 59.38% |
| Two-layer residual GRU | 6 | 50.78% |
| Causal event Transformer | 6 | 84.38% |
| Causal event Transformer | 1 | 85.16% |
| Causal event Transformer | 2 | **89.84%** |

The winning 1.27M-parameter event Transformer with two recurrent thought steps
crossed 95% at update 117, exceeded 98% by update 127, and reached a stable 100%
by update 169. Its 200-update checkpoint and report are
`h100_event_transformer_t2_overfit_128_200.pt` and
`h100_event_transformer_t2_overfit_128_200.json`.

This is an optimization sanity result, not evidence of reasoning or
generalization: the model memorized 128 repeated episodes using only final
verifier actions. It establishes that the generic sensory-to-memory pathway is
trainable. The next gate is held-out accuracy on procedurally generated
two-premise episodes.

### Generated two-premise learning

The winning event Transformer was trained for five passes over procedurally
generated episodes with per-episode card-color variation. Evaluation used 2,000
sealed seeds and the unseen `Z` entity alphabet.

| Unique training episodes | Held-out accuracy |
|---:|---:|
| 5,000 | 50.00% |
| 10,000 | 86.10% |
| 25,000 | **97.45%** |

This sharp transition is genuine held-out generalization rather than fixed-set
memorization. The model received only final public verifier actions; it received
no entity, relation, graph, or proof supervision.

Outcome-only REINFORCE fine-tuning from the 25k checkpoint improved held-out
accuracy to **99.40%** after three 10k-episode passes. A matched reward-only run
from random initialization remained at exactly **50.00%**. Thus sparse reward
can refine an acquired sensory policy but did not discover it efficiently from
scratch under this budget.

With no additional updates, the refined policy scored **73.80%** on unseen
three-premise chains and **70.70%** on four-premise chains (72.25% combined).
This is promising length transfer, though not yet evidence of a general
reasoning algorithm.

### Mixed curriculum and extreme-length transfer

Starting from the two-premise policy, a staged verifier curriculum introduced
three- and four-premise chains. A short fixed-compute, outcome-only RL pass then
refined the trained range. Held-out accuracy was:

| Premises | Accuracy | Training status |
|---:|---:|---|
| 2 | 99.60% | trained |
| 3 | 99.40% | trained |
| 4 | 94.60% | trained |
| 6 | 82.00% | zero-shot |
| 8 | 83.60% | zero-shot |
| 12 | 82.30% | zero-shot |
| 16 | 81.30% | zero-shot |
| 24 | 79.87% | zero-shot |
| 32 | 79.87% | zero-shot |
| 64 | 77.20% | zero-shot |

Removing all per-episode color variation left the extreme-length result nearly
unchanged (78.89% overall versus 78.98%), ruling out dependence on randomized
card colors.

The stronger paired counterfactual audit retained every premise and changed
only the visible conclusion relation while inverting the correct answer. Across
2,500 pairs at 2, 4, 8, 16, and 64 premises:

- original accuracy: 86.80%
- counterfactual accuracy: 87.60%
- both members correct: 81.76%
- prediction changed in the required direction: 89.12%

Paired-both-correct accuracy was 98.60% at two premises, 94.60% at four, and
69.20% at 64. These interventions strongly reject fixed-answer, seed, layout,
length, and premise-only shortcuts. They support learned relational behavior,
although transfer across genuinely different task families remains untested.

### Selective evidence and distractor branches

The first structural transfer suite hid a short relevant path among shuffled,
disconnected premises from mixed relation families. With no such training, the
chain policy scored only **53.86%** overall and fell to 47.83% for 64 total cards
with four relevant links. This localized a real limitation: length robustness
did not imply selective evidence retrieval.

A mixed curriculum then used 50% ordinary chains and 50% branched/distractor
episodes, without adding an evidence selector or graph operation. After 50,000
episodes, held-out branched accuracy rose to **99.72%**. It achieved 100% on
64-card configurations that were absent from training.

To test configuration memorization, a second sealed suite combined unseen total
lengths with unseen relevant depths:

| Total premises | Relevant depth | Accuracy |
|---:|---:|---:|
| 12 | 3 | 89.00% |
| 24 | 3 | 85.40% |
| 24 | 6 | 82.20% |
| 48 | 6 | 87.60% |
| 48 | 12 | 96.60% |
| 64 | 12 | 97.80% |

Overall novel-composition accuracy was **89.77%**. Ordinary all-relevant chain
accuracy remained 97.63% at two premises, 91.13% at four, and 75.00% at 64.
This shows learned selective evidence behavior and compositional transfer within
the relational domain, with some cost to the original chain distribution.

### Cross-family transfer: parity propagation

The first genuinely different operation used symmetric `SAME`/`FLIP`
constraints. Solving a query requires XOR-like parity composition rather than
directional transitive reachability. The relational model was at chance zero
shot (49.88%), as expected for unseen visual relation words and a new operation.

Matched 50,000-episode parity curricula produced a large transfer effect:

| Initialization | Overall held-out accuracy | Length 2 | Length 4 |
|---|---:|---:|---:|
| Relational checkpoint | 63.26% | 87.20% | 85.30% |
| Random initialization | 50.06% | 50.70% | 50.10% |

The pretrained model learned two-premise parity to essentially 100% during its
second epoch, while the scratch model never rose above chance. This is evidence
of cross-family transfer in learning efficiency, not zero-shot task solving.

An extended 2→8 parity curriculum reached 100% at lengths two and four and
69.80% at length eight. Zero-shot performance was 52.90% at length sixteen and
49.70% at thirty-two. Thus the shared representation accelerates acquisition,
but a length-general iterative parity algorithm has not yet emerged.

A matched recurrent-compute sweep tested whether simply allowing more internal
thought iterations fixes that failure. All runs used the same pretrained
relational checkpoint, curriculum, data volume, and weight-shared thought cell:

| Thought steps | Length 2 | Length 4 | Length 8 | Length 16 | Length 32 | Length 64 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 100.0% | 100.0% | **69.8%** | **52.9%** | 49.7% | not measured |
| 4 | 100.0% | 97.4% | 58.0% | 51.5% | **50.6%** | **49.2%** |
| 8 | 100.0% | 100.0% | 62.1% | **52.9%** | 50.5% | 48.5% |

More recurrent computation did not improve extrapolation and actually reduced
length-eight accuracy. The present thought transition therefore does not behave
like a reusable parity-update algorithm: repeated application can perturb a
useful state instead of monotonically refining it. The next architecture test
should make the thought dynamics state-preserving (a learned gated residual
update initialized near identity), train with randomized thought counts, and
supervise the same final answer at several compute depths. That remains a fully
learned sensory-to-action system; it changes the optimization pressure, not by
inserting a symbolic parity routine.

The follow-up implemented that test with an 8-step learned gated-residual
transition initialized near identity. Each training example selected a random
depth from one through eight, with outcome supervision at that depth and a
small cross-depth consistency loss. It added only generic learned parameters
(1.290M total versus 1.272M) and retained the raw RGB/PCM-only boundary.

The held-out result was **68.08%** overall: 99.6% at length two, 97.3% at four,
59.3% at eight, 51.4% at sixteen, 49.7% at thirty-two, and 51.2% at sixty-four.
An audit of the same examples at every internal compute depth measured 66.75%,
67.00%, 67.75%, 67.92%, 68.38%, 68.23%, 68.25%, and 68.08%. Thus the gated
update successfully removed the earlier destructive-depth behavior and made
additional thinking mildly useful, but did **not** yield length-general parity.
The remaining failure is representational/algorithmic rather than simple state
instability: all compute depths learn essentially the same bounded-length
strategy.

An accuracy-gated curriculum then trained only lengths 2, 4, and 8. A new
length was unlocked only after every active length reached at least 95% on a
held-out validation set for two consecutive epochs. Training replay ratios were
100% length 2, then 30/70 for 2/4, then 15/25/60 for 2/4/8. Length 4 unlocked at
epoch 2 and length 8 at epoch 6. The curriculum completed at epoch 15 with
validation scores of 100%, 100%, and 98.2%. An independent final evaluation
reproduced **100% / 100% / 98.2%** at lengths 2/4/8, compared with 59.3% at
length 8 in the fixed schedule. Lengths 16, 32, and 64 remained at chance.
This establishes that mastery-gated progression solves the immediate learning
efficiency failure; the next controlled stage can now introduce length 16 while
replaying the three mastered lengths.

That checkpoint then unlocked length 16 with a 10/15/25/50 replay mix. The
stage completed after nine epochs and independently scored **100% / 100% /
100% / 99.4%** at lengths 2/4/8/16; unseen length 32 remained at chance.

An initially aggressive length-32 attempt (50% frontier examples at the
original learning rate) left length 32 at chance and rapidly damaged lengths 8
and 16, so it was stopped after three epochs. A clean restart used a `1e-4`
learning rate and a conservative 10/15/20/25/30 replay mix for 2/4/8/16/32.
Length 32 crossed the gate after 24 epochs. Independent final accuracy was
**100% / 100% / 100% / 99.6% / 96.7%** through length 32, while unseen length
64 remained at chance (50.6%). These results show that powers-of-two mastery
gating plus substantial replay can extend the learned skill, although the lack
of zero-shot length doubling still argues against a fully length-general parity
algorithm.

The length-64 stage added best-validation checkpoint restoration and used a
lower `5e-5` learning rate with 5/10/15/20/20/30 replay weights through length
64. It exhausted its 40-epoch budget without meeting the two-epoch 95% mastery
gate. The best joint validation state occurred at epoch 37. On a larger
independent evaluation, restored-best accuracy was **100% / 100% / 100% /
99.8% / 99.3% / 93.3%** at lengths 2/4/8/16/32/64. Unseen length 96 scored
54.8%. Thus curriculum training moved length 64 from chance to strong but
incomplete performance while preserving earlier skills. This checkpoint is an
intermediate milestone, not a mastered stage; length 96 remains locked.

A consolidation run resumed the restored length-64 checkpoint at a `2e-5`
learning rate with 500 held-out validation examples per active length. Length
64 crossed 95% several times before finally sustaining 95.4% and 95.0% in
epochs 14 and 15, completing the mastery gate. Restoring the best joint state
produced independent accuracy of **100% / 100% / 100% / 100% / 99.3% / 95.4%**
at lengths 2/4/8/16/32/64. Unseen length 96 was 49.6%, so mastery has extended
through 64 but still does not extrapolate zero-shot to the next untrained
length. Length 96 may now be explicitly unlocked.

The first length-96 stage resumed the mastered length-64 checkpoint with a
`1e-5` learning rate and approximately 73% replay across lengths 2 through 64.
It exhausted 30 epochs without approaching the mastery gate. The best joint
validation state was epoch 20 at 72.7% for length 96. Independent restored-best
accuracy was **100% / 100% / 100% / 99.9% / 98.7% / 91.4% / 73.5%** through
length 96; unseen length 112 scored 39.1%, reflecting a systematic prediction
bias rather than positive transfer. This is an intermediate diagnostic only:
length 96 is not mastered, length 64 was partially destabilized, and length 112
must remain locked. The sharp slowdown compared with the 32→64 stage suggests
that the 1.29M-parameter system has reached a meaningful capacity or
optimization boundary.

### Scaling pilot and representation pretraining

The nominal `5m` event-Transformer preset contains 8.65M parameters once its
sequential core and gated thoughts are included. A scratch parity pilot stayed
at **49.98%** after 200,000 episodes, including exactly 50% at length two.
Full-precision and high-learning-rate memorization diagnostics ruled out BF16
as the cause. Raw parameter count alone therefore did not improve acquisition.

An intermediate `2m` preset was added; its full event model has 2.35M
parameters. It learned more slowly than the 1M model but exposed the missing
causal ingredient in the old metadata: successful parity transfer used a
three-stage chain → mixed-structural → parity pipeline. The 2M chain model
reached 98.18% after consolidation, and its structural checkpoint retained
98.25% at two premises with useful relational length transfer.

Starting parity from that learned sensory representation produced immediate
transfer. The model mastered lengths 2/4/8 in ten epochs and length 16 in six
more, scoring **100% / 100% / 100% / 99.9%** independently. This sharply
contrasts with the larger scratch model's chance result and shows that learned
representation history dominates scale in the current regime. A conservative
length-32 stage reached **86.7%** independently (with 2/4/8 at 100% and 16 at
99.7%) but did not meet mastery. Extra capacity accelerated short-to-medium
acquisition; it did not automatically produce length-general parity or beat
the heavily consolidated 1.29M model at long sequences.

A lower-rate length-32 consolidation improved the 2M model to **100% / 100% /
100% / 99.8% / 95.9%** independently at lengths 2/4/8/16/32. A subsequent
strict confirmation run did not produce two consecutive 95% validation passes,
but restored its best state and independently scored **100% / 100% / 100% /
99.8% / 96.2%**; unseen length 64 remained at chance (47.1%). This distinction
is intentional: the checkpoint is strong at length 32, but the original
powers-of-two mastery gate is not recorded as formally complete.

The next curriculum removes the large length jumps. It starts with mastered
anchors 2/4/8/16, requires at least 99% on 250 held-out examples for every
active length in two consecutive epochs, and then unlocks exactly one additional
premise: 17, 18, 19, and so forth. Length 17 passed this stricter gate at epochs
8 and 9 (100% at the frontier on both passes, with the minimum retained score
99.2%) and unlocked length 18. This experiment tests whether smooth increases
in sequence length produce a learned iterative procedure more efficiently than
doubling the required horizon.

The 100-epoch incremental run subsequently mastered length 18 at epochs 73 and
74 and unlocked length 19. Length 19 did not satisfy the two-pass gate within
the remaining budget. Best-checkpoint restoration selected epoch 97 at stage 7,
where the minimum held-out accuracy over 2/4/8/16/17/18/19 was **98.8%**.
Independent evaluation of the restored model confirmed 100% / 100% / 100% /
99.73% / 99.45% at lengths 2/4/8/16/17; the configured final suite accidentally
omitted lengths 18 and 19 and should include them in the next audit. Untrained
longer lengths remained mostly near chance. The safe resume point is therefore
stage 7 (length 19 unlocked but not mastered), using
`h100_parity_2m_incremental_64.pt` as initialization.

### Choice reaction and selective attention

The next task family adds a simpler cognitive primitive beneath relational
reasoning: a light appears at one of two to eight fixed response locations and
the agent must emit the matching motor action. The loop is deliberately basic:
perceive → discriminate → select → act. Trials can add blank delay frames,
irrelevant hollow visual objects, and unrelated audio tones. Every trial is
deterministic and has exactly one verifiable answer.

`choice_reaction.py` renders these trials into the same 160×96 RGB and raw PCM
interface used by the reasoning games. No target index, task identifier, or
renderer state enters the model. A shared `mixed_cognitive` dataset can
interleave reaction and parity/syllogism-style episodes through the same CNN,
audio encoder, event Transformer, recurrent thought cell, and motor head.
The motor vocabulary expands from five to eight outputs while retaining the
five learned rows from an older checkpoint.

The intended curriculum is accuracy-first: master 2 choices, then unlock
3 through 8 one at a time; next add distractor count, delay, and audio noise.
Only after stable accuracy should outcome-only RL tune the learned halting head
with a small speed bonus. This prevents fast guessing from being rewarded while
still applying direct pressure toward rapid correct reactions.

A local Apple M5 smoke test used the 1.291M-parameter model, two recurrent
thought steps, 12,000 training trials, and a 2,800-trial held-out evaluation.
The three training epochs introduced choices 2–3, then 2–5, then 2–8. Accuracy
rose from 70.19% in epoch one to 100% in epochs two and three. Final held-out
accuracy was **100% at every choice count from 2 through 8**, including at the
first thought depth. Estimated training throughput was 876 trials/s, evaluation
throughput was 1,818 trials/s (0.550 ms/episode), and measured whole-process
throughput including startup and checkpointing was 913 trials/s. Raw deterministic
trial rendering alone reached 8,414 trials/s. This is a pipeline smoke test with
no distractors or delay, not yet evidence of selective-attention mastery.

`choice_reaction_realtime.py` adds the batch-one streaming evaluation boundary.
The private environment releases timestamped RGB/PCM packets one frame at a
time; the policy retains only those public packets, performs inference after
stimulus onset, emits one motor action, and receives correctness plus a small
deadline-normalized speed bonus. Accelerator compilation is handled by explicit
unscored warm-up trials, and `.item()` synchronization ensures latency ends only
after the action has actually left the accelerator.

On the Apple M5, 1,000 warm-started eight-choice trials achieved **100%** accuracy
with 5.00 ms mean, 4.89 ms median, 5.70 ms p95, and 6.81 ms p99 response latency
at batch size one. A harder untrained stream with ten visual distractors, three
extra delay frames paced at 60 Hz, and five irrelevant audio tones also achieved
**100%** over 100 trials, with 8.62 ms mean and 10.79 ms p95 latency measured
from stimulus presentation to action emission. Total paced throughput was 11.41
trials/s because each trial included four 16.67 ms display intervals before the
response. These measurements exercise the full renderer → packet → tensor →
model → synchronized motor-action path, but not yet an OS window/screen-capture
round trip.

The harder attention generator adds bright response-aligned decoys and salient
premature flashes at incorrect response locations. A local curriculum combined
20 ordinary distractors, six target-like decoys, three temporal false cues,
four delay frames, and eight irrelevant audio tones. The 1.291M model reached
100% training accuracy by epoch two and scored **100% at every choice count from
2 through 8** over 2,800 held-out trials. In a 100-trial batch-one stream paced
at 60 Hz it retained **100%** accuracy with 11.99 ms mean, 12.08 ms median,
13.75 ms p95, and zero 100 ms deadline misses.

The syllogism/parity checkpoint was deliberately not fine-tuned locally. Its
SHA-256 remains `fac9a6e2c709cf99a7d3878294cb07362a31e2c428a9259b63de628eaca42d2f`.
An attempted local regression audit revealed a real raw-pixel domain shift:
Linux and macOS font/Pillow rasterization produce different card pixels, and
the Linux-trained checkpoint falls to chance on locally re-rendered cards even
when using the same font family. This is not evidence of weight regression.
The next mixed-cognitive run must therefore occur in the original Linux/CUDA
renderer environment and enforce a sealed before/after parity gate at
2/4/8/16/17/18/19. The attention checkpoint remains a separate branch until
that same-domain audit passes.

The first same-domain RTX 5090 mixed pilot expanded the protected 2.35M reasoner
to eight actions and trained two epochs at `5e-6` on a 50/50 mixture of parity
replay and the hardest attention configuration. A sealed 1,400-trial reasoning
suite scored **99.5% both before and after**. Per-length accuracy changed from
100/100/100/99.5/99.0/98.0/100% to
100/100/100/99.5/99.5/97.5/100% at 2/4/8/16/17/18/19, so no length regressed by
more than 0.5 percentage points. The no-regression gate passed.

Attention did not yet pass: the separate 2,800-trial hard-attention evaluation
scored **35.21%** overall, ranging from 55.0% at three choices to 19.5% at eight.
The model was still improving during training (58.8% mixed training accuracy in
epoch one and 64.6% in epoch two), but the run was intentionally too short for
mastery. `small_mixed_50_50.pt` is therefore a useful diagnostic/resume point,
not a replacement for either protected best checkpoint. The next pilot should
continue with short audited blocks and retain the same reasoning regression
gate.

A second identical two-epoch 50/50 block continued from the accepted pilot.
Hard-attention accuracy improved from **35.21% to 49.43%**, and mixed training
accuracy reached 72.67%. However, the sealed reasoning audit fell from 99.5% to
98.71% overall: length 16 dropped from 99.5% to 97.5% and length 18 from 98.0%
to 95.0%. Those 2–3 point losses violate the per-length one-point regression
limit, so `small_mixed_50_50_block2.pt` is explicitly rejected despite its useful
attention gain. The accepted resume point remains `small_mixed_50_50.pt`.
The next continuation should restart there with substantially more reasoning
replay, rather than attempting to repair the rejected weights.

Restarting from the accepted block-one checkpoint with **75% reasoning replay
and 25% hard attention** produced a better tradeoff. After two epochs, the same
sealed reasoning audit scored **99.57%**, versus the original 99.5% baseline.
Per-length accuracy was 100/100/100/99.5/100/98.0/99.5% at
2/4/8/16/17/18/19; no length moved by more than 0.5 points and the regression
gate passed. Hard-attention accuracy reached **47.57%**, up from 35.21% at the
accepted resume point and close to the rejected 50/50 continuation's 49.43%.
`small_mixed_75_25.pt` is therefore the new accepted shared checkpoint. This
establishes that attention can improve substantially without damaging reasoning
when rehearsal dominates the mixture.
