# Neural Computer Agent

A compact research repository for building a real-time neural computer that
learns reusable cognitive primitives from sensory streams and deterministic
outcomes.

Audited model checkpoints are stored in the private Hugging Face repository:
<https://huggingface.co/torarin87/neural-computer-agent>.

The learner receives rendered vision/audio/text streams, its own opaque
actions, its own latent state and memory, and scalar verifier outcomes. It
does not receive game state, coordinates, semantic task labels, rule IDs,
correct-action labels, English chain-of-thought, or counterfactual labels for
actions it did not attempt.

## North star

Maximize verified reusable capability gained per unique interaction.

The project distinguishes:

- unique verifier/reward bits;
- unique logical lifetimes;
- replayed examples;
- optimizer updates;
- GPU and wall time;
- action latency;
- retention and forward-transfer ratios.

Final accuracy alone is not an adequate score.

## Current audited frontier

The new unified-controller line now has its first retained compounding
milestone. A single 298,252-parameter controller with one vision encoder,
recurrent state, generic differentiable workspace, latent intention, and
replaceable actuator adapter learns hidden visual-action functions from its
own attempted opaque actions and scalar outcomes.

Prior visual grounding changed a matched 600-step four-rule task from a stable
75% shortcut to 99.85–99.90% on two independent seeds. The next rung inferred
an identity-versus-flipped mapping after one support outcome; inherited
training reached 100%, while matched fresh stayed at 49.26%. Balanced rehearsal
then preserved both the one-support skill and the broader two-support
four-function skill. The selected checkpoint passed disjoint 2,048-lifetime
normal, private-rule reversal, prediction-flip, blank-vision,
shuffled-feedback, and active-state-reset audits:

- one-support bijection: 99.98% normal, 99.95% reversed;
- retained four-function task: 100% normal and reversed;
- paired counterfactual flips: 99.93% and 100%.

This is evidence of fast within-lifetime binding, positive forward transfer,
and behavioral retention in one controller.

The same controller now also performs content-addressed latent recall across
active-state resets. A 600-update capacity-two rung reached 96.53% blind
recall; 150-update bridges at capacities 8 and 16 produced zero-shot transfer
to capacities 16 and 32. A later five-second rung used only 20 new-memory
updates at capacity 40 and reached 90.00% blind recall, then transferred
zero-shot to capacity 48 at 88.28% and capacity 56 at 87.33%. An independent
five-second acquisition replicated the result, and the two checkpoints crossed
the old capacity-64 frontier at 85.57% and 86.33%. Empty, shuffled, and
corrupted memories collapse toward chance; disk save/load reproduces hard
retrieval; the earlier one-support and four-rule skills remain retained. The
frozen retrieval frontier is now capacity 72; both parents fail capacity 80.

A subsequent selective-memory atom learned from verified success minus a
generic write cost. On blind data it wrote on 61.16% of first encounters but
only 5.10% of redundant repeats, averaging 0.663 writes per context while
retaining 99.90% query accuracy. Removing writes, shuffling admissions,
corrupting values, or hiding the prior memory read causally degraded the
appropriate behavior.

The first physical integration audit exposed the next boundary: intentionally
absent default rows retrieve unrelated neighbors in a shared disk bank. A
reward-learned scalar rejection threshold restored 87.99–88.96% disk accuracy,
but duplicate growth narrowly missed its gate at 20.02–21.39%. Raw-cosine
confidence improved the signal but still produced 27–29% false accepts. The
learned sparse selector and shared disk reader are therefore not yet claimed
as one admitted end-to-end capability.

Unseen elongated diamonds and disconnected dot-pair stimuli also transfer
zero-shot at 94.95–98.14%, tightening the evidence that visual identity is
relational rather than tied to the original rectangles.

Replacement, consolidation, unbounded memory growth, and cross-modality
transfer remain open. See
`experiments/unified_cognitive_controller/README.md`.

The two-decision identify-then-act task requires the agent to:

1. emit an opaque probe action;
2. observe its visible consequence;
3. infer the hidden actuator mapping;
4. observe a target;
5. emit the correct opaque action.

The current fresh predictive learner reached on seed 211:

- 100% held-out accuracy at 64 unique verifier bits;
- 100% accuracy and 100% prediction flips under valid protocol rerenders;
- 100% accuracy and 100% prediction flips under target reversal;
- chance performance when the probe consequence is removed.

An incremental 8→16→32→64-bit learner reached 93.36% with 256 cumulative
optimizer updates. A 32-bit arm with 512 updates failed at 52.73%, so extra
replay does not substitute for the missing unique outcomes.

A subsequent exact three-seed map corrected the robustness claim. At 64 bits,
normal accuracy was 55.47%, 99.61%, and 81.64% for seeds 151, 211, and 307;
only seed 211 passed every causal and anti-fluke gate. Thus 64 bits is the
current single-seed capability frontier, not a robust sample threshold.

Earlier fixed-target weights caused negative transfer to the full task.
Inherited weights are therefore retained only when they improve the next
held-out learning curve.

See:

- `experiments/forward_transfer_attention/SAMPLE_EFFICIENCY_LEDGER.md`
- `experiments/forward_transfer_attention/MICRO_INTERCEPT_DESIGN.md`
- `experiments/forward_transfer_attention/README.md`

## Repository map

| Path | Contents |
|---|---|
| `experiments/unified_cognitive_controller/` | Single-controller few-shot binding, retention, and persistent-memory interface |
| `experiments/forward_transfer_attention/` | Main sample-efficiency, transfer, memory, binding, and causal-audit research |
| `experiments/syllogimous_neural_computer/` | Learned external-memory neural computer |
| `experiments/syllogimous_latent_agent/` | Latent real-time agent and sensory models |
| `experiments/syllogimous_bitter_lesson/` | Emergent reasoning experiments without symbolic solution machinery |
| `experiments/syllogimous_realtime/` | Real-time deterministic syllogism environment and Elisa sources |
| `experiments/sensory_codec/` | Sparse sensory stream experiments |
| `artifacts/checkpoints/` | Curated current checkpoints that are small enough for Git |
| `artifacts/manifests/` | Checksums for curated and excluded historical artifacts |
| `session_records/` | Compact historical reports and continuation notes |
| `legacy/screenwatch_streamer/` | Original sensory-streamer sources, without compiled binaries |

## Setup

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA-capable PyTorch is recommended for training. CPU and Apple unified-memory
backends are useful for tests and tiny diagnostics.

## Verify

```bash
python -m pytest experiments/forward_transfer_attention -q
./scripts/verify_curated_artifacts.sh
```

The narrow current-task regression suite is:

```bash
python -m pytest \
  experiments/forward_transfer_attention/test_identify_then_act.py -q
```

## Reproduce the current 64-bit task

```bash
python -m experiments.forward_transfer_attention.train_identify_then_act \
  --report experiments/forward_transfer_attention/reports/reproduction.json \
  --checkpoint-out artifacts/checkpoints/reproduction.pt \
  --device cuda \
  --seed 211 \
  --curriculum-rung random_probe \
  --intention-width 64 \
  --pretrain-lifetimes 128 \
  --pretrain-steps 40 \
  --policy-lifetimes 64 \
  --test-lifetimes 256 \
  --fit-updates 256 \
  --batch-size 32
```

## Next experiment

Two different 32-bit searches are now complete. The feature-interface winner
fell from 69.53% blind accuracy to 55.47% on its replication seed. A subsequent
learning-mechanism population compared a frozen core, zero-initialized residual
adapters, and conservative action/predictor/recurrent adaptation. Its rank-16
adapter reached 66.41% blind accuracy on seed 211 but also fell to 55.47% on
seed 307, with invalid causal reversal behavior. No checkpoint was promoted.

The cheap "more readout capacity or optimizer freedom" branch is closed at 32
unique outcomes. A subsequent eight-clone reward-free predictive-objective
screen also failed: contrastive refinement reached only 58.59% blind accuracy,
and the unrefined core had the best final selection score. Extra auxiliary
prediction losses therefore do not earn a longer run.

The variance decomposition is complete. Across a nine-horse race at 64 bits,
predictive-core initialization changed the causal floor by 74.22 percentage
points, versus 7.03 points for readout initialization and 5.86 points for
readout replay sampling. All frozen cores passed exact retention checks.

The next sub-minute population should therefore race predictive-core
initializations under identical experience and optimizers, using successive
halving at 32, 48, and 64 outcomes. A winner must then reproduce on a disjoint
lifetime stream and pass old-capability retention before promotion.

That race is now complete. Core seed 263 passed every causal and anti-fluke
gate at 48 and 64 outcomes on a disjoint policy stream with a different
downstream initialization. Its replicated causal floors were 98.05% and
97.27%, respectively. Core seed 211 did not reproduce a stable pass.

The new frontier is therefore a **population-selected, replicated 48-bit
learner**, with search compute accounted separately. Seed 263 is admitted to
the prior-primitive retention/compatibility suite; no general-agent checkpoint
is promoted until that suite passes.

The selected core is now materialized as an immutable 2.9 MB candidate with
SHA-256
`d027b80a631f61c3a9769b60a079494e0a669e1211d3324a13e5ad7b65a1006d`.
Exact reloads reproduce metric-for-metric. With exact complemented negative
controls it passes every gate at 48 and 64 outcomes. A tempting 40-bit point
reaches 95.31% accuracy but fails the missing-evidence uncertainty gate and is
honestly rejected.

Compatibility testing preserves fixed-probe mastery at 16 outcomes and
fixed-target mastery at 48 outcomes, with the predictive core bit-identical
throughout behavioral learning. This establishes a reproducible 25% reduction
from the previous 64-outcome frontier without observed forgetting inside the
identify-then-act family.

The first compounding ladder is also complete. With the immutable core frozen,
novel target-side and observed-effect-side questions each require 8 outcomes,
and their effect-target composition requires 24. The composition replicated on
two disjoint streams while matched-fresh stayed at chance through 64 outcomes.
The gain localizes to the learned vision encoder.

A gradual appearance bridge then changed the palette, object geometry, and
finally both. Stable composition mastery remained 24 outcomes on every rung;
the combined shift replicated with 100% normal/counterfactual accuracy and
100% causal flips. A third-stream retention audit preserved the earlier ladder
at 8/8/16 outcomes. This is verified surface generalization and earlier ability
reuse, not yet broad amodal transfer: spatial relation and event structure are
still shared.

The next bridge replaces position with color identity. It uncovered selective
negative transfer—position-trained vision accelerated observed-effect color
but suppressed target color—so the system retained the useful branch and reset
the harmful one. After acquiring both color primitives from attempted answers
and scalar outcomes, a new relation head reached stable causal mastery from 16
new outcomes on both the selected and blind streams. The identical unacquired
architecture, and either primitive alone, failed through 64 outcomes: a
replicated transfer-ratio lower bound of 4×.

The blind audit reached 100% normal accuracy, 100% accuracy and flips under
both protocol and target rerenders, chance with either fact missing, and 0%
under exact complement controls. Stratified shuffled-label controls produced
no causal pass. The earlier position ladder remained 8/8/24 with bit-identical
cores.

The curated 5.5 MB milestone is
`artifacts/checkpoints/color_primitive_compounder_bits16_seed1901.pt`.

See
`experiments/forward_transfer_attention/ROBUST_SAMPLE_EFFICIENCY_STRATEGY.md`
for the population-search decision and pre-registered diagnostic.

The longer-term optimization is a gradient-trained population with
successive-halving compute allocation. Fitness is held-out learning AULC,
stable bits-to-threshold, retention, latency, and positive transfer to the next
primitive—not old-task accuracy.
