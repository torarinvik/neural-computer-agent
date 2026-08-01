# Script-alignment experimental ladder

All architecture terms in this ladder use the normative
[`../../docs/AMODAL_N_TO_M_ARCHITECTURE.md`](../../docs/AMODAL_N_TO_M_ARCHITECTURE.md)
definition. In particular, the final target permits variable N simultaneous or
asynchronous input encoders and variable M output decoders; the present
vision-only integrated model is a migration source, not an amodal completion.

The project now distinguishes a collection of successful descendants from one
continually growing agent.  A capability counts toward the video-script claim
only when the same immutable checkpoint passes it alongside every previously
admitted capability.

## Gate 0: one-controller baseline

Run `audit_controller_alignment.py` on one checkpoint.  The initial repertoire
is binary/four-rule few-shot binding, cross-appearance relation perception,
persistent disk recall, and span-three/span-five visual working memory.  A
failure is a localization result, not permission to average different models.

## Gate 1: frozen-weight external-memory acquisition

The controller and every adapter are immutable.  An adjacent task may change
only RAM/VRAM state and serializable disk rows.  Admission requires:

- above-chance held-out behavior after one or a few ordinary outcome-bearing
  demonstrations;
- the same behavior after process-style state erasure and disk reload;
- collapse under empty, shuffled, and corrupted memory;
- a valid pixel-level rule reversal with corresponding prediction flips; and
- exact equality of every model parameter before and after acquisition.

The first target is private relation remapping: reuse the mastered visual
same/different primitive while each visual context privately maps that
relation to opaque actions normally or in reverse.

**Gate 1 passed (2026-08-01, non-parametric memory path):** the frozen
same/different controller acquired the private context-to-action convention
from scalar outcomes alone.  A generic content-addressed disk store held
controller-produced event keys and successful opaque-action intentions; the
existing frozen decoder rendered the returned intention.  Three independent
1,024-context audits passed after real save/reload: 98.14%, 98.14%, and 98.34%
normal accuracy; the pixel-identical private-rule reversal reached the same
accuracies with 100% paired prediction flips; no-memory, shuffled-value, and
corrupted-value controls collapsed to approximately chance; and every model
parameter hash was identical before and after.  The complete record is in
`../../session_records/frozen_relation_memory_2026-08-01/`.

This closes frozen-weight acquisition for a generic non-parametric episodic
action-memory baseline.  It does not yet claim that the native recurrent
`retrieved_memory` vector path can compose the new rule, nor that the memory
can replace protocol action intentions with a learned amodal concept.  Those
are the next interface and representation tests.

**Gate 1b passed (2026-08-01, learned memory-intention bridge):** a small
reusable memory-code bridge and intention composer were trained from 256 scalar
support outcomes and the frozen controller's own attempted query action, with
no hand labels or private query answers.  They were then frozen; adaptation
changed only generic content-addressed RAM/disk rows.  Three independent seeds
scored 100.00%, 100.00%, and 99.95% on 2,048 held-out contexts, with 100%
pixel-valid reversal prediction flips.  Empty, shuffled, and corrupted memory
controls were at chance, exact disk retrieval was 100%, and the controller
state digest was unchanged in every run.  Each context used a unique random
visual key independent of the hidden remapping, preventing a key-to-answer
shortcut.  See
`../../session_records/memory_intention_bridge_2026-08-01/`.

This qualifies a learned amodal memory-code/intention boundary, but not the
controller's native `retrieved_memory` path: the diagnostic still uses a fixed
visual key encoder plus an external bridge and composer.  The next interface
test is to migrate this boundary onto the standardized event/intention buses
without introducing a task-specific reasoning branch.

## Gate 2: compounding ledger

Compare the experienced agent with a fresh matched learner on the next
adjacent primitive.  Record unique verifier outcomes and the first threshold
that remains passed at every later prefix.  Architecture reuse alone does not
count: inherited memory must reduce stable bits-to-threshold without violating
retention.

## Gate 3: extracted neural IR and decoder fan-out

The vision encoder and actuator are externally owned and independently
checkpointed. The refactored one-event/one-action path must be bit-identical to
the current integrated model before any new modality is trained.

**Gate 3a passed (2026-08-01):** the encoder, controller core, and decoder now
have disjoint ownership and separate state dicts. The real five-capability
checkpoint is bit-identical through the extracted path over 64 held-out
lifetimes, including zero maximum logit difference and exact reconstruction of
all 66 source tensors. A two-coordinate legacy action-residual suffix remains
explicitly marked as migration debt.

**Gate 3b passed (2026-08-01):** the legacy two-action residual was folded
algebraically into a 24-dimensional intention residual. It used no examples,
outcomes, labels, or optimizer updates. Across 12,288 paired decisions there
were no action flips and maximum logit drift was `5.72e-6`; all five repertoire
gates passed at 4,096 held-out lifetimes. The compatibility suffix is now
structurally zero in the promoted checkpoint.

**Gate 3c passed (2026-08-01):** a second, independently calibrated opaque
protocol decoder consumes the same frozen 24-dimensional intention as the
inherited decoder. Three independent seeds crossed after 64 verifier bits and
passed the five-capability closed-loop audit at 512 lifetimes; the promoted
seed passed at 4,096. Reward-shuffled, intention-shuffled, and zero-intention
controls failed. A runtime-variable output bus exercised zero, one, and both
decoders, with the inherited output bit-exact. See
`../../session_records/amodal_output_fanout_2026-08-01/`.

**Gate 4a passed (2026-08-01):** a runtime-variable event collection and a
generic permutation-invariant set bus preserve N=1 and identical duplicates
bit-for-bit. A 4,817-parameter set residual learned complementary N=2 relation
composition from attempted actions and scalar outcomes while the controller
and all adapters remained frozen. Three seeds crossed after 768–1,344 verifier
bits. The promoted bus scored 96.46% on bars while either stream alone remained
near chance, and transferred above 90% to unseen diamonds and dot pairs.
Shuffled partners stayed at chance and contradictory partners causally reversed
predictions. See `../../session_records/amodal_input_composition_2026-08-01/`.

**Gate 4b-synchronous passed (2026-08-01):** `AmodalEventTimeline` sorts
timestamped events independently of arrival order and groups bounded jitter.
At 4,096 lifetimes, out-of-order and 0.25-unit-jitter delivery were both
96.36% and action-identical to synchronous N=2. Mismatched timestamps remained
two separate windows. This is transport alignment, not learned delay policy.

**Gate 4b-next:** replicate cross-renderer transfer, then qualify noisy and
missing streams plus a learned latency-versus-wait policy without breaking exact
N=1.

One latent intention is connected to independently trained output adapters
(opaque action, integer/bit code, and audio event).  Replacing or permuting an
adapter may require adapting that thin decoder, but must not require relearning
the controller-side concept. Multiple decoders must be able to consume the same
intention simultaneously.

## Gate 4: variable N-input composition

The frozen controller accepts a variable-size event collection with no fixed
modality slots. Aligned redundant streams should improve sample efficiency;
shuffled, delayed, missing, duplicated, and noisy streams must behave
predictably. A complementary-evidence task must require two encoders so that
neither frontend can solve it alone.

## Gate 5: new frontend/backend qualification

A previously unseen encoder and decoder are trained as neural-IR adapters while
the controller remains frozen. Existing cognitive skills must become available
through the new input and output without relearning those skills in the core.

## Standing rules

- The learner sees pixels, its opaque attempted actions, scalar outcomes, and
  its own latent/memory state only.
- Verifier metadata can score and audit but never enters deployed inference.
- Start below one minute; promote only after a causal or mechanistic signal.
- Rehearse the complete admitted repertoire during any weight-changing run.
- Do not add a permanent task-specific branch when a generic memory operation
  or shared adapter can express the capability.
