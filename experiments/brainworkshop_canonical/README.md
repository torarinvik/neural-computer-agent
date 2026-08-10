# Canonical Brain Workshop runtime boundary

This experiment is the production-runtime composition smoke test for the next
continual-learning rung:

```text
symbol frontend -> amodal event bus -> one controller/memory
                -> opaque intention -> keypress decoder
                -> opaque keypress feedback -> episodic context/retention state
```

The verifier privately holds the n-back comparison target. The learner receives
only learned event tensors, its own opaque keypress feedback, and deterministic
scalar outcomes. `KeypressDecoder` owns key-index lowering and propensity
accounting; `CapabilityRetentionLedger` receives only an opaque episodic
context key and the scalar episode score.

Run the short composition check with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.train \
  --n-back 4 --batch-size 4 --report-out /tmp/brainworkshop-canonical-smoke.json
```

The `--updates` option runs the bounded reward-only external-reader pilot. It
freezes the controller and frontend, uses fresh lifetimes, and reports fresh,
time-shuffle, and history-reset controls. This establishes a causal signal but
does not by itself claim Brain Workshop mastery or catastrophic-forgetting
resistance. Promotion still requires the complete retention ladder, reversal,
and grow-when-full controls.

The `relation` reader is the current promoted narrow mechanism for this rung:
it uses a bounded external event window with learned content-and-age attention
and reached `0.9375/1.0000/1.0000` fresh accuracy across three seeds while the
time-shuffle and history-reset controls stayed near chance. Acquisition and
retention accounting are separate: training and controls do not write the
retention ledger, while an explicit post-acquisition audit can protect a
candidate. This is still bounded n-back-2 capability, not general continual
learning.

The sequential growth runner then appends a new relation reader, adapter, and
keypress decoder as an isolated external slot. Controlled slot exploration is
propensity-accounted and starts only after learner-visible feedback; the old
slot is otherwise advanced by its own scalar failure. The replicated
n-back-2-to-n-back-3 rung preserved the old slot at `1.0000` and acquired the
new slot at `0.8021` or better across three seeds, with chance-level causal
controls, protected old/new retention records, an unchanged old-state hash,
and zero replay. This is a bounded two-slot result; growth beyond two slots,
reversal, eviction, persistence, and general continual learning remain open.

Run the sequential audit with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.sequential_train \
  --old-n-back 2 --new-n-back 3 --old-updates 128 --new-updates 256 \
  --batch-size 32 --steps 8 --slot-exploration 0.25 \
  --report-out /tmp/brainworkshop-sequential.json
```

The three-slot ladder is not promoted: forced candidate audits showed every
reader at `1.0000`, but dynamic fresh routing fell to `0.5703` on n-back-4.
The next high-ROI task is an outcome-trained opaque route learner that can
discover a new slot while retaining existing slots.

The first proactive route-head implementation is retained only as rejected
infrastructure: it improved n-back-4 discovery to `0.8828` but collapsed the
old n-back-2/3 routes to `0.6771/0.5167`. Any promoted route learner must have
an explicit old-route preservation gate.

A 12-step lifetime control raised safe dynamic n-back-4 routing to `0.7513`
but still missed the stable gate. This points to persistent route inference
across short lifetimes as the next bottleneck.

The persistent-route implementation now keeps opaque outcome evidence outside
the controller and promotes a slot only after a stable eight-observation
prefix. The eight-step ladder remains rejected at `1.0000/0.8042/0.5703` for
n-back-2/3/4: without an observable task cue, route discovery necessarily
costs early scored trials. A replicated 16-step ladder passes bounded-growth
gates across seeds 17, 18, and 19 at `1.0000/0.9231/0.8333`, with chance-level
shuffle controls, perfect forced candidate audits, unchanged prior slots,
frozen controller state, and zero replay. This is promoted only as
long-lifetime bounded external growth; it is not general continual learning.

Run the promoted long-lifetime audit with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.capacity_ladder \
  --first-updates 96 --second-updates 192 --third-updates 192 \
  --batch-size 32 --steps 16 --slot-exploration 0.25 \
  --persistent-route --seed 17 \
  --report-out /tmp/brainworkshop-capacity-persistent.json
```

The remaining blocker is route identifiability on short lifetimes, not the
reader's candidate computation or retention isolation. A future cue-conditioned
route memory must use a learned event/context key and scalar outcomes only; a
handwritten n-back selector would invalidate the pressure test.

That route memory is now implemented as
`PersistentOpaqueContextRouteEvidence`. It indexes independent opaque scalar
ledgers by learned event keys. Training can read the table, but only explicit
forced candidate audits write persistent route evidence, so exploratory
failures cannot permanently poison promotion.

The cue-conditioned short-lifetime audit uses an ordinary rendered cue token
through the event encoder. Across seeds 17, 18, and 19, 8-step fresh accuracy
was `1.0000/1.0000/1.0000` for n-back-2/3/4; time-shuffle and history-reset
controls stayed near chance, cue-shuffled controls separated by at least 0.1,
all candidate audits were perfect, prior slots were unchanged, and replay was
zero. The cue-absent control remains `1.0000/0.80/0.57--0.58`, so this is a
promoted cue-conditioned bounded route result, not general continual learning.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.context_route \
  --first-updates 64 --second-updates 256 --third-updates 256 \
  --batch-size 32 --steps 8 --slot-exploration 0.5 --seed 17 \
  --report-out /tmp/brainworkshop-context-route.json
```

The route ledger now also has reversal hysteresis and a reloadable external
state payload. Those mechanisms pass unit and round-trip tests; a full
nonstationary cue-reversal ladder now has a promoted same-cue replacement
rung. The subsequent failure-only demotion rung is also promoted: it records
grouped scalar outcomes from fresh changed-task lifetimes and allows a stale
protected route to retire without same-cue calibration.

Run the replacement audit with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.context_route_reversal \
  --old-updates 64 --new-updates 256 --batch-size 32 --steps 8 \
  --slot-exploration 0.5 --seed 17 \
  --report-out /tmp/brainworkshop-context-route-reversal.json
```

Run the failure-only demotion audit with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.context_route_failure_demotion \
  --old-updates 64 --new-updates 256 --transition-batches 12 \
  --batch-size 32 --steps 8 --slot-exploration 0.5 --seed 17 \
  --report-out /tmp/brainworkshop-context-route-failure-demotion.json
```

The promoted result is bounded failure-driven route memory. It does not
establish unrestricted memory growth, arbitrary new computation, or general
continual learning.

The next promoted rung removes the task horizon from appended capability
provisioning. `train_adaptive_relation_capability()` gives each slot only a
fixed external event-window capacity; the benchmark horizon remains private to
the verifier harness. The adaptive reader evaluates candidate relations before
mixing them, avoiding the extra-history blur of the pooled reader.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.adaptive_capability_growth \
  --old-updates 64 --adaptive-updates 256 --batch-size 32 --steps 8 \
  --calibration-lifetimes 8 --slot-exploration 0.5 --seed 17 \
  --report-out /tmp/brainworkshop-adaptive-capability-growth.json
```

Seeds 17, 18, and 19 all passed the n-back-2/3/4 fresh, causal-control,
retention, reload, frozen-core, and zero-replay gates. This is bounded generic
capability growth, not arbitrary computation or general continual learning.

The automatic-discovery rung removes the forced calibration transaction for a
new cue. Ordinary fallback episodes write grouped scalar evidence until the
new slot becomes protected:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.automatic_route_discovery \
  --old-updates 64 --new-updates 256 --discovery-batches 12 \
  --batch-size 32 --steps 8 --memory-capacity 5 --slot-exploration 0.5 \
  --seed 17 --report-out /tmp/brainworkshop-automatic-route-discovery.json
```

Seeds 17, 18, and 19 all passed automatic promotion, old retention, cue
controls, route reload, frozen-core, prior-bank, and zero-replay gates with
the new-cue calibration flag set to false.

The failure-triggered capacity-growth audit tests the next boundary. A
capacity-five generic adaptive reader is trained on a new capability; fresh
opaque failures below the `0.8` stable mastery threshold trigger replacement
of that unmastered reader with the same blueprint at capacity six. The old
capability remains isolated and mastered, while ordinary post-growth fallback
outcomes discover the new route without cue calibration:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.capacity_growth_from_failure \
  --old-updates 64 --initial-updates 256 --continued-updates 512 \
  --probe-batches 4 --discovery-batches 12 --batch-size 32 --steps 10 \
  --initial-capacity 5 --grown-capacity 6 --slot-exploration 0.5 \
  --failure-threshold 0.8 --seed 17 \
  --report-out /tmp/brainworkshop-capacity-growth-from-failure.json
```

Seeds 17, 18, and 19 pass fresh mastery, old retention, causal controls,
route reload, frozen-core, prior-bank, and zero-replay gates. This promotes
bounded external capacity growth only; unrestricted memory growth and general
continual learning remain unqualified.

The recursive audit appends a second capability after the first capacity
growth. It tests capacity `5 -> 6` followed by `6 -> 7`, with the full failed
external slot reset at each transaction so a damaged adapter or decoder cannot
contaminate the replacement:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.recursive_capacity_growth \
  --old-updates 64 --initial-updates 256 --continued-updates 512 \
  --probe-batches 8 --discovery-batches 12 --batch-size 32 --steps 11 \
  --first-initial-capacity 5 --first-grown-capacity 6 \
  --second-initial-capacity 6 --second-grown-capacity 7 \
  --failure-threshold 0.8 --seed 17 \
  --report-out /tmp/brainworkshop-recursive-capacity-growth.json
```

Seeds 17, 18, and 19 pass both failure triggers, fresh mastery, retention of
both prior capabilities, automatic route discovery, causal controls, three-cue
route reload, frozen-core, prior-state, and zero-replay gates. This promotes
recursive bounded growth only; unrestricted continual learning remains open.

The bounded-bank lifecycle audit then tests safe replacement rather than
append-only growth:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.protected_eviction_growth \
  --old-updates 64 --first-initial-updates 256 --first-continued-updates 512 \
  --candidate-updates 64 --replacement-updates 512 --probe-batches 8 \
  --discovery-batches 12 --batch-size 32 --steps 12 \
  --first-initial-capacity 5 --first-grown-capacity 6 \
  --candidate-capacity 7 --replacement-capacity 8 \
  --failure-threshold 0.8 --seed 17 \
  --report-out /tmp/brainworkshop-protected-eviction-growth.json
```

Seeds 17, 18, and 19 pass protected-slot masking, fresh candidate failure,
stale-route reset, replacement mastery, old retention, causal controls,
route reload, frozen-core, prior-bank, full-bank refusal, and zero-replay
gates. This promotes retention-safe bounded eviction and slot reuse only.
The current selector is an opaque outcome-derived score with a retention mask,
not learned general utility or consolidation; unbounded memory growth,
arbitrary new computation, and general continual learning remain open.

The learned utility audit removes the direct outcome-history score from the
selector. `ExternalCapabilityEvictionPolicy` receives only an incoming learned
event tensor and opaque capability-address tensors; fresh scalar verifier
outcomes train its utility ranking outside the frozen controller. Candidate
identities and physical slots are independently permuted, and the incoming
task context alternates between the two candidate capabilities:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.learned_eviction \
  --train-worlds 4 --eval-worlds 8 --policy-steps-per-world 24 \
  --old-updates 64 --candidate-updates 256 --replacement-updates 512 \
  --replacement-learning-rate 3e-3 --probe-batches 1 --base-audits 3 \
  --batch-size 32 --steps 11 --seed 17 \
  --report-out /tmp/brainworkshop-learned-eviction.json
```

Across seeds 17, 18, and 19, learned selection was `1.000` in every run;
reward-shuffled and corrupted-feature controls were `0.500` in every run;
replacement fresh accuracy was `0.919/1.000/1.000`; retained capability and
base retention were `1.000` in every run; stale route state was cleared; the
controller was frozen; and replay was zero. Each run used `945,792` unique
verifier bits, `236,288` logical lifetimes, `2,912` optimizer updates, and
`2,540,544` verifier outcome events.

This promotes a narrow learned context-conditioned utility policy for a
bounded opaque capability bank. The lower replacement learning rate is part of
the write-stability protocol: the rejected `1e-2` schedule passed selection
but failed fresh replacement on two seeds. Persistent consolidation,
unbounded growth, arbitrary new computation, and general continual learning
remain unqualified. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/learned_eviction_context_seed17.json`,
`...seed18.json`, and `...seed19.json`.

The replay-free transition acquisition module also exposes the online
discovery rung:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.replay_free_transition_acquisition \
  --audit online-discovery --seed 93 --steps 6 \
  --source-training-lifetimes 2 --target-training-lifetimes 2 \
  --report-out /tmp/brainworkshop-online-transition-discovery.json
```

This starts with only a source factual slot, stages a novel rendered target
slot from opaque transition evidence, and commits it only after held-out,
recursive, source-retention, and matched fresh-candidate gates. It checks
one-pass learning, route recovery, and source retention. The policy-free
runtime can use `ExternalControllerEventWindowStateAdapter` to retain bounded
event-window statistics at the historical planner width, and the promotion
call passes three independent recursive held-out rollouts into the atomic
commit gate before a post-promotion route lifetime. It is deliberately a
boundary audit rather than a general continual-learning claim; the seed
ledger is in
`session_records/multi_lifetime_transition_promotion_2026-08-11/sample_efficiency_ledger.json`.
