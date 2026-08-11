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

The current v5 harness additionally keeps provisional candidates attached by
an opaque context-continuity proposal when strict prediction matching is
temporarily too narrow, and verifies target candidates across affine and
random-feature replay-free model families. This reduces provisional capacity
loss but has not improved the six-seed promotion rate beyond `2/6`; it remains
an engineering boundary rather than a capability claim. Its accounting is in
`session_records/context_continuity_mixed_transition_2026-08-11/sample_efficiency_ledger.json`.

The v6 harness supports the causal
`recency_weighted_and_latest_v1` event-window state contract with gain `0.05`
and decay `0.75`. Explicitly selecting that mode completes `5/12` runs on
seeds 90–101, compared with `3/12` for the compatibility mean/max contract;
every passing run recovers the post-promotion route and beats its fresh
challenger. The compatibility mean/max mode remains the smoke-test default so
existing checkpoint and audit behavior stays stable. Seven recency runs still
reject, so this remains a bounded transfer result rather than general
continual learning. The full ledger is in
`session_records/recency_window_transition_2026-08-11/sample_efficiency_ledger.json`.

The discovery harness now uses a write firewall during target acquisition:
provisional candidates continue to learn one-pass evidence, while source slots
remain read-only until promotion. With recency/latest state, seeds 80–103
complete `7/24` runs and retain the source slot in `24/24`; eight candidates
pass promotion but one fails later route recovery. This is an improved bounded
memory boundary, not general continual learning. The full accounting is in
`session_records/recency_window_isolated_transition_2026-08-11/sample_efficiency_ledger.json`.

The v8 continuation seam keeps the promoted opaque slot bound for the first
post-promotion recovery lifetime and permits only a bounded caller-owned
continuation tolerance. It does not relax global matching or write to committed
memory. The same 24 seeds complete `8/24` runs: all eight promoted routes
recover, all eight beat a matched fresh challenger, and source retention stays
`24/24`, with zero replay and zero optimizer updates. This is a bounded route
recovery improvement, not general continual learning. The full ledger is in
`session_records/recency_window_preferred_recovery_2026-08-11/sample_efficiency_ledger.json`.

The downstream destination-composition rung can be run with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.goal_conditioned_planning
```

It freezes the controller and decoder, learns a replay-free factual transition
slot from rendered lifetimes, admits a learned state as an opaque goal file,
and compares two-step goal-conditioned search with a matched fresh factual
slot. Seed `93` measured `0.00360` trained terminal error versus `0.04376`
fresh, with zero replay and zero optimizer updates. This qualifies downstream
use of a goal fragment, not end-task mastery or general continual learning; the
next pressure test is a multi-step, structurally diverse acquisition curve. Seeds
`91`, `92`, and `93` all passed the bounded rung; the ledger is in
`session_records/goal_conditioned_planning_2026-08-11/sample_efficiency_ledger.json`.

The source-retention/target-acquisition rung can be run with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.nonstationary_goal_conditioned_planning
```

It learns rendered n-back-2 family A, then n-back-3 family B in an isolated
external factual slot, admits B's opaque goal file, and compares two-step
search with a matched fresh target slot while checking A's byte stability.
Seeds `91`, `92`, and `93` all pass this bounded rung; the ledger is in
`session_records/nonstationary_goal_conditioned_planning_2026-08-11/sample_efficiency_ledger.json`.

The online version adds the goal gate after opaque target-context discovery:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.replay_free_transition_acquisition \
  --audit online-discovery --seed 91 --window-statistics recency_weighted_and_latest_v1 \
  --window-gain 0.05 --goal-conditioned --prior-selection-fresh-cost 1.0 \
  --prior-selection-cost-weight 0.2
```

Across seeds `80–103`, `9/24` runs completed route discovery, source
retention, target goal admission/use, and the matched fresh-goal challenger
with a cost-aware prior receipt; the other fifteen were rejected by the
complete gate. The ledger is in
`session_records/online_goal_conditioned_discovery_2026-08-11/sample_efficiency_ledger.json`.

To exercise the replaceable memory-side acquisition policy instead of static
caller costs, add `--learned-prior-selection-cost`. The router then persists a
shared opaque cost ledger and updates it only after verified promotion; this is
a wiring/control mode until held-out multi-family acquisition shows a gain.
The current seed-91–93 smoke passed `1/3`; the two rejected runs failed before
ledger observation, so target discovery/promotion stability remains the active
bottleneck.
The optional `--adaptive-address` flag exercises copy-on-write key adaptation;
its candidate key is resolved from the post-holdout bank before retention
verification, preserving historical addresses without changing the controller.

The promotion verifier now evaluates the three independent recursive holdouts
as a conservative aggregate: every candidate rollout must remain below the
absolute error bound, the candidate must win a majority of fresh comparisons,
and its mean error must be lower than the matched fresh challenger. This avoids
letting one noisy lifetime veto an otherwise retained capability without
removing the catastrophic-regression bound. On the same seeds `80–103`, the
complete gate rose from `9/24` to `10/24`; source retention stayed `24/24`,
with `624` unique verifier bits, `504` transition rows consumed once, zero
replay, zero optimizer updates, and an unchanged controller. This is a bounded
promotion-stability gain, not general continual learning. The ledger is in
`session_records/online_goal_conditioned_discovery_aggregate_retention_2026-08-11/sample_efficiency_ledger.json`.

The external context encoder also exposes a copy-on-write contrastive update
contract that consumes one fresh paired-view batch and performs exactly one
optimizer update without retaining rows. One-pass pretraining did not improve
the current discovery pass rate, so it remains a reusable memory-side seam and
is not a promoted capability claim. Random-feature widths `128–1024` likewise
did not change the failing seeds; staged evidence coherence, not model width,
is the active bottleneck.

The default online audit now separates novelty routing from promotion fit:
committed-slot routing uses a tighter `0.02` match tolerance, while promotion
keeps the original `0.05` held-out prediction threshold. This makes the router
stage a candidate when evidence is novel without weakening the commit gate.
On the same seeds `80–103`, the complete gate rose from `10/24` to `14/24`;
source retention remained `24/24`, and all complete runs passed route recovery,
goal admission/use, and the fresh-goal challenger. The tolerance sweep plateaued
at `14/24` from `0.02` through `0.005`, so this is not a single knife-edge.
The ledger is in
`session_records/online_goal_conditioned_discovery_routing_threshold_2026-08-11/sample_efficiency_ledger.json`.

The factored frozen-base pressure audit can be run with:

```bash
PYTHONPATH=src:. uv run python -m experiments.brainworkshop_canonical.factored_residual_base_pressure
```

It trains a replay-free affine source base, freezes it with the controller,
then learns a novel opaque target through a context-local random-feature
residual. Across seeds `91`, `92`, `93`, `95`, `99`, `100`, `101`, `102`, and
`103`, all `9/9` candidates staged, passed promotion, beat the frozen-base
challenger, retained the source, and round-tripped the replaceable base. The
audit used `189` unique verifier bits, `216` one-pass transition rows, zero
replay, and zero optimizer updates. This is a bounded factored external-memory
gain, not general continual learning; the ledger is in
`session_records/factored_residual_base_pressure_2026-08-11/sample_efficiency_ledger.json`.

The stricter multi-regime sequence audit can be run with:

```bash
PYTHONPATH=src:. uv run python -m experiments.brainworkshop_canonical.factored_residual_sequence_pressure
```

It uses ten-step lifetimes, three sequential opaque regimes, independent
recursive holdouts, analytic copy-on-write ridge selection, reversal, partial
evidence, and checksum-corruption controls. The current three-seed gate is
intentionally rejected: `8/9` target slots promote, but one seed fails to
stage its third regime and close partial evidence remains ambiguous. This is
the correct safe failure, not a reason to force a route. The ledger is in
`session_records/factored_residual_sequence_pressure_2026-08-11/sample_efficiency_ledger.json`.

For the explicit cross-prefix identity-confirmation diagnostic, add
`--missing-evidence-stable-confirmation`. It permits close per-prefix factual
margins only when the same slot remains the winner across the confirmation
sequence; contradictions and slot flips still refuse. The six-seed matched
diagnostic is archived under
`session_records/factored_stable_identity_confirmation_2026-08-11/` and is not
promoted because the fresh-seed replication was tied.

The causal external working-memory transfer audit can be run with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.causal_working_memory_transfer \
  --seed 17 --report-out /tmp/causal-working-memory-transfer-17.json
```

It trains one replaceable working-memory codec on fresh n-back-2 lifetimes,
freezes it, and evaluates fresh external state through the canonical causal
read-before-write loop. Seeds `17` and `18` both reach `1.0` n-back-2
fresh-state accuracy versus `0.5` matched fresh controls, while shuffled
outcomes and history reset remain near chance. The n-back-3 probe stays near
chance, so this promotes causal memory-state transfer but not longer-rule
generalization. Evidence is archived in
`session_records/brainworkshop_causal_working_memory_transfer_2026-08-11/`.

## Causal protected external rule growth (2026-08-11)

The next pressure test is implemented in `causal_rule_growth.py`. It trains a
source n-back-2 working-memory cell, appends a separate n-back-3
`ExternalWorkingMemoryCell`, freezes the source cell and controller, and
learns the target route from rendered cue events. The target cell is the only
new trainable memory-side capacity. The audit also checks exact route-state
reload with the compatible learned-event encoder and tests that reversal
evidence on a copied route table cannot mutate the live route table.

Seeds `17` and `18` both pass complete-prefix retention, new-rule mastery,
unchanged controller/source-codec digests, cue-conditioned route separation,
reload, reversal, and zero replay. The shuffled-cue control selects the target
slot only `0.5398` and `0.5540` of the time. This promotes bounded protected
rule growth, not general continual learning: route state is keyed by a
versioned learned-event representation, and arbitrary rule induction,
unrestricted memory growth, and compression remain open.

The full reports and accounting ledger are archived in
`session_records/brainworkshop_causal_rule_growth_2026-08-11/`.

## Causal repeated depth growth (2026-08-11)

`causal_depth_growth.py` extends the protected-file audit through three
external working-memory files: n-back-2, n-back-3, and n-back-4. The shared
controller, event encoder, and every earlier cell/adapter/decoder are frozen
before the next file is acquired. All three rendered cues route to the correct
opaque slot, while shuffled cues fail to target the intended slot.

Seeds `17` and `18` both retain every rule at `1.0000` across eight fresh
lifetime probes. Both pass exact route reload, explicit rejection of a route
payload paired with an incompatible learned-event encoder, copied-table
reversal, protected-prefix digests, and zero replay. The full evidence and
ledger are archived in
`session_records/brainworkshop_causal_depth_growth_2026-08-11/`.

This promotes repeated bounded rule growth, not general continual learning.
The next pressure test must vary the rule family and cue representation on
held-out lifetimes; fixed n-back depth plus a fixed rendered cue family is not
yet arbitrary rule acquisition.

## Held-out external rule growth (2026-08-11)

`heldout_rule_growth.py` extends the same boundary one step further. It trains
an n-back-5 external file under rendered cue `7`, withholds cue `8` from the
route ledger, and then discovers the correct opaque file from scalar outcomes.
Seeds `17` and `18` both retain the n-back-2/3/4 prefix and the new n-back-5
file at `1.0000`, recover the held-out route at `1.0000`, preserve the
controller and frontend digests, reload the route state exactly, reject an
incompatible learned-event representation, and use zero replayed examples.
This is bounded outcome-only route discovery, not arbitrary new computation,
unrestricted memory growth, or general continual learning.

Run it with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.heldout_rule_growth \
  --source-updates 64 --target-updates 256 --batch-size 32 --steps 14 \
  --calibration-lifetimes 8 --discovery-lifetimes 8 --retention-lifetimes 8 \
  --seed 17 --report-out /tmp/brainworkshop-heldout-rule-growth.json
```

The authoritative reports and sample-efficiency ledger are archived in
`session_records/brainworkshop_heldout_rule_growth_2026-08-11/`.

## Cross-family rule growth and route hysteresis (2026-08-11)

`cross_family_rule_growth.py` varies the private verifier family while keeping
the learner-facing protocol fixed. It grows isolated external files for
n-back-2, pair parity, adjacent switching, and single-symbol parity. The final
family is trained under cue `7`, then cue `8` is introduced only after
training; the controller receives rendered events, opaque actions, and scalar
outcomes, never the private family or target labels.

Seeds `17` and `18` both pass complete-prefix retention, new-family mastery,
frozen controller and event encoder digests, held-out outcome-only route
discovery, shuffled-cue controls, exact route reload, incompatible event
representation rejection, and zero replay. Held-out recovery reached
`0.9978/0.9598` accuracy with `1.0000/1.0000` target-slot selection. The
lowest retained primitive was `0.8594` on seed `18`; all other retained
families reached `1.0000`.

The audit exposed and fixes an important memory-side bottleneck: immediate
fallback on one noisy outcome can demote an otherwise competent route. Route
discovery uses failure patience `1` to gather evidence; exploitation uses
patience `4` before falling back. This changes only external route policy, not
the frozen controller, event encoder, or capability files.

Each seed used `832` optimizer updates, `344,064` training verifier bits,
`59,648` audit bits, and zero replayed examples. This promotes cross-family
outcome-only route discovery over bounded external rule growth. It still does
not establish arbitrary new computation, unrestricted memory growth,
compression, or general continual learning. Evidence is archived in
`session_records/brainworkshop_cross_family_rule_growth_2026-08-11/`.

Reproduce the promoted gate with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.cross_family_rule_growth \
  --source-updates 64 --target-updates 256 --batch-size 32 --steps 14 \
  --calibration-lifetimes 32 --discovery-lifetimes 32 --retention-lifetimes 4 \
  --seed 17 --report-out /tmp/brainworkshop-cross-family-17.json
```

## Replay-free factual transition acquisition (2026-08-11)

`replay_free_transition_acquisition.py` is the rendered “CPU/files” pressure
test. The controller, frontend, and keypress decoder are frozen; each opaque
transition row is consumed once by an external factual bank, and a planner
derives behavior from an external goal rather than storing a task policy.

The default short rung improves held-out recursive transition error from
`0.06614` to `0.02737` with `18` one-pass rows, zero replay, zero optimizer
updates, and an unchanged controller. The two-family rung retains the source
slot byte-for-byte while the target improves from `0.04522` fresh error to
`0.01334`.

The audit also supports the opt-in
`ordered_payload_and_presence_v1` state adapter, which preserves the bounded
learned event-token order and empty positions. It improved the matched
eight-seed raw candidate-admission rate from `3/8` to `5/8` and the complete
goal-conditioned promotion rate from `3/8` to `4/8` when paired with factual
routing tolerance `0.005`. This is retained as a qualified representation
boundary, not promoted as general continual learning: routing calibration and
reliable held-out candidate retention remain unresolved.

Example:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  -m experiments.brainworkshop_canonical.replay_free_transition_acquisition \
  --audit online-discovery --goal-conditioned \
  --window-statistics ordered_payload_and_presence_v1 \
  --routing-match-tolerance 0.005 --seed 93 \
--report-out /tmp/brainworkshop-transition-discovery-93.json
```

## Held-out rule-family frontier: triplet parity (2026-08-11)

`cross_family_rule_growth.py` now accepts the target rule family and cue as
arguments, so the cross-family audit is no longer hard-coded to one target.
The learner-facing protocol remains unchanged: rendered symbol events, opaque
keypress actions/feedback, and scalar verifier outcomes only. The target
family is verifier-private orchestration, not controller metadata.

The first genuinely held-out target, `triplet_parity`, exposes the next real
boundary. With the existing four-family architecture and 256 target updates,
forced target execution reached only `63.4--66.8%`; 512 updates improved this
to `68.2--73.0%` but still missed the `80%` mastery gate. Prior n-back-2,
pair-parity, and switching files stayed at `100%`, and their prefix digests,
controller, and encoder remained unchanged. Held-out cue routing did not
recover because the target file itself was not mastered.

A same-cue curriculum warmup on `parity2` before `triplet_parity` was worse,
ending at `48.6--54.8%`. Reusing one mutable external file across changing
rules causes interference; it is not compositional learning. The audit also
corrects the `prefix_retention` gate so it measures only protected prior
families rather than accidentally including the new target.

This rejects further route tuning as the immediate answer. The next high-ROI
architecture task is a generic executable/compositional external capability
that can acquire a new temporal rule from scalar outcomes, while retaining
the protected family files. Evidence and accounting are in
`session_records/brainworkshop_cross_family_triplet_frontier_2026-08-11/`.

## External compute-file growth: triplet parity (2026-08-11)

`external_compute_growth.py` closes the computation-acquisition gap with the
generic external register substrate. The frozen controller and event encoder
feed standardized learned events into an opaque instruction plus an
append-only event-window compute basis. The source `symbol_parity` file is
mastered first; its basis, instruction, adapter, and decoder are then frozen
while a fresh `triplet_parity` file learns from scalar verifier outcomes only.

The new `event_window_only` basis mode is important: a newly appended file can
read its own persistent event window without depending on the previous file's
hidden register distribution. That is an ABI-level isolation mechanism, not a
triplet-specific branch. Across seeds `17` and `18`, source and target
retention both passed the `0.80` stable-prefix gate (`1.0000` source; target
`0.8736` and `1.0000`), the source file was byte-stable, the frozen controller
and frontend digests were unchanged, and resetting the external history
dropped target performance to chance. The matched fresh-file comparator was
`0.8786` and `1.0000`, so inherited state is not being claimed as a universal
sample-efficiency gain; the promoted result is the isolated, append-only
blueprint and its retention boundary.

Each seed consumed `176,128` unique training verifier bits, performed `448`
optimizer updates, replayed zero examples, and used an explicit file slot so
route discovery could not hide computation failure. This promotes one
outcome-only new rendered computation with frozen core and no source replay;
it does not establish arbitrary program induction, route discovery, unrestricted
memory growth, or general continual learning. Reports are archived in
`session_records/brainworkshop_external_compute_growth_promoted_2026-08-11/`.

Run the promoted rung with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.external_compute_growth \
  --source-updates 192 --target-updates 256 --fresh-updates 256 \
  --batch-size 32 --steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-external-compute-growth.json
```
