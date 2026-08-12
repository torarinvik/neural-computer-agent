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

## Content-addressed external compute-file route discovery (2026-08-11)

`external_compute_route.py` removes the explicit file selection. It first
calibrates and protects the `symbol_parity` file under rendered cue `7`, then
appends the `triplet_parity` file under cue `8`. A
`PersistentOpaqueContextRouteEvidence` table learns one independent scalar
route ledger per learned event key. Unknown cue `9` falls back to append order,
so an unrecognized context cannot activate the newest file through accidental
linear generalization.

Seeds `17` and `18` both pass direct file mastery, routed source/target
mastery, exact source/target slot selection, unseen-cue fallback, no-file
chance control, route reload, protected source context, immutable files,
frozen controller/frontend, and zero replay. Routed target accuracy was
`0.8672` and `1.0000`; unseen-cue accuracy remained near chance at
`0.5014` and `0.4964`. This promotes cue-conditioned outcome-only route
discovery over isolated generic compute files, not arbitrary program
induction, unrestricted memory growth, or general continual learning.

Each seed used `269,824` training verifier bits, `9,216` audit bits, `448`
optimizer updates for file acquisition, `264` external route-memory updates,
and zero replayed examples. Evidence is archived in
`session_records/brainworkshop_external_compute_route_promoted_2026-08-11/`.

Run it with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.external_compute_route \
  --source-updates 192 --target-updates 256 --route-updates 256 \
  --route-calibration-lifetimes 8 --batch-size 32 --retention-lifetimes 4 \
  --seed 17 --report-out /tmp/brainworkshop-external-compute-route.json
```

## Four-file content-addressed route bank (2026-08-11)

`external_compute_route_bank.py` generalizes the route ledger and executable
file builder from two slots to an append-only four-file bank. The files cover
`symbol_parity`, `triplet_parity`, `parity2`, and a balanced binary-symbol
`switch_binary` family. Each file is acquired in isolation from scalar
outcomes, then frozen before the next file is added. The controller, event
frontend, and generic register interpreter remain frozen while the route table
learns each opaque file address from the learned event key.

Seeds `17` and `18` both pass stable direct and routed mastery for all four
files, exact correct-file selection (`1.0000` for every known context),
unseen-context append-order fallback, no-file chance control, exact route
reload, byte-identical prior files, frozen controller/frontend, and zero
replay. The unseen-context accuracy is `0.5096` and `0.5072`; no-file accuracy
is `0.4952` and `0.4790`.

The first four-symbol `switch` diagnostic was rejected as an invalid chance
control because `current != previous` is true 75% of the time over four
symbols. `switch_binary` fixes only that measurement confound by using a
balanced two-symbol rendered alphabet; it does not expose a target or rule ID
to the learner.

Each seed used `605,696` unique verifier bits, `49,152` unique logical
lifetimes, `768` optimizer updates, `776` route-memory updates, and zero
replay. This promotes bounded append-only external-file routing, not
unrestricted memory expansion, arbitrary program induction, or general
continual learning. Evidence is archived in
`session_records/brainworkshop_external_compute_route_bank_promoted_2026-08-11/`.

Run it with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.external_compute_route_bank \
  --slot-count 4 --file-updates 192 --route-updates 256 \
  --route-calibration-lifetimes 8 --batch-size 32 --retention-lifetimes 4 \
  --seed 17 --report-out /tmp/brainworkshop-external-compute-route-bank.json
```

## Failure-driven same-cue external route reversal (2026-08-11)

`external_compute_route_reversal.py` tests the next nonstationary-memory
pressure point. A `symbol_parity` file is first mastered behind cue `7`, and a
`triplet_parity` file is acquired behind cue `8`. The verifier then changes the
task behind cue `7` to the replacement family. The controller and event
frontend remain frozen. A parallel probe runs both opaque files and gives route
memory only terminal scalar episode outcomes; four consecutive failures then
demote the stale route and prefer the replacement for the same learned cue.

Seeds `17` and `18` both pass changed same-cue mastery (`0.8750` and `1.0000`),
old-file forced retention (`1.0000` on both), exact route reload, frozen
controller/frontend, byte-identical files, and zero replay. An unseen cue stays
near chance (`0.5099` and `0.5107`) and does not select the replacement.

This promotes failure-driven bounded route reversal with retained old
computation. It does not yet establish unrestricted memory growth, arbitrary
new computation, semantic ambiguity resolution, or general continual learning.
Raw reports and accounting are archived in
`session_records/brainworkshop_external_compute_route_reversal_promoted_2026-08-11/`.

Run one calibrated seed with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.external_compute_route_reversal \
  --source-updates 192 --target-updates 256 --route-updates 256 \
  --calibration-lifetimes 8 --transition-batches 12 --batch-size 32 \
  --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-external-compute-route-reversal.json
```

## Outcome-gated open external-compute growth baseline (2026-08-11)

`external_compute_open_growth.py` removes the fixed-bank assumption from the
canonical growth harness. It starts with one file, trains each fresh candidate
in an isolated slot, admits only a stable direct mastery prefix, and rolls
back an unmastered candidate before trying the next one. The shared
controller and event encoder are frozen after the source file; route evidence
is appended only after a candidate is admitted.

Seeds `17` and `18` each admitted five files from six candidates. The
`nback2` candidate failed stable mastery and was rolled back in both seeds;
the later `symbol_parity_odd` candidate reused that slot and was admitted.
All five direct and routed files passed, same-cue reversal passed at `1.0000`,
old-file retention remained `1.0000`, route reload was exact, and replay was
zero. The weakest routed-file accuracy was `0.8693`.

This promotes outcome-gated append-only capacity growth with failed-candidate
rollback and protected-prefix retention. It does not establish unrestricted
growth, arbitrary program induction, learned compression, or general
continual learning. Reports and full accounting are archived in
`session_records/brainworkshop_external_compute_open_growth_promoted_2026-08-11/`.

Run the calibrated promotion with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.external_compute_open_growth \
  --target-file-count 5 --candidate-budget 6 \
  --file-updates 192 --route-updates 256 \
  --route-calibration-lifetimes 8 --transition-batches 12 \
  --batch-size 32 --retention-lifetimes 4 --seed 17 \
  --credit-mode reinforce --entropy-weight 0.0 \
  --report-out /tmp/brainworkshop-external-compute-open-growth.json
```

## Attempted-outcome credit for external n-back acquisition (2026-08-12)

The previous reinforce objective could represent n-back-2 but collapsed to its
75% majority-action baseline. The generic `attempted_bce` mode now trains the
logit of the action actually attempted against only that action's scalar
outcome, with a small entropy term for exploration. A matched fresh candidate
trained on shuffled outcomes remains below mastery.

Seeds `17` and `18` both admit and route five files, including n-back-2.
Direct n-back-2 accuracy is `1.0000` on both; the weakest routed-file
accuracy is `0.8828` and `1.0000`. Same-cue reversal, old-file retention,
exact reload, frozen controller/frontend, and zero replay all pass. The
shuffled-feedback control maxima are `0.4479` and `0.2760`.

This promotes outcome-only scalar credit for a reusable external working-memory
capability, not arbitrary computation or general continual learning. Evidence
and separate control accounting are archived in
`session_records/brainworkshop_external_compute_nback2_credit_promoted_2026-08-12/`.

Run the calibrated credit promotion with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.external_compute_open_growth \
  --target-file-count 5 --candidate-budget 5 \
  --file-updates 192 --route-updates 256 \
  --route-calibration-lifetimes 8 --transition-batches 12 \
  --batch-size 32 --retention-lifetimes 4 --seed 17 \
  --credit-mode attempted_bce --entropy-weight 0.01 \
  --report-out /tmp/brainworkshop-external-compute-nback2-credit.json
```

## Deeper n-back growth with a parameterized event window (2026-08-12)

The private verifier now generates `nbackN` families generically, and the
external compute basis accepts a versioned event-window size. The four-event
window acquired n-back-3 at `1.0000` on every fresh lifetime. A direct
n-back-4 probe remained below mastery because current plus the previous three
events cannot expose a lag-four comparison reliably. Widening the same generic
window to five events acquired n-back-4 at `1.0000` on every fresh lifetime on
both seeds.

The full five-event open-growth promotion admitted and routed eight files on
seeds `17` and `18`, including n-back-2, n-back-3, and n-back-4. Same-context
reversal, old-file retention, route reload, shuffled-feedback rejection,
frozen controller/frontend, unchanged admitted files, and zero replay all
passed. The route schedule uses distinct rendered cue symbols so each file has
an unambiguous context key.

This promotes a replicated deeper working-memory capability and a
domain-general representation extension. It remains bounded by the event
window and external file bank; it does not establish unrestricted history,
learned compression, arbitrary program induction, or general continual
learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_compute_deeper_nback_promoted_2026-08-12/`.

Run the five-event promotion with:

```bash
PYTHONPATH=src:. ./.venv/bin/python -m experiments.brainworkshop_canonical.external_compute_open_growth \
  --target-file-count 8 --candidate-budget 8 --event-window-size 5 \
  --file-updates 192 --route-updates 256 \
  --route-calibration-lifetimes 8 --transition-batches 12 \
  --batch-size 32 --retention-lifetimes 4 --seed 17 \
  --credit-mode attempted_bce --entropy-weight 0.01 \
  --report-out /tmp/brainworkshop-external-compute-deeper-nback.json
```

## External temporal-history memory contract (2026-08-12)

The fixed event window is now backed by a separate memory-side temporal
contract. `ExternalTemporalHistoryMemory` stores learned event tensors in
scoped append-only records and supports opaque relative-offset reads. Storage
can grow without resizing the controller, and missing history is returned as a
mask rather than fabricated zero evidence.

The ABI probe stored 128 records in each of two scopes, read distant offsets
exactly, reloaded its checksummed payload exactly, isolated scope clearing, and
rejected corruption. This is a storage qualification only: it uses no verifier
bits or optimizer updates and does not claim learned addressing. The next
capability experiment must train the offset selector from scalar outcomes.
Evidence is archived in
`session_records/brainworkshop_external_temporal_memory_contract_2026-08-12/`.

## Stable external temporal address-index contract (2026-08-12)

`ExternalTemporalAddressIndex` now provides the canonical content-addressed
extension. It maps a learned opaque query key to an opaque namespace and a
stable absolute history position while keeping event payloads in the separate
`ExternalTemporalHistoryMemory` file. Relative offsets were deliberately not
used for this pointer: appending newer records would otherwise retarget older
addresses.

The focused ABI/runtime tests cover metadata preservation, explicit misses,
out-of-range targets, checksum reload, corruption rejection, stable retrieval
after appends, transient addressed context, and current-token persistence. This
is a storage qualification only; no verifier bits, optimizer updates, replay,
or learned-addressing claim is made. Evidence is archived in
`session_records/brainworkshop_external_temporal_address_index_contract_2026-08-12/`.

## Canonical runtime history bridge (2026-08-12)

The production runtime now exposes
`AmodalControllerRuntime.step_streams_with_external_history()`. It reads
caller-selected relative offsets from `ExternalTemporalHistoryMemory` before
appending the current learned event tokens, then passes historical and current
tokens together through the ordinary amodal event bus. Historical tokens are
transient processing context; only current tokens enter the controller's fixed
event window. The bridge preserves per-token presence masks,
and rejects both unpreserved timing/source metadata and event-window overflow
instead of silently dropping information.

This closes the transport seam between the external temporal file and the
canonical runtime; it does not promote a new learned-capability result. The
current history ABI stores learned payloads only, offset selection remains
external, and the controller still has a bounded query window. The next
experiment should train a query-conditioned offset/address policy against
scalar outcomes and measure stable-prefix retention, reversal, missing-history,
memory-corruption, shuffled-outcome, and zero-replay controls.

## Outcome-only learned temporal offset growth (2026-08-12)

The next step trains an external file to choose a relative history offset from
scalar outcomes. A frozen-controller n-back-4 file is retained while a fresh
n-back-5 file learns its offset policy over offsets 1–8. Seeds `17` and `18`
both selected offset `5` on every evaluated lifetime and passed direct mastery,
old-file retention, wrong-offset, missing-history, shuffled-outcome, frozen
core/frontend, and zero-replay gates.

This promotes one narrow learned-addressing mechanism, not general memory
search. The selector is currently a single global offset distribution per
external file; query-conditioned content addressing and multiple useful
offsets remain open. Evidence and separate accounting are archived in
`session_records/brainworkshop_external_temporal_offset_growth_promoted_2026-08-12/`.
The seed-17 canonical read-before-append revalidation is archived separately
in `session_records/brainworkshop_external_temporal_offset_canonical_bridge_revalidated_2026-08-12/`.

## Context-conditioned temporal route growth (2026-08-12)

`external_temporal_context_route_growth.py` composes the temporal offset file
with `PersistentOpaqueContextRouteEvidence`. A normalized learned event tensor
is the opaque context key; a terminal scalar episode outcome teaches the route
table which isolated temporal capability file to select. The controller and
event encoder remain frozen, and the source file is protected before the
target file is acquired.

Seeds `17` and `18` both mastered n-back-4 at cue `11` and n-back-5 at cue
`12`, learned file-local offsets `4` and `5`, selected the correct file at
`1.0000` on every routed lifetime, and retained the source at `1.0000`.
Target routed accuracy was `1.0000` and `0.9514`. Unknown-context fallback,
wrong-file, wrong-offset, missing-history, shuffled-route-feedback, exact
reload, frozen-core, and zero-replay gates all passed.

This promotes bounded composition of context routing and learned temporal
addressing. It does not yet establish multiple useful addresses under one
context, content search, learned compression, unrestricted memory growth, or
general continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_context_route_growth_promoted_2026-08-12/`.

## Opaque external address-index materialization (2026-08-12)

The context-route experiment now materializes the learned route decisions in
`ExternalTemporalAddressIndex`. The index stores only the learned event key and
an opaque `(target_scope, target_position)` location; it does not decode a
random basis vector and it does not enter the controller. Exact and nearby-key
lookups, explicit unknown misses, reload, checksum rejection, and clear are
checked independently from route mastery.

This is the canonical storage-side completion of the external address seam. It
does not promote address discovery by itself: the index writes are made only
after the outcome-only route learner has selected mastered files. Learned
same-context content search, compression, unrestricted growth, and general
continual learning remain open.

Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_context_address_index_promoted_2026-08-12/`.

## Same-cue query-conditioned temporal address growth (historical v1)

The historical promotion below is retained for provenance. The current
canonical outcome-only rerun does not reproduce its learned mastery at the
present training boundary, although the storage controls remain valid; it is
not current evidence for general continual learning.

`external_temporal_query_address_growth.py` removes the cue-to-file shortcut.
Every episode has the same rendered cue; the first learned query event is the
opaque context key. A source external file first learns a generic temporal
readout and offset 4 from scalar outcomes. The entire file is then frozen, and
the external context-keyed route table must acquire offset 5 for a new query
without replay or readout updates.

Seeds `17` and `18` both retained the source query at accuracy `1.0000`,
acquired the target query at `1.0000`, and selected offsets 4 and 5 on every
retained lifetime. Unknown-query, wrong-offset, missing-history,
shuffled-outcome, exact-reload, frozen-readout, frozen-core, and zero-replay
gates all passed.

This promotes bounded same-cue multi-address acquisition through learned
event keys. The next boundary is content-conditioned retrieval beyond exact
query keys, followed by learned compression and capacity pressure. Evidence
and accounting are archived in
`session_records/brainworkshop_external_temporal_query_address_growth_promoted_2026-08-12/`.

## Canonical-bridge query addressing replication (historical v2)

This historical report is also superseded by the current canonical rerun. It
remains useful as transport provenance, but its learned query-address
promotion is not treated as reproducible current evidence.

The same query-conditioned experiment now runs through
`AmodalControllerRuntime.step_streams_with_external_history()` rather than a
parallel manual append/read path. Logical lags remain the public opaque
addresses (`4` and `5`); the bridge receives `3` and `4` because it reads the
prior history before appending the current event. The address is selected once
per query and reused for the episode, which gives scalar terminal credit a
consistent temporal target.

With `256` source and `256` target updates, seeds `17` and `18` both passed all
`14/14` gates: source retention, target acquisition, lags `4`/`5`, unknown and
wrong-address rejection, missing-history rejection, shuffled-outcome rejection,
exact route reload, frozen readout/controller/frontend, and zero replay. Seed
`17` reached `1.0000` on every retained source and target lifetime; seed `18`
reached minimum retained accuracies `0.921875` and `0.90625`. Each seed consumed
`158,208` unique verifier bits, `16,640` logical lifetimes, `256` optimizer
updates, and `264` external route-memory updates.

This promotes canonical transport integration plus bounded query-conditioned
external addressing. It still does not establish learned unrestricted search,
compression, arbitrary new computation, or general continual learning. Evidence
is archived in
`session_records/brainworkshop_external_temporal_query_address_bridge_v2_promoted_2026-08-12/`.

## Outcome-only counterfactual query-address growth (promoted bounded rung)

`external_temporal_query_counterfactual_growth.py` removes the remaining
joint-address/readout ambiguity from the query experiment without unfreezing
the controller. Eight fresh external capability files are trained at opaque
candidate offsets `1..8` using paired common-random action arms and scalar
keypress outcomes only. A candidate is admitted only when its held-out
accuracy remains at least `0.95` across every measured lifetime. Exactly one
candidate—the verifier-valid offset `4`—is admitted; the other seven remain
below mastery.

The admitted file is frozen. The source query is recorded at offset `4`, then
the target query is evaluated against all eight opaque offsets on fresh paired
episodes. The persistent context-keyed route ledger receives only those scalar
outcomes and independently protects offset `5`. The target reaches `1.0000`
on every retained lifetime, source mastery remains `1.0000`, and exact route
reload reproduces the result. Unknown-query, wrong-offset, missing-history,
shuffled-feedback, frozen-controller, frozen-event-encoder, frozen-file, and
zero-replay gates all pass on seeds `17`, `18`, and `19`.

Each seed accounts for `165,632` unique verifier bits, `336,896`
counterfactual-arm bits, `16,704` unique logical lifetimes, `33,792`
counterfactual logical lifetimes, `1,024` optimizer updates, `72` promoted
route-memory updates, and zero replay. The unique-bit count treats the eight
candidate training streams as distinct, the common held-out candidate stream
once, and each target probe stream once; paired arms are reported separately.

This promotes outcome-only acquisition of a bounded external temporal address
and demonstrates replay-free growth of an isolated route memory while the
controller and promoted readout remain frozen. It does not establish
unrestricted memory growth, arbitrary new computation, general program
induction, or general continual learning. The private verifier is used only
for audit gates; it is not supplied to the learner. Evidence and the complete
accounting reports are archived in
`session_records/brainworkshop_external_temporal_query_counterfactual_growth_promoted_2026-08-12/`.

Run the promoted rung with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_query_counterfactual_growth \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-query-counterfactual-growth.json
```

## Outcome-only related-key content retrieval (promoted bounded rung)

`external_temporal_content_counterfactual_growth.py` composes the promoted
counterfactual capability acquisition with the canonical
`ExternalTemporalAddressIndex`. The memory write is materialized from the
learned opaque route ledger: it stores only learned context keys and opaque
absolute positions. It does not receive the private query depth or a semantic
task identifier.

Across seeds `17`, `18`, and `19`, exactly one of eight source candidates is
admitted at offset `4`; the frozen capability acquires target offset `5` from
fresh paired scalar probes. Exact and 20%-perturbed source and target keys
retrieve the correct routes at `1.0000` on every retained lifetime. Unknown
keys miss, reload preserves related-key retrieval, checksum corruption is
rejected, clear removes hits, and controller, event encoder, and promoted
file digests remain unchanged. All `23/23` gates pass with zero replay.

Each seed accounts for `166,912` unique verifier bits, `336,896`
counterfactual-arm bits, `16,704` unique logical lifetimes, `33,792`
counterfactual logical lifetimes, `1,024` optimizer updates, `72` route-memory
updates, two content-memory writes, and zero replay. This promotes bounded
related-key retrieval composed with replay-free external capability growth; it
does not establish unrestricted memory growth, arbitrary new computation,
program induction, or general continual learning. Evidence and complete raw
reports are archived in
`session_records/brainworkshop_external_temporal_content_counterfactual_growth_promoted_2026-08-12/`.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_content_counterfactual_growth \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-content-counterfactual-growth.json
```

## Sequential open query/address growth (promoted bounded rung)

`external_temporal_open_query_growth.py` removes the two-route ceiling from
the current canonical temporal experiment. After one source capability is
acquired and frozen, four new opaque query/address routes are learned
sequentially for depths `5`, `6`, `7`, and `8`. Each new route is trained only
from fresh paired scalar probes; after every addition the complete prior
prefix is evaluated again without replay.

The variable-capacity `ExternalTemporalAddressIndex` grows from one to five
learned routes. Across seeds `17`, `18`, and `19`, every stage retained all
earlier routes at `1.0000`, every new route reached `1.0000`, and all five
20%-related keys retrieved their correct opaque positions at `1.0000`.
Unknown-key miss, shuffled-feedback rejection, exact reload, checksum
corruption rejection, clear, frozen controller/event encoder/promoted file,
and zero-replay gates all pass (`16/16` per seed).

Each seed accounts for `169,600` unique verifier bits, `358,400`
counterfactual-arm bits, `17,088` unique logical lifetimes, `36,864`
counterfactual logical lifetimes, `1,024` optimizer updates, `264` route-memory
updates, five content-memory writes, and zero replay. This promotes repeated
replay-free growth of an external query/address bank and prefix retention; it
does not establish unrestricted capacity, learned compression under pressure,
arbitrary new computation, or general continual learning. Evidence is
archived in
`session_records/brainworkshop_external_temporal_open_query_growth_promoted_2026-08-12/`.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_open_query_growth \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-open-query-growth.json
```

## Replay-free temporal capacity pressure (promoted bounded rung)

`external_temporal_open_query_capacity.py` pressure-tests the five-route
external bank with a two-slot active cache. A memory-side policy selects an
opaque victim using learned signatures plus generic reliability/age telemetry;
archived routes are reactivated through copy-on-write retention probes. The
source route is protected only after a stable scalar prefix, and the controller,
event encoder, and acquired capability file remain frozen throughout.

Across seeds `17`, `18`, and `19`, the five-route archive remains addressable
while the two-slot cache performs `11` replacements and `3` active no-op
probes. Every reactivation is accepted, the protected source is never evicted,
all final active routes remain at `1.0000`, and the generic victim policy
reaches held-out accuracy `0.9414`, `0.9590`, and `0.9727`. Reward-shuffled
controls remain below the `0.70` mastery boundary (`0.5723`, `0.5723`,
`0.4980`). Related-key lookup, unknown-key miss, exact reload, checksum
corruption rejection, frozen-core/file, and zero-replay gates all pass (`17/17`
per seed).

Each seed accounts for `169,600` unique verifier bits, `12,224` capacity
retention verifier bits, `3,000` policy verifier bits, `3,080` logical
lifetimes, `4,024` optimizer updates, five archive records, two active slots,
and zero replay. This promotes bounded replay-free capacity pressure and
verifier-gated reactivation; it does not establish unrestricted memory growth,
learned compression, arbitrary new computation, or general continual learning.
Evidence and complete raw reports are archived in
`session_records/brainworkshop_external_temporal_open_query_capacity_promoted_2026-08-12/`.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_open_query_capacity \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 \
  --policy-updates 3000 --policy-eval-episodes 512 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-open-query-capacity.json
```

## Interleaved replay-free capability admission (promoted bounded rung)

`external_temporal_interleaved_admission.py` removes the remaining
experiment-level separation between acquisition and capacity pressure. Six
fresh opaque query/address routes are learned sequentially from new scalar
counterfactual probes; each is admitted immediately into a three-slot hot
cache backed by a seven-record growable archive. Previously acquired routes
are revisited between admissions, and every replacement is copy-on-write and
verifier-gated. The protected source, controller, event encoder, and frozen
capability file remain unchanged.

Across seeds `17`, `18`, and `19`, all `20/20` gates pass. The archive grows to
seven routes while the cache performs `14` replacements and `14` genuine
learned-policy victim choices. Every fresh admission and cold reactivation is
accepted; the protected source is never evicted; active, archived, and 20%
related-key routes remain at `1.0000`; unknown-key miss, reload, checksum
rejection, frozen-core/file, and zero-replay controls pass. Held-out victim
policy accuracy is `0.9336`, `0.9297`, and `0.9199`, while reward-shuffled
controls remain below mastery (`0.3027`, `0.2070`, `0.3418`).

Each seed accounts for `171,776` unique verifier bits, `48,128`
counterfactual-arm bits, `25,344/25,920/25,728` capacity-retention bits,
`3,000` policy updates, seven archive records, three active slots, and zero
replay. This promotes bounded replay-free interleaved capability admission
under capacity pressure; it does not establish unrestricted memory growth,
learned compression, arbitrary new computation, or general continual learning.
Evidence and complete raw reports are archived in
`session_records/brainworkshop_external_temporal_interleaved_admission_promoted_2026-08-12/`.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_interleaved_admission \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 --admission-count 6 \
  --policy-updates 3000 --policy-eval-episodes 512 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-interleaved-admission.json
```

## Learned temporal executable-artifact alias consolidation (promoted bounded rung)

`external_temporal_artifact_alias_consolidation.py` composes the frozen
temporal capability path with `ExecutableArtifactMemory`. Three acquired
opaque routes are materialized as external route artifacts; one route is
represented twice under exact and nearby learned keys. A generic
`OpaqueConsolidationPolicy` must select that redundant pair across every
physical row permutation, after which an independent verifier authorizes a
copy-on-write rewrite.

Across seeds `17`, `18`, and `19`, all `15/15` gates pass. Learned redundant-
pair selection is `1.0000` across 24 permutations, reward-shuffled controls
are `0.0000`, and untrained controls are `0.1667`. The accepted rewrite
reduces four physical rows to three while retaining exact and alias routes at
`1.0000`; rejected compaction is source-byte-stable, reload preserves routes,
artifact corruption is rejected, and controller/encoder/file/replay controls
pass.

Each seed accounts for `168,576` unique temporal verifier bits, `1,344,000`
policy verifier bits, `16` alias-retention verifier bits, `3,000` policy
updates, `48,024` policy logical lifetimes, and zero replay. This promotes
narrow learned external alias consolidation, not arbitrary semantic
compression, unrestricted memory growth, arbitrary new computation, or
general continual learning. Evidence is archived in
`session_records/brainworkshop_external_temporal_artifact_alias_consolidation_promoted_2026-08-12/`.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_artifact_alias_consolidation \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 \
  --policy-updates 3000 --policy-batch-size 16 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-temporal-artifact-alias.json
```

## Learned temporal shared-artifact consolidation (promoted bounded rung)

`external_temporal_shared_artifact_consolidation.py` extends alias
consolidation to two distinct temporal routes that share an opaque artifact
basis but require separate residual and address views. A generic
`OpaqueConsolidationPolicy` selects the compositional pair from opaque
candidate tensors; an independent verifier authorizes a copy-on-write rewrite
from four physical rows to three.

The policy controls use address-scrubbed candidate views. This removes an
address-geometry shortcut found in the first audit, where a reward-shuffled
policy could select the target pair despite not learning the shared artifact
structure. The deployed transactional streams still use the learned route
keys. Across seeds `17`, `18`, and `19`, learned pair selection is `1.0000`
across all 24 physical permutations; address-scrubbed reward-shuffled rates
are `0.0000`, `0.6667`, and `0.1667`, and untrained rates are `0.1667`.
Forward and reversed rewrites retain both distinct routes and the nearby
source alias, reduce serialized bytes from `10,212` to `8,275`, preserve
reload behavior, reject corruption, leave rejected transactions unchanged,
and keep the controller, event encoder, and promoted file frozen with zero
replay.

Each seed accounts for `4,352` temporal route-verifier bits, `24,576`
counterfactual temporal-verifier bits, `28,928` combined unique temporal
bits, `1,344,000` policy bits, `24` shared-view retention bits, `48,024`
policy logical lifetimes, and `3,000` optimizer updates. This promotes narrow
learned shared-view external consolidation, not arbitrary semantic
compression, unrestricted memory growth, arbitrary new computation, or
general continual learning. Evidence and complete raw reports are archived
in `session_records/brainworkshop_external_temporal_shared_artifact_consolidation_promoted_2026-08-12/`.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_shared_artifact_consolidation \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 \
  --policy-updates 3000 --policy-batch-size 16 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-temporal-shared-artifact.json
```

## Repeated online compositional temporal growth (promoted bounded rung)

`external_temporal_online_compositional_growth.py` composes fresh route
acquisition with repeated shared-view consolidation. Six anonymous temporal
routes arrive as three distinct shared-basis pairs. Each pair is learned from
fresh scalar evidence, inserted into a four-row artifact capacity, reduced
from two rows to one by a generic opaque policy, and re-read before the next
pair arrives. The compacted snapshot is grown back to hot capacity explicitly.

Across seeds `17`, `18`, and `19`, all forward and reversed streams accept all
three rewrites, and every earlier route remains at `1.0000` minimum fresh
verification accuracy after each later pair. Learned pair selection is
`1.0000` at every stage. The first stage has only two candidates, so its
shuffled and untrained rates are necessarily `1.0000`; after alternatives
exist, shuffled rates are `0.8333/0.8333/0.0000` and
`0.0000/0.2500/0.0000` across the latter two stages for seeds 17/18/19,
while untrained rates are `0.3333` and `0.1667`. Both insertion orders pass
reload, corruption, and non-mutating rejection controls. The controller,
event encoder, and acquired file remain frozen with zero replay.

Each seed accounts for `43,904` unique verifier bits, `48` temporal logical
lifetimes, `48,000` policy-training lifetimes, `64` policy-audit lifetimes,
`48,112` total logical lifetimes, `3,000` optimizer updates, and zero replay.
Serialized bytes reduce `6,338 → 3,169`, `9,507 → 6,338`, and
`12,676 → 9,507`, ending with three physical rows for six logical routes.
This promotes bounded repeated online compositional growth, not unrestricted
memory growth, arbitrary new computation, semantic ontology formation, or
general continual learning. Evidence and complete reports are archived in
`session_records/brainworkshop_external_temporal_online_compositional_growth_promoted_2026-08-12/`.

Run it with:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.external_temporal_online_compositional_growth \
  --source-updates 128 --source-evaluation-lifetimes 4 \
  --source-route-lifetimes 8 --target-route-updates 8 \
  --policy-updates 3000 --policy-batch-size 16 --batch-size 16 \
  --data-steps 14 --retention-lifetimes 4 --seed 17 \
  --report-out /tmp/brainworkshop-temporal-online-compositional.json
```

## Related-key temporal content retrieval (historical v1; current v2 rejected)

`external_temporal_content_retrieval_growth.py` composes that address
capability with the canonical persistent append-only content-addressed memory.
The memory stores two learned event keys and opaque capability-address values;
the controller and capability file are frozen before retrieval. Exact keys and
nearby learned keys (20% normalized perturbation) must recover offsets 4 and 5
and preserve the source capability. Unknown-key no-hit, clear, reload,
checksum-corruption, frozen-core, frozen-file, and zero-replay controls are
part of the promotion gate.

The historical v1 report used an older random value-basis decoder and is not
the canonical implementation. The current v2 rerun replaces that decoder with
`ExternalTemporalAddressIndex`; storage and integrity gates pass, but the
outcome-only query-conditioned source/target readout remains below mastery.
It therefore does not qualify learned content retrieval, compression,
capacity management, arbitrary new computation, or general continual
learning. The rejection and accounting are archived in
`session_records/brainworkshop_external_temporal_content_retrieval_canonical_rejected_2026-08-12/`.

## Canonical-bridge related-key retrieval replication (historical v2)

This historical bridge report is superseded by the current v2 rejection above;
the transport contract remains canonical, but the learned retrieval result is
not promoted until it reproduces under the current address-index boundary.

The related-key retrieval composition now consumes temporal events produced by
the canonical pre-append history bridge. The persistent content-addressed
memory stores only opaque learned query keys and capability-address values; it
does not enter the controller or reinterpret protocol fields. Exact and 20%
normalized-perturbation keys must recover the logical lags `4` and `5`, while
the bridge receives their prior-record offsets `3` and `4`.

Seeds `17` and `18` each pass the full retrieval, persistence, clear, reload,
checksum-corruption, frozen-capability, frozen-core, and zero-replay gates at
the stable bridge-integrated query-address rung. This remains bounded related-
key retrieval, not learned compression, capacity management, arbitrary new
computation, or general continual learning. Evidence is archived in
`session_records/brainworkshop_external_temporal_content_retrieval_bridge_v2_promoted_2026-08-12/`.

## Verified external temporal-memory compaction (2026-08-12)

`external_temporal_verified_compaction_growth.py` adds the missing commit
boundary to the append-only memory. It writes an exact source key, a nearby
source alias with the same opaque capability address, and an exact target key.
A held-out route verifier approves only the redundant source/alias merge; a
destructive source/target merge is rejected before mutation. The memory commit
is versioned and scope-safe, so stale verifiers cannot overwrite newer state.

Seeds 17 and 18 both retained exact and related-key source and target routes at
`1.0000` after reducing three records to two. Reload, checksum-corruption,
stale-version, frozen-core, frozen-file, and zero-replay controls passed. This
promotes verifier-gated compaction of redundant learned content keys, not
arbitrary compression, unrestricted memory growth, arbitrary new computation,
or general continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_verified_compaction_growth_promoted_2026-08-12/`.

## Learned live-memory compaction selection (2026-08-12)

`external_temporal_learned_compaction_growth.py` transfers the existing opaque
consolidation learner to the canonical persistent append-only memory. The
policy is trained from scalar duplicate-rewrite utility, then sees only
learned event keys and opaque values for a live memory containing an exact
source key, a nearby source alias, and a target key. It must select the
redundant pair under all physical row permutations before the held-out verifier
allows compaction.

Seeds 17 and 18 selected the redundant pair on all six permutations, versus two
of six for the untrained policy, and both committed a one-row compaction with
reload and checksum controls passing. This promotes memory-side learned
proposal selection in live external memory, not end-to-end capability
acquisition, arbitrary compression, unrestricted memory growth, or general
continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_learned_compaction_growth_promoted_2026-08-12/`.

## Learned temporal capacity scheduling (2026-08-12)

`external_temporal_capacity_schedule_growth.py` composes the learned opaque
capacity planner with the canonical persistent append-only content memory.
`MemoryCandidates.pad_to_capacity()` exposes a fixed external policy budget
with zero-filled, unoccupied rows while leaving the storage backend variable-
capacity and the controller fixed-size.

The planner is trained only from scalar utility on generic candidate banks.
In the live transfer, two distinct learned event-key addresses each have a
redundant alias. Under a four-row budget, two new addresses can be admitted
only after the planner selects and verifies a redundant pair. The stream is
run in forward and reversed physical row order; route retention is checked
after every transaction and after exact reload.

Seeds 17 and 18 reached `1.0` held-out utility for admit, evict,
consolidate, and grow. The trained planner reached `1.0` consolidation
transfer versus `0.15625` and `0.25` for fresh policies. Both seeds completed
two compactions and two admissions in both row orders, retained four distinct
routes at every stage, rejected checksum corruption, kept the controller and
event encoder frozen, and used zero replay.

This promotes bounded replay-free capacity scheduling and sequential
verifier-gated multi-row compaction in the canonical temporal-memory path. It
does not establish arbitrary shared-structure compression, semantic
equivalence discovery, unbounded memory, autonomous verifier design, or
general continual learning. Evidence and accounting are archived in
`session_records/brainworkshop_external_temporal_capacity_schedule_promoted_2026-08-12/`.

## Shared-basis external value compression (2026-08-12)

`external_temporal_shared_basis_compression_growth.py` pressure-tests a
factorized value store for distinct opaque learned event-key records. The
controller and event encoder remain frozen. Logical keys stay independent;
only value payloads share an external orthonormal basis, and ordinary reads
still return materialized learned values.

The verifier rejects a lossy rank-one candidate and accepts a rank-two
copy-on-write candidate only when all twelve routes remain distinct and within
tolerance. Across seeds `17` and `18`, forward and reversed physical row
orders reduced basis/coefficient storage from `336` to `56` scalars, with exact
reload, checksum-corruption, stale-version, frozen-core, and zero-replay gates
passing.

This is promoted as safe shared-structure memory compression. The rank choice
is deterministic SVD, so it is not yet learned compression, semantic
equivalence discovery, arbitrary computation, unrestricted growth, or general
continual learning. The next experiment must learn the structure/rank proposal
from scalar outcomes and challenge it with evolving residuals and long
nonstationary retention streams. Evidence is archived in
`session_records/brainworkshop_external_temporal_shared_basis_compression_promoted_2026-08-12/`.

## Outcome-trained shared-basis policy growth (2026-08-12)

`external_temporal_shared_basis_policy_growth.py` trains a generic external
compression selector from one scalar utility per fresh candidate bank. It sees
only rank, reconstruction error, physical-size, occupancy, and width
statistics, and emits a candidate index. The memory verifier remains
authoritative for route/value retention and versioned copy-on-write commit.

Across seeds `17` and `18`, held-out rank-selection accuracy was
`0.875/1.000/1.000` for ranks `1/2/4`. In the live frozen canonical stream the
policy selected rank `2` for six old records, then rank `4` after six successor
records arrived; all old and new routes remained readable without replay.
Reversal, reload, stale-version, corruption, frozen-core, and zero-replay
gates passed.

This promotes one replay-free outcome-trained compression preference and one
nonstationary growth transfer. It does not establish online semantic structure
discovery, unrestricted memory growth, arbitrary computation, or general
continual learning: reconstruction error is still supplied as a candidate
feature, and only one successor transition is tested. Evidence is archived in
`session_records/brainworkshop_external_temporal_shared_basis_policy_growth_promoted_2026-08-12/`.

## Raw-value shared-structure policy growth (2026-08-12)

`external_temporal_shared_basis_structure_growth.py` trains the external
`OpaqueSharedBasisStructurePolicy` from scalar verifier utility while exposing
only opaque value rows and occupancy. The policy computes a
row-permutation-invariant singular-spectrum summary internally and never
receives a precomputed candidate reconstruction-error feature. Candidate
proposals remain advisory; the persistent shared-basis memory verifies route
retention and commits atomically.

Seeds 17 and 18 reached held-out rank-1/2/4 scores of
`0.9375/1.0000/1.0000` and `0.9219/1.0000/1.0000`. Both live streams selected
rank `2 → 4`, retained six old routes after successor growth, admitted six new
routes, passed forward/reversed-order, reload, stale-version, corruption,
frozen-core, and zero-replay controls. Promotion required 50,000 scalar
updates per seed; the rejected 10k/20k calibration is archived alongside the
promoted reports. This remains bounded structure selection, not general
continual learning. Evidence is in
`session_records/brainworkshop_external_temporal_shared_basis_structure_growth_promoted_2026-08-12/`.

## Repeated raw-value shared-structure growth (2026-08-12)

`external_temporal_shared_basis_repeated_growth.py` transfers the v2 raw-value
structure policy through four cohorts: rank `2 → 4 → 4 → 4`. Each successor
cohort arrives without replaying earlier values. Forward and reversed physical
insertion order, complete prefix retention after every stage, eight
verifier-gated copy-on-write commits, exact reload, stale-version rejection,
checksum corruption, frozen controller/encoder, and zero replay are tested.

Seeds 17 and 18 reached held-out rank-1/2/4 scores of
`0.8594/0.9844/1.0000` and `0.9219/1.0000/1.0000` after 20,000 scalar
updates, and both selected `2 → 4 → 4 → 4` while retaining all 24 routes. The
3,000-update safety-only calibration was rejected for weak transfer and is
archived. This is repeated bounded structure selection, not general
continual learning. Evidence is in
`session_records/brainworkshop_external_temporal_shared_basis_repeated_growth_promoted_2026-08-12/`.

## Competing-subspace dynamic-rank growth (2026-08-12)

`external_temporal_shared_basis_competing_subspaces.py` expands candidates to
`(2, 4, 8)` and presents four incompatible orthogonal rank-two cohorts. The
policy must choose `2 → 4 → 8 → 8` as the union grows. Both subspace-arrival
orders and both physical row orders are tested, with complete prefix retention
after every verifier-gated commit.

Seeds 17 and 18 both reached held-out rank-2/4/8 scores of
`0.9688/1.0000/1.0000`; all 16 commits per seed passed, all 24 routes survived,
and reload, stale-version, corruption, frozen-core, and zero-replay controls
passed. This remains bounded competing-subspace structure selection, not
general continual learning. Evidence is in
`session_records/brainworkshop_external_temporal_shared_basis_competing_subspaces_promoted_2026-08-12/`.

## Verifier-gated shared-basis regime replacement (2026-08-12)

`external_temporal_shared_basis_regime_replacement.py` uses the new
`shared_basis_rewrite_v1` API to replace one working memory scope while
retaining a protected source scope. The old working regime selects rank `8`;
after replacing twelve old routes with twelve new routes, the new regime
selects rank `4`. Both seeds retained six protected routes, removed the old
working keys, admitted the new working keys, and passed persistence,
stale-version, corruption, frozen-core, and zero-replay controls. This is
bounded regime replacement, not general continual learning. Evidence is in
`session_records/brainworkshop_external_temporal_shared_basis_regime_replacement_promoted_2026-08-12/`.

## Learned external regime trigger (2026-08-12)

`external_temporal_shared_basis_learned_regime_trigger.py` trains
`OpaqueRegimeChangePolicy` from scalar verifier utility over opaque current
and incoming banks. It must keep stable evidence as an exact no-op and trigger
replacement only after a structural shift. The independent shared-basis
verifier then performs a protected-scope rewrite, retaining protected routes,
removing stale working routes, and admitting the new regime.

Seeds 17 and 18 both reached `1.0000/1.0000` stable-keep/shift-replace
transfer after 1,000 detector updates, while complementary fresh controls
averaged `0.5000`. Persistence, corruption, frozen-core, and zero-replay
gates passed. This is a narrow learned regime-trigger boundary, not general
change-point discovery or general continual learning. Evidence is archived at
`session_records/brainworkshop_external_temporal_shared_basis_learned_regime_trigger_promoted_2026-08-12/`.

## Alternating hidden regimes with protected scopes (2026-08-12)

`external_temporal_shared_basis_alternating_regimes.py` transfers the learned
keep/replace trigger through five hidden `A ↔ B` working-regime reversals. It
tests a stable no-op before every boundary, fresh opaque addresses for every
working occurrence, three protected scopes, verifier-gated rewrites, and
shared-basis compression after each replacement. Forward and reversed
physical row orders are both required to pass.

Seeds 17 and 18 passed all 34 gates: every boundary was detected and replaced,
all protected routes survived, stale routes disappeared, and the logical
record count stayed at 26 while physical storage stayed at 168 versus 416
dense value scalars. This is a bounded repeated-reversal and capacity-reuse
result, not general continual learning. Evidence is archived at
`session_records/brainworkshop_external_temporal_shared_basis_alternating_regimes_promoted_2026-08-12/`.

## Gated residual online adaptation (2026-08-12)

`external_temporal_regime_policy_online_adaptation.py` first trains the raw
external regime detector on stable and disjoint banks, then freezes it and
grows a zero-initialized `GatedResidualRegimeChangePolicy` from fresh scalar
utilities. The online stream has non-periodic stable intervals, partially
overlapping shifts, and disjoint shifts. No earlier pair is replayed.

Seeds 17 and 18 raised partial-overlap replacement from `0.0156` to `0.8203`
and `0.8906` while retaining `1.0000` stable keep and disjoint replacement.
The rejected naive single-policy update is retained as a negative control: it
learned the new shift but erased stable behavior. Evidence is archived at
`session_records/brainworkshop_external_temporal_regime_policy_online_adaptation_promoted_2026-08-12/`.

## Opaque binding-routed residual slots (2026-08-12)

`external_temporal_regime_policy_binding_slots.py` pressure-tests the
geometry collision exposed by the residual experiment. Two opaque context
keys route the same relational bank geometry to independent residual slots.
Slot A is learned first and slot B second; each slot is updated only from its
fresh scalar utilities. The test verifies that the other slot and the frozen
base do not change, then checks stable and disjoint retention for both keys.

Seeds 17 and 18 passed all gates. Partial replacement reached at least
`0.8828` for both bindings while stable keep and disjoint replacement stayed
above `0.9766` and `1.0000`. The 96-update overadaptation calibration is
archived as a rejection because stable retention collapsed on one seed. Evidence
is archived at
`session_records/brainworkshop_external_temporal_regime_policy_binding_slots_promoted_2026-08-12/`.

## Learned full-bank maintenance choice (2026-08-12)

`external_temporal_regime_policy_learned_maintenance.py` trains the existing
external eviction policy from scalar verifier utilities, then uses it to
select which full residual slot to reuse. The policy receives only opaque keys
and generic reliability/age telemetry; candidate order is reversed in a
control. The selected slot is replaced through the residual bank's
copy-on-write verifier, while the sibling is retained and the new binding is
learned afterward.

Seeds 17 and 18 passed selector transfer, both orderings, sibling retention,
new-binding acquisition, frozen-core, and zero-replay gates. Evidence is
archived at
`session_records/brainworkshop_external_temporal_regime_policy_learned_maintenance_promoted_2026-08-12/`.

## Routed nonstationary maintenance residuals (2026-08-12)

`external_temporal_regime_policy_nonstationary_maintenance.py` keeps the
reliability-trained maintenance base frozen and routes two isolated residuals
behind opaque context keys. One learns reliability victim selection and one
learns age victim selection from fresh scalar utilities. Unknown contexts must
fall back to the base, and candidate order is reversed as a control.

Seeds 17 and 18 passed both objective transfers, order reversal, fallback,
slot-isolation, frozen-core, and zero-replay gates. Evidence is archived at
`session_records/brainworkshop_external_temporal_regime_policy_nonstationary_maintenance_promoted_2026-08-12/`.

The promoted lifecycle run also builds copy-on-write replacement candidates.
An unsafe candidate is rejected without mutation; a verified candidate reuses
slot B for a new opaque binding while binding A remains retained. Eviction,
capacity reuse, and subsequent promotion are all checked under the same
external verifier boundary.

The promoted run also freezes each slot after its retention probe and rejects
updates to frozen slots plus a third allocation beyond the configured bank
capacity. This makes slot stopping an explicit verifier-gated operation rather
than an optimizer convention.

## Learned episodic binding discovery (2026-08-12)

`external_temporal_learned_binding_routing.py` closes the next boundary after
externally supplied context keys. An external `EpisodicBindingRouter` encodes
fresh learned event/action/outcome trajectories, provisions two opaque slots
from first observations, and learns routing from only the scalar utility of the
slot actually attempted. The canonical controller and learned event encoder
remain frozen and no examples are replayed.

Seeds 17 and 18 both reached `1.0000` forward route accuracy and `1.0000`
candidate-order permutation accuracy. Freezing and exact reload preserved the
route exactly; reward-shuffled controls stayed at `0.5000` balanced accuracy.
This promotes bounded learned binding discovery, not autonomous ontology
formation, unrestricted slot growth, or general continual learning. Evidence
is archived at
`session_records/brainworkshop_external_temporal_learned_binding_routing_promoted_2026-08-12/`.

## Online episodic binding capacity (2026-08-12)

`external_temporal_online_binding_capacity.py` pressure-tests the next
lifecycle boundary. The router learns two bindings, consolidates their keys
through held-out copy-on-write probes, recognizes a third binding as novel,
rejects an unsafe replacement, and verifier-gates replacement of one slot
while retaining its sibling. The learned route embedding is paired with an
immutable generic episode signature so novelty detection does not depend on
the learned encoder having already seen the future binding.

Seeds 17 and 18 passed all gates: initial and consolidated known rates,
retained-sibling and new-binding accuracy, retired-binding unknown rate,
permutation, reload, frozen-core, and shuffled-route controls were all clean.
Evidence is archived at
`session_records/brainworkshop_external_temporal_online_binding_capacity_promoted_2026-08-12/`.

## Learned episodic binding victim selection (2026-08-12)

`external_temporal_learned_binding_victim_selection.py` removes the remaining
manual victim-slot choice. The generic `ExternalCapabilityEvictionPolicy`
learns from opaque incoming signatures and reliability/age telemetry, then
selects the disposable physical slot under forward and reversed candidate
orders. A verifier protects sibling A and admits new binding C only after the
copy-on-write candidate passes retention.

Seeds 17 and 18 reached `0.8047/0.8184` held-out victim transfer and
`0.3086/0.3301` reward-shuffled controls. Both orderings selected the weak
slot and both transactions retained the sibling and acquired the new binding.
Evidence is archived at
`session_records/brainworkshop_external_temporal_learned_binding_victim_selection_promoted_2026-08-12/`.

## Growable interleaved episodic binding archive (2026-08-12)

`external_temporal_interleaved_binding_archive.py` separates long-term
external storage from the bounded active router. Four anonymous bindings are
introduced and revisited through six interleaved replacement cycles. The
archive keeps inactive records searchable by immutable generic signatures;
the active cache uses the scalar stable-prefix gate to protect a mastered
resident and avoid replacement thrashing.

Seeds 17 and 18 both passed six verifier-gated replacements, zero protected or
avoidable evictions, twelve active no-op probes, four archived records with two
inactive recoverable records, reload, frozen-core, and zero-replay gates. The
held-out victim policy remained at `0.8047/0.8184`, while reward-shuffled
controls were `0.3086/0.3301`. This promotes growable external episodic storage
and bounded anti-thrashing lifecycle behavior, not unrestricted memory growth
or general continual learning. Evidence is archived at
`session_records/brainworkshop_external_temporal_interleaved_binding_archive_promoted_2026-08-12/`.

## Archive scale, reversal, and integrity (2026-08-12)

`external_temporal_archive_scale_reversal.py` pressure-tests the archive as a
real long-term memory substrate rather than only a two-slot lifecycle helper.
`EpisodicBindingArchive` v2 adds cached/batched signature retrieval, explicit
scalar reversal telemetry, and a canonical SHA-256 payload checksum.

Across seeds 17 and 18, 1,024-record archives achieved `1.0000` known-query
retrieval, `0.0000` unknown false-known rate, `1.0000` query-order invariance,
perfect reload, protected-sibling survival through reversal, and corruption
rejection. The controller and event encoder were frozen and replay was zero.
The compact tensor snapshot reduced the 1,024-record persistence artifact to
about 166 KB while preserving retrieval and rejecting snapshot corruption.
This promotes scalable archive retrieval and integrity, not 1,024 learned
capabilities, semantic compression, or general continual learning. Evidence is
archived at
`session_records/brainworkshop_external_temporal_archive_scale_reversal_promoted_2026-08-12/`.

## Episodic executable-artifact reactivation (2026-08-12)

`episodic_artifact_reactivation.py` connects the growable episodic archive to
the existing opaque external executable-file bank. A learned context/signature
lookup returns only an artifact digest; a separate held-out execution probe
must verify the staged hot-bank replacement before the index commits it.

Across seeds 24101 and 24102, four cold executable artifacts were interleaved
through a two-slot hot cache. A protected artifact survived every swap, two
cold artifacts were reactivated, an old artifact was revisited without replay,
missing and corrupt handles were rejected, and the shared interpreter stayed
byte-identical. Failed and mutating retention probes left the episodic index
unchanged. Online optimizer updates and replayed examples were zero.

This promotes a bounded replay-free reactivation contract for external
capability files. It does not establish unrestricted memory growth, automatic
new computation, or general continual learning. Evidence is archived at
`session_records/brainworkshop_episodic_artifact_reactivation_promoted_2026-08-12/`.

## Variable external-history computation growth (2026-08-12)

`external_compute_open_growth.py --event-window-size 0 --history-query-count 0`
now exercises the external compute ABI with an append-only, variable-length
history of learned event tensors. Relative ages are explicit opaque addressing
features, while the reducer consumes records causally from oldest to newest and
then the current event. The controller and event encoder remain frozen; history
is read before append; missing records use an explicit mask; and no replay is
used.

Across seeds `17` and `18`, a fresh `symbol_parity` file and a held-out
`triplet_parity` file both reached `1.0000` on every direct and protected-prefix
probe. Same-cue reversal demoted the old route and preferred the replacement;
exact reload, unknown-context near-chance, reward-shuffled rejection,
controller/frontend immutability, and zero replay all passed. This is the first
replicated promotion of compositional external computation with variable
history, but it remains bounded two-file growth—not arbitrary program
induction, unrestricted memory growth, or general continual learning.
Evidence is archived at
`session_records/brainworkshop_external_history_open_growth_promoted_2026-08-12/`.

## Unbounded external history with bounded active reads (2026-08-12)

The next audit separates memory growth from per-step computation. The external
history remains append-only and unbounded, but a file may request a bounded
active slice. With `--history-query-count 4`, the fixed event ABI receives the
three preceding records plus the current event; the current event is included
before execution and appended afterward. This preserves the causal boundary
that the earlier adapter accidentally violated when it supplied four old
records and omitted the current one.

Across seeds `17` and `18`, the five-file promotion admitted and routed
`symbol_parity`, `triplet_parity`, `parity2`, `switch_binary`, and `nback2`.
Every direct and protected-prefix gate passed; `nback2` reached `1.0000` on all
four direct lifetimes on both seeds. Reversal, old-file retention, exact route
reload, unknown-context near-chance, reward-shuffled rejection, frozen-core
and frontend immutability, admitted-file immutability, and zero replay all
passed. The weakest direct file probe was `0.8636` on seed 17; all routed
probes passed their mastery gates.

This promotes a scalable external-memory/query boundary and five-file growth.
It does not establish unrestricted learned temporal dependency, learned
compression, arbitrary program induction, or general continual learning.
Evidence and accounting are archived at
`session_records/brainworkshop_external_history_bounded_open_growth_promoted_2026-08-12/`.

## External-history n-back-4 depth probe (2026-08-12)

With `--history-query-count 5`, a fresh external file receives four preceding
learned events plus the current event while the complete lifetime remains in
append-only storage. On seeds `17` and `18`, direct n-back-4 acquisition
reached `1.0000` on all four fresh probe lifetimes after 192 attempted-outcome
updates. This confirms that the corrected external-history boundary supports a
deeper temporal dependency than n-back-2.

This is a bounded direct-depth promotion only. It does not establish
multi-file n-back-4 retention, unrestricted learned query depth, learned
compression, arbitrary program induction, or general continual learning.
Evidence and accounting are archived at
`session_records/brainworkshop_external_history_nback4_depth_probe_promoted_2026-08-12/`.

## Mixed-depth external-history retention and routing (2026-08-12)

The append-only history boundary now supports a per-file active query profile
without resizing the shared controller or interpreter. The promoted profile is
`(4, 4, 4, 4, 5)`: each query count includes the current event, so the first
four opaque files read three preceding records plus the current event and the
fifth `nback4` file reads four preceding records plus the current event. The
full lifetime remains in external storage and inactive fixed-window positions
are explicitly masked.

Across seeds `17` and `18`, five sequentially acquired files reached `1.0000`
on all four fresh lifetimes. The learned route selected all five files at
`1.0000`; same-context replacement reached `1.0000`; the protected first file
remained at `1.0000`; unknown-context controls were near chance (`0.4875` and
`0.5117`); and every file digest stayed unchanged during route learning.
Direct acquisition used 960 optimizer updates per seed and route/replacement
learning used 1,280 memory-side updates per seed; replayed examples were zero.

This promotes replay-free multi-file retention and routing across mixed
bounded temporal depths. The profile is currently configured by the external
memory policy rather than learned, so this does not establish learned depth
selection, learned compression, unrestricted memory growth, arbitrary program
induction, or general continual learning. Evidence and accounting are
archived at
`session_records/brainworkshop_external_history_multiscale_route_promoted_2026-08-12/`.

## Mixed-depth external-history n-back-5 retention and routing (2026-08-13)

The same generic route-bank harness now accepts an explicit per-file query
profile and verifier-dependent episode length. With profile `(4, 4, 4, 4,
6)` and a six-record flattened event window, the fifth file receives five
preceding records plus the current event and learns `nback5` without resizing
the controller, event frontend, or shared interpreter. The training signal is
an explicit attempted-outcome BCE option with entropy regularization; the
previous REINFORCE default was retained for backward-compatible baseline runs.

Across seeds `17` and `18`, all five direct files and all five learned routes
reached `1.0000` on four fresh lifetimes. Every route selected the correct file
at `1.0000`; route reload was exact; unknown-context controls were near chance
(`0.5195` and `0.5094`); no-file controls were near chance (`0.4859` and
`0.4828`); all file digests were unchanged during routing; and the controller,
event frontend, and all replay counts satisfied their freeze/zero gates.

This promotes replicated replay-free retention and routing at a deeper mixed
bounded temporal profile. The active-depth profile remains an external memory
configuration, so this does not establish learned depth selection, learned
compression, unrestricted memory growth, arbitrary program induction, or
general continual learning. Evidence and accounting are archived at
`session_records/brainworkshop_external_history_nback5_route_promoted_2026-08-13/`.

## Outcome-only external-history depth selection (2026-08-12)

`external_compute_depth_selection.py` adds an isolated memory-side policy for
choosing a file's active history depth. It probes candidate query counts in
ascending order and commits a depth only after every probe lifetime passes the
stable-prefix mastery gate. If all candidates fail, the policy exposes no
depth. The controller, event frontend, and external files are not updated by
calibration.

Across seeds `17` and `18`, the policy selected the same minimal stable
profile `[1, 3, 2, 2, 5]` for `symbol_parity`, `triplet_parity`, `parity2`,
`switch_binary`, and `nback4`. Every selected depth retained `1.0000` on four
fresh lifetimes. Policy reload, frozen-core/frontend, file-immutability,
shuffled-outcome fail-closed, and zero-replay gates all passed. Calibration
performed no optimizer updates.

This promotes outcome-only retrieval-depth selection and a fail-closed
memory policy. It is not neural depth learning, learned compression,
unrestricted memory growth, arbitrary program induction, or general
continual learning. Evidence and accounting are archived at
`session_records/brainworkshop_external_history_depth_selection_promoted_2026-08-12/`.

## Outcome-only external-history depth selection through n-back-5 (2026-08-13)

The same policy now evaluates candidate query counts `1..6` against a six-event
shared window and a five-file bank ending in `nback5`. Across seeds `17` and
`18`, n-back-5 selected depth `6`; the complete profiles were `[1, 4, 2, 2,
6]` and `[1, 3, 2, 2, 6]`. Every selected depth retained mastery on four fresh
lifetimes. Four patient failures demoted the first file's stale depth `1`,
and a fresh depth `2` replacement re-mastered it on eight lifetimes in both
seeds.

Policy reload, controller/frontend immutability, external-file immutability,
shuffled-outcome fail-closed behavior, and zero replay all passed. This
promotes outcome-only active-depth selection and replay-free maintenance
through n-back-5. It remains an external bounded policy rather than learned
neural depth inference or learned compression, and does not establish
unrestricted memory growth, arbitrary program induction, or general continual
learning. Evidence is archived at
`session_records/brainworkshop_external_history_nback5_depth_selection_promoted_2026-08-13/`.

## Replay-free external-history depth maintenance (2026-08-12)

The depth policy also handles stale evidence. After protecting the selected
profile, four patient scalar failures demote the stale depth. A reversed
candidate is skipped during immediate re-probing so it cannot trap the policy
in a retry loop; the next candidate is evaluated from fresh outcomes and is
promoted only after its stable-prefix gate passes.

Across seeds `17` and `18`, depth `1` for the first file was demoted and depth
`2` was promoted as the replacement. The replacement reached `1.0000` on all
eight probe lifetimes, while the initially selected profile retained `1.0000`
on four fresh lifetimes. The controller, event frontend, and file parameters
remained unchanged; policy reload, shuffled-outcome fail-closed, and zero
replay also passed.

This promotes replay-free stale-depth maintenance, not unrestricted
nonstationary learning. The reversal failures are an explicit scalar-outcome
control rather than a learned environmental change. Evidence is archived at
`session_records/brainworkshop_external_history_depth_maintenance_promoted_2026-08-12/`.

## Generic external-history n-back-5 depth (2026-08-12)

The generic flattened external event ABI now supports a six-record active
window: five preceding learned events plus the current event. Across seeds
`17` and `18`, n-back-5 reached `1.0000` on all four fresh lifetimes after
512 attempted-outcome updates, with the controller and event frontend frozen
and zero replay.

The variable-history attention path was run as a negative control at the same
depth and plateaued around `0.74–0.78` after both 192 and 512 updates on seed
17. The implementation nevertheless received an important correctness fix:
masked history is compacted before recurrence and mapped back to opaque read
positions, so padding placement cannot alter valid-token computation.

This promotes deeper bounded generic temporal computation. It does not promote
unrestricted history, learned compression, arbitrary program induction, or
general continual learning. Evidence and accounting are archived at
`session_records/brainworkshop_external_history_nback5_depth_promoted_2026-08-12/`.

## Relative-age indexed external history (2026-08-13)

The variable-history reader now has a generic `history_indexed` variant. It
maps each valid external record into an opaque relative-age slot using the
memory-provided age and explicit presence mask, then exposes the separate
slots plus the current event to the replaceable compute file. This removes the
learned attention reducer's fragile content-selection step without giving the
file a task name, target bit, or raw modality format. The controller, event
frontend, and keypress boundary remain unchanged.

With dynamic history (`event_window_size=0`, query count `0`), n-back-5 reached
`1.0000` on all four fresh lifetimes across seeds `17` and `18` after `512`
attempted-outcome updates. Both runs passed frozen-controller/frontend and
zero-replay gates. Missing-history, corrupted-history, action-shuffled,
reward-shuffled, and n-back-4 depth-shift controls all failed the mastery
threshold. The result promotes bounded relative-age addressing, not learned
unbounded compression, arbitrary program induction, or general continual
learning. Evidence and accounting are archived at
`session_records/brainworkshop_external_history_indexed_nback5_depth_promoted_2026-08-13/`.
