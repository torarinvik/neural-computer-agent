# Canonical Brain Workshop external-context rung (2026-08-05)

This record moves the Brain Workshop pressure test onto the production
runtime. Each lifetime uses a balanced hidden n-back target and exposes only
learned event tensors, opaque keypress feedback, and deterministic scalar
outcomes. The controller, event frontend, and keypress feedback encoder stay
frozen during the reward-only pilot; only external episodic context, an
intention adapter, and the replaceable keypress decoder are updated.

## Boundary implementation

```text
raw symbol -> learned event encoder -> amodal controller/memory
           -> online episodic context -> zero-impact intention adapter
           -> keypress output bus -> opaque action/outcome feedback
```

The online context state is separate from the controller recurrent state. A
stable opaque capability address is separate from the transient episode query;
the retention ledger records the former, not every changing context vector.
This distinction fixed an initial harness bug that would have created one
retention record per lifetime.

## Composition smoke

The no-update composition command passed with the expected chance-level
behavior, valid keypress propensities, one retention address, and zero replay:

```bash
PYTHONPATH=src uv run python -m experiments.brainworkshop_canonical.train \
  --n-back 4 --batch-size 4 --report-out /tmp/brainworkshop-canonical-smoke.json
```

## Reward-only reader pilot

The balanced n-back-2 pilot used fresh lifetimes only, a frozen shared
controller, 128 optimizer updates, batch size 64, and learning rate `1e-2`.
Across seeds 17, 18, and 19, post-training fresh accuracy was
`0.678/0.560/0.690`; time-shuffle controls were `0.488/0.513/0.508`, and
history-reset controls were `0.512/0.483/0.500`. Each seed used 32,768 unique
verifier bits, 8,192 logical lifetimes, and zero replayed examples. Reports
are in the matching `reward_only_lr1e2_nback2_seed*_128.json` files.

This is a replicated causal signal that the online external reader can use
trajectory history, but it is not mastery: the population remains below the
`0.8` stable-prefix threshold, and no earlier capability was yet eligible for
retention protection. The 256-update follow-up did not materially improve the
fresh scores, so adding a second capability is not justified yet.

The current bottleneck is therefore the reward-only reader's sample
efficiency and stability, not the I/O boundary or retention bookkeeping. The
next experiment should improve credit variance or add a causally justified
external reader mechanism while preserving the frozen-core and full control
set. No Brain Workshop capability is promoted from this record.

## Replicated relation-reader rung (2026-08-05)

The next reader uses a bounded external event window with learned content and
age attention. It exposes only the current learned event, the retrieved event
relation, opaque previous action, and scalar previous outcome to the external
intention adapter. The controller, event encoder, and keypress feedback encoder
remain frozen; training updates only the replaceable reader, intention
adapter, and keypress decoder.

Across fresh seeds 17, 18, and 19, 32 reward-only updates reached fresh
eligible accuracy `0.9375/1.0000/1.0000`. Time-shuffle controls were
`0.4818/0.4792/0.5365`, history-reset controls were `0.5000/0.5000/0.5000`,
and every seed protected the capability after an explicit post-acquisition
retention audit. Each seed used 4,096 unique verifier bits, 1,024 logical
lifetimes, 32 optimizer updates, and zero replayed examples.

This promotes the narrow relation-reader mechanism for balanced n-back-2
acquisition under a frozen core. It does not establish n-back-3 transfer,
sequential isolated capability growth, learned eviction, unrestricted memory
growth, or general continual learning. The next rung is a longer acquisition
and retention audit followed by adding a second capability without unfreezing
the first external slot.

The longer continuation rung used 512 fresh-lifetime updates per seed. Seeds
17, 18, and 19 all reached `1.0000` fresh accuracy; time-shuffle controls were
`0.4792/0.4792/0.5365`, history-reset controls were `0.5000/0.5000/0.5000`,
and three repeated fresh retention audits were `1.0000/1.0000/1.0000` for
each seed. Each run used 65,536 unique verifier bits, 16,384 logical
lifetimes, and zero replayed examples. This closes the bounded single-
capability stability rung. It does not test whether a second capability can be
added without damaging the first.

## Replicated two-slot sequential growth (2026-08-05)

The append-only path now adds a second relation reader, intention adapter,
keypress decoder, and opaque retention address without changing the controller
or event frontend. The new slot receives a 25% controlled exploration rate
after feedback is available, plus failure-gated advancement from the current
slot. The rollout records the exact marginal keypress propensity under that
slot mixture; no task ID or unattempted-action outcome is used.

After 128 n-back-2 updates, the n-back-2 slot was frozen and 256 fresh n-back-3
updates trained only the appended slot. Across seeds 17, 18, and 19, old-slot
fresh accuracy was `1.0000/1.0000/1.0000`; new-slot fresh accuracy was
`0.8042/0.8042/0.8021`. Old-slot time-shuffle controls were
`0.5052/0.5139/0.5035`, new-slot controls were `0.5438/0.5375/0.5396`, and
history-reset controls were `0.5000/0.5000/0.5000` for both families. Both
retention records were protected in every seed, the old controller/reader/
adapter/decoder hash was unchanged, and replay was zero. Each run used 65,536
unique verifier bits, 384 optimizer updates, and 12,288 logical lifetimes.

This promotes bounded two-slot sequential capability growth under a frozen
controller. It does not establish growth beyond two slots, n-back-4 transfer,
learned eviction, reversal recovery, persistent reload, or general continual
learning. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/sequential_nback2_to_nback3_seed*.json`.

## Three-slot routing boundary (2026-08-05)

The first n-back-2 to n-back-3 to n-back-4 capacity ladder exposed the next
real bottleneck. The three external slots all learned their candidate family:
forced candidate audits were `1.0000` for n-back-2, n-back-3, and n-back-4,
all three retention records were protected, and the first two slot hashes were
unchanged during third-slot growth. But the deployed outcome-driven route only
reached dynamic fresh accuracy `1.0000/0.8042/0.5703` for n-back-2/3/4.

This is a routing-discovery failure, not evidence that the third reader cannot
compute the capability. The third slot was selected for only `16.97%` of
training positions, and the dynamic router did not identify it quickly enough
on fresh n-back-4 lifetimes. The three-slot result is therefore rejected until
the external route learner can discover a new opaque slot without degrading
the retained slots. Evidence is in
`session_records/brainworkshop_canonical_2026-08-05/capacity_ladder_nback2_to_nback4_seed17.json`.

A proactive outcome-trained route head improved n-back-4 discovery to `0.8828`
but interfered with the retained n-back-2/3 routes, which fell to
`0.6771/0.5167`. It is rejected. The route learner needs an explicit
old-route-preservation invariant or a persistent task-inference state; a new
family's scalar rewards alone are insufficient to safely reshape shared route
selection.

A longer 12-step lifetime control improved the safe failure-gated route to
`1.0000/0.8889/0.7513` on n-back-2/3/4, while forced candidate audits stayed
perfect. More scalar evidence helps, but n-back-4 still misses the `0.8`
stable threshold. The remaining route bottleneck is persistent task inference
across short lifetimes, not reader computation or retention.

## Persistent route-evidence gate and long-lifetime promotion (2026-08-05)

The next route implementation adds `PersistentOpaqueRouteEvidence` as isolated
external state. It stores only opaque slot indices, scalar outcomes, and a
versioned stable-prefix gate; it stores no task name, n-back value, semantic
label, or correct action. A route slot becomes preferred only after eight
candidate-specific observations whose cumulative prefix remains above the
mastery threshold. Controlled slot exploration remains propensity-accounted
and is enabled during persistent-route acquisition.

The short eight-step control is rejected: fresh n-back-2/3/4 accuracy was
`1.0000/0.8042/0.5703` after the gate and candidate audits passed. This is the
irreducible cold-start cost of this cue-free verifier: an agent cannot know
which opaque reader to use on the first scored trial when the observable symbol
stream has the same distribution for every n-back family. Persistent global
preference also cannot be allowed to replace old routes on a task switch.

The 16-step ladder was replicated across seeds 17, 18, and 19. Fresh
n-back-2/3/4 accuracy was `1.0000/0.9231/0.8333` for every seed; time-shuffle
controls stayed near chance, history-reset controls stayed at or below `0.54`,
forced candidate audits were `1.0000` for every slot, prior-slot state hashes
were unchanged during third-slot growth, and the controller remained frozen.
Each seed used 196,608 unique verifier bits, 15,360 logical lifetimes, 480
optimizer updates, and zero replayed examples.

This promotes bounded three-slot external growth for long enough lifetimes,
not general continual learning. The next high-ROI task is a generic
cue-conditioned route memory or an explicitly longer-horizon Brain Workshop
protocol, with task-switch and short-lifetime controls retained. The system
must not claim arbitrary new-task routing until an observable route key exists.
Evidence is in
`capacity_ladder_persistent_route_steps16_seed17.json`,
`capacity_ladder_persistent_route_steps16_seed18.json`, and
`capacity_ladder_persistent_route_steps16_seed19.json`.

## Context-conditioned short-lifetime route promotion (2026-08-05)

The next route policy is `PersistentOpaqueContextRouteEvidence`. It indexes
the append-only slot ledger by a learned event key, not by a task name. Route
reads are available during acquisition, but persistent writes are an explicit
candidate-calibration transaction: only forced, verified candidate audits can
promote a context-to-slot mapping. This prevents exploratory failures from
poisoning the stable-prefix gate.

The verifier now has an optional rendered cue token. The cue enters through
the ordinary event encoder and never reaches the controller as metadata. With
8-step lifetimes, 256 updates for each appended capability, and 50% controlled
slot exploration, seeds 17, 18, and 19 all reached cue-conditioned fresh
accuracy `1.0000/1.0000/1.0000` for n-back-2/3/4. Time-shuffle and
history-reset controls remained near chance, forced candidate audits were
perfect, cue-shuffled fresh accuracy was `0.6823/0.8000/0.5703` for seed 17,
`0.6806/0.8021/0.5781` for seed 18, and
`0.6806/0.8021/0.5833` for seed 19. Every seed passed the pre-registered
cue-separation margin, prior-bank hash, frozen-controller, and zero-replay
gates. Each run used 86,016 unique verifier bits, 18,432 logical lifetimes,
576 optimizer updates, and zero replayed examples.

The canonical agent's versioned route-state payload also round-trips the
global and context ledgers and restores route selection without loading or
changing controller weights.

The cue-absent control remains the earlier cold-start behavior
(`1.0000/0.8042/0.57--0.58`), so this promotes cue-conditioned bounded route
selection, not arbitrary task inference. A genuine agent needs an observable
instruction or context stream; a hidden task ID or handwritten selector would
invalidate the result. Evidence is in
`context_route_short_lifetime_seed17.json`,
`context_route_short_lifetime_seed18.json`, and
`context_route_short_lifetime_seed19.json`.

The route ledger now includes reversal patience and a versioned reload
boundary. A batch writer reduces attempted slot outcomes before advancing the
ledger, so reversal patience is measured in fresh rollout batches rather than
raw eligible trials.

## Nonstationary same-cue replacement (2026-08-05)

The same rendered cue was first calibrated to n-back-2 in slot zero, then a
new n-back-4 capability was acquired from fresh scalar outcomes without
replaying the old task. Across seeds 17, 18, and 19, the new cue-conditioned
route reached `1.0000` fresh accuracy, the old slot retained `1.0000` under a
forced post-reversal audit, and route-state reload selected the new slot.
Cue-shuffled fresh controls were `0.7578/0.7604/0.7708`; prior controller and
slot state remained unchanged, and replay was zero. Each seed used 45,056
unique verifier bits, 10,240 logical lifetimes, and 320 optimizer updates.

This promotes bounded nonstationary route replacement: new verified external
capabilities can take over an existing cue without deleting the old artifact.
It does not yet show automatic stale-route demotion from uncontrolled failure
streams, unbounded growth, arbitrary new computation, or general continual
learning. Evidence is in
`context_route_reversal_seed17.json`, `context_route_reversal_seed18.json`,
and `context_route_reversal_seed19.json`.

## Failure-only stale-route demotion (2026-08-05)

The same-cue reversal was then tested without calibrating the replacement
under that cue. Cue 4 was first mastered by n-back-2 in slot zero; n-back-4
was independently learned and calibrated under cue 5 in slot one. Fresh
n-back-4 lifetimes were then rendered with cue 4, and only their scalar
verifier outcomes could update cue 4's route ledger.

Across seeds 17, 18, and 19, all promotion gates passed. After eight
transition batches, protected slot zero had reversal count `1`, slot one was
preferred, and the new route reached `1.0000` fresh accuracy. Forced old-slot
retention remained `1.0000`; controller and prior-bank hashes were unchanged;
route state reloaded to slot one; and replay was zero. Each seed used 46,592
eligible verifier bits, 95,616 total verifier-step outcomes, 46,592 feedback
events, 10,624 logical lifetimes, and 320 optimizer updates.

This promotes bounded failure-driven nonstationary external memory: fresh
failure evidence can demote a stale route and promote an already learned
replacement without deleting the old capability or replaying old data. It is
still not general continual learning, unrestricted memory growth, or arbitrary
new computation. Evidence is in
`context_route_failure_demotion_seed17.json`,
`context_route_failure_demotion_seed18.json`, and
`context_route_failure_demotion_seed19.json`.

## Generic adaptive capability growth (2026-08-05)

The next boundary removed n-back-shaped provisioning from appended slots. The
old compatibility slot learned n-back-2, then two new slots were provisioned
with the identical adaptive per-candidate relation reader and fixed event
window capacity `5`. Their constructors received no n-back value; only the
benchmark verifier was configured with n-back-3 and n-back-4. Fresh scalar
outcomes trained the slots, and forced candidate audits calibrated their
rendered cue routes.

Across seeds 17, 18, and 19, fresh accuracy was `1.0000` for n-back-2, 3, and
4. Time-shuffle controls stayed at `0.482--0.560`, history-reset controls at
`0.500--0.600`, cue-shuffled fresh controls separated causally, old forced
retention stayed `1.0000`, route reload restored slots `0/1/2`, controller and
prior-bank hashes were unchanged, and replay was zero. Each seed used 86,016
eligible verifier bits, 165,888 total verifier-step outcomes, 86,016 feedback
events, 18,432 logical lifetimes, and 576 optimizer updates.

This promotes bounded generic external capability growth: new memory-side
capabilities can share one task-agnostic bounded-window blueprint while the
controller remains frozen. It still requires an observable cue and candidate
calibration, and does not establish arbitrary program induction, unrestricted
memory growth, or general continual learning. Evidence is in
`adaptive_capability_growth_seed17.json`, `adaptive_capability_growth_seed18.json`,
and `adaptive_capability_growth_seed19.json`.

## Automatic route discovery without new-cue calibration (2026-08-05)

The explicit candidate-calibration bottleneck was then removed. After the
n-back-3 generic capability was trained under cue 5, no cue-5 calibration
transaction was performed. Twelve fresh fallback batches alone updated the
cue-5 route ledger through grouped scalar outcomes.

Across seeds 17, 18, and 19, the new cue record accumulated eight low slot-0
observations and twelve successful slot-1 observations. Slot 1 became
protected and preferred automatically; fresh cue-5 accuracy was `1.0000`, old
forced n-back-2 retention stayed `1.0000`, reload selected slot 1, the frozen
controller and prior bank were unchanged, and replay was zero. The new-cue
calibration flag was false in every report.

This promotes bounded automatic route discovery from ordinary fallback
outcomes. It still depends on an observable cue, a pre-existing generic
capability blueprint, and stable evidence; arbitrary new computation,
unrestricted memory growth, and general continual learning remain open.
Evidence is in `automatic_route_discovery_seed17.json`,
`automatic_route_discovery_seed18.json`, and
`automatic_route_discovery_seed19.json`.

## Failure-triggered adaptive capacity growth (2026-08-05)

The next open-ended-growth pressure test starts with the same generic
adaptive reader at event-window capacity `5`, then presents a new n-back-6
capability. Fresh opaque outcome probes fail below the registered stable
mastery threshold of `0.8`, so the unmastered candidate alone is replaced by
the same reader blueprint at capacity `6`. The frozen controller, old slot,
and prior route state are not resized or reset; no old examples are replayed.

Across seeds 17, 18, and 19, the failure trigger fired in every run. The
grown candidate reached `1.0000` on fresh n-back-6 outcomes, ordinary fallback
outcomes automatically discovered and preferred its cue-5 route, and the old
n-back-2 capability retained `1.0000`. Fresh time-shuffle controls were
`0.487/0.503/0.518`, history-reset controls were `0.500` in every seed, route
state reload selected the new slot, prior-bank hashes were unchanged, the
controller stayed frozen, and replay was zero. No new cue-calibration
transaction was performed.

This promotes bounded failure-triggered external capacity growth: an
unmastered memory-side candidate can be reset and expanded when fresh scalar
failure justifies it, while mastered capability state remains isolated. It
does not establish unrestricted memory growth, arbitrary new computation, or
general continual learning. Evidence is in
`capacity_growth_from_failure_seed17.json`,
`capacity_growth_from_failure_seed18.json`, and
`capacity_growth_from_failure_seed19.json`.

## Recursive failure-triggered capacity growth (2026-08-05)

The one-generation result was pressure-tested by appending a second generic
capability after the first had grown and been routed. The first unmastered
candidate expanded from capacity `5` to `6`; the second independently failed
at capacity `6` and expanded to `7`. Each failed candidate was replaced as a
complete external slot—reader, intention adapter, route scorer, opaque key,
and keypress decoder—because carrying a damaged adapter across a growth event
was shown to be unsafe. Mastered slots and the controller were never reset.

Across seeds 17, 18, and 19, both failure triggers fired, both new
capabilities reached `1.0000`, and the original n-back-2 plus first grown
n-back-6 capabilities retained `1.0000`. Cue-5 and cue-6 routes were learned
from ordinary fallback outcomes, all fresh/time-shuffle/history-reset controls
passed, all three cue routes survived state reload, prior mastered state was
unchanged, and replay was zero. Each run used `255,108` unique verifier bits,
`53,990` fresh logical lifetimes, `1,600` optimizer updates, `255,108`
eligible feedback events, and `647,880` total verifier outcome events.

This promotes recursive bounded failure-triggered external capacity growth.
It still does not establish unbounded growth, learned consolidation/eviction,
arbitrary new computation, or general continual learning. Evidence is in
`recursive_capacity_growth_seed17.json`,
`recursive_capacity_growth_seed18.json`, and
`recursive_capacity_growth_seed19.json`.

## Retention-safe bounded eviction and replacement (2026-08-05)

The next lifecycle boundary reuses a fixed three-slot external capability
bank. Slots zero and one first master n-back-2 and n-back-6 and become
protected. A third n-back-7 candidate is intentionally left unmastered;
fresh opaque n-back-8 failure evidence identifies it as replaceable. The
transaction masks protected rows, clears stale global and context-conditioned
route evidence, and replaces the candidate as a complete slot at capacity
eight.

Across seeds 17, 18, and 19, candidate failure accuracy was `0.645`, `0.655`,
and `0.592`; replacement fresh accuracy and both prior retention scores were
`1.000` in every seed. Stale route state was cleared, the replacement route
was rediscovered from ordinary outcomes, cue reload selected slots `0/1/2`,
the prior mastered bank was unchanged, a fully protected bank refused
eviction, the controller stayed frozen, and replay was zero. Each run used
`260,968` unique verifier bits, `47,846` logical lifetimes, `1,408` optimizer
updates, `260,968` eligible feedback events, and `621,998` total verifier
outcome events.

This promotes retention-safe bounded eviction and slot reuse. The selector is
currently an opaque outcome-derived utility score plus a mastery-protection
mask; learned general utility, consolidation, unbounded growth, arbitrary new
computation, and general continual learning remain open. Evidence is in
`protected_eviction_growth_seed17.json`,
`protected_eviction_growth_seed18.json`, and
`protected_eviction_growth_seed19.json`.

## Learned context-conditioned capability utility (2026-08-05)

`ExternalCapabilityEvictionPolicy` consumes only an incoming learned event
tensor and detached opaque capability addresses. Fresh scalar verifier
outcomes train the external utility policy; the controller never receives
outcome summaries, task IDs, raw verifier state, or physical slot semantics.
Candidate identities and physical slots are independently permuted, and the
incoming task context alternates between n-back-6 and n-back-7.

Across seeds 17, 18, and 19, learned selection was `1.000` in every run;
reward-shuffled and corrupted-feature controls were `0.500` in every run;
replacement fresh accuracy was `0.919/1.000/1.000`; retained capability and
base retention were `1.000` in every run; stale route evidence was cleared;
the controller was frozen; and replay was zero. Each run used `945,792`
unique verifier bits, `236,288` logical lifetimes, `2,912` optimizer updates,
and `2,540,544` verifier outcome events. The stable replacement schedule uses
a `3e-3` learning rate; the rejected `1e-2` schedule failed fresh replacement
on two seeds despite passing selection.

This promotes a narrow learned context-conditioned utility policy for a
bounded opaque capability bank with explicit retention masking. Persistent
consolidation, unbounded growth, arbitrary new computation, and general
continual learning remain unqualified. Evidence is in
`learned_eviction_context_seed17.json`,
`learned_eviction_context_seed18.json`, and
`learned_eviction_context_seed19.json`.
