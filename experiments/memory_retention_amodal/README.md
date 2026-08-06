# Outcome-only cue-guided retention

This experiment tests the next memory bottleneck after scalar outcome recall:
the controller sees an opaque cue for one of two later event tokens, receives
only opaque probe actions and scalar outcomes, and must retain the cued outcome
in a one-row memory while a distractor arrives.

The curriculum first trains single-event scalar recall, then presents the two
events in a random order. The controller/runtime remains one v27 model; the
cue is an ordinary learned event token, not a verifier label or a special
retention branch.

The first sub-minute rung was rejected. Seed 17 reached `0.4922` intact recall,
`0.5146` after clearing memory, `0.4971` after value corruption, and `0.4902`
under reversed order. The reward-shuffled control also remained at chance
(`0.4863` intact). The ordinary arm's committed-write rate was `0.2409`, so
the write gate changed behavior, but there was no causal memory-use gap and no
retention qualification.

Reports and accounting are in
`session_records/memory_retention_amodal_v20_2026-08-04/`. This negative rung
is useful, but the short parent was undertrained. A bridged run with 1,024
single-event updates reached a stable 100% parent prefix before the
distractor phase; it still ended at `0.5020` intact, `0.5049` clear,
`0.5195` corrupt, and `0.4805` reversed-order recall. The ordinary arm
committed `94.69%` of writes, proving that the phase transition is not failing
because writes are absent. The remaining bottleneck is outcome-only
credit assignment and retention-policy stability after the curriculum jump.
Do not promote v20 weights from either rung.

Run the initial rung with:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.memory_retention_amodal.train \
  --phase1-steps 64 --phase2-steps 128 --batch-size 16 --seed 17 \
  --report-out /tmp/memory-retention.json
```

## v21 write-credit diagnostic

The v21 follow-up added a generic current-to-prior event match and an opt-in
Bernoulli straight-through write sampler. The sampler exposes only an opaque
write log-probability to the outcome-only policy-gradient loop; it does not
add a task-specific retention branch.

With 1,024 parent updates followed by 512 target-first retention updates, the
parent reward reached a stable `1.0` prefix, but the retention conditions were
still near chance:

| condition | recall |
|---|---:|
| intact | 0.5010 |
| clear memory | 0.4951 |
| corrupt values | 0.4795 |
| reversed order | 0.4814 |
| random action | 0.4688 |

The mean write strength was `0.8902`, with `85.50%` durable commits. The
`+0.0059` intact/clear gap fails the `+0.15` promotion gate. This diagnostic is
rejected as a learned retention improvement. The sampler remains available as
explicitly opt-in training infrastructure, but no v21 weights are promoted.
See `session_records/memory_retention_amodal_v21_2026-08-04/`.

## Current qualification status

The corrected protocol presents the opaque target cue again at recall. That is
necessary: resetting recurrent state without re-presenting the ordinary query
event makes content-addressed selection impossible, because every query would
otherwise use the same blank address. This correction is an environment
protocol fix, not a learned capability gain.

The v27 runtime uses a payload-only latest-event address with a residual
learned-event identity path and strongest-prior binding, parent-capability
audits, alternating target-first/target-last
warmup, balanced-order training, held-out validation selection, and missing-cue
controls. The three-seed balanced population in
`session_records/memory_retention_amodal_v32_2026-08-04/` is rejected:

| seed | intact | target first | target last | clear | missing write cue |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.503 | 0.491 | 0.527 | 0.505 | 0.512 |
| 18 | 0.715 | 0.998 | 0.524 | 0.516 | 0.752 |
| 19 | 0.717 | 0.754 | 0.705 | 0.496 | 0.617 |

The order asymmetry identifies a last-write/first-write shortcut rather than
stable cue-conditioned utility. Seed 19 passes the per-run gate, but the
population does not. The first-to-earliest-token variant was also tested and
removed after a seed-19 run collapsed to chance; it is not part of the
production schema. No v23 weights are promoted. See the v32 ledger for the
full verifier-bit, lifetime, optimizer, diagnostic, and wall-time accounting.

A discarded private write-policy probe then measured the gate after training.
All three seeds produced approximately `0.99–1.00` target and distractor write
strengths and `1.0` commit rates in both orderings. This rules out a simple
address or event-representation failure: the policy has collapsed to “always
write.” The next experiment is therefore a lower-variance utility-policy
training sweep, not another permanent event feature.

The parent-stable v33 mini-rung tested the existing generic write cost. The
zero-cost arm passed its single-seed gate (`0.884` target-first, `0.869`
target-last), while cost `0.02` fell to `0.749`/`0.753` and failed cue gain.
Neither arm is a population promotion. Full reports are in
`session_records/memory_retention_amodal_v33_2026-08-04/`.

A v34 candidate that added generic memory-read similarity/hit features to the
write head preserved the parent audit but collapsed retention to chance. It is
rejected and removed; the canonical runtime remains v25. See
`session_records/memory_retention_amodal_v34_2026-08-04/`.

Two controls close the diagnosis. Batch size 64 still stayed at chance, so
more trajectories per update are not enough (`v35`). Freezing the mastered
parent and training only the write head still produced target-last `0.764`
versus target-first `0.506` (`v36`). The failure is therefore conditional-write
credit assignment itself, not parent co-adaptation, batch variance, event
representation, or memory-read metadata.

The v37 write-critic baseline and v38 hard-retention control both reproduce
the same last-write shortcut (`0.522` target-first, `0.997` target-last).
They remain opt-in training infrastructure, but are rejected as capability
gains. The next implementation must supply counterfactual credit for the
write decision itself rather than another scalar baseline.

## v39 counterfactual write-utility qualification

The v39 protocol supplies that missing signal. For one randomly selected
position, paired common-random arms force write versus skip; all other writes
remain shared sampled decisions. The generic write logit is trained from the
scalar recall difference between the two arms. This is a trainer-only
intervention: no target slot, verifier bit, branch position, or symbolic label
enters the controller.

After a neutral write-policy reset, the unprotected three-seed population
passes the existing retention gate. Seeds 17, 18, and 19 reach mean intact
recall `0.956`, clear-memory `0.511`, corrupt-memory `0.488`, reversed-order
`0.954`, target-first `0.954`, and target-last `0.940`; random-action recall
is `0.505`. The reward-shuffled control remains at chance. This is the first
qualified learned cue-conditioned utility result in this ladder.

The v39 protocol is promoted at the sub-minute rung. No checkpoint is
promoted yet: the next required work is a roughly three-minute replication,
retention on mastered primitives, fresh-learner transfer, and persistent
memory qualification. Reports and exact accounting are in
`session_records/memory_retention_amodal_v39_2026-08-04/`.

The parent-acquisition trainer now uses the canonical factorized counterfactual
credit primitive for paired probe and recall action decisions. Their verifier
outcomes remain separate factors; the write decision remains separately
credited because this phase does not yet isolate a write/skip arm. A small
integration smoke run executes the factorized path with zero replay and is
recorded as infrastructure validation only. This does not expand the v39
capability claim.

## v40–v41: primitive retention and parent-preserving qualification

The first longer v40 run reached `0.998` intact retention but forgot the
mastered single-event primitive (`0.738`) and is rejected. v41 fixes this with
one ordinary outcome-only parent rehearsal update after each retention update,
plus validation-time selection that requires both retention utility and parent
retention to remain above threshold. The three-seed unprotected population
then passes stable-prefix accounting: mean intact `0.995`, clear `0.512`,
corrupt `0.493`, reversed `0.994`, target-first `0.993`, target-last `0.998`,
and mastered-primitive retention `0.991`. The reward-shuffled control remains
at chance.

v41 promotes the parent-preserving training protocol and narrow sub-minute
retention result. No checkpoint is promoted. The next rung is a longer
rehearsal-preserving replication with a matched fresh-learner transfer curve;
persistent memory remains unqualified. Records are in
`session_records/memory_retention_amodal_v40_2026-08-04/` and
`session_records/memory_retention_amodal_v41_2026-08-04/`.

## v42–v43: consolidation and transfer

The v42 2,048-update stress test retained the parent but eventually violated
the stable-prefix rule. v43 adds an explicit stop after three consecutive
held-out validations pass; it stops at 320 retention updates and preserves
both the mastered primitive and unseen-token retention at `1.000`.

A matched fresh-learner transfer curve on identical unseen tokens gives
13,312 stable bits for the retained learner versus 20,480 for the fresh
learner, a `1.538x` fresh-over-transferred sample-efficiency ratio. This is a
one-seed transfer lead only. Replicate the ratio across the v41 population and
then qualify persistent-memory reload/corruption and transfer retention.

## v44: persistent-memory boundary

v44 repeats the consolidated rung with a disk-backed memory replacement. The
learned retention episode writes through `PersistentContentAddressedMemory`,
reopens the atomic snapshot, and recalls at `1.000`, matching in-process
retention. The one-seed transfer ratio remains `1.538x` and unseen-token
zero-shot recall is `0.996`.

This qualifies the persistent-memory boundary for this narrow verifier only.
Replicate reload, corruption recovery, and transfer retention across the v41
population before promoting a checkpoint or claiming general persistent
episodic memory. The record is in
`session_records/memory_retention_amodal_v44_2026-08-04/`.

## v46: population persistent-memory boundary

v46 repeats the consolidated protocol on seeds 17, 18, and 19 with an
end-to-end persistence control. Reload averaged `0.991` recall; every seed
rejected a checksum-invalid snapshot, and every restored snapshot returned
`1.000` recall. The ordinary retention gate promoted seeds 17 and 19, while
seed 18 failed the stable-prefix rule after the long requested budget.

Transfer remains the bottleneck: seed 19 reproduces the `1.538x`
fresh-over-transferred stable-bit ratio, but seeds 17 and 18 do not reach a
stable transfer threshold under the matched short budget. The persistent
storage boundary is therefore qualified across three seeds, but no checkpoint,
general episodic-memory claim, or population-level transfer claim is
promoted. Evidence is in
`session_records/memory_retention_amodal_v46_2026-08-04/`.

## v47–v48: fresh-parent transfer control

The seed-17 transferred learner reaches stable threshold at `28,672` bits,
but the matched fresh learner never qualifies its parent. Extending the fresh
phase from 512 to 2,048 updates does not change that result; the fresh learner
still has zero retention-phase updates. The transfer denominator is therefore
undefined, and no ratio is claimed.

This makes fresh-parent qualification an explicit transfer gate. The next
transfer experiment must use multiple fresh initialization seeds and report
parent acquisition separately from retention transfer. Records are in
`session_records/memory_retention_amodal_v47_2026-08-04/` and
`session_records/memory_retention_amodal_v48_2026-08-04/`.

## v49: fresh-initialization transfer gate

v49 repeats the seed-19 transfer comparison with three independent fresh
initializations. Only one fresh learner qualifies its parent and reaches
`20,480` stable bits; two fail parent qualification and never enter retention
training. The transferred learner reaches `13,312` bits, but the run is
marked `fresh_parent_not_qualified` and no population ratio is claimed.

The earlier `1.538x` ratio is therefore a single favorable fresh-control
comparison, not a reusable-capability result. The report now makes the gate
explicit. Evidence is in
`session_records/memory_retention_amodal_v49_2026-08-04/`.

## v50: value-baseline diagnostic rejection

v50 tests an opt-in training-only learned value baseline for scalar parent
policy learning. It improves parent qualification for some fresh
initializations, but the retained seed-19 model fails the stable retention
gate and the transferred parent is not qualified. The mechanism is rejected
as the default path and remains opt-in until its loss weighting is isolated.
Evidence is in
`session_records/memory_retention_amodal_v50_2026-08-04/`.

## v51–v52: parent-action intervention diagnostics

v51 trains parent probe and recall actions from coupled forced-action
outcomes, but remains at chance on seed 19 and fails parent qualification.
v52 adds a fixed-write parent scaffold: fresh parent qualification improves to
2/3 initializations, but mastered-primitive retention falls to 0.773 and no
stable threshold is reached.

Both protocols are rejected. The remaining bottleneck is the phase transition
between parent action acquisition and learned write-policy adaptation. The
original policy-gradient parent protocol remains the default control. Evidence
is in `session_records/memory_retention_amodal_v51_2026-08-04/` and
`session_records/memory_retention_amodal_v52_2026-08-04/`.

## v53: mixed parent acquisition and rehearsal

v53 uses fixed-write coupled action credit only to acquire the parent, then
returns to ordinary outcome-only rehearsal during retention. It preserves
mastered-primitive retention at 1.000 and reaches stable threshold, but
corrupt-memory recall rises to 0.519, failing the causal gap gate. Transfer
also remains unqualified. The original policy-gradient protocol remains the
default. Evidence is in
`session_records/memory_retention_amodal_v53_2026-08-04/`.

## v54–v55: phase-transition controls

v54 closes a real optimizer-state hazard: resetting the write-policy output
without clearing its Adam moments could immediately reassert the old policy.
The implementation now clears those moments. In the matched seed-19 short
rung, the reset arm reaches stable threshold at `23,040` verifier bits versus
`17,920` without the reset, with identical causal metrics. It is retained as a
phase-isolation correctness fix, not as a capability gain.

v55 freezes the generic write policy during parent acquisition and unfreezes it
at retention. This explicit gradient-routing hypothesis reaches `17,920`
stable bits with `0.999` intact, `0.480` clear, and `0.519` corrupt recall,
matching rather than improving the unfrozen control. It is rejected and was
not promoted to the longer transfer rung. Evidence is in
`session_records/memory_retention_amodal_v54_2026-08-04/` and
`session_records/memory_retention_amodal_v55_2026-08-04/`.

## v56–v57: identifiable parent-credit control

The parent intervention also contained an unidentifiable term: it optimized
the probe action even though the hidden probe bit is random and cannot be
known before the action. v56 removes that term and trains only recall, but
using the intervention during rehearsal reduces mastered-parent retention to
`0.770` and fails the stable prefix. v57 restores ordinary rehearsal and keeps
the recall-only intervention only for acquisition; its first two validations
pass, but later validation collapses and the stable-prefix gate still fails.
Both are rejected. The original policy-gradient protocol remains the default.
Evidence is in `session_records/memory_retention_amodal_v56_2026-08-04/` and
`session_records/memory_retention_amodal_v57_2026-08-04/`.

## v58: feedback-residual transfer audit

v58 adds a zero-initialized, protocol-agnostic feedback residual to the
memory-value path. All three retained runs pass the per-run parent, causal,
and persistence gates, but the matched transfer population remains
unqualified: only seed 19 reaches a qualified ratio. Unseen-token recall is
`0.742`, `0.719`, and `0.754`, so the residual does not solve address
generalization. It remains opt-in training infrastructure. Evidence is in
`session_records/memory_retention_amodal_v58_2026-08-04/`.

## v59: address initialization diagnostics

Identity-initialized and identity-residual address variants are rejected. The
long identity run is seed-dependent on unseen-token recall (`0.547`, `0.719`,
`0.996`) and qualifies transfer for only two of three seeds; the short rung
qualifies transfer but does not stabilize main retention. The diagnostic
identified a deeper contract issue: the prior address included transport age,
which differs between writes and later recalls. Evidence is in
`session_records/memory_retention_amodal_v59_2026-08-04/`.

## v60: stable content-address boundary

v60 tests the direct address correction. New v24 writes and reads project the
latest learned event payload, while age, duration, timestamp presence, and
confidence remain reasoning features rather than address features. All three
runs pass the narrow retention/persistence gates; unseen-token recall is
`0.789`, `0.738`, and `0.844`, but transfer still qualifies only two of three
seeds. The payload-only address is therefore promoted as an interface fix,
not as a learned population-transfer claim. Evidence is in
`session_records/memory_retention_amodal_v60_2026-08-04/`.

## v61: token-diverse retention control

The fixed two-token curriculum was a lookup shortcut. v61 varies the opaque
event-token pair across episodes while preserving the same token for each
episode's write and recall. Full randomization is seed-sensitive during parent
acquisition, so the promoted narrow control keeps parent acquisition fixed
and randomizes the main retention/rehearsal episodes. All three retained
models reach `0.996–1.000` unseen-token recall and pass the narrow causal and
persistence gates, but the fresh transfer population remains unstable. This
qualifies a training control, not population transfer or a checkpoint.
Evidence is in `session_records/memory_retention_amodal_v61_2026-08-04/`.

## v62–v64: fresh-transfer controls

Increasing fresh parent acquisition to 1,024 updates, doubling fresh
retention to 512 updates, and holding tokens fixed through the retention
warmup each fail to stabilize the same fresh seeds. The retained models remain
strong; the unresolved bottleneck is transfer variance, not main-task causal
retention. Evidence is in
`session_records/memory_retention_amodal_v62_2026-08-04/` through
`session_records/memory_retention_amodal_v64_2026-08-04/`.

## v65: canonical feedback-residual ablation

Removing the opaque-feedback residual while retaining the v25 payload-only
address and token-diverse schedule leaves unseen-token recall at `1.000`, but
stable retention falls to `0.805`, target-last to `0.636`, and persistent
reload to `0.813`. The causal and persistence gates fail. The generic
feedback-to-memory-value residual is therefore canonical in v25; the
no-feedback path remains only an explicit ablation. Evidence is in
`session_records/memory_retention_amodal_v65_2026-08-04/`.

## v66: matched transfer-arm configuration

The transfer harness had been silently disabling the retention write-policy
reset and omitting several declared training controls for both transferred and
fresh arms. v66 forwards the main training configuration to every matched arm.
The short rung improves some fresh controls but remains underpowered and is not
promoted. This is a correctness fix in the experiment harness.

## v67–v71: phase-transition and token-block controls

Longer fixed-token warmups alone move the failure between seeds. Reusing each
randomized opaque token pair for a bounded block is more stable: reuse two
fixes the known fresh failures but destabilizes the main seed-19 curve, while
reuse four preserves the main gate and the fresh controls when the budget is
extended to 1,024 retention updates. These are trainer-only schedules; no
token identity, target slot, or verifier label enters the controller.

## v72–v73: population transfer qualification

v72 repeats the 1,024/1,024 reuse-four protocol with persistent reload,
checksum rejection, and recovery. All three seeds pass those controls and all
fresh transfer arms qualify with positive ratios. v73 strengthens the unseen
token audit from one held-out pair to four independent pairs and records the
per-pair scores. Seeds 17, 18, and 19 all pass the main gate; minimum unseen
pair recall is `0.879`, `0.727`, and `0.840`, persistent reload is `0.996`,
`1.000`, and `1.000`, and fresh-over-transferred ratios are `2.103x`,
`1.538x`, and `2.000x`. This is the first promoted narrow outcome-only
retention/transfer result in this ladder. It does not establish natural
language, physical, or general episodic-memory capability. Evidence is in
`session_records/memory_retention_amodal_v73_2026-08-04/`.

## v74: superseded three-slot / two-row harness rung

The harness supports a bounded three-slot world with two memory rows and
derives the event-window capacity from the retained slots. The original
protected 1,024/1,024 seed-17 rung is retained for provenance, but its balanced
counterfactual arms duplicated verifier rows without duplicating the trainer's
target-position assignment. Its position-specific metrics are therefore
confounded. The record is superseded by v76; evidence is in
`session_records/memory_retention_amodal_v74_2026-08-04/`.

## v76: promoted three-slot / two-row retention

The corrected rung fixes the counterfactual pairing and adds two generic
boundary improvements: a residual learned-event identity path in the shared
memory address, and strongest-prior event binding for the write utility policy.
The v27 controller passes three seeds with intact recall `0.924/0.997/0.999`,
target-first `0.998/0.997/0.998`, target-last `0.846/0.999/0.997`, and unseen
population minima `0.914/0.836/0.930`. Persistent reload is
`0.934/0.992/0.996`, recovery is `0.938/1.000/1.000`, and all checksum
corruption controls reject altered state. This promotes only the narrow
outcome-only three-slot/two-row retention capability, not general episodic
memory or natural-modality grounding. Evidence is in
`session_records/memory_retention_amodal_v76_2026-08-04/`.

## 2026-08-04: three-factor parent credit and retention boundary

The parent-acquisition trainer now separates probe action, write/skip, and
recall action into three common-random intervention factors. Recall uses a
differentiable memory transaction so the value path can learn, while a forced
write can explicitly detach its gate gradient. A direct unit test verifies
that recall gradients reach the memory-value path without reaching the write
policy head.

At the 1,024-update parent rung, seed 17 reached `0.980` mastered-primitive
retention and `0.744` mean unseen-token retention; seed 69415 replicated
`0.992` and `0.734`. Intact recall was `0.787`/`0.747` and reverse-order
recall `0.742`/`0.720`. Reward-shuffled training stayed at chance. This is a
replicated narrow parent-acquisition and memory-value credit signal, not a
promotion: target-last recall was about `0.99`, while target-first remained
about `0.50` in both seeds.

The retention counterfactual was also corrected so non-branch positions are
forced to skip, matching its stated intervention. Parent-protected retention
controls preserve mastered skills but do not remove the recency shortcut;
unprotected retention updates catastrophically forget the parent. Those
controls are rejected. The next experiment must create negative utility for
overwriting an already retained target while keeping the parent frozen; more
generic write features or longer unprotected retention training are not the
answer. Reports and ledgers are in the
`counterfactual_three_factor_value_gradient_*` and
`counterfactual_three_factor_retention_v2_*` directories under
`session_records/sequence_working_memory_2026-08-02/`.

## v77: isolated external-writer overwrite credit

The next bottleneck was not the frozen controller's representation or memory
address; it was credit assignment for a write that can preserve or destroy an
already stored event. The `external_overwrite_v2` protocol freezes the
controller and trains a separately versioned memory-side writer from three
common-random outcome factors: target write versus skip, true distractor
overwrite versus skip after the target is stored, and target write after a
distractor is stored. The target and distractor remain trainer-only
intervention state.

The first attempt used an unbounded residual and returned to the last-write
shortcut at 64 updates. A corrected factor order, sharper generic frozen
relevance prior, and bounded `tanh` residual produced the accepted writer. The
64-update seed-17 rung reached target-first `0.949`, target-last `0.971`,
intact `0.965`, mastered-parent retention `0.973`, and unseen-token minimum
`0.957`. Seed 69415 replicated target-first `0.983`, target-last `0.982`,
intact `0.977`, mastered retention `0.965`, and unseen minimum `0.961`.
The reward-shuffled control stayed at chance (`0.483` intact,
`0.508`/`0.499` target-first/last) and never qualified its parent. All runs
used zero replayed examples.

This is a promoted narrow causal overwrite-credit rung, not a claim of
general continual learning or arbitrary new computation. Reports and ledgers
are in the three `external_write_relevance_prior_v10*` directories under
`session_records/sequence_working_memory_2026-08-02/`. The next gate is
transfer to larger slot banks and persistent memory, followed by tests that
the writer learns new write utilities rather than merely executing the frozen
relevance prior.

## v78: stable controller-native memory values for larger banks

The writer-only approach failed when the bounded bank grew from two to three
slots: its relevance gate selected the correct target, but the stored value
depended on the preceding distractor context. The controller now includes an
identity-initialized generic value path from the current learned event and
opaque feedback. It is learned during parent acquisition, then frozen while an
isolated external writer learns overwrite utility.

At the three-slot, one-row, 64-update rung, seed 17 reached target-first/
target-last `0.963/0.940`, intact `0.947`, mastered retention `0.980`, unseen
minimum `0.945`, and passed the promotion gate with zero replay. A replication
with seed 69415 required the smallest tested phase-1 extension from 704 to 800
steps, then reached `0.986/0.991`, intact `0.989`, mastered retention `0.977`,
and unseen minimum `0.961`. The reward-shuffled control remained at chance.
The same persistent backend then reloaded at `0.965`/`0.996`, rejected checksum
corruption in both seeds, and recovered at `0.938`/`1.000`.

This promotes the stable-value boundary and its checksum-protected persistence
audit for the narrow three-slot outcome-only pressure test. It does not
establish general continual learning, arbitrary new computation, or general
durable episodic memory. Reports and ledgers are in the six
`external_write_stable_value_v14*` directories under
`session_records/sequence_working_memory_2026-08-02/`.

The same result also passed a two-row bank with the same metrics in both seeds;
the independent two-row replication again passed persistence and checksum
recovery. Those two reports are the `external_write_stable_value_v14_two_row*`
directories.

## v79: four-slot temporal-bank replication

The stable-value/external-writer mechanism then passed a four-slot, two-row
bank. Seed 17 reached target-first/last `0.981/0.982`, intact `0.986`,
mastered retention `0.992`, and unseen minimum `0.980` with zero replay. Seed
69415 replicated at `0.982/0.970`, intact `0.983`, mastered retention `0.977`,
and unseen minimum `0.973`; persistent reload was `0.988`, checksum corruption
was rejected, and recovery was `1.000`. The four-slot reward-shuffled control
stayed at chance.

This promotes the bounded four-slot/two-row retention and persistence rung, not
general continual learning or arbitrary new computation. Reports and ledgers
are in the three `external_write_stable_value_v15*` directories.

## v80: five-slot temporal-bank scaling and persistence

The same stable controller-native value path and isolated external writer were
extended to a five-slot, two-row bank. Seed 17 passed at the original 704-step
phase-1 budget with target-first/last `0.974/0.973`, intact `0.975`, mastered
retention `0.961`, and unseen-token minimum `0.965`. Seed 69415 initially failed
at 800 phase-1 steps because the parent never stabilized; this is retained as a
curriculum-budget rejection, not as an architecture failure. Extending only
that phase to 1,600 requested steps produced a stable parent after 1,152
effective updates and passed with target-first/last `0.978/0.987`, intact
`0.983`, mastered retention `0.992`, and unseen-token minimum `0.969`.

The matched persistence audit for the successful replication reloaded at
`0.984`, rejected checksum corruption, and recovered intact recall at `1.000`.
The five-slot reward-shuffled control stayed at chance (`0.476` intact,
`0.514`/`0.503` target-first/last) and failed parent acquisition. Every run
used zero replayed examples. This promotes a narrow five-slot scaling and
persistence rung; it does not establish general continual learning, arbitrary
new computation, or general durable episodic memory. The phase-1 extension is
part of the replication result, and all accounting is in the five
`external_write_stable_value_v16*` directories under
`session_records/sequence_working_memory_2026-08-02/`.

## v81: learned utility-based eviction

The next bottleneck was the full-bank fallback: strength-based eviction could
discard a row that remained useful even when the frozen controller and the
external writer had already learned the correct value path. The new
`ExternalMemoryEvictionPolicy` scores opaque candidate rows from generic
controller-native write context and memory tensors. Its paired counterfactual
training compares forced row-0 and row-1 outcomes; physical row identity and
target labels remain trainer-only.

With the parent acquired on fresh randomized opaque tokens and frozen before
eviction training, seed 17 reached balanced/target-first/target-last recall
`0.916/0.903/0.981`; seed 69415 replicated `0.963/0.912/0.999`. Strength
eviction scored only `0.488/0.512` on target-first, random eviction scored
`0.737/0.756`, and both learned runs passed clear-memory, corruption,
persistent reload, checksum-rejection, and recovery controls. The
reward-shuffled control stayed at chance (`0.526` balanced, `0.501` target
first) and failed parent acquisition. Every run used zero replayed examples.

This promotes only a narrow, replicated learned-utility eviction mechanism for
a three-slot/two-row verifier with a frozen controller. It is not a claim of
general episodic memory, natural-modality transfer, arbitrary computation, or
general continual learning. Reports and ledgers are in the three
`learned_eviction_v1_*` directories under
`session_records/sequence_working_memory_2026-08-02/`.
