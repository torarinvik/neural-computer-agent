# Curated checkpoints

## Successor-slot frontier diagnostics

`span11_slot_extension_critic_seed93604.pt` and
`span11_slot_extension_critic5_seed93605.pt` preserve the two temporary
action-conditioned critic arms used in the span-eleven extension probe. They
are intentionally **not promoted**: both preserved the parent skills but did
not create a zeroed-slot causal gain. The corresponding reports, including
all audit metrics and the discarded-critic boundary, are in
`session_records/sequence_working_memory_2026-08-02/`.

`span11_replay_credit_candidate_seed93712.pt` is the strongest unpromoted
replay-credit candidate: hidden successor gate plus binary outcome/critic
diagnostics. It reached a 0.89-point causal gain while retaining old spans
within two points, below the 5-point promotion bar. Its replay evidence is
preserved as `artifacts/memory/span11_replay_buffer_seed93703.pt`.

## Adjacent complement primitive

`complement_slot_candidate_seed93748.pt` is a validated but not yet mastered
successor-slot candidate for a visibly distinct complement operation. It
reached 59.77% on an independent 1,024-lifetime audit versus 50.64% for the
parent and 50.64% with the appended slot zeroed: a 9.12-point causal gain.
Old span-nine/span-ten retention changed by −0.01/−0.04 points, and the
matched outcome-shuffled control scored 47.56%. The complete cue-blank,
memory-reset, and retention audit is
`session_records/sequence_working_memory_2026-08-02/complement_slot_audit_seed294800.json`.

- SHA-256 candidate:
  `8db94b3f3fe95bdbf40f53c1bc6effcbac96f4e3cfcbcdc5caeed55207462e4c`
- SHA-256 shuffled control:
  `6c081a068a934e215a68481e803651db9a3ab135af19d47f9ab237e44f9f5243`

`complement_slot_replica_seed93750.pt` is the second truthful seed. It reached
58.12% with a 7.51-point causal gain; span-nine/span-ten retention remained
within the two-point gate. `complement_slot_shuffled_replica_seed93751.pt` is
its matched adversarial control and scored 47.47% on the independent audit.

- SHA-256 replica:
  `69973283351f41b91a29827e9af634bdd000be1c0a0b68a397746362461180a5`
- SHA-256 shuffled replica:
  `162afc28462dc499122c20d49d05582dd7af1371075f2818d1361ff5701cb04e`

The primitive is now replicated but not mastered: 58–60% is a partial
acquisition result, not a deployment default. Continue it only through a
protected fresh-lifetime sample-efficiency curve before increasing task
difficulty.

`complement_slot_1024_best_single_seed93763.pt` is the best single 1,024-
lifetime diagnostic arm: 69.19% complement accuracy, an 18.67-point causal
gain, and old-span retention of −1.88/−1.81 points. Two matched seeds failed
the retention gate, so this checkpoint is explicitly **unpromoted** and must
not replace the replicated 256-lifetime candidate.

- SHA-256:
  `faa68ef37e6f435daad26d18067a4060f2da1c7d0c1ea564d6e949ad2475593c`
- Complement outcome replay buffer:
  `artifacts/memory/complement_slot_replay_seed93753.pt`
- Buffer SHA-256:
  `338168fed5bede7fb0fad070c37fe78f3fa6ef37a97496dbb92e7a341f52d64a`

### Replicated margin-loss complement checkpoint

`complement_margin_slot_seed93775.pt` and
`complement_margin_slot_seed93776.pt` are the current replicated,
retention-safe 512-lifetime complement acquisition checkpoints. They use the
same successor-slot architecture but train the binary complement residual with
a constant-gradient margin objective. Independent audits reached 63.52% and
61.59%, with +13.04/+10.88-point zeroed-slot causal gains; span-nine/span-ten
retention stayed within −1.55/−1.54 points. A matched shuffled-outcome control
(`complement_margin_slot_shuffled_seed93782.pt`) scored 50.09%.

These are partial-capability checkpoints, not mastery defaults. The complete
margin/continuation/1,024-lifetime curve is in
`session_records/sequence_working_memory_2026-08-02/complement_margin_frontier_2026-08-03.md`.

- SHA-256 seed 93775: `ca46796008a851cfa4915dbed6fbdfc623564d4ff34d24f2215166db5ae5a698`
- SHA-256 seed 93776: `027692ea475430598994d7085cc6fd1b2796b89ad4d8efd99cb37357bc7a934a`
- SHA-256 shuffled control: `01cdbe5a7a0fd6f7fbfcebc0fa656bd1004c2c62565b9eb004495e33f1a11b71`

### Complement population-selection checkpoints

`complement_population_winner_seed93785.pt` and
`complement_population_winner_seed93789.pt` are the two promoted winners of
three-clone races on disjoint 1,024-lifetime streams. Their full audits reached
66.81%/+16.28 points and 67.64%/+16.90 points complement/causal performance;
span-nine/span-ten retention stayed within −1.97/−1.61 and −1.58/−1.22
points. Full reset controls were 50.00%; matched shuffled controls
(`complement_population_shuffled_seed93786.pt` and
`complement_population_shuffled_seed93790.pt`) returned to 51.68% and 50.50%.

These are promoted partial-capability checkpoints, not mastery defaults. The
population costs three clone trainings, so it currently demonstrates robust
selection rather than a lower verifier-bit cost. A 2,048-lifetime population
race selected seed 93791 but was rejected after a −2.67-point span-nine
retention drop; its local candidate is retained as
`complement_population_2048_rejected_seed93791.pt`. Reports and selection
logic are in
`session_records/sequence_working_memory_2026-08-02/population_races_2026-08-03/`.

The follow-up continuation/gate candidates are intentionally unpromoted and
kept only for analysis: `complement_population_continuation_*`,
`complement_population_append4_seed93830.pt`, and
`complement_population_entropy_append_seed93840.pt`. Their independent
audits are recorded in
`session_records/sequence_working_memory_2026-08-02/continuation_and_gate_frontier_2026-08-03/`.

### Verified fourth-slot complement compounding

`complement_population_fourth_slot_seed93871.pt` is the first promoted
fourth-slot successor of the three-slot population parent. It used 1,536 new
target lifetimes plus span-nine/span-ten rehearsal, and reached 71.90% versus
66.58% for the parent (+5.31 points). Two additional independent audits gave
+5.95/+5.80 points; reset and blank controls were ~50%, shuffled outcomes were
54.48%, and old-span retention stayed within 0.05 points. Its SHA-256 is
`61f97ee8f7ce0d2ec32e065aeaa6c72ce05a8ff7332698a3b1b89f0f58fcf262`.

`complement_population_fourth_slot_shuffled_seed93872.pt` is the matched
adversarial control. The same recipe from seed 93873 is retained but
unpromoted as `complement_population_fourth_slot_replica_seed93873_unpromoted.pt`
because its causal gain was +2.97 points. Full reports are in
`session_records/sequence_working_memory_2026-08-02/fourth_slot_compounding_2026-08-03/`.

`complement_population_fifth_slot_seed93880_unpromoted.pt` is a retained
negative control: a fifth same-task slot added only +0.46 points and did not
improve zero-shot span-eleven transfer. The corresponding record is in
`session_records/sequence_working_memory_2026-08-02/fifth_slot_saturation_2026-08-03/`.

## Procedural-shape relative operations

`unified_next_error_balanced_primary_seed43601.pt` and
`unified_next_error_balanced_replica_seed43651.pt` break the compound-2 local
saturation boundary with a replicated verifier-side 3:2 conflict/non-conflict
curriculum.

- SHA-256 primary:
  `2c219ca8e7370e735e1dcfd06d3b326bd08b32ebeb2493a17c0f255807072545`
- SHA-256 replica:
  `f83999c98f589cc68a4f615753598d459c265595db98f76b7d6bce6b7c5424c9`
- high-precision matched `next`: 72.15% → 73.50%
- causal conflicts: 81.60% → 82.02%
- non-conflicts: 62.55% → 64.85%
- hardest non-conflict/action-zero cell: 52.99% → 56.81%
- previous-item overall/conflict retention: 96.30% / 95.65%
- exact paired truthful run beats both unchanged-parent and shuffled-outcome
  controls; complete memory reset remains at chance
- acquisition per run: 1,536 target bits, 10,752 rehearsal bits, 6,144
  unique lifetimes, 16 gradient evaluations, and four optimizer updates

The weights use verifier-known logical subgroups only for loss allocation and
are not exposed to the controller. Focal loss, non-conflict-only weighting,
and equal subgroup weighting were rejected before this asymmetric setting.

`unified_next_trust_compound2_primary_seed43001.pt` and
`unified_next_trust_compound2_replica_seed43051.pt` are the first replicated
second compounding increment.

- SHA-256 primary:
  `3879d91a4dd0b349b3396b0679c8d4d49c81ab56de6cb56686992cc84640d192`
- SHA-256 replica:
  `f4d041d76ff42c3f3d213b079196dc7935135e9de360dd6a1a29089c3cb65ffa`
- matched `next` gain: 69.66% → 71.61% / 68.23% → 71.68%
- matched causal-conflict gain: 79.46% → 81.26% /
  78.07% → 82.09%
- previous-item overall/conflict retention: 96.50%/95.96% and
  96.44%/95.60%
- exact shuffled target control: no `next` or independent-slot gain
- acquisition per increment: 1,536 target bits, 10,752 rehearsal bits,
  16 gradient evaluations, and four optimizer updates

A third half-step is not promoted: it gained only 0.46 `next` points and no
causal-conflict accuracy. These checkpoints define the current compounding
frontier and its measured local saturation boundary.

`unified_next_reference_only_trust_region_primary_seed42801.pt` and
`unified_next_reference_only_trust_region_replica_seed42851.pt` are the
replicated constraint-only rehearsal milestones.

- SHA-256 primary:
  `57f463296a3564da4fe7cb432f3650158b245bf8179cc0b4999c1cc4b26bf7be`
- SHA-256 replica:
  `ae419d571ea3976d28b257d59da32f4a2dedd51d6611f18abe21932fc81b9565`
- acquisition per run: 1,536 target verifier bits, 10,752 rehearsal
  verifier bits, 16 gradient evaluations, and four optimizer updates
- matched `next` gain: 63.93% → 69.99% / 64.71% → 69.99%
- matched causal-conflict gain: 70.43% → 78.58% /
  74.29% → 80.49%
- every retained old-skill overall and causal-conflict gate: above 95%
- complete memory reset: chance
- same-seed truthful target outcomes beat shuffled outcomes on `next`,
  causal conflicts, and the independent new-slot subgroup

Rehearsal batches define the protected gradient direction but do not take
old-skill optimizer steps. A 0.000025 target learning rate is the first tested
trust region that replicated learning without leaving the locally safe loss
region. This is incremental protected learning, not second-anchor mastery.

`unified_next_gradient_projection_primary_seed41901.pt` and
`unified_next_gradient_projection_replica_seed42101.pt` are the first
replicated protected-plasticity checkpoints.

- SHA-256 primary:
  `4d3536228ffff37cc9f71a24c11f3937fdb0168a2e18797cc4ae38fae083adc8`
- SHA-256 replica:
  `9e0d21e7f857c04fc5de6b9cdcf2fcc18fd754bdce7aab503d86b5d8fd86dd24`
- acquisition: 1,536 target verifier bits plus 10,752 truthful rehearsal
  bits in each run
- new aligned `next`: 64.00% / 64.45%, from a 58.12% zero-shot baseline
- new causal conflicts: 73.23% / 71.37%
- all redundant-anchor, first-next, and previous-item overall and
  causal-conflict gates: above 95%
- complete memory reset: chance
- matched target-only shuffled control: 52.99% `next`, 50.86% conflicts

The generic training rule projects only a target-gradient component that
opposes the current cycle's aggregate rehearsal gradient. It uses no semantic
task ID or correct unattempted action. A longer continuation reached 92.36%
new `next` but is deliberately not promoted because previous-item causal
conflicts fell to 94.43%.

`unified_procedural_shape_next_bridge_seed41151.pt` is an explicitly
unpromoted research checkpoint at the second `next item` anchor frontier.

- SHA-256:
  `91e2b4108eb7c51eb593bbe1414ff6e29727b6e0e3d0bf37c1ccb6ed07516a54`
- first `next item` anchor retention: 98.89%
- `previous item` retention: 99.29%
- second independent-anchor performance: 58.69% (`next`)
- full memory reset: 50.00%

The two previous-item milestone checksums, omitted from the manifest in their
original commit, are now also registered there.

## Color primitive compounder

`color_primitive_compounder_bits16_seed1901.pt` is the first replicated
cross-attribute compounding checkpoint.

- SHA-256:
  `698cf1a56914ff733e808c189043caea66e0355b1da1742bbcd78fd9be2f156f`
- Size: approximately 5.5 MB
- Learner-visible training signals: rendered RGB events, attempted opaque
  binary answers, and scalar verifier outcomes
- Atom acquisition: 64 target-color outcomes and 24 effect-color outcomes
- New relation acquisition: stable causal mastery from 16 outcomes
- Exact unacquired architecture control: no mastery through 64 outcomes
- Blind disjoint replication: 100% normal and counterfactual accuracy with
  100% causal flips
- Missing-target and missing-effect audits: chance
- Retained position ladder: 8/8/24 outcomes

The checkpoint contains two compact vision branches, their reward-trained
primitive heads, and the 16-outcome relation head. The two-branch design is an
honest response to measured negative transfer: position-trained vision helps
effect-color learning but suppresses target-color learning. Unifying or
compressing these branches without losing the transfer gain remains open.

## Unified controller lineage

`unified_compound_rehearsal_seed2505.pt` is the first one-controller
few-shot-and-retention milestone.

- SHA-256:
  `8e346fd2863f595fd0ee96b5f4a8353cae48e96e7739ad0febd11345a60f9099`
- Size: approximately 1.2 MB; 298,252 parameters
- Learner-visible signals: rendered RGB, attempted opaque action, scalar
  verified outcome, and the controller's own latent active state
- One-support hidden bijection: 99.98% blind normal and 99.95% reversed
- Retained two-support four-function task: 100% blind normal and reversed
- Counterfactual prediction flips: 99.93% and 100%
- Shuffled-feedback, active-state-reset, and blank-vision controls prevent
  reward-history, static-action, and renderer-shortcut explanations
- No semantic task IDs, correct-action labels, unattempted-action labels, or
  within-lifetime weight updates

The three preceding checkpoints preserve the exact gradual lineage:

- `unified_constant_action_v1_seed2501.pt`: active-state memory atom;
- `unified_visible_identity_from_constant_seed2501.pt`: visual-action atom;
- `unified_four_rule_support2_from_visual_seed2502.pt`: audited two-support
  memory × perception parent.

The final rung used balanced rehearsal with different evidence schedules,
preserving the broader parent while acquiring the harder one-support mapping.
Persistent disk memory exists only as an inactive tested interface and is not
part of this capability claim.

## Persistent-memory ladder

`unified_persistent_capacity40_temp50_seed4801.pt` is the current unified
controller milestone.

- SHA-256:
  `3adc437e87e3ec65f02aeb22fe56bb16d0d48b43543d9858762eb2f27e2b3d9d`
- Capacity-40 blind recall: 90.00%
- Zero-shot capacity-48 recall: 88.28%
- Zero-shot capacity-56 recall: 87.33%
- Zero-shot capacity-64 recall: 85.57%
- Acquisition cost: 20 memory updates, 4,800 unique recurring contexts,
  and 4.93 seconds on the measured GPU run
- Empty, shuffled, and corrupted memory controls: large causal degradation
- Private-rule reversal and paired prediction-flip controls: passed
- Original one-support and four-rule behavioral retention: passed
- Serializable disk read reproduces the hard latent-memory read
- A matched temperature-10 pilot regressed and was rejected; sharpening the
  differentiable training read to temperature 50 was the isolated change

`unified_persistent_capacity40_temp50_seed4802.pt` independently replicated
the same acquisition in 5.01 seconds:

- SHA-256:
  `b8244b6e2e1dbbe84578f8f0f48d5689c750d99cee47ff5ddd9f647879a57a69`
- Capacity-40 blind recall: 89.83%
- Zero-shot capacity-64 recall: 86.33%
- Capacity-64 retrieval top-1: 72.34%
- All causal, disk, and behavioral-retention gates passed

The exact gradual parents are:

- `unified_persistent_capacity2_ratio2_seed4004.pt`, SHA-256
  `2be66cd8b13f4eee86a80bd2c0369a7cb138d1b83f489cef5e962ab0d086a34c`;
- `unified_persistent_capacity8_bridge_seed4005.pt`, SHA-256
  `90d4a28c8c855190fc58f4f536a927ef3548229768a9bd5ca6117655fa24528c`;
- `unified_persistent_capacity16_bridge_seed4006.pt`, SHA-256
  `7ae568ea2007241dcc167764bd91ba4e4c19bb95431c968d42e0f4c6f766a215`.

The memory rows contain controller-created opaque vectors, not semantic
records.

## Selective-memory atom

`unified_selective_memory_atom_seed5402.pt` is the first checkpoint with an
audited learned admission decision.

- SHA-256:
  `3fa82275e37ba5de686d4ec9966c1345e15b46e89938a8cf9bc0e0da94b15c30`
- Parent: `unified_persistent_capacity40_temp50_seed4802.pt`
- Acquisition: 80 write-gate updates and 20,480 generated contexts
- Blind first-encounter write rate: 61.16%
- Blind redundant-repeat write rate: 5.10%
- Blind writes per context: 0.663
- Query accuracy after either encounter: 99.90%
- No-write, shuffled-admission, corrupted-value, and hidden-read controls:
  passed
- One-support and four-rule retention: passed

The policy emergently uses memory absence as the default case and writes rows
for exceptions. This is a bounded one-slot admission atom, not yet a claim of
replacement, merging, compression, or unbounded-stream consolidation.

## Frequency–recency replacement

`unified_memory_frequency_recency_capacity6_seed6607.pt` is the promoted
bounded-memory utility-composition milestone.

- SHA-256:
  `1346da994de4ba20864c5f1bc1da12684fc13d8dcda480a76cfc6f713da0181c`
- Parent: `unified_memory_replacement_capacity6_seed6310.pt`
- Size: approximately 1.2 MB; 298,359 parameters
- Acquisition: 20 reward-only updates, 51,200 unique verifier bits,
  no replay, 3.23 seconds training
- Held-out future accuracy: 95.32%
- Correct noisy frequency-plus-recency evictions: 87.30%
- Physical disk audit: 96.81%, 92.97% correct evictions
- Every audited access history survived save/reload; row count stayed bounded
- Age and frequency corruption controls both caused material degradation
- Recency, one-support, and four-rule retention gates passed
- No semantic utility labels or correct eviction actions entered training

`unified_memory_frequency_recency_capacity6_seed6608.pt` independently
replicated the milestone:

- SHA-256:
  `b50a3338ef197c4cd955b45a465994052df6443772c73c2bd97c421f1440bc8f`
- Acquisition: 20 updates, 51,200 verifier bits, 3.23 seconds training
- Held-out / physical accuracy: 95.10% / 96.29%
- Physical correct evictions: 93.36%
- All causal, bounded-disk, persistence, and retention gates passed

## Online utility adaptation

`unified_memory_online_utility_seed6810.pt` is the promoted online-adaptation
milestone.

- SHA-256:
  `c3e837c6512a30c11b1c861b79242296b76cfa0cd9fe62aa414d3e5b2aa10750`
- Parent: `unified_memory_frequency_recency_capacity6_seed6607.pt`
- Size: approximately 1.2 MB; 298,359 parameters
- Acquisition: 64 symmetric perturbation horse-race updates, 212,992 unique
  verifier bits counting both candidates, zero replay, 28.89 seconds total
- Continuous phases: recency-dominant → frequency-dominant → recency return →
  equal return, with no learner-visible boundary or optimizer reset
- Correct-target rates: 90.82%, 87.16%, 91.31%, 89.99%
- Physical disk audit: 96.94% learned versus 96.81% visible oracle
- 6,144 rows and all 1,024 access histories survived save/reload; zero growth
- Age/frequency corruption reduced accuracy to 92.74%/88.66%
- Binary mapping and four-rule retention gates passed
- Only `memory_replacement_extra_gate.weight` changed

`unified_memory_online_utility_seed6809.pt` independently replicated all online
and retention gates:

- SHA-256:
  `d25d26c4d34ff86e50474b5ff38c630a2d92b782dea10d4782b01a363bb64a81`
- Acquisition: 64 updates, 212,992 unique verifier bits, zero replay,
  28.66 seconds
- Correct-target rates: 90.67%, 86.43%, 91.16%, 90.53%

The matched reward-shuffled control failed the frequency switch at 57.71%
correct targets and saved no checkpoint.

## Previous-item sequence operation

`unified_procedural_shape_previous_primary_seed37201.pt` and
`unified_procedural_shape_previous_replica_seed39251.pt` are the first
replicated procedural-shape sequence-manipulation checkpoints.

- SHA-256 primary:
  `2b359ea78595ecca06030c86c46d0306e0149a732cbd3d06b4edff0119f14a95`
- SHA-256 replica:
  `00c87374b2f734e83f2115c0bed2c8411802e9fecd570a34a316f1666a03e152`
- Size: approximately 650 KB each
- Natural held-out accuracy: 98.30% / 98.59%
- Previous-item accuracy: 97.78% / 98.23%
- Previous-item conflict accuracy: 96.65% / 97.43%
- Weakest query-position/target cell: 96.44% / 96.88%
- Complete fast-memory reset: 49.95% / 49.90%
- Valid operation-counterfactual accuracy: 97.57% / 97.84%
- Matched shuffled-outcome conflict accuracy: 52.57%

The checkpoints retain three abstract visual events in RAM/VRAM and use an
arbitrary visual operation glyph to select direct or previous-item lookup. No
operation ID, ordinal, correct action, target identity, or unattempted outcome
entered training. Full evidence and the ultra-gradual lineage are recorded in
`session_records/procedural_shape_previous_operation_2026-07-30/README.md`.

## Conditional memory retrieval

`unified_conditional_memory_usage_prior_seed17603.pt` is the promoted per-query
content-versus-verified-strength retrieval milestone.

- SHA-256:
  `1f52e037ff9af5e116f2b6b7ca8852242d8e560a248ed01e6b998cddcbd03b1e`
- Parent: `unified_memory_usage_prior_seed17401.pt`
- Approximately 1.2 MB; only the new 49-parameter policy changed
- Stable mastery: 5,120 verifier bits, 40 updates, no replay
- Held-out exact/ambiguous accuracy: 100%/100%
- Fixed scale-zero / scale-one controls: 74.22%/74.41%
- Physical audit: 256/256 banks correct after exact disk reload
- Feature-shuffle and value-corruption controls: passed
- Reward-shuffled control: no mastery
- Binary mapping, four-rule, selective-disk, and volatility retention: passed

The policy consumes only generic retrieval statistics and scalar visual-task
outcomes. Private arm identity is audit metadata and was not a training label.

## Continuous memory retrieval

`unified_continuous_memory_usage_prior_seed17718.pt` is the promoted continuous
resource-control milestone.

- SHA-256:
  `dbd8b5306be515d8abcedaf95ea6c16f16797e760d6ac5f06f0a6c76fe02189a`
- Parent: `unified_conditional_memory_usage_prior_seed17603.pt`
- Same architecture and parameter count; only the 49-parameter policy changed
- Stable improvement: 640 verifier bits, five updates, no replay
- Final acquisition: 1,024 bits, eight updates, 5.17 seconds
- Held-out two-/three-/four-row accuracy: 100%/100%/100%
- Mean usage-prior scale: 0.312 versus the inherited binary policy's 0.50
- Reward-shuffled and reset-policy controls: 50% row accuracy
- Feature-shuffle and value-corruption controls: passed
- 128 physical banks per row count reloaded exactly and scored 100%
- Parent conditional retrieval and all older retention gates: passed

The checkpoint was selected on a task where no constant scale can solve both
arms. Correctness dominates a smaller generic cost on retrieval strength.

## Four-way physical memory retrieval

`unified_four_target_memory_retrieval_seed17828.pt` is the promoted four-way
retrieval milestone.

- SHA-256:
  `22152214154e1935b35539954c22f8f9bbfb3acd3dee9aaeddde6e3ddcafa00f`
- Parent: `unified_continuous_memory_usage_prior_seed17718.pt`
- Same architecture and parameter count; only the 49-parameter policy changed
- One new batch: 512 verifier bits and 512 logical contexts
- 1,000 internal updates and 639,872 explicitly counted replayed examples
- Four held-out target regimes: 100% each
- Best fixed scalar: 25%; feature shuffle: 25.39%
- Value corruption: 0% visual success
- 128 physical banks: 100% correct and every reload exact
- Parent continuous/conditional retrieval: 100%/100%
- Binary-mapping and four-rule retention: passed
- Independent seeds 17827 and 17829 replicated every gate
- Shuffled verifier reward learned one class at 25% and was rejected

The learner uses only successful scalar-action intervals. Parent rehearsal
preserves behaviorally equivalent action regions rather than exact numeric
outputs, which permits plasticity without catastrophic forgetting.

## Unseen-boundary four-way retrieval

`unified_four_target_boundary_transfer_seed17915.pt` is the promoted
out-of-distribution retrieval milestone.

- SHA-256:
  `f36a42d95e3cf20b93091ae62b1924540efb456a094013c96a2438dd76ada345`
- Parent: `unified_four_target_memory_retrieval_seed17828.pt`
- 298,524 total parameters; only a new 113-parameter residual changed
- Residual input: four legacy statistics plus sorted cosine/usage for four rows
- Training shift range: `[-0.09, 0.12]`
- Disjoint held-out bands: `[-0.099, -0.095]` and `[0.13, 0.16]`
- Stable unseen mastery: 1,536 verifier bits; 4,096 total
- Every class in both unseen bands: 100%
- Best fixed scalar: 25%; feature shuffle: 23.3–25.0%
- Value corruption: 0% visual success
- 256 unseen-band physical banks: 100% correct, every reload exact
- Parent continuous/conditional retrieval: 98.93%/100%
- Binary-mapping and four-rule retention: passed
- Independent seed 17916 replicated every gate
- Shuffled reward and exact four-feature controls were rejected

The residual is zero-output at insertion. Its full-row evidence is generic,
sorted, and invariant to physical memory-row permutation.

## Multi-feature online utility

`unified_memory_multifeature_reliability_seed6932.pt` is the promoted
two-dimensional online-utility milestone.

- SHA-256:
  `bb5cd158c08f4b92061aca7bfae0751d4e18408e8e37f53cac13dffaed8ac9f4`
- Parent: `unified_memory_online_utility_seed6810.pt`
- 298,360 parameters; one zero-initialized reliability coefficient added
- 48 move/stay horse-race updates, 196,608 verifier bits, zero replay,
  29.37 seconds
- Correct-target rates: 89.75%, 78.22%, 88.48%, 87.45%
- Physical audit: 96.21% learned versus 96.35% visible oracle
- All 6,144 rows and 1,024 access/success/failure histories survived reload
- Age, frequency, and reliability corruption gates passed; zero growth
- Binary mapping and four-rule retention passed

`unified_memory_multifeature_reliability_seed6938.pt` replicated all training
and retention gates:

- SHA-256:
  `0342a8266bde7bc5a0f79004792ce29668f758904aa954755b7bf7130993730d`
- 48 updates, 196,608 verifier bits, zero replay, 29.34 seconds
- Correct-target rates: 88.67%, 88.43%, 84.72%, 83.35%

The exact reward-shuffled control failed and saved no checkpoint.

## Physical online utility adaptation

`unified_memory_physical_online_seed7012.pt` is the selected checkpoint whose
adaptation decisions were made from physical disk-backed memories.

- SHA-256:
  `2c6e61b5e2689d46dfc43dd5cfc9c5b234736d217aae28f6221501bd5ddeea70`
- Parent: `unified_memory_multifeature_reliability_seed6932.pt`
- 298,360 parameters; only the existing two-coefficient residual changed
- 48 updates, 196,608 unique verifier bits, zero replay, 136.33 seconds
- 6,144 complete physical histories persisted through save/reload
- Correct-target rates: 91.06%, 82.13%, 88.23%, 82.91%
- Physical/tensor choices equivalent on all 48 updates
- Binary mapping and four-rule retention passed

`unified_memory_physical_online_seed7015.pt` independently replicated every
registered gate:

- SHA-256:
  `7ae96b44ec6bed0db8eb7f9b78640fe40b621875195303e3e3c604f357bb441d`
- Correct-target rates: 85.74%, 77.25%, 86.72%, 82.67%
- 136.69 seconds; identical verifier-bit and persistence accounting

The matched reward-shuffled control failed and saved no checkpoint.

## Balanced maximin population winner

`balanced_maximin_stream7085_clone7211_round54.pt` is the exact resumable state
selected by the replicated balanced population objective.

- SHA-256:
  `a4202886e3a8d712baa0d1e6ee003bb86c93cc3ff3bc96d7f59e3d54b90a1839`
- Parent controller:
  `unified_memory_online_utility_seed6810.pt`
- State includes controller residual weights, bounded physical disk banks,
  latent strategy memory, context encoder and optimizer, RNG states, trace,
  and exact accounting
- Selection used no shadow audit or privileged labels
- Round-42 score: +1.157 reliability reward points and +2.778 return points;
  maximin +1.157
- Final: +1.157 reliability and +6.250 return reward points
- Final target gains: +6.94 reliability points and +63.89 return points
- Binary and four-rule retention, physical/tensor parity, persistence, and
  exact-resume gates passed
- Matched verifier-reward shuffle failed the full gate

This checkpoint is curated for the next transfer-ledger experiment. Its
weights are not yet claimed to improve a genuinely later held-out task; that
is the next scientific gate.

## Integrated binary and visible-context controller

`unified_binary_context_integrated_seed8397.pt` is the first one-controller
integration of two independently acquired skills.

- SHA-256:
  `332fb1d2d51eea210ac695e101b64fa53ef8ec5059cf1f5bd26755a297089d9c`
- Parent: `unified_memory_online_utility_seed6810.pt`
- Context specialist: `unified_visible_context_seed8383.pt`
- Integration: static behavior routing into a zero-initialized, learned gated
  action residual, then 32 updates of whole-trajectory rehearsal.
- Held-out binary few-shot: 97.42% normal / 98.01% reversed; all binary
  vision, feedback, state-reset, and reversal gates passed.
- Held-out visible context: 95.54% normal / 94.79% counterfactual; blank
  vision was 48.63% and prediction flips were 88.40%.
- An independently seeded static-routing and trajectory-rehearsal pipeline
  replicated both complete gates. The shuffled-specialist control was rejected.

The integration target contained only the two learned controllers' opaque
behavior distributions. Verifier labels, semantic context IDs, and correct
unattempted actions were never used as distillation targets.

## Three-skill compounding controller

`unified_three_skill_compounding_seed8413.pt` is the first controller in this
lineage to acquire a later compositional primitive faster because it retained
the two prerequisite skills.

- SHA-256:
  `0192403bc3d3655a7861947303a42a221645ea9200d8ba508076103fcebb785f`
- Parent: `unified_binary_context_integrated_seed8397.pt`
- New primitive: rendered identity XOR rendered context, selected by a learned
  public visual operation cue.
- A zero-output, sensory-gated relation adapter was added; every pre-existing
  parameter remained bit-identical.
- At 64 updates: 2,048 new lifetimes plus 512 binary and 512 context replay
  lifetimes, or 18,432 total verifier bits.
- On 1,024 held-out lifetimes: 91.57% normal and 91.83% pixel-counterfactual
  new-skill accuracy; removing only the operation cue reduced it to 53.92%.
- Retention: 97.40% binary few-shot and 93.98% direct visible context.
- A shuffled-teacher consolidation control failed the complete gate.

Across paired seeds 8411–8413, the integrated controller reached stable,
retention-safe mastery at 68/72/64 updates versus 76/80/72 for the old parent:
exactly eight updates, 256 new lifetimes, and 384 total lifetimes earlier on
every seed.

## Verified-use memory-volatility controller

`unified_controller_memory_volatility_seed17107.pt` adds one generic temporal
plasticity feature to the seven-feature physical replacement parent.

- SHA-256:
  `da89893ffe67a20907755c48b4dfbd0755a1469dd05ed7211473e07d95c21c07`
- Parent: `unified_memory_persistent_physical_seed7032.pt`
- Only the new volatility coefficient differs from the behavior-preserving
  expanded parent; inherited replacement coefficients are bit-identical.
- Stable mastery arrived after 24 updates and 6,144 verifier bits.
- Held-out valid replacement was 98.83%, versus 57.81% for matched
  reward-shuffled training.
- The selected state passed a 128-bank physical save/reload audit at 100%
  valid replacement; shuffled volatility fell to 52.34% and reversing outcome
  histories reversed 100% of choices.

The promoted physical claim holds admission strength equal during history
collection. Unequal retrieval priors remain an explicitly recorded frontier.

## Reward-selected memory usage-prior controller

`unified_memory_usage_prior_seed17401.pt` closes that unequal-strength boundary
for exact-content retrieval.

- SHA-256:
  `2c1fe6c47a7b13efa1f3cfdc6349260b0f7959443e98e9c3c5a1841ed594cc65`
- Parent: `unified_controller_memory_volatility_seed17107.pt`
- One scalar was added at the old behavior-preserving value `1.0`.
- Five physical candidate clones competed at scales
  `0, 0.25, 0.5, 0.75, 1.0`; only scalar visual verifier reward selected the
  survivor.
- The race used 1,280 verifier bits and selected scale `0.0`.
- Unequal-strength valid replacement improved from 64.06% to 100%, with 98.05%
  visual accuracy.
- Volatility shuffle fell to 46.88%; reversing histories reversed every choice.
- An independent 512-context selective-disk audit passed all gates at 93.55%
  first-reload and 93.36% repeat-reload accuracy.

This checkpoint uses a global content-first scale. Conditional per-query
retrieval remains the next frontier.

## Cross-family pair-relation repertoire

`unified_pair_relation_repertoire_seed9112.pt` is the first unified-controller
checkpoint to add a simultaneous same/different visual primitive while
retaining the three-skill parent.

- SHA-256:
  `50cad66c1853a691e3d426ec522c5758e3d645b354add5c6b31f0891f41f7908`
- Parent: `unified_three_skill_compounding_seed8413.pt`
- Acquisition: 64 updates, 12,288 new-task verifier bits, 30,720 total bits
  including balanced rehearsal, 4.39 seconds on the measured GPU run
- Held-out colors/positions: 99.56%
- Missing second object: 49.12%
- Valid relation counterfactual, prediction-flip, blank-vision, and all three
  inherited retention gates: passed
- Learner-visible information: RGB, opaque attempted action, scalar verifier
  outcome, and the controller's own latent state

The checkpoint does not yet carry an appearance-independent same/different
concept: zero-shot diamonds scored 26.20% and dot pairs 68.65%. Those numbers
are the next gradual curriculum boundary.

## Cross-appearance pair-relation bridge

`unified_pair_relation_appearance_bridge_seed9303.pt` extends the same learned
relation across bars and diamonds.

- SHA-256:
  `3fbb53049a1ecb5496c308eba195371531d1ec87c8be3edb0c3ddf980a0b9919`
- Parent: `unified_pair_relation_repertoire_seed9112.pt`
- Architecture change: 64-unit zero-output nonlinear gate refiner
- Training: 32 acquisition plus 288 consolidation updates
- Experience: 61,440 new-relation verifier bits; 184,320 total with replay
- Held-out: 99.61% bars, 97.52% diamonds
- Retention: 93.37% binary mapping, 91.74% visible context, 90.58%
  visible-context XOR; every complete causal gate passed
- Matched reset-slot control: 97.69% diamonds but only 72.03% bars
- Independent missing-object audit: 49.61%

The fixed consolidation duration is seed-sensitive and dot pairs remain
unmastered at 70.61%, so this is a capability/causal-reuse milestone rather
than a robust stable-bits threshold.

## Three-appearance pair-relation compounder

`unified_pair_relation_three_appearance_seed9622.pt` extends the same relation
to disconnected dot pairs without losing bars, diamonds, or the unrelated
repertoire.

- SHA-256:
  `6dee3d9545f537d041edfe4e7a29df579f41be2b50eae8740d1c06318998ba4e`
- Parent: `unified_pair_relation_appearance_bridge_seed9303.pt`
- Training: 8 acquisition plus 48 consolidation updates
- Experience: 1,792 new lifetimes and 37,632 total verifier bits with replay
- Independent audit: 99.96% bars, 97.83% diamonds, 96.44% dot pairs
- Missing-second-object controls: 49.60%, 50.05%, and 49.90%
- Two-contour versus bars-only lineage advantage: +7.79 percentage points
  averaged over three paired seeds, positive in every pair
- Inference: all three appearances master at one controller pass per event,
  with zero optional thought passes

The 56-update endpoint is seed-sensitive and is not claimed as a robust
stable-bits threshold.

## Robust three-appearance pair-relation controller

`unified_pair_relation_robust_three_appearance_seed9672.pt` replaces the
seed-sensitive endpoint with a replicated fixed recipe and no architecture
change.

- SHA-256:
  `1ff5d38258a8c683fbc7dcfd6a1098e20c18ae35d5372d377fd8874e29544f54`
- Parent: `unified_pair_relation_appearance_bridge_seed9303.pt`
- Training: 64 acquisition plus 32 consolidation updates
- Experience: 3,072 new lifetimes, 18,432 new verifier bits, and 64,512 total
  verifier bits including balanced rehearsal
- Three fresh seeds: 3/3 complete capability and retention gates passed
- Mean held-out bars/diamonds/dot-pairs: 99.61%/96.62%/97.81%
- Independent 8,192-lifetime audit: 99.65%/97.85%/97.78%
- Missing-second-object controls: 49.41%/49.81%/50.01%
- Inference: one controller pass per event and zero optional thought passes

A matched zero-output gate extension also replicated, but the unchanged
architecture passed the same seeds and budget.  The promoted claim is
therefore a robust curriculum threshold, not an architecture improvement.

## Causal pair-magnitude compounder

`unified_pair_magnitude_compound_seed21475.pt` is hosted in the Hugging Face
model repository under `checkpoints/`.

- SHA-256:
  `3717bc318a35c7508c6d9fe7be0b2a196b0883af99d9f08c7205f196c61a7dfa`
- Parent: `unified_pair_relation_robust_three_appearance_seed9672.pt`
- Size: approximately 1.4 MB; 369,926 controller parameters
- Acquisition: 4,096 updates, 131,072 new-task lifetimes, 1,179,648 total
  verifier bits including balanced retention replay
- Independent 8,192/16,384-lifetime magnitude audits: 92.05%/91.96%
- Missing-second-object controls: 60.53%/60.40%; the task's optimal
  one-object shortcut is bounded at 62.5%
- Removing the inherited relation read costs 10.70/10.34 percentage points
- Bars, diamonds, and dot-pair relation skills plus all unrelated replay
  skills retain their complete causal gates
- Inference: one controller pass per event; extra thought reduces accuracy

The promoted skill is bars magnitude. Zero-shot magnitude on unseen diamonds
and dot pairs remains near 60% and is the next gradual appearance bridge.

## Gradual magnitude appearance bridge

`unified_pair_magnitude_gradual_bridge_seed21515.pt` is hosted in the
Hugging Face model repository under `checkpoints/`.

- SHA-256:
  `594f9f45b99c3d6d78536d2d0d1af40cb988e62cb6d128a774a95496ffb4f392`
- Parent: `unified_pair_magnitude_compound_seed21475.pt`
- Size: approximately 1.5 MB; 388,191 controller parameters
- Acquisition: 256 new lifetimes, 1,536 new verifier bits, 8 updates
- Total with seven complete replay streams: 480 lifetimes / 2,880 bits
- Three fresh seeds: 3/3 complete acquisition, causality, and retention gates
- Independent 16,384-lifetime target audit: 91.36%
- Missing-second-object control: 60.52%
- Inherited-read ablation: 79.20%, a 12.16-point causal loss
- Original magnitude, three relation appearances, and all unrelated skills
  retained
- Zero-shot next-rung transfer: 88.57% parent versus 90.68% child
- Inference: one controller pass per event; optional thought hurts

The controller learns the first just-beyond-ability bars→diamonds morph and
zero-shot masters the next two harder morph levels. Full diamonds remain open.

## Fixed-parameter magnitude experience consolidation

`unified_pair_magnitude_experience_consolidation_seed21653.pt` is hosted in
the Hugging Face model repository under `checkpoints/`.

- SHA-256:
  `ffb09143b452f5b9e94b74bc382cf82e83b0e80d7edb1638631d60d4b8d3d6ce`
- Parent: `unified_pair_magnitude_gradual_bridge_seed21515.pt`
- Size: approximately 1.5 MB; 388,191 controller parameters
- Acquisition: 128 new lifetimes / 768 new verifier bits
- Rehearsal: 128 unique lifetimes / 768 verifier bits
- Internal consolidation: 16 optimizer passes / 4,096 lifetime exposures
- Three fresh seeds: 3/3 acquisition, causality, and retention gates
- Independent 32,768-lifetime target audit: 90.22%
- Missing-second-object control: 60.61%
- Inherited-read ablation: 78.50%, an 11.71-point causal loss
- Parent fails at unseen 20.5078%; child passes at 90.17% and also masters
  20.7031%
- Inference: one controller pass per event; optional thought hurts

The architecture does not grow. Reusing one diverse experience packet beats
both one-pass learning and a four-times-larger fresh stream at the matched
optimizer budget. This is an endpoint experience budget, not yet a minimum
stable bits-to-threshold.

## Half-compute magnitude successor

`unified_pair_magnitude_half_compute_seed21702.pt` is hosted in the Hugging
Face model repository under `checkpoints/`.

- SHA-256:
  `e3ae0cd90ec0dc6f2e98c829c2c064d7a6a6008b36fb982213b3b50c795e8ba9`
- Parent: `unified_pair_magnitude_experience_consolidation_seed21653.pt`
- Size: approximately 1.5 MB; 388,191 controller parameters
- New experience: 128 lifetimes / 768 verifier bits
- Total unique evidence with ten replay streams: 288 lifetimes / 1,728 bits
- Consolidation: 8 optimizer passes / 2,304 lifetime exposures
- Stable prefix ladder: 4/6/7 failed; 8/12/16 passed
- Three fresh seeds: 3/3 acquisition, causality, and retention gates
- Independent 32,768-lifetime audit: 90.21%
- Missing-second-object control: 60.64%
- Inherited-read ablation: 78.12%, a 12.09-point causal loss
- Inference: one controller pass per event; optional thought hurts

The separate unseen-rung gain gate missed its registered threshold and is not
part of this checkpoint's promoted claim.

## Compounding magnitude successor

`unified_pair_magnitude_compounding_seed22022.pt` is hosted in the Hugging
Face model repository under `checkpoints/`.

- SHA-256:
  `5aa030f0fb11d0765752f05cf6c6ecb6334ee31fa1b12a41eeef2603212fe1d4`
- Parent: `unified_pair_magnitude_half_compute_seed21702.pt`
- Size: approximately 1.5 MB; unchanged 388,191 controller parameters
- Genuine frontier: 21.484375% bars→diamonds contour
- New experience: 96 lifetimes / 576 verifier bits, down 25% from the
  preceding 128-lifetime acquisition rung
- Total unique evidence: 228 lifetimes / 1,368 bits
- Consolidation: 12 passes / 2,736 lifetime exposures
- Evidence/compute ladder: 64×8 failed 0/3; 96×8 passed 1/3; 96×12 passed 3/3
- Matched population: parent 0/8 mastery, child 8/8, +0.4677 percentage
  points mean accuracy; every stream improved
- Reset inherited knowledge: 87.95%; shuffled outcomes: 89.39%
- Independent 32,768-lifetime audit: 90.45%; every causality and retention
  gate passed

This is the first magnitude rung in this lineage where accumulated skill
reduces the next genuine acquisition's new verifier experience. It deliberately
spends more private consolidation compute to do so.

## Repeated-compounding magnitude successor

`unified_pair_magnitude_repeated_compounding_seed23105.pt` is hosted in the
Hugging Face model repository under `checkpoints/`.

- SHA-256:
  `c136841d60a5220bd09cd12029b6d59d903dc73d5deddb39248e7327ae48f2a2`
- Parent: `unified_pair_magnitude_compounding_seed22022.pt`
- Size: approximately 1.5 MB; unchanged 388,191 controller parameters
- Genuine frontier: 22.65625% bars→diamonds contour
- New experience: 44 lifetimes / 264 verifier bits, down 54.2% from the
  preceding 96-lifetime acquisition
- Total unique evidence: 188 lifetimes / 1,128 bits
- Consolidation: 12 passes / 2,256 lifetime exposures
- Evidence floor: 32 failed; 40 failed; 42 passed 1/3; 44 passed 3/3
- Matched population: parent 2/8 mastery, child 8/8, +0.2683 percentage
  points mean accuracy; every stream improved
- Reset inherited knowledge: 87.62%; shuffled outcomes: 88.66%
- Independent 32,768-lifetime audit: 90.26%; every causality and retention
  gate passed

This is the second consecutive magnitude rung where accumulated skill reduces
the experience needed for a harder frontier.

## Intention-only amodal migration checkpoint

`unified_repertoire_span2_amodal_intention_seed122005.pt` is the promoted
behavior-preserving successor to the five-capability repertoire checkpoint.

- SHA-256:
  `9eea7ab479cb8450737f040b76495cc5ec737e970cdc165af2446873e530cd6c`
- Parent: `unified_repertoire_span2_strict_seed122005.pt`
- Migration cost: zero examples, verifier bits, and optimizer updates
- Mechanism: fold the learned two-action residual through the frozen decoder's
  minimum-norm right inverse into a 24-dimensional intention residual
- Paired parity: zero action flips across 12,288 decisions; maximum logit drift
  `5.72e-6`
- Independent audit: all five repertoire gates passed on 4,096 lifetimes
- Compatibility suffix: structurally zero

This checkpoint removes active device-protocol content from the migration
suffix. It does not yet establish variable-N inputs or multiple output decoders.
