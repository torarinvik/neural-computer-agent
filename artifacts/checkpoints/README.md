# Curated checkpoints

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
