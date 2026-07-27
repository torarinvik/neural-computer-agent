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
