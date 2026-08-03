# Latent-basis alignment through paired sensory consistency (2026-08-03)

## Question

Can a new encoder with a different latent coordinate system become useful to a
frozen controller without semantic labels, task IDs, correct-action labels, or
verifier outcomes? The diagnostic wraps a copied vision frontend in a fixed
reverse permutation of its latent coordinates. A trainable linear adapter must
recover the controller's existing basis. The controller, original encoder,
input bus, and decoder remain frozen; only the second encoder's adapter is
trainable.

## Results

The self-supervised signal was paired encoded-event consistency: for the same
unlabeled raw frame, the new stream's adapted encoding is pulled toward the
original stream's encoding. No semantic or verifier bits are used. Each run
used 32 updates, batch size 256, all three appearances (`bars`, `diamonds`,
`dot_pairs`), and 147,456 unlabeled calibration frames. Two identity-initialized
seeds and one randomly initialized adapter passed the pre-registered gates:

| seed | bars fused | diamonds fused | dot-pairs fused | controller changed |
|---:|---:|---:|---:|:---:|
| 2871 | 96.04% | 96.27% | 95.90% | no |
| 2872 | 96.07% | 96.43% | 95.98% | no |
| 2873 (random init) | 98.09% | 95.31% | 93.93% | no |

The individual streams stayed near chance, shuffled partners stayed near
chance, and contradictory partners caused the expected prediction reversals
(approximately 80--93% across appearances). Full N=1 behavior remained about
98--100%, within the retention gate. The random-init curve crossed the useful
composition range around update 8, rather than relying on the adapter's
identity initialization. The reports contain the complete curves and hashes.

This qualifies a narrow but important property: **paired, unlabeled sensory
consistency can align a new neural-IR basis while the sovereign controller is
frozen, and the aligned stream participates in an already learned
cross-stream relation.** It is not a claim that arbitrary natural audio,
language, or a cold-start encoder will align without paired correspondence.

## Reward-only controls

Three reward-only arms were deliberately kept as negative evidence. They
briefly improved fused accuracy on their small training stream, but failed
shuffled/contradictory controls and did not cross the pass gate:

| arm | final fused | shuffled partner | contradictory flip |
|---|---:|---:|---:|
| bus + adapter, seed 285001 | 79.38% | 53.01% | 55.82% |
| bus + adapter, seed 285002 | 75.90% | 51.09% | 53.67% |
| adapter only, seed 285003 | 71.64% | 58.44% | 39.06% |

This is evidence against promoting sparse reward as the first alignment
signal—not evidence that reward learning is impossible at a larger, properly
curriculated scale. It is also a reminder that a high fused score without a
causal contradiction audit is not sufficient.

## Sample-efficiency interpretation

The alignment run used no verifier bits and learned from a dense, automatically
available relation between two views of the same sensory event. That is a
useful sample-efficiency result, but its unit is unlabeled paired frames, not
reward episodes. The next comparison must measure the cost of acquiring those
pairs and test whether the adapter can align a genuinely different raw
modality under the same no-semantic-label rule.

## Artifacts and next gate

- `self_supervised_seed2871.json`
- `self_supervised_seed2872.json`
- `self_supervised_random_seed2873.json`
- `reward_only_bus_adapter_seed285001_u4.json`
- `reward_only_bus_adapter_seed285002_u4.json`
- `reward_only_adapter_only_seed285003_u16.json`
- controller: `artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt`
- input bus: `artifacts/checkpoints/amodal_input_bus_complementary_seed145001.pt`

No production checkpoint is promoted from this diagnostic. The next highest-ROI
step is to save the adapter as an independently loadable artifact, replay the
frozen-controller composition audit from that artifact, and then repeat the
paired-consistency alignment with a non-vision encoder (audio or token stream)
before allowing any controller or memory updates.
