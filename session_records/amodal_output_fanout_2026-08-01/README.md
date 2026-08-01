# Amodal output fan-out milestone — 2026-08-01

## Result

Gate 3c passes. One immutable controller intention now drives two independently
owned backends at the same time: the inherited opaque-action decoder and a
separately calibrated two-command protocol decoder whose numbering is reversed.

The controller, vision frontend, inherited decoder, and memory parameters were
frozen. The new decoder received only the 24-dimensional base intention, its
own attempted opaque command, and one scalar success bit. It never received a
correct command, unattempted outcomes, task identity, primary-decoder logits,
or primary-decoder weights.

## Fast experiment ladder

Binary-only calibration looked strong statically but failed the span-two
sequence-reversal causal gate. Increasing that diet from 640 to 1,920 verifier
bits did not repair it. A binary/span mixture also failed the closed-loop audit.

The smallest targeted adjacent curriculum won: span-two outcomes alone. Each
update supplied 64 attempted-command outcome bits. Three independent decoder
seeds crossed the static 85% gate after the first 64 bits, trained for only 768
bits total, and passed the five-capability closed-loop audit at 512 held-out
lifetimes. More diverse experience was not automatically better; training on
the distribution that exposed the causal weakness produced the most reusable
decoder.

## Adversarial controls

- A matched reward-shuffled span-two run did not pass or cross stably.
- Shuffled intentions and zero intentions stayed near 50% on every capability.
- The inherited decoder's simultaneous logits were bit-exact with direct use.
- All controller and decoder parameters were unchanged during audit.
- The output bus accepted zero, one, or two backends without resizing the core.
- A closed-loop alternate-backend audit lowered the reversed protocol to
  canonical legacy action IDs solely to reuse the historical harness. It passed
  rule reversal, memory corruption/reset, missing-evidence, candidate-flip, and
  true sequence-reversal controls.

## Promoted 4,096-lifetime audit

| Capability | Accuracy |
| --- | ---: |
| Binary mapping | 98.04% |
| Four-rule binding | 99.62% |
| Relation — bars | 96.87% |
| Relation — diamonds | 99.98% |
| Relation — dot pairs | 98.41% |
| Persistent memory | 98.63% |
| Span two | 97.13% |

For span two, full memory reset was 50.07%, blank presentation was 49.80%,
candidate intervention flipped 94.20% of changed answers, and true sequence
reversal flipped 88.60%.

The simultaneous-fanout audit at 512 lifetimes also passed all five task
families. Protocol accuracy ranged from 96.88% to 100%; intention-shuffled and
zero-intention controls remained at chance.

## Artifacts

- Controller:
  `artifacts/checkpoints/unified_repertoire_span2_amodal_intention_seed122005.pt`
- Protocol decoder:
  `artifacts/checkpoints/opaque_protocol_decoder_span2_seed133001.pt`
- Decoder SHA-256:
  `0258822d056a0bc5cf430a3035d81f84ede477eb0dbdd7fc9365d6be66bb03a7`
- Training, replication, negative-control, fan-out, and closed-loop reports are
  stored beside this README.

## Honest boundary and next frontier

This proves independently learned M-output fan-out from one frozen intention;
it does not yet prove a device-independent recurrent feedback interface. The
legacy controller still receives canonical previous-action IDs internally. An
alternate decoder therefore needs a thin command-to-canonical-action lowering
when it drives the historical closed loop.

The next architecture gate is variable N-input composition, beginning with an
exact N=1 control and then redundant N=2 events. Complementary N=2 evidence
must eventually make neither frontend sufficient alone.
