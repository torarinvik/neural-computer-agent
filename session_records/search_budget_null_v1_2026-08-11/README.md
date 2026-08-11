# Search budget is not the bottleneck (F145)

Probe 245, paired on 2 seeds against F143 (depth 4, beam 4).

    depth 6 : +0.1008 / +0.1337  mean +0.1172  (+0.0015)
    beam 8  : +0.0959 / +0.1350  mean +0.1154  (-0.0003)
    base    : +0.0980 / +0.1334  mean +0.1157

Both nulls. F110's remaining 31.9% contains no purchasable search
component; it sits in the model. Predicted in advance by
docs/LITERATURE.md from the compounding-model-error literature.

Next suspect: the state abstraction (nearest object per polarity only),
which F109 tested when the value model was still the binding
constraint and which now deserves re-testing.
