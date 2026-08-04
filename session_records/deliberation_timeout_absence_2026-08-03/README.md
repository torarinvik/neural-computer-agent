# Bounded missing-evidence timeout

This rung extends the promoted outcome-only `WAIT`/`THINK`/`COMMIT` policy
through the production `AmodalEventWindowBuffer`. The controller first trains
on balanced complete, delayed, and think-required episodes. Its transport and
execution heads are then frozen; only the age-gated timeout residual is
trained on balanced complete/delayed/missing/think-required episodes.

The audit uses timestamped opaque events, out-of-order complete arrivals,
`0.1` timestamp jitter under a `0.25` buffer tolerance, and a permanently
withheld partner for the missing condition. Each learned episode is paired with
an immediate-`COMMIT` control using the same verifier sequence.

Across seeds 17, 18, and 19, the learned policy selects `COMMIT` on complete
windows, `WAIT` on delayed windows, `THINK` on think-required windows, and
`WAIT` followed by timeout `COMMIT` when the partner is absent. Paired mixed
utility gains over immediate commit are `0.0561`, `0.0678`, and `0.0551`.
An indefinite-wait policy has no finite utility on the missing condition
because it never completes; the promoted policy terminates after one bounded
quiet tick instead.

This promotes a narrow bounded missing-evidence termination primitive. It does
not qualify broad absence inference, learned delay compensation, or
multimodal transfer.
