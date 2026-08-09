# Promoted online transition-context identity rung

Two seeds (`70011`, `70012`) test an alternating opaque transition stream.
Individual rows are deliberately insufficiently identifying in this fixture,
so `ExternalOnlineTransitionContextRouter` accumulates a 12-row current-stream
window and uses aggregate factual prediction error plus a best-vs-second-best
margin. Ambiguous windows are never written to a model.

The router correctly assigned base and auxiliary windows to their prior slots,
admitted the held-out target exactly once without a regime label, and reused
that slot after returning to the target later in the stream. Target mastery was
`1.0` in both seeds after 22 optimizer updates, versus 31 and 37 for matched
fresh targets. Both prior slots remained byte-stable with `1.0` retention. Old
prior rows replayed during target adaptation were zero; current target-window
replay is reported separately. A fourth regime at the three-slot capacity
limit was refused without growth or writes. Wrong-context MSE, corruption,
frozen-controller, and exact persistence controls passed.

This promotes bounded online identity and ambiguity-safe routing, not general
continual learning. The context encoder is pretrained, identity is windowed,
capacity pressure is refused rather than solved, and current target windows
are replayed. The next step is verifier-gated growth, consolidation, and
compression while retaining all alternating capabilities.
