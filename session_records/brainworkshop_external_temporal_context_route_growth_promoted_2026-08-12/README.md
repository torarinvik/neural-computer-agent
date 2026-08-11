# Outcome-only context-conditioned temporal routing

This two-seed promotion composes the external temporal history contract with
the persistent opaque context-route table. A frozen controller and learned
event encoder feed two isolated capability files. Each file learns its own
relative history offset from scalar verifier outcomes; the route table then
learns which file to select from a normalized learned event key, also using
only terminal scalar episode outcomes.

The source file learns n-back-4 at rendered cue `11`. Without replay, a fresh
file learns n-back-5 at cue `12`. Across seeds `17` and `18`, both files reach
stable direct mastery, both contexts select the correct file on every retained
lifetime, and routed n-back-5 reaches `1.0000` and `0.9514`. The source file
remains at `1.0000` after growth and is byte-identical. The files learn modes
4 and 5 respectively.

Controls reject the shortcut explanations: an unknown cue falls back to the
oldest file and remains below the `0.80` mastery threshold (`0.6372` and
`0.6484`); wrong-file accuracy remains below `0.674`; wrong-offset accuracy
remains below `0.667`; missing history remains below `0.80`; and a
one-step-shuffled route-outcome control never selects the new file. Context
table reload is exact, the controller and event encoder are unchanged, and
replay is zero.

Each seed consumed `387,584` unique verifier bits, `41,216` unique logical
lifetimes, `1,024` optimizer updates, `264` route-memory updates, and zero
replayed examples. The result promotes a bounded composition: learned
context-conditioned routing into isolated temporal capability files. It does
not establish same-context multi-address binding, content search, learned
compression, unrestricted memory growth, arbitrary new computation, or
general continual learning.

Reports are `seed-17.json` and `seed-18.json`. The experiment is implemented
in `experiments/brainworkshop_canonical/external_temporal_context_route_growth.py`.
