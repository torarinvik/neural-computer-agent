# Same-cue query-conditioned temporal address growth

This two-seed promotion is a stronger pressure test than cue-to-file routing.
Every episode uses the same rendered cue (`12`). The first learned query event
is the opaque address key: query `0` requires a private n-back-4 relation and
query `1` requires a private n-back-5 relation. The external route table must
retain offset `4` for the first key and acquire offset `5` for the second.

The source external capability file first learns a generic readout and its
first offset from scalar outcomes. Its complete state is then frozen. Without
replay, only the external context-keyed route table is allowed to grow. Across
seeds `17` and `18`, source and target accuracy were `1.0000` on every retained
lifetime, with selected offsets `[4]` and `[5]`. The readout, controller, and
event encoder remained byte-stable.

Controls reject shortcut explanations: an unseen query falls back to offset
`1` and remains below mastery (`0.6042–0.6389`); forced offset `1` remains
below `0.657`; missing history remains below `0.80`; shuffled scalar outcomes
do not acquire offset `5`; route reload is exact; and replay is zero.

Each seed consumed `313,856` unique verifier bits, `33,024` unique logical
lifetimes, `512` optimizer updates, `520` route-memory updates, and zero
replayed examples. This promotes bounded same-cue multi-address acquisition
through learned query event keys. It does not establish content search,
learned compression, unrestricted memory growth, arbitrary new computation,
or general continual learning.

Reports are `seed-17.json` and `seed-18.json`. The experiment is implemented
in `experiments/brainworkshop_canonical/external_temporal_query_address_growth.py`.
