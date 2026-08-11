# Related-key temporal content retrieval

This two-seed promotion composes same-cue query-conditioned temporal address
growth with the canonical persistent append-only content-addressed memory. The
memory stores two learned event keys and opaque capability-address values. The
controller, learned event encoder, and external capability file are frozen
before retrieval; only the external memory is read during the probes.

Exact keys and nearby learned keys, made with a fixed 20% normalized
perturbation, recovered offsets `4` and `5` at `1.0000` accuracy on every
retained lifetime for seeds `17` and `18`. Cosine scores for the related-key
reads were `0.9712`/`0.9870` and `0.9841`/`0.9869` respectively.

The promotion also required an explicit unknown-key no-hit, clear removing
hits, exact reload preserving related-key reads, checksum-corruption rejection,
unchanged controller/event-encoder/file digests, and zero replay. Each seed
consumed `313,856` unique verifier bits, `33,024` logical lifetimes, `512`
optimizer updates, `520` route-memory updates, and two content-memory writes.

This qualifies one bounded related-key content-addressed retrieval composition.
It does not establish learned compression, capacity management, arbitrary new
computation, unrestricted memory growth, or general continual learning.

Reports are `seed-17.json` and `seed-18.json`. The experiment is implemented
in `experiments/brainworkshop_canonical/external_temporal_content_retrieval_growth.py`.
