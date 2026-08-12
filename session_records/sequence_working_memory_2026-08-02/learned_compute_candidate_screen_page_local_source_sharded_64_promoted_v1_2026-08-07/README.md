# Page-local source sharding at 64 candidates promoted (2026-08-07)

The source-sharded page-local boundary scales from 46 to 64 opaque candidates:
30 protected source rows occupy three independently trained normalized pages of
ten, and 34 unseen rows occupy 17 raw identity append pages. Later pages open
only after cumulative scalar verifier failure.

Both seeds pass strict per-candidate known and unseen mastery at `1.0000/1.0000`,
known and unseen permutation, reward-shuffled null, source-page immutability,
reload, frozen-core, and zero-replay gates. The run uses 512 updates per source
page, 512 raw-prior updates, and 32 fresh updates per unseen page: 3,136
optimizer updates and 652,800 verifier bits.

This promotes a larger bounded source-competition isolation result. Page order
remains physical rather than learned, and arbitrary page retrieval,
representation selection, and general continual learning remain open.
