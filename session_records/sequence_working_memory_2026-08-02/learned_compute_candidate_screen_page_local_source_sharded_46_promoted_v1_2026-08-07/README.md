# Page-local source sharding promoted (2026-08-07)

The 46-candidate source boundary is repaired by splitting the 20 protected
source rows into two independent normalized pages of ten candidates each. The
second source page is exposed only after scalar verifier failure for the first;
the 26 unseen rows remain raw identity append pages with representation-matched
copy-on-write priors.

Both seeds pass strict per-candidate known and unseen mastery at `1.0000/1.0000`,
known and unseen candidate permutation, reward-shuffled null, source-page
immutability, reload, frozen-core, and zero-replay gates. The result uses 512
updates per source page, 512 raw-prior updates, and 32 fresh updates per
unseen append page: 2,496 optimizer updates and 477,696 verifier bits. The
unsharded page-local 46 control required 3,008 updates and 1,423,872 verifier
bits and still failed source retention.

This promotes bounded source-competition isolation and a concrete
sample-efficiency gain. Page order is still a physical experimental factor;
learned page retrieval, arbitrary page composition, and general continual
learning remain open.
