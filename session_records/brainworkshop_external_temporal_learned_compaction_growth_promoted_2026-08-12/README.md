# Learned live-memory compaction selection

This two-seed composition transfers the existing opaque consolidation learner
onto the canonical persistent append-only content-memory ABI. The policy is
trained from scalar duplicate-rewrite utility, then receives only learned event
keys, learned values, strength, and age for a live three-row memory: an exact
source key, a nearby source alias sharing its opaque address, and a target key.

On all six physical row permutations for seeds `17` and `18`, the learned
policy selected the redundant source/alias pair, while the untrained policy
selected it only twice. The held-out verifier accepted the live compaction,
which saved one row; reload and checksum-corruption controls passed. The
controller and event encoder were frozen and replay was zero. Each seed used
`229,376` scalar utility observations, `512` optimizer updates, seven memory
writes, and six permutation probes.

This promotes transfer of learned opaque proposal selection into live external
memory. It does not establish end-to-end capability acquisition, arbitrary
compression, unrestricted memory growth, arbitrary new computation, or general
continual learning.

Reports are `seed-17.json` and `seed-18.json`. The experiment is implemented
in `experiments/brainworkshop_canonical/external_temporal_learned_compaction_growth.py`.
