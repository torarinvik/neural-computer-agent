# Online learned-context nonlinear retention — promoted bounded result

Across seeds `1601` and `1602`, four disjoint nonlinear transition streams
arrived one row at a time. The opaque context encoder formed candidate keys;
each candidate consumed 64 rows without replay, and every promotion passed
the `0.02` held-out floor while retention probes protected earlier slots.
Exact router persistence passed for both seeds.

This promotes bounded online replay-free nonlinear retention with an
untrained-but-opaque context encoder and four-slot capacity. It does not
establish general continual learning or unrestricted growth. Reports are
protected by `SHA256SUMS`.
