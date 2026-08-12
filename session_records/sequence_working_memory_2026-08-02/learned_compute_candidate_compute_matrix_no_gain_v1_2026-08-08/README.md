# Protected-compute matrix audit — no new compression gain

Date: 2026-08-08

The adapter-sharing growth policy was extended to fresh-probe a new adapter
against every protected physical compute slot before allocating fresh compute.
Both five-capability seeds pass mastery, retention, frozen-core, reload,
corruption, and no-replay gates, but both still select fresh compute for the
hard middle capability:

| seed | final behavior | physical compute | physical adapters | outcome |
| --- | --- | ---: | ---: | --- |
| 69316 | `1.000 / 0.895 / 1.000 / 0.797 / 0.766` | 5 | 2 | no new compute compression |
| 69317 | `1.000 / 0.750 / 0.867 / 0.855 / 0.770` | 4 | 2 | no new compute compression |

The matrix admission mechanism is retained as a reusable ABI improvement, but
the experiment does not promote a compute-consolidation result. Fresh compute
growth remains the next compression bottleneck.

Reports are covered by `SHA256SUMS`.
