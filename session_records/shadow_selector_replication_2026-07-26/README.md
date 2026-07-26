# Shadow-selector replication and retention localization

Date: 2026-07-26

## Frozen-rule replication

The promoted round-18 selector was applied unchanged to fresh physical stream
7076 and new clones 7110–7113. Conservative shadow scores were:

| Clone | Round-18 shadow score |
|---|---:|
| 7110 | -1.04 points |
| 7111 | 0.00 points |
| 7112 | **+6.25 points** |
| 7113 | 0.00 points |

Only selected 7112 and fixed lowest-ID control 7110 were extended. Selected
7112 learned the reliability context better (+20.83 target points versus
+4.17), so acquisition direction replicated. Old-return magnitude did not:
7112 gained only +1.39 target points and +0.23 reward points; control gained
0.00 target points and +0.46 reward points.

The round-18 audit is therefore an acquisition selector, not a demonstrated
retention selector.

## Rejected round-36 proxy

The same read-only bank audit was added after the reliability block. Every
audited 36-round trace exactly matched its corresponding unaudited full prefix.
Nevertheless, the proxy ranked weak 7100 above strong 7101 on stream 7075.
Read-only strategy-bank ceiling at round 36 is rejected as a predictor of
return learning.

## Actual-return selector

Retrospective prefixes from all completed populations show that mean verified
reward advantage over the first six genuine old-return rounds selects the
eventual best safe clone:

- stream 7074: clone 7092;
- stream 7075: clone 7101;
- stream 7076: clone 7112.

One return round is noisy. Three rounds tie 7091 and 7092 on stream 7074. Six
rounds separate them and are retained as the candidate retention rung.

## Next gate

Build exact resumable prefix checkpoints. Then prospectively run a two-stage
race: all clones to round 18 plus the shadow acquisition audit, top candidates
to round 42, and only the six-round return winner to round 54. This is expected
to reduce training compute while preserving unique-experience accounting.
