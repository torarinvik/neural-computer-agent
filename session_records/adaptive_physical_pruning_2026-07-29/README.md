# Verified-history adaptive physical pruning

## Result

The controller's verified experience can now decide which consolidated memory
banks deserve a larger physical diversity reserve.

Every bank begins with two representatives for each relation-discovered
behavior. A third representative remains volatile unless a past deep read
causally rescued an error: deep was requested, deep succeeded, and the
corresponding shallow read failed. No appearance name, rule bit, semantic class,
or correct pruning action is visible to the policy.

The row-specific hypothesis was rejected cheaply. Which exact third row was
used did not transfer reliably. The promoted task-agnostic rule protects both
extra rows in a bank after any one of them earns causal protection. This
preserves within-class diversity without pretending that an individual latent
is a permanent semantic object.

## Replicated graduation

All runs use the immutable adaptive-read checkpoint, 4,096 disjoint physical
banks, 16 history rounds, and three held-out future rounds.

| metric | seed 44001 | seed 45001 | seed 46001 |
|---|---:|---:|---:|
| full six-row accuracy | 99.6012% | 99.5687% | 99.5985% |
| adaptive-pruned accuracy | **99.5646%** | **99.5266%** | **99.5361%** |
| fixed four-row accuracy | 99.4832% | 99.4344% | 99.4358% |
| exact-budget shuffled-history accuracy | 99.5076% | 99.4602% | 99.4561% |
| adaptive mean physical rows | **4.344** | **4.340** | **4.371** |
| logical rows versus full | **72.46%** | **72.37%** | **72.88%** |
| mean logical comparisons | 2.074 | 2.083 | 2.078 |
| corrupted-memory accuracy | 49.99% | 49.99% | 49.98% |
| exact compact disk reloads | 128/128 | 128/128 | 128/128 |
| serialized-byte ratio | 84.72% | 84.52% | 86.06% |

All pre-registered gates pass in all three reports. Logical payload falls by roughly
28%; serialized files fall by roughly 15% because every tiny per-bank file
retains fixed PyTorch and history metadata.

The learned placement matters. A stratified shuffle preserves the exact
physical-row budget but loses 0.057–0.080 percentage points.
Zeroing memory values collapses to chance. The reversed-rule task
stays within 0.075 percentage points of the full store. Compaction physically
removes rows, preserves access/success/volatility histories, and reloads them
exactly.

## Sample and compute accounting

Each graduation run observes 393,216 history verifier outcomes:

`4,096 banks × 2 balanced rules × 3 appearances × 16 rounds`.

These are online experience outcomes, not semantic supervision and not
controller optimizer examples. The future audit consumes 73,728 disjoint
verifier outcomes. No controller tensor changes during this experiment.

The short ladder was deliberately run first:

- four history rounds: about 4.17 rows, small but real gain over fixed four;
- eight rounds: 4.23 rows and 99.5294% held-out accuracy;
- twelve rounds: 4.29 rows and 99.5483%;
- sixteen rounds: stable graduation on two new seeds.

The 16-round point is promoted because accuracy is primary. The 8- and 12-round
points remain valid lower-experience/lower-storage operating points.

## Claim boundary

Demonstrated:

- scalar verified outcomes create a useful causal protection signal;
- protection generalizes to disjoint future queries;
- causal history placement beats a matched shuffle;
- the policy physically shrinks disk-backed memory and reloads exactly;
- adaptive storage composes with adaptive read depth;
- old controller weights are untouched.

Not demonstrated:

- protection across unrelated cognitive primitives;
- an optimal decay/thaw schedule under a changing environment;
- net wall-clock gains once consolidation and compaction overhead are included;
- that 16 history rounds are the globally most sample-efficient operating point.

## Artifacts

- `graduation_seed44001.json`
- `graduation_seed45001.json`
- `graduation_seed46001.json`
- `rep_seed43001_h8.json`
- `rep_seed43001_h12.json`
- `bankwise_h4.json`
- `bankwise_h16.json`

The executable audit is
`experiments/archive/unified_cognitive_controller/audit_adaptive_physical_pruning.py`.
