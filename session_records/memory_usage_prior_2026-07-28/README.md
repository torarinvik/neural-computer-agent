# Reward-selected retrieval prior fixes unequal physical memory

## The failure

The temporal-volatility controller reached 100% physical replacement only when
stored rows had equal admission strength. With naturally unequal learned
strengths, retrieval scored:

`cosine similarity + log(write strength)`

The prior was sometimes large enough to redirect an exact query to a nearby
high-strength row. Volatility histories then described the rows actually read,
not the presumed source rows, and correct replacement fell to 64–73%.

## Backward-compatible mechanism

The controller now optionally owns one scalar
`memory_usage_prior_scale`. It is initialized to `1.0`, making insertion
bit-identical to every previous checkpoint. Physical and tensor retrieval use:

`cosine similarity + scale * log(write strength)`

Old checkpoints omit the parameter and continue to use `1.0`.

## Reward-only five-way race

Five temporary controller clones used scales:

`0.0, 0.25, 0.5, 0.75, 1.0`

Every clone acted on the same physical banks. Selection consumed only mean
pixel-task verifier reward. Stable/decoy labels and correct replacement actions
were private audit metadata and never entered selection.

Each normal race used four fresh rounds of 16 banks:

- 448 unique logical contexts;
- 20 counterfactual candidate evaluations;
- 1,280 verifier bits;
- zero gradient updates;
- zero replayed examples;
- under 25 seconds total.

| seed | selected scale | parent valid | selected valid | visual accuracy | shuffled valid | reversed stable eviction |
|---:|---:|---:|---:|---:|---:|---:|
| 17401 | **0.0** | 64.06% | **100%** | 98.05% | 46.88% | **100%** |
| 17402 | **0.0** | 64.06% | **100%** | 99.02% | 50.00% | **100%** |

The reward-shuffled control used twice as many rounds and 2,560 verifier bits.
It selected `0.5`, reached only 69.53% valid replacement, and failed every
learning/causality gate while retaining unrelated old skills.

All physical histories persisted exactly and every bank stayed bounded.
Reliability-dominant utility, binary mapping, and four-rule behavior passed.
Only `memory_usage_prior_scale` changed.

## Prior selective-disk retention

Content-first scale `0.0` was then tested on the older sparse write/read loop:

| seed | first reload | repeat reload | corrupt values | duplicate rows |
|---:|---:|---:|---:|---:|
| 17404 | 93.55% | 93.36% | 66.41% | 10.55% |
| 17405 | 92.77% | 93.36% | 66.60% | 11.52% |

Both 512-context audits passed every pre-existing gate. No-memory accuracy was
50%; tensor sparse accuracy was 99.61–100%.

## Promoted checkpoint

`artifacts/checkpoints/unified_memory_usage_prior_seed17401.pt`

SHA-256:
`2c1fe6c47a7b13efa1f3cfdc6349260b0f7959443e98e9c3c5a1841ed594cc65`

## Scope and next frontier

The result establishes that the controller can use a tiny, reward-selected
resource parameter to repair a real interaction between memory admission and
retrieval, with only 1,280 verifier bits.

It does not show that zero usage prior is universally optimal. Ambiguous or
duplicate memories may benefit from verified utility as a tie-breaker. The next
breakthrough target is a per-query conditional scale learned from generic
confidence, margin, and outcome history, with exact-content and duplicate-row
arms in the same curriculum.
