# Transactional long-term-memory updates

## Breakthrough

The external memory can now treat a learned rewrite as a transaction:

1. clone the current memory;
2. apply the candidate update only to the clone;
3. score old-skill retention and new-skill gain with verifier callbacks;
4. commit the clone only when all safety gates pass;
5. otherwise return an exact rollback.

The verifier interface sees only latent memory and produces scalar scores. It
does not add task labels, semantic fields, or hard-coded skill identities to
the memory system.

## Deterministic probe

The harmful candidate replaced an old skill with a new one. New-task score
improved by `+1.0`, but old retention fell from `1.0` to `0.5`; the transaction
was rejected and the keys/values were unchanged. A safe candidate replaced a
decoy row: new-task gain was `+1.0`, old retention stayed at `1.0`, and the
candidate committed. The committed tensors survived disk save/reload exactly.

| gate | result |
|---|---:|
| harmful update rejected | pass |
| harmful update rolled back exactly | pass |
| safe update committed | pass |
| positive candidate gain | pass |
| old-skill retention | pass |
| disk persistence | pass |

The executable probe is
`experiments/unified_cognitive_controller/probe_memory_transaction.py`.
The reusable transaction API is
`DiskLatentMemory.transactional_replace()` in
`experiments/unified_cognitive_controller/memory.py`.

## Learned-head integration and adversarial audit

The promoted receipt-trained plasticity head is now connected to the
transaction path by
`experiments/unified_cognitive_controller/audit_transactional_plasticity.py`.
The audit uses the same latent receipt histories for two arms:

* **learned:** the eight-feature head chooses the replacement row;
* **adversarial control:** a verifier-side control deliberately targets a
  stable row, without exposing semantic labels to the learner.

With 32 banks of capacity 6, the learned arm committed 32/32 proposals, with
32 positive candidate gains and zero observed unguarded or guarded forgetting.
The adversarial arm caused 8 unguarded regressions; all 8 were rolled back by
the transaction gate, with zero guarded regressions. Twenty-four adversarial
updates were safe enough to commit. The complete JSON report is
`learned_plasticity_transaction_audit.json`; all runs completed in 0.83 seconds
on CPU and committed-memory disk round trips were exact.

| arm | proposals | commits | rollbacks | unguarded forgetting | guarded forgetting |
|---|---:|---:|---:|---:|---:|
| learned head | 32 | 32 | 0 | 0 | 0 |
| adversarial control | 32 | 24 | 8 | 8 | 0 |

This is an integrated safety result, not yet proof of open-ended continual
learning. The next rung is to evaluate the same transaction mechanism over a
longer stream of genuinely novel primitives, with explicit old-skill
retention, memory-corruption dependence, and sample-efficiency measurements.

## Persistent multi-round stream audit

The longer-stream rung now keeps each physical bank alive while six
independent candidate streams arrive. Across 32 banks this produced 384
proposals in 5.52 seconds:

| arm | proposals | commits | strict positive commits | rollbacks | unguarded forgetting | guarded forgetting |
|---|---:|---:|---:|---:|---:|---:|
| learned head | 192 | 122 | 49 | 70 | 69 | 0 |
| adversarial control | 192 | 101 | 33 | 91 | 89 | 0 |

The learned arm accumulated up to six old-task verifiers on the same bounded
physical banks. Seventy learned proposals were rejected because their
candidate would have damaged an earlier verifier; none of those regressions
reached committed memory. Shuffling stored values after the stream degraded at
least one retained verifier in 5/32 learned banks (mean minimum-score drop
0.125), proving causal dependence on the stored contents. Every committed
bank survived exact save/reload of keys, values, and volatility. The complete
report is `transactional_stream_audit.json`, and the executable is
`experiments/unified_cognitive_controller/audit_transactional_stream.py`.

This is a persistent-memory and safety breakthrough, not yet a claim that the
controller has learned an unrelated cognitive primitive. The next frontier is
to replace the synthetic candidate stream with a genuinely different primitive
and measure verified forward transfer per retained-memory bit.

## Rejected relational-reader fork

The next cognitive-reader hypothesis was tested before any longer run. A
zero-initialized task-agnostic reader received the current intention, the
relational context, and their elementwise product. On the matched 16-update
supervised probe it moved eligible text accuracy from **47.66% to 53.91%**, but
the time-shuffle control also reached **53.32%**. A reward-only 4-back variant
ended at **47.66%** versus **48.05%** before training, with time-shuffle at
**48.24%**. These are shortcut/no-gain results, not capability claims; the
reader code remains available behind explicit diagnostic flags, but no longer
gets a larger budget until a causal representation gate improves.
