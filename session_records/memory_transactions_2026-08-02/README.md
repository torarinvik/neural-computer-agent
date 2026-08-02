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

## Boundary and next rung

This proves the safety mechanism, not yet a full learned controller loop. The
next integration must feed the promoted receipt-trained plasticity head's
replacement proposal into this transaction, score old and candidate behaviors
with the real controller/verifier, and measure commit rate, rollback rate,
learning gain per verifier bit, and retention under memory corruption.
