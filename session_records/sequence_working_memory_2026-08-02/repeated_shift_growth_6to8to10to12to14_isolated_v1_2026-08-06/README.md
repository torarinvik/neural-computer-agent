# Isolated fourth-shift growth to 46 capabilities (2026-08-06)

This audit revisits the rejected 46-capability shared-router frontier with
the modular route/credit mechanism. The schedule starts with two protected
length-six capabilities, then adds 8 length-eight, 10 length-ten, 12
length-twelve, and 14 length-fourteen capabilities. Each new capability gets
an isolated external route extension and credit head; the controller, old
route, and earlier external state remain frozen. No earlier examples are
replayed after a shift.

| metric | seed 69316 | seed 69317 |
| --- | ---: | ---: |
| total capabilities | 46 | 46 |
| shift-1 minimum route | 0.9219 | 0.8906 |
| shift-2 minimum route | 0.8906 | 0.9063 |
| shift-3 minimum route | 0.9219 | 0.8594 |
| shift-4 minimum route | 0.8594 | 0.8750 |
| old route / permutation | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| credit old / combined | 1.0000 / 0.9348 | 1.0000 / 0.9130 |
| full-bank protection / reversal / recovery | pass | pass |
| reward-shuffled route selection | pass | pass |
| replayed examples | 0 | 0 |

Both seeds pass every declared gate. Compared with the rejected shared-router
46-capability rung, this isolates the failure to interference in one mutable
candidate scorer rather than to external-bank capacity. The tradeoff is that
the modular mechanism allocates one replaceable route/credit state per new
capability and does not yet provide a shared reusable acquisition prior.

## Accounting

Per seed: 2,916,728 unique verifier bits, 477,560 unique logical lifetimes,
30,720 optimizer updates, 23,552 route updates, four distribution shifts,
zero replayed examples, and approximately 33 seconds wall time. Fresh-learner
transfer was not measured in this audit.

## Claim boundary

This promotes a bounded four-shift, 46-capability replay-free external-growth
mechanism with isolated route/credit state and hard retention protection. It
does not establish unbounded growth, arbitrary program induction, a
sample-efficiency gain over a fresh learner, or general continual learning.

Reports and the accounting ledger are in this directory; `SHA256SUMS`
verifies the archived evidence.
