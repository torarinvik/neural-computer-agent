# Replicated external skill-fragment composition — 2026-08-11

This record promotes bounded reusable compositional transfer across two seeds.
The parent amodal controller is frozen after forward-sequence acquisition. Two
opaque external fragments are acquired sequentially (`reverse`, then `rotate`)
without replay. A held-out ordered composition is learned by an external trace
combiner and decoder. The matched fresh learner has the same external
architecture and fresh verifier exposure but no inherited fragment interpreter
state.

| seed | inherited stable bits | fresh stable bits | fresh/inherited | composition | wrong order | zero codes | reward shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 69316 | 6,144 | 24,576 | 4.00x | 0.9661 | 0.7708 | 0.6510 | 0.6797 |
| 69317 | 9,216 | 12,288 | 1.33x | 0.9948 | 0.7630 | 0.6745 | 0.3307 |

Both seeds passed every promotion gate, including reverse retention, stable
composition and fresh baselines, positive transfer, order sensitivity, no
fragment bypass, reward-shuffled rejection, frozen parent, zero replay, and
opaque routing resolution. The reported result remains deliberately bounded:
it is not general continual learning, unrestricted memory growth, arbitrary
program induction, or proof of Turing-complete learned behavior.

The implementation gain is the external execution trace plus trace combiner.
The interpreter exposes ordered post-instruction states with a mask, and the
combiner learns composition-specific credit assignment without changing the
controller. Normalized external code materialization prevents a silent
near-zero instruction scale failure.
