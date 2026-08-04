# Think-required execution primitive

This rung isolates the remaining execution state. The initial event is
low-confidence and the partner event is released only after an internal quiet
controller tick. A wait or immediate commit therefore cannot recover the
target; `THINK` is causally necessary. The learner still sees only opaque
events and the scalar verifier outcome.

Seeds 17, 18, and 19 all learned `THINK` at 100% evaluation frequency and
reached 1.0 reward. The result promotes a narrow think-required primitive, not
general mixed-state scheduling: the mixed distribution still needs a
state-conditioned execution critic or curriculum to arbitrate reliably among
all three actions.
