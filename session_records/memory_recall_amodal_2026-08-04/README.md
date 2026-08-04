# Outcome-only memory recall

This record qualifies the smallest learned capability that depends on the
canonical memory boundary. A hidden binary probe target produces an opaque
probe action and scalar reward. The controller receives that feedback, writes
it through a one-row memory, resets recurrent state, and must reproduce the
scalar outcome with no sensory evidence.

Seeds 17, 18, and 19 all pass the preregistered narrow gate at 256 optimizer
updates. Intact recall and persistent replacement recall are `1.0` for every
seed. Clearing memory gives `0.4727`, `0.4922`, and `0.4941`; zeroing the stored
value gives `0.4922`, `0.5059`, and `0.5371`. The corrected independent
reward-randomization control remains at `0.5078` and does not promote.

This is a promoted scalar-outcome recall primitive, not general episodic memory.
The rung exposes a differentiable write-strength path at threshold `0.5`; the
ordinary seeds commit at rates `1.0`, `0.6563`, and `1.0`. That shows the gate
can affect writes, but does not qualify learned skipping or utility-based
retention. Multi-row retention, content-key binding under interference,
batch-isolated memory, and cross-adapter retrieval remain open.
