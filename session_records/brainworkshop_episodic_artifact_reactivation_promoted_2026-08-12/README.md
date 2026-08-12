# Promoted episodic executable-artifact reactivation

This record promotes a narrow memory-side capability: a frozen shared
interpreter can reactivate opaque executable artifacts from a growable
episodic archive into a bounded hot cache without replaying their acquisition
stream or updating the controller/interpreter.

Seeds 24101 and 24102 both passed:

- reactivation of two cold executable artifacts through a two-slot hot cache;
- revisiting an older artifact after multiple swaps without replay;
- retention of a protected artifact through every replacement;
- rejection of protected, failed, mutating, missing, and corrupt candidates
  without changing live state;
- exact index and executable-memory reload;
- byte-identical frozen shared interpreter;
- zero online optimizer updates and zero replayed examples.

Each seed charged 512 held-out verifier bits: 384 for reactivation probes and
128 for final hot-bank execution. The archive contains the full JSON reports
and `sample_efficiency_ledger.json`.

## Claim boundary

This is bounded replay-free reactivation of stored executable capability files.
It does not establish unrestricted memory growth, automatic synthesis of new
computation while frozen, or general continual learning.
