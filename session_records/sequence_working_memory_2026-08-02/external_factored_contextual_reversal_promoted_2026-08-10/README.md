# Contextual reliability reversal without factual replay

This three-seed audit tests nonstationary verifier evidence. Two factual slots
receive the same near-tolerance drift, but their verifier labels alternate
between negative and positive across two reversals.

With local count decay, the reliability gate flips in one subsequent evidence
window: the formerly rejected slot routes again, the formerly accepted slot
is vetoed, and retained quarantined rows resolve to the correct stable slot.
The factual bank never changes, persistence is exact, and the controller,
base, and context encoder remain frozen with zero replay.

This promotes a bounded recency mechanism for external reliability state. It
does not establish unrestricted memory growth, general task transfer, or
general continual learning.
