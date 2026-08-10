# Factual quarantine resolution into an isolated candidate

This archive extends the factored quarantine boundary with a later-evidence
continuation. Five seeds first retain one opaque novel bundle in quarantine,
then stage a copy-on-write factual candidate from a later bundle. The resolver
tests the retained bundle against that candidate, consumes it once when it
agrees, and leaves the committed model unchanged. Candidate state persists
through the router payload and remains subject to the independent held-out
promotion gate.

All five seeds passed the existing partial-stream, retention, corruption, and
frozen-component controls. The candidate resolver performs no controller or
base updates and makes no claim that a novel candidate is ready for promotion
without held-out evidence.

This promotes safe later-evidence absorption into an isolated factual
candidate. It does not establish a learned open-world identity resolver,
automatic version formation, or general continual learning.

The same run attached a persistent sparse factual-overlap index to the
factored router. It retained `56` unique stream facts per seed and passed
round-trip identity checks. The index is a contradiction-aware proposal
accelerator; factual verification and independent promotion probes remain
authoritative.
