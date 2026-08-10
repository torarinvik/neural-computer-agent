# Delayed identity resolution for overlapping goal frontends

This promoted four-seed pressure test deliberately gives two active frontend
alignments the same identity signature. The bank refuses the overlap, retains
two bounded deferred signatures, refuses a third at quarantine capacity, and
blocks eviction of referenced slots. A later disambiguating anchor is added
through an explicit verifier-approved update; verifier rejection leaves the
deferred records byte-stable, while acceptance resolves both records and
permits safe eviction.

All four seeds passed. Resolved frontend mastery was `0.975`–`1.0`. The
controller, factual model, and verifier memory stayed frozen, persistence was
exact, and replay was zero.

This is bounded verifier-gated delayed identity evidence. It does not establish
semantic open-world identity discovery, unrestricted memory growth, or general
continual learning.
