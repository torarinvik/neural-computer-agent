# Retention-safe external memory representation migration

This audit replaces opaque memory key and value spaces while keeping the
controller frozen. Every occupied row is mapped one-to-one, protected
retention evidence is transferred as state without replaying outcomes, and
paired held-out queries must return the same values and hit decisions. A
candidate with changed stored values is rejected.

This promotes external-memory migration safety, not arbitrary learned value
alignment or unrestricted memory growth.
