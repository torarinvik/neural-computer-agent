# Population persistent-memory and transfer boundary qualification

v46 reruns the consolidated parent-preserving retention protocol on seeds 17,
18, and 19. Each run writes through `PersistentContentAddressedMemory`,
reopens the atomic snapshot, deliberately mutates serialized state without
updating its checksum, verifies that a new backend rejects the snapshot, then
restores the known-good state atomically and verifies recall again.

Persistent reload averaged `0.991` recall across the population, recovery was
`1.000` for every seed, and checksum corruption was rejected for every seed.
The ordinary retention gate promoted seeds 17 and 19; seed 18 failed the
stable-prefix rule after the requested long budget. The three-seed transfer
replication is inconclusive: seed 19 reproduces the `1.538x` fresh-over-
transferred stable-bit ratio, while seeds 17 and 18 do not reach a stable
transfer threshold under the matched short budget.

This qualifies the persistent storage boundary for the narrow verifier across
three seeds. It does not qualify a checkpoint, general persistent episodic
capability, or population-level transfer efficiency. Keep the transfer
failure visible and investigate curriculum/seed variance before increasing
the transfer budget. Exact per-seed reports and accounting are in this
directory.
