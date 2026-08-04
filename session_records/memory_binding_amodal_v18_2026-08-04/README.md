# v18 outcome-only two-slot binding

This is the narrow follow-up to the rejected v17 two-slot audit. The v18
controller gives memory reads and writes one shared event-window address that
does not include recurrent state or feedback. The memory backend uses two rows
and four opaque per-trajectory scopes, with fixed writes at threshold `0.0` so
retention is not part of the claim.

Seeds 17, 18, and 19 all pass the preregistered gate at 128 optimizer updates:
intact recall is `1.0` for every seed; clear-memory is
`0.5391/0.4375/0.4297`; corruption is `0.5625/0.4688/0.5313`; swapped-slot is
`0.4766/0.5391/0.4844`; and swapped-scope is `0.5000/0.4844/0.3906`. The
independent reward-shuffled control reaches `0.5234` and does not promote.

This promotes fixed-write two-slot content binding and batch isolation through
the canonical controller. It does not promote learned skip policy,
utility-based eviction, persistent episodic utility, or cross-adapter
retrieval. The prior failed rung is preserved under
`session_records/memory_binding_amodal_2026-08-04/`.
