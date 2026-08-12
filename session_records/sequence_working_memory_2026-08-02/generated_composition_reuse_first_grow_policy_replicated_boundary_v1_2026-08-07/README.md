# Fresh-verified reuse-first, grow-on-failure admission (2026-08-07)

The memory admission policy now tries an existing physical compute module with
a new logical binding. It accepts reuse only when every fresh retention probe
clears the mastery floor; otherwise it discards the trial binding and appends a
new physical compute module. No semantic procedure label is used.

Results:

- both related registry seeds reuse one physical module and retain both
  bindings at `1.0000/0.8906` and `1.0000/1.0000`;
- opaque seed `69316` rejects the reuse trial at `0.6172` and grows a second
  physical module, then retains at `1.0000/0.9219`;
- opaque seed `69317` accepts reuse because fresh probes reach `0.7813`, and
  retains at `1.0000/0.7813`;
- all runs pass exact reload, old-binding isolation, shared-base protection,
  checksum recovery, frozen-core, and zero-replay gates.

This is a stronger memory policy than fixed reuse: it compresses compatible
capabilities while preserving a verified path for genuinely new computation.
The decision remains bounded and verifier-driven; the next challenge is
content-addressed candidate selection across many physical modules rather
than trying only the first candidate.
