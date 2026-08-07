# Append-prior strength boundary (2026-08-07)

This audit tests whether a new append-only screen should inherit the frozen
base query path at full strength. `prior_strength` blends copied floating
point state with the extension's fresh initialization; the base remains
frozen, key-side state remains fresh under `query_path`, and extensions stay
independently replaceable.

At three singleton stages and 128 calibration updates per stage, fresh
initialization passes both seeds. Full-strength query transfer passes only one
seed (the other reaches `0.7396`); half strength passes both (`0.8542/0.9063`),
while quarter strength again fails one (`1.0000/0.6667`). At 64 updates,
half strength fails both for three stages (`0.3333/0.3333`). On the earlier
two-stage mixed `[1, 2]` boundary at 64 updates, half strength passes one seed
and fails the other (`1.0000/0.6667`), so it does not replace the full-strength
prior as a universal default.

This promotes the tunable copy-on-write prior-strength API as a safe escape
from harmful inherited basins, not a new sample-efficiency claim. The
evidence shows that prior transfer is depth- and strength-dependent; a future
memory policy should select or validate the prior from fresh verifier evidence
rather than assume one strength is optimal forever. Full raw reports and
checksums are included here.
