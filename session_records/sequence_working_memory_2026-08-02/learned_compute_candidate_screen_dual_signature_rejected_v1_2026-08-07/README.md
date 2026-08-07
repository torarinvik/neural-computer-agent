# Dual raw/normalized signature rejected (2026-08-07)

This control retains both the original learned key and its frozen affine-
normalized view in one opaque 64-wide signature, applying the same transform
to queries and keys. It avoids choosing one representation by hand while
remaining permutation-equivariant and controller-independent.

It is rejected at the bank-26/six-stage/1024-update/full-prior boundary:
unseen routing is `0.9167/0.7396`, and both seeds have per-target holes. The
extra representation increases rank but does not produce a stable learned
alignment. The next strategy is page-local representation selection under
fresh verifier control, not a larger concatenated signature.
