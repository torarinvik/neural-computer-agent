# v65 no-feedback residual ablation

v65 removes the canonical opaque-feedback residual while retaining the stable
payload-only address and token-diverse retention schedule. Parent recall and
unseen-token recall remain high, but stable retention, order symmetry, causal
gap, and persistent reload all fail. The feedback-to-memory-value residual is
therefore required by the current narrow learned interface and is canonical in
runtime v25.
