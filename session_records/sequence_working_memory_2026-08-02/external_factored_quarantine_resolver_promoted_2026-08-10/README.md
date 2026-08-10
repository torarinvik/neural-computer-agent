# Factual quarantine resolution

This archive records the five-seed resolver continuation of the factored
quarantine boundary. Once four factual versions were committed, four
quarantined one-row bundles were re-tested independently against the current
models and resolved to stable slots `[0, 0, 1, 1]`. A corrupted quarantined
bundle did not match any version, remained quarantined, and was released only
through the explicit caller-owned drain operation.

The resolver never trains, merges, or promotes a candidate. It uses the same
independent factual partial-read gate as normal routing and removes only
bundles that pass it; unresolved evidence remains isolated. All five seeds
also passed the underlying partial-stream acquisition and frozen-component
gates.

This promotes factual recovery of known versions from quarantine. It does not
establish learned open-world resolution or automatic new-version promotion.
