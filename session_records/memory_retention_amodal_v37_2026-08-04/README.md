# Write-critic credit-assignment rejection

This mini-rung added an opt-in training-only critic. The critic consumes
detached learned state, memory key/value, and write-probability tensors and
provides an advantage baseline only for write Bernoulli log-probabilities. It
never enters the deployed runtime or receives verifier-private labels.

The parent stabilized, but the policy learned the same last-write shortcut:
target-first recall was `0.522` and target-last was `0.997`. The critic is
retained only as reusable infrastructure; it does not qualify a capability or
solve conditional-write credit.
