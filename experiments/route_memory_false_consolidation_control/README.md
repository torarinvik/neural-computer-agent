# False-consolidation control with high-similarity distractors

Each fresh route-memory state contains one true redundant pair and one
unrelated pair with higher raw key cosine. The true pair has consistent generic
evidence masks; the distractor has incompatible masks. All rows are protected,
so the planner must choose consolidation or grow. A copy-on-write retention
verifier rejects every non-target pair and records false-consolidation
attempts separately from false commits.

The planner trains online from one scalar utility per proposal across two mask
patterns, then transfers to an unseen third pattern. Promotion requires true
consolidation transfer, zero false consolidation commits, frozen controller,
zero replay, and exact utility/update accounting.
