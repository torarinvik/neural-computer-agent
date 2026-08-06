# Stable controller value path — five-slot, two-row seed 69415 at 800 steps

Status: rejected curriculum-budget rung.

The five-slot architecture was held fixed, but the parent did not stabilize at
800 requested phase-1 steps. Retention was correctly blocked rather than
interpreted as an architecture failure.

- intact: `0.495`
- target-first/last: `0.502`/`0.500`
- mastered-parent retention: `0.496`
- parent stable: `false`
- retention updates: `0`
- replayed examples: `0`

The persistent backend rejected checksum corruption, but reload and recovery
were chance-level because no parent capability had been acquired. Following
the experiment ladder, the next run changed only the phase-1 curriculum
budget.
