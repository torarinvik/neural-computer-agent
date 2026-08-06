# Rejected: two-skill externalization with decoy training only (2026-08-06)

Decoy-ignorance training closed the v1 presence-cue shortcut: random
same-norm artifacts no longer unlock play (snake 0.0020/0.0020) and the
bank-withheld collapse held (~0). But the cross-artifact gate failed on
both seeds: pong plays at 0.8750/0.9063 while holding the SNAKE artifact.
The core learned to verify artifact authenticity, not artifact identity -
any genuine trained artifact acts as a master key, with game identity
leaking from the per-game peripherals and event streams.

Repair (v3): cross-artifact ignorance training - holding the other
game's artifact (detached) is shaped toward uniform play exactly like
noise, forcing the core to check program-data consistency.
