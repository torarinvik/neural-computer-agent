# Learned opaque-row eviction — seed 17

Status: promoted narrow learned-eviction rung.

The controller acquired its parent on fresh opaque event tokens and was then
frozen. A separate memory-side scorer learned to choose which row to overwrite
from two paired counterfactual factors: target-first and target-middle. Policy
updates consumed only scalar recall differences between forced row-0 and row-1
arms.

- held-out balanced recall: `0.916`
- target-first recall: `0.903`
- target-last recall: `0.981`
- strength-eviction target-first baseline: `0.488`
- random target-first control: `0.737`
- clear-memory/corruption controls: `0.489`/`0.489`
- persistent reload/recovery: `0.916`/`0.969`
- checksum corruption rejected: `true`
- replayed examples: `0`

The controller remained frozen during eviction learning. This promotes a
narrow learned utility-based eviction boundary, not general episodic memory or
arbitrary new computation.
