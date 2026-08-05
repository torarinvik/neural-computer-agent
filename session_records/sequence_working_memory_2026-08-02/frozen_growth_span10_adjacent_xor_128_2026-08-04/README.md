# Adjacent-XOR frozen-growth probe — rejected

The orthogonal adjacent-XOR primitive was tested at the required sub-minute
diagnostic rung against the same frozen span-eight parent used by the promoted
complement artifacts.

- Target: span 10, adjacent-XOR, 128 successor-slot updates.
- Target gain: `-0.625` percentage points.
- Growth ablation was not causal, while insertion, rehydration, retention, and
  corruption controls passed.
- The candidate was not promoted or scaled.

This is a useful negative result: the current generic successor slot can
retain the parent while failing to acquire a structurally different local
comparison primitive at this budget. The next genuine factor-chaining test
must therefore improve credit assignment or use a task with a verified causal
signal before attempting composition.
