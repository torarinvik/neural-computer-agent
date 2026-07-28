# Gate-selected replay does not pay

Once interference is at zero, replay is what grows: linear per rung, quadratic
over a ladder, and already 4:1 against new-task experience by rung 5. The slot's
own gate measures which old skills it can still disturb, so skills it is exactly
shut on looked like replay that buys a guarantee already held.

Measured on a sixth rung, five inherited skills, four seeds per setting. A lower
threshold drops replay more aggressively.

| threshold | replay spent | new skill | all gates | worst retention delta |
|---|---:|---:|---|---:|
| 0.50 | **74.8%** | 0.9948 | **3/4** | **−0.01382** |
| 0.70 | 95.6% | 0.9899 | 4/4 | −0.00122 |
| 0.85 | 96.4% | 0.9866 | 4/4 | −0.00073 |
| 0.98 | 99.4% | 0.9896 | 4/4 | −0.00073 |
| every skill, always | 100.0% | 0.9884 | 4/4 | −0.00059 |

**The saving is negligible where it is safe and unsafe where it saves.** The one
setting that cuts a quarter of replay fails a retention gate and degrades the
worst-hit skill more than twentyfold against full replay.

The reason is timing, and it was visible in the first pilot: a slot does not
reach a high shut fraction until late in training, so early updates keep every
skill selected. Lowering the threshold enough to drop skills early drops them
while the gate is still moving.

The feedback risk noted when this was implemented is real, not hypothetical.
Replay is part of what holds a slot shut, so dropping a skill's replay lets that
slot drift open on it — which is exactly the −0.014 at threshold 0.50. Selection
on a quantity that the selection itself changes needs a stability argument this
design does not have.

Replay cost remains the open scaling problem. What this rules out is using the
gate as the selector.
