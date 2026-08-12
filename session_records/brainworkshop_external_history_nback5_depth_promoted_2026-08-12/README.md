# Generic external-history n-back-5 depth probe (2026-08-12)

This promotion widens the replaceable flattened external event window from
five to six records. The query count is six, meaning five preceding learned
events plus the current event. The controller, event frontend, instruction
interpreter, and keypress boundary remain unchanged; only the external compute
basis receives the larger generic window.

Across seeds 17 and 18, n-back-5 reached `1.0000` on all four fresh lifetimes
after 512 attempted-outcome updates. The controller and event frontend stayed
byte-identical and replay was zero.

The variable-history attention path was retained as a diagnostic control. At
the same n-back-5 depth it plateaued around `0.74–0.78` after both 192 and 512
updates on seed 17, so it was not promoted as a learned-capability result.
This identifies the next implementation bottleneck: making the variable
history reducer as sample-efficient and lag-selective as the flattened ABI,
without introducing a task-specific branch.

The history-attention implementation also now compacts masked records before
the recurrent pass and maps states back to their opaque positions. A focused
invariance test confirms that left-padding or sparse mask placement cannot
change the computation when valid records and ages are unchanged.

This promotes deeper generic bounded temporal computation, not unrestricted
history growth, learned compression, arbitrary program induction, or general
continual learning.
