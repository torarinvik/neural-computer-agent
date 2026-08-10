# The n-step value head: 25.4% -> 45.6% of headroom (F111)

F110's oracle substitution convicted the outcome model and showed the working
interface: real-valued cell worth. This mimics it — a scalar head regressing
the discounted n-step return, used raw by the search; the ignorance term pins
the entry-free prediction to the batch mean.

| arm | held-out | twin | entry effect | % headroom |
| --- | ---: | ---: | ---: | ---: |
| 3-class outcome | -0.0205 | -0.0782 | +0.0577 | 25.4% |
| n-step value head | +0.0069 | -0.0968 | +0.1036 | 45.6% |
| oracle values (target) | +0.1234 | | | |

Held-out positive on both seeds for the first time (+0.0010, +0.0127); the twin
penalty deepens on both — reading harder.

The 3-class quantisation was itself much of the constraint: all positive futures
in one bin threw away the gradient the search needed, and it was degenerate
twice over (F102, F106). Regression has no bins.

Ladder: 0.2% -> 22.0% (ignorance) -> 25.4% (freeze) -> 45.6% (value head).
Remaining to oracle-value target: +0.1165.
