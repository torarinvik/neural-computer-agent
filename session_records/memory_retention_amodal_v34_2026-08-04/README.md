# Memory-read-state utility candidate rejection

This discarded candidate exposed generic memory-read top similarity and hit
state to the write-utility head, with a v23-to-v24 checkpoint migration and
zero/RNG-safe initialization. It preserved the parent audit, but the
parent-stable seed-19 mini-rung collapsed to chance retention:

| condition | recall |
|---|---:|
| intact | 0.5215 |
| clear memory | 0.4980 |
| corrupt values | 0.4932 |
| reverse order | 0.4863 |
| target first | 0.4902 |
| target last | 0.5068 |

The candidate is rejected and all v24 production/schema changes are removed.
The current canonical runtime remains v23. This result argues against adding
more memory-read metadata to the utility head before solving the outcome-only
credit-assignment problem.
