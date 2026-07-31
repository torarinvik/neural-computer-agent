# Five-item relation readout breakthrough

## Result

Using the frozen `unified_next_error_balanced_primary_seed43601.pt`
controller, a disposable probe was given five-item episodes and one query
asking whether the fourth item matches the fifth item.  The probe saw only the
post-query hidden state or workspace; verifier labels were used only to train
the discarded probe.

| representation | normal held-out | shuffled-label held-out |
| --- | ---: | ---: |
| hidden, linear | 66.02% | 52.05% |
| workspace, linear | 50.49% | 49.32% |
| combined, linear | 66.21% | 50.68% |
| hidden, MLP | **93.95%** | 51.27% |
| workspace, MLP | 81.93% | 49.22% |
| combined, MLP | 92.77% | 51.76% |

The generator required 1,024 examples per split for span five.  The result is
representation evidence only: no controller weights changed and no
behavioral five-item skill is claimed yet.  It justifies the smallest next
behavioral arm: one pure `next` relation, one extra query thought step, and
the already validated two-to-one target/rehearsal replay schedule.

## Interpretation

Span five is not blocked by sensory or memory representation.  As in span four,
the useful signal is strongly nonlinear and the workspace alone is weaker than
the recurrent hidden state.  The next experiment must therefore test credit
assignment, not invent a new encoder or memory architecture.
