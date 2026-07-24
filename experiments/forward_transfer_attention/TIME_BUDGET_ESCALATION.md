# Time-budgeted experiment policy

Every candidate mechanism starts with a 15–30 second run. The run is allowed
to proceed only when it is mechanically healthy: finite metrics, live
gradients, bounded residuals, and a decreasing loss. This is a health check,
not evidence of capability; a phase-transition valley is expected.

Escalation requires a capability signal on unseen logical lifetimes and an
honest control (for example, shuffled labels at chance). Repeat the tiny run
with a second seed before moving to 3 minutes, then require replication before
moving to 10 minutes. A red flag or two independent no-signal runs means change
the experiment rather than increasing its budget.

Use `experiment_gate.py` to classify JSON reports. The gate never treats live
gradients alone as a reason to spend more compute.
