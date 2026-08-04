# v56 recall-only parent-credit diagnostic

v56 removes the probe-action policy-gradient term from the coupled parent
intervention because the probe bit is random and unobservable before the
action. It trains only the identifiable post-outcome recall path, using fixed
writes.

Applying the intervention during parent rehearsal makes the parent initially
stable but reduces mastered-parent retention to `0.770`; the final retention
run has no stable prefix. The arm is rejected. The probe-credit removal is not
a capability gain and is not part of the default protocol.
