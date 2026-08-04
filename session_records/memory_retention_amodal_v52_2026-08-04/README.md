# Fixed-write parent scaffold rejection

v52 isolates parent action acquisition by forcing the parent memory write to
commit while training paired action interventions. Fresh parent acquisition
improves to 2/3 initializations, but the retained model loses the parent
boundary during retention: mastered-primitive retention is 0.773, best
validation parent retention is 0.766, and no stable threshold is reached.

The fixed-write scaffold is rejected as a capability protocol. It is useful
diagnostically: it shows that action acquisition and retention-phase
co-adaptation must be separated without freezing or privileging the deployed
agent.
