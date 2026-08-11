# Held-out parallel composition — causal narrow promotion

This two-seed in-repository audit tests a generic parallel composition under
randomized slot domains. The richer training arm includes random parallel
programs but explicitly excludes the exact target
`PARALLEL(INC(0,m=2), INC(1,m=2))`, in either child order. The atomic-only arm
is the byte-identical interpreter trained on the same random-program budget
without parallel instructions.

At the stable-prefix threshold `0.9`, both parallel arms learn the held-out
target stably by update `300`; both atomic controls remain at zero. This is a
narrow composition-generalization result: the model can combine known local
effects into an unseen simultaneous atomic effect when composition examples
are available. The result does not claim that the richer arm improves old-basis
sample efficiency, arbitrary computation, general continual learning, or
unrestricted memory growth.

Each arm consumed `192,000` unique random-program steps with zero replay. The
arithmetic-family probes remain available in the same report and retain their
causal wrong-modulus controls. The audit schema is
`neural-computer.recipe-expressibility-audit.v4`.
