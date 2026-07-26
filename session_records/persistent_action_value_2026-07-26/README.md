# Persistent action-value skill — pre-registration

The action-value challenger has passed seven consecutive independently
confirmed streams. The final milestone is durable integration.

Seeds 8011 and 8012 run the fixed 3,600-bit safe-learning configuration with
separate empty verified skill stores. Each must:

1. leave the mastered incumbent unchanged;
2. produce and independently confirm a gap promotion;
3. atomically commit both the verified parent and promoted action-value child;
4. reload both into fresh model objects with bit-exact outputs;
5. reproduce the child's pre-save audited utility exactly;
6. detect child corruption while preserving parent retrieval;
7. retain binary-mapping and four-rule controller capabilities;
8. preserve verifier-bit and unlabeled-context accounting.

Both runs must pass. The child payload is modality/task-name agnostic: an
opaque latent policy head, generic context key, causal verifier evidence, and
lineage provenance.

## Result

Both independent persistent runs passed every gate.

| Seed | Proposal | Confirmed | Lower 95% | Utility gain |
|---|---:|---:|---:|---:|
| 8011 | 960 bits | 3,360 bits | `+0.1316` | `+7.49` points |
| 8012 | 1,200 bits | 3,600 bits | `+0.0494` | `+6.66` points |

For both runs:

- mastered incumbent outputs and utility remained unchanged;
- the confirmed child was an action-value latent head, not a task label;
- parent and child reloaded bit-exactly from a fresh store instance;
- reloaded child audited utility matched pre-save utility exactly;
- child corruption was detected and parent retrieval still succeeded;
- binary-mapping and four-rule controller retention gates passed;
- accounting recorded 3,600 attempted-outcome bits and 14,400 unlabeled
  candidate contexts.

Together with the preceding seven confirmed non-persistent streams, this gives
nine consecutive safe gap promotions, two of them complete durable
promote→confirm→commit→reload replications.

This is a real milestone: the system can keep a verified incumbent immutable,
learn a stronger latent action concept from experience, prove improvement on
disjoint outcomes, and append it to long-term disk memory without corrupting
its parent.

The next frontier is sequential retrieval and reuse: begin a new related
primitive from the stored child, compare samples-to-confirmation against the
stored parent and reset, and require all ancestor skills to remain retrievable.
