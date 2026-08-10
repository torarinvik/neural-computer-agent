# Policy-free amodal runtime

This archive records the first end-to-end execution audit of the new
canonical seam inspired by the exported games session:

```text
opaque events -> one frozen amodal controller -> opaque state
             -> factual model search toward an opaque goal
             -> intention bus -> independent decoder
```

Seeds `85001`, `85002`, and `85003` all pass the integration gates. The
one-pass affine factual model reaches every one of four novel goals through
planner-derived intentions, while goal-shuffled controls score `0.0`. The
controller remains byte-stable, the factual model remains unchanged during
search, and model persistence is exact.

Mean search latency is `2.59–4.25 ms` across the three local runs, with `84`
expanded nodes per target. The direct controller intention is recorded only
as a diagnostic control; it is not decoded. Replay and optimizer updates are
zero after the single factual sufficient-statistics update.

This promotes canonical policy-free runtime wiring and behavior derivation
from factual search. It does not establish learned state grounding,
unrestricted planning, arbitrary new computation, or general continual
learning. The next promotion must connect the seam to a genuinely learned
state/goal representation and a nontrivial sequential target stream.
