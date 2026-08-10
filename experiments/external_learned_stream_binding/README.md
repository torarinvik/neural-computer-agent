# Learned external stream binding

This pressure test trains one generic transition-context encoder from paired
same-stream views, freezes it, and then asks an external memory to bind
interleaved transition arrivals without caller-supplied stream keys.

The deployed path is:

```text
opaque transition arrival
    -> frozen context encoder
    -> anonymous track / delay / reliability memory
    -> opaque key
    -> one shared factual multi-stream router
    -> transition model memory
```

The controller is frozen and receives no stream labels. Trainer-only stream
indices are used to construct positive pairs and score diagnostic assignment
accuracy; they never enter the binding memory. A fresh untrained encoder,
interleaving-order control, missing-arrival control, persistence check, and
controller digest check are reported separately.

The open-set extension runs the same boundary at live capacity. An unseen
stream is accumulated in bounded provisional memory rather than being forced
into an existing identity or mutating the shared factual bank. Provisional
state can be promoted only after a caller-owned retention probe; retirement is
transactional for the same reason. Irregular timestamps, rejected admission,
rejected retirement, exact persistence, and live-track byte stability are
reported separately.

This promotes a bounded learned identity/binding lifecycle with safe
quarantine and replacement. It does not promote learned semantic identity,
general continual learning, unrestricted memory growth, or natural-language
grounding.
