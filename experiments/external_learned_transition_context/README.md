# External learned transition context

This pressure test closes the first gap in the contextual model-bank result:
contexts are learned from opaque transition evidence instead of supplied as a
regime label. A frozen controller is paired with an independently trainable
`ExternalTransitionContextEncoder` and an append-only
`ExternalTransitionModelBank`.

The encoder is trained only with paired noisy views of transition bundles. It
must make views of the same dynamics stable while keeping different dynamics
separable. A held-out dynamics regime then receives a new bank slot using the
encoder-generated key; the nearest learned factual model is used only as an
initialization prior. Behavior is still derived by model-based search, not
stored as a policy.

The report includes context stability/separation, frozen-controller and
no-label controls, base-regime retention, target-vs-fresh acquisition cost,
wrong-context and corrupted-target controls, and persistence. This remains a
bounded context-acquisition result: the encoder sees a finite transition
bundle, and it does not yet demonstrate indefinite online identity formation,
slot compression, or unrestricted task transfer.

```text
.venv/bin/python experiments/external_learned_transition_context/train.py \
  --seed 69911 \
  --report-out /tmp/learned-transition-context.json
```
