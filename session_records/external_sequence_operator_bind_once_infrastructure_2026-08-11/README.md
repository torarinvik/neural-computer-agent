# Bind-once external operator routing — infrastructure audit

Date: 2026-08-11

## Question

Does the external operator-memory path make contextual lookup once per
rollout, then reuse that binding through a fixed recurrent instruction chain?
This is the direct implementation lesson transferred from the exported
working-memory session. It is an execution-contract audit, not a learned
capability experiment.

## Mechanism

`ExternalSequenceOperatorMemory.bind(query)` materializes the learned route
distribution and returns an ephemeral `BoundExternalSequenceOperatorMemory`.
The shared register interpreter accepts that handle without a route query or
slot ID. The underlying bank remains external and independently mutable; bank
growth invalidates the handle and requires an explicit rebind.

## Result

The seed-914, eight-step, batch-16 audit observed:

- raw routed execution: 8 route-encoder calls;
- bound execution after binding: 1 route-encoder call;
- raw and bound outputs: exactly equal (`max_output_delta = 0.0`);
- route-query gradient: live and finite;
- external-bank growth: correctly rejected until rebinding;
- unique verifier bits, logical lifetimes, optimizer updates, and replayed
  examples: all zero, because this was not a capability run.

The route encoder was therefore reduced by 8x for this fixed-depth case while
preserving semantics and gradients. This does not demonstrate faster learning,
positive transfer, arbitrary new computation, unrestricted memory growth, or
general continual learning. It only removes repeated contextual lookup from
the recurrent execution contract and makes growth behavior explicit.

Raw slot and route-query APIs remain available for diagnostics and route
probes. The binding is deliberately ephemeral and is not controller state.

## Reproduction

```bash
PYTHONPATH=src:. .venv/bin/python \
  experiments/external_register_composition_amodal/audit_bind_once_contract.py \
  --seed 914 --steps 8 --batch-size 16
```

