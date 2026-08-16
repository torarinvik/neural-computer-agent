# Live amodal identity-assignment seam (2026-08-16)

Status: **interface diagnostic; not a holdout and not a promotion**.

The collision experiment remains synthetic and stays outside the curated bank.
This record documents the first production seam for using a caller-owned
identity artifact without adding an identity branch to the controller.

## Contract exercised

```text
learned event collection
  -> opaque causal evidence [external self/assignment artifact]
  -> ExternalCausalIdentityAssignment
  -> opaque goal-state candidate selection
  -> PolicyFreeAmodalRuntime
  -> opaque intention / protocol decoder
```

The gate consumes only finite evidence scores. It selects a candidate by
top-two margin and emits an explicit abstention on a tie. Abstention returns no
action; it never fabricates a goal, zero evidence, or a protocol command. The
selected identity slot is receipt-local credit metadata and is not fed into
the controller. Existing source guards still reject verifier coordinates,
task/rule IDs, cluster maps, and scoring oracles in the production adapter.

## Tests

- high-margin evidence selects an opaque goal state and emits one decoded
  intention;
- the selected slot is retained only in `PolicyFreeLiveCredit`;
- equal evidence produces no proposal and leaves the planner output untouched;
- incompatible protocol feedback remains rejected;
- the production source remains free of navigation/verifier oracle names.

This is an interface test, not a behavior claim. No unique verifier bits,
logical lifetimes, optimizer updates, or replayed examples were spent, and no
bank or checkpoint was modified.

## Next gate

Run the collision assignment artifact through a real learned event frontend on
valid pixel rerenders, compare against a matched no-assignment learner, and
measure stable prefixes before connecting it to a temporary external transition
or self-model artifact. Do not admit this seam or the synthetic beam to the
curated bank yet.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_navigation_runtime.py
```

The machine implementation is
`src/neural_computer/navigation_runtime.py`; the versioned assignment gate is
`src/neural_computer/identity.py`.
