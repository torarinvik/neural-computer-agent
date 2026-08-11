"""Audit the external operator bind-once execution contract.

This is an infrastructure audit, not a capability experiment.  It verifies
that a fixed external route is materialized once and reused by a recurrent
chain, that gradients still reach the route query, and that external-bank
growth invalidates an active binding instead of silently changing semantics.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from neural_computer import (
    ExternalCapabilityRegisterMachine,
    ExternalRegisterInstruction,
    ExternalSequenceOperatorMemory,
)


def _machine() -> ExternalCapabilityRegisterMachine:
    return ExternalCapabilityRegisterMachine(
        event_width=4,
        action_width=2,
        intention_width=6,
        register_width=8,
        instruction_width=5,
        operator_mode="factorized_protected_bounded_meta",
        instructions=tuple(ExternalRegisterInstruction(5) for _ in range(2)),
    )


def _memory() -> ExternalSequenceOperatorMemory:
    memory = ExternalSequenceOperatorMemory(8, 5, operator_rank=2)
    for _ in range(3):
        memory.add_slot()
    return memory


def run(seed: int, steps: int, batch_size: int) -> dict[str, object]:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    torch.manual_seed(seed)
    started = time.perf_counter()
    machine = _machine()
    memory = _memory()
    query = torch.randn(batch_size, 5)
    register = torch.randn(batch_size, 8)
    codes = torch.randn(batch_size, steps, 5)
    calls = 0
    original_route_weights = memory.route_weights

    def counted_route_weights(route_query: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_route_weights(route_query)

    memory.route_weights = counted_route_weights  # type: ignore[method-assign]
    with torch.no_grad():
        raw_final = machine.execute_code_chain(
            register,
            codes,
            sequence_operator_memory=memory,
            sequence_operator_route_query=query,
        )
        raw_calls = calls
        bound = memory.bind(query)
        bound_final = machine.execute_code_chain(
            register,
            codes,
            sequence_operator_memory=bound,
        )
        bound_calls = calls - raw_calls

    gradient_memory = _memory()
    gradient_query = torch.randn(batch_size, 5, requires_grad=True)
    gradient_bound = gradient_memory.bind(gradient_query)
    gradient_final = machine.execute_code_chain(
        register,
        codes,
        sequence_operator_memory=gradient_bound,
    )
    gradient_final.square().mean().backward()
    gradient_live = gradient_query.grad is not None and bool(
        torch.isfinite(gradient_query.grad).all()
    )

    growth_requires_rebind = False
    gradient_memory.add_slot()
    try:
        gradient_bound.residual(register, codes[:, 0])
    except RuntimeError as error:
        growth_requires_rebind = "rebind" in str(error)

    return {
        "schema": "neural-computer.external-sequence-operator-bind-once-audit.v1",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "raw_route_calls": raw_calls,
        "bound_route_calls_after_bind": bound_calls,
        "expected_raw_route_calls": steps,
        "expected_bound_route_calls_after_bind": 1,
        "route_call_reduction": raw_calls / max(bound_calls, 1),
        "raw_and_bound_outputs_equal": bool(torch.allclose(raw_final, bound_final)),
        "max_output_delta": float((raw_final - bound_final).abs().max()),
        "route_gradient_live": gradient_live,
        "growth_requires_rebind": growth_requires_rebind,
        "unique_verifier_bits": 0,
        "logical_lifetimes": 0,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
        "capability_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=914)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.seed, args.steps, args.batch_size), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
