"""Replay the deliberation policy through the production timestamp buffer.

This module is an integration audit, not a second controller. It converts the
experiment's raw streams into timestamped opaque events, routes arrivals
through ``AmodalEventWindowBuffer``, and then invokes the same controller state
and execution policy used by the runtime.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from neural_computer import (
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
)


def _quiet_feedback(width: int, device: torch.device) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, width, device=device),
        reward=torch.zeros(1, device=device),
        propensity=torch.ones(1, device=device),
        has_feedback=torch.zeros(1, device=device),
    )


def timestamped_events(
    runtime: AmodalControllerRuntime,
    streams: Mapping[str, torch.Tensor],
    *,
    timestamp: float = 0.0,
) -> dict[str, AmodalEvent]:
    """Encode raw streams and attach transport timestamps without labels."""
    collection = runtime.encode_streams(streams)
    events: dict[str, AmodalEvent] = {}
    for index, name in enumerate(streams):
        events[name] = AmodalEvent(
            payload=collection.payload[:, index],
            source_key=(
                None
                if collection.source_key is None
                else collection.source_key[:, index]
            ),
            confidence=collection.confidence[:, index],
            timestamp=torch.full((1,), timestamp, device=collection.payload.device),
        )
    return events


def buffered_rollout(
    runtime: AmodalControllerRuntime,
    verifier,
    *,
    mode_override: str | None = None,
    out_of_order: bool = False,
    timestamp_jitter: float = 0.0,
    force_missing: bool = False,
    include_timeout: bool = False,
) -> tuple[str, float] | tuple[str, str | None, float]:
    """Run one bounded execution cycle through the real async event buffer.

    ``force_missing`` is a diagnostic timeout control: the verifier still
    scores the private target, but the delayed partner is withheld and the
    buffer releases its partial timestamp window.
    """
    device = next(runtime.parameters()).device
    streams = verifier.reset(1)
    state = runtime.initial_state(1, device=device)
    feedback = _quiet_feedback(runtime.controller.feedback_width, device)
    buffer = runtime.window_buffer(("a", "b"), tolerance=0.25, max_wait=1.0)
    arrivals = timestamped_events(runtime, streams)

    if "b" in arrivals:
        ordered = (
            {"b": arrivals["b"], "a": arrivals["a"]}
            if out_of_order
            else {"a": arrivals["a"], "b": arrivals["b"]}
        )
        windows = buffer.push(ordered)
        if len(windows) != 1 or not windows[0].complete:
            raise RuntimeError("complete timestamped arrivals did not form one window")
        initial_events = windows[0].collection
    else:
        buffer.push({"a": arrivals["a"]})
        initial_events = runtime.encode_streams({"a": streams["a"]})

    initial, state = runtime.step_events(initial_events, state, feedback)
    decision = mode_override or ("wait", "think", "commit")[
        int(initial.execution_logits[0].argmax())
    ]
    final = initial
    timeout_decision: str | None = None

    if decision == "wait":
        if "b" in streams:
            # The initial buffered window is already complete. Waiting should
            # not turn available evidence into a synthetic timeout.
            final = initial
        else:
            partner = {} if force_missing else verifier.release_delayed()
            if partner:
                partner_event = timestamped_events(
                    runtime, partner, timestamp=timestamp_jitter
                )["b"]
                windows = buffer.push({"b": partner_event})
                if len(windows) != 1 or not windows[0].complete:
                    raise RuntimeError("delayed partner did not complete its timestamp window")
                final, state = runtime.step_events(
                    windows[0].collection,
                    state,
                    _quiet_feedback(runtime.controller.feedback_width, device),
                    elapsed=1.0,
                )
            else:
                buffer.release_pending(0.0)
                quiet = AmodalEventCollection.empty(1, runtime.event_width, device=device)
                final, state = runtime.step_events(
                    quiet,
                    state,
                    _quiet_feedback(runtime.controller.feedback_width, device),
                    elapsed=1.0,
                )
                timeout_decision = ("wait", "think", "commit")[
                    int(final.execution_logits[0].argmax())
                ]
                if timeout_decision == "think":
                    final, state = runtime.step_events(
                        quiet,
                        state,
                        _quiet_feedback(runtime.controller.feedback_width, device),
                        elapsed=1.0,
                    )
    elif decision == "think":
        quiet = AmodalEventCollection.empty(1, runtime.event_width, device=device)
        final, state = runtime.step_events(
            quiet,
            state,
            _quiet_feedback(runtime.controller.feedback_width, device),
            elapsed=1.0,
        )
        partner = {} if force_missing else verifier.release_delayed(after_think=True)
        if partner:
            partner_event = timestamped_events(
                runtime, partner, timestamp=timestamp_jitter
            )["b"]
            windows = buffer.push({"b": partner_event})
            if len(windows) != 1 or not windows[0].complete:
                raise RuntimeError("thought partner did not complete its timestamp window")
            final, state = runtime.step_events(
                windows[0].collection,
                state,
                _quiet_feedback(runtime.controller.feedback_width, device),
                elapsed=1.0,
            )

    action = final.decoded["protocol"].argmax(dim=-1)
    reward = verifier.step(action).item()
    if include_timeout:
        return decision, timeout_decision, float(reward)
    return decision, float(reward)
