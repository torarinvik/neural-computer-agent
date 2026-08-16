"""A bounded live adapter for the policy-free amodal navigation slice.

This module is deliberately an adapter, not a second navigation controller.
The production controller still consumes only learned event tensors; factual
transition search and candidate intentions stay in the external
``PolicyFreeAmodalRuntime``. Protocol actions are decoded after planning and
are converted back to opaque controller feedback by a replaceable adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .identity import (
    ExternalCausalIdentityAssignment,
    ExternalIdentityAssignment,
)
from .interface import AmodalEventCollection, ControllerFeedback
from .live import (
    LiveActionProposal,
    LiveCognitiveMachine,
    LiveDecoderDecision,
    LiveIntentionDecoder,
    ResolvedLiveOutcome,
)
from .runtime import PolicyFreeAmodalRuntime, PolicyFreeRuntimeOutput

POLICY_FREE_LIVE_MACHINE_SCHEMA = "neural-computer.policy-free-live-machine.v1"


@dataclass(frozen=True)
class PolicyFreeLiveCredit:
    """Receipt-local external planning identity for delayed diagnostics."""

    selected_slot_id: int | None
    planner_schema: str
    identity_slot_id: int | None = None
    schema: str = POLICY_FREE_LIVE_MACHINE_SCHEMA

    def validate(self) -> PolicyFreeLiveCredit:
        if self.schema != POLICY_FREE_LIVE_MACHINE_SCHEMA:
            raise ValueError("unsupported policy-free live credit schema")
        if not isinstance(self.planner_schema, str) or not self.planner_schema:
            raise ValueError("policy-free live planner schema is missing")
        if self.selected_slot_id is not None and self.selected_slot_id < 0:
            raise ValueError("policy-free live selected slot is invalid")
        if self.identity_slot_id is not None and self.identity_slot_id < 0:
            raise ValueError("policy-free live identity slot is invalid")
        return self


class PolicyFreeAmodalLiveMachine(LiveCognitiveMachine):
    """Drive one policy-free planner through the live tick contract.

    ``goal_state`` and ``candidate_intentions`` are opaque external artifacts.
    They are copied at construction and never inferred from verifier state.
    ``feedback_encoder`` is the sole bridge from externally decoded protocol
    actions back into the controller's learned feedback space.
    """

    schema = POLICY_FREE_LIVE_MACHINE_SCHEMA

    def __init__(
        self,
        runtime: PolicyFreeAmodalRuntime,
        decoder: LiveIntentionDecoder,
        *,
        goal_state: torch.Tensor,
        candidate_intentions: torch.Tensor,
        output_key: str,
        batch_size: int = 1,
        horizon: int = 1,
        feedback_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        identity_assignment: ExternalCausalIdentityAssignment | None = None,
        goal_state_candidates: torch.Tensor | None = None,
        require_known: bool = False,
        model_version: int = 0,
        sample: bool = True,
    ) -> None:
        if not isinstance(runtime, PolicyFreeAmodalRuntime):
            raise TypeError("policy-free live machine needs a policy-free runtime")
        if not callable(getattr(decoder, "decide", None)):
            raise TypeError("policy-free live decoder must implement decide")
        if batch_size < 1 or horizon < 1 or not output_key:
            raise ValueError("policy-free live dimensions and output are invalid")
        if model_version < 0:
            raise ValueError("policy-free live model version cannot be negative")
        if goal_state.ndim != 2 or goal_state.shape[0] != batch_size:
            raise ValueError("policy-free live goal state must have shape [batch, width]")
        if goal_state.shape[1] != runtime.planner.model.state_width:
            raise ValueError("policy-free live goal state width is incompatible")
        if candidate_intentions.ndim not in (2, 3):
            raise ValueError("policy-free live candidates must be [count, width] or [batch, count, width]")
        if candidate_intentions.shape[-1] != runtime.runtime.intention_width:
            raise ValueError("policy-free live candidate width is incompatible")
        candidate_count = (
            candidate_intentions.shape[0]
            if candidate_intentions.ndim == 2
            else candidate_intentions.shape[1]
        )
        if candidate_count < 1:
            raise ValueError("policy-free live candidates cannot be empty")
        if candidate_intentions.ndim == 3 and candidate_intentions.shape[0] != batch_size:
            raise ValueError("policy-free live candidate batch is incompatible")
        if identity_assignment is None and goal_state_candidates is not None:
            raise ValueError("goal-state candidates require an identity assignment artifact")
        if identity_assignment is not None and goal_state_candidates is None:
            raise ValueError("identity assignment requires goal-state candidates")
        if goal_state_candidates is not None:
            if goal_state_candidates.ndim not in (2, 3):
                raise ValueError("goal-state candidates must be [slots, width] or [batch, slots, width]")
            if goal_state_candidates.shape[-1] != runtime.planner.model.state_width:
                raise ValueError("goal-state candidate width is incompatible")
            if goal_state_candidates.ndim == 3 and goal_state_candidates.shape[0] != batch_size:
                raise ValueError("goal-state candidate batch is incompatible")
            if goal_state_candidates.shape[-2] < 1:
                raise ValueError("goal-state candidates cannot be empty")
        decoder_width = getattr(decoder, "intention_width", None)
        if decoder_width != runtime.runtime.intention_width:
            raise ValueError("policy-free live decoder width is incompatible")
        self.runtime = runtime
        self.decoder = decoder
        self.output_key = output_key
        self.batch_size = int(batch_size)
        self.event_width = runtime.runtime.event_width
        self.horizon = int(horizon)
        self.require_known = bool(require_known)
        self.model_version = int(model_version)
        self.sample = bool(sample)
        self.goal_state = goal_state.detach().clone()
        self.candidate_intentions = candidate_intentions.detach().clone()
        if feedback_encoder is None:
            feedback_encoder = self._identity_feedback
        if not callable(feedback_encoder):
            raise TypeError("policy-free live feedback encoder must be callable")
        self.feedback_encoder = feedback_encoder
        self.identity_assignment = identity_assignment
        self.goal_state_candidates = (
            None
            if goal_state_candidates is None
            else goal_state_candidates.detach().clone()
        )
        self._state = runtime.runtime.initial_state(
            self.batch_size,
            device=self.goal_state.device,
            dtype=self.goal_state.dtype,
        )
        self._last_output: PolicyFreeRuntimeOutput | None = None
        self._last_identity_assignment: ExternalIdentityAssignment | None = None

    @staticmethod
    def _identity_feedback(action: torch.Tensor) -> torch.Tensor:
        return action

    @property
    def state(self):
        """The live working state lease; callers cannot mutate the planner."""

        return self._state

    @property
    def last_output(self) -> PolicyFreeRuntimeOutput | None:
        return self._last_output

    @property
    def last_identity_assignment(self) -> ExternalIdentityAssignment | None:
        return self._last_identity_assignment

    def configuration(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "boundary": "learned-events_to_external-model-search_to_opaque-intention_v1",
            "controller_input": "amodal_event_collection_only",
            "goal": "external_opaque_planner_state",
            "feedback": "replaceable_protocol_to_opaque_action_encoder",
            "candidate_intentions": "external_opaque_set",
            "identity_assignment": (
                None
                if self.identity_assignment is None
                else self.identity_assignment.configuration()
            ),
            "horizon": self.horizon,
            "require_known": self.require_known,
            "model_version": self.model_version,
            "policy_free_runtime": self.runtime.configuration(),
        }

    def reset(self) -> None:
        """Start a new live lifetime without changing external artifacts."""

        self._state = self.runtime.runtime.initial_state(
            self.batch_size,
            device=self.goal_state.device,
            dtype=self.goal_state.dtype,
        )
        self._last_output = None
        self._last_identity_assignment = None

    def _resolved_goal_state(
        self,
        identity_evidence: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ExternalIdentityAssignment | None]:
        if self.identity_assignment is None:
            if identity_evidence is not None:
                raise ValueError("identity evidence supplied without an assignment artifact")
            return self.goal_state, None
        if identity_evidence is None:
            raise ValueError("identity assignment requires learned evidence each tick")
        assert self.goal_state_candidates is not None
        candidates = self.goal_state_candidates
        slot_count = candidates.shape[-2]
        assignment = self.identity_assignment.resolve(identity_evidence)
        assignment.validate(batch_size=self.batch_size, slot_count=slot_count)
        if bool(torch.any(assignment.abstained)):
            return self.goal_state, assignment
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0).expand(self.batch_size, -1, -1)
        selected = assignment.selected_slot.to(device=candidates.device)
        goal_state = candidates.gather(
            1,
            selected.view(self.batch_size, 1, 1).expand(
                -1, 1, candidates.shape[-1]
            ),
        ).squeeze(1)
        return goal_state, assignment

    def _feedback(
        self, outcomes: Sequence[ResolvedLiveOutcome]
    ) -> ControllerFeedback:
        device = self.goal_state.device
        dtype = self.goal_state.dtype
        action = torch.zeros(
            self.batch_size,
            self.runtime.runtime.controller.feedback_width,
            device=device,
            dtype=dtype,
        )
        reward = torch.zeros(self.batch_size, device=device, dtype=dtype)
        propensity = torch.ones(self.batch_size, device=device, dtype=dtype)
        present = torch.zeros(self.batch_size, device=device, dtype=dtype)
        if outcomes:
            resolved = max(outcomes, key=lambda item: item.receipt.receipt_id)
            encoded = self.feedback_encoder(resolved.receipt.action)
            if not isinstance(encoded, torch.Tensor):
                raise TypeError("policy-free live feedback encoder must return a tensor")
            if encoded.shape != action.shape:
                raise ValueError("policy-free live feedback encoder returned the wrong shape")
            action = encoded.to(device=device, dtype=dtype)
            reward = resolved.event.reward.to(device=device, dtype=dtype)
            propensity = resolved.receipt.propensity.to(device=device, dtype=dtype)
            present = resolved.event.present.to(device=device, dtype=dtype)
        return ControllerFeedback(
            action=action,
            reward=reward,
            propensity=propensity,
            has_feedback=present,
        ).validate(
            batch=self.batch_size,
            action_width=self.runtime.runtime.controller.feedback_width,
        )

    def _consume_without_action(
        self,
        events: AmodalEventCollection,
        outcomes: Sequence[ResolvedLiveOutcome],
        *,
        elapsed: float,
    ) -> None:
        """Advance learned state while an external identity gate abstains."""

        collection = self.runtime.runtime.input_bus(events)
        _, self._state = self.runtime.runtime.controller.step(
            collection,
            self._state,
            self._feedback(outcomes),
            self.runtime.runtime.memory,
            elapsed=elapsed,
        )

    def tick(
        self,
        events: AmodalEventCollection,
        outcomes: Sequence[ResolvedLiveOutcome],
        *,
        now: float,
        elapsed: float,
        identity_evidence: torch.Tensor | None = None,
    ) -> tuple[LiveActionProposal, ...]:
        del now
        events.validate(width=self.event_width)
        if events.payload.shape[0] != self.batch_size:
            raise ValueError("policy-free live event batch is incompatible")
        goal_state, identity_assignment = self._resolved_goal_state(identity_evidence)
        self._last_identity_assignment = identity_assignment
        if identity_assignment is not None and bool(torch.any(identity_assignment.abstained)):
            # A self-model tie is a safe no-op, never a guessed goal or action.
            self._consume_without_action(events, outcomes, elapsed=elapsed)
            return ()
        output, self._state = self.runtime.step_events(
            events,
            self._state,
            self._feedback(outcomes),
            goal_state,
            self.candidate_intentions,
            horizon=self.horizon,
            require_known=self.require_known,
            elapsed=elapsed,
        )
        self._last_output = output
        decision: LiveDecoderDecision = self.decoder.decide(
            output.intention, sample=self.sample
        )
        credit = PolicyFreeLiveCredit(
            selected_slot_id=output.selected_slot_id,
            planner_schema=self.runtime.planner.schema,
            identity_slot_id=(
                None
                if identity_assignment is None or self.batch_size != 1
                else int(identity_assignment.selected_slot[0].item())
            ),
        ).validate()
        return (
            LiveActionProposal(
                intention=output.intention,
                action=decision.action.detach(),
                propensity=decision.propensity.detach(),
                output_key=self.output_key,
                model_version=self.model_version,
                credit_state=credit,
            ).validate(batch_size=self.batch_size),
        )


__all__ = [
    "POLICY_FREE_LIVE_MACHINE_SCHEMA",
    "PolicyFreeAmodalLiveMachine",
    "PolicyFreeLiveCredit",
]
