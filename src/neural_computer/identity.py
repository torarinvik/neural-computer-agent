"""Replaceable identity-assignment artifacts at the external runtime seam.

The controller never consumes this object.  A frontend or external self model
turns learned causal evidence into an opaque slot choice; the live navigation
adapter may then select among caller-owned goal fragments.  Ties are explicit
abstentions rather than guessed identities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA = "neural-computer.external-identity-assignment.v1"
EXTERNAL_CAUSAL_IDENTITY_ARTIFACT_SCHEMA = (
    "neural-computer.external-causal-identity-artifact.v1"
)
EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V2_SCHEMA = (
    "neural-computer.persistent-causal-identity.v2"
)
EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V3_SCHEMA = (
    "neural-computer.persistent-causal-identity.v3"
)


@dataclass(frozen=True)
class ExternalIdentityAssignment:
    """One verified external slot assignment, or an explicit abstention."""

    selected_slot: torch.Tensor
    confidence: torch.Tensor
    abstained: torch.Tensor
    schema: str = EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA

    def validate(
        self,
        *,
        batch_size: int,
        slot_count: int,
    ) -> ExternalIdentityAssignment:
        if self.schema != EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA:
            raise ValueError("unsupported external identity-assignment schema")
        if slot_count < 1:
            raise ValueError("identity assignment requires at least one slot")
        if self.selected_slot.shape != (batch_size,) or self.selected_slot.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("selected identity slot must be an integer [batch]")
        if self.confidence.shape != (batch_size,):
            raise ValueError("identity confidence must have shape [batch]")
        if self.abstained.shape != (batch_size,) or self.abstained.dtype != torch.bool:
            raise ValueError("identity abstention must be boolean [batch]")
        if not bool(torch.isfinite(self.confidence).all()):
            raise ValueError("identity confidence must be finite")
        if bool(torch.any(self.confidence < 0.0)):
            raise ValueError("identity confidence cannot be negative")
        if bool(torch.any((self.selected_slot < 0) | (self.selected_slot >= slot_count))):
            raise ValueError("selected identity slot is outside the candidate set")
        return self


class ExternalCausalIdentityAssignment:
    """Gate learned causal slot evidence before external goal selection.

    ``evidence`` is an opaque score emitted by a replaceable self/assignment
    artifact.  This gate does not inspect coordinates, labels, rewards, or
    protocol actions.  It only checks the top-two margin and abstains when the
    evidence cannot distinguish candidates.
    """

    schema = EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA

    def __init__(
        self, *, margin: float = 0.15, minimum_evidence: float = 0.0
    ) -> None:
        if not torch.isfinite(torch.tensor(margin)) or margin < 0.0:
            raise ValueError("identity assignment margin must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(minimum_evidence)) or minimum_evidence < 0.0:
            raise ValueError(
                "identity assignment minimum evidence must be finite and nonnegative"
            )
        self.margin = float(margin)
        self.minimum_evidence = float(minimum_evidence)

    def configuration(self) -> dict[str, float | str]:
        return {
            "schema": self.schema,
            "behavior": "opaque-causal-evidence_top-slot_with_explicit_abstention_v1",
            "margin": self.margin,
            "minimum_evidence": self.minimum_evidence,
        }

    def resolve(self, evidence: torch.Tensor) -> ExternalIdentityAssignment:
        if evidence.ndim != 2 or evidence.shape[1] < 1:
            raise ValueError("identity evidence must have shape [batch, slots]")
        if not bool(torch.isfinite(evidence).all()):
            raise ValueError("identity evidence must be finite")
        batch_size, slot_count = evidence.shape
        values, indices = torch.topk(evidence, k=min(2, slot_count), dim=1)
        selected = indices[:, 0].to(dtype=torch.long)
        confidence = torch.softmax(evidence, dim=1).amax(dim=1)
        if slot_count == 1:
            margin = torch.full_like(confidence, float("inf"))
        else:
            margin = values[:, 0] - values[:, 1]
        abstained = (margin < self.margin) | (values[:, 0] < self.minimum_evidence)
        return ExternalIdentityAssignment(
            selected_slot=selected,
            confidence=confidence,
            abstained=abstained,
        ).validate(batch_size=batch_size, slot_count=slot_count)


class ExternalCausalIdentityArtifact:
    """Score action-conditioned dependence in bound learned event histories.

    The artifact is intentionally outside the controller.  It receives only a
    bounded history of learned event tensors and opaque action/intention
    features.  A centered cross-covariance score measures how much each track's
    event change is explained by those features; no coordinates, task labels,
    or verifier outcomes are accepted.
    """

    schema = EXTERNAL_CAUSAL_IDENTITY_ARTIFACT_SCHEMA

    def __init__(self, *, minimum_history: int = 4, epsilon: float = 1e-8) -> None:
        if minimum_history < 2:
            raise ValueError("causal identity history must contain at least two events")
        if not torch.isfinite(torch.tensor(epsilon)) or epsilon <= 0.0:
            raise ValueError("causal identity epsilon must be positive and finite")
        self.minimum_history = int(minimum_history)
        self.epsilon = float(epsilon)

    def configuration(self) -> dict[str, float | int | str]:
        return {
            "schema": self.schema,
            "behavior": "centered-action-event-cross-covariance_v1",
            "minimum_history": self.minimum_history,
            "epsilon": self.epsilon,
        }

    def evidence(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        if event_history.ndim != 4:
            raise ValueError("event history must have shape [batch, time, tracks, width]")
        if action_history.ndim != 3:
            raise ValueError("action history must have shape [batch, time-1, width]")
        batch_size, time_steps, track_count, _ = event_history.shape
        if time_steps < self.minimum_history:
            raise ValueError("causal identity history is shorter than its minimum")
        if action_history.shape[0] != batch_size or action_history.shape[1] != time_steps - 1:
            raise ValueError("action history must align with event transitions")
        if track_count < 1 or action_history.shape[2] < 1:
            raise ValueError("causal identity history needs tracks and action features")
        if not bool(torch.isfinite(event_history).all()) or not bool(
            torch.isfinite(action_history).all()
        ):
            raise ValueError("causal identity histories must be finite")
        event_delta = event_history[:, 1:] - event_history[:, :-1]
        centered_actions = action_history - action_history.mean(dim=1, keepdim=True)
        centered_delta = event_delta - event_delta.mean(dim=1, keepdim=True)
        covariance = torch.einsum(
            "bta,btkd->bkad", centered_actions, centered_delta
        )
        action_energy = centered_actions.square().sum(dim=(1, 2))
        delta_energy = centered_delta.square().sum(dim=(1, 3))
        denominator = action_energy.unsqueeze(1) * delta_energy
        explained = covariance.square().sum(dim=(2, 3)) / denominator.clamp_min(
            self.epsilon
        )
        return explained.clamp_min(0.0).clamp_max(1.0).sqrt()


class PersistentCausalIdentityV2:
    """Persist action-conditioned self dynamics and rebind them to tracks.

    This artifact stores a compact causal signature, not a slot/object index.
    Each new episode is matched against that signature, so object replacement
    and slot permutation do not become persistent identity.  A high-confidence
    episode is the only event that updates the signature.  Contradictions,
    weak applicability, and missing evidence quarantine the model; while
    quarantined, statistics are frozen until fresh high-confidence episodes
    provide a replacement signature.

    The artifact is external to the controller and accepts only learned event
    tensors plus opaque action/intention features.  ``episode_id`` is an
    external lifetime token: passing it makes repeated live ticks idempotent
    for persistent-statistics updates.
    """

    schema = EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V2_SCHEMA

    def __init__(
        self,
        *,
        minimum_history: int = 4,
        margin: float = 0.15,
        minimum_evidence: float = 0.2,
        minimum_similarity: float = 0.65,
        recovery_episodes: int = 2,
        epsilon: float = 1e-8,
    ) -> None:
        if minimum_history < 2:
            raise ValueError("persistent identity history must contain at least two events")
        values = {
            "margin": margin,
            "minimum_evidence": minimum_evidence,
            "minimum_similarity": minimum_similarity,
            "epsilon": epsilon,
        }
        if any(not torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise ValueError("persistent identity parameters must be finite")
        if margin < 0.0 or minimum_evidence < 0.0:
            raise ValueError("persistent identity gates must be nonnegative")
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("persistent identity similarity must lie in [0, 1]")
        if epsilon <= 0.0:
            raise ValueError("persistent identity epsilon must be positive")
        if recovery_episodes < 1:
            raise ValueError("persistent identity recovery needs at least one episode")
        self.minimum_history = int(minimum_history)
        self.margin = float(margin)
        self.minimum_evidence = float(minimum_evidence)
        self.minimum_similarity = float(minimum_similarity)
        self.recovery_episodes = int(recovery_episodes)
        self.epsilon = float(epsilon)
        self.local_artifact = ExternalCausalIdentityArtifact(
            minimum_history=self.minimum_history,
            epsilon=self.epsilon,
        )
        self.assignment_gate = ExternalCausalIdentityAssignment(
            margin=self.margin,
            minimum_evidence=self.minimum_evidence,
        )
        self._prototype: torch.Tensor | None = None
        self._support = 0
        self._status = "uninitialized"
        self._reason = "no_persistent_model"
        self._quarantine_count = 0
        self._recovery_buffer: list[torch.Tensor] = []
        self._committed_episode_id: object | None = None
        self._buffered_episode_ids: set[object] = set()
        self._last_evidence: torch.Tensor | None = None
        self._last_similarity: torch.Tensor | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def support(self) -> int:
        return self._support

    @property
    def quarantine_count(self) -> int:
        return self._quarantine_count

    @property
    def last_evidence(self) -> torch.Tensor | None:
        return None if self._last_evidence is None else self._last_evidence.clone()

    @property
    def last_similarity(self) -> torch.Tensor | None:
        return None if self._last_similarity is None else self._last_similarity.clone()

    def configuration(self) -> dict[str, float | int | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "behavior": "persistent-action-conditioned-dynamics_rebind-quarantine-relearn_v2",
            "minimum_history": self.minimum_history,
            "margin": self.margin,
            "minimum_evidence": self.minimum_evidence,
            "minimum_similarity": self.minimum_similarity,
            "recovery_episodes": self.recovery_episodes,
            "epsilon": self.epsilon,
            "assignment_gate": self.assignment_gate.configuration(),
            "local_artifact": self.local_artifact.configuration(),
        }

    def reset(self) -> None:
        """Forget persistent statistics and return to an uninitialized state."""

        self._prototype = None
        self._support = 0
        self._status = "uninitialized"
        self._reason = "no_persistent_model"
        self._quarantine_count = 0
        self._recovery_buffer.clear()
        self._committed_episode_id = None
        self._buffered_episode_ids.clear()
        self._last_evidence = None
        self._last_similarity = None

    @staticmethod
    def _validate_present(
        event_present: torch.Tensor | None,
        *,
        batch_size: int,
        time_steps: int,
        track_count: int,
    ) -> torch.Tensor:
        if event_present is None:
            return torch.ones(
                batch_size,
                time_steps,
                track_count,
                dtype=torch.bool,
            )
        if event_present.shape != (batch_size, time_steps, track_count):
            raise ValueError("event presence must have shape [batch, time, tracks]")
        if event_present.dtype != torch.bool:
            raise ValueError("event presence must be boolean")
        return event_present

    def _signatures(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized action-conditioned delta signatures [tracks, actions, width]."""

        event_delta = event_history[:, 1:] - event_history[:, :-1]
        centered_actions = action_history - action_history.mean(dim=1, keepdim=True)
        centered_delta = event_delta - event_delta.mean(dim=1, keepdim=True)
        covariance = torch.einsum(
            "bta,btkd->bkad", centered_actions, centered_delta
        )
        action_energy = centered_actions.square().sum(dim=(1, 2))
        signatures = covariance / action_energy[:, None, None, None].clamp_min(
            self.epsilon
        )
        return signatures[0]

    def _empty_assignment(self, slot_count: int) -> ExternalIdentityAssignment:
        evidence = torch.zeros(1, slot_count)
        self._last_evidence = evidence
        return self.assignment_gate.resolve(evidence)

    def _quarantine(self, reason: str) -> None:
        if self._status != "quarantined":
            self._quarantine_count += 1
        self._status = "quarantined"
        self._reason = reason
        self._recovery_buffer.clear()
        self._buffered_episode_ids.clear()

    def _update_prototype(self, signature: torch.Tensor) -> None:
        signature = signature.detach().clone()
        if self._prototype is None:
            self._prototype = signature
        else:
            self._prototype = (
                self._prototype * self._support + signature
            ) / float(self._support + 1)
        self._support += 1
        self._status = "active"
        self._reason = "high_confidence_assignment"

    def _similarities(self, signatures: torch.Tensor) -> torch.Tensor:
        if self._prototype is None:
            return torch.ones(signatures.shape[0])
        prototype = self._prototype.to(device=signatures.device, dtype=signatures.dtype)
        flattened = signatures.flatten(start_dim=1)
        reference = prototype.flatten()
        reference_norm = reference.norm()
        norms = flattened.norm(dim=1)
        denominator = norms * reference_norm
        cosine = (flattened @ reference) / denominator.clamp_min(self.epsilon)
        return cosine.clamp_min(0.0).clamp_max(1.0)

    def resolve(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
        *,
        event_present: torch.Tensor | None = None,
        episode_id: object | None = None,
    ) -> ExternalIdentityAssignment:
        """Match the current tracks and update only after a gated assignment."""

        local_evidence = self.local_artifact.evidence(event_history, action_history)
        batch_size, time_steps, track_count, _ = event_history.shape
        if batch_size != 1:
            raise ValueError("persistent causal identity v2 currently requires batch size one")
        present = self._validate_present(
            event_present,
            batch_size=batch_size,
            time_steps=time_steps,
            track_count=track_count,
        )
        self._last_similarity = torch.zeros(track_count)
        if not bool(present.all()):
            self._last_evidence = torch.zeros_like(local_evidence)
            if self._prototype is not None:
                self._quarantine("missing_evidence")
            else:
                self._reason = "missing_evidence"
            return self._empty_assignment(track_count)

        signatures = self._signatures(event_history, action_history)
        if self._status == "quarantined":
            candidate = self.assignment_gate.resolve(local_evidence)
            self._last_evidence = local_evidence.detach().clone()
            if bool(candidate.abstained[0]):
                self._reason = "recovery_evidence_insufficient"
                self._recovery_buffer.clear()
                self._buffered_episode_ids.clear()
                return self._empty_assignment(track_count)
            if episode_id is None or episode_id not in self._buffered_episode_ids:
                self._recovery_buffer.append(
                    signatures[int(candidate.selected_slot[0].item())].detach().clone()
                )
                if episode_id is not None:
                    self._buffered_episode_ids.add(episode_id)
            if len(self._recovery_buffer) < self.recovery_episodes:
                self._reason = "quarantined_relearning"
                return self._empty_assignment(track_count)
            self._prototype = torch.stack(self._recovery_buffer).mean(dim=0)
            self._support = len(self._recovery_buffer)
            self._status = "active"
            self._reason = "relearned_requires_confirmation"
            self._recovery_buffer.clear()
            self._buffered_episode_ids.clear()
            self._committed_episode_id = episode_id
            return self._empty_assignment(track_count)

        similarities = self._similarities(signatures)
        self._last_similarity = similarities.detach().clone()
        if self._prototype is None:
            evidence = local_evidence
        else:
            evidence = local_evidence * similarities.unsqueeze(0)
        self._last_evidence = evidence.detach().clone()
        assignment = self.assignment_gate.resolve(evidence)
        if bool(assignment.abstained[0]):
            if self._prototype is not None:
                self._quarantine("low_applicability_or_margin")
            else:
                self._reason = "initial_evidence_insufficient"
            return assignment
        selected = int(assignment.selected_slot[0].item())
        if self._prototype is not None and float(similarities[selected]) < self.minimum_similarity:
            self._quarantine("causal_signature_contradiction")
            return self._empty_assignment(track_count)
        if episode_id is None or episode_id != self._committed_episode_id:
            self._update_prototype(signatures[selected])
            self._committed_episode_id = episode_id
        return assignment


@dataclass
class _PersistentCausalGraph:
    """Compact state-conditioned transition evidence for identity v3.

    State vectors are learned event tensors, never coordinates.  Edges retain
    the opaque action vector that caused a transition.  Keeping the graph in
    the external artifact lets the identity model compare *which action moves
    which learned state* instead of comparing a global covariance whose value
    drifts whenever a history prefix grows.
    """

    states: torch.Tensor
    edges: set[tuple[int, tuple[float, ...], int]]

    def clone(self) -> _PersistentCausalGraph:
        return _PersistentCausalGraph(self.states.detach().clone(), set(self.edges))


class PersistentCausalIdentityV3:
    """Persist a state-conditioned action-labelled transition graph.

    V2 persisted a global action/event covariance.  That statistic is
    sensitive to the absolute learned event vectors and to the length of the
    currently visible prefix.  V3 instead stores a compact graph over learned
    event states and opaque action vectors.  A candidate must match the
    action-labelled transitions whose states are already known; unfamiliar
    state support is ignored only until enough known edges exist, after which
    the model abstains rather than extrapolating confidently.

    The artifact remains external to the controller.  A high-confidence
    assignment is cached for the rest of its episode, while later evidence
    may quarantine it if known transitions contradict the persistent graph.
    """

    schema = EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V3_SCHEMA

    def __init__(
        self,
        *,
        minimum_history: int = 4,
        margin: float = 0.15,
        minimum_evidence: float = 0.2,
        minimum_similarity: float = 0.65,
        minimum_state_similarity: float = 0.995,
        minimum_known_edges: int = 2,
        recovery_episodes: int = 2,
        epsilon: float = 1e-8,
    ) -> None:
        if minimum_history < 2:
            raise ValueError("persistent identity history must contain at least two events")
        values = {
            "margin": margin,
            "minimum_evidence": minimum_evidence,
            "minimum_similarity": minimum_similarity,
            "minimum_state_similarity": minimum_state_similarity,
            "epsilon": epsilon,
        }
        if any(not torch.isfinite(torch.tensor(value)) for value in values.values()):
            raise ValueError("persistent identity parameters must be finite")
        if margin < 0.0 or minimum_evidence < 0.0:
            raise ValueError("persistent identity gates must be nonnegative")
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("persistent identity similarity must lie in [0, 1]")
        if not 0.0 < minimum_state_similarity <= 1.0:
            raise ValueError("state similarity must lie in (0, 1]")
        if minimum_known_edges < 1:
            raise ValueError("persistent identity needs at least one known edge")
        if epsilon <= 0.0:
            raise ValueError("persistent identity epsilon must be positive")
        if recovery_episodes < 1:
            raise ValueError("persistent identity recovery needs at least one episode")
        self.minimum_history = int(minimum_history)
        self.margin = float(margin)
        self.minimum_evidence = float(minimum_evidence)
        self.minimum_similarity = float(minimum_similarity)
        self.minimum_state_similarity = float(minimum_state_similarity)
        self.minimum_known_edges = int(minimum_known_edges)
        self.recovery_episodes = int(recovery_episodes)
        self.epsilon = float(epsilon)
        self.local_artifact = ExternalCausalIdentityArtifact(
            minimum_history=self.minimum_history,
            epsilon=self.epsilon,
        )
        self.assignment_gate = ExternalCausalIdentityAssignment(
            margin=self.margin,
            minimum_evidence=self.minimum_evidence,
        )
        self._prototype: _PersistentCausalGraph | None = None
        self._support = 0
        self._status = "uninitialized"
        self._reason = "no_persistent_model"
        self._quarantine_count = 0
        self._recovery_buffer: list[_PersistentCausalGraph] = []
        self._buffered_episode_ids: set[object] = set()
        self._episode_id: object | None = None
        self._episode_assignment: ExternalIdentityAssignment | None = None
        self._last_evidence: torch.Tensor | None = None
        self._last_similarity: torch.Tensor | None = None

    @property
    def status(self) -> str:
        return self._status

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def support(self) -> int:
        return self._support

    @property
    def quarantine_count(self) -> int:
        return self._quarantine_count

    @property
    def last_evidence(self) -> torch.Tensor | None:
        return None if self._last_evidence is None else self._last_evidence.clone()

    @property
    def last_similarity(self) -> torch.Tensor | None:
        return None if self._last_similarity is None else self._last_similarity.clone()

    def configuration(self) -> dict[str, float | int | str | dict[str, object]]:
        return {
            "schema": self.schema,
            "behavior": "persistent-state-conditioned-action-graph_rebind-quarantine-relearn_v3",
            "minimum_history": self.minimum_history,
            "margin": self.margin,
            "minimum_evidence": self.minimum_evidence,
            "minimum_similarity": self.minimum_similarity,
            "minimum_state_similarity": self.minimum_state_similarity,
            "minimum_known_edges": self.minimum_known_edges,
            "recovery_episodes": self.recovery_episodes,
            "epsilon": self.epsilon,
            "assignment_gate": self.assignment_gate.configuration(),
            "local_artifact": self.local_artifact.configuration(),
        }

    def reset(self) -> None:
        self._prototype = None
        self._support = 0
        self._status = "uninitialized"
        self._reason = "no_persistent_model"
        self._quarantine_count = 0
        self._recovery_buffer.clear()
        self._buffered_episode_ids.clear()
        self._episode_id = None
        self._episode_assignment = None
        self._last_evidence = None
        self._last_similarity = None

    @staticmethod
    def _validate_present(
        event_present: torch.Tensor | None,
        *,
        batch_size: int,
        time_steps: int,
        track_count: int,
    ) -> torch.Tensor:
        if event_present is None:
            return torch.ones(
                batch_size,
                time_steps,
                track_count,
                dtype=torch.bool,
            )
        if event_present.shape != (batch_size, time_steps, track_count):
            raise ValueError("event presence must have shape [batch, time, tracks]")
        if event_present.dtype != torch.bool:
            raise ValueError("event presence must be boolean")
        return event_present

    def _action_label(self, action: torch.Tensor) -> tuple[float, ...]:
        return tuple(round(float(value), 6) for value in action.tolist())

    def _state_match(
        self,
        state: torch.Tensor,
        states: torch.Tensor,
    ) -> tuple[float, int | None]:
        if states.numel() == 0:
            return 0.0, None
        state = state.to(device=states.device, dtype=states.dtype)
        scores = torch.nn.functional.cosine_similarity(
            states, state.unsqueeze(0), dim=1, eps=self.epsilon
        )
        value, index = scores.max(dim=0)
        return float(value), int(index.item())

    def _graph(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
        track: int,
    ) -> _PersistentCausalGraph:
        states: list[torch.Tensor] = []
        labels: list[int] = []
        for state in event_history[0, :, track]:
            similarity, index = self._state_match(
                state,
                torch.stack(states) if states else state.new_empty((0, state.numel())),
            )
            if index is None or similarity < self.minimum_state_similarity:
                states.append(state.detach().clone())
                index = len(states) - 1
            labels.append(index)
        edges = {
            (labels[index], self._action_label(action_history[0, index]), labels[index + 1])
            for index in range(action_history.shape[1])
        }
        return _PersistentCausalGraph(torch.stack(states), edges)

    def _responsiveness(self, graph: _PersistentCausalGraph) -> float:
        if not graph.edges:
            return 0.0
        changing = sum(source != target for source, _, target in graph.edges)
        if not changing:
            return 0.0
        successors: dict[tuple[int, tuple[float, ...]], set[int]] = {}
        for source, action, target in graph.edges:
            successors.setdefault((source, action), set()).add(target)
        branch = sum(len(targets) > 1 for targets in successors.values())
        return min(
            1.0,
            0.5 * changing / len(graph.edges)
            + 0.5 * branch / max(1, len(successors)),
        )

    def _graph_similarity(
        self,
        graph: _PersistentCausalGraph,
        prototype: _PersistentCausalGraph,
    ) -> float:
        if not graph.edges or not prototype.edges:
            return 0.0
        mapping: dict[int, int] = {}
        for index, state in enumerate(graph.states):
            similarity, target = self._state_match(state, prototype.states)
            if target is not None and similarity >= self.minimum_state_similarity:
                mapping[index] = target
        # A growing episode legitimately reveals new state/action pairs.  They
        # are not contradictions and must not make the score decay with every
        # additional prefix.  Only pairs already present in the persistent
        # graph are comparable; a known pair with a different successor is a
        # genuine dynamics contradiction.
        transitions: dict[tuple[int, tuple[float, ...]], set[int]] = {}
        for source, action, target in prototype.edges:
            transitions.setdefault((source, action), set()).add(target)
        known = 0
        matched = 0
        for source, action, target in graph.edges:
            if source not in mapping or target not in mapping:
                continue
            key = (mapping[source], action)
            if key not in transitions:
                continue
            known += 1
            if mapping[target] in transitions[key]:
                matched += 1
        if known < self.minimum_known_edges:
            # New state/action pairs are expected as an episode grows.  They
            # are not contradictions until the persistent graph has enough
            # shared pairs to make a comparison.
            return -1.0
        return matched / float(known)

    def _merge_graph(
        self,
        graph: _PersistentCausalGraph,
    ) -> None:
        if self._prototype is None:
            self._prototype = graph.clone()
            return
        prototype = self._prototype
        states = [state.detach().clone() for state in prototype.states]
        mapping: dict[int, int] = {}
        for index, state in enumerate(graph.states):
            similarity, target = self._state_match(
                state,
                torch.stack(states) if states else state.new_empty((0, state.numel())),
            )
            if target is None or similarity < self.minimum_state_similarity:
                states.append(state.detach().clone())
                target = len(states) - 1
            mapping[index] = target
        prototype.states = torch.stack(states)
        prototype.edges.update(
            (mapping[source], action, mapping[target])
            for source, action, target in graph.edges
        )

    def _empty_assignment(self, slot_count: int) -> ExternalIdentityAssignment:
        evidence = torch.zeros(1, slot_count)
        self._last_evidence = evidence
        return self.assignment_gate.resolve(evidence)

    def _quarantine(self, reason: str) -> None:
        if self._status != "quarantined":
            self._quarantine_count += 1
        self._status = "quarantined"
        self._reason = reason
        self._episode_assignment = None
        self._recovery_buffer.clear()
        self._buffered_episode_ids.clear()

    def _update_prototype(self, graph: _PersistentCausalGraph) -> None:
        self._merge_graph(graph)
        self._support += 1
        self._status = "active"
        self._reason = "high_confidence_assignment"

    @torch.no_grad()
    def resolve(
        self,
        event_history: torch.Tensor,
        action_history: torch.Tensor,
        *,
        event_present: torch.Tensor | None = None,
        episode_id: object | None = None,
    ) -> ExternalIdentityAssignment:
        if event_history.ndim != 4 or action_history.ndim != 3:
            raise ValueError("persistent identity histories have invalid rank")
        batch_size, time_steps, track_count, event_width = event_history.shape
        if batch_size != 1:
            raise ValueError("persistent causal identity v3 currently requires batch size one")
        if time_steps < self.minimum_history:
            raise ValueError("causal identity history is shorter than its minimum")
        if action_history.shape[0] != 1 or action_history.shape[1] != time_steps - 1:
            raise ValueError("action history must align with event transitions")
        if event_width < 1 or action_history.shape[2] < 1:
            raise ValueError("persistent identity histories need non-empty widths")
        if not bool(torch.isfinite(event_history).all()) or not bool(
            torch.isfinite(action_history).all()
        ):
            raise ValueError("persistent identity histories must be finite")
        present = self._validate_present(
            event_present,
            batch_size=batch_size,
            time_steps=time_steps,
            track_count=track_count,
        )
        if not bool(present.all()):
            self._last_similarity = torch.zeros(track_count)
            self._last_evidence = torch.zeros(1, track_count)
            if self._prototype is not None:
                self._quarantine("missing_evidence")
            else:
                self._reason = "missing_evidence"
            return self._empty_assignment(track_count)

        graphs = [self._graph(event_history, action_history, track) for track in range(track_count)]
        local_evidence = self.local_artifact.evidence(event_history, action_history)
        responsiveness = torch.tensor(
            [self._responsiveness(graph) for graph in graphs],
            dtype=local_evidence.dtype,
        ).unsqueeze(0)
        # Preserve the calibrated covariance scale while still rejecting a
        # static track.  A linear product made short, valid histories fail
        # merely because they had not yet visited many distinct states.
        response_gate = responsiveness.clamp_min(0.0).sqrt()
        if self._status == "quarantined":
            recovery_evidence = local_evidence * response_gate
            candidate = self.assignment_gate.resolve(recovery_evidence)
            self._last_evidence = recovery_evidence.detach().clone()
            if bool(candidate.abstained[0]):
                self._reason = "recovery_evidence_insufficient"
                self._recovery_buffer.clear()
                self._buffered_episode_ids.clear()
                return self._empty_assignment(track_count)
            if episode_id is None or episode_id not in self._buffered_episode_ids:
                self._recovery_buffer.append(
                    graphs[int(candidate.selected_slot[0].item())].clone()
                )
                if episode_id is not None:
                    self._buffered_episode_ids.add(episode_id)
            if len(self._recovery_buffer) < self.recovery_episodes:
                self._reason = "quarantined_relearning"
                return self._empty_assignment(track_count)
            self._prototype = self._recovery_buffer[0].clone()
            for graph in self._recovery_buffer[1:]:
                self._merge_graph(graph)
            self._support = len(self._recovery_buffer)
            self._status = "active"
            self._reason = "relearned_requires_confirmation"
            self._recovery_buffer.clear()
            self._buffered_episode_ids.clear()
            self._episode_assignment = None
            self._episode_id = episode_id
            return self._empty_assignment(track_count)

        similarities = torch.ones(track_count, dtype=local_evidence.dtype)
        if self._prototype is not None:
            similarities = torch.tensor(
                [self._graph_similarity(graph, self._prototype) for graph in graphs],
                dtype=local_evidence.dtype,
            )
        evidence = (
            local_evidence
            * response_gate
            * similarities.clamp_min(0.0).unsqueeze(0)
        )
        self._last_similarity = similarities.detach().clone()
        self._last_evidence = evidence.detach().clone()

        if (
            episode_id is not None
            and episode_id == self._episode_id
            and self._episode_assignment is not None
        ):
            selected = int(self._episode_assignment.selected_slot[0].item())
            if selected >= track_count:
                self._episode_assignment = None
            elif (
                self._prototype is not None
                and float(similarities[selected]) >= 0.0
                and float(similarities[selected]) < self.minimum_similarity
            ):
                self._quarantine("causal_graph_contradiction")
                return self._empty_assignment(track_count)
            else:
                # Extend the already verified episode graph without counting
                # another persistent update.  This makes the prototype
                # prefix-stable while still letting later episodes test more
                # state/action pairs against it.
                self._merge_graph(graphs[selected])
                return self._episode_assignment

        assignment = self.assignment_gate.resolve(evidence)
        if bool(assignment.abstained[0]):
            if self._prototype is not None:
                if bool(torch.all(similarities < 0.0)):
                    self._reason = "state_evidence_insufficient"
                    return self._empty_assignment(track_count)
                self._quarantine("low_applicability_or_margin")
            else:
                self._reason = "initial_evidence_insufficient"
            return assignment
        selected = int(assignment.selected_slot[0].item())
        if (
            self._prototype is not None
            and float(similarities[selected]) >= 0.0
            and float(similarities[selected]) < self.minimum_similarity
        ):
            self._quarantine("causal_graph_contradiction")
            return self._empty_assignment(track_count)
        self._update_prototype(graphs[selected])
        self._episode_id = episode_id
        self._episode_assignment = assignment
        return assignment


__all__ = [
    "EXTERNAL_CAUSAL_IDENTITY_ARTIFACT_SCHEMA",
    "EXTERNAL_IDENTITY_ASSIGNMENT_SCHEMA",
    "EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V2_SCHEMA",
    "EXTERNAL_PERSISTENT_CAUSAL_IDENTITY_V3_SCHEMA",
    "ExternalCausalIdentityArtifact",
    "ExternalCausalIdentityAssignment",
    "ExternalIdentityAssignment",
    "PersistentCausalIdentityV2",
    "PersistentCausalIdentityV3",
]
