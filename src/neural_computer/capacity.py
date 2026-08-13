"""Opaque planning for external-memory admission under capacity pressure.

The planner is deliberately a memory-side component.  It consumes learned
keys/values and generic storage metadata, not task labels, raw modalities, or
device protocols.  Its output is only a proposal.  Protection masks and the
behavior verifier remain authoritative at the transaction boundary.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .memory import MemoryCandidates

CAPACITY_PLANNER_SCHEMA = "neural-computer.opaque-capacity-planner.v8"
CAPACITY_PLANNER_ADAPTER_SCHEMA = (
    "neural-computer.opaque-capacity-planner-residual-adapter.v1"
)
ADMISSION_ACTIONS = ("admit", "evict", "consolidate", "grow")


@dataclass(frozen=True)
class CapacityPlannerOutput:
    """Permutation-equivariant scores for one admission decision."""

    action_logits: torch.Tensor
    eviction_scores: torch.Tensor
    pair_scores: torch.Tensor
    valid_evictions: torch.Tensor
    valid_pairs: torch.Tensor
    available_actions: torch.Tensor


@dataclass(frozen=True)
class CapacityPlan:
    """One opaque admission proposal for an external-memory transaction."""

    action: str
    action_index: int
    eviction_index: int | None
    pair: tuple[int, int] | None
    score: torch.Tensor


class OpaqueCapacityPlannerAdapter(nn.Module):
    """Zero-initialized residual policy state for replay-free adaptation.

    The adapter consumes the same generic invariant features as the frozen
    planner and adds residual action, eviction, and pair scores.  It is an
    independently replaceable external policy component: adapting it does not
    change the base planner weights.
    """

    schema = CAPACITY_PLANNER_ADAPTER_SCHEMA

    def __init__(self, *, width: int, hidden: int = 64) -> None:
        super().__init__()
        if min(width, hidden) < 1:
            raise ValueError("capacity planner adapter widths must be positive")
        self.width = int(width)
        self.hidden = int(hidden)
        self.global_width = 15
        self.row_width = 4
        self.eviction_width = self.global_width + self.row_width + 2
        self.pair_width = 10
        self.action_network = nn.Sequential(
            nn.Linear(self.global_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(ADMISSION_ACTIONS)),
        )
        self.eviction_network = nn.Sequential(
            nn.Linear(self.eviction_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.pair_network = nn.Sequential(
            nn.Linear(self.pair_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Keep the residual exactly zero at construction, but retain a live
        # feature extractor.  Zeroing every layer would make the hidden
        # representation learn only after the output head moved, needlessly
        # starving a replay-free adapter of credit during the first updates.
        for network in (
            self.action_network,
            self.eviction_network,
            self.pair_network,
        ):
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "width": self.width,
            "hidden": self.hidden,
            "updates": "single_verifier_utility_without_replay_v1",
            "base_weights": "frozen_external_policy_v1",
        }


class OpaqueCapacityPlanner(nn.Module):
    """Learn a generic admission action without physical-slot semantics.

    The planner scores the action set ``admit``, ``evict``, ``consolidate``,
    and ``grow`` from an incoming learned key/value plus an unordered bank of
    learned rows.  Candidate rows and pair scores are equivariant to a joint
    permutation of the bank.  The caller supplies ``protected`` and whether
    verified consolidation is currently available; those are safety facts,
    not learned substitutes for retention or behavior verification.
    """

    schema = CAPACITY_PLANNER_SCHEMA

    def __init__(
        self,
        *,
        width: int,
        hidden: int = 64,
        learning_rate: float = 1e-2,
        pair_similarity_prior: float = 0.5,
        eviction_similarity_prior: float = 0.5,
    ) -> None:
        super().__init__()
        if min(width, hidden) < 1:
            raise ValueError("capacity planner widths must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("capacity planner learning rate must be positive")
        if pair_similarity_prior < 0.0 or not math.isfinite(pair_similarity_prior):
            raise ValueError("capacity planner pair similarity prior is invalid")
        if eviction_similarity_prior < 0.0 or not math.isfinite(eviction_similarity_prior):
            raise ValueError("capacity planner eviction similarity prior is invalid")
        self.width = int(width)
        self.hidden = int(hidden)
        self.learning_rate = float(learning_rate)
        self.pair_similarity_prior = float(pair_similarity_prior)
        self.eviction_similarity_prior = float(eviction_similarity_prior)
        row_width = 4
        incoming_width = 2
        self.row_width = row_width
        self.incoming_width = incoming_width
        global_width = incoming_width + row_width + 9
        self.global_width = global_width
        self.action_network = nn.Sequential(
            nn.Linear(global_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(ADMISSION_ACTIONS)),
        )
        with torch.no_grad():
            self.action_network[-1].bias.copy_(
                torch.tensor([0.25, 0.0, 0.25, -0.25])
            )
        self.eviction_network = nn.Sequential(
            nn.Linear(global_width + row_width + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        pair_relation_width = 10
        self.pair_relation_width = pair_relation_width
        pair_width = pair_relation_width
        self.pair_network = nn.Sequential(
            nn.Linear(pair_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def configuration(self) -> dict[str, int | float | str | tuple[str, ...]]:
        return {
            "schema": self.schema,
            "width": self.width,
            "hidden": self.hidden,
            "learning_rate": self.learning_rate,
            "pair_similarity_prior": self.pair_similarity_prior,
            "eviction_similarity_prior": self.eviction_similarity_prior,
            "actions": ADMISSION_ACTIONS,
            "row_features": "invariant_key_value_norm_strength_relative_age",
            "pair_relations": (
                "key_cosine",
                "value_cosine",
                "support_sum",
                "support_abs_difference",
                "age_sum",
                "age_abs_difference",
                "incoming_key_cosines",
                "incoming_value_cosines",
            ),
            "updates": "single_verifier_utility_without_replay_v1",
            "action_mask": "caller_constrained_structural_availability_v1",
            "stability_regularization": "replay_free_parameter_anchor_v1",
        }

    def _validate_inputs(
        self,
        bank: MemoryCandidates,
        incoming_key: torch.Tensor,
        incoming_value: torch.Tensor,
        protected: torch.Tensor,
        *,
        consolidation_available: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        bank.validate(width=self.width, capacity=bank.keys.shape[1])
        batch, capacity, _ = bank.keys.shape
        if incoming_key.shape != (batch, self.width):
            raise ValueError(
                f"incoming_key must have shape [{batch}, {self.width}]"
            )
        if incoming_value.shape != (batch, self.width):
            raise ValueError(
                f"incoming_value must have shape [{batch}, {self.width}]"
            )
        if protected.shape != (batch, capacity) or protected.dtype != torch.bool:
            raise ValueError(
                f"protected must be bool [{batch}, {capacity}]"
            )
        if consolidation_available.shape != (batch,) or consolidation_available.dtype != torch.bool:
            raise ValueError("consolidation_available must be bool [batch]")
        if action_mask.shape != (batch, len(ADMISSION_ACTIONS)) or action_mask.dtype != torch.bool:
            raise ValueError(
                "action_mask must be bool [batch, admission_action_count]"
            )
        tensors = (incoming_key, incoming_value)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("incoming planner tensors must be finite")

    def _row_features(self, bank: MemoryCandidates) -> torch.Tensor:
        timestamps = bank.timestamps
        occupied = bank.occupied
        occupied_timestamps = timestamps.masked_fill(~occupied, 0.0)
        latest = occupied_timestamps.amax(dim=-1, keepdim=True)
        earliest = torch.where(occupied, timestamps, latest).amin(
            dim=-1, keepdim=True
        )
        relative_age = (latest - timestamps) / (latest - earliest).clamp_min(1.0)
        return torch.cat(
            (
                torch.linalg.vector_norm(bank.keys, dim=-1, keepdim=True),
                torch.linalg.vector_norm(bank.values, dim=-1, keepdim=True),
                bank.strengths.unsqueeze(-1),
                relative_age.unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        bank: MemoryCandidates,
        incoming_key: torch.Tensor,
        incoming_value: torch.Tensor,
        protected: torch.Tensor,
        *,
        consolidation_available: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        adapter: OpaqueCapacityPlannerAdapter | None = None,
    ) -> CapacityPlannerOutput:
        """Score an admission state without assigning meaning to rows."""

        batch, capacity, _ = bank.keys.shape
        if consolidation_available is None:
            consolidation_available = torch.ones(
                batch, dtype=torch.bool, device=bank.keys.device
            )
        if action_mask is None:
            action_mask = torch.ones(
                batch,
                len(ADMISSION_ACTIONS),
                dtype=torch.bool,
                device=bank.keys.device,
            )
        if adapter is not None:
            if not isinstance(adapter, OpaqueCapacityPlannerAdapter):
                raise TypeError("capacity planner adapter is invalid")
            if adapter.width != self.width:
                raise ValueError("capacity planner adapter width is invalid")
        self._validate_inputs(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=consolidation_available,
            action_mask=action_mask,
        )
        row_features = self._row_features(bank)
        occupied = bank.occupied
        counts = occupied.sum(dim=-1, keepdim=True).to(row_features.dtype)
        pooled = (row_features * occupied.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1.0)
        normalized_keys = torch.nn.functional.normalize(bank.keys, dim=-1)
        normalized_values = torch.nn.functional.normalize(bank.values, dim=-1)
        normalized_incoming_key = torch.nn.functional.normalize(incoming_key, dim=-1)
        normalized_incoming_value = torch.nn.functional.normalize(incoming_value, dim=-1)
        incoming_key_cosines = (
            normalized_keys * normalized_incoming_key[:, None, :]
        ).sum(dim=-1)
        incoming_value_cosines = (
            normalized_values * normalized_incoming_value[:, None, :]
        ).sum(dim=-1)
        occupied_float = occupied.to(dtype=row_features.dtype)
        relation_means = torch.stack(
            (
                (incoming_key_cosines * occupied_float).sum(dim=-1)
                / counts.squeeze(-1).clamp_min(1.0),
                (incoming_value_cosines * occupied_float).sum(dim=-1)
                / counts.squeeze(-1).clamp_min(1.0),
            ),
            dim=-1,
        )
        relation_maxima = torch.stack(
            (
                incoming_key_cosines.masked_fill(~occupied, -torch.inf).amax(dim=-1),
                incoming_value_cosines.masked_fill(~occupied, -torch.inf).amax(dim=-1),
            ),
            dim=-1,
        )
        incoming = torch.stack(
            (
                torch.linalg.vector_norm(incoming_key, dim=-1),
                torch.linalg.vector_norm(incoming_value, dim=-1),
            ),
            dim=-1,
        )
        occupancy_fraction = counts / float(capacity)
        protected_fraction = (
            (protected & occupied).sum(dim=-1, keepdim=True).to(row_features.dtype)
            / counts.clamp_min(1.0)
        )
        similarity = normalized_values @ normalized_values.transpose(-1, -2)
        pair_mask = bank.occupied[:, :, None] & bank.occupied[:, None, :]
        pair_mask = pair_mask & ~torch.eye(
            capacity, dtype=torch.bool, device=bank.keys.device
        )
        pair_similarity = similarity.masked_fill(~pair_mask, -torch.inf).amax(
            dim=(-1, -2)
        ).unsqueeze(-1)
        pair_similarity = torch.where(
            torch.isfinite(pair_similarity), pair_similarity, torch.zeros_like(pair_similarity)
        )
        global_features = torch.cat(
            (
                incoming,
                pooled,
                occupancy_fraction,
                protected_fraction,
                pair_similarity,
                occupied.any(dim=-1, keepdim=True).to(row_features.dtype),
                consolidation_available.unsqueeze(-1).to(row_features.dtype),
                relation_means,
                relation_maxima,
            ),
            dim=-1,
        )
        action_logits = self.action_network(global_features)
        repeated_global = global_features[:, None, :].expand(-1, capacity, -1)
        eviction_features = torch.cat(
            (
                repeated_global,
                row_features,
                incoming_key_cosines.unsqueeze(-1),
                incoming_value_cosines.unsqueeze(-1),
            ),
            dim=-1,
        )
        eviction_scores = self.eviction_network(eviction_features).squeeze(-1) - self.eviction_similarity_prior * (
            incoming_key_cosines + incoming_value_cosines
        )
        left_metadata = row_features[:, :, None, -2:].expand(-1, -1, capacity, -1)
        right_metadata = row_features[:, None, :, -2:].expand(-1, capacity, -1, -1)
        pair_relations = torch.stack(
            (
                normalized_keys @ normalized_keys.transpose(-1, -2),
                normalized_values @ normalized_values.transpose(-1, -2),
            ),
            dim=-1,
        )
        incoming_key_cosines = (
            normalized_keys * normalized_incoming_key[:, None, :]
        ).sum(dim=-1)
        incoming_value_cosines = (
            normalized_values * normalized_incoming_value[:, None, :]
        ).sum(dim=-1)
        incoming_pair_relations = torch.cat(
            (
                incoming_key_cosines[:, :, None, None].expand(-1, -1, capacity, -1),
                incoming_key_cosines[:, None, :, None].expand(-1, capacity, -1, -1),
                incoming_value_cosines[:, :, None, None].expand(-1, -1, capacity, -1),
                incoming_value_cosines[:, None, :, None].expand(-1, capacity, -1, -1),
            ),
            dim=-1,
        )
        metadata_relations = torch.cat(
            (
                left_metadata + right_metadata,
                (left_metadata - right_metadata).abs(),
            ),
            dim=-1,
        )
        pair_features = torch.cat(
            (pair_relations, metadata_relations, incoming_pair_relations), dim=-1
        )
        pair_scores = self.pair_network(pair_features).squeeze(-1) + self.pair_similarity_prior * pair_relations[..., 0]
        if adapter is not None:
            action_logits = action_logits + adapter.action_network(global_features)
            eviction_scores = eviction_scores + adapter.eviction_network(
                eviction_features
            ).squeeze(-1)
            pair_scores = pair_scores + adapter.pair_network(pair_features).squeeze(-1)
        valid_evictions = occupied & ~protected
        valid_pairs = occupied[:, :, None] & occupied[:, None, :]
        diagonal = torch.eye(capacity, dtype=torch.bool, device=bank.keys.device)
        valid_pairs = valid_pairs & ~diagonal
        available_actions = torch.stack(
            (
                (counts < float(capacity)).squeeze(-1),
                valid_evictions.any(dim=-1),
                valid_pairs.any(dim=(-1, -2)) & consolidation_available,
                torch.ones(batch, dtype=torch.bool, device=bank.keys.device),
            ),
            dim=-1,
        )
        available_actions = available_actions & action_mask
        if not bool(available_actions.any(dim=-1).all()):
            raise ValueError("action_mask leaves no structurally available action")
        return CapacityPlannerOutput(
            action_logits=action_logits,
            eviction_scores=eviction_scores.masked_fill(~valid_evictions, -torch.inf),
            pair_scores=pair_scores.masked_fill(~valid_pairs, -torch.inf),
            valid_evictions=valid_evictions,
            valid_pairs=valid_pairs,
            available_actions=available_actions,
        )

    @torch.no_grad()
    def propose(
        self,
        bank: MemoryCandidates,
        incoming_key: torch.Tensor,
        incoming_value: torch.Tensor,
        protected: torch.Tensor,
        *,
        consolidation_available: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        adapter: OpaqueCapacityPlannerAdapter | None = None,
        explore: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> CapacityPlan | tuple[CapacityPlan, ...]:
        """Return a deterministic or exploratory safe action proposal.

        ``explore=True`` samples from the masked action/selector
        distributions for online verifier learning. Deployment defaults to
        deterministic argmax selection.
        """

        if bank.keys.shape[0] < 1:
            raise ValueError("capacity planner requires at least one bank")
        if not isinstance(explore, bool):
            raise TypeError("capacity planner explore flag must be boolean")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("capacity planner proposal temperature must be positive")
        output = self(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=consolidation_available,
            action_mask=action_mask,
            adapter=adapter,
        )
        plans: list[CapacityPlan] = []
        for batch_index in range(bank.keys.shape[0]):
            available = output.available_actions[batch_index]
            logits = output.action_logits[batch_index].masked_fill(~available, -torch.inf)
            if explore:
                action_distribution = torch.softmax(logits / temperature, dim=-1)
                action_index = int(
                    torch.multinomial(
                        action_distribution,
                        1,
                        generator=generator,
                    )
                )
            else:
                action_index = int(logits.argmax())
            eviction_index: int | None = None
            pair: tuple[int, int] | None = None
            score = logits[action_index]
            if action_index == 1:
                eviction_logits = output.eviction_scores[batch_index]
                if explore:
                    eviction_distribution = torch.softmax(
                        eviction_logits / temperature,
                        dim=-1,
                    )
                    eviction_index = int(
                        torch.multinomial(
                            eviction_distribution,
                            1,
                            generator=generator,
                        )
                    )
                else:
                    eviction_index = int(eviction_logits.argmax())
                score = output.eviction_scores[batch_index, eviction_index]
            elif action_index == 2:
                pair_scores = output.pair_scores[batch_index].masked_fill(
                    ~torch.triu(
                        torch.ones_like(output.valid_pairs[batch_index]),
                        diagonal=1,
                    ),
                    -torch.inf,
                )
                flat_scores = pair_scores.reshape(-1)
                if explore:
                    pair_distribution = torch.softmax(
                        flat_scores / temperature,
                        dim=-1,
                    )
                    flat_index = int(
                        torch.multinomial(
                            pair_distribution,
                            1,
                            generator=generator,
                        )
                    )
                else:
                    flat_index = int(flat_scores.argmax())
                first, second = divmod(flat_index, pair_scores.shape[1])
                pair = (first, second)
                score = pair_scores[first, second]
            plans.append(
                CapacityPlan(
                    action=ADMISSION_ACTIONS[action_index],
                    action_index=action_index,
                    eviction_index=eviction_index,
                    pair=pair,
                    score=score.detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)

    def adaptation_step(
        self,
        bank: MemoryCandidates,
        incoming_key: torch.Tensor,
        incoming_value: torch.Tensor,
        protected: torch.Tensor,
        plan: CapacityPlan,
        verifier_utility: float,
        *,
        consolidation_available: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        adapter: OpaqueCapacityPlannerAdapter | None = None,
        anchor_state: Mapping[str, torch.Tensor] | None = None,
        stability_coefficient: float = 0.0,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        """Update the policy from one verifier utility bit without replay.

        Utility is a scalar in ``[0, 1]`` supplied by an external behavior
        verifier.  The selected action, eviction row, or consolidation pair
        is reinforced when utility is high and suppressed when utility is
        low.  No memory row, controller parameter, or old example is stored
        or modified by this method.
        """

        if bank.keys.shape[0] != 1:
            raise ValueError("capacity planner adaptation requires one bank")
        if not isinstance(plan, CapacityPlan):
            raise TypeError("capacity planner adaptation plan is invalid")
        if stability_coefficient < 0.0 or not math.isfinite(stability_coefficient):
            raise ValueError("capacity planner stability coefficient is invalid")
        if anchor_state is not None:
            if not isinstance(anchor_state, Mapping):
                raise TypeError("capacity planner anchor state is invalid")
            parameter_names = {name for name, _ in self.named_parameters()}
            if set(anchor_state) != parameter_names:
                raise ValueError("capacity planner anchor state keys are invalid")
        if not math.isfinite(verifier_utility) or not 0.0 <= verifier_utility <= 1.0:
            raise ValueError("capacity planner verifier utility must lie in [0, 1]")
        if not 0 <= plan.action_index < len(ADMISSION_ACTIONS):
            raise ValueError("capacity planner plan action index is invalid")
        if plan.action != ADMISSION_ACTIONS[plan.action_index]:
            raise ValueError("capacity planner plan action does not match its index")
        output = self(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=consolidation_available,
            action_mask=action_mask,
            adapter=adapter,
        )
        if not bool(output.available_actions[0, plan.action_index]):
            raise ValueError("capacity planner plan action is unavailable")
        action_logits = output.action_logits[0].masked_fill(
            ~output.available_actions[0],
            -torch.inf,
        )
        action_probability = torch.softmax(action_logits, dim=-1)[
            plan.action_index
        ].clamp(1e-6, 1.0 - 1e-6)
        action_log_probability = torch.log(action_probability)
        advantage = verifier_utility - 0.5
        loss = -advantage * action_log_probability

        if plan.action_index == 1:
            if plan.eviction_index is None or plan.pair is not None:
                raise ValueError("capacity planner eviction plan fields are invalid")
            if not bool(output.valid_evictions[0, plan.eviction_index]):
                raise ValueError("capacity planner eviction plan row is invalid")
            selector_logits = output.eviction_scores[0]
            selector_log_probability = torch.log_softmax(selector_logits, dim=-1)[
                plan.eviction_index
            ]
            loss = loss - advantage * selector_log_probability
        elif plan.action_index == 2:
            if plan.pair is None or plan.eviction_index is not None:
                raise ValueError("capacity planner consolidation plan fields are invalid")
            first, second = plan.pair
            capacity = bank.keys.shape[1]
            if not 0 <= first < second < capacity:
                raise ValueError("capacity planner consolidation pair is invalid")
            valid_pairs = output.valid_pairs[0] & torch.triu(
                torch.ones_like(output.valid_pairs[0]),
                diagonal=1,
            )
            if not bool(valid_pairs[first, second]):
                raise ValueError("capacity planner consolidation pair is unavailable")
            selector_logits = output.pair_scores[0].masked_fill(
                ~valid_pairs,
                -torch.inf,
            ).reshape(-1)
            pair_index = first * capacity + second
            selector_log_probability = torch.log_softmax(selector_logits, dim=-1)[
                pair_index
            ]
            loss = loss - advantage * selector_log_probability
        elif plan.eviction_index is not None or plan.pair is not None:
            raise ValueError("capacity planner non-selector plan fields are invalid")

        if stability_coefficient > 0.0:
            if anchor_state is None:
                raise ValueError(
                    "capacity planner stability coefficient needs an anchor state"
                )
            stability_penalty = torch.zeros((), dtype=loss.dtype)
            for name, parameter in self.named_parameters():
                anchor = anchor_state[name].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                if anchor.shape != parameter.shape:
                    raise ValueError("capacity planner anchor tensor shape is invalid")
                stability_penalty = stability_penalty + (parameter - anchor).pow(2).mean()
            loss = loss + stability_coefficient * stability_penalty

        selected_optimizer = optimizer
        if selected_optimizer is None:
            optimized_parameters = (
                adapter.parameters() if adapter is not None else self.parameters()
            )
            selected_optimizer = torch.optim.SGD(
                optimized_parameters,
                lr=self.learning_rate,
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())


__all__ = [
    "ADMISSION_ACTIONS",
    "CAPACITY_PLANNER_ADAPTER_SCHEMA",
    "CAPACITY_PLANNER_SCHEMA",
    "CapacityPlan",
    "CapacityPlannerOutput",
    "OpaqueCapacityPlanner",
    "OpaqueCapacityPlannerAdapter",
]
