"""External raw-value regime-boundary policy.

This policy is deliberately outside the controller.  It compares opaque
current and incoming value banks through permutation-invariant structural
summaries and emits only a keep/replace decision.  A scalar verifier utility
is the sole learning signal; the policy never receives task labels,
reconstruction errors, or a hand-written regime identifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

REGIME_CHANGE_POLICY_SCHEMA = "neural-computer.opaque-regime-change-policy.v1"
GATED_RESIDUAL_REGIME_POLICY_SCHEMA = (
    "neural-computer.gated-residual-regime-change-policy.v1"
)
GATED_RESIDUAL_REGIME_POLICY_BANK_SCHEMA = (
    "neural-computer.gated-residual-regime-policy-bank.v1"
)


@dataclass(frozen=True)
class RegimeChangePolicyOutput:
    """Keep/replace logits emitted by the external detector."""

    logits: torch.Tensor


@dataclass(frozen=True)
class RegimeChangePlan:
    """One advisory keep/replace proposal."""

    replace: bool
    score: torch.Tensor


class OpaqueRegimeChangePolicy(nn.Module):
    """Learn whether an incoming opaque bank warrants a replacement."""

    schema = REGIME_CHANGE_POLICY_SCHEMA

    def __init__(
        self,
        *,
        value_width: int,
        hidden: int = 64,
        max_spectral_bins: int = 8,
        learning_rate: float = 1e-2,
    ) -> None:
        super().__init__()
        if min(value_width, hidden, max_spectral_bins) < 1:
            raise ValueError("regime policy dimensions must be positive")
        if learning_rate <= 0.0 or not math.isfinite(learning_rate):
            raise ValueError("regime policy learning rate must be positive")
        self.value_width = int(value_width)
        self.hidden = int(hidden)
        self.max_spectral_bins = int(max_spectral_bins)
        self.learning_rate = float(learning_rate)
        self.summary_width = 2 * self.max_spectral_bins + 2
        self.feature_width = 2 * self.summary_width + 7
        self.scorer = nn.Sequential(
            nn.Linear(self.feature_width, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 2),
        )

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "value_width": self.value_width,
            "hidden": self.hidden,
            "max_spectral_bins": self.max_spectral_bins,
            "learning_rate": self.learning_rate,
            "features": "opaque_spectral_cross_bank_structure_v1",
            "forbidden_features": (
                "task_labels_regime_ids_candidate_reconstruction_error_v1"
            ),
            "updates": "single_scalar_verifier_utility_without_replay_v1",
            "proposal": "keep_or_replace_v1",
        }

    def _validate_bank(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
    ) -> None:
        if values.ndim != 3 or values.shape[-1] != self.value_width:
            raise ValueError(
                "regime policy values must have shape "
                f"[batch, rows, {self.value_width}]"
            )
        if occupied.shape != values.shape[:2] or occupied.dtype != torch.bool:
            raise ValueError("regime policy occupancy has the wrong shape")
        if values.shape[0] < 1 or not bool(occupied.any(dim=-1).all()):
            raise ValueError("regime policy needs non-empty banks")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("regime policy values must be finite")

    def _summary(
        self,
        values: torch.Tensor,
        occupied: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_bank(values, occupied)
        masked = values * occupied.unsqueeze(-1).to(values.dtype)
        singular = torch.linalg.svdvals(masked)
        bins = min(self.max_spectral_bins, singular.shape[-1])
        singular = singular[..., :bins]
        energy = singular.square() / singular.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        cumulative = energy.cumsum(dim=-1)
        normalized = singular / singular[..., :1].clamp_min(1e-6)
        normalized_padded = torch.zeros(
            values.shape[0], self.max_spectral_bins,
            dtype=values.dtype,
            device=values.device,
        )
        cumulative_padded = torch.ones_like(normalized_padded)
        normalized_padded[:, :bins] = normalized
        cumulative_padded[:, :bins] = cumulative
        row_count = occupied.sum(dim=-1, keepdim=True).to(values.dtype)
        mean_norm = singular.square().sum(dim=-1, keepdim=True).sqrt() / row_count.sqrt().clamp_min(1.0)
        return torch.cat(
            (
                normalized_padded,
                cumulative_padded,
                row_count / float(self.value_width),
                mean_norm,
            ),
            dim=-1,
        )

    def _cross_summary(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> torch.Tensor:
        current_norm = current_values / current_values.square().sum(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        incoming_norm = incoming_values / incoming_values.square().sum(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        similarity = torch.abs(current_norm @ incoming_norm.transpose(-1, -2))
        mask = current_occupied.unsqueeze(-1) & incoming_occupied.unsqueeze(-2)
        count = mask.sum(dim=(-1, -2)).to(current_values.dtype)
        safe_count = count.to(torch.long).clamp_min(1)
        selected = similarity.masked_fill(~mask, float("inf")).reshape(
            current_values.shape[0], -1
        ).sort(dim=-1).values
        indices = torch.stack(
            (
                (safe_count - 1) // 4,
                (safe_count - 1) // 2,
                (safe_count - 1) * 3 // 4,
            ),
            dim=-1,
        )
        quantiles = selected.gather(1, indices)
        quantiles = torch.where(count.unsqueeze(-1) > 0, quantiles, torch.zeros_like(quantiles))
        weighted = similarity * mask.to(similarity.dtype)
        mean = weighted.sum(dim=(-1, -2)) / count.clamp_min(1.0)
        second = weighted.square().sum(dim=(-1, -2)) / count.clamp_min(1.0)
        std = (second - mean.square()).clamp_min(0.0).sqrt()
        maximum = weighted.masked_fill(~mask, 0.0).amax(dim=(-1, -2))
        current_count = current_occupied.sum(dim=-1).to(current_values.dtype)
        incoming_count = incoming_occupied.sum(dim=-1).to(current_values.dtype)
        return torch.cat(
            (
                mean.unsqueeze(-1),
                std.unsqueeze(-1),
                maximum.unsqueeze(-1),
                quantiles,
                (incoming_count / current_count.clamp_min(1.0)).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _features(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_bank(current_values, current_occupied)
        self._validate_bank(incoming_values, incoming_occupied)
        if current_values.shape[0] != incoming_values.shape[0]:
            raise ValueError("regime policy banks must have matching batches")
        return torch.cat(
            (
                self._summary(current_values, current_occupied),
                self._summary(incoming_values, incoming_occupied),
                self._cross_summary(
                    current_values,
                    current_occupied,
                    incoming_values,
                    incoming_occupied,
                ),
            ),
            dim=-1,
        )

    def forward(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> RegimeChangePolicyOutput:
        logits = self.scorer(
            self._features(
                current_values,
                current_occupied,
                incoming_values,
                incoming_occupied,
            )
        )
        return RegimeChangePolicyOutput(logits=logits)

    @torch.no_grad()
    def propose(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        *,
        explore: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> RegimeChangePlan | tuple[RegimeChangePlan, ...]:
        if not isinstance(explore, bool):
            raise TypeError("regime policy explore flag must be boolean")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("regime policy temperature must be positive")
        logits = self(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        ).logits
        plans: list[RegimeChangePlan] = []
        for row in logits:
            if explore:
                probabilities = torch.softmax(row / temperature, dim=-1)
                index = int(torch.multinomial(probabilities, 1, generator=generator))
            else:
                index = int(row.argmax())
            plans.append(
                RegimeChangePlan(
                    replace=index == 1,
                    score=row[index].detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)

    def adaptation_step(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        plan: RegimeChangePlan,
        verifier_utility: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        self._validate_bank(current_values, current_occupied)
        self._validate_bank(incoming_values, incoming_occupied)
        if current_values.shape[0] != 1 or incoming_values.shape[0] != 1:
            raise ValueError("regime policy adaptation needs one bank pair")
        if not isinstance(plan, RegimeChangePlan):
            raise TypeError("regime policy plan is invalid")
        if not math.isfinite(verifier_utility) or not 0.0 <= verifier_utility <= 1.0:
            raise ValueError("regime policy utility must lie in [0, 1]")
        logits = self(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        ).logits[0]
        index = int(plan.replace)
        log_probability = torch.log_softmax(logits, dim=-1)[index]
        loss = -(verifier_utility - 0.5) * log_probability
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())


class GatedResidualRegimeChangePolicy(nn.Module):
    """Grow a trainable external residual without changing a frozen policy.

    The base detector is an immutable capability snapshot.  A zero-initialized
    residual can learn a new boundary online, but deterministic inference uses
    it only when its preferred action has at least ``override_margin`` more
    logit evidence than the base detector's preferred action.  This keeps the
    old policy available as a protected fallback while the external residual
    grows.
    """

    schema = GATED_RESIDUAL_REGIME_POLICY_SCHEMA

    def __init__(
        self,
        base: OpaqueRegimeChangePolicy,
        *,
        override_margin: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, OpaqueRegimeChangePolicy):
            raise TypeError("gated residual base must be an opaque regime policy")
        if not math.isfinite(override_margin) or override_margin < 0.0:
            raise ValueError("gated residual override margin must be non-negative")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.override_margin = float(override_margin)
        self.residual = nn.Linear(base.feature_width, 2)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @property
    def value_width(self) -> int:
        return self.base.value_width

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "base_schema": self.base.schema,
            "value_width": self.value_width,
            "feature_contract": "base_opaque_spectral_cross_bank_structure_v1",
            "residual": "zero_initialized_external_linear_v1",
            "override": "base_fallback_margin_gate_v1",
            "override_margin": self.override_margin,
            "frozen_base": True,
            "updates": "single_scalar_verifier_utility_without_replay_v1",
        }

    def trainable_parameters(self):
        """Return only the new external residual parameters."""

        return self.residual.parameters()

    def _logits(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base_logits = self.base(
                current_values,
                current_occupied,
                incoming_values,
                incoming_occupied,
            ).logits
        features = self.base._features(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        )
        residual_logits = self.residual(features)
        return base_logits, residual_logits, base_logits + residual_logits

    def forward(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
    ) -> RegimeChangePolicyOutput:
        _base_logits, _residual_logits, combined = self._logits(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        )
        return RegimeChangePolicyOutput(logits=combined)

    def _gated_logits(
        self,
        base_logits: torch.Tensor,
        residual_logits: torch.Tensor,
    ) -> torch.Tensor:
        base_index = base_logits.argmax(dim=-1)
        residual_index = residual_logits.argmax(dim=-1)
        base_score = base_logits.gather(-1, base_index.unsqueeze(-1)).squeeze(-1)
        residual_score = residual_logits.gather(
            -1, residual_index.unsqueeze(-1)
        ).squeeze(-1)
        use_residual = (
            residual_score > self.override_margin
        ) & (residual_score >= base_score + self.override_margin)
        return torch.where(
            use_residual.unsqueeze(-1),
            residual_logits,
            base_logits,
        )

    @torch.no_grad()
    def propose(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        *,
        explore: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> RegimeChangePlan | tuple[RegimeChangePlan, ...]:
        if not isinstance(explore, bool):
            raise TypeError("gated residual explore flag must be boolean")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("gated residual temperature must be positive")
        base_logits, residual_logits, combined = self._logits(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        )
        logits = combined if explore else self._gated_logits(
            base_logits, residual_logits
        )
        plans: list[RegimeChangePlan] = []
        for row in logits:
            if explore:
                probabilities = torch.softmax(row / temperature, dim=-1)
                index = int(torch.multinomial(probabilities, 1, generator=generator))
            else:
                index = int(row.argmax())
            plans.append(
                RegimeChangePlan(
                    replace=index == 1,
                    score=row[index].detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)

    def adaptation_step(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        plan: RegimeChangePlan,
        verifier_utility: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        self.base._validate_bank(current_values, current_occupied)
        self.base._validate_bank(incoming_values, incoming_occupied)
        if current_values.shape[0] != 1 or incoming_values.shape[0] != 1:
            raise ValueError("gated residual adaptation needs one bank pair")
        if not isinstance(plan, RegimeChangePlan):
            raise TypeError("gated residual plan is invalid")
        if not math.isfinite(verifier_utility) or not 0.0 <= verifier_utility <= 1.0:
            raise ValueError("gated residual utility must lie in [0, 1]")
        _base_logits, _residual_logits, logits = self._logits(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        )
        index = int(plan.replace)
        log_probability = torch.log_softmax(logits[0], dim=-1)[index]
        loss = -(verifier_utility - 0.5) * log_probability
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.trainable_parameters(), lr=self.base.learning_rate
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())


class GatedResidualRegimePolicyBank(nn.Module):
    """Route isolated residuals with an opaque external binding context.

    Geometry-only regime summaries cannot distinguish two bindings that share
    the same relational structure.  This bank therefore accepts a learned,
    opaque context key from external state.  Keys select independent residual
    slots; the frozen base remains the fallback.  The bank does not interpret
    keys, assign semantic fields, or receive task labels.
    """

    schema = GATED_RESIDUAL_REGIME_POLICY_BANK_SCHEMA

    def __init__(
        self,
        base: OpaqueRegimeChangePolicy,
        *,
        context_width: int,
        override_margin: float = 0.0,
        route_threshold: float = 0.75,
    ) -> None:
        super().__init__()
        if not isinstance(base, OpaqueRegimeChangePolicy):
            raise TypeError("residual policy bank base is invalid")
        if context_width < 1:
            raise ValueError("residual policy bank context width is invalid")
        if not math.isfinite(override_margin) or override_margin < 0.0:
            raise ValueError("residual policy bank override margin is invalid")
        if not math.isfinite(route_threshold) or not -1.0 <= route_threshold <= 1.0:
            raise ValueError("residual policy bank route threshold is invalid")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.context_width = int(context_width)
        self.override_margin = float(override_margin)
        self.route_threshold = float(route_threshold)
        self.residual_slots = nn.ModuleList()
        self.slot_keys = nn.ParameterList()

    @property
    def slot_count(self) -> int:
        return len(self.residual_slots)

    def configuration(self) -> dict[str, int | float | str]:
        return {
            "schema": self.schema,
            "base_schema": self.base.schema,
            "value_width": self.base.value_width,
            "context_width": self.context_width,
            "slot_count": self.slot_count,
            "residual": "independent_zero_initialized_linear_slots_v1",
            "routing": "opaque_binding_cosine_key_v1",
            "override": "base_fallback_margin_gate_v1",
            "override_margin": self.override_margin,
            "route_threshold": self.route_threshold,
            "frozen_base": True,
            "updates": "single_scalar_verifier_utility_without_replay_v1",
        }

    def _validate_context(self, context: torch.Tensor) -> None:
        if context.ndim != 2 or context.shape[1] != self.context_width:
            raise ValueError(
                "residual policy bank context must have shape "
                f"[batch, {self.context_width}]"
            )
        if not bool(torch.isfinite(context).all()):
            raise ValueError("residual policy bank context must be finite")
        if not bool(context.square().sum(dim=-1).gt(1e-12).all()):
            raise ValueError("residual policy bank context cannot be zero")

    @torch.no_grad()
    def add_slot(self, context_key: torch.Tensor) -> int:
        """Append an isolated residual bound to one opaque context key."""

        if context_key.ndim != 1 or context_key.shape[0] != self.context_width:
            raise ValueError("residual policy bank slot key has the wrong shape")
        self._validate_context(context_key.unsqueeze(0))
        reference = next(self.base.parameters())
        residual = nn.Linear(
            self.base.feature_width + self.context_width,
            2,
        ).to(device=reference.device, dtype=reference.dtype)
        nn.init.zeros_(residual.weight)
        nn.init.zeros_(residual.bias)
        key = F.normalize(context_key.detach(), dim=0).to(reference)
        self.residual_slots.append(residual)
        self.slot_keys.append(nn.Parameter(key, requires_grad=False))
        return self.slot_count - 1

    def trainable_parameters(self, slot_index: int):
        if not 0 <= slot_index < self.slot_count:
            raise IndexError("residual policy bank slot is out of range")
        return self.residual_slots[slot_index].parameters()

    def route_scores(self, context: torch.Tensor) -> torch.Tensor:
        self._validate_context(context)
        if not self.slot_count:
            raise RuntimeError("residual policy bank has no slots")
        normalized = F.normalize(context, dim=-1)
        keys = torch.stack(tuple(self.slot_keys), dim=0).to(context)
        return normalized @ keys.transpose(0, 1)

    def route_slot(self, context: torch.Tensor) -> torch.Tensor:
        return self.route_scores(context).argmax(dim=-1)

    def _logits(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        context: torch.Tensor,
        slot_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not 0 <= slot_index < self.slot_count:
            raise IndexError("residual policy bank slot is out of range")
        self._validate_context(context)
        if context.shape[0] != current_values.shape[0]:
            raise ValueError("residual policy bank context batch does not match banks")
        with torch.no_grad():
            base_logits = self.base(
                current_values,
                current_occupied,
                incoming_values,
                incoming_occupied,
            ).logits
        features = self.base._features(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
        )
        residual_input = torch.cat((features, context), dim=-1)
        residual_logits = self.residual_slots[slot_index](residual_input)
        return base_logits, residual_logits, base_logits + residual_logits

    def forward(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        context: torch.Tensor,
        *,
        slot_index: int | None = None,
    ) -> RegimeChangePolicyOutput:
        selected = int(self.route_slot(context)[0]) if slot_index is None else slot_index
        _base, _residual, combined = self._logits(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
            context,
            selected,
        )
        return RegimeChangePolicyOutput(logits=combined)

    @torch.no_grad()
    def propose(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        context: torch.Tensor,
        *,
        explore: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> RegimeChangePlan | tuple[RegimeChangePlan, ...]:
        if not isinstance(explore, bool):
            raise TypeError("residual policy bank explore flag must be boolean")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("residual policy bank temperature is invalid")
        self._validate_context(context)
        route_scores = self.route_scores(context)
        plans: list[RegimeChangePlan] = []
        for batch_index in range(current_values.shape[0]):
            slot_index = int(route_scores[batch_index].argmax())
            base_logits, residual_logits, combined = self._logits(
                current_values[batch_index : batch_index + 1],
                current_occupied[batch_index : batch_index + 1],
                incoming_values[batch_index : batch_index + 1],
                incoming_occupied[batch_index : batch_index + 1],
                context[batch_index : batch_index + 1],
                slot_index,
            )
            if explore:
                logits = combined
            else:
                base_index = base_logits.argmax(dim=-1)
                residual_index = residual_logits.argmax(dim=-1)
                base_score = base_logits.gather(-1, base_index[:, None]).squeeze(-1)
                residual_score = residual_logits.gather(
                    -1, residual_index[:, None]
                ).squeeze(-1)
                use_residual = (
                    route_scores[batch_index, slot_index]
                    >= self.route_threshold
                ) & (residual_score > self.override_margin) & (
                    residual_score >= base_score + self.override_margin
                )
                logits = torch.where(
                    use_residual.view(1, 1), residual_logits, base_logits
                )
            row = logits[0]
            if explore:
                probabilities = torch.softmax(row / temperature, dim=-1)
                index = int(torch.multinomial(probabilities, 1, generator=generator))
            else:
                index = int(row.argmax())
            plans.append(
                RegimeChangePlan(
                    replace=index == 1,
                    score=row[index].detach().clone(),
                )
            )
        return plans[0] if len(plans) == 1 else tuple(plans)

    def adaptation_step(
        self,
        current_values: torch.Tensor,
        current_occupied: torch.Tensor,
        incoming_values: torch.Tensor,
        incoming_occupied: torch.Tensor,
        context: torch.Tensor,
        slot_index: int,
        plan: RegimeChangePlan,
        verifier_utility: float,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        self.base._validate_bank(current_values, current_occupied)
        self.base._validate_bank(incoming_values, incoming_occupied)
        self._validate_context(context)
        if current_values.shape[0] != 1 or incoming_values.shape[0] != 1:
            raise ValueError("residual policy bank adaptation needs one bank pair")
        if not isinstance(plan, RegimeChangePlan):
            raise TypeError("residual policy bank plan is invalid")
        if not math.isfinite(verifier_utility) or not 0.0 <= verifier_utility <= 1.0:
            raise ValueError("residual policy bank utility must lie in [0, 1]")
        _base, _residual, logits = self._logits(
            current_values,
            current_occupied,
            incoming_values,
            incoming_occupied,
            context,
            slot_index,
        )
        index = int(plan.replace)
        loss = -(verifier_utility - 0.5) * torch.log_softmax(logits[0], dim=-1)[index]
        selected_optimizer = optimizer
        if selected_optimizer is None:
            selected_optimizer = torch.optim.SGD(
                self.trainable_parameters(slot_index), lr=self.base.learning_rate
            )
        selected_optimizer.zero_grad()
        loss.backward()
        selected_optimizer.step()
        return float(loss.detach())


__all__ = [
    "GATED_RESIDUAL_REGIME_POLICY_BANK_SCHEMA",
    "GATED_RESIDUAL_REGIME_POLICY_SCHEMA",
    "REGIME_CHANGE_POLICY_SCHEMA",
    "GatedResidualRegimeChangePolicy",
    "GatedResidualRegimePolicyBank",
    "OpaqueRegimeChangePolicy",
    "RegimeChangePlan",
    "RegimeChangePolicyOutput",
]
