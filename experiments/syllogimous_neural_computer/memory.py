from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PersistentMemory:
    """A growing, serializable key-value memory outside the network weights.

    This object contains no task labels or game state. The controller supplies
    every key, value, and write strength. Rows are append-only at this layer:
    consolidation may later write a better abstraction, but raw experience is
    not silently destroyed. Tensor storage grows in chunks and can be saved to
    disk independently of model weights.
    """

    keys: torch.Tensor
    values: torch.Tensor
    usage: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor
    access_count: torch.Tensor
    success_count: torch.Tensor
    failure_count: torch.Tensor
    volatility: torch.Tensor
    clock: int = 0
    growth_chunk: int = 1024

    @classmethod
    def empty(cls, capacity: int, width: int, *, device: torch.device | str = "cpu",
              dtype: torch.dtype = torch.float32,
              growth_chunk: int = 1024) -> "PersistentMemory":
        if capacity < 1 or width < 1:
            raise ValueError("capacity and width must be positive")
        return cls(
            torch.zeros(capacity, width, device=device, dtype=dtype),
            torch.zeros(capacity, width, device=device, dtype=dtype),
            torch.zeros(capacity, device=device, dtype=dtype),
            torch.zeros(capacity, device=device, dtype=torch.long),
            torch.zeros(capacity, device=device, dtype=torch.bool),
            torch.zeros(capacity, device=device, dtype=torch.long),
            torch.zeros(capacity, device=device, dtype=torch.long),
            torch.zeros(capacity, device=device, dtype=torch.long),
            torch.ones(capacity, device=device, dtype=dtype),
            0, growth_chunk,
        )

    @property
    def capacity(self) -> int:
        return self.keys.shape[0]

    @property
    def width(self) -> int:
        return self.keys.shape[1]

    @property
    def count(self) -> int:
        return int(self.valid.sum())

    def to(self, device: torch.device | str) -> "PersistentMemory":
        return PersistentMemory(self.keys.to(device), self.values.to(device),
                                self.usage.to(device), self.age.to(device),
                                self.valid.to(device), self.access_count.to(device),
                                self.success_count.to(device),
                                self.failure_count.to(device),
                                self.volatility.to(device),
                                self.clock, self.growth_chunk)

    def clone(self) -> "PersistentMemory":
        return PersistentMemory(self.keys.clone(), self.values.clone(), self.usage.clone(),
                                self.age.clone(), self.valid.clone(),
                                self.access_count.clone(),
                                self.success_count.clone(),
                                self.failure_count.clone(),
                                self.volatility.clone(),
                                self.clock, self.growth_chunk)

    def select(self, indices: torch.Tensor | list[int]) -> "PersistentMemory":
        """Copy selected valid rows into a compact active-memory store."""
        selected = torch.as_tensor(indices, device=self.keys.device, dtype=torch.long)
        if selected.ndim != 1:
            raise ValueError("memory indices must be one-dimensional")
        if selected.numel() and (not bool(self.valid[selected].all())):
            raise ValueError("active memory can contain only valid long-term rows")
        active = PersistentMemory.empty(
            max(1, int(selected.numel())), self.width, device=self.keys.device,
            dtype=self.keys.dtype, growth_chunk=self.growth_chunk)
        if selected.numel():
            count = int(selected.numel())
            active.keys[:count].copy_(self.keys[selected])
            active.values[:count].copy_(self.values[selected])
            active.usage[:count].copy_(self.usage[selected])
            active.age[:count].copy_(self.age[selected])
            active.access_count[:count].copy_(self.access_count[selected])
            active.success_count[:count].copy_(self.success_count[selected])
            active.failure_count[:count].copy_(self.failure_count[selected])
            active.volatility[:count].copy_(self.volatility[selected])
            active.valid[:count] = True
        active.clock = self.clock
        return active

    def _grow(self) -> None:
        amount = max(1, self.growth_chunk)
        self.keys = torch.cat((self.keys, self.keys.new_zeros(amount, self.width)))
        self.values = torch.cat((self.values, self.values.new_zeros(amount, self.width)))
        self.usage = torch.cat((self.usage, self.usage.new_zeros(amount)))
        self.age = torch.cat((self.age, self.age.new_zeros(amount)))
        self.valid = torch.cat((self.valid, self.valid.new_zeros(amount)))
        self.access_count = torch.cat((
            self.access_count, self.access_count.new_zeros(amount)))
        self.success_count = torch.cat((
            self.success_count, self.success_count.new_zeros(amount)))
        self.failure_count = torch.cat((
            self.failure_count, self.failure_count.new_zeros(amount)))
        self.volatility = torch.cat((
            self.volatility, self.volatility.new_ones(amount)))

    @torch.no_grad()
    def record_outcomes(
            self, queries: torch.Tensor, outcomes: torch.Tensor, *,
            update_volatility: bool = False,
            success_protection_rate: float = 0.1,
            failure_thaw_rate: float = 0.2,
            stale_thaw_rate: float = 0.005,
            usage_prior_scale: torch.Tensor | float = 1.0) -> None:
        """Attribute verified binary outcomes to content-addressed top-1 rows."""
        if queries.ndim != 2 or queries.shape[1] != self.width:
            raise ValueError("queries must have shape [batch, memory width]")
        if outcomes.shape != (queries.shape[0],):
            raise ValueError("outcomes must have one value per query")
        if self.count == 0:
            return
        for name, rate in (
                ("success_protection_rate", success_protection_rate),
                ("failure_thaw_rate", failure_thaw_rate),
                ("stale_thaw_rate", stale_thaw_rate)):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if (
                isinstance(usage_prior_scale, (int, float))
                and usage_prior_scale < 0.0):
            raise ValueError("usage_prior_scale must be nonnegative")
        indices = self.valid.nonzero(as_tuple=False).squeeze(1)
        keys = torch.nn.functional.normalize(self.keys[indices], dim=-1)
        normalized_queries = torch.nn.functional.normalize(
            queries, dim=-1)
        similarity = normalized_queries @ keys.T
        scale = torch.as_tensor(
            usage_prior_scale, device=similarity.device,
            dtype=similarity.dtype)
        if scale.ndim == 1:
            if scale.shape[0] != queries.shape[0]:
                raise ValueError(
                    "per-query usage_prior_scale must match query batch")
            scale = scale.unsqueeze(-1)
        similarity = similarity + scale * (
            self.usage[indices].clamp_min(1e-6).log().unsqueeze(0))
        chosen = indices[similarity.argmax(-1)]
        successes = (outcomes > 0.5).to(self.success_count.dtype)
        failures = 1 - successes
        self.success_count.scatter_add_(0, chosen, successes)
        self.failure_count.scatter_add_(0, chosen, failures)
        if update_volatility:
            # Every outcome interval makes untouched rows slightly easier to
            # rewrite. Verified success protects the row; verified failure
            # thaws it. Access alone never grants protection.
            current_volatility = self.volatility[indices]
            self.volatility[indices] = current_volatility + (
                stale_thaw_rate * (1.0 - current_volatility))
            unique, inverse = chosen.unique(return_inverse=True)
            success_totals = torch.zeros(
                unique.shape[0], device=chosen.device,
                dtype=self.volatility.dtype)
            failure_totals = torch.zeros_like(success_totals)
            success_totals.scatter_add_(
                0, inverse, successes.to(success_totals.dtype))
            failure_totals.scatter_add_(
                0, inverse, failures.to(failure_totals.dtype))
            protected = self.volatility[unique] * (
                (1.0 - success_protection_rate) ** success_totals)
            self.volatility[unique] = 1.0 - (
                (1.0 - protected)
                * ((1.0 - failure_thaw_rate) ** failure_totals))
            self.volatility.clamp_(0.0, 1.0)

    def read(self, queries: torch.Tensor, top_k: int = 4,
             temperature: torch.Tensor | float = 1.0,
             confidence_mode: str = "ranked",
             record_access: bool = False,
             usage_prior_scale: torch.Tensor | float = 1.0,
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Content-addressed sparse reads; returns values and confidence."""
        if queries.ndim != 2 or queries.shape[1] != self.width:
            raise ValueError("queries must have shape [batch, memory width]")
        if self.count == 0:
            return torch.zeros_like(queries), queries.new_zeros(queries.shape[0])
        indices = self.valid.nonzero(as_tuple=False).squeeze(1)
        keys = torch.nn.functional.normalize(self.keys[indices], dim=-1)
        queries = torch.nn.functional.normalize(queries, dim=-1)
        cosine_similarity = queries @ keys.T
        similarity = cosine_similarity * temperature
        # Learned write strength is both an admission decision and a soft
        # retrieval prior. This lets delayed reward train the write gate in the
        # differentiable lifetime implementation used by the benchmark.
        scale = torch.as_tensor(
            usage_prior_scale, device=similarity.device,
            dtype=similarity.dtype)
        if scale.ndim == 1:
            if scale.shape[0] != queries.shape[0]:
                raise ValueError(
                    "per-query usage_prior_scale must match query batch")
            scale = scale.unsqueeze(-1)
        similarity = similarity + (
            self.usage[indices].clamp_min(1e-6).log().unsqueeze(0)
            * scale)
        selected = min(top_k, indices.numel())
        scores, local_indices = similarity.topk(selected, dim=-1)
        if record_access:
            with torch.no_grad():
                chosen = indices[local_indices[:, 0]]
                increments = torch.ones_like(
                    chosen, dtype=self.access_count.dtype)
                self.access_count.scatter_add_(0, chosen, increments)
        weights = torch.softmax(scores, dim=-1)
        values = self.values[indices[local_indices]]
        read = (weights.unsqueeze(-1) * values).sum(dim=1)
        if confidence_mode == "ranked":
            confidence = scores[:, 0]
        elif confidence_mode == "cosine":
            confidence = torch.gather(
                cosine_similarity, 1, local_indices[:, :1]).squeeze(1)
        else:
            raise ValueError("unsupported confidence mode")
        return read, confidence

    def read_with_receipt(
            self, queries: torch.Tensor, top_k: int = 4,
            temperature: torch.Tensor | float = 1.0,
            confidence_mode: str = "ranked",
            record_access: bool = False,
            usage_prior_scale: torch.Tensor | float = 1.0,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read values and return the physical top-1 row for each query.

        The receipt is the causal provenance of the read.  It is deliberately
        only a physical index, not a semantic task label.  This lets a
        verifier attribute a later outcome to the row that actually supplied
        the value even when write-strength priors redirect retrieval.
        Empty memories return ``-1`` receipts; callers must not attribute an
        outcome to those rows.
        """
        if queries.ndim != 2 or queries.shape[1] != self.width:
            raise ValueError("queries must have shape [batch, memory width]")
        if self.count == 0:
            return (
                torch.zeros_like(queries),
                queries.new_zeros(queries.shape[0]),
                torch.full(
                    (queries.shape[0],), -1, device=queries.device,
                    dtype=torch.long),
            )
        indices = self.valid.nonzero(as_tuple=False).squeeze(1)
        keys = torch.nn.functional.normalize(self.keys[indices], dim=-1)
        normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
        cosine_similarity = normalized_queries @ keys.T
        similarity = cosine_similarity * temperature
        scale = torch.as_tensor(
            usage_prior_scale, device=similarity.device,
            dtype=similarity.dtype)
        if scale.ndim == 1:
            if scale.shape[0] != queries.shape[0]:
                raise ValueError(
                    "per-query usage_prior_scale must match query batch")
            scale = scale.unsqueeze(-1)
        similarity = similarity + (
            self.usage[indices].clamp_min(1e-6).log().unsqueeze(0)
            * scale)
        selected = min(top_k, indices.numel())
        scores, local_indices = similarity.topk(selected, dim=-1)
        receipts = indices[local_indices[:, 0]]
        if record_access:
            with torch.no_grad():
                increments = torch.ones_like(
                    receipts, dtype=self.access_count.dtype)
                self.access_count.scatter_add_(0, receipts, increments)
        weights = torch.softmax(scores, dim=-1)
        values = self.values[indices[local_indices]]
        read = (weights.unsqueeze(-1) * values).sum(dim=1)
        if confidence_mode == "ranked":
            confidence = scores[:, 0]
        elif confidence_mode == "cosine":
            confidence = torch.gather(
                cosine_similarity, 1, local_indices[:, :1]).squeeze(1)
        else:
            raise ValueError("unsupported confidence mode")
        return read, confidence, receipts

    @torch.no_grad()
    def record_outcomes_from_receipts(
            self, receipts: torch.Tensor, outcomes: torch.Tensor, *,
            update_volatility: bool = False,
            success_protection_rate: float = 0.1,
            failure_thaw_rate: float = 0.2,
            stale_thaw_rate: float = 0.005,
            ) -> None:
        """Attribute verifier outcomes to already-resolved physical rows.

        This is the safe counterpart to :meth:`record_outcomes`.  The latter
        resolves a fresh top-1 read and can therefore attribute an outcome to
        a different row when admission strength is part of retrieval.  A
        receipt freezes that causal choice at read time.
        """
        if receipts.ndim != 1 or outcomes.shape != receipts.shape:
            raise ValueError("receipts and outcomes must be one-dimensional")
        if receipts.numel() and (
                bool((receipts < 0).any())
                or bool((receipts >= self.capacity).any())
                or not bool(self.valid[receipts].all())):
            raise ValueError("receipts must identify valid physical rows")
        for name, rate in (
                ("success_protection_rate", success_protection_rate),
                ("failure_thaw_rate", failure_thaw_rate),
                ("stale_thaw_rate", stale_thaw_rate)):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not receipts.numel():
            return
        successes = (outcomes > 0.5).to(self.success_count.dtype)
        failures = 1 - successes
        self.success_count.scatter_add_(0, receipts, successes)
        self.failure_count.scatter_add_(0, receipts, failures)
        if not update_volatility:
            return
        valid = self.valid
        self.volatility[valid] += stale_thaw_rate * (
            1.0 - self.volatility[valid])
        unique, inverse = receipts.unique(return_inverse=True)
        success_totals = torch.zeros(
            unique.shape[0], device=receipts.device,
            dtype=self.volatility.dtype)
        failure_totals = torch.zeros_like(success_totals)
        success_totals.scatter_add_(
            0, inverse, successes.to(success_totals.dtype))
        failure_totals.scatter_add_(
            0, inverse, failures.to(failure_totals.dtype))
        protected = self.volatility[unique] * (
            (1.0 - success_protection_rate) ** success_totals)
        self.volatility[unique] = 1.0 - (
            (1.0 - protected)
            * ((1.0 - failure_thaw_rate) ** failure_totals))
        self.volatility.clamp_(0.0, 1.0)

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor, strengths: torch.Tensor,
              *, threshold: float = 0.5) -> int:
        """Commit sampled/thresholded controller writes between episodes."""
        if keys.shape != values.shape or keys.ndim != 2 or keys.shape[1] != self.width:
            raise ValueError("keys and values must share shape [writes, memory width]")
        if strengths.shape != (keys.shape[0],):
            raise ValueError("strengths must have one value per write")
        committed = 0
        for key, value, strength in zip(keys, values, strengths):
            if float(strength) < threshold:
                continue
            empty = (~self.valid).nonzero(as_tuple=False)
            if not empty.numel():
                self._grow()
                empty = (~self.valid).nonzero(as_tuple=False)
            slot = int(empty[0, 0])
            self.clock += 1
            self.keys[slot].copy_(key.detach())
            self.values[slot].copy_(value.detach())
            self.usage[slot] = strength.detach()
            self.age[slot] = self.clock
            self.access_count[slot] = 0
            self.success_count[slot] = 0
            self.failure_count[slot] = 0
            self.volatility[slot] = 1.0
            self.valid[slot] = True
            committed += 1
        return committed

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "schema": "syllogimous-neural-computer-memory-v4",
            "keys": self.keys.detach().cpu(), "values": self.values.detach().cpu(),
            "usage": self.usage.detach().cpu(), "age": self.age.detach().cpu(),
            "valid": self.valid.detach().cpu(),
            "access_count": self.access_count.detach().cpu(),
            "success_count": self.success_count.detach().cpu(),
            "failure_count": self.failure_count.detach().cpu(),
            "volatility": self.volatility.detach().cpu(),
            "clock": self.clock,
            "growth_chunk": self.growth_chunk,
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, device: torch.device | str = "cpu") -> "PersistentMemory":
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("schema") not in {
                "syllogimous-neural-computer-memory-v1",
                "syllogimous-neural-computer-memory-v2",
                "syllogimous-neural-computer-memory-v3",
                "syllogimous-neural-computer-memory-v4"}:
            raise ValueError("unsupported persistent-memory schema")
        access_count = payload.get(
            "access_count", torch.zeros_like(payload["age"]))
        success_count = payload.get(
            "success_count", torch.zeros_like(payload["age"]))
        failure_count = payload.get(
            "failure_count", torch.zeros_like(payload["age"]))
        volatility = payload.get(
            "volatility", torch.ones_like(payload["usage"]))
        return cls(payload["keys"], payload["values"], payload["usage"],
                   payload["age"], payload["valid"], access_count,
                   success_count, failure_count, volatility,
                   int(payload["clock"]),
                   int(payload.get("growth_chunk", 1024)))
