from __future__ import annotations

import torch


class DifferentiableBatchMemory:
    """Per-lifetime functional memory used to propagate delayed query loss.

    Unlike the disk store, this short training view keeps autograd history. At
    deployment its learned proposals are committed to PersistentMemory.
    """

    def __init__(self, batch: int, width: int, *, device: torch.device,
                 keys: torch.Tensor | None = None,
                 values: torch.Tensor | None = None,
                 strengths: torch.Tensor | None = None,
                 admissions: torch.Tensor | None = None):
        self.batch = batch
        self.width = width
        self.device = device
        self.keys = (torch.empty(batch, 0, width, device=device) if keys is None else keys)
        self.values = (torch.empty(batch, 0, width, device=device)
                       if values is None else values)
        self.strengths = (torch.empty(batch, 0, device=device)
                          if strengths is None else strengths)
        self.admissions = (torch.empty(batch, 0, device=device)
                           if admissions is None else admissions)

    @property
    def count(self) -> int:
        return self.keys.shape[1]

    def read(self, queries: torch.Tensor, top_k: int = 4,
             temperature: torch.Tensor | float = 1.0
             ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            return torch.zeros_like(queries), queries.new_zeros(queries.shape[0])
        keys = torch.nn.functional.normalize(self.keys, dim=-1)
        normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
        similarity = torch.einsum("bd,bnd->bn", normalized_queries, keys) * temperature
        similarity = similarity + self.strengths.clamp_min(1e-6).log()
        similarity = similarity + self.admissions.clamp_min(1e-6).log()
        selected = min(top_k, self.count)
        scores, indices = similarity.topk(selected, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        rows = torch.arange(self.batch, device=self.device).unsqueeze(1)
        values = self.values[rows, indices]
        admitted = self.admissions[rows, indices]
        return (weights.unsqueeze(-1) * values * admitted.unsqueeze(-1)).sum(dim=1), scores[:, 0]

    def append(self, keys: torch.Tensor, values: torch.Tensor,
               strengths: torch.Tensor,
               admissions: torch.Tensor | None = None) -> "DifferentiableBatchMemory":
        if keys.shape != (self.batch, self.width) or values.shape != keys.shape:
            raise ValueError("one key and value per lifetime required")
        if strengths.shape != (self.batch,):
            raise ValueError("one strength per lifetime required")
        if admissions is None:
            admissions = torch.ones_like(strengths)
        if admissions.shape != (self.batch,):
            raise ValueError("one admission decision per lifetime required")
        return DifferentiableBatchMemory(
            self.batch, self.width, device=self.device,
            keys=torch.cat((self.keys, keys.unsqueeze(1)), dim=1),
            values=torch.cat((self.values, values.unsqueeze(1)), dim=1),
            strengths=torch.cat((self.strengths, strengths.unsqueeze(1)), dim=1),
            admissions=torch.cat((self.admissions, admissions.unsqueeze(1)), dim=1),
        )

    def counterfactual(self, mode: str) -> "DifferentiableBatchMemory":
        """Return a non-mutating causal intervention for memory-use audits."""
        if mode == "intact":
            return self
        if mode == "empty":
            return DifferentiableBatchMemory(self.batch, self.width, device=self.device)
        if mode == "shuffled":
            # Preserve keys, sizes, and value distribution while assigning each
            # lifetime another lifetime's memories.
            return DifferentiableBatchMemory(
                self.batch, self.width, device=self.device, keys=self.keys,
                values=self.values.roll(1, dims=0), strengths=self.strengths,
                admissions=self.admissions)
        if mode == "garbage":
            # Deterministic pseudo-garbage avoids changing results with RNG state.
            key_index = torch.arange(self.keys.numel(), device=self.device,
                                     dtype=self.keys.dtype).reshape_as(self.keys)
            value_index = key_index + 0.5
            return DifferentiableBatchMemory(
                self.batch, self.width, device=self.device,
                keys=torch.sin(key_index * 1.618),
                values=torch.cos(value_index * 2.414),
                strengths=self.strengths, admissions=self.admissions)
        raise ValueError(f"unknown memory intervention {mode!r}")
