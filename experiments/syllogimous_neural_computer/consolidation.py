from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import PublicEpisode, collate_episodes

from .memory import PersistentMemory
from .model import NeuralComputerAgent


@dataclass(frozen=True)
class ReplayScore:
    correct: int
    total: int
    loss: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)


@dataclass(frozen=True)
class ConsolidationProposal:
    """A learned two-row-to-one-row rewrite in latent memory space."""

    first: int
    second: int
    key: torch.Tensor
    value: torch.Tensor
    priority: torch.Tensor
    operation_logits: torch.Tensor
    log_probability: torch.Tensor | None = None


@dataclass(frozen=True)
class ConsolidationResult:
    committed: bool
    memory: PersistentMemory
    before: ReplayScore
    after: ReplayScore
    rows_saved: int
    reward: float
    provenance: tuple[torch.Tensor, torch.Tensor] | None


class LearnedConsolidator(nn.Module):
    """Proposes generic latent rewrites without seeing labels or task state.

    The network sees only entries created by the controller. Its three operation
    logits mean merge, keep-first/drop-second, and keep-second/drop-first. These
    names describe mechanics, not the semantic content of an entry.
    """

    def __init__(self, width: int, hidden: int = 128):
        super().__init__()
        self.width = width
        pair_width = width * 4 + 4
        self.pair_score = nn.Sequential(nn.Linear(pair_width, hidden), nn.GELU(),
                                        nn.Linear(hidden, 1))
        self.operation = nn.Sequential(nn.Linear(pair_width, hidden), nn.GELU(),
                                       nn.Linear(hidden, 3))
        self.merge_gate = nn.Sequential(nn.Linear(pair_width, hidden), nn.GELU(),
                                        nn.Linear(hidden, width * 2))
        self.stop_head = nn.Sequential(nn.Linear(width * 2 + 3, hidden), nn.GELU(),
                                       nn.Linear(hidden, 1))
        self.rewrite_head = nn.Sequential(nn.Linear(width * 6 + 9, hidden), nn.GELU(),
                                          nn.Linear(hidden, 1))

    def _features(self, memory: PersistentMemory, indices: torch.Tensor) -> torch.Tensor:
        keys = memory.keys[indices]
        values = memory.values[indices]
        usage = memory.usage[indices, None]
        relative_age = (memory.clock - memory.age[indices]).to(keys.dtype)[:, None]
        relative_age = relative_age / relative_age.max().clamp_min(1.0)
        return torch.cat((keys, values, usage, relative_age), dim=-1)

    def _propose(self, memory: PersistentMemory, *, stochastic: bool,
                 merge_std: float = 0.25) -> ConsolidationProposal | None:
        indices = memory.valid.nonzero(as_tuple=False).squeeze(1)
        if indices.numel() < 2:
            return None
        features = self._features(memory, indices)
        left, right = torch.triu_indices(indices.numel(), indices.numel(), offset=1,
                                         device=indices.device)
        pair_features = torch.cat((features[left], features[right]), dim=-1)
        pair_logits = self.pair_score(pair_features).squeeze(-1)
        pair_distribution = torch.distributions.Categorical(logits=pair_logits)
        selected = pair_distribution.sample() if stochastic else pair_logits.argmax()
        first, second = indices[left[selected]], indices[right[selected]]
        pair = pair_features[selected]
        operation_logits = self.operation(pair)
        operation_distribution = torch.distributions.Categorical(logits=operation_logits)
        operation_tensor = (operation_distribution.sample() if stochastic else
                            operation_logits.argmax())
        operation = int(operation_tensor)
        log_probability = (pair_distribution.log_prob(selected) +
                           operation_distribution.log_prob(operation_tensor))
        if operation == 0:
            gate_means = self.merge_gate(pair)
            if stochastic:
                gate_distribution = torch.distributions.Normal(
                    gate_means, torch.full_like(gate_means, merge_std))
                gate_logits = gate_distribution.rsample()
                log_probability = log_probability + gate_distribution.log_prob(
                    gate_logits.detach()).mean()
            else:
                gate_logits = gate_means
            gates = torch.sigmoid(gate_logits)
            key_gate, value_gate = gates[:self.width], gates[self.width:]
            key = key_gate * memory.keys[first] + (1.0 - key_gate) * memory.keys[second]
            value = value_gate * memory.values[first] + (1.0 - value_gate) * memory.values[second]
            priority = memory.usage[torch.stack((first, second))].max()
        else:
            chosen = first if operation == 1 else second
            key, value, priority = (memory.keys[chosen], memory.values[chosen],
                                    memory.usage[chosen])
        return ConsolidationProposal(int(first), int(second), key, value, priority,
                                     operation_logits, log_probability)

    def propose(self, memory: PersistentMemory) -> ConsolidationProposal | None:
        return self._propose(memory, stochastic=False)

    def sample(self, memory: PersistentMemory) -> ConsolidationProposal | None:
        """Sample a rewrite and retain its log probability for policy gradients."""
        return self._propose(memory, stochastic=True)

    def stop_logit(self, memory: PersistentMemory) -> torch.Tensor:
        """Estimate whether further rewriting is worse than retaining the store."""
        indices = memory.valid.nonzero(as_tuple=False).squeeze(1)
        if not indices.numel():
            return self.stop_head[0].weight.new_tensor(20.0)
        features = self._features(memory, indices).mean(dim=0)
        count = features.new_tensor([float(indices.numel())]).log1p()
        return self.stop_head(torch.cat((features, count), dim=0)).squeeze(-1)

    def rewrite_logit(self, memory: PersistentMemory,
                      proposal: ConsolidationProposal) -> torch.Tensor:
        """Predict whether this particular latent rewrite is preferable to STOP."""
        indices = torch.tensor([proposal.first, proposal.second],
                               device=memory.keys.device)
        pair = self._features(memory, indices).reshape(-1)
        proposal_features = torch.cat((proposal.key, proposal.value,
                                       proposal.priority.reshape(1),
                                       proposal.operation_logits,
                                       pair.new_tensor([float(memory.count)]).log1p()))
        return self.rewrite_head(torch.cat((pair, proposal_features))).squeeze(-1)


def apply_proposal(memory: PersistentMemory,
                   proposal: ConsolidationProposal) -> PersistentMemory:
    """Apply to a clone; the supplied memory is an immutable transaction base."""
    candidate = memory.clone()
    if proposal.first == proposal.second:
        raise ValueError("a consolidation proposal must address two distinct rows")
    if not (candidate.valid[proposal.first] and candidate.valid[proposal.second]):
        raise ValueError("a consolidation proposal must address valid rows")
    candidate.keys[proposal.first].copy_(proposal.key.detach())
    candidate.values[proposal.first].copy_(proposal.value.detach())
    candidate.usage[proposal.first] = proposal.priority.detach()
    candidate.valid[proposal.second] = False
    return candidate


@torch.no_grad()
def score_sensory_replay(model: NeuralComputerAgent, memory: PersistentMemory,
                         episodes: Sequence[PublicEpisode], device: torch.device,
                         *, batch_size: int = 64) -> ReplayScore:
    """Verify a memory using raw public observations and public action outcomes."""
    was_training = model.training
    model.eval()
    correct, total, loss = 0, 0, 0.0
    for offset in range(0, len(episodes), batch_size):
        batch = collate_episodes(episodes[offset:offset + batch_size])
        targets = batch["actions"][:, 0].to(device)
        output = model(batch["frames"].to(device), batch["pcm"].to(device),
                       batch["mask"].to(device), memory)
        logits = output.answer_logits[:, -1]
        correct += int((logits.argmax(-1) == targets).sum())
        total += targets.numel()
        loss += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
    model.train(was_training)
    return ReplayScore(correct, total, loss / max(1, total))


def transactional_consolidate(
        memory: PersistentMemory, proposal: ConsolidationProposal,
        replay_verifier: Callable[[PersistentMemory], ReplayScore],
        heldout_verifier: Callable[[PersistentMemory], ReplayScore] | None = None,
        *, accuracy_tolerance: float = 0.0, storage_reward: float = 0.001,
        error_penalty: float = 1.0,
        loss_tolerance: float | None = None) -> ConsolidationResult:
    """Commit a smaller memory only if replay and held-out accuracy are preserved."""
    verifiers = [replay_verifier]
    if heldout_verifier is not None:
        verifiers.append(heldout_verifier)
    return transactional_consolidate_many(
        memory, proposal, verifiers, accuracy_tolerance=accuracy_tolerance,
        storage_reward=storage_reward, error_penalty=error_penalty,
        loss_tolerance=loss_tolerance)


def transactional_consolidate_many(
        memory: PersistentMemory, proposal: ConsolidationProposal,
        verifiers: Sequence[Callable[[PersistentMemory], ReplayScore]], *,
        accuracy_tolerance: float = 0.0, storage_reward: float = 0.001,
        error_penalty: float = 1.0,
        loss_tolerance: float | None = None) -> ConsolidationResult:
    """Require a proposal to pass every independently scored rehearsal group."""
    if not verifiers:
        raise ValueError("at least one verifier is required")
    before_scores = [check(memory) for check in verifiers]
    candidate = apply_proposal(memory, proposal)
    after_scores = [check(candidate) for check in verifiers]
    drops = [max(0.0, before.accuracy - after.accuracy)
             for before, after in zip(before_scores, after_scores)]
    accuracy_drop = max(drops)
    accepted = all(drop <= accuracy_tolerance for drop in drops)
    if loss_tolerance is not None:
        accepted = accepted and all(
            after.loss <= before.loss + loss_tolerance
            for before, after in zip(before_scores, after_scores))
    rows_saved = memory.count - candidate.count
    reward = storage_reward * rows_saved - error_penalty * accuracy_drop
    provenance = (memory.keys[[proposal.first, proposal.second]].detach().cpu().clone(),
                  memory.values[[proposal.first, proposal.second]].detach().cpu().clone())
    return ConsolidationResult(accepted, candidate if accepted else memory,
                               before_scores[0], after_scores[0],
                               rows_saved if accepted else 0, reward, provenance)
