"""Canonical Brain Workshop pressure tests for the production runtime."""

from .environment import BrainWorkshopEventEncoder, NBackVerifier, NBackVerifierStep
from .runner import CanonicalBrainWorkshopAgent, CanonicalRollout
from .trainer import (
    RewardOnlyUpdate,
    audit_retention,
    evaluate_policy,
    freeze_shared_path,
    train_adaptive_relation_capability,
    train_existing_adaptive_relation_capability,
    train_isolated_relation_capability,
    train_reward_only,
)

__all__ = [
    "BrainWorkshopEventEncoder",
    "CanonicalBrainWorkshopAgent",
    "CanonicalRollout",
    "NBackVerifier",
    "NBackVerifierStep",
    "RewardOnlyUpdate",
    "audit_retention",
    "evaluate_policy",
    "freeze_shared_path",
    "train_adaptive_relation_capability",
    "train_existing_adaptive_relation_capability",
    "train_isolated_relation_capability",
    "train_reward_only",
]
