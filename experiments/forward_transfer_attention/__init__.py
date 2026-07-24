"""Few-shot forward transfer through latent external memory."""

from .environment import (AttentionTransferLifetime, generate_attention_lifetime,
                          generate_shape_attention_lifetime,
                          generate_temporal_attention_lifetime)

__all__ = ["AttentionTransferLifetime", "generate_attention_lifetime",
           "generate_shape_attention_lifetime", "generate_temporal_attention_lifetime"]
