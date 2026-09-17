from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLTokenJointConfig:
    """Configuration for joint VLA + RL-token training metadata."""

    enable: bool = False
    weight: float = 1.0
    num_rl_tokens: int = 1
    nhead: int = 8
    num_enc_layers: int = 3
    num_dec_layers: int = 3
    ff_dim: int | None = 4096
    token_pool_size: int = 64
    image_only: bool = True
    lr_multiplier: float = 1.0
    gradient_checkpointing: bool = False
