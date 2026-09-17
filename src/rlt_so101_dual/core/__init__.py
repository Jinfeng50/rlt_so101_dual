from __future__ import annotations

from rlt_so101_dual.core.interfaces import ChunkTransition, Observation, VLAOutput
from rlt_so101_dual.core.config import RLTConfig, OfflineRLConfig
from rlt_so101_dual.core.vla_adapter import VLAAdapter, DummyVLAAdapter
from rlt_so101_dual.core.rl_token import RLTokenModule
from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.critic import ChunkCritic, TwinCritic
from rlt_so101_dual.core.losses import discounted_chunk_return, critic_loss, actor_loss
from rlt_so101_dual.core.replay_buffer import ReplayBuffer
from rlt_so101_dual.core.utils import (
    soft_update,
    flatten_chunk,
    unflatten_chunk,
    compute_discount_vector,
    build_mlp,
    filter_encoder_only,
    infer_actor_architecture,
)
from rlt_so101_dual.core.policy import RLTPolicy
from rlt_so101_dual.core.algorithm import RLTAlgorithm
from rlt_so101_dual.core.collector import Environment, DummyEnvironment, execute_chunk
from rlt_so101_dual.core.rewards import build_reward_seq

__all__ = [
    "ChunkTransition",
    "Observation",
    "VLAOutput",
    "RLTConfig",
    "OfflineRLConfig",
    "VLAAdapter",
    "DummyVLAAdapter",
    "RLTokenModule",
    "ChunkActor",
    "ChunkCritic",
    "TwinCritic",
    "discounted_chunk_return",
    "critic_loss",
    "actor_loss",
    "ReplayBuffer",
    "RLTPolicy",
    "RLTAlgorithm",
    "soft_update",
    "flatten_chunk",
    "unflatten_chunk",
    "compute_discount_vector",
    "build_mlp",
    "filter_encoder_only",
    "infer_actor_architecture",
    "Environment",
    "DummyEnvironment",
    "execute_chunk",
    "build_reward_seq",
]
