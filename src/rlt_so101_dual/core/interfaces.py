"""Shared dataclasses and protocol types."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

TRANSITION_SOURCE_DEMO = 0
TRANSITION_SOURCE_WARMUP_VLA = 1
TRANSITION_SOURCE_RL_AUTONOMOUS = 2
TRANSITION_SOURCE_HUMAN_OVERRIDE = 3

STATE_VEC = "state_vec"
EXEC_CHUNK_FLAT = "exec_chunk_flat"
REF_CHUNK_FLAT = "ref_chunk_flat"
REWARD_SEQ = "reward_seq"
NEXT_STATE_VEC = "next_state_vec"
NEXT_REF_FLAT = "next_ref_flat"
DONE = "done"
ACTUAL_STEPS = "actual_steps"
SOURCE = "source"
EPISODE_ID = "episode_id"
IS_CRITICAL = "is_critical"
OUTCOME = "outcome"

@dataclass
class Observation:

    images: dict[str, torch.Tensor]  # camera_name -> (B, C, H, W)
    proprio: torch.Tensor  # (B, proprio_dim)
    instruction_ids: torch.Tensor | None = None
    timestamp: float | None = None

@dataclass
class VLAOutput:

    final_tokens: torch.Tensor  # (B, M, token_dim)
    sampled_action_chunk: torch.Tensor  # (B, H, action_dim)
    extra: dict = field(default_factory=dict)

@dataclass
class ChunkTransition:

    state_vec: torch.Tensor  # (state_dim,)
    exec_chunk: torch.Tensor  # (C, action_dim)
    ref_chunk: torch.Tensor  # (C, action_dim)
    reward_seq: torch.Tensor  # (C,)
    next_state_vec: torch.Tensor  # (state_dim,)
    next_ref_chunk: torch.Tensor  # (C, action_dim)
    done: torch.Tensor  # scalar
    intervention: torch.Tensor  # scalar, 0/1 flag
    actual_steps: torch.Tensor  # scalar int, steps actually executed (<= C)
    source: torch.Tensor = field(default_factory=lambda: torch.tensor(0))
    episode_id: torch.Tensor = field(default_factory=lambda: torch.tensor(-1))
    is_critical: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    # Explicit episode outcome: 1.0 success, 0.0 failure, -1.0 unresolved.
    # Deliberately separate from reward_seq. With milestone shaping a failed
    # attempt can still carry a positive reward, so inferring the outcome from
    # the sign of the reward -- which is what the code used to do -- would file
    # failures into the success bucket of stratified sampling. -1.0 keeps
    # transitions written before this field existed readable.
    outcome: torch.Tensor = field(default_factory=lambda: torch.tensor(-1.0))
