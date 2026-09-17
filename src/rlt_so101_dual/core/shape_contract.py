"""SO101 dual-arm shape / camera / chunk contract (single source of truth)."""

from __future__ import annotations

from typing import Literal, Sequence

import torch

# ---------------------------------------------------------------------------
# 1) LeRobot / BiSO
# ---------------------------------------------------------------------------

ROBOT_TYPE = "bi_so_follower"

_SINGLE_JOINTS: list[str] = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

JOINT_NAMES: list[str] = [
    *(f"left_{n}" for n in _SINGLE_JOINTS),
    *(f"right_{n}" for n in _SINGLE_JOINTS),
]

ACTION_DIM = len(JOINT_NAMES)  # 12
PROPRIO_DIM = len(JOINT_NAMES)

# ---------------------------------------------------------------------------
# 2) openpi / π0.5 image slots
# ---------------------------------------------------------------------------

PI05_IMAGE_SIDE = 224
PI05_MAX_DIM = 32

PI05_SLOT_BASE = "observation.images.base_0_rgb"
PI05_SLOT_LEFT_WRIST = "observation.images.left_wrist_0_rgb"
PI05_SLOT_RIGHT_WRIST = "observation.images.right_wrist_0_rgb"

PI05_CAMERA_ORDER: list[str] = [
    PI05_SLOT_BASE,
    PI05_SLOT_LEFT_WRIST,
    PI05_SLOT_RIGHT_WRIST,
]

# ---------------------------------------------------------------------------
# 3) This repo: cameras / fps / RLT dims
# ---------------------------------------------------------------------------

FPS = 30
IMAGE_HW: tuple[int, int] = (480, 640)  # (H, W); feature shape [H, W, 3]

CAMERA_KEYS: list[str] = ["left_wrist", "right_wrist", "right_front"]

CAMERA_ARM_LOCAL: dict[str, str] = {
    "left_wrist": "wrist",
    "right_wrist": "wrist",
    "right_front": "front",
}

CAMERA_ARM_SIDE: dict[str, Literal["left", "right"]] = {
    "left_wrist": "left",
    "right_wrist": "right",
    "right_front": "right",
}

PI05_CAMERA_MAP: dict[str, str] = {
    "right_front": PI05_SLOT_BASE,
    "left_wrist": PI05_SLOT_LEFT_WRIST,
    "right_wrist": PI05_SLOT_RIGHT_WRIST,
}

VLA_HORIZON = 50
CHUNK_LENGTH = 10
RL_TOKEN_DIM = 2048

def _assert_camera_contract() -> None:
    """Validate camera tables are consistent."""
    keys = set(CAMERA_KEYS)
    if keys != set(CAMERA_ARM_LOCAL):
        raise RuntimeError(f"CAMERA_ARM_LOCAL keys {set(CAMERA_ARM_LOCAL)} != CAMERA_KEYS")
    if keys != set(CAMERA_ARM_SIDE):
        raise RuntimeError(f"CAMERA_ARM_SIDE keys {set(CAMERA_ARM_SIDE)} != CAMERA_KEYS")
    if keys != set(PI05_CAMERA_MAP):
        raise RuntimeError(f"PI05_CAMERA_MAP keys {set(PI05_CAMERA_MAP)} != CAMERA_KEYS")
    if set(PI05_CAMERA_MAP.values()) != set(PI05_CAMERA_ORDER):
        raise RuntimeError("PI05_CAMERA_MAP values must be exactly PI05_CAMERA_ORDER")
    for alias, local in CAMERA_ARM_LOCAL.items():
        side = CAMERA_ARM_SIDE[alias]
        rebuilt = f"{side}_{local}"
        if rebuilt != alias:
            raise RuntimeError(
                f"camera alias {alias!r} != BiSO rebuild {rebuilt!r} "
                f"(side={side}, local={local}); keep alias = {{side}}_{{arm_local}}"
            )

_assert_camera_contract()

def dataset_image_key(camera: str) -> str:
    """``left_wrist`` → ``observation.images.left_wrist``."""
    return f"observation.images.{camera}"

def state_vec_dim(proprio_dim: int = PROPRIO_DIM, rl_token_dim: int = RL_TOKEN_DIM) -> int:
    """Width of the actor/critic state input: [rl_token ; proprio]."""
    return rl_token_dim + proprio_dim

def chunk_flat_dim(chunk_length: int = CHUNK_LENGTH, action_dim: int = ACTION_DIM) -> int:
    """Width of a flattened action chunk."""
    return chunk_length * action_dim

def check_dim(name: str, got: int, expected: int, hint: str = "") -> None:
    """Raise unless ``got == expected``."""
    if got != expected:
        suffix = f"\n{hint}" if hint else ""
        raise ValueError(f"{name}: expected {expected}, got {got}.{suffix}")

def check_last_dim(name: str, tensor: torch.Tensor, expected: int, hint: str = "") -> None:
    """Raise unless ``tensor.shape[-1] == expected``."""
    got = int(tensor.shape[-1])
    if got != expected:
        suffix = f"\n{hint}" if hint else ""
        raise ValueError(
            f"{name}: expected last dim {expected}, got {got} (shape {tuple(tensor.shape)}).{suffix}"
        )

def check_cameras(got: Sequence[str], expected: Sequence[str] = CAMERA_KEYS) -> None:
    """Raise unless the camera key lists match exactly, order included."""
    if list(got) != list(expected):
        raise ValueError(
            f"camera keys: expected {list(expected)}, got {list(got)}. "
            "Camera order determines prefix-token layout, so a permutation is "
            "as wrong as a missing camera."
        )

def infer_dims_from_cache(
    state_vec: torch.Tensor,
    ref_chunk: torch.Tensor,
    chunk_length: int,
    rl_token_dim: int = RL_TOKEN_DIM,
) -> tuple[int, int]:
    """Recover ``(proprio_dim, action_dim)`` from cached transition tensors."""
    proprio_dim = int(state_vec.shape[-1]) - rl_token_dim
    flat = int(ref_chunk.shape[-1]) if ref_chunk.ndim == 2 else int(ref_chunk.shape[-1]) * chunk_length
    if ref_chunk.ndim == 2:
        if flat % chunk_length:
            raise ValueError(
                f"flattened ref_chunk width {flat} is not divisible by chunk_length {chunk_length}"
            )
        action_dim = flat // chunk_length
    else:
        action_dim = int(ref_chunk.shape[-1])
    return proprio_dim, action_dim
