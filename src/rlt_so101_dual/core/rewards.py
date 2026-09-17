from __future__ import annotations

import torch


def build_reward_seq(
    chunk_length: int,
    is_terminal_chunk: bool,
    episode_success: bool,
    actual_steps: int | torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sparse terminal reward for a single chunk.

    Returns (C,) with 1.0 at the last valid step iff the chunk ends a
    successful episode; zeros otherwise. Matches the papers assumption
    that the supervisor only provides r_T=1 on success.
    """
    if isinstance(actual_steps, torch.Tensor):
        actual_steps = int(actual_steps.item())
    steps = (
        chunk_length if actual_steps is None
        else max(0, min(chunk_length, int(actual_steps)))
    )
    reward = torch.zeros(chunk_length, device=device)
    if is_terminal_chunk and episode_success and steps > 0:
        reward[steps - 1] = 1.0
    return reward
