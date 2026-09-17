"""RLT training losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.critic import TwinCritic
from rlt_so101_dual.core.utils import compute_discount_vector

def discounted_chunk_return(
    reward_seq: torch.Tensor, gamma: float, actual_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted return over a chunk of rewards.

    `actual_steps` is accepted but not used in the arithmetic, and that is
    correct rather than an oversight: the sum runs over all C entries, and
    every writer pads the region past `actual_steps` with zeros, so those
    terms vanish whatever discount they are multiplied by. A reward written
    at index `actual-1` correctly picks up `gamma ** (actual - 1)`.

    That only holds while the padding really is zero, which is an invariant
    of the callers rather than of this function -- milestone shaping put
    rewards on non-terminal chunks and it still holds. The argument is kept
    so the assertion below can enforce the invariant, since a future writer
    reaching past `actual_steps` would otherwise silently inflate returns.

    Args:
        reward_seq: (B, C) rewards per timestep, zero beyond actual_steps
        gamma: discount factor
        actual_steps: (B,) valid steps per chunk; None means all C are valid

    Returns:
        (B, 1) discounted return
    """
    C = reward_seq.shape[1]
    if actual_steps is not None and __debug__:
        steps = actual_steps.reshape(-1).long().clamp(0, C)
        past_end = torch.arange(C, device=reward_seq.device).unsqueeze(0) >= steps.unsqueeze(1)
        if bool((reward_seq * past_end).any()):
            raise ValueError(
                "reward_seq carries a nonzero value past actual_steps. Rewards must be "
                "written at index actual_steps-1 or earlier; anything beyond it is "
                "padding and would be discounted as though it happened."
            )
    discounts = compute_discount_vector(gamma, C, device=reward_seq.device)
    return (reward_seq * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)

def critic_loss(
    critic: TwinCritic,
    target_critic: TwinCritic,
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
    target_q_clip: float | None = 100.0,
    target_policy_noise: float = 0.0,
    target_noise_clip: float = 0.5,
) -> torch.Tensor:
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        # Use deterministic mean for target action (TD3-style), clamped to [-1,1]
        mu_next, _ = actor.forward(x_next, ref_next)
        if target_policy_noise > 0.0:
            noise = (torch.randn_like(mu_next) * target_policy_noise).clamp(
                -target_noise_clip, target_noise_clip
            )
            mu_next = mu_next + noise
        mu_next = mu_next.clamp(-1.0, 1.0)
        q_next = target_critic.min_q(x_next, mu_next)
        if target_q_clip is not None and target_q_clip > 0:
            q_next = q_next.clamp(-target_q_clip, target_q_clip)
        r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        # Bootstrap with gamma^k where k = actual steps executed
        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = r + bootstrap

    q1, q2 = critic(x, a)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)

def actor_loss(
    actor: ChunkActor,
    critic: TwinCritic,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    mu, _ = actor.forward(x, ref, training=True)
    q = critic.min_q(x, mu)
    bc_reg = ((mu - ref) ** 2).sum(dim=-1).mean()
    return -q.mean() + beta * bc_reg
