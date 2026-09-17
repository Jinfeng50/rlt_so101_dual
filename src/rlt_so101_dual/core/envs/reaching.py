from __future__ import annotations

import torch

from rlt_so101_dual.core.collector import Environment
from rlt_so101_dual.core.interfaces import Observation


class ReachingEnvironment(Environment):
    """12-DOF reaching task for validating the RL training pipeline.

    The agent must move a 12-dimensional position vector to match a random target.
    Reward is the negative normalized L2 distance; episode ends when close enough.
    

    Toy environment, unrelated to the SO101: its dimensions are arbitrary and
    deliberately NOT taken from shape_contract. Used only by its own test; no
    production path constructs it.
    """

    def __init__(self, proprio_dim: int = 12, action_dim: int = 12):
        self._proprio_dim = proprio_dim
        self._action_dim = action_dim
        self.position = torch.zeros(self._proprio_dim)
        self.target = torch.zeros(self._proprio_dim)

    @property
    def proprio_dim(self) -> int:
        return self._proprio_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _make_obs(self) -> Observation:
        return Observation(
            images={"base": torch.zeros(1, 3, 64, 64)},
            proprio=self.position.unsqueeze(0).clone(),
        )

    def reset(self) -> Observation:
        self.target = torch.rand(self._proprio_dim) * 2 - 1  # uniform in [-1, 1]
        self.position = torch.zeros(self._proprio_dim)
        return self._make_obs()

    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, dict]:
        if action.dim() == 2:
            action = action.squeeze(0)

        clipped = action.clamp(-0.1, 0.1)
        self.position = self.position + clipped.detach()

        distance = torch.norm(self.position - self.target).item()
        reward = -distance / self._proprio_dim
        done = distance < 0.05 * (self._proprio_dim ** 0.5)
        info = {"distance": distance, "target": self.target.clone()}

        return self._make_obs(), reward, done, info
