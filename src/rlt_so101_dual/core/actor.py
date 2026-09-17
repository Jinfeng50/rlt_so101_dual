"""Chunk actor network."""

from __future__ import annotations

import torch
import torch.nn as nn

from rlt_so101_dual.core.utils import build_mlp, _get_activation

class ResidualMLP(nn.Module):

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block_layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim)]
            if layer_norm:
                block_layers.append(nn.LayerNorm(hidden_dim))
            block_layers.append(_get_activation(activation))
            blocks.append(nn.Sequential(*block_layers))
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)

class ChunkActor(nn.Module):

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.05,
        ref_dropout_p: float = 0.5,
        activation: str = "relu",
        layer_norm: bool = False,
        residual: bool = False,
        residual_to_ref: bool = False,
    ):
        super().__init__()
        if residual:
            self.net = ResidualMLP(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        else:
            self.net = build_mlp(
                state_dim + chunk_dim, hidden_dim, chunk_dim, num_layers,
                activation=activation, layer_norm=layer_norm,
            )
        self.fixed_std = fixed_std
        self.ref_dropout_p = ref_dropout_p
        self.residual_to_ref = residual_to_ref
        self.register_buffer("_pin_mask", None, persistent=False)
        if residual_to_ref:
            # Zero-init the final layer so mu == ref_chunk exactly at init
            # (delta == 0), giving a safe starting policy for online RL on
            # real hardware: the untrained actor is a no-op over the VLA
            # reference until gradient updates move it away from zero.
            out_layer = self.net.output_proj if residual else self.net[-1]
            nn.init.zeros_(out_layer.weight)
            nn.init.zeros_(out_layer.bias)

    def set_pin_mask(self, pin_mask: torch.Tensor | None) -> None:
        """Pin chosen flattened action entries to the VLA reference.

        For a joint the task never moves, dropping only its BC term is not
        enough: it hands that dimension to Q alone. wrist_roll is the worst
        possible candidate for that -- its exec-ref spread in the run4 buffer is
        std 0.957 against 0.087-0.200 for every other joint, and that spread is
        not action variation at all. The operator nudges the wrist while
        dragging the leader arm back and cannot restore it exactly, and
        QUANTILES amplifies that residual ~79x (q99 - q01 = 0.558 deg). So the
        critic fits structure into pure repositioning noise on precisely the
        dimension the actor can barely move (a 0.1 normalised clip is 0.028 deg
        there). Pinning mu to ref makes the BC contribution exactly zero, keeps
        Q from chasing that noise, and makes the loss query the same action the
        robot executes on that joint.
        """
        self.register_buffer("_pin_mask", pin_mask, persistent=False)

    def forward(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (mu, std)."""
        net_ref = ref_chunk_flat
        if training:
            mask = (
                torch.rand(state_vec.shape[0], 1, device=state_vec.device) > self.ref_dropout_p
            ).float()
            net_ref = ref_chunk_flat * mask
        x = torch.cat([state_vec, net_ref], dim=-1)
        delta = self.net(x)
        mu = ref_chunk_flat + delta if self.residual_to_ref else delta
        mu = self._apply_pin(mu, ref_chunk_flat)
        std = torch.full_like(mu, self.fixed_std)
        return mu, std

    def _apply_pin(self, mu: torch.Tensor, ref_chunk_flat: torch.Tensor) -> torch.Tensor:
        """Pin to the *undropped* reference -- net_ref may have been zeroed."""
        pin = getattr(self, "_pin_mask", None)
        if pin is None:
            return mu
        pin = pin.to(mu.device, mu.dtype)
        return mu * (1.0 - pin) + ref_chunk_flat * pin

    def sample(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with Gaussian noise. Returns (action, mu)."""
        mu, std = self.forward(state_vec, ref_chunk_flat, training)
        # Pin after the noise too, or exploration would move a pinned joint.
        action = self._apply_pin(mu + std * torch.randn_like(std), ref_chunk_flat)
        return action, mu
