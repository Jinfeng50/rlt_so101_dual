from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_token import (
    VLA_SAFETENSORS_FILE,
    RLTokenPolicy,
)


def _make_policy(monkeypatch, vla_ft_weight: float) -> RLTokenPolicy:
    """Build an RLTokenPolicy with a tiny ``nn.Linear`` stand-in for pi0.5.

    Sidesteps the real pi0.5 load (slow, needs network) and the image-token
    shape probe (which dereferences pi05.model.paligemma_with_expert.…).
    """
    mock_pi05 = nn.Linear(8, 16)
    for p in mock_pi05.parameters():
        p.requires_grad = vla_ft_weight > 0

    monkeypatch.setattr(RLTokenPolicy, "_load_pi05_backbone", lambda self: mock_pi05)
    monkeypatch.setattr(RLTokenPolicy, "_compute_num_image_tokens", lambda self, pi05: 0)

    cfg = RLTokenPolicyConfig(
        device="cpu",
        rl_token_dim=16,
        rl_token_nhead=2,
        rl_token_enc_layers=1,
        rl_token_dec_layers=1,
        rl_token_ff_dim=32,
        rl_token_num_rl_tokens=1,
        vla_ft_weight=vla_ft_weight,
    )
    return RLTokenPolicy(cfg)


def test_round_trip_with_vla_ft(monkeypatch, tmp_path: Path) -> None:
    """Joint-training ckpt persists pi05 weights and restores them on load."""
    policy = _make_policy(monkeypatch, vla_ft_weight=1.0)
    with torch.no_grad():
        policy._pi05.weight.fill_(0.123)
        policy._pi05.bias.fill_(-0.5)

    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    policy._save_pretrained(save_dir)

    vla_file = save_dir / VLA_SAFETENSORS_FILE
    assert vla_file.exists(), f"expected cotrained VLA at {vla_file}"
    assert (save_dir / "model.safetensors").exists()

    fresh = _make_policy(monkeypatch, vla_ft_weight=1.0)
    assert not torch.allclose(fresh._pi05.weight, torch.full_like(fresh._pi05.weight, 0.123))

    model_file = str(save_dir / "model.safetensors")
    loaded = RLTokenPolicy._load_as_safetensor(fresh, model_file, "cpu", strict=True)
    assert loaded is fresh
    assert torch.allclose(loaded._pi05.weight, torch.full_like(loaded._pi05.weight, 0.123))
    assert torch.allclose(loaded._pi05.bias, torch.full_like(loaded._pi05.bias, -0.5))


def test_no_vla_file_when_frozen(monkeypatch, tmp_path: Path) -> None:
    """Frozen-VLA runs (vla_ft_weight==0) skip the auxiliary file."""
    policy = _make_policy(monkeypatch, vla_ft_weight=0.0)

    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    policy._save_pretrained(save_dir)

    assert (save_dir / "model.safetensors").exists()
    assert not (save_dir / VLA_SAFETENSORS_FILE).exists()


def test_load_ckpt_without_vla_file(monkeypatch, tmp_path: Path) -> None:
    """Old / frozen-VLA ckpts (no vla.safetensors) load without crash, pi05 untouched."""
    saver = _make_policy(monkeypatch, vla_ft_weight=0.0)
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    saver._save_pretrained(save_dir)
    assert not (save_dir / VLA_SAFETENSORS_FILE).exists()

    fresh = _make_policy(monkeypatch, vla_ft_weight=0.0)
    sentinel_w = fresh._pi05.weight.detach().clone()
    sentinel_b = fresh._pi05.bias.detach().clone()

    model_file = str(save_dir / "model.safetensors")
    loaded = RLTokenPolicy._load_as_safetensor(fresh, model_file, "cpu", strict=True)
    assert torch.equal(loaded._pi05.weight, sentinel_w)
    assert torch.equal(loaded._pi05.bias, sentinel_b)
