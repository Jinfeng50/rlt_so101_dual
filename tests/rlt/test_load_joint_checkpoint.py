#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Test that downstream code can load an RL Token saved by a joint train run.

Simulates a downstream actor-critic loader path: it reads
<ckpt>/rl_token/state_dict.safetensors + meta.json, rebuilds an RLTokenModule
from meta.json, and verifies parameter equality.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from rlt_so101_dual.adapters.lerobot.training_config import RLTokenJointConfig
from rlt_so101_dual.core.rl_token import RLTokenModule
from rlt_so101_dual.adapters.lerobot.train_utils import (
    RL_TOKEN_DIR,
    RL_TOKEN_META_FILENAME,
    RL_TOKEN_STATE_FILENAME,
    save_rl_token_state,
)


def test_downstream_loads_rl_token_from_joint_checkpoint(tmp_path: Path):
    """Downstream rebuild of RLTokenModule from <ckpt>/rl_token/ matches saved weights."""
    cfg = RLTokenJointConfig(
        enable=True,
        num_rl_tokens=2,
        nhead=4,
        num_enc_layers=2,
        num_dec_layers=2,
        ff_dim=128,
        token_pool_size=32,
        image_only=True,
    )

    saved = RLTokenModule(
        token_dim=32,
        nhead=cfg.nhead,
        num_enc_layers=cfg.num_enc_layers,
        num_dec_layers=cfg.num_dec_layers,
        ff_dim=cfg.ff_dim,
        num_rl_tokens=cfg.num_rl_tokens,
        inference_only=False,
    )
    ckpt_dir = tmp_path / "checkpoints" / "step000010"
    ckpt_dir.mkdir(parents=True)
    save_rl_token_state(ckpt_dir, saved, cfg, num_image_tokens=128)

    # Downstream-style rebuild: read meta.json, instantiate fresh module from module_config,
    # load state dict.
    rl_dir = ckpt_dir / RL_TOKEN_DIR
    state_path = rl_dir / RL_TOKEN_STATE_FILENAME
    meta_path = rl_dir / RL_TOKEN_META_FILENAME
    assert state_path.exists()
    assert meta_path.exists()

    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["module"] == "rlt_so101_dual.core.rl_token.RLTokenModule"
    mc = meta["module_config"]

    rebuilt = RLTokenModule(
        token_dim=mc["token_dim"],
        nhead=mc["nhead"],
        num_enc_layers=mc["num_enc_layers"],
        num_dec_layers=mc["num_dec_layers"],
        ff_dim=mc["ff_dim"],
        num_rl_tokens=mc["num_rl_tokens"],
        inference_only=mc["inference_only"],
    )

    sd = load_file(str(state_path))
    rebuilt.load_state_dict(sd, strict=True)

    # Param equality
    for (na, pa), (nb, pb) in zip(saved.named_parameters(), rebuilt.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb), f"{na} differs"

    # And the encoder still computes a real value
    fake = torch.randn(2, 16, mc["token_dim"])
    z_rl = rebuilt.encode(fake)
    assert z_rl.shape == (2, mc["token_dim"])
    assert torch.isfinite(z_rl).all()

    # postprocess metadata round-trip
    pp = meta["postprocess"]
    assert pp["image_only"] is True
    assert pp["token_pool_size"] == 32
    assert pp["num_image_tokens"] == 128
