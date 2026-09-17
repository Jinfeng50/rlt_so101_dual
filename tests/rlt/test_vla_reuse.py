"""VLA open-loop reuse: fewer π0.5 forwards while keeping C-step RL windows."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from rlt_so101_dual.adapters.lerobot.policies.action_modifier import RLTActionModifier
from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from rlt_so101_dual.core.actor import ChunkActor


def test_vla_reuse_horizon_must_be_multiple_of_chunk_length():
    try:
        ChunkACPolicyConfig(vla_reuse_horizon=25, chunk_length=10)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_select_action_reuses_one_vla_forward_across_windows():
    C, AD, PD, H = 10, 12, 12, 30
    calls = {"n": 0}

    policy = ChunkACPolicy.__new__(ChunkACPolicy)
    policy.config = type("Cfg", (), {})()
    policy.config.chunk_length = C
    policy.config.action_dim = AD
    policy.config.proprio_dim = PD
    policy.config.vla_reuse_horizon = H
    # _rtc_runtime_enabled is a property over _rtc_config.
    object.__setattr__(policy, "_rtc_config", None)
    object.__setattr__(policy, "_vla_reuse_actions", None)
    object.__setattr__(policy, "_vla_reuse_prefix", None)
    object.__setattr__(policy, "_vla_reuse_cursor", 0)
    object.__setattr__(policy, "_logged_vla_reuse", False)

    actor = ChunkActor(
        state_dim=8 + PD,
        chunk_dim=C * AD,
        hidden_dim=8,
        num_layers=2,
        fixed_std=0.0,
        residual_to_ref=False,
    )

    class _Tok:
        def encode(self, prefix):
            return torch.zeros(1, 8)

    class _Phase:
        is_critical = True

    mod = RLTActionModifier(
        rl_token=_Tok(),
        actor=actor,
        phase_ctrl=_Phase(),
        chunk_length=C,
        action_dim=AD,
        proprio_dim=PD,
        chunk_exec_steps=C,
        vla_ref=True,
        action_clip_delta=None,
    )
    mod.actor_enabled = False
    object.__setattr__(policy, "modifier", mod)

    def fake_raw(batch, **kwargs):
        calls["n"] += 1
        vla = torch.zeros(1, 50, AD)
        for i in range(50):
            vla[0, i] = float(i)
        return vla, torch.zeros(1, 1, 16)

    object.__setattr__(policy, "_predict_vla_raw", fake_raw)
    object.__setattr__(policy, "_ensure_modifier", lambda: mod)

    batch = {"observation.state": torch.zeros(1, PD)}
    for _ in range(H):
        ChunkACPolicy.select_action(policy, batch)
    assert calls["n"] == 1
    ChunkACPolicy.select_action(policy, batch)
    assert calls["n"] == 2


def test_online_cli_defaults_and_argv_flag(tmp_path):
    from rlt_so101_dual.adapters.lerobot.record.online_cli import (
        build_online_train_argv,
        build_parser,
    )

    args = build_parser().parse_args([
        "--vla-path", "/tmp/vla",
        "--rl-token-path", "/tmp/tok",
        "--tokenizer-path", "/tmp/tokz",
        "--task", "t",
        "--gamma", "0.999",
    ])
    assert args.vla_reuse_horizon == 10
    assert args.chunk_exec_steps == 50
    assert args.vcodec == "h264"

    follower_list = [
        {"port": "/dev/ttyUSB0", "id": "left"},
        {"port": "/dev/ttyUSB1", "id": "right"},
    ]

    class _Setup:
        setup = {}
        followers = follower_list
        leaders = []
        left_cameras = {}
        right_cameras = {}

    class _Paths:
        dataset_name = "eval_online_rl_x"
        dataset_root = tmp_path / "ds"
        day_dir = tmp_path / "day"
        log_file = tmp_path / "log.txt"

    argv = build_online_train_argv(
        Namespace(**vars(args)), _Setup(), _Paths(), str(tmp_path / "cal"), []
    )
    assert "--policy.vla_reuse_horizon=10" in argv
    assert "--dataset.vcodec=h264" in argv
