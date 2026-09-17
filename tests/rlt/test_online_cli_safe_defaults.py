"""Online CLI defaults aligned with the RLT paper path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rlt_so101_dual.adapters.lerobot.record.online_cli import build_online_train_argv, build_parser


def _minimal_args(**overrides):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--vla-path", "/tmp/vla",
            "--rl-token-path", "/tmp/tok",
            "--tokenizer-path", "/tmp/tokz",
            "--task", "dummy task",
            "--gamma", "0.999",
        ]
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _paths():
    return SimpleNamespace(
        dataset_name="eval_online_rl_test",
        dataset_root=Path("/tmp/ds"),
        day_dir=Path("/tmp"),
    )


def _setup():
    return SimpleNamespace(
        followers=[{"port": "/tmp/left"}, {"port": "/tmp/right"}],
        left_cameras={},
        right_cameras={},
    )


def test_online_cli_defaults_match_paper():
    args = _minimal_args()
    assert args.actor_residual_to_ref is False
    assert args.actor_fixed_std == 0.05
    assert args.actor_action_clip_delta is None
    assert args.actor_ref_dropout_p == 0.5
    assert args.critic_layer_norm is False
    assert args.stratified_sampling is False
    assert args.milestone_reward == 0.0
    assert args.time_decay == 1.0
    assert args.utd_ratio == 5
    assert args.critic_only_episodes == 10


def test_build_argv_paper_defaults():
    args = _minimal_args()
    argv = build_online_train_argv(
        args, setup=_setup(), paths=_paths(), cal_dir="/tmp/cal", teleop_argv=[],
    )
    assert "--policy.actor_residual_to_ref=false" in argv
    assert "--policy.actor_ref_dropout_p=0.5" in argv
    assert "--policy.actor_fixed_std=0.05" in argv
    assert "--policy.critic_layer_norm=false" in argv
    assert "--online_rl.use_stratified_sampling=false" in argv
    assert "--online_rl.milestone_reward=0.0" in argv
    assert "--online_rl.time_decay=1.0" in argv
    assert not any(a.startswith("--policy.actor_action_clip_delta=") for a in argv)


def test_build_argv_forces_zero_ref_dropout_when_residual_to_ref():
    args = _minimal_args(actor_residual_to_ref=True, actor_ref_dropout_p=0.5)
    argv = build_online_train_argv(
        args, setup=_setup(), paths=_paths(), cal_dir="/tmp/cal", teleop_argv=[],
    )
    assert "--policy.actor_residual_to_ref=true" in argv
    assert "--policy.actor_ref_dropout_p=0.0" in argv


def test_so101_safety_overrides_reach_argv():
    args = _minimal_args(
        actor_residual_to_ref=True,
        actor_fixed_std=0.0,
        actor_action_clip_delta=0.05,
        stratified_sampling=True,
        milestone_reward=0.3,
        time_decay=0.98,
    )
    argv = build_online_train_argv(
        args, setup=_setup(), paths=_paths(), cal_dir="/tmp/cal", teleop_argv=[],
    )
    assert "--policy.actor_residual_to_ref=true" in argv
    assert "--policy.actor_fixed_std=0.0" in argv
    assert "--policy.actor_action_clip_delta=0.05" in argv
    assert "--online_rl.use_stratified_sampling=true" in argv
    assert "--online_rl.milestone_reward=0.3" in argv
    assert "--online_rl.time_decay=0.98" in argv
