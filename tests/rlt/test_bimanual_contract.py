"""Guards for the bimanual SO101 data contract (rlt_so101_dual)."""

import json
import tempfile
from pathlib import Path

import torch

from rlt_so101_dual.adapters.lerobot.record.common import (
    build_camera_configs,
    build_robot_argv,
    build_teleop_argv,
    load_robot_setup,
)
from rlt_so101_dual.core import shape_contract as sc

class _Raises:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"expected {self.exc_type}")
        if not issubclass(et, self.exc_type):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(f"expected match {self.match!r} in {ev!r}")
        return True

def raises(exc_type, match=None):
    return _Raises(exc_type, match)

def _manifest(tmp_path, arms, cameras):
    path = tmp_path / "setup.json"
    path.write_text(
        json.dumps(
            {
                "datasets": {"root": str(tmp_path / "datasets")},
                "arms": arms,
                "cameras": cameras,
            }
        )
    )
    return str(path)

def _cams(**overrides):
    cams = [
        {"alias": "left_wrist", "port": 0, "fourcc": "YUYV"},
        {"alias": "right_wrist", "port": 4, "fourcc": "YUYV"},
        {"alias": "right_front", "port": 2, "fourcc": "YUYV"},
    ]
    if overrides.get("alias_of_first"):
        cams[0]["alias"] = overrides["alias_of_first"]
    if overrides.get("drop_last"):
        cams = cams[:-1]
    return cams

FOLLOWERS = [
    {"alias": "left_follower", "type": "follower", "port": "/dev/left_fol"},
    {"alias": "right_follower", "type": "follower", "port": "/dev/right_fol"},
]
LEADERS = [
    {"alias": "left_leader", "type": "leader", "port": "/dev/left_led"},
    {"alias": "right_leader", "type": "leader", "port": "/dev/right_led"},
]

def test_contract_is_bimanual_so101():
    assert sc.ROBOT_TYPE == "bi_so_follower"
    assert sc.ACTION_DIM == sc.PROPRIO_DIM == 12
    assert len(sc.JOINT_NAMES) == 12
    assert all(n.startswith(("left_", "right_")) for n in sc.JOINT_NAMES)
    assert sc.CAMERA_KEYS == ["left_wrist", "right_wrist", "right_front"]
    assert sc.FPS == 30
    assert sc.state_vec_dim() == 2048 + 12
    assert sc.chunk_flat_dim() == 10 * 12

def test_pi05_camera_slots_are_three():
    assert len(sc.PI05_CAMERA_ORDER) == 3
    assert set(sc.PI05_CAMERA_MAP) == set(sc.CAMERA_KEYS)
    assert sc.PI05_CAMERA_MAP["right_front"].endswith("base_0_rgb")
    assert sc.PI05_CAMERA_MAP["left_wrist"].endswith("left_wrist_0_rgb")
    assert sc.PI05_CAMERA_MAP["right_wrist"].endswith("right_wrist_0_rgb")

def test_camera_alias_rebuilds_from_arm_local():
    for alias in sc.CAMERA_KEYS:
        side = sc.CAMERA_ARM_SIDE[alias]
        local = sc.CAMERA_ARM_LOCAL[alias]
        assert f"{side}_{local}" == alias
    assert sc.CAMERA_ARM_LOCAL == {
        "left_wrist": "wrist",
        "right_wrist": "wrist",
        "right_front": "front",
    }

def test_dual_manifest_is_accepted(tmp_path):
    setup = load_robot_setup(_manifest(tmp_path, FOLLOWERS + LEADERS, _cams()))
    assert len(setup.followers) == 2
    assert len(setup.leaders) == 2
    assert sorted(setup.left_cameras) == ["wrist"]
    assert sorted(setup.right_cameras) == ["front", "wrist"]

def test_single_arm_manifest_is_rejected(tmp_path):
    arms = [FOLLOWERS[0], LEADERS[0]]
    with raises(ValueError, match="exactly 2 follower"):
        load_robot_setup(_manifest(tmp_path, arms, _cams()))

def test_unknown_camera_alias_is_rejected_not_dropped(tmp_path):
    with raises(ValueError, match="Unknown camera alias"):
        build_camera_configs(_cams(alias_of_first="top"))

def test_missing_camera_is_rejected(tmp_path):
    with raises(ValueError, match="missing camera"):
        build_camera_configs(_cams(drop_last=True))

def test_camera_configs_expose_arm_local_names(tmp_path):
    left, right = build_camera_configs(_cams())
    assert sorted(left) == ["wrist"]
    assert sorted(right) == ["front", "wrist"]

def test_robot_argv_is_bimanual(tmp_path):
    setup = load_robot_setup(_manifest(tmp_path, FOLLOWERS + LEADERS, _cams()))
    argv = build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, "/tmp/cal")
    joined = " ".join(argv)
    assert "--robot.type=bi_so_follower" in argv
    assert "left_arm_config.port" in joined
    assert "right_arm_config.port" in joined
    assert "--robot.left_arm_config.max_relative_target=12.0" in argv
    assert "--robot.right_arm_config.max_relative_target=12.0" in argv
    left_cams = json.loads(next(a for a in argv if "left_arm_config.cameras=" in a).split("=", 1)[1])
    right_cams = json.loads(next(a for a in argv if "right_arm_config.cameras=" in a).split("=", 1)[1])
    assert sorted(left_cams) == ["wrist"]
    assert sorted(right_cams) == ["front", "wrist"]

def test_teleop_argv_is_bi_leader(tmp_path):
    setup = load_robot_setup(_manifest(tmp_path, FOLLOWERS + LEADERS, _cams()))
    argv = build_teleop_argv(setup.leaders, no_teleop=False)
    assert "--teleop.type=bi_so_leader" in argv

def test_yaml_shape_fields_are_not_overwritten(tmp_path):
    from rlt_so101_dual.cli.common import load_training_config

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("action_dim: 7\nproprio_dim: 8\nchunk_length: 3\ncameras: [alpha]\n")
    cfg = load_training_config(str(cfg_path))
    assert cfg.action_dim == 7
    assert cfg.proprio_dim == 8
    assert cfg.chunk_length == 3
    assert cfg.cameras == ["alpha"]

def test_narrow_state_is_rejected_before_padding():
    with raises(ValueError, match="expected last dim 12, got 6"):
        sc.check_last_dim("observation.state", torch.zeros(2, 6), sc.PROPRIO_DIM)
    sc.check_last_dim("observation.state", torch.zeros(2, 12), sc.PROPRIO_DIM)

def test_cache_dims_can_be_recovered_for_comparison():
    state_vec = torch.zeros(4, sc.state_vec_dim())
    ref_flat = torch.zeros(4, sc.chunk_flat_dim())
    proprio_dim, action_dim = sc.infer_dims_from_cache(state_vec, ref_flat, sc.CHUNK_LENGTH)
    assert (proprio_dim, action_dim) == (sc.PROPRIO_DIM, sc.ACTION_DIM)

def test_go_home_accepts_bimanual_joint_names():
    assert "left_shoulder_pan.pos" in sc.JOINT_NAMES
    assert "right_gripper.pos" in sc.JOINT_NAMES
    unknown = sorted({"foo.pos"} - set(sc.JOINT_NAMES))
    assert unknown == ["foo.pos"]

def test_reward_shaping_flags_reach_the_record_config(tmp_path):
    from rlt_so101_dual.adapters.lerobot.record.common import resolve_run_paths
    from rlt_so101_dual.adapters.lerobot.record.online_cli import (
        build_online_train_argv,
        build_parser,
    )

    setup_json = _manifest(tmp_path, FOLLOWERS + LEADERS, _cams())
    args = build_parser().parse_args([
        "--setup-json", setup_json,
        "--vla-path", "/tmp/vla",
        "--rl-token-path", "/tmp/tok",
        "--tokenizer-path", "/tmp/tokz",
        "--task", "pick up the bolt and insert it into the sleeve",
        "--gamma", "0.999",
        "--milestone-reward", "0.25",
        "--terminal-reward", "2.0",
        "--time-decay", "0.99",
        "--milestone-key", "m",
    ])
    setup = load_robot_setup(setup_json)
    paths = resolve_run_paths(setup.setup, "t", "p")
    argv = build_online_train_argv(args, setup, paths, "/tmp/cal", [])

    assert "--online_rl.milestone_reward=0.25" in argv
    assert "--online_rl.terminal_reward=2.0" in argv
    assert "--online_rl.time_decay=0.99" in argv
    assert "--rlt.milestone_key=m" in argv
    assert "--policy.gamma=0.999" in argv

def test_milestone_key_does_not_collide_with_existing_bindings():
    from rlt_so101_dual.adapters.lerobot.record.backend import RLTRecordConfig

    cfg = RLTRecordConfig()
    bound = {cfg.rl_phase_key, cfg.rl_phase_failure_key, cfg.end_success_key, cfg.end_failure_key}
    assert cfg.milestone_key not in bound

def test_yaml_disagreeing_with_the_contract_is_warned_about(tmp_path, caplog):
    import logging

    from rlt_so101_dual.cli.common import load_training_config

    cfg_path = tmp_path / "single.yaml"
    cfg_path.write_text("action_dim: 6\nproprio_dim: 6\n")
    with caplog.at_level(logging.WARNING):
        cfg = load_training_config(str(cfg_path))
    assert cfg.action_dim == 6, "the YAML still wins"
    assert "does NOT match" in caplog.text

def test_online_train_refuses_to_start_without_gamma():
    from rlt_so101_dual.adapters.lerobot.record.online_cli import build_parser

    with raises(SystemExit):
        build_parser().parse_args([
            "--setup-json", "/tmp/x.json", "--vla-path", "/tmp/v",
            "--rl-token-path", "/tmp/t", "--tokenizer-path", "/tmp/tz",
            "--task", "test task",
        ])

def test_online_train_accepts_the_documented_gamma():
    from rlt_so101_dual.adapters.lerobot.record.online_cli import build_parser

    args = build_parser().parse_args([
        "--setup-json", "/tmp/x.json", "--vla-path", "/tmp/v",
        "--rl-token-path", "/tmp/t", "--tokenizer-path", "/tmp/tz",
        "--task", "test task",
        "--gamma", "0.999",
    ])
    assert args.gamma == 0.999
