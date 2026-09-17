import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlt_so101_dual.adapters.lerobot.record.cli import build_parser
from rlt_so101_dual.adapters.lerobot.record import runner
from rlt_so101_dual.adapters.lerobot.record.common import (
    build_policy_overrides,
    checkpoint_rejects_field,
    resolve_camera_rename_map,
)
from rlt_so101_dual.adapters.lerobot.record.loop import _validate_policy_image_features
from rlt_so101_dual.adapters.lerobot.record.runner import (
    _collect_external_episode_outcome_key,
    _patch_episode_outcome_listener,
    _patch_skip_policyless_reset_loop,
    build_default_collect_record_argv,
    build_segment_record_argv,
)
from rlt_so101_dual.core import shape_contract as sc


def dual_arm_manifest(tmp_path, *, with_leaders=True, cameras=None):
    """A manifest matching the bimanual shape contract.

    Camera ports are placeholders; nothing in these tests opens a device.
    """
    arms = [
        {
            "alias": "left_follower",
            "type": "follower",
            "port": "/tmp/left-follower-port",
            "calibration_dir": str(tmp_path / "calibration" / "left_follower"),
        },
        {
            "alias": "right_follower",
            "type": "follower",
            "port": "/tmp/right-follower-port",
            "calibration_dir": str(tmp_path / "calibration" / "right_follower"),
        },
    ]
    if with_leaders:
        arms.extend(
            [
                {
                    "alias": "left_leader",
                    "type": "leader",
                    "port": "/tmp/left-leader-port",
                    "calibration_dir": str(tmp_path / "calibration" / "left_leader"),
                },
                {
                    "alias": "right_leader",
                    "type": "leader",
                    "port": "/tmp/right-leader-port",
                    "calibration_dir": str(tmp_path / "calibration" / "right_leader"),
                },
            ]
        )
    if cameras is None:
        cameras = [
            {"alias": key, "port": f"/dev/video{i}", "fourcc": "YUYV"}
            for i, key in enumerate(sc.CAMERA_KEYS)
        ]
    return {
        "datasets": {"root": str(tmp_path / "datasets")},
        "arms": arms,
        "cameras": cameras,
    }


def test_initial_source_rejects_rlt():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "segment",
        "--task",
        "test task",
            "--initial-source",
            "rlt",
            "--critical-source",
            "rlt",
        ])


def test_segment_defaults_to_rtc_enabled():
    parser = build_parser()
    args = parser.parse_args([
        "segment",
        "--task",
        "test task",
        "--initial-source",
        "teleop",
        "--critical-source",
        "rlt",
        "--policy-path",
        "/tmp/ac",
    ])
    assert args.rtc is True


def test_segment_rlt_argv_marks_key_segment_with_teleop_start_and_rtc():
    args = SimpleNamespace(
        critical_source="rlt",
        initial_source="teleop",
        policy_path="/tmp/ac",
        vla_path="/tmp/vla",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        reset_time_s=None,
        fps=30,
        vcodec="h264",
        intervention_action_blend_time_s=0.4,
        rtc=True,
        rtc_execution_horizon=10,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=None,
        vla_rtc_execution_horizon=None,
        vla_ref=True,
        chunk_exec_steps=25,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(
        dataset_name="local/test",
        dataset_root="/tmp/dataset",
    )
    argv = build_segment_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.rl_phase_key_toggles_episode=true" in argv
    assert "--rlt.start_in_teleop=true" in argv
    assert "--rlt.rtc_enabled=true" in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--policy.path=/tmp/ac" in argv


def test_pedal_listener_routes_record_events_and_episode_outcome(monkeypatch):
    control_utils = pytest.importorskip("lerobot.utils.control_utils")
    from rlt_so101_dual.adapters.lerobot.record import pedal_listener

    captured = {}

    class FakePedalListener:
        def __init__(self, on_press):
            captured["on_press"] = on_press

        def start(self):
            return True

        def stop(self):
            captured["stopped"] = True

    def original_init_keyboard_listener(*args, **kwargs):
        return None, {"episode_outcome": None, "exit_early": False}

    monkeypatch.setattr(control_utils, "is_headless", lambda: True)
    monkeypatch.setattr(control_utils, "init_keyboard_listener", original_init_keyboard_listener)
    monkeypatch.setattr(pedal_listener, "PedalListener", FakePedalListener)

    _patch_episode_outcome_listener("e")
    listener, events = control_utils.init_keyboard_listener(
        intervention_toggle_key=" ",
        rl_phase_key="r",
    )

    captured["on_press"]("space")
    assert events["toggle_intervention"] is True

    captured["on_press"]("r")
    assert events["start_rl_phase"] is True

    captured["on_press"]("e")
    assert events["episode_outcome"] == "success"
    assert events["exit_early"] is True

    events["episode_outcome"] = None
    events["exit_early"] = False
    captured["on_press"]("e")
    captured["on_press"]("u")
    assert events["episode_outcome"] == "failure"
    assert events["exit_early"] is True
    listener.stop()


def test_skip_policyless_reset_loop_keeps_recording_loop(monkeypatch):
    from rlt_so101_dual.adapters.lerobot.record import backend as lerobot_rlt_record

    calls = []

    def original_record_loop(*args, **kwargs):
        calls.append((args, kwargs))
        return "called"

    monkeypatch.setattr(lerobot_rlt_record, "record_loop", original_record_loop)

    _patch_skip_policyless_reset_loop()

    assert lerobot_rlt_record.record_loop(teleop=object(), control_time_s=10) is None
    assert calls == []
    assert lerobot_rlt_record.record_loop(policy=object(), dataset=object()) == "called"
    assert len(calls) == 1


def test_save_episode_patch_preserves_official_background_video_encoding(monkeypatch):
    calls = []

    class FakeMeta:
        total_episodes = 0
        video_keys = ["observation.images.left_wrist"]

        def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats, episode_metadata):
            calls.append(("meta", dict(episode_metadata)))
            self.total_episodes += 1

    class FakeWriter:
        def __init__(self):
            self._batch_encoding_size = 6

    class FakeLeRobotDataset:
        def __init__(self):
            self.meta = FakeMeta()
            self.writer = FakeWriter()

        def save_episode(self, *args, **kwargs):
            calls.append(("save", self.writer._batch_encoding_size, kwargs))
            self.meta.save_episode(0, 1, ["task"], {}, {"base": "metadata"})
            return "saved"

    fake_lerobot_dataset = type(sys)("lerobot.datasets.lerobot_dataset")
    fake_lerobot_dataset.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", fake_lerobot_dataset)

    runner._patch_save_episode_extra_metadata()

    dataset = FakeLeRobotDataset()
    assert dataset.save_episode(extra_episode_metadata={"episode_success": "success"}) == "saved"
    assert ("save", 6, {}) in calls
    assert ("meta", {"base": "metadata", "episode_success": "success"}) in calls
    assert dataset.writer._batch_encoding_size == 6


def test_default_collect_parser_requires_user_policy_path():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["collect"])


def test_default_collect_parser_uses_open_source_safe_defaults():
    parser = build_parser()
    args = parser.parse_args(["collect",
        "--task",
        "test task", "--policy-path", "/tmp/ac"])

    assert args.policy_path == "/tmp/ac"
    assert args.vla_path is None
    assert args.rl_token_path is None
    assert args.dataset_tag == "so101_dual_collect"
    assert args.num_episodes == 5
    assert args.rlt_toggle_key == "r"
    assert args.teleop_toggle_key == "space"
    assert args.start_with_teleop is False
    assert args.only_critical is False
    assert args.rtc is True
    assert args.rtc_execution_horizon == 10
    assert args.vla_rtc_execution_horizon == 25
    assert args.rtc_action_queue_size_to_get_new_actions == 30


def test_default_collect_full_mode_uses_r_key_as_episode_outcome():
    parser = build_parser()
    args = parser.parse_args(["collect",
        "--task",
        "test task", "--policy-path", "/tmp/ac", "--rlt-toggle-key", "r"])

    assert _collect_external_episode_outcome_key(args) == "r"


def test_default_collect_argv_matches_best_real_robot_rtc_chunks():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        default_episode_success=None,
        start_with_teleop=False,
        only_critical=False,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={
            "wrist": {
                "type": "opencv",
                "index_or_path": "/tmp/left-camera",
                "width": 640,
                "height": 480,
                "fps": 30,
                "fourcc": "MJPG",
            }
        },
        right_cameras={"wrist": {}, "front": {}},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--policy.phase_mode=manual" in argv
    assert "--rlt.enable=true" in argv
    assert "--rlt.rl_phase_key=r" in argv
    assert "--rlt.start_in_teleop=false" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" not in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" not in argv
    assert "--rlt.skip_prefix_recording=true" not in argv
    assert "--rlt.rtc_execution_horizon=10" in argv
    assert "--rlt.vla_rtc_execution_horizon=25" in argv
    assert "--rlt.rtc_action_queue_size_to_get_new_actions=30" in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--require_episode_success_label=true" in argv
    assert "--dataset.video_encoding_batch_size=6" in argv
    assert "--dataset.streaming_encoding=true" in argv
    assert "--policy_sync_to_teleop=true" in argv
    assert "--vla_ref=true" in argv


def test_default_collect_only_critical_starts_recording_on_first_r_and_ends_on_second_r():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        start_with_teleop=False,
        only_critical=True,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.skip_prefix_recording=true" in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" in argv
    assert "--rlt.start_in_teleop=false" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" not in argv
    assert "--enable_episode_outcome_labeling=true" in argv
    assert "--require_episode_success_label=true" in argv
    assert "--policy_sync_to_teleop=true" in argv


def test_default_collect_start_with_teleop_sets_episode_initial_source():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=5,
        episode_time_s=3000,
        fps=30,
        vcodec="h264",
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        start_with_teleop=True,
        only_critical=False,
    )
    setup = SimpleNamespace(
        followers=[{"port": "left"}, {"port": "right"}],
        left_cameras={},
        right_cameras={},
    )
    paths = SimpleNamespace(dataset_name="local/eval_vla_rlt_vla_123456", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(
        args=args,
        setup=setup,
        paths=paths,
        cal_dir="/tmp/cal",
        teleop_argv=["--teleop.type=bi_so_leader"],
    )

    assert "--rlt.start_in_teleop=true" in argv
    assert "--rlt.rl_phase_key_toggles_critical_phase=true" not in argv
    assert "--rlt.rl_phase_key_toggles_episode=true" not in argv


def test_full_vla_pedal_outcome_parser():
    parser = build_parser()
    args = parser.parse_args([
        "full",
        "--task",
        "test task",
        "--initial-source",
        "vla",
        "--policy-path",
        "/tmp/ac",
        "--vla-path",
        "/tmp/base.pt",
        "--phase-mode",
        "always_vla",
        "--chunk-exec-steps",
        "25",
        "--pedal-outcome",
        "--episode-outcome-key",
        "e",
        "--reset-time-s",
        "0",
    ])

    assert args.rtc is True
    assert args.pedal_outcome is True
    assert args.episode_outcome_key == "e"
    assert args.phase_mode == "always_vla"
    assert args.chunk_exec_steps == 50
    assert args.reset_time_s == 0


def test_full_vla_dry_run_accepts_headless_default_episode_success(tmp_path, capsys):
    for side in ("left", "right"):
        cal_dir = tmp_path / "calibration" / f"{side}_follower"
        cal_dir.mkdir(parents=True)
        (cal_dir / f"{side}_follower.json").write_text("{}")

    setup_json = tmp_path / "setup.json"
    setup_json.write_text(json.dumps(dual_arm_manifest(tmp_path)))

    parser = build_parser()
    args = parser.parse_args([
        "full",
        "--task",
        "test task",
        "--initial-source",
        "vla",
        "--policy-path",
        "/tmp/ac",
        "--setup-json",
        str(setup_json),
        "--dataset-tag",
        "headless_full",
        "--no-teleop",
        "--default-episode-success",
        "success",
        "--dry-run",
    ])

    runner.run_full(args)

    assert "--default_episode_success=success" in capsys.readouterr().out


def test_rlt_so101_dual_recording_does_not_import_lerobot_fork_only_modules():
    source_root = Path(__file__).parents[2] / "src" / "rlt_so101_dual"
    banned_imports = [
        "lerobot.scripts.lerobot_rlt_record",
        "lerobot.scripts.recording_hil",
        "lerobot.scripts.recording_loop",
        "lerobot.scripts.robot_config_loader",
        "lerobot.utils.recording_annotations",
        "lerobot.rl.acp_tags",
        "lerobot.policies.rlt",
    ]

    offenders = []
    for py_file in source_root.rglob("*.py"):
        text = py_file.read_text()
        for banned in banned_imports:
            if banned in text:
                offenders.append(f"{py_file.relative_to(source_root)}: {banned}")

    assert offenders == []


class _FakeLeaderBus:
    def __init__(self):
        self.calls = []

    def enable_torque(self):
        self.calls.append(("enable_torque",))

    def disable_torque(self):
        self.calls.append(("disable_torque",))

    def sync_write(self, data_name, values):
        self.calls.append(("sync_write", data_name, dict(values)))


class _FakeLeaderArm:
    def __init__(self):
        self.bus = _FakeLeaderBus()
        self.config = SimpleNamespace(port="/dev/fake")


def test_official_so_leader_feedback_is_sent_through_bus():
    from rlt_so101_dual.adapters.lerobot.record.hil import send_teleop_feedback, set_teleop_manual_control

    leader = _FakeLeaderArm()

    send_teleop_feedback(leader, {"shoulder_pan.pos": 1.0, "ignored": 2.0})
    send_teleop_feedback(leader, {"shoulder_lift.pos": 3.0})
    set_teleop_manual_control(leader, True)

    assert leader.bus.calls == [
        ("enable_torque",),
        ("sync_write", "Goal_Position", {"shoulder_pan": 1.0}),
        ("sync_write", "Goal_Position", {"shoulder_lift": 3.0}),
        ("disable_torque",),
    ]


def test_official_so_leader_connection_error_is_compatible_with_lerobot_v051():
    from rlt_so101_dual.adapters.lerobot.record.hil import send_teleop_feedback

    leader = _FakeLeaderArm()

    def fail_enable_torque():
        raise ConnectionError("no status packet")

    leader.bus.enable_torque = fail_enable_torque

    with pytest.raises(ConnectionError, match=r"leader arm on /dev/fake.*no status packet"):
        send_teleop_feedback(leader, {"shoulder_pan.pos": 1.0})


def test_official_bi_so_leader_feedback_splits_prefixed_actions():
    from rlt_so101_dual.adapters.lerobot.record.hil import send_teleop_feedback

    teleop = SimpleNamespace(left_arm=_FakeLeaderArm(), right_arm=_FakeLeaderArm())

    send_teleop_feedback(
        teleop,
        {
            "left_shoulder_pan.pos": 1.0,
            "right_elbow_flex.pos": 2.0,
            "action_is_pad": 0.0,
        },
    )

    assert teleop.left_arm.bus.calls == [
        ("enable_torque",),
        ("sync_write", "Goal_Position", {"shoulder_pan": 1.0}),
    ]
    assert teleop.right_arm.bus.calls == [
        ("enable_torque",),
        ("sync_write", "Goal_Position", {"elbow_flex": 2.0}),
    ]


def test_default_collect_argv_accepts_headless_default_episode_success():
    args = SimpleNamespace(
        policy_path="/tmp/ac",
        vla_path="/tmp/vla.pt",
        rl_token_path="/tmp/rlt",
        task="task",
        num_episodes=1,
        episode_time_s=10,
        fps=30,
        vcodec="h264",
        rtc=True,
        rtc_execution_horizon=10,
        vla_rtc_execution_horizon=25,
        rtc_max_guidance_weight=10.0,
        rtc_prefix_attention_schedule="EXP",
        rtc_action_queue_size_to_get_new_actions=30,
        vla_ref=True,
        play_sounds=True,
        rlt_toggle_key="r",
        teleop_toggle_key="space",
        default_episode_success="success",
        start_with_teleop=False,
        only_critical=False,
    )
    setup = SimpleNamespace(followers=[{"port": "left"}, {"port": "right"}], left_cameras={}, right_cameras={})
    paths = SimpleNamespace(dataset_name="local/test", dataset_root="/tmp/dataset")

    argv = build_default_collect_record_argv(args, setup, paths, "/tmp/cal", ["--teleop.type=bi_so_leader"])

    assert "--default_episode_success=success" in argv
    assert "--require_episode_success_label=true" in argv


def _visual_policy(*image_keys):
    """A stand-in policy declaring the given keys as VISUAL input features."""
    return SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                k: SimpleNamespace(type=SimpleNamespace(value="VISUAL")) for k in image_keys
            }
        )
    )


def _video_features(*keys):
    return {k: {"dtype": "video"} for k in keys}


PI05_SLOTS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
DUAL_RENAME = {
    f"observation.images.{k}": v for k, v in sc.PI05_CAMERA_MAP.items()
}
RAW_CAMS = tuple(f"observation.images.{k}" for k in sc.CAMERA_KEYS)


def test_image_feature_check_accepts_exact_match():
    _validate_policy_image_features(_visual_policy(*PI05_SLOTS), _video_features(*PI05_SLOTS))


def test_image_feature_check_accepts_renamed_dual_cameras():
    _validate_policy_image_features(
        _visual_policy(*PI05_SLOTS),
        _video_features(*RAW_CAMS),
        rename_map=DUAL_RENAME,
    )


def test_image_feature_check_rejects_dataset_with_no_images():
    """The --dataset.video=false symptom this check was originally written for."""
    with pytest.raises(ValueError, match="dataset.video=true"):
        _validate_policy_image_features(
            _visual_policy(*PI05_SLOTS),
            {"observation.state": {"dtype": "float32"}},
            rename_map=DUAL_RENAME,
        )


def test_image_feature_check_rejects_images_that_never_match_policy():
    """Images present, but no rename_map -- so nothing lines up with pi0.5's slots.

    Having *some* images is not enough: the renamed keys must actually overlap
    what the policy will read, or every camera is silently padded away and the
    robot runs blind.
    """
    with pytest.raises(ValueError):
        _validate_policy_image_features(
            _visual_policy(*PI05_SLOTS),
            _video_features(*RAW_CAMS),
        )


def test_image_feature_check_rejects_rename_map_with_wrong_targets():
    with pytest.raises(ValueError):
        _validate_policy_image_features(
            _visual_policy(*PI05_SLOTS),
            _video_features(*RAW_CAMS),
            rename_map={
                RAW_CAMS[0]: "observation.images.typo_rgb",
                RAW_CAMS[1]: "observation.images.also_wrong",
                RAW_CAMS[2]: "observation.images.still_wrong",
            },
        )


def test_image_feature_check_rejects_reversed_rename_map():
    """Mapping pi0.5's names back to raw aliases is the wrong direction."""
    with pytest.raises(ValueError):
        _validate_policy_image_features(
            _visual_policy(*PI05_SLOTS),
            _video_features(*RAW_CAMS),
            rename_map={v: k for k, v in DUAL_RENAME.items()},
        )


def test_full_parser_leaves_rename_map_unset_for_autodetection():
    args = build_parser().parse_args(
        ["full",
        "--task",
        "test task", "--initial-source", "vla", "--policy-path", "/tmp/ckpt"]
    )
    assert args.rename_map is None


def _write_checkpoint(tmp_path, visual_keys):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    **{k: {"type": "VISUAL", "shape": [3, 224, 224]} for k in visual_keys},
                    "observation.state": {"type": "STATE", "shape": [32]},
                }
            }
        )
    )
    return tmp_path


def test_rename_map_autodetected_for_plain_pi05_checkpoint(tmp_path):
    """vla_ft declares pi0.5 slot names, so raw dual cams must be renamed onto them."""
    ckpt = _write_checkpoint(tmp_path, PI05_SLOTS)
    assert resolve_camera_rename_map(ckpt) == {
        f"observation.images.{k}": v for k, v in sc.PI05_CAMERA_MAP.items()
    }


def test_rename_map_autodetect_skips_rlt_ac_checkpoint(tmp_path):
    """rlt_ac reads the raw aliases; renaming would map every camera away from it."""
    ckpt = _write_checkpoint(
        tmp_path, [f"observation.images.{k}" for k in sc.CAMERA_KEYS]
    )
    assert resolve_camera_rename_map(ckpt) == {}


def test_rename_map_autodetect_falls_back_when_config_unreadable(tmp_path):
    assert resolve_camera_rename_map(tmp_path / "nonexistent") == {}
    assert resolve_camera_rename_map(None) == {}


def test_explicit_rename_map_overrides_autodetection(tmp_path):
    ckpt = _write_checkpoint(tmp_path, PI05_SLOTS)
    assert resolve_camera_rename_map(ckpt, {}) == {}
    assert resolve_camera_rename_map(ckpt, {"a": "b"}) == {"a": "b"}


def _checkpoint(tmp_path, name, config):
    """A checkpoint directory holding just the config.json these tests read."""
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_pi05_checkpoint_does_not_receive_rlt_ac_only_overrides(tmp_path):
    """`--split-critical-phase` asks for phase_mode=manual, but a plain pi0.5
    config has no such field and draccus aborts the whole run over it:
    `record_full: error: unrecognized arguments: --phase_mode=manual`.
    """
    ckpt = _checkpoint(tmp_path, "pi05", {"type": "pi05", "chunk_size": 50})

    overrides = build_policy_overrides(
        policy_path=str(ckpt), vla_path=None, rl_token_path=None,
        phase_mode="manual", chunk_exec_steps=25,
    )

    assert overrides == [f"--policy.path={ckpt}"]


def test_rlt_ac_checkpoint_still_receives_them(tmp_path):
    ckpt = _checkpoint(
        tmp_path, "rlt_ac", {"type": "rlt_ac", "phase_mode": "always_rl", "chunk_exec_steps": 25},
    )

    overrides = build_policy_overrides(
        policy_path=str(ckpt), vla_path=None, rl_token_path=None,
        phase_mode="manual", chunk_exec_steps=10,
    )

    assert "--policy.phase_mode=manual" in overrides
    assert "--policy.chunk_exec_steps=10" in overrides


def test_unreadable_config_passes_the_override_through(tmp_path):
    """A missing config.json is a real case -- the 70000-step SFT checkpoint
    arrived without one. Dropping phase_mode there would leave an rlt_ac
    actor silently never invoked, so an unknown config must not suppress it.
    """
    missing = tmp_path / "not_there"
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{ not json")

    assert checkpoint_rejects_field(str(missing), "phase_mode") is False
    assert checkpoint_rejects_field(None, "phase_mode") is False
    assert checkpoint_rejects_field(str(broken), "phase_mode") is False

    overrides = build_policy_overrides(
        policy_path=str(missing), vla_path=None, rl_token_path=None, phase_mode="manual",
    )
    assert "--policy.phase_mode=manual" in overrides


def test_overrides_unaffected_when_no_rlt_ac_knob_was_requested(tmp_path):
    """Omitting phase_mode must not start consulting the checkpoint for it."""
    ckpt = _checkpoint(tmp_path, "pi05", {"type": "pi05"})

    overrides = build_policy_overrides(
        policy_path=str(ckpt), vla_path=None, rl_token_path=None,
    )

    assert overrides == [f"--policy.path={ckpt}"]


def _captured_bindings(monkeypatch, **listener_kwargs):
    """Run the patched init_keyboard_listener and return what each sub-listener got."""
    import lerobot.utils.control_utils as control_utils
    from rlt_so101_dual.adapters.lerobot.record import runner as record_runner

    monkeypatch.setattr(control_utils, "is_headless", lambda: False)
    monkeypatch.setattr(
        control_utils, "init_keyboard_listener", lambda *a, **k: (None, {}), raising=False,
    )
    if hasattr(control_utils.init_keyboard_listener, "_rlt_so101_dual_record_keys"):
        del control_utils.init_keyboard_listener._rlt_so101_dual_record_keys

    seen = {}
    monkeypatch.setattr(
        record_runner, "_start_record_event_keyboard_listener",
        lambda events, bindings: seen.setdefault("regular", bindings),
    )
    monkeypatch.setattr(
        record_runner, "_start_episode_outcome_keyboard_listener",
        lambda events, bindings: seen.setdefault("outcome", bindings),
    )
    record_runner._patch_record_keyboard_listener()
    control_utils.init_keyboard_listener(**listener_kwargs)
    return seen


def test_split_critical_phase_leaves_s_f_labelling_the_episode(monkeypatch):
    """r/u resolve the critical phase, so s/f must still reach the episode
    outcome listener -- that listener is the only thing that sets exit_early,
    and without it an episode runs to episode_time_s and silently takes
    default_episode_success.
    """
    seen = _captured_bindings(
        monkeypatch,
        rl_phase_key="r", rl_phase_failure_key="u",
        episode_success_key="s", episode_failure_key="f",
        end_success_key=None, end_failure_key=None,
    )

    # the listener stores the label, not the event name (removeprefix("episode_"))
    assert seen["outcome"] == {"s": "success", "f": "failure"}
    assert seen["regular"]["r"] == "start_rl_phase"
    assert seen["regular"]["u"] == "mark_rl_phase_failure"


def test_two_events_on_one_key_are_reported_not_silently_dropped(monkeypatch, caplog):
    """The online-RL path deliberately lets end_* win on s/f. It must at least
    say so: this collision is what cost a 30-episode run its episode labels.
    """
    with caplog.at_level("WARNING"):
        seen = _captured_bindings(
            monkeypatch,
            episode_success_key="s", episode_failure_key="f",
            end_success_key="s", end_failure_key="f",
        )

    assert seen["outcome"] == {}
    assert seen["regular"]["s"] == "end_phase_success"
    assert "claimed by both" in caplog.text


def _cfg(*, rlt_enable=False, toggles_cp=False, vla_model="", labeling=False):
    return SimpleNamespace(
        enable_critical_phase_labeling=labeling,
        rlt=SimpleNamespace(
            enable=rlt_enable,
            vla_model=vla_model,
            rl_phase_key_toggles_critical_phase=toggles_cp,
        ),
    )


def test_critical_phase_intervals_are_tracked_whenever_r_toggles_a_phase():
    """The r key toggling a critical phase is the reason to record where those
    phases were. Both `full --split-critical-phase` and rlt-so101-dual-online-train
    set this, and neither sets enable_critical_phase_labeling or rlt.vla_model --
    which is why the 30-episode baseline produced no intervals file.
    """
    from rlt_so101_dual.adapters.lerobot.record.backend import _should_track_critical_phase

    assert _should_track_critical_phase(_cfg(rlt_enable=True, toggles_cp=True), False) is True


def test_no_tracker_when_nothing_can_toggle_a_phase():
    from rlt_so101_dual.adapters.lerobot.record.backend import _should_track_critical_phase

    assert _should_track_critical_phase(_cfg(), False) is False
    # rlt on but the r key drives the episode, not a sub-phase (plain full)
    assert _should_track_critical_phase(_cfg(rlt_enable=True), False) is False
    # the toggle alone, with rlt off, is not enough -- no phase state machine runs
    assert _should_track_critical_phase(_cfg(toggles_cp=True), False) is False


def test_pre_existing_tracker_conditions_still_hold():
    from rlt_so101_dual.adapters.lerobot.record.backend import _should_track_critical_phase

    assert _should_track_critical_phase(_cfg(labeling=True), False) is True
    assert _should_track_critical_phase(_cfg(rlt_enable=True, vla_model="/p"), False) is True
    assert _should_track_critical_phase(_cfg(), True) is True


def test_missing_upstream_tracker_degrades_instead_of_crashing(tmp_path, caplog):
    """CriticalPhaseTracker lives in the upstream lerobot fork, not the 0.5.1 this
    project pins. Its import sat behind a condition that never fired until
    741c744 made the condition reachable -- turning dead code into a
    ModuleNotFoundError on the first real --split-critical-phase run.
    """
    from rlt_so101_dual.adapters.lerobot.record.backend import _build_phase_trackers

    with caplog.at_level("WARNING"):
        tracker, intervention = _build_phase_trackers(tmp_path / "intervals.json", True)

    assert tracker is None
    assert intervention is None
    assert "critical_phase_tracker" in caplog.text
    assert "will NOT be written" in caplog.text
