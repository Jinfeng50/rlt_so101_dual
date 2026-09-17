from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from lerobot.utils.constants import OBS_IMAGES

from rlt_so101_dual.core import shape_contract as sc

# Default dual-arm cell for this tree (override with --setup-json).
# common.py → record → lerobot → adapters → rlt_so101_dual → src → repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SETUP_PATH = _REPO_ROOT / "configs" / "hardware" / "so101_dual_manifest.json"
DEFAULT_DATASET_ROOT = Path.home() / "rlt_so101_dual" / "data" / "so101_dual"


def load_setup_json(path: str | None = None) -> dict[str, Any]:
    setup_path = Path(path).expanduser() if path else DEFAULT_SETUP_PATH
    with open(setup_path) as fh:
        return json.load(fh)


def resolve_dataset_root(setup: dict[str, Any]) -> Path:
    dataset_root = setup.get("datasets", {}).get("root", "")
    if not dataset_root:
        return DEFAULT_DATASET_ROOT
    return Path(dataset_root).expanduser()


def get_sorted_followers(setup: dict[str, Any]) -> list[dict[str, Any]]:
    followers = [arm for arm in setup["arms"] if "follower" in arm["type"]]
    followers.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return followers


def get_sorted_leaders(setup: dict[str, Any]) -> list[dict[str, Any]]:
    leaders = [arm for arm in setup["arms"] if "leader" in arm["type"]]
    leaders.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return leaders

log = logging.getLogger(__name__)

TELEOP_ID = "bimanual_leader"
FOLLOWER_ID = "bimanual"


@dataclass(frozen=True)
class RobotSetup:
    setup: dict[str, Any]
    followers: list[dict[str, Any]]
    leaders: list[dict[str, Any]]
    left_cameras: dict[str, Any]
    right_cameras: dict[str, Any]


@dataclass(frozen=True)
class RunPaths:
    dataset_name: str
    dataset_root: Path
    day_dir: Path
    log_file: Path


def load_robot_setup(setup_json: str | None) -> RobotSetup:
    """Parse a manifest into a *bimanual* SO101 robot setup.

    Expects 2 followers + 2 leaders and the three cameras in
    ``shape_contract.CAMERA_KEYS``. Any other layout is rejected.
    """
    setup = load_setup_json(setup_json)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)
    if len(followers) != 2:
        raise ValueError(
            f"Expected exactly 2 follower arms (bimanual SO101), got {len(followers)}. "
            "Use configs/hardware/so101_dual_manifest.json."
        )
    if len(leaders) != 2:
        raise ValueError(
            f"Expected exactly 2 leader arms (bimanual SO101), got {len(leaders)}."
        )
    left_cameras, right_cameras = build_camera_configs(setup.get("cameras", []))
    return RobotSetup(setup, followers, leaders, left_cameras, right_cameras)


def build_camera_configs(cameras: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split manifest cameras onto left/right BiSOFollower arm configs.

    Manifest aliases stay in ``shape_contract.CAMERA_KEYS``. Under each arm the
    local name is shortened (``left_wrist``→``wrist``, …); BiSOFollower then
    re-prefixes ``left_``/``right_`` so dataset keys match the contract.
    """
    left_cameras: dict[str, Any] = {}
    right_cameras: dict[str, Any] = {}
    seen_aliases: list[str] = []
    for camera in cameras:
        alias = camera["alias"]
        if alias not in sc.CAMERA_KEYS:
            raise ValueError(
                f"Unknown camera alias {alias!r}. Expected one of {sc.CAMERA_KEYS}; "
                "the alias becomes the dataset key and selects the pi0.5 camera slot."
            )
        seen_aliases.append(alias)
        camera_config: dict[str, Any] = {
            "type": "opencv",
            "index_or_path": camera["port"],
            "width": camera.get("width", 640),
            "height": camera.get("height", 480),
            "fps": camera.get("fps", sc.FPS),
        }
        if camera.get("fourcc"):
            camera_config["fourcc"] = camera["fourcc"]
        local_name = sc.CAMERA_ARM_LOCAL[alias]
        side = sc.CAMERA_ARM_SIDE[alias]
        if side == "left":
            left_cameras[local_name] = camera_config
        else:
            right_cameras[local_name] = camera_config
    missing = [k for k in sc.CAMERA_KEYS if k not in seen_aliases]
    if missing:
        raise ValueError(
            f"Manifest is missing camera(s) {missing}. All of {sc.CAMERA_KEYS} are "
            "required: the policy's camera list is fixed at training time."
        )
    return left_cameras, right_cameras


def resolve_run_paths(setup: dict[str, Any], dataset_tag: str, dataset_prefix: str) -> RunPaths:
    now = datetime.now()
    date_folder = f"{now:%m%d}_{dataset_tag}"
    dataset_leaf = f"{dataset_prefix}_{now:%H%M%S}"
    day_dir = resolve_dataset_root(setup) / date_folder
    dataset_root = day_dir / dataset_leaf
    return RunPaths(
        dataset_name=f"local/{dataset_leaf}",
        dataset_root=dataset_root,
        day_dir=day_dir,
        log_file=day_dir / f"{dataset_leaf}.log",
    )


def configure_logging(log_file: Path, log_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    # LeRobot logs a WARNING + full pformat dict on *every* frame where
    # max_relative_target clips even 0.1°. That floods the console and hides
    # keyboard events (r/s/f/space). Keep the safety clamp; just throttle the spam.
    _install_relative_goal_clamp_log_throttle()


_CLAMP_LOG_MSG = "Relative goal position magnitude had to be clamped"
_CLAMP_LOG_INTERVAL_S = 5.0
_clamp_log_state = {"last": 0.0, "suppressed": 0, "installed": False}


class _RelativeGoalClampThrottle(logging.Filter):
    """Allow at most one clamp WARNING every `_CLAMP_LOG_INTERVAL_S` seconds."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if _CLAMP_LOG_MSG not in msg:
            return True
        import time

        now = time.monotonic()
        last = float(_clamp_log_state["last"])
        if now - last < _CLAMP_LOG_INTERVAL_S:
            _clamp_log_state["suppressed"] = int(_clamp_log_state["suppressed"]) + 1
            return False
        suppressed = int(_clamp_log_state["suppressed"])
        _clamp_log_state["last"] = now
        _clamp_log_state["suppressed"] = 0
        if suppressed > 0:
            record.msg = (
                f"{record.msg}\n  … suppressed {suppressed} similar clamp warning(s) "
                f"in the last {_CLAMP_LOG_INTERVAL_S:.0f}s (max_relative_target still active)."
            )
            record.args = ()
        return True


def _install_relative_goal_clamp_log_throttle() -> None:
    if _clamp_log_state["installed"]:
        return
    filt = _RelativeGoalClampThrottle()
    root = logging.getLogger()
    root.addFilter(filt)
    # Also attach to handlers already created by basicConfig / FileHandler so
    # both console and file are throttled the same way.
    for handler in root.handlers:
        handler.addFilter(filt)
    _clamp_log_state["installed"] = True


def remove_existing_dataset(dataset_root: Path) -> None:
    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)


def stage_arm_calibration(arm: dict[str, Any], dst: Path) -> None:
    calibration_file = arm.get("calibration_file")
    if calibration_file:
        src = Path(calibration_file).expanduser()
    else:
        serial = Path(arm["calibration_dir"]).name
        src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
    if src.exists():
        shutil.copy2(src, dst)
        log.info("Calibration staged: %s -> %s", src, dst)
        return
    log.warning("Calibration file not found: %s", src)


def stage_follower_calibrations(followers: list[dict[str, Any]], cal_dir: str) -> None:
    if len(followers) != 2:
        raise ValueError(f"Expected 2 followers for bimanual staging, got {len(followers)}")
    for side, arm in (("left", followers[0]), ("right", followers[1])):
        stage_arm_calibration(arm, Path(cal_dir) / f"{FOLLOWER_ID}_{side}.json")


def build_teleop_argv(leaders: list[dict[str, Any]], no_teleop: bool) -> list[str]:
    if no_teleop:
        log.warning("Teleop disabled by --no-teleop")
        return []
    if len(leaders) != 2:
        log.warning("Teleop disabled: need 2 leader arms, got %d", len(leaders))
        return []
    log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    return [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        f"--teleop.id={TELEOP_ID}",
    ]


def stage_leader_calibrations(
    leaders: list[dict[str, Any]], teleop_argv: list[str]
) -> TemporaryDirectory[str] | None:
    if not teleop_argv:
        return None
    if len(leaders) != 2:
        raise ValueError(f"Expected 2 leaders for bimanual staging, got {len(leaders)}")
    leader_cal_dir = TemporaryDirectory(prefix="record-leader-cal-")
    for side, arm in (("left", leaders[0]), ("right", leaders[1])):
        stage_arm_calibration(arm, Path(leader_cal_dir.name) / f"{TELEOP_ID}_{side}.json")
    teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")
    return leader_cal_dir


def build_robot_argv(
    followers: list[dict[str, Any]],
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
    cal_dir: str,
    *,
    max_relative_target: float = 12.0,
) -> list[str]:
    if len(followers) != 2:
        raise ValueError(f"Expected 2 followers for bi_so_follower argv, got {len(followers)}")
    # max_relative_target is the last line of defense against a bad VLA/RL chunk
    # commanding an 80° jump in one control step (seen in online try3 recovery).
    # Docs (hardware.md / GUIDE) already require 12° for teleop; online was missing it.
    return [
        "--robot.type=bi_so_follower",
        f"--robot.id={FOLLOWER_ID}",
        f"--robot.calibration_dir={cal_dir}",
        f"--robot.left_arm_config.port={followers[0]['port']}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.max_relative_target={max_relative_target}",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={followers[1]['port']}",
        "--robot.right_arm_config.use_degrees=true",
        f"--robot.right_arm_config.max_relative_target={max_relative_target}",
        f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
    ]


PI05_SLOT_RENAME_MAP: dict[str, str] = {
    f"{OBS_IMAGES}.{alias}": slot for alias, slot in sc.PI05_CAMERA_MAP.items()
}


def resolve_camera_rename_map(
    policy_path: str | Path | None,
    explicit: dict[str, str] | None = None,
) -> dict[str, str]:
    """Decide whether the recording dataset's camera keys need renaming.

    Two kinds of checkpoint get loaded through the same `--policy-path`, and
    they want opposite things:

    * A plain pi0.5 VLA (`vla_ft`) declares pi0.5's own slot names
      (`base_0_rgb`, `left_wrist_0_rgb`, …), so the robot's raw camera
      aliases must be renamed onto them or the policy reads nothing.
    * An `rlt_ac` checkpoint declares the raw aliases (see
      `ChunkACPolicyConfig.validate_features`, which builds them from
      `shape_contract.CAMERA_KEYS`) and does its own mapping to pi0.5 slots
      inside `Pi05Adapter.camera_name_map`. Renaming here would map every
      camera *away* from what it reads.

    So the map is read off the checkpoint's own `config.json` rather than
    assumed. An explicit `--rename-map` (including an empty one) always wins;
    an unreadable or unrecognized config falls back to no renaming, leaving
    `_validate_policy_image_features` to report the mismatch.
    """
    if explicit is not None:
        return explicit
    if policy_path is None:
        return {}
    config_file = Path(policy_path).expanduser() / "config.json"
    try:
        features = json.loads(config_file.read_text()).get("input_features") or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    visual_keys = {
        key for key, spec in features.items()
        if isinstance(spec, dict) and spec.get("type") == "VISUAL"
    }
    # Raw aliases first: an rlt_ac checkpoint declares those and must not be
    # renamed, even though it also knows about the pi0.5 slot names.
    if visual_keys & set(PI05_SLOT_RENAME_MAP):
        return {}
    if visual_keys & set(PI05_SLOT_RENAME_MAP.values()):
        return dict(PI05_SLOT_RENAME_MAP)
    return {}


def checkpoint_rejects_field(policy_path: str | None, field: str) -> bool:
    """True only when the checkpoint's `config.json` is readable and lacks `field`.

    draccus rejects `--policy.<field>` outright when the loaded policy's config
    class has no such attribute, and the error names the bare flag rather than
    the policy that refused it:

        record_full: error: unrecognized arguments: --phase_mode=manual

    `phase_mode` and `chunk_exec_steps` live on `ChunkACPolicyConfig`; a plain
    pi0.5 SFT checkpoint declares neither, so `--split-critical-phase` used to
    abort before recording a single frame. Read the checkpoint instead of
    assuming, the same way `resolve_rename_map` decides camera renaming.

    The unreadable case deliberately answers False, i.e. pass the override
    through and let draccus decide. The asymmetry is the point: a config.json
    can genuinely be missing (the 70000-step SFT checkpoint arrived without
    one and it had to be back-filled), and silently dropping `phase_mode` on a
    real rlt_ac checkpoint would leave the actor never invoked while the run
    looks healthy -- the same silent-wrong shape as cd24167. An override the
    policy cannot parse fails loudly, immediately, and costs nothing.
    """
    if policy_path is None:
        return False
    config_file = Path(policy_path).expanduser() / "config.json"
    try:
        config = json.loads(config_file.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(config, dict) and field not in config


def build_dataset_argv(
    *,
    dataset_name: str,
    dataset_root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: int,
    fps: int,
    vcodec: str,
    rename_map: dict[str, str] | None = None,
) -> list[str]:
    argv = [
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.fps={fps}",
        f"--dataset.vcodec={vcodec}",
        "--dataset.push_to_hub=false",
        f"--dataset.video_encoding_batch_size={num_episodes + 1}",
        "--dataset.streaming_encoding=true",
    ]
    if rename_map:
        argv.append(f"--dataset.rename_map={json.dumps(rename_map)}")
    return argv


def build_policy_overrides(
    *,
    policy_path: str | None,
    vla_path: str | None,
    rl_token_path: str | None,
    phase_mode: str | None = None,
    chunk_exec_steps: int | None = None,
) -> list[str]:
    if policy_path is None:
        return []
    overrides = [f"--policy.path={policy_path}"]
    # Only forward the rlt_ac-only knobs when this checkpoint actually declares
    # them; see checkpoint_declares(). Log the drop rather than doing it
    # quietly -- a silently ignored --phase-mode on a real rlt_ac checkpoint
    # would mean the actor never runs, which is the failure mode cd24167 was
    # about.
    for field, value in (("phase_mode", phase_mode), ("chunk_exec_steps", chunk_exec_steps)):
        if value is None:
            continue
        if checkpoint_rejects_field(policy_path, field):
            logging.info(
                "Not passing --policy.%s=%s: %s/config.json declares no such field "
                "(a plain pi0.5 checkpoint has neither phase_mode nor chunk_exec_steps).",
                field, value, policy_path,
            )
            continue
        overrides.append(f"--policy.{field}={value}")
    if vla_path is not None:
        overrides.append(f"--policy.vla_pretrained_path={vla_path}")
    if rl_token_path is not None:
        overrides.append(f"--policy.rl_token_pretrained_path={rl_token_path}")
    return overrides


def build_rtc_argv(
    *,
    enabled: bool,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
    vla_execution_horizon: int | None,
    action_queue_size_to_get_new_actions: int | None,
) -> list[str]:
    argv = [
        f"--rlt.rtc_enabled={'true' if enabled else 'false'}",
        f"--rlt.rtc_execution_horizon={execution_horizon}",
        f"--rlt.rtc_max_guidance_weight={max_guidance_weight}",
        f"--rlt.rtc_prefix_attention_schedule={prefix_attention_schedule}",
    ]
    if vla_execution_horizon is not None:
        argv.append(f"--rlt.vla_rtc_execution_horizon={vla_execution_horizon}")
    if action_queue_size_to_get_new_actions is not None:
        argv.append(
            "--rlt.rtc_action_queue_size_to_get_new_actions="
            f"{action_queue_size_to_get_new_actions}"
        )
    return argv


def preflight_motor_connections(
    followers: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
    cal_dir: str,
    leader_cal_dir: str | None,
) -> None:
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollowerConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SOLeaderConfig

    def disconnect(device: Any) -> None:
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(device, arm_name, None)
            if arm is not None and arm.is_connected:
                arm.disconnect()
        if getattr(device, "is_connected", False):
            device.disconnect()

    if len(followers) != 2:
        raise ValueError(f"Expected 2 followers for motor preflight, got {len(followers)}")

    log.info("Preflight checking follower motor connections before loading policy")
    robot = BiSOFollower(
        BiSOFollowerConfig(
            id=FOLLOWER_ID,
            calibration_dir=Path(cal_dir),
            left_arm_config=SOFollowerConfig(port=followers[0]["port"], use_degrees=True),
            right_arm_config=SOFollowerConfig(port=followers[1]["port"], use_degrees=True),
        )
    )
    try:
        robot.connect(calibrate=True)
        log.info("Preflight follower motor check passed")
    finally:
        disconnect(robot)

    if not leaders or leader_cal_dir is None:
        return
    if len(leaders) != 2:
        raise ValueError(f"Expected 2 leaders for motor preflight, got {len(leaders)}")

    log.info("Preflight checking leader motor connections before loading policy")
    teleop = BiSOLeader(
        BiSOLeaderConfig(
            id=TELEOP_ID,
            calibration_dir=Path(leader_cal_dir),
            left_arm_config=SOLeaderConfig(port=leaders[0]["port"], use_degrees=True),
            right_arm_config=SOLeaderConfig(port=leaders[1]["port"], use_degrees=True),
        )
    )
    try:
        teleop.connect(calibrate=True)
        log.info("Preflight leader motor check passed")
    finally:
        disconnect(teleop)


def load_dataset_stats_from_pretrained(pretrained_path: str | Path) -> dict[str, dict[str, Any]] | None:
    """Load the (feature -> {stat_name: tensor}) dataset_stats dict bundled
    with a saved lerobot policy checkpoint's own preprocessor pipeline --
    i.e. the normalization the model was actually TRAINED with.

    For online RL (rlt_ac), the outer ChunkACPolicy wrapper is built fresh
    every session (no --policy.path of its own) against a brand-new,
    zero-episode dataset, so `make_pre_post_processors()`'s usual
    `dataset_stats` source (the recording dataset's own stats) is always
    empty. Without this, the frozen VLA -- which DOES expect properly
    normalized state/action, per its own saved
    `policy_preprocessor.json`/`*_normalizer_processor.safetensors` -- ends
    up fed effectively un-normalized (or default-normalized) observations
    despite loading the right weights, which reads as the VLA "acting
    randomly" even outside any RL involvement. This loads that checkpoint's
    real stats so they can be passed through instead.

    Returns None if the checkpoint has no normalizer_processor step (e.g. a
    non-lerobot-standard checkpoint, or one saved without normalization).
    """
    from safetensors.torch import load_file

    pretrained_path = Path(pretrained_path)
    preprocessor_json = pretrained_path / "policy_preprocessor.json"
    if not preprocessor_json.is_file():
        return None
    with open(preprocessor_json) as fh:
        spec = json.load(fh)
    state_file = next(
        (
            step.get("state_file")
            for step in spec.get("steps", [])
            if step.get("registry_name") == "normalizer_processor"
        ),
        None,
    )
    if not state_file:
        return None
    flat = load_file(str(pretrained_path / state_file))
    stats: dict[str, dict[str, Any]] = {}
    for key, tensor in flat.items():
        # Feature names themselves contain dots (observation.state,
        # observation.images.left_wrist); stat names (mean/std/min/max/
        # q01.../q99) never do, so the LAST dot always separates them.
        feature_name, stat_name = key.rsplit(".", 1)
        stats.setdefault(feature_name, {})[stat_name] = tensor
    return stats


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    _quiet_video_encoder_logs()


def _quiet_video_encoder_logs() -> None:
    """Suppress libx264's per-container startup banner (cpu caps, codec info)
    that streaming video encoding prints once per camera per episode chunk.
    PyAV's log callback is process-global, so setting it once here covers
    encoders created later in background threads too."""
    try:
        import av

        av.logging.set_level(av.logging.ERROR)
    except ImportError:
        pass
