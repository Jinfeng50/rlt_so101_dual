"""Build LeRobot robot/teleop configs from the hardware manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

from rlt_so101_dual.core import shape_contract as sc

logger = logging.getLogger(__name__)

def _find_arm_port(arms: list[dict], side: str) -> str:
    """Find the follower port for the given side from the arms list."""
    side_lower = side.lower()
    for arm in arms:
        alias = arm.get("alias", "").lower()
        arm_type = arm.get("type", "").lower()
        if side_lower in alias and ("follower" in alias or "follower" in arm_type):
            return arm["port"]
    raise ValueError(
        f"Cannot find {side} follower arm in setup.json. "
        f"Available arms: {[a.get('alias') for a in arms]}"
    )

def load_robot_config_from_json(path: str | Path) -> BiSOFollowerConfig:
    """Load a bimanual robot config from a roboclaw-compatible setup.json."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    arms = data.get("arms", [])
    cameras = data.get("cameras", [])
    robot_id = data.get("robot_id", data.get("id"))

    followers = [a for a in arms if "follower" in a.get("type", "").lower()]
    if len(followers) != 2:
        raise ValueError(
            f"Expected exactly 2 follower arms in {path}, got {len(followers)}. "
            "Use configs/hardware/so101_dual_manifest.json."
        )

    left_port = _find_arm_port(arms, "left")
    right_port = _find_arm_port(arms, "right")

    left_cams: dict[str, OpenCVCameraConfig] = {}
    right_cams: dict[str, OpenCVCameraConfig] = {}
    seen: list[str] = []

    for cam in cameras:
        alias = cam["alias"]
        if alias not in sc.CAMERA_KEYS:
            raise ValueError(
                f"Unknown camera alias {alias!r}. Expected one of {sc.CAMERA_KEYS}."
            )
        seen.append(alias)
        cam_cfg = OpenCVCameraConfig(
            index_or_path=cam["port"],
            fps=cam.get("fps", sc.FPS),
            width=cam.get("width", sc.IMAGE_HW[1]),
            height=cam.get("height", sc.IMAGE_HW[0]),
        )
        local = sc.CAMERA_ARM_LOCAL[alias]
        if sc.CAMERA_ARM_SIDE[alias] == "left":
            left_cams[local] = cam_cfg
        else:
            right_cams[local] = cam_cfg

    missing = [k for k in sc.CAMERA_KEYS if k not in seen]
    if missing:
        raise ValueError(f"Manifest is missing camera(s) {missing}. Required: {sc.CAMERA_KEYS}")

    logger.info(
        "Loaded robot config from %s: left_port=%s (%d cams), right_port=%s (%d cams)",
        path, left_port, len(left_cams), right_port, len(right_cams),
    )

    cfg = BiSOFollowerConfig(
        left_arm_config=SOFollowerConfig(port=left_port, cameras=left_cams),
        right_arm_config=SOFollowerConfig(port=right_port, cameras=right_cams),
    )
    if robot_id:
        cfg.id = robot_id
    return cfg
