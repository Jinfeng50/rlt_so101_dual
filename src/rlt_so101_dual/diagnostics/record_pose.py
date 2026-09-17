"""Read both followers' current poses and print --go-home-positions JSON.

`--go-home-positions` takes raw motor ticks. Pose both arms where you want
them to park, hold them, then:

    rlt-so101-dual-record-pose --setup-json configs/hardware/so101_dual_manifest.json

Prints a line you can paste into rlt-so101-dual-online-train. Arms are never
commanded -- this only reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rlt_so101_dual.adapters.lerobot.record.common import (
    FOLLOWER_ID,
    load_robot_setup,
    stage_follower_calibrations,
)
from rlt_so101_dual.core import shape_contract as sc


def read_raw_ticks(arm) -> dict[str, int]:
    """Raw tick per motor on one arm bus (no left_/right_ prefix)."""
    raw = arm.bus.sync_read("Present_Position", normalize=False)
    return {f"{motor}.pos": int(value) for motor, value in raw.items()}


def read_bimanual_ticks(robot) -> dict[str, int]:
    """Prefixed ticks matching ``shape_contract.JOINT_NAMES``."""
    out: dict[str, int] = {}
    for side, arm in (("left", robot.left_arm), ("right", robot.right_arm)):
        for key, value in read_raw_ticks(arm).items():
            out[f"{side}_{key}"] = value
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--setup-json",
        type=Path,
        default=Path("configs/hardware/so101_dual_manifest.json"),
    )
    p.add_argument("--compact", action="store_true", help="One line, ready to paste")
    args = p.parse_args(argv)

    from tempfile import TemporaryDirectory

    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollowerConfig

    setup = load_robot_setup(str(args.setup_json))
    with TemporaryDirectory(prefix="record-pose-cal-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        robot = BiSOFollower(
            BiSOFollowerConfig(
                id=FOLLOWER_ID,
                calibration_dir=Path(cal_dir),
                left_arm_config=SOFollowerConfig(
                    port=setup.followers[0]["port"], use_degrees=True
                ),
                right_arm_config=SOFollowerConfig(
                    port=setup.followers[1]["port"], use_degrees=True
                ),
            )
        )
        try:
            robot.connect(calibrate=False)
            ticks = read_bimanual_ticks(robot)
        finally:
            for arm_name in ("left_arm", "right_arm"):
                arm = getattr(robot, arm_name, None)
                if arm is not None and getattr(arm, "is_connected", False):
                    arm.disconnect()
            if getattr(robot, "is_connected", False):
                robot.disconnect()

    missing = sorted(set(sc.JOINT_NAMES) - set(ticks))
    if missing:
        print(f"warning: missing joints after read: {missing}", file=sys.stderr)

    print("\nCurrent bimanual pose, as raw motor ticks:\n")
    if args.compact:
        print(f"  --go-home-positions '{json.dumps(ticks, separators=(',', ':'))}'")
    else:
        print(f"  --go-home-positions '{json.dumps(ticks)}'")
    print(
        "\nPaste that into rlt-so101-dual-online-train. It is specific to this robot and "
        "this calibration -- re-record it after any re-calibration, and after moving "
        "to another machine."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
