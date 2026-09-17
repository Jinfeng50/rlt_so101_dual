"""CLI: hardware preflight checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from rlt_so101_dual.core import shape_contract as sc

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_MARK = {PASS: "\033[32m ok \033[0m", WARN: "\033[33mwarn\033[0m", FAIL: "\033[31mFAIL\033[0m"}

class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, what: str, detail: str = "") -> None:
        self.rows.append((status, what, detail))
        print(f"  [{_MARK[status]}] {what}" + (f"\n         {detail}" if detail else ""))

    def section(self, title: str) -> None:
        print(f"\n{title}")

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == WARN)

def check_environment(r: Report) -> None:
    r.section("Environment")

    v = sys.version_info
    r.add(PASS if v >= (3, 12) else FAIL, f"python {v.major}.{v.minor}.{v.micro}",
          "" if v >= (3, 12) else "requires-python is >=3.12")

    try:
        import torch

        arches = torch.cuda.get_arch_list()
        r.add(PASS, f"torch {torch.__version__}")
        if not torch.cuda.is_available():
            r.add(FAIL, "CUDA not available", "training and deployment both need a GPU")
        else:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            r.add(PASS, f"GPU {name} ({total:.1f} GiB)")
            cap = torch.cuda.get_device_capability(0)
            arch = f"sm_{cap[0]}{cap[1]}"
            if arch in arches:
                r.add(PASS, f"torch was built for this GPU ({arch})")
            else:
                r.add(FAIL, f"torch has no kernels for {arch}",
                      f"built for {arches}. A Blackwell card needs a +cu128 build.")
    except ImportError as exc:
        r.add(FAIL, "torch not importable", str(exc))

    try:
        import lerobot

        r.add(PASS, f"lerobot {getattr(lerobot, '__version__', '?')}")
    except ImportError as exc:
        r.add(FAIL, "lerobot not importable", str(exc))

    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        r.add(PASS, "torchcodec can load")
    except Exception as exc:
        r.add(FAIL, "torchcodec cannot load",
              f"{type(exc).__name__}: video encode and decode will both fail. "
              "conda install -c conda-forge ffmpeg=7")

    try:
        import cv2

        r.add(PASS, f"opencv {cv2.__version__}")
    except ImportError as exc:
        r.add(FAIL, "opencv not importable", str(exc))

    try:
        import scservo_sdk  # noqa: F401

        r.add(PASS, "feetech servo SDK")
    except ImportError:
        # lerobot[pi] does not pull this in; the SO101 bus needs lerobot[feetech].
        r.add(FAIL, "feetech servo SDK missing (scservo_sdk)",
              "nothing can talk to the arms without it. "
              'pip install "feetech-servo-sdk>=1.0.0,<2.0.0"')

    for var, why in [
        ("HF_HUB_OFFLINE", "stops a run from stalling on a Hub lookup mid-session"),
        ("TORCHDYNAMO_DISABLE", "torch.compile fights the real-time online-RL loop"),
    ]:
        if os.environ.get(var) == "1":
            r.add(PASS, f"{var}=1")
        else:
            r.add(WARN, f"{var} is not set", why)

def check_manifest(r: Report, setup_json: Path) -> dict | None:
    r.section(f"Manifest ({setup_json})")
    if not setup_json.exists():
        r.add(FAIL, "manifest not found")
        return None
    try:
        setup = json.loads(setup_json.read_text())
    except json.JSONDecodeError as exc:
        r.add(FAIL, "manifest is not valid JSON", str(exc))
        return None

    from rlt_so101_dual.adapters.lerobot.record.common import load_robot_setup

    try:
        parsed = load_robot_setup(str(setup_json))
    except ValueError as exc:
        r.add(FAIL, "manifest rejected", str(exc))
        return None

    r.add(PASS, f"{len(parsed.followers)} follower(s), {len(parsed.leaders)} leader(s)")
    if len(parsed.followers) != 2 or len(parsed.leaders) != 2:
        r.add(WARN, "expected bimanual 2+2 arms", "use configs/hardware/so101_dual_manifest.json")
    if not parsed.leaders:
        r.add(WARN, "no leader arm", "teleoperation and human intervention are unavailable")
    cam_keys = sorted([*(f"left_{k}" for k in parsed.left_cameras), *(f"right_{k}" for k in parsed.right_cameras)])
    r.add(PASS, f"cameras {cam_keys}")
    return setup

def check_serial(r: Report, setup: dict | None) -> None:
    r.section("Serial ports")
    if setup is None:
        return
    for arm in setup.get("arms", []):
        port = Path(os.path.expanduser(arm["port"]))
        alias = arm.get("alias", "?")
        if not port.exists():
            r.add(FAIL, f"{alias}: {port} missing",
                  "plug the arm in; ports must be /dev/serial/by-id/... as in "
                  "configs/hardware/so101_dual_manifest.json")
            continue
        readable = os.access(port, os.R_OK | os.W_OK)
        r.add(PASS if readable else FAIL, f"{alias}: {port}" + ("" if readable else " (no permission)"),
              "" if readable else "add yourself to the dialout group")

def check_calibration(r: Report, setup: dict | None) -> None:
    r.section("Calibration")
    if setup is None:
        return
    for arm in setup.get("arms", []):
        alias = arm.get("alias", "?")
        cal = arm.get("calibration_file")
        if not cal:
            r.add(WARN, f"{alias}: no calibration_file in the manifest")
            continue
        path = Path(os.path.expanduser(cal))
        if path.exists():
            r.add(PASS, f"{alias}: {path.name}")
        else:
            r.add(FAIL, f"{alias}: {path} missing",
                  "run lerobot-calibrate. Calibration lives outside the repo and does not "
                  "travel with git -- copy it when moving machines")

def check_cameras(r: Report, setup: dict | None) -> None:
    r.section("Cameras")
    if setup is None:
        return
    import cv2

    for cam in setup.get("cameras", []):
        alias = cam.get("alias", "?")
        port = cam.get("port")
        width, height = cam.get("width", 640), cam.get("height", 480)
        index_or_path: int | str
        if isinstance(port, int) or str(port).isdigit():
            index_or_path = int(port)
            label = f"/dev/video{index_or_path}" if isinstance(port, int) or str(port).isdigit() else str(port)
        else:
            path = Path(os.path.expanduser(str(port)))
            if not path.exists():
                r.add(FAIL, f"{alias}: {port} missing",
                      "update the index in configs/hardware/so101_dual_manifest.json")
                continue
            index_or_path = str(path)
            label = str(path)

        cap = cv2.VideoCapture(index_or_path, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                r.add(FAIL, f"{alias}: {label} cannot be opened",
                      "another process may be holding it, or the index moved after replug")
                continue
            fourcc = cam.get("fourcc", "YUYV")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if not ok or frame is None:
                r.add(FAIL, f"{alias}: opened but read no frame")
                continue
            got_h, got_w = frame.shape[:2]
            if (got_w, got_h) != (width, height):
                r.add(FAIL, f"{alias}: got {got_w}x{got_h}, manifest asks for {width}x{height}",
                      "a wrong node can still return a valid-looking frame at another size")
                continue

            from rlt_so101_dual.diagnostics.camera_preview import measure_fps

            want = cam.get("fps", sc.FPS)
            cap.set(cv2.CAP_PROP_FPS, want)
            measured = measure_fps(cap, n=45)
            brightness = float(frame.mean())
            if measured < want * 0.9:
                r.add(FAIL, f"{alias}: {measured:.1f} fps delivered, manifest asks for {want}",
                      "usually auto-exposure in dim light. Fix with: "
                      "rlt-so101-dual-camera-preview --fix-exposure  (the setting is "
                      "camera-side and survives, so it only needs running once per "
                      "power cycle)")
            elif brightness < 40:
                r.add(WARN, f"{alias}: {label} {got_w}x{got_h} {measured:.1f} fps, but very dark "
                            f"(mean {brightness:.0f})",
                      "a short exposure is the price of the full frame rate -- add light")
            else:
                r.add(PASS, f"{alias}: {label} {got_w}x{got_h} {measured:.1f} fps "
                            f"(brightness {brightness:.0f})")
        finally:
            cap.release()

def check_storage(r: Report, setup: dict | None, min_free_gb: float) -> None:
    r.section("Storage")
    root = Path(os.path.expanduser(
        (setup or {}).get("datasets", {}).get(
            "root", "~/rlt_so101_dual/data/so101_dual"
        )
    ))
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".preflight_write_test"
        probe.write_text("x")
        probe.unlink()
        r.add(PASS, f"dataset root writable: {root}")
    except OSError as exc:
        r.add(FAIL, f"dataset root not writable: {root}", str(exc))

    free_gb = shutil.disk_usage(root if root.exists() else Path.home()).free / 1024**3
    # 30 fps, three 640x480 h264 streams plus state.
    r.add(PASS if free_gb >= min_free_gb else WARN, f"{free_gb:.0f} GiB free",
          "" if free_gb >= min_free_gb else
          f"below {min_free_gb:.0f} GiB; a long session records roughly 1 GiB per 15 minutes")

def check_contract(r: Report) -> None:
    r.section("Shape contract")
    r.add(PASS, f"{sc.ACTION_DIM}-DoF {sc.ROBOT_TYPE} at {sc.FPS} fps")
    r.add(PASS, f"cameras {sc.CAMERA_KEYS} -> pi0.5 slots "
                f"{[sc.PI05_CAMERA_MAP[k].rsplit('.', 1)[-1] for k in sc.CAMERA_KEYS]}")
    r.add(PASS, f"RL chunk {sc.CHUNK_LENGTH} over a {sc.VLA_HORIZON}-step VLA horizon")

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--setup-json",
        type=Path,
        default=Path("configs/hardware/so101_dual_manifest.json"),
    )
    p.add_argument("--skip-cameras", action="store_true", help="No cameras attached yet")
    p.add_argument("--skip-arms", action="store_true", help="No arms attached yet")
    p.add_argument("--min-free-gb", type=float, default=50.0)
    args = p.parse_args(argv)

    r = Report()
    check_contract(r)
    check_environment(r)
    setup = check_manifest(r, args.setup_json)
    if not args.skip_arms:
        check_serial(r, setup)
        check_calibration(r, setup)
    if not args.skip_cameras:
        check_cameras(r, setup)
    check_storage(r, setup, args.min_free_gb)

    print()
    if r.failed:
        print(f"{r.failed} check(s) FAILED, {r.warned} warning(s). Fix the failures before starting.")
        return 1
    print(f"All checks passed ({r.warned} warning(s)). Good to start.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
