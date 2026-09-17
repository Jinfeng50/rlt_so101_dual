"""Live camera preview for aiming the three dual-arm cameras.

The environment ships opencv-python-headless, so cv2.imshow() does not exist.
This serves frames as MJPEG over HTTP (also works over SSH without X):

    rlt-so101-dual-camera-preview --setup-json configs/hardware/so101_dual_manifest.json
    rlt-so101-dual-camera-preview --devices /dev/video0 /dev/video4 /dev/video2
    rlt-so101-dual-camera-preview --snapshot /tmp

Prefer ``--setup-json``: three UGREEN cameras share the same USB id, so
sysfs autodetection cannot tell them apart. Manifest indices are the
authoritative mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from rlt_so101_dual.core import shape_contract as sc

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETUP = _REPO_ROOT / "configs" / "hardware" / "so101_dual_manifest.json"
# Offline / dry-run placeholders when no manifest ports are available.
DEFAULT_DEVICES = {
    "left_wrist": "/dev/video0",
    "right_wrist": "/dev/video4",
    "right_front": "/dev/video2",
}
PI05_SIDE = 224

TARGET_BRIGHTNESS = 110.0
FLICKER_LIMIT = 4.0


def devices_from_setup(setup_path: Path) -> dict[str, str]:
    data = json.loads(setup_path.expanduser().read_text())
    out: dict[str, str] = {}
    for cam in data.get("cameras", []):
        alias = cam["alias"]
        if alias not in sc.CAMERA_KEYS:
            raise SystemExit(
                f"Unknown camera alias {alias!r} in {setup_path}; "
                f"expected {sc.CAMERA_KEYS}"
            )
        port = cam["port"]
        out[alias] = str(port) if isinstance(port, int) or str(port).isdigit() else str(port)
    missing = [k for k in sc.CAMERA_KEYS if k not in out]
    if missing:
        raise SystemExit(f"{setup_path} missing cameras {missing}")
    # Preserve contract order.
    return {k: out[k] for k in sc.CAMERA_KEYS}


def resolve_devices(
    explicit: list[str] | None,
    setup_json: Path | None,
) -> tuple[dict[str, str], str]:
    if explicit:
        if len(explicit) != len(sc.CAMERA_KEYS):
            raise SystemExit(
                f"--devices needs {len(sc.CAMERA_KEYS)} paths "
                f"({', '.join(sc.CAMERA_KEYS)}), got {len(explicit)}"
            )
        return dict(zip(sc.CAMERA_KEYS, explicit)), "given on the command line"

    path = (setup_json or DEFAULT_SETUP).expanduser()
    if path.exists():
        return devices_from_setup(path), f"setup-json {path}"

    aliases = {n: p for n, p in DEFAULT_DEVICES.items() if Path(p).exists()}
    if len(aliases) == len(DEFAULT_DEVICES):
        return {k: aliases[k] for k in sc.CAMERA_KEYS}, "default /dev/video* placeholders"

    raise SystemExit(
        "No cameras resolved.\n"
        f"  Pass --setup-json (default {DEFAULT_SETUP}), or\n"
        "  --devices <left_wrist> <right_wrist> <right_front>"
    )


def measure_fps(cap: cv2.VideoCapture, n: int = 60) -> float:
    for _ in range(10):
        cap.read()
    start = time.perf_counter()
    for _ in range(n):
        cap.read()
    return n / max(time.perf_counter() - start, 1e-9)


def exposure_ladder(mains_hz: int, fps: int) -> list[int]:
    period_units = round(1e4 / (2 * mains_hz))
    max_units = int(1e4 / fps)
    return [n * period_units for n in range(1, max_units // period_units + 1)]


def _sample(cap: cv2.VideoCapture, exposure: int, n: int = 20) -> tuple[float, float, float, float]:
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    for _ in range(8):
        cap.read()
    fps = measure_fps(cap, n=n)
    frames = []
    for _ in range(6):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame.astype(np.float32))
    if not frames:
        return fps, float("nan"), float("nan"), float("nan")
    flicker = (
        float(np.mean([np.abs(b - a).mean() for a, b in zip(frames, frames[1:])]))
        if len(frames) > 1
        else 0.0
    )
    last = frames[-1]
    return fps, float(last.mean()), float((last.max(axis=2) >= 250).mean()), flicker


def fix_exposure(
    path: str,
    exposure: int | None = None,
    target_brightness: float = TARGET_BRIGHTNESS,
    mains_hz: int = 50,
) -> dict:
    index_or_path: str | int = int(path) if str(path).isdigit() else path
    cap = cv2.VideoCapture(index_or_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open {path}. Another process may hold it:\n"
            f"    fuser -v {path}"
        )
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, sc.FPS)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        if exposure is not None:
            fps, brightness, blown, flicker = _sample(cap, exposure, n=45)
            return {
                "exposure": exposure, "fps": fps, "brightness": brightness,
                "blown": blown, "flicker": flicker, "sweep": [], "mains_hz": mains_hz,
            }

        sweep = []
        for value in exposure_ladder(mains_hz, sc.FPS):
            fps, brightness, blown, flicker = _sample(cap, value)
            sweep.append({
                "exposure": value, "fps": fps, "brightness": brightness,
                "blown": blown, "flicker": flicker,
            })

        eligible = [
            s for s in sweep
            if s["fps"] >= sc.FPS * 0.9 and s["brightness"] == s["brightness"]
        ]
        if not eligible:
            eligible = [s for s in sweep if s["brightness"] == s["brightness"]]
        best = min(
            eligible,
            key=lambda s: (abs(s["brightness"] - target_brightness), s["exposure"]),
        )

        fps, brightness, blown, flicker = _sample(cap, best["exposure"], n=45)
        return {
            "exposure": best["exposure"], "fps": fps, "brightness": brightness,
            "blown": blown, "flicker": flicker, "sweep": sweep, "mains_hz": mains_hz,
        }
    finally:
        cap.release()


def open_camera(path: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    index_or_path: str | int = int(path) if str(path).isdigit() else path
    cap = cv2.VideoCapture(index_or_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {path}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def resize_with_pad(frame: np.ndarray, side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(side / w, side / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    out = np.zeros((side, side, 3), dtype=frame.dtype)
    y = (side - resized.shape[0]) // 2
    x = (side - resized.shape[1]) // 2
    out[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return out


def annotate(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 28, 1)
    cv2.rectangle(out, (0, 0), (w - 1, 24), (0, 0, 0), -1)
    cv2.putText(out, f"{label}  {w}x{h}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    thumb = resize_with_pad(frame, PI05_SIDE)
    cv2.rectangle(thumb, (0, 0), (PI05_SIDE - 1, PI05_SIDE - 1), (0, 200, 0), 1)
    cv2.putText(thumb, "as pi0.5 sees it", (5, PI05_SIDE - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    pad = np.zeros((h, PI05_SIDE, 3), dtype=out.dtype)
    pad[: min(h, PI05_SIDE)] = thumb[: min(h, PI05_SIDE)]
    return np.hstack([out, pad])


class Cameras:
    def __init__(self, devices: dict[str, str], width: int, height: int, fps: int):
        self.devices = devices
        self.caps = {name: open_camera(path, width, height, fps) for name, path in devices.items()}
        self.lock = threading.Lock()

    def grab(self) -> np.ndarray | None:
        panes = []
        with self.lock:
            for name, cap in self.caps.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    return None
                panes.append(annotate(frame, f"{name}  ({self.devices[name]})"))
        height = max(p.shape[0] for p in panes)
        panes = [
            np.vstack([p, np.zeros((height - p.shape[0], p.shape[1], 3), dtype=p.dtype)])
            if p.shape[0] < height else p
            for p in panes
        ]
        return np.hstack(panes)

    def release(self) -> None:
        for cap in self.caps.values():
            cap.release()


def make_handler(cameras: Cameras):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path not in ("/", "/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    combined = cameras.grab()
                    if combined is None:
                        continue
                    ok, jpg = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--setup-json",
        type=Path,
        default=DEFAULT_SETUP,
        help="Dual-arm manifest (camera indices)",
    )
    p.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help=f"Override paths in order: {' '.join(sc.CAMERA_KEYS)}",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=sc.FPS)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--snapshot", type=Path, default=None)
    p.add_argument("--fix-exposure", action="store_true", default=False)
    p.add_argument("--target-brightness", type=float, default=TARGET_BRIGHTNESS)
    p.add_argument("--exposure", type=int, default=None)
    p.add_argument("--mains-hz", type=int, default=50, choices=(50, 60))
    args = p.parse_args()

    devices, how = resolve_devices(args.devices, args.setup_json)
    print(f"cameras ({how}):")
    for name, path in devices.items():
        print(f"  {name:<12} {path}")

    if args.fix_exposure:
        for name, path in devices.items():
            result = fix_exposure(path, args.exposure, args.target_brightness, args.mains_hz)
            print(f"\n  {name} ({path})")
            for s in result["sweep"]:
                mark = " <-- chosen" if s["exposure"] == result["exposure"] else ""
                slow = "" if s["fps"] >= sc.FPS * 0.9 else "  too slow"
                print(
                    f"      exposure {s['exposure']:>4} ({s['exposure'] / 10:>4.1f} ms)  "
                    f"{s['fps']:>5.1f} fps  brightness {s['brightness']:>5.1f}  "
                    f"blown {s['blown'] * 100:>4.1f}%  flicker {s['flicker']:>5.2f}"
                    f"{slow}{mark}"
                )
            print(
                f"      -> exposure {result['exposure']} ({result['exposure'] / 10:.1f} ms), "
                f"{result['fps']:.1f} fps, brightness {result['brightness']:.0f}, "
                f"{result['blown'] * 100:.1f}% blown out, flicker {result['flicker']:.2f}"
            )
        return

    cameras = Cameras(devices, args.width, args.height, args.fps)
    try:
        if args.snapshot is not None:
            args.snapshot.mkdir(parents=True, exist_ok=True)
            for name, cap in cameras.caps.items():
                for _ in range(int(2 * args.fps)):
                    cap.read()
                ok, frame = cap.read()
                if not ok or frame is None:
                    print(f"{name}: FAILED to read a frame")
                    continue
                out = args.snapshot / f"{name}.png"
                cv2.imwrite(str(out), frame)
                cv2.imwrite(
                    str(args.snapshot / f"{name}_pi05_224.png"),
                    resize_with_pad(frame, PI05_SIDE),
                )
                print(f"{name}: {frame.shape} -> {out}")
            return

        server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(cameras))
        print(f"Live preview on http://localhost:{args.port}   (Ctrl-C to stop)")
        print(f"  cameras: {', '.join(f'{k}={v}' for k, v in devices.items())}")
        print("  over SSH:  ssh -L {0}:localhost:{0} <user>@<host>".format(args.port))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    finally:
        cameras.release()


if __name__ == "__main__":
    sys.exit(main())
