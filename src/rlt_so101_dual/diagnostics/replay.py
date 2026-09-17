# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Watch back what was just recorded, one episode at a time.

Read-only. It resolves the session for you rather than asking you to paste a
path: a stale path copied out of the docs is how an empty
`session_164946/` ended up inside a live session directory.

Episodes are not separate files. LeRobot concatenates a whole session into one
mp4 per camera and records each episode's span in `meta/episodes/`, so playing
"episode 7" means seeking to its timestamps -- which is what this does.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

from rlt_so101_dual.core import shape_contract as sc

DEFAULT_ROOT = Path.home() / "rlt_so101_dual" / "data" / "so101_dual"
CAMERAS = tuple(sc.CAMERA_KEYS)


def find_sessions(root: Path = DEFAULT_ROOT) -> list[Path]:
    """Sessions that have at least one finished episode, newest last.

    A session being recorded right now already has `meta/info.json` but no
    episode table until the first episode is saved, and picking it would fail
    for a reason that has nothing to do with what was asked.
    """
    # Match session_*, record_teleop_full_* (legacy), eval_* policy runs, etc.
    found = [
        p
        for p in root.glob("*/*")
        if p.is_dir()
        and (p / "meta" / "info.json").exists()
        and any((p / "meta").glob("episodes/**/*.parquet"))
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime)


def resolve_session(arg: str | None, root: Path = DEFAULT_ROOT) -> Path:
    if arg:
        session = Path(arg).expanduser()
        if not (session / "meta" / "info.json").exists():
            raise SystemExit(f"not a session (no meta/info.json): {session}")
        return session
    sessions = find_sessions(root)
    if not sessions:
        raise SystemExit(f"no session with a finished episode under {root}")
    return sessions[-1]


def episode_table(session: Path) -> "list[dict]":
    """Per-episode length, span in the video, outcome, and repeated frames."""
    import json

    import pandas as pd

    files = sorted((session / "meta").glob("episodes/**/*.parquet"))
    if not files:
        raise SystemExit(f"no episode metadata under {session}/meta/episodes")
    df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")

    # Repeated camera buffers, if this session was recorded with the counter.
    # Later rows win: a discarded attempt leaves its index behind for the retake.
    repeats: dict[int, dict] = {}
    sidecar = session / "camera_delivery.jsonl"
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                repeats[row["episode_index"]] = row

    out = []
    primary = CAMERAS[0]
    for _, r in df.iterrows():
        idx = int(r["episode_index"])
        rec = repeats.get(idx, {})
        cams = rec.get("cameras", {})
        from_key = f"videos/observation.images.{primary}/from_timestamp"
        to_key = f"videos/observation.images.{primary}/to_timestamp"
        out.append({
            "index": idx,
            "length": int(r["length"]),
            "success": str(r.get("episode_success", "?")),
            "from": float(r[from_key]),
            "to": float(r[to_key]),
            "aligned": rec.get("loop_samples") == int(r["length"]) if rec else None,
            "repeats": {c: cams.get(c, {}).get("repeated_buffers") for c in CAMERAS} if cams else {},
        })
    return out


def video_path(session: Path, camera: str) -> Path:
    key = f"observation.images.{camera}"
    files = sorted(glob.glob(str(session / "videos" / key / "**" / "*.mp4"), recursive=True))
    if not files:
        raise SystemExit(f"no video for {camera} under {session}/videos/{key}/")
    if len(files) > 1:
        print(f"note: {camera} has {len(files)} files; playing the first", file=sys.stderr)
    return Path(files[0])


def _escape(path: Path) -> str:
    """lavfi's movie= source treats these as syntax."""
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_command(session: Path, ep: dict, camera: str, speed: float) -> list[str]:
    start, span = ep["from"], ep["to"] - ep["from"]
    title = f"ep{ep['index']}  {ep['length']}f  {span:.1f}s  {ep['success']}"

    if camera != "both":
        return ["ffplay", "-hide_banner", "-loglevel", "error", "-autoexit",
                "-window_title", f"{title}  [{camera}]",
                "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
                "-vf", f"setpts={1 / speed:.4f}*PTS",
                str(video_path(session, camera))]

    # Side-by-side all contract cameras; each movie source seeks independently.
    parts = []
    labels = []
    for i, cam in enumerate(CAMERAS):
        path = video_path(session, cam)
        tag = f"c{i}"
        labels.append(f"[{tag}]")
        parts.append(
            f"movie={_escape(path)}:seek_point={start:.3f},"
            f"trim=duration={span:.3f},setpts=PTS-STARTPTS,"
            f"drawtext=text='{cam}':x=10:y=10:fontsize=24:fontcolor=white:"
            f"box=1:boxcolor=black@0.5[{tag}]"
        )
    stack = "".join(labels) + ("hstack=inputs=" + str(len(CAMERAS)))
    graph = ";".join(parts) + f";{stack},setpts={1 / speed:.4f}*PTS"
    return ["ffplay", "-hide_banner", "-loglevel", "error", "-autoexit",
            "-window_title", f"{title}  [{' | '.join(CAMERAS)}]", "-f", "lavfi", "-i", graph]


def build_viz_command(session: Path, index: int) -> list[str]:
    return ["lerobot-dataset-viz", "--repo-id", f"local/{session.name}",
            "--root", str(session), "--episode-index", str(index), "--num-workers", "0"]


def print_table(session: Path, rows: list[dict]) -> None:
    print(f"{session}\n{len(rows)} episodes\n")
    has_repeats = any(r["repeats"] for r in rows)
    head = f"{'ep':>4} {'frames':>7} {'sec':>6} {'outcome':>8}"
    if has_repeats:
        for cam in CAMERAS:
            head += f" {cam[:8]:>10}"
        head += f" {'sidecar':>9}"
    print(head)
    for r in rows:
        line = f"{r['index']:>4} {r['length']:>7} {r['to'] - r['from']:>6.1f} {r['success']:>8}"
        if has_repeats:
            for cam in CAMERAS:
                n = r["repeats"].get(cam)
                if n is None:
                    cell = "-"
                else:
                    cell = f"{n} ({100 * n / r['length']:.0f}%)"
                line += f" {cell:>10}"
            aligned = {True: "ok", False: "MISMATCH", None: "-"}[r["aligned"]]
            line += f" {aligned:>9}"
        print(line)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Play back a recorded episode. Read-only.",
        epilog="With no SESSION, the most recently written session is used.")
    p.add_argument("session", nargs="?", default=None,
                   help="session directory; defaults to the newest under data/so101_dual")
    p.add_argument("-e", "--episode", type=int, default=0, help="episode index (default 0)")
    p.add_argument("-c", "--camera", choices=[*CAMERAS, "both"], default="both",
                   help="which camera to show (default both, side by side)")
    p.add_argument("-s", "--speed", type=float, default=1.0, help="playback speed, e.g. 0.5")
    p.add_argument("-l", "--list", action="store_true", help="list the episodes and exit")
    p.add_argument("--viz", action="store_true",
                   help="open Rerun with images, action and state instead of the video")
    p.add_argument("--print-only", action="store_true", help="print the command without running it")
    args = p.parse_args(argv)

    session = resolve_session(args.session)
    rows = episode_table(session)

    if args.list:
        print_table(session, rows)
        return 0

    match = [r for r in rows if r["index"] == args.episode]
    if not match:
        available = ", ".join(str(r["index"]) for r in rows)
        raise SystemExit(f"no episode {args.episode} in {session}\navailable: {available}")
    ep = match[0]

    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    cmd = (build_viz_command(session, args.episode) if args.viz
           else build_command(session, ep, args.camera, args.speed))
    print(" ".join(cmd) if args.print_only else
          f"{session.name}  ep{ep['index']}  {ep['length']} frames  "
          f"{ep['to'] - ep['from']:.1f}s  {ep['success']}")
    if args.print_only:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
