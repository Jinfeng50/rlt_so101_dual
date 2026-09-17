"""Read-only acceptance check for one recorded session.

A session that is malformed in a way nobody notices until VLA fine-tuning
costs a day of training-GPU time and a re-collection. Every check here is either
part of the data contract in ``core/shape_contract.py`` or something that
has actually gone wrong on this rig.

    rlt-so101-dual-check-dataset ~/rlt_so101_dual/data/so101_dual/<session>
    rlt-so101-dual-check-dataset <session> --full        # scan every frame, not a sample
    rlt-so101-dual-check-dataset <session> --skip-video  # metadata only, no decoding

This tool never writes, moves, renames or deletes anything. It opens the
session read-only and prints a verdict. Exit code is 1 if any check FAILs,
0 otherwise; WARNings never fail the run.

Run it twice: once on each raw session before transferring it, and once on
the merged dataset before training. ``lerobot-edit-dataset --operation.type
merge`` returning success is not evidence that the merge is usable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from rlt_so101_dual.core import shape_contract as sc
from rlt_so101_dual.diagnostics.preflight import FAIL, PASS, WARN, Report

# This threshold cannot detect a repeated frame in a recorded session, and the
# number it produces is not evidence about the cameras. It is kept only because
# removing it is a separate change; read the value below as "how often two
# decoded frames came out nearly equal", nothing more.
#
# Two things are true and were both established late:
#
# 1. The mean-based cutoff this replaced was worse. The top camera is
#    wide-angle, so the arm covers a small part of its frame and 33 ms of
#    genuine motion changes the mean by only ~2. A mean cutoff called 30% of
#    those frames duplicates.
# 2. The largest-per-pixel-change cutoff below does not fix the problem, it
#    only hides it. LeRobot encodes with `g=2`, so every other frame is an
#    I-frame, and two *byte-identical* input frames decode to images differing
#    by tens of grey levels. At a cutoff of 2 this check therefore reports
#    0.0% on every real session, whether or not frames were repeated.
#    tests/rlt/test_h264_roundtrip.py pins that on known-duplicate input.
#
#    Raising the cutoff does not rescue it either: on session 0810_dup_probe
#    (top camera, 1238 pairs) duplicates measured 46-58 while genuine motion
#    in the quiet parts of an episode fell to 16, so the two overlap. That
#    measurement is from recorded footage, not from the test above.
#
# The measurement that answers the question runs during recording, before
# encoding: see adapters/lerobot/record/frame_delivery.py, which compares the
# object the camera handed over.
DUP_MAX_PIXEL_DIFF = 2

# Below this total joint movement (sum of absolute per-joint deltas, in the
# same units as observation.state) the arm is treated as stationary. A
# stationary arm produces near-identical frames legitimately, so counting
# those as duplicates overstates the problem -- the motion-gated rate is the
# number that actually means "the camera failed to deliver a new frame".
MOTION_EPS = 0.2

REQUIRED_FEATURES = [
    "action",
    "observation.state",
    "complementary_info.is_intervention",
    "complementary_info.phase",
    "complementary_info.state",
    "complementary_info.collector_policy_id",
    "timestamp",
    "frame_index",
    "episode_index",
]


def _load_info(r: Report, session: Path) -> dict | None:
    r.section(f"Manifest  ({session})")
    info_path = session / "meta" / "info.json"
    if not info_path.exists():
        r.add(FAIL, "meta/info.json not found",
              "this is not a LeRobot session root; point at the "
              "session_<HHMMSS> directory (or legacy record_teleop_full_*), not its parent")
        return None
    try:
        info = json.loads(info_path.read_text())
    except json.JSONDecodeError as exc:
        r.add(FAIL, "meta/info.json is not valid JSON", str(exc))
        return None

    eps, frames = info.get("total_episodes", 0), info.get("total_frames", 0)
    r.add(PASS if eps > 0 else FAIL, f"total_episodes = {eps}",
          "" if eps > 0 else "an empty session must not be transferred or merged")
    r.add(PASS if frames > 0 else FAIL, f"total_frames = {frames}"
          + (f"  ({frames / max(info.get('fps', 1), 1):.1f} s)" if frames else ""))

    robot = info.get("robot_type")
    r.add(PASS if robot == sc.ROBOT_TYPE else FAIL, f"robot_type = {robot}",
          "" if robot == sc.ROBOT_TYPE else f"the contract requires {sc.ROBOT_TYPE}")
    fps = info.get("fps")
    r.add(PASS if fps == sc.FPS else FAIL, f"fps = {fps}",
          "" if fps == sc.FPS else f"the contract requires {sc.FPS}")
    r.add(PASS, f"codebase_version = {info.get('codebase_version')}")
    return info


def _check_schema(r: Report, info: dict) -> None:
    r.section("Schema against the data contract")
    feats = info.get("features", {})

    missing = [k for k in REQUIRED_FEATURES if k not in feats]
    if missing:
        r.add(FAIL, f"{len(missing)} required feature(s) missing", ", ".join(missing))
    else:
        r.add(PASS, f"all {len(REQUIRED_FEATURES)} required features present")

    for key in ("action", "observation.state"):
        f = feats.get(key)
        if f is None:
            continue
        shape = list(f.get("shape", []))
        ok = shape == [sc.ACTION_DIM]
        r.add(PASS if ok else FAIL, f"{key} shape {shape}",
              "" if ok else f"the contract requires [{sc.ACTION_DIM}]")
        names = list(f.get("names") or [])
        ok = names == sc.JOINT_NAMES
        r.add(PASS if ok else FAIL, f"{key} joint names",
              "" if ok else f"got {names}\n         want {sc.JOINT_NAMES}\n"
                            "         order matters: it fixes which column is which joint")

    for cam in sc.CAMERA_KEYS:
        key = f"observation.images.{cam}"
        f = feats.get(key)
        if f is None:
            r.add(FAIL, f"{key} missing",
                  f"the contract requires both of {sc.CAMERA_KEYS}")
            continue
        shape = tuple(f.get("shape", []))[:2]
        want = sc.IMAGE_HW
        ok = shape == want
        r.add(PASS if ok else FAIL, f"{key} {shape[0]}x{shape[1]}"
              if len(shape) == 2 else f"{key} shape {shape}",
              "" if ok else f"the contract requires {want[0]}x{want[1]}")


def _check_episodes(r: Report, session: Path) -> tuple[list[str], int] | None:
    """Returns (per-episode success labels, expected total frames)."""
    import pandas as pd

    r.section("Episode labels and task text")
    files = sorted((session / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        r.add(FAIL, "meta/episodes/**.parquet not found")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    if "episode_success" not in df.columns:
        r.add(FAIL, "episode_success column missing",
              "this session was recorded without success labels and cannot be "
              "used for supervised training")
        return None

    labels = df["episode_success"].astype(str).tolist()
    unlabeled = [i for i, v in enumerate(labels) if v in ("", "None", "nan", "<NA>")]
    if unlabeled:
        r.add(FAIL, f"{len(unlabeled)} episode(s) unlabeled", f"indices {unlabeled}")
    else:
        r.add(PASS, f"all {len(labels)} episodes labeled")

    failures = [i for i, v in enumerate(labels) if v != "success"]
    if failures:
        r.add(WARN, f"{len(failures)} episode(s) not marked success", f"indices {failures}\n"
              "         the demo dataset trains VLA and the RL token by supervised "
              "learning, so it must contain successes only. Drop these before merging.")
    else:
        r.add(PASS, f"all {len(labels)} episodes are successes")

    if "tasks" in df.columns:
        tasks = {t for row in df["tasks"] for t in (row if isinstance(row, (list, np.ndarray)) else [row])}
        if len(tasks) == 1:
            r.add(PASS, f'task text is uniform: "{next(iter(tasks))}"')
        else:
            r.add(FAIL, f"{len(tasks)} distinct task strings in one session",
                  "\n         ".join(sorted(str(t) for t in tasks))
                  + "\n         every stage from SFT to online RL must use the same "
                    "string verbatim; differing text splits the dataset's task index")

    lengths = df["length"].tolist() if "length" in df.columns else []
    if lengths:
        r.add(PASS, f"episode lengths {min(lengths)}-{max(lengths)} frames "
                    f"({min(lengths) / sc.FPS:.1f}-{max(lengths) / sc.FPS:.1f} s)")
    return labels, int(sum(lengths))


def _check_frames(r: Report, session: Path) -> "np.ndarray | None":
    """Validates the per-frame table. Returns per-frame joint speed for reuse."""
    import pandas as pd

    r.section("Frame table")
    files = sorted((session / "data").rglob("*.parquet"))
    if not files:
        r.add(FAIL, "data/**.parquet not found")
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    r.add(PASS, f"{len(df)} rows across {len(files)} parquet file(s)")

    bad = []
    for key in ("action", "observation.state"):
        if key not in df.columns:
            continue
        arr = np.stack(df[key].to_numpy())
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            bad.append(f"{key}: {n_nan} NaN, {n_inf} Inf")
    if bad:
        r.add(FAIL, "non-finite values present", "; ".join(bad))
    else:
        r.add(PASS, "action and observation.state are finite everywhere")

    # frame_index must restart at 0 each episode and increase by one; a gap
    # means frames were dropped between the capture loop and the writer.
    gaps, ts_bad = [], []
    for ep, g in df.groupby("episode_index"):
        fi = g["frame_index"].to_numpy()
        if not np.array_equal(fi, np.arange(len(fi))):
            gaps.append(int(ep))
        ts = g["timestamp"].to_numpy().astype(float)
        if len(ts) > 1 and not np.all(np.diff(ts) > 0):
            ts_bad.append(int(ep))
    r.add(PASS if not gaps else FAIL,
          "frame_index contiguous in every episode" if not gaps
          else f"frame_index has gaps in episode(s) {gaps}")
    r.add(PASS if not ts_bad else FAIL,
          "timestamp strictly increasing in every episode" if not ts_bad
          else f"timestamp not monotonic in episode(s) {ts_bad}")

    # LeRobot synthesises timestamp as frame_index / fps, so a perfectly even
    # spacing here is not evidence that the capture loop kept time. Say so
    # rather than letting the PASS above imply more than it does.
    r.add(WARN, "timestamps are synthesised from frame_index, not measured",
          "an even spacing here cannot detect a camera that under-delivered, and "
          "neither can the duplicate-frame check below (it cannot see through the "
          "encoder). The recording-time counter in camera_delivery.jsonl is the "
          "only measurement that can; this session has none if it predates it.")

    if "observation.state" in df.columns:
        S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        speed = np.abs(np.diff(S, axis=0)).sum(axis=1)
        return speed
    return None


def _check_videos(r: Report, session: Path, expected_frames: int,
                  speed: "np.ndarray | None", full: bool) -> None:
    from torchcodec.decoders import VideoDecoder

    r.section("Video streams")
    for cam in sc.CAMERA_KEYS:
        key = f"observation.images.{cam}"
        files = sorted((session / "videos" / key).rglob("*.mp4"))
        if not files:
            r.add(FAIL, f"{cam}: no mp4 under videos/{key}/")
            continue
        if len(files) > 1:
            r.add(WARN, f"{cam}: {len(files)} video files",
                  "duplicate-frame sampling below only reads the first")

        dec = VideoDecoder(str(files[0]))
        n = dec.metadata.num_frames
        total = sum(VideoDecoder(str(f)).metadata.num_frames for f in files)
        ok = total == expected_frames
        r.add(PASS if ok else FAIL,
              f"{cam}: {total} decoded frames vs {expected_frames} parquet rows",
              "" if ok else "a mismatch means the video and the state table are "
                            "misaligned; every training sample after the first "
                            "gap pairs the wrong image with the wrong action")

        # Decoding the first and the last frame proves the stream is readable
        # end to end. A truncated file (power loss during finalize) still
        # reports a plausible frame count in its header.
        try:
            _ = dec[0]
            _ = dec[n - 1]
            r.add(PASS, f"{cam}: first and last frame decode")
        except Exception as exc:
            r.add(FAIL, f"{cam}: cannot decode first/last frame",
                  f"{type(exc).__name__}: {exc}")
            continue

        _duplicate_report(r, dec, cam, n, speed, full)


def _duplicate_report(r: Report, dec, cam: str, n: int,
                      speed: "np.ndarray | None", full: bool) -> None:
    """Fraction of frames that merely repeat the previous one.

    The capture loop takes whatever the camera thread last produced rather
    than waiting for a new frame, so a camera that misses the loop deadline
    silently contributes the previous frame again. Nothing else in the
    pipeline notices: the frame count, the declared fps and the wall-clock
    rate all stay correct.
    """
    if full:
        lo, hi = 0, n
    else:
        # A contiguous window from the middle, where the arm is in the thick
        # of the task rather than parked at the start pose.
        want = min(900, n)
        lo = max(0, (n - want) // 2)
        hi = lo + want

    peaks, means = [], []
    B = 300
    prev = None
    for s in range(lo, hi, B):
        # Full resolution: subsampling pixels can step over the only region
        # that moved, which is exactly the failure the mean-based check had.
        blk = dec[s:min(s + B, hi)].numpy().astype(np.int16)
        if prev is not None:
            blk = np.concatenate([prev[None], blk], 0)
        d = np.abs(np.diff(blk, axis=0))
        peaks.append(d.reshape(len(d), -1).max(axis=1))
        means.append(d.mean(axis=(1, 2, 3)))
        prev = blk[-1]
    peak = np.concatenate(peaks)
    mean = np.concatenate(means)
    dup = peak <= DUP_MAX_PIXEL_DIFF
    frac = float(dup.mean())

    scope = "all frames" if full else f"frames {lo}-{hi}"
    detail = (f"largest per-pixel change: p10={np.percentile(peak, 10):.0f} "
              f"p50={np.percentile(peak, 50):.0f}   scene activity (mean change): "
              f"p50={np.percentile(mean, 50):.2f}  ({scope})")

    # Repeats while the arm is parked are expected and harmless; repeats while
    # it is moving are the camera failing to keep up. Separating the two is
    # what distinguishes a real defect from a still scene.
    # `speed` is one shorter than the frame count (it is a diff), so the last
    # window has no speed sample. Requiring len(speed) >= hi silently skipped
    # the whole gate whenever the window ran to the end of the video, which is
    # exactly what --full does.
    moving_note = ""
    if speed is not None and len(speed) >= hi - 1:
        mv = speed[lo:hi - 1][: len(dup)] > MOTION_EPS
        if mv.sum() >= 30:
            moving_note = (f"\n         while the arm is moving: "
                           f"{100 * dup[mv].mean():.1f}% "
                           f"({int(mv.sum())} frames sampled)")

    # Neither branch is evidence about the cameras: after encoding, a repeated
    # frame is indistinguishable from a moving one (see DUP_MAX_PIXEL_DIFF).
    # Both lines say what was measured and point at the one that can answer it.
    cannot_tell = ("\n         This cannot detect a repeated frame: the encoder "
                   "makes identical inputs differ by more than motion does. "
                   "camera_delivery.jsonl in this session is the measurement "
                   "that can, and it is written during recording.")
    if frac >= 0.05:
        r.add(WARN, f"{cam}: {100 * frac:.1f}% of decoded frame pairs are near-identical",
              detail + moving_note + cannot_tell)
    else:
        r.add(PASS, f"{cam}: {100 * frac:.1f}% of decoded frame pairs are near-identical",
              detail + moving_note + cannot_tell)


def _check_loadable(r: Report, session: Path, repo_id: str | None) -> None:
    r.section("LeRobotDataset load")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rid = repo_id or f"local/{session.name}"
    try:
        ds = LeRobotDataset(repo_id=rid, root=session, revision="main",
                            tolerance_s=0.04, video_backend="pyav")
    except Exception as exc:
        r.add(FAIL, "LeRobotDataset could not open the session",
              f"{type(exc).__name__}: {exc}")
        return
    r.add(PASS, f"opened as {rid}: {ds.num_episodes} episodes, {ds.num_frames} frames")

    # The first frame, both sides of every episode boundary, and the last
    # frame. Boundaries are where video seeking and the episode index have to
    # agree, and they are where a truncated stream shows up first.
    probes = [0, ds.num_frames - 1]
    try:
        # v3 keeps the row range per episode on the episodes table; the v2
        # `episode_data_index` attribute is gone.
        eps = ds.meta.episodes
        lo = eps["dataset_from_index"]
        hi = eps["dataset_to_index"]
        for ep in range(ds.num_episodes):
            probes += [int(lo[ep]), int(hi[ep]) - 1]
        r.add(PASS, f"episode row ranges resolved for all {ds.num_episodes} episodes")
    except Exception as exc:
        r.add(WARN, "episode row ranges unavailable; probing endpoints only",
              f"{type(exc).__name__}: {exc}")

    bad = []
    for i in sorted(set(p for p in probes if 0 <= p < ds.num_frames)):
        try:
            item = ds[i]
            for cam in sc.CAMERA_KEYS:
                k = f"observation.images.{cam}"
                if k not in item:
                    bad.append(f"frame {i}: {k} missing")
                    continue
                if tuple(item[k].shape[-2:]) != sc.IMAGE_HW:
                    bad.append(f"frame {i}: {k} is {tuple(item[k].shape)}")
            if tuple(item["observation.state"].shape) != (sc.ACTION_DIM,):
                bad.append(f"frame {i}: state is {tuple(item['observation.state'].shape)}")
        except Exception as exc:
            bad.append(f"frame {i}: {type(exc).__name__}: {exc}")

    if bad:
        r.add(FAIL, f"{len(bad)} probe(s) failed", "\n         ".join(bad[:8]))
    else:
        r.add(PASS, f"{len(set(probes))} probe frames load with both cameras "
                    f"and a {sc.ACTION_DIM}-dim state")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path,
                   help="a session_<HHMMSS> directory (or legacy record_teleop_full_*), or a merged dataset root")
    p.add_argument("--repo-id", default=None,
                   help="defaults to local/<directory name>")
    p.add_argument("--full", action="store_true",
                   help="scan every frame for duplicates instead of a 900-frame window")
    p.add_argument("--skip-video", action="store_true",
                   help="metadata only; skips decoding entirely")
    args = p.parse_args(argv)

    session = args.session.expanduser().resolve()
    r = Report()

    info = _load_info(r, session)
    if info is None:
        print("\n1 check FAILED. Nothing else could be checked.")
        return 1

    _check_schema(r, info)
    eps = _check_episodes(r, session)
    speed = _check_frames(r, session)

    if not args.skip_video:
        expected = info.get("total_frames", 0)
        if eps is not None and eps[1] and eps[1] != expected:
            r.add(WARN, f"episode lengths sum to {eps[1]} but info.json says {expected}")
        _check_videos(r, session, expected, speed, args.full)
        _check_loadable(r, session, args.repo_id)
    else:
        r.section("Video streams")
        r.add(WARN, "skipped (--skip-video)",
              "decoding is the only check that can catch a truncated stream "
              "or a camera that under-delivered")

    print()
    if r.failed:
        print(f"FAIL — {r.failed} failed, {r.warned} warning(s). "
              f"Do not transfer or merge this session until they are resolved.")
        return 1
    print(f"PASS — 0 failed, {r.warned} warning(s).")
    if r.warned:
        print("Warnings do not block, but read them before committing to a "
              "600-episode collection run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
