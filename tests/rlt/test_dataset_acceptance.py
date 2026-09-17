"""Guards for the read-only session acceptance check.

The point of `rlt-so101-dual-check-dataset` is to fail before a bad session is
carried to another machine and trained on. These tests pin the conditions
it must refuse, because a checker that quietly passes everything is worse
than no checker -- it converts "nobody looked" into "it was verified".
"""

import json

import numpy as np
import pytest

from rlt_so101_dual.core import shape_contract as sc
from rlt_so101_dual.diagnostics import dataset_acceptance as da
from rlt_so101_dual.diagnostics.preflight import FAIL, PASS, WARN, Report


def _feature(shape, names=None, dtype="float32"):
    f = {"dtype": dtype, "shape": list(shape)}
    if names is not None:
        f["names"] = list(names)
    return f


def _good_info(**overrides):
    feats = {
        "action": _feature([sc.ACTION_DIM], sc.JOINT_NAMES),
        "observation.state": _feature([sc.PROPRIO_DIM], sc.JOINT_NAMES),
        "complementary_info.is_intervention": _feature([1]),
        "complementary_info.phase": _feature([1]),
        "complementary_info.state": _feature([1]),
        "complementary_info.collector_policy_id": _feature([1], dtype="int64"),
        "timestamp": _feature([1]),
        "frame_index": _feature([1], dtype="int64"),
        "episode_index": _feature([1], dtype="int64"),
    }
    for cam in sc.CAMERA_KEYS:
        feats[f"observation.images.{cam}"] = _feature(
            [*sc.IMAGE_HW, 3], ["height", "width", "channels"], dtype="video"
        )
    info = {
        "codebase_version": "v3.0",
        "robot_type": sc.ROBOT_TYPE,
        "fps": sc.FPS,
        "total_episodes": 10,
        "total_frames": 4676,
        "features": feats,
    }
    info.update(overrides)
    return info


def _statuses(rows):
    return [s for s, _, _ in rows]


class TestManifest:
    def test_a_good_manifest_passes(self, tmp_path):
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text(json.dumps(_good_info()))
        r = Report()
        info = da._load_info(r, tmp_path)
        assert info is not None
        assert FAIL not in _statuses(r.rows)

    def test_missing_info_json_is_fatal(self, tmp_path):
        r = Report()
        assert da._load_info(r, tmp_path) is None
        assert FAIL in _statuses(r.rows)

    def test_an_empty_session_fails(self, tmp_path):
        """The three aborted 0-episode runs on 2026-08-10 must never merge."""
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text(
            json.dumps(_good_info(total_episodes=0, total_frames=0))
        )
        r = Report()
        da._load_info(r, tmp_path)
        assert _statuses(r.rows).count(FAIL) == 2   # episodes and frames

    @pytest.mark.parametrize("field,value", [("robot_type", "so100"), ("fps", 25)])
    def test_contract_mismatch_fails(self, tmp_path, field, value):
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text(
            json.dumps(_good_info(**{field: value}))
        )
        r = Report()
        da._load_info(r, tmp_path)
        assert FAIL in _statuses(r.rows)


class TestSchema:
    def test_contract_conformant_schema_passes(self):
        r = Report()
        da._check_schema(r, _good_info())
        assert FAIL not in _statuses(r.rows)

    def test_wrong_action_width_fails(self):
        """A 6-dim (single-arm) action must not slip through the dual-arm checker."""
        info = _good_info()
        info["features"]["action"] = _feature([6], sc.JOINT_NAMES[:6])
        r = Report()
        da._check_schema(r, info)
        assert FAIL in _statuses(r.rows)

    def test_permuted_joint_names_fail(self):
        """Same width, wrong order: every column means a different joint."""
        info = _good_info()
        swapped = list(sc.JOINT_NAMES)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        info["features"]["action"] = _feature([sc.ACTION_DIM], swapped)
        r = Report()
        da._check_schema(r, info)
        assert FAIL in _statuses(r.rows)

    def test_missing_complementary_info_fails(self):
        """Native lerobot-record omits these; such a session is unusable."""
        info = _good_info()
        del info["features"]["complementary_info.is_intervention"]
        r = Report()
        da._check_schema(r, info)
        assert FAIL in _statuses(r.rows)

    def test_missing_camera_fails(self):
        info = _good_info()
        del info["features"]["observation.images.left_wrist"]
        r = Report()
        da._check_schema(r, info)
        assert FAIL in _statuses(r.rows)

    def test_wrong_resolution_fails(self):
        info = _good_info()
        info["features"]["observation.images.right_front"] = _feature(
            [720, 1280, 3], ["height", "width", "channels"], dtype="video"
        )
        r = Report()
        da._check_schema(r, info)
        assert FAIL in _statuses(r.rows)


class TestDuplicateAccounting:
    """The duplicate rate must separate a stalled camera from a still arm."""

    class _FakeDecoder:
        """Frames where every other one repeats, over a moving scene."""

        def __init__(self, n, period):
            rng = np.random.default_rng(0)
            base = rng.integers(0, 255, size=(n, 3, 64, 64), dtype=np.int16)
            for i in range(n):
                if period and i % period != 0:
                    base[i] = base[i - 1]
            self._f = base

        def __getitem__(self, s):
            class _A:
                def __init__(self, a): self._a = a
                def numpy(self): return self._a
            return _A(self._f[s])

    def test_every_other_frame_repeating_is_reported(self):
        r = Report()
        da._duplicate_report(r, self._FakeDecoder(400, 2), "right_front", 400, None, full=True)
        assert WARN in _statuses(r.rows)
        assert "49" in r.rows[0][1] or "50" in r.rows[0][1]

    def test_a_clean_stream_passes(self):
        r = Report()
        da._duplicate_report(r, self._FakeDecoder(400, 0), "right_front", 400, None, full=True)
        assert _statuses(r.rows) == [PASS]

    def test_motion_gating_is_reported_separately(self):
        """Repeats while the arm is parked must not read as a camera fault."""
        n = 400
        speed = np.zeros(n)
        speed[: n // 2] = 5.0                      # first half moving, second half still
        r = Report()
        da._duplicate_report(r, self._FakeDecoder(n, 2), "right_front", n, speed, full=True)
        detail = r.rows[0][2]
        assert "while the arm is moving" in detail


def test_thresholds_are_documented_constants():
    """Keep these two visible and stable.

    DUP_MAX_PIXEL_DIFF no longer decides anything a camera fault would trip --
    it cannot see through the encoder (tests/rlt/test_h264_roundtrip.py). It is
    pinned so that the constant and that test move together.
    """
    assert da.DUP_MAX_PIXEL_DIFF == 2
    assert da.MOTION_EPS == 0.2


def test_a_small_change_in_a_wide_view_is_not_a_duplicate():
    """One half of the check: a wide view with a small moving object.

    A mean-based cutoff called a frame a duplicate whenever most of the image
    held still, which is every frame of a wide-angle view of a small arm. This
    pins that the largest-per-pixel-change statistic does not repeat that
    mistake.

    It says nothing about the other half -- whether a genuine repeat is found --
    and for a long time nothing did. That gap is what let a metric which cannot
    fire be read as evidence that no frames repeated. See
    tests/rlt/test_h264_roundtrip.py.
    """
    n = 200
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, size=(3, 64, 64), dtype=np.int16)
    frames = np.repeat(base[None], n, axis=0)
    # One 4x4 patch moves: a large change, over a tiny part of the frame.
    for i in range(n):
        frames[i, :, 10:14, (i % 40):(i % 40) + 4] = 255

    class _Dec:
        def __getitem__(self, s):
            class _A:
                def __init__(self, a): self._a = a
                def numpy(self): return self._a
            return _A(frames[s])

    r = Report()
    da._duplicate_report(r, _Dec(), "right_front", n, None, full=True)
    assert _statuses(r.rows) == [PASS]
    assert r.rows[0][1].startswith("right_front: 0.0%")
