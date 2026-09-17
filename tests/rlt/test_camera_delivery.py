"""The recording-time repeated-buffer counter.

What makes this measurement trustworthy is that it has no threshold: a frame
either is the same object the loop already wrote, or it is not. These tests pin
both directions -- a real repeat is counted, and a fresh frame is not -- which
is the check the pixel-based duplicate metric never had.
"""

import json

import numpy as np

from rlt_so101_dual.adapters.lerobot.record import loop
from rlt_so101_dual.adapters.lerobot.record.frame_delivery import (
    SIDECAR_NAME,
    FrameDeliveryCounter,
    append_sidecar,
    frame_digest,
    persist,
    resolve_episode_index,
)

CAMS = ["left_wrist", "right_wrist", "right_front"]


def _frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)


class TestObjectIdentity:
    def test_the_same_object_twice_is_a_repeated_buffer(self):
        """read_latest() returns latest_frame without copying, so this is the
        exact shape of a camera that missed the loop deadline."""
        c = FrameDeliveryCounter()
        stale = _frame(1)
        c.observe({"right_front": stale, "left_wrist": _frame(2)}, CAMS)
        c.observe({"right_front": stale, "left_wrist": _frame(3)}, CAMS)

        assert c.repeated_buffers["right_front"] == 1
        assert c.repeated_buffers["left_wrist"] == 0
        assert c.hash_repeats["right_front"] == 0, "an identical object is not also a hash repeat"
        assert c.samples == 2

    def test_fresh_frames_are_never_counted(self):
        c = FrameDeliveryCounter()
        for i in range(10):
            c.observe({"right_front": _frame(i), "left_wrist": _frame(100 + i)}, CAMS)
        assert c.repeated_buffers == {"right_front": 0, "left_wrist": 0}
        assert c.hash_repeats == {"right_front": 0, "left_wrist": 0}

    def test_an_equal_but_distinct_array_is_a_hash_repeat_not_a_buffer_repeat(self):
        """A different object holding identical bytes is the camera or driver
        repeating a frame, not the loop outrunning the camera. Different fault,
        counted apart."""
        c = FrameDeliveryCounter()
        f = _frame(7)
        c.observe({"right_front": f}, ["right_front"])
        c.observe({"right_front": f.copy()}, ["right_front"])

        assert c.repeated_buffers["right_front"] == 0
        assert c.hash_repeats["right_front"] == 1

    def test_the_frame_being_compared_against_is_kept_alive(self):
        """What makes `is` trustworthy: the counter holds a reference to the
        frame it will compare the next one against, so that object cannot be
        freed and its address cannot be handed to a different array.

        An earlier version of this test asserted that the *final* frame never
        lands on the first one's freed address. That is not the invariant --
        Python is free to recycle an address for any later object, so the
        assertion was both meaningless and able to fail by chance.
        """
        c = FrameDeliveryCounter()
        first = _frame(1)
        c.observe({"right_front": first}, ["right_front"])
        assert c._prev_frame["right_front"] is first     # held, so it cannot be freed
        del first

        for i in range(50):                      # churn allocations of the same size
            held = c._prev_frame["right_front"]
            c.observe({"right_front": _frame(200 + i)}, ["right_front"])
            assert c._prev_frame["right_front"] is not held, "each fresh frame replaces the held one"
        assert c.repeated_buffers["right_front"] == 0

    def test_a_missing_camera_key_is_skipped_not_crashed(self):
        c = FrameDeliveryCounter()
        c.observe({"right_front": _frame(1)}, CAMS)
        assert "left_wrist" not in c.repeated_buffers
        assert c.samples == 1

    def test_the_row_indices_of_the_repeats_are_recorded(self):
        """Totals cannot distinguish repeats clustered at episode start from
        repeats spread evenly, and that difference decides whether a threshold
        should be a rate at all. The encoded video cannot be re-examined for it.
        """
        c = FrameDeliveryCounter()
        stale = _frame(1)
        c.observe({"right_front": stale}, ["right_front"])          # frame 0, fresh
        c.observe({"right_front": stale}, ["right_front"])          # frame 1, repeat
        c.observe({"right_front": stale}, ["right_front"])          # frame 2, repeat
        c.observe({"right_front": _frame(2)}, ["right_front"])      # frame 3, fresh
        c.observe({"right_front": _frame(3)}, ["right_front"])      # frame 4, fresh

        assert c.at_frames["right_front"] == [1, 2]
        assert c.repeated_buffers["right_front"] == len(c.at_frames["right_front"])

    def test_hash_repeats_are_indexed_separately(self):
        c = FrameDeliveryCounter()
        f = _frame(7)
        c.observe({"right_front": f}, ["right_front"])              # frame 0
        c.observe({"right_front": f.copy()}, ["right_front"])       # frame 1, same bytes
        assert c.at_frames["right_front"] == []
        assert c.hash_at_frames["right_front"] == [1]

    def test_the_index_is_the_dataset_row_index(self):
        """Every loop sample becomes one dataset row, so these indices address
        rows in the episode's parquet directly. Verified on real data: the
        smoke session's loop_samples matched its per-episode row counts exactly
        (600/502/326)."""
        c = FrameDeliveryCounter()
        stale = _frame(99)
        frames = [_frame(i) for i in range(10)]
        frames[7] = frames[6] = stale                # rows 6 and 7 hold one object
        for f in frames:
            c.observe({"right_front": f}, ["right_front"])

        assert c.samples == 10
        assert c.at_frames["right_front"] == [7], "row 7 repeats row 6; row 6 itself is fresh"

    def test_hashing_can_be_switched_off(self):
        c = FrameDeliveryCounter(hash_content=False)
        f = _frame(3)
        c.observe({"right_front": f}, ["right_front"])
        c.observe({"right_front": f.copy()}, ["right_front"])
        assert c.hash_repeats["right_front"] == 0


def test_frame_digest_is_content_addressed():
    f = _frame(4)
    assert frame_digest(f) == frame_digest(f.copy())
    assert frame_digest(f) != frame_digest(_frame(5))


class TestSidecar:
    def test_the_row_stores_raw_counts_not_percentages(self):
        """A WARN or FAIL threshold chosen later has to be applicable to
        sessions recorded before it existed, which requires the denominator."""
        c = FrameDeliveryCounter()
        stale = _frame(1)
        c.observe({"right_front": stale}, ["right_front"])
        c.observe({"right_front": stale}, ["right_front"])
        c.observe({"right_front": _frame(2)}, ["right_front"])

        row = c.record(episode_index=7, elapsed_s=1.234)
        assert row["episode_index"] == 7
        assert row["loop_samples"] == 3
        assert row["elapsed_s"] == 1.23
        assert row["cameras"]["right_front"] == {
            "repeated_buffers": 1, "hash_repeats": 0,
            "at_frames": [1], "hash_at_frames": [],
        }
        assert not any("pct" in k or "%" in k for k in row["cameras"]["right_front"])

    def test_rows_append_one_json_object_per_episode(self, tmp_path):
        path = tmp_path / SIDECAR_NAME
        for ep in range(3):
            c = FrameDeliveryCounter()
            c.observe({"right_front": _frame(ep)}, ["right_front"])
            append_sidecar(path, c.record(episode_index=ep, elapsed_s=1.0))

        rows = [json.loads(ln) for ln in path.read_text().splitlines()]
        assert [r["episode_index"] for r in rows] == [0, 1, 2]

    def test_an_unwritable_path_does_not_end_the_run(self, tmp_path):
        """Instrumentation must never be the reason an episode is lost."""
        append_sidecar(tmp_path / "no_such_dir" / SIDECAR_NAME, {"episode_index": 0})


class TestPersist:
    """The first version of this read `dataset.episode_buffer`, which exists on
    `dataset.writer` and not on `LeRobotDataset`. It raised AttributeError at the
    end of the first episode and ended the recording session. Nothing here was
    covered, because the accessor was only ever read from the docs.
    """

    class _Dataset:
        def __init__(self, root, num_episodes=4):
            self.root = root
            self._n = num_episodes

        @property
        def num_episodes(self):
            if isinstance(self._n, Exception):
                raise self._n
            return self._n

    def test_the_row_lands_in_the_session_directory(self, tmp_path):
        c = FrameDeliveryCounter()
        c.observe({"right_front": _frame(1)}, ["right_front"])
        row = persist(c, self._Dataset(tmp_path, num_episodes=4), elapsed_s=2.0)

        assert row["episode_index"] == 4
        written = json.loads((tmp_path / SIDECAR_NAME).read_text().strip())
        assert written == row

    def test_a_dataset_that_cannot_report_its_index_still_writes_a_row(self, tmp_path):
        """A missing number is a gap in a diagnostic. A raised one costs the episode."""
        c = FrameDeliveryCounter()
        c.observe({"right_front": _frame(1)}, ["right_front"])
        broken = self._Dataset(tmp_path, num_episodes=AttributeError("no attribute"))
        row = persist(c, broken, elapsed_s=2.0)

        assert row["episode_index"] is None
        assert row["cameras"]["right_front"]["repeated_buffers"] == 0
        assert (tmp_path / SIDECAR_NAME).exists()

    def test_no_dataset_is_not_an_error(self):
        c = FrameDeliveryCounter()
        c.observe({"right_front": _frame(1)}, ["right_front"])
        assert persist(c, None, elapsed_s=1.0)["episode_index"] is None

    def test_a_string_root_is_accepted(self, tmp_path):
        c = FrameDeliveryCounter()
        c.observe({"right_front": _frame(1)}, ["right_front"])
        persist(c, self._Dataset(str(tmp_path)), elapsed_s=1.0)
        assert (tmp_path / SIDECAR_NAME).exists()

    def test_resolve_episode_index_never_raises(self):
        for value in (AttributeError("x"), KeyError("x"), TypeError("x"), None):
            ds = self._Dataset("/nowhere", num_episodes=value)
            assert resolve_episode_index(ds) is None or isinstance(
                resolve_episode_index(ds), int)


class TestEpisodeFrameCount:
    """`episode_buffer` is on `dataset.writer`, not on the dataset.

    Eight call sites in the recording loop spelled `dataset.episode_buffer`
    inline. All eight were on the online-RL path -- critical-phase marks and
    human takeovers -- so plain teleop recording never reached them and the
    AttributeError stayed invisible until an unconditional read hit it.
    """

    class _Writer:
        def __init__(self, buffer):
            self.episode_buffer = buffer

    class _Dataset:
        def __init__(self, writer):
            self.writer = writer

    def test_it_reads_through_the_writer(self):
        ds = self._Dataset(self._Writer({"size": 42}))
        assert loop.episode_frame_count(ds) == 42

    def test_a_dataset_without_a_writer_is_zero_not_an_error(self):
        assert loop.episode_frame_count(object()) == 0

    def test_no_dataset_is_zero(self):
        assert loop.episode_frame_count(None) == 0

    def test_a_writer_without_a_buffer_is_zero(self):
        assert loop.episode_frame_count(self._Dataset(self._Writer(None))) == 0

    def test_a_buffer_without_a_size_is_zero(self):
        assert loop.episode_frame_count(self._Dataset(self._Writer({}))) == 0

    def test_reading_it_off_the_dataset_is_what_used_to_break(self):
        """Pin the shape of the mistake: the attribute is genuinely absent."""
        ds = self._Dataset(self._Writer({"size": 7}))
        assert not hasattr(ds, "episode_buffer")
        assert loop.episode_frame_count(ds) == 7


def test_summary_line_reports_both_kinds_of_repeat():
    c = FrameDeliveryCounter()
    stale = _frame(1)
    c.observe({"right_front": stale}, ["right_front"])
    c.observe({"right_front": stale}, ["right_front"])
    line = c.summary_line()
    assert "right_front 1 repeated buffers (50.0%)" in line
