"""Episode playback: path resolution and the seek maths."""

import json

import pandas as pd
import pytest

from rlt_so101_dual.core import shape_contract as sc
from rlt_so101_dual.diagnostics import replay

PRIMARY = sc.CAMERA_KEYS[0]


def _session(root, name, episodes, *, sidecar=None, videos=True):
    s = root / "0811_tag" / name
    (s / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (s / "meta" / "info.json").write_text(json.dumps({"total_episodes": len(episodes)}))

    rows, t = [], 0.0
    for i, length in enumerate(episodes):
        rows.append({
            "episode_index": i, "length": length, "episode_success": "success",
            f"videos/observation.images.{PRIMARY}/from_timestamp": t,
            f"videos/observation.images.{PRIMARY}/to_timestamp": t + length / 30,
        })
        t += length / 30
    pd.DataFrame(rows).to_parquet(s / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    if videos:
        for cam in sc.CAMERA_KEYS:
            d = s / "videos" / f"observation.images.{cam}" / "chunk-000"
            d.mkdir(parents=True)
            (d / "file-000.mp4").write_bytes(b"")
    if sidecar is not None:
        (s / "camera_delivery.jsonl").write_text(
            "\n".join(json.dumps(r) for r in sidecar) + "\n")
    return s


def _cam_repeats(**counts):
    return {cam: {"repeated_buffers": counts.get(cam, 0)} for cam in sc.CAMERA_KEYS}


class TestResolveSession:
    def test_the_newest_finished_session_is_the_default(self, tmp_path):
        _session(tmp_path, "record_teleop_full_100544", [10])
        newer = _session(tmp_path, "record_teleop_full_101213", [10, 20])
        assert replay.resolve_session(None, root=tmp_path) == newer

    def test_a_session_being_recorded_right_now_is_skipped(self, tmp_path):
        done = _session(tmp_path, "record_teleop_full_100544", [10])
        live = tmp_path / "0811_tag" / "record_teleop_full_104814"
        (live / "meta").mkdir(parents=True)
        (live / "meta" / "info.json").write_text('{"total_episodes": 0}')

        assert replay.resolve_session(None, root=tmp_path) == done

    def test_an_explicit_path_is_used_as_given(self, tmp_path):
        older = _session(tmp_path, "record_teleop_full_100544", [10])
        _session(tmp_path, "record_teleop_full_101213", [10])
        assert replay.resolve_session(str(older), root=tmp_path) == older

    def test_a_path_that_is_not_a_session_says_so(self, tmp_path):
        with pytest.raises(SystemExit, match="not a session"):
            replay.resolve_session(str(tmp_path))

    def test_no_sessions_at_all_says_so(self, tmp_path):
        with pytest.raises(SystemExit, match="no session"):
            replay.resolve_session(None, root=tmp_path)


class TestEpisodeTable:
    def test_spans_follow_the_episode_lengths(self, tmp_path):
        s = _session(tmp_path, "record_teleop_full_100544", [434, 574])
        rows = replay.episode_table(s)
        assert [r["length"] for r in rows] == [434, 574]
        assert rows[1]["from"] == pytest.approx(434 / 30)
        assert rows[1]["to"] - rows[1]["from"] == pytest.approx(574 / 30)

    def test_repeat_counts_come_from_the_sidecar(self, tmp_path):
        side = [{"episode_index": 0, "loop_samples": 434,
                 "cameras": _cam_repeats(left_wrist=20, right_wrist=1, right_front=0)}]
        s = _session(tmp_path, "record_teleop_full_100544", [434], sidecar=side)
        row = replay.episode_table(s)[0]
        assert row["repeats"]["left_wrist"] == 20
        assert row["repeats"]["right_wrist"] == 1
        assert row["aligned"] is True

    def test_a_retake_supersedes_the_attempt_it_replaced(self, tmp_path):
        side = [
            {"episode_index": 0, "loop_samples": 186, "cameras": _cam_repeats()},
            {"episode_index": 0, "loop_samples": 434,
             "cameras": _cam_repeats(left_wrist=20, right_wrist=1)},
        ]
        s = _session(tmp_path, "record_teleop_full_100544", [434], sidecar=side)
        row = replay.episode_table(s)[0]
        assert row["repeats"]["left_wrist"] == 20
        assert row["aligned"] is True

    def test_a_sidecar_that_disagrees_with_the_data_is_flagged(self, tmp_path):
        side = [{"episode_index": 0, "loop_samples": 999, "cameras": _cam_repeats()}]
        s = _session(tmp_path, "record_teleop_full_100544", [434], sidecar=side)
        assert replay.episode_table(s)[0]["aligned"] is False

    def test_a_session_without_a_sidecar_still_lists(self, tmp_path):
        s = _session(tmp_path, "record_teleop_full_100544", [434])
        row = replay.episode_table(s)[0]
        assert row["repeats"] == {} and row["aligned"] is None


class TestCommand:
    def _ep(self, tmp_path, index=1):
        s = _session(tmp_path, "record_teleop_full_100544", [434, 574, 534])
        return s, replay.episode_table(s)[index]

    def test_it_seeks_to_the_episode_instead_of_playing_the_session(self, tmp_path):
        s, ep = self._ep(tmp_path)
        cmd = replay.build_command(s, ep, PRIMARY, speed=1.0)
        assert cmd[cmd.index("-ss") + 1] == f"{434 / 30:.3f}"
        assert cmd[cmd.index("-t") + 1] == f"{574 / 30:.3f}"

    def test_half_speed_doubles_the_presentation_timestamps(self, tmp_path):
        s, ep = self._ep(tmp_path)
        cmd = replay.build_command(s, ep, PRIMARY, speed=0.5)
        assert "setpts=2.0000*PTS" in cmd[cmd.index("-vf") + 1]

    def test_both_cameras_seek_to_the_same_place(self, tmp_path):
        s, ep = self._ep(tmp_path)
        graph = replay.build_command(s, ep, "both", speed=1.0)[-1]
        assert graph.count(f"seek_point={434 / 30:.3f}") == len(sc.CAMERA_KEYS)
        assert "hstack" in graph
        for cam in sc.CAMERA_KEYS:
            assert f"observation.images.{cam}" in graph

    def test_viz_targets_the_same_episode(self, tmp_path):
        s = _session(tmp_path, "record_teleop_full_100544", [434, 574])
        cmd = replay.build_viz_command(s, 1)
        assert cmd[cmd.index("--episode-index") + 1] == "1"
        assert cmd[cmd.index("--root") + 1] == str(s)


class TestMain:
    def test_print_only_does_not_launch_anything(self, tmp_path, capsys, monkeypatch):
        s = _session(tmp_path, "record_teleop_full_100544", [434, 574])
        monkeypatch.setattr(replay.subprocess, "call",
                            lambda *a, **k: pytest.fail("must not run"))
        assert replay.main([str(s), "-e", "1", "--print-only"]) == 0
        assert "ffplay" in capsys.readouterr().out

    def test_an_unknown_episode_lists_what_there_is(self, tmp_path):
        s = _session(tmp_path, "record_teleop_full_100544", [434, 574])
        with pytest.raises(SystemExit, match="available: 0, 1"):
            replay.main([str(s), "-e", "9"])
