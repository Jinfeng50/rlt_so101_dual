"""Leader go-home / pre-reset align helpers."""

from __future__ import annotations

from rlt_so101_dual.adapters.lerobot.record.hil import ramp_teleop_to_action
from rlt_so101_dual.adapters.lerobot.record.runner import (
    HOTKEY_ARM_DELAY_S,
    _debounce_allow,
    _ensure_record_events,
    _hotkeys_armed,
)


class _FakeTeleop:
    def __init__(self):
        self.sent: list[dict[str, float]] = []
        self._pose = {
            "left_shoulder_lift.pos": -40.0,
            "right_shoulder_lift.pos": -40.0,
        }

    def get_action(self):
        return dict(self._pose)


def test_ramp_teleop_blends_toward_target(monkeypatch):
    teleop = _FakeTeleop()
    sent: list[dict[str, float]] = []

    def _capture(_teleop, action):
        sent.append(dict(action))
        teleop._pose.update(action)

    monkeypatch.setattr(
        "rlt_so101_dual.adapters.lerobot.record.hil.send_teleop_feedback",
        _capture,
    )
    monkeypatch.setattr(
        "lerobot.utils.robot_utils.precise_sleep",
        lambda _s: None,
    )
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda _s: None)

    target = {
        "left_shoulder_lift.pos": -100.0,
        "right_shoulder_lift.pos": -100.0,
        "ignored_image": "x",
    }
    ramp_teleop_to_action(teleop, target, steps=4, fps=30.0, reason="test")
    assert len(sent) == 4
    assert sent[0]["left_shoulder_lift.pos"] == -55.0
    assert sent[-1]["left_shoulder_lift.pos"] == -100.0
    assert "ignored_image" not in sent[-1]


def test_hotkey_arming_and_debounce():
    events: dict = {}
    _ensure_record_events(events)
    events["ignore_hotkeys_until"] = 1e18  # far future
    assert not _hotkeys_armed(events)
    events["ignore_hotkeys_until"] = 0.0
    assert _hotkeys_armed(events)
    assert HOTKEY_ARM_DELAY_S >= 1.0
    assert _debounce_allow(events, "episode_failure")
    assert not _debounce_allow(events, "episode_failure")
