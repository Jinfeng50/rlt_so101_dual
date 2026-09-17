"""Camera resolution from dual-arm setup.json (not USB autodetection).

Three UGREEN cameras share the same USB id, so indices come from the
manifest / --devices, not sysfs scanning.
"""

import json

import pytest

from rlt_so101_dual.diagnostics import camera_preview as cp


def test_devices_from_setup_json(tmp_path):
    path = tmp_path / "setup.json"
    path.write_text(
        json.dumps(
            {
                "cameras": [
                    {"alias": "left_wrist", "port": 0},
                    {"alias": "right_wrist", "port": 4},
                    {"alias": "right_front", "port": 2},
                ]
            }
        )
    )
    assert cp.devices_from_setup(path) == {
        "left_wrist": "0",
        "right_wrist": "4",
        "right_front": "2",
    }


def test_explicit_devices_win(tmp_path):
    devices, how = cp.resolve_devices(
        ["/dev/videoX", "/dev/videoY", "/dev/videoZ"],
        setup_json=tmp_path / "missing.json",
    )
    assert list(devices.values()) == ["/dev/videoX", "/dev/videoY", "/dev/videoZ"]
    assert list(devices) == ["left_wrist", "right_wrist", "right_front"]
    assert "command line" in how


def test_wrong_device_count_exits():
    with pytest.raises(SystemExit, match="needs 3"):
        cp.resolve_devices(["/dev/video0"], setup_json=None)


def test_missing_setup_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "DEFAULT_DEVICES", {
        "left_wrist": "/nonexistent0",
        "right_wrist": "/nonexistent1",
        "right_front": "/nonexistent2",
    })
    with pytest.raises(SystemExit, match="No cameras resolved"):
        cp.resolve_devices(None, setup_json=tmp_path / "nope.json")


def test_resize_with_pad_matches_pi05_shape():
    import numpy as np

    out = cp.resize_with_pad(np.zeros((480, 640, 3), dtype=np.uint8), cp.PI05_SIDE)
    assert out.shape == (224, 224, 3)
