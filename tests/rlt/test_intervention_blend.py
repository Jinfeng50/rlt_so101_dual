from __future__ import annotations

import numpy as np
import pytest

from rlt_so101_dual.adapters.lerobot.recording import blend_robot_actions


def test_blend_robot_actions_interpolates_scalar_joints():
    start = {"left_shoulder.pos": 10.0, "right_shoulder.pos": -2.0}
    target = {"left_shoulder.pos": 14.0, "right_shoulder.pos": 6.0}

    blended = blend_robot_actions(list(start), start, target, 0.25)

    assert blended == {"left_shoulder.pos": 11.0, "right_shoulder.pos": 0.0}


def test_blend_robot_actions_clips_alpha_and_preserves_arrays():
    start = {"arm.pos": np.array([0.0, 10.0], dtype=np.float32)}
    target = {"arm.pos": np.array([10.0, 20.0], dtype=np.float32)}

    low = blend_robot_actions(["arm.pos"], start, target, -1.0)
    high = blend_robot_actions(["arm.pos"], start, target, 2.0)

    np.testing.assert_allclose(low["arm.pos"], start["arm.pos"])
    np.testing.assert_allclose(high["arm.pos"], target["arm.pos"])
    assert low["arm.pos"].dtype == np.float32


def test_blend_robot_actions_surfaces_shape_mismatch():
    start = {"arm.pos": np.array([0.0, 10.0], dtype=np.float32)}
    target = {"arm.pos": np.array([10.0, 20.0, 30.0], dtype=np.float32)}

    with pytest.raises(ValueError):
        blend_robot_actions(["arm.pos"], start, target, 0.5)
