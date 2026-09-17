"""Go-home pose capture for bimanual followers."""

from rlt_so101_dual.diagnostics.record_pose import read_bimanual_ticks, read_raw_ticks


class _FakeBus:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def sync_read(self, data_name, motors=None, *, normalize=True, num_retry=0):
        self.calls.append((data_name, normalize))
        return self.values


class _FakeArm:
    def __init__(self, values):
        self.bus = _FakeBus(values)


class _FakeBiRobot:
    def __init__(self, left_values, right_values):
        self.left_arm = _FakeArm(left_values)
        self.right_arm = _FakeArm(right_values)


def test_ticks_are_read_raw_not_normalised():
    arm = _FakeArm({"shoulder_pan": 2054, "gripper": 1789})
    read_raw_ticks(arm)
    assert arm.bus.calls == [("Present_Position", False)]


def test_motor_names_get_the_pos_suffix_the_flag_expects():
    arm = _FakeArm({"shoulder_pan": 2054.7, "elbow_flex": 3041})
    assert read_raw_ticks(arm) == {"shoulder_pan.pos": 2054, "elbow_flex.pos": 3041}


def test_bimanual_names_match_the_contract():
    from rlt_so101_dual.core import shape_contract as sc

    joint = "shoulder_pan"
    robot = _FakeBiRobot({joint: 2048}, {joint: 2049})
    ticks = read_bimanual_ticks(robot)
    assert ticks["left_shoulder_pan.pos"] == 2048
    assert ticks["right_shoulder_pan.pos"] == 2049
    assert set(ticks) <= set(sc.JOINT_NAMES)
