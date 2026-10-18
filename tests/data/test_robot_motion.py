"""Retargeted-data adapter against an independent analytic rigid-body chain."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from unilab.data.robot_motion import prepare_retargeted_motion
from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader

_XML = """<mujoco><compiler angle="radian"/><worldbody>
<body name="root"><freejoint/>
 <inertial pos="0.1 0 0" mass="2" diaginertia="0.1 0.1 0.1"/>
 <body name="upper" pos="0.3 0 0.1"><joint name="hip" axis="0 0 1" range="-1 1"/>
  <inertial pos="0 0.1 0" mass="1" diaginertia="0.05 0.05 0.05"/>
  <body name="lower" pos="0.4 0.1 -0.2"><joint name="knee" axis="0 1 0" range="-1 1"/>
   <inertial pos="0.1 0 0" mass="1" diaginertia="0.04 0.04 0.04"/>
   <body name="tip" pos="0.2 -0.1 0.3"/>
  </body>
 </body>
</body></worldbody></mujoco>"""


def _analytic(time):
    root = np.array([0.2 + time, -0.3 + 0.2 * time, 0.8])
    r0 = Rotation.from_euler("xyz", [0.2, -0.1, 0.3 + 0.6 * time])
    r1 = r0 * Rotation.from_rotvec([0, 0, 0.2 + 0.3 * time])
    r2 = r1 * Rotation.from_rotvec([0, -0.1 + 0.4 * time, 0])
    upper = root + r0.apply([0.3, 0, 0.1])
    lower = upper + r1.apply([0.4, 0.1, -0.2])
    tip = lower + r2.apply([0.2, -0.1, 0.3])
    return np.stack([root, upper, lower, tip]), [r0, r1, r2, r2]


@pytest.fixture
def robot_case(tmp_path):
    model = tmp_path / "robot.xml"
    model.write_text(_XML)
    times = np.arange(4) / 30
    positions, quaternions = [], []
    for t in times:
        p, rotations = _analytic(t)
        # Source has a different body order, no world0, and an extra fixed body.
        positions.append(p[[3, 2, 0, 1, 0]])
        q = np.stack([r.as_quat()[[3, 0, 1, 2]] for r in rotations])
        quaternions.append(q[[3, 2, 0, 1, 0]])
    fields = {
        "fps": np.array([30]),
        "joint_names": np.array(["knee", "hip"]),
        "joint_pos": np.stack([-0.1 + 0.4 * times, 0.2 + 0.3 * times], axis=-1),
        "joint_vel": np.full((4, 2), 1234.0),
        "body_pos_w": np.array(positions),
        "body_quat_w": np.array(quaternions).astype(np.float32),
    }
    fields["body_quat_w"][::2] *= -1
    source, output = tmp_path / "source.npz", tmp_path / "target.npz"
    np.savez(source, **fields)
    return source, output, model, fields


def _convert(case, **kwargs):
    source, output, model, _fields = case
    return prepare_retargeted_motion(
        source,
        output,
        model_file=model,
        joint_names=("hip", "knee"),
        root_body_index=kwargs.pop("root_body_index", 2),
        **kwargs,
    )


def test_named_joint_mapping_time_grid_target_fk_and_loader(robot_case):
    _convert(robot_case, metadata={"source_dataset": "independent fixture"})
    with np.load(robot_case[1], allow_pickle=False) as motion:
        times = np.arange(6) / 50
        assert motion["fps"].tolist() == [50]
        assert motion["joint_names"].tolist() == ["hip", "knee"]
        assert motion["body_names"].tolist() == ["world", "root", "upper", "lower", "tip"]
        np.testing.assert_allclose(
            motion["joint_pos"], np.stack([0.2 + 0.3 * times, -0.1 + 0.4 * times], axis=-1)
        )
        np.testing.assert_allclose(motion["joint_vel"], np.tile([0.3, 0.4], (6, 1)), atol=1e-13)
        for i, time in enumerate(times):
            positions, _rotations = _analytic(time)
            np.testing.assert_allclose(motion["body_pos_w"][i, 1:], positions, atol=3e-8)
        metadata = json.loads(motion["metadata_json"].item())
        assert metadata["source_root_body_index"] == 2
        assert metadata["max_source_fk_position_error_m"] < 1e-7
        assert metadata["source_fps"] == 30
    loaded = MotionLoader(str(robot_case[1]))
    assert (loaded.num_frames, loaded.num_bodies, loaded.num_joints) == (6, 5, 2)


@pytest.mark.parametrize(
    "kind",
    [
        "root",
        "quaternion_order",
        "bad_norm",
        "joint_order",
        "body_pose",
        "limit",
        "nonfinite",
        "fps",
    ],
)
def test_rejects_invalid_source_without_publishing(robot_case, kind):
    source, output, _model, fields = robot_case
    kwargs = {}
    if kind == "root":
        kwargs["root_body_index"] = 0
    elif kind == "quaternion_order":
        fields["body_quat_w"] = fields["body_quat_w"][..., [1, 2, 3, 0]]
    elif kind == "bad_norm":
        fields["body_quat_w"][0, 2] *= 1.1
    elif kind == "joint_order":
        fields["joint_names"] = np.array(["hip", "knee"])
    elif kind == "body_pose":
        fields["body_pos_w"][:, 0, 0] += 0.02
    elif kind == "limit":
        fields["joint_pos"][0, 0] = 1.01
    elif kind == "nonfinite":
        fields["joint_pos"][0, 0] = np.nan
    elif kind == "fps":
        fields["fps"] = np.array([0])
    np.savez(source, **fields)
    with pytest.raises(ValueError):
        _convert(robot_case, **kwargs)
    assert not output.exists()
