"""Offline FK export checked against independent continuous rigid-body motion."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from unilab.data.tracking_export import export_tracking_motion
from unilab.tasks.motion_tracking.common.motion_loader import MotionLoader

_XML = """<mujoco><compiler angle="radian"/><worldbody>
<body name="root"><freejoint/>
 <inertial pos="0.2 0.1 -0.1" mass="2" diaginertia="0.1 0.1 0.1"/>
 <body name="upper" pos="0.3 -0.2 0.1"><joint name="hip" axis="0 0 1" range="-1 1"/>
  <inertial pos="0.1 0.2 0" mass="1" diaginertia="0.05 0.05 0.05"/>
  <body name="lower" pos="0.4 0.1 -0.2"><joint name="knee" axis="0 1 0" range="-1 1"/>
   <inertial pos="0.2 0 0.1" mass="1" diaginertia="0.04 0.04 0.04"/>
   <body name="tip" pos="0.2 -0.1 0.3"/>
  </body>
 </body>
</body></worldbody></mujoco>"""


def _root_rotation(time):
    return Rotation.from_rotvec([0, 0, 0.7 * time]) * Rotation.from_euler("xyz", [0.6, -0.4, 0.9])


def _configuration(times):
    poses = np.zeros((len(times), 9))
    poses[:, :3] = np.array([1.0, -0.5, 2.0]) + times[:, None] * np.array([0.3, -0.2, 0.4])
    poses[:, 3:7] = np.stack([_root_rotation(t).as_quat()[[3, 0, 1, 2]] for t in times])
    poses[:, 7] = 0.1 + 0.2 * times
    poses[:, 8] = -0.2 - 0.3 * times
    return poses


def _world_pose(time):
    """Analytic hierarchy from the fixture, independent of MuJoCo FK/Jacobians."""
    p0 = np.array([1.0, -0.5, 2.0]) + time * np.array([0.3, -0.2, 0.4])
    r0 = _root_rotation(time)
    r1 = r0 * Rotation.from_rotvec([0, 0, 0.1 + 0.2 * time])
    r2 = r1 * Rotation.from_rotvec([0, -0.2 - 0.3 * time, 0])
    p1 = p0 + r0.apply([0.3, -0.2, 0.1])
    p2 = p1 + r1.apply([0.4, 0.1, -0.2])
    p3 = p2 + r2.apply([0.2, -0.1, 0.3])
    return np.stack((np.zeros(3), p0, p1, p2, p3)), [Rotation.identity(), r0, r1, r2, r2]


@pytest.fixture
def export_case(tmp_path):
    model = tmp_path / "tiny.xml"
    model.write_text(_XML)
    times = np.arange(6) / 50
    return model, times, _configuration(times), tmp_path / "motion.npz"


def test_world_body_origin_fk_velocity_names_and_real_loader(export_case):
    model, times, poses, output = export_case
    # Flip signs without changing physical orientation.
    poses[::2, 3:7] *= -1
    export_tracking_motion(
        output,
        model_file=model,
        qpos=poses,
        joint_names=("knee", "hip"),
        fps=50,
        metadata={"source": "independent test fixture"},
    )
    with np.load(output, allow_pickle=False) as data:
        assert data["joint_names"].tolist() == ["knee", "hip"]
        assert data["body_names"].tolist() == ["world", "root", "upper", "lower", "tip"]
        assert json.loads(data["metadata_json"].item())["source"] == "independent test fixture"
        np.testing.assert_allclose(data["joint_pos"], poses[:, [8, 7]])
        np.testing.assert_allclose(
            data["joint_vel"], np.tile([-0.3, 0.2], (len(times), 1)), atol=1e-14
        )
        epsilon = 1e-5
        for i, time in enumerate(times):
            positions, rotations = _world_pose(time)
            previous, before = _world_pose(time - epsilon)
            following, after = _world_pose(time + epsilon)
            linear = (following - previous) / (2 * epsilon)
            angular = np.stack(
                [(a * b.inv()).as_rotvec() / (2 * epsilon) for a, b in zip(after, before)]
            )
            np.testing.assert_allclose(data["body_pos_w"][i], positions, atol=2e-14)
            actual_rotation = Rotation.from_quat(data["body_quat_w"][i, :, [1, 2, 3, 0]].T)
            np.testing.assert_allclose(
                actual_rotation.as_matrix(), [r.as_matrix() for r in rotations], atol=2e-14
            )
            np.testing.assert_allclose(data["body_lin_vel_w"][i], linear, atol=1e-9)
            np.testing.assert_allclose(data["body_ang_vel_w"][i], angular, atol=1e-9)
        # Off-centre inertials make these checks distinguish body-origin from COM.
        np.testing.assert_array_equal(data["body_lin_vel_w"][:, 0], 0)
        np.testing.assert_array_equal(data["body_ang_vel_w"][:, 0], 0)
        np.testing.assert_array_equal(
            data["body_quat_w"][:, 0], np.tile([1, 0, 0, 0], (len(times), 1))
        )
        all_arrays = {
            key: data[key].copy()
            for key in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        }
    full = MotionLoader(str(output))
    selected = MotionLoader(str(output), body_indices=np.array([4, 1]))
    assert full.num_frames == 6 and full.num_bodies == 5 and full.fps == 50
    assert selected.num_bodies == 2
    for name, expected in all_arrays.items():
        np.testing.assert_allclose(getattr(full, name), expected, atol=2e-7)
        subset = expected[:, [4, 1]] if name.startswith("body_") else expected
        np.testing.assert_allclose(getattr(selected, name), subset, atol=2e-7)


@pytest.mark.parametrize(
    "kind", ["limit", "quat", "nonfinite", "missing_name", "duplicate", "wrong_name", "width"]
)
def test_export_rejects_invalid_trajectory_without_creating_output(export_case, kind):
    model, _, poses, output = export_case
    names = ("hip", "knee")
    if kind == "limit":
        poses[2, 7] = 2
    elif kind == "quat":
        poses[2, 3:7] *= 2
    elif kind == "nonfinite":
        poses[2, 0] = np.nan
    elif kind == "missing_name":
        names = ("hip",)
    elif kind == "duplicate":
        names = ("hip", "hip")
    elif kind == "wrong_name":
        names = ("hip", "ankle")
    else:
        poses = poses[:, :-1]
    with pytest.raises(ValueError):
        export_tracking_motion(output, model_file=model, qpos=poses, joint_names=names, fps=50)
    assert not output.exists()


def test_export_refuses_overwrite(export_case):
    model, _, poses, output = export_case
    output.write_bytes(b"preserve original")
    with pytest.raises(FileExistsError, match="overwrite"):
        export_tracking_motion(
            output, model_file=model, qpos=poses, joint_names=("hip", "knee"), fps=50
        )
    assert output.read_bytes() == b"preserve original"


@pytest.mark.parametrize("fps", [True, 0, -50, 50.0])
def test_export_rejects_invalid_fps(export_case, fps):
    model, _, poses, output = export_case
    with pytest.raises(ValueError, match="positive integer"):
        export_tracking_motion(
            output, model_file=model, qpos=poses, joint_names=("hip", "knee"), fps=fps
        )
    assert not output.exists()
