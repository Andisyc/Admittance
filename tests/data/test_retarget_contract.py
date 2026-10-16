"""Reject invalid coordinate transfers and publish only complete tracking files."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from unilab.data.retarget import file_sha256, kinematics_sha256, validate_g1_joint_mapping
from unilab.data.tracking_export import export_tracking_motion

_JOINTS = tuple(f"joint{j}" for j in range(29))


def _xml(*, axis="0 0 1", root_pos="0 0 1", root_quat="1 0 0 0", ref="0", extra=False):
    chain = (
        "".join(
            f'<body name="link{j}" pos="0 0 .03"><joint name="joint{j}" '
            f'axis="{axis if j == 3 else "0 0 1"}" range="-1 1" ref="{ref}"/>'
            '<geom type="sphere" size=".01" mass=".1"/>'
            for j in range(29)
        )
        + "</body>" * 29
    )
    fixed = '<body name="extra_fixed_frame" pos=".1 0 0"/>' if extra else ""
    return (
        '<mujoco><compiler angle="radian"/><worldbody>'
        f'<body name="pelvis" pos="{root_pos}" quat="{root_quat}"><freejoint/>'
        '<geom type="sphere" size=".05" mass="1"/>'
        f"{fixed}{chain}</body></worldbody></mujoco>"
    )


def _model(**kwargs):
    return mj.MjModel.from_xml_string(_xml(**kwargs))


def test_matching_named_frames_allow_fixed_extras_and_quaternion_sign():
    source = _model(extra=True)
    target = _model(root_quat="-1 0 0 0")
    source_addr, target_addr = validate_g1_joint_mapping(source, target, _JOINTS[::-1])
    np.testing.assert_array_equal(source_addr, np.arange(35, 6, -1))
    np.testing.assert_array_equal(target_addr, source_addr)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"axis": "0 0 -1"}, "jnt_axis"),
        ({"root_pos": "0 0 2"}, "frame position"),
        ({"root_quat": "0 0 0 1"}, "frame rotation"),
        ({"ref": "0.2"}, "reference angle"),
    ],
)
def test_rejects_changed_joint_or_root_coordinate(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_g1_joint_mapping(_model(), _model(**kwargs), _JOINTS)


def test_rejects_unknown_or_incomplete_joint_mapping():
    source, target = _model(), _model()
    with pytest.raises(ValueError, match="every target hinge"):
        xml = _xml().replace('name="joint28"', 'name="other_joint"')
        validate_g1_joint_mapping(source, mj.MjModel.from_xml_string(xml), _JOINTS)
    with pytest.raises(ValueError, match="29 unique"):
        validate_g1_joint_mapping(source, target, _JOINTS[:-1])
    with pytest.raises(ValueError, match="29 unique"):
        validate_g1_joint_mapping(source, target, (*_JOINTS[:-1], _JOINTS[0]))


def test_rejects_unnamed_intermediate_frame_that_cannot_be_verified():
    xml = _xml().replace('name="link8"', "")
    with pytest.raises(ValueError, match="named frames"):
        validate_g1_joint_mapping(_model(), mj.MjModel.from_xml_string(xml), _JOINTS)


def test_compiled_kinematics_hash_covers_included_robot(tmp_path):
    robot = tmp_path / "robot.xml"
    robot.write_text(_xml())
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><include file="robot.xml"/></mujoco>')
    scene_hash = file_sha256(scene)
    first = kinematics_sha256(mj.MjModel.from_xml_path(str(scene)))
    robot.write_text(_xml(axis="0 0 -1"))
    assert file_sha256(scene) == scene_hash
    assert kinematics_sha256(mj.MjModel.from_xml_path(str(scene))) != first


def _export_arguments(tmp_path):
    robot = tmp_path / "robot.xml"
    robot.write_text(_xml())
    model = mj.MjModel.from_xml_path(str(robot))
    return dict(
        model_file=robot,
        qpos=np.broadcast_to(model.qpos0, (2, model.nq)).copy(),
        joint_names=_JOINTS,
        fps=50,
    )


@pytest.mark.parametrize("metadata", [{"bad": float("nan")}, {"bad": object()}])
def test_bad_metadata_does_not_create_final_file(tmp_path, metadata):
    arguments = _export_arguments(tmp_path)
    target = tmp_path / "motion.npz"
    with pytest.raises((TypeError, ValueError)):
        export_tracking_motion(target, metadata=metadata, **arguments)
    assert not target.exists()
    assert not list(tmp_path.glob(".motion.npz.*.tmp"))


def test_interrupted_compression_cleans_temporary_and_final_file(tmp_path, monkeypatch):
    arguments = _export_arguments(tmp_path)
    target = tmp_path / "motion.npz"

    def interrupted(stream, **kwargs):
        stream.write(b"incomplete archive")
        raise OSError("simulated write interruption")

    monkeypatch.setattr(np, "savez_compressed", interrupted)
    with pytest.raises(OSError, match="interruption"):
        export_tracking_motion(target, **arguments)
    assert not target.exists()
    assert not list(tmp_path.glob(".motion.npz.*.tmp"))


def test_atomic_publication_refuses_concurrent_existing_file(tmp_path, monkeypatch):
    arguments = _export_arguments(tmp_path)
    target = tmp_path / "motion.npz"
    original_link = os.link

    def concurrent_writer(source, destination):
        Path(destination).write_bytes(b"other writer")
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", concurrent_writer)
    with pytest.raises(FileExistsError):
        export_tracking_motion(target, **arguments)
    assert target.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".motion.npz.*.tmp"))
