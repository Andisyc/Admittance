"""Offline target-model FK and tracking NPZ export; no simulation stepping."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np


def export_tracking_motion(
    output: str | Path,
    *,
    model_file: str | Path,
    qpos: np.ndarray,
    joint_names: Sequence[str],
    fps: int,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export wxyz/world0/body-origin data in the *target* MuJoCo body layout.

    The input qpos is already in this model's order. All articulated joints
    must be supplied exactly once, in the desired policy order. This function
    never clips, normalizes malformed rotations, steps physics or changes XML.
    """
    import mujoco as _mj

    # Native MuJoCo exports are populated from extension modules without stubs.
    mj: Any = _mj

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("tracking fps must be a positive integer")
    # Validate metadata before creating any output. Only publish a completed
    # archive; a failed conversion must remain safely retryable at this path.
    metadata_json = json.dumps(dict(metadata or {}), sort_keys=True, allow_nan=False)
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing motion: {target}")
    model = mj.MjModel.from_xml_path(str(Path(model_file).resolve(strict=True)))
    poses = np.asarray(qpos, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] < 2 or poses.shape[1] != model.nq:
        raise ValueError(f"qpos must be [T>=2,{model.nq}] in target model order")
    if not np.isfinite(poses).all():
        raise ValueError("qpos must be finite")
    types = model.jnt_type
    if (
        model.njnt < 2
        or types[0] != mj.mjtJoint.mjJNT_FREE
        or np.any(types[1:] != mj.mjtJoint.mjJNT_HINGE)
    ):
        raise ValueError("export requires one root free joint followed by hinge joints")
    names = tuple(joint_names)
    expected = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)}
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("joint_names must name every target hinge exactly once")
    ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in names])
    qadr, vadr = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    if not np.allclose(np.linalg.norm(poses[:, 3:7], axis=-1), 1, atol=1e-7, rtol=0):
        raise ValueError("root quaternion must be unit wxyz")
    limited = model.jnt_limited[ids].astype(bool)
    values, ranges = poses[:, qadr][:, limited], model.jnt_range[ids][limited]
    if np.any(values < ranges[:, 0] - 1e-6) or np.any(values > ranges[:, 1] + 1e-6):
        raise ValueError("trajectory exceeds target joint limits")
    count, dt = len(poses), 1 / fps
    velocities = np.empty((count, model.nv))
    ahead, behind = np.empty(model.nv), np.empty(model.nv)
    for i in range(count):
        # Both increments use the CURRENT orientation's tangent frame, avoiding
        # the world/local ambiguity of quaternion component finite differences.
        if i == 0:
            mj.mj_differentiatePos(model, velocities[i], dt, poses[i], poses[i + 1])
        elif i == count - 1:
            mj.mj_differentiatePos(model, behind, dt, poses[i], poses[i - 1])
            velocities[i] = -behind
        else:
            mj.mj_differentiatePos(model, ahead, dt, poses[i], poses[i + 1])
            mj.mj_differentiatePos(model, behind, dt, poses[i], poses[i - 1])
            velocities[i] = (ahead - behind) / 2
    positions, quaternions, linear, angular = [
        np.empty((count, model.nbody, width)) for width in (3, 4, 3, 3)
    ]
    data = mj.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for i in range(count):
        data.qpos[:], data.qvel[:] = poses[i], velocities[i]
        mj.mj_forward(model, data)
        positions[i], quaternions[i] = data.xpos, data.xquat
        for body in range(model.nbody):
            mj.mj_jacBody(model, data, jp, jr, body)
            linear[i, body], angular[i, body] = jp @ velocities[i], jr @ velocities[i]
    payload = {
        "fps": np.array([fps]),
        "joint_pos": poses[:, qadr],
        "joint_vel": velocities[:, vadr],
        "body_pos_w": positions,
        "body_quat_w": quaternions,
        "body_lin_vel_w": linear,
        "body_ang_vel_w": angular,
    }
    if any(not np.isfinite(array).all() for array in payload.values()):
        raise ValueError("target FK produced nonfinite output")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream,
                **payload,
                joint_names=np.asarray(names),
                body_names=np.asarray(
                    [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
                ),
                metadata_json=np.array(metadata_json),
            )
        # A same-directory hard link publishes atomically and refuses to replace
        # a file that another process created after our initial existence check.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target
