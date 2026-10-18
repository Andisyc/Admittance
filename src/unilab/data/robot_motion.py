"""Adapt named, already retargeted robot trajectories to target-model tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from unilab.data.human_motion import HumanMotion, resample_human_motion
from unilab.data.retarget import file_sha256, kinematics_sha256
from unilab.data.tracking_export import export_tracking_motion


def prepare_retargeted_motion(
    source: str | Path,
    output: str | Path,
    *,
    model_file: str | Path,
    joint_names: Sequence[str],
    root_body_index: int,
    target_fps: int = 50,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Convert explicit Z-up/wxyz named-joint NPZs, never trusting body indices.

    ``root_body_index`` declares the source root frame; it is not inferred from
    array width. At nine source frames, an injective pose match must reproduce
    every target non-world body within 0.1 mm and 0.1 mrad. Extra fixed source
    bodies are allowed. Positions/velocities are then regenerated with target FK.
    Small float32 quaternion norm roundoff (<=1e-5) is explicitly renormalized;
    malformed rotations, unknown joints and target-limit violations fail.
    """
    import mujoco as _mj
    from scipy.optimize import linear_sum_assignment

    mj: Any = _mj
    with np.load(source, allow_pickle=False) as arrays:
        required = {"fps", "joint_names", "joint_pos", "body_pos_w", "body_quat_w"}
        if not required.issubset(arrays.files):
            raise ValueError(f"retargeted source is missing {sorted(required - set(arrays.files))}")
        fps_array = arrays["fps"]
        if fps_array.size != 1 or fps_array.dtype.kind not in "fiu":
            raise ValueError("source fps must be a numeric scalar")
        source_fps = float(fps_array.reshape(-1)[0])
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("source fps must be finite and positive")
        raw_names = arrays["joint_names"]
        if raw_names.ndim != 1 or raw_names.dtype.kind != "U":
            raise ValueError("source joint_names must be a Unicode vector")
        source_names = raw_names.tolist()
        values = {}
        for key in ("joint_pos", "body_pos_w", "body_quat_w"):
            raw = arrays[key]
            if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
                raise ValueError(f"source {key} must be finite numeric data")
            values[key] = np.asarray(raw, dtype=np.float64)
    names = tuple(joint_names)
    if (
        not names
        or len(set(names)) != len(names)
        or len(set(source_names)) != len(source_names)
        or set(source_names) != set(names)
    ):
        raise ValueError("source must name every target hinge exactly once")
    joints, positions, quaternions = (values[k] for k in ("joint_pos", "body_pos_w", "body_quat_w"))
    if joints.ndim != 2 or joints.shape[0] < 2 or joints.shape[1] != len(names):
        raise ValueError("source joint_pos must be [T>=2,J]")
    if positions.ndim != 3 or positions.shape[0] != len(joints) or positions.shape[2] != 3:
        raise ValueError("source body_pos_w must be [T,B,3]")
    if quaternions.shape != (*positions.shape[:2], 4):
        raise ValueError("source body_quat_w must be [T,B,4]")
    if (
        isinstance(root_body_index, bool)
        or not isinstance(root_body_index, int)
        or not 0 <= root_body_index < positions.shape[1]
    ):
        raise ValueError("root_body_index is outside the source body axis")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if not np.allclose(norms, 1, atol=1e-5, rtol=0):
        raise ValueError("source body quaternions must be unit wxyz")
    quaternions = quaternions / norms
    model = mj.MjModel.from_xml_path(str(Path(model_file).resolve(strict=True)))
    target_names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)}
    if (
        model.njnt < 2
        or model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE
        or np.any(model.jnt_type[1:] != mj.mjtJoint.mjJNT_HINGE)
        or set(names) != target_names
        or positions.shape[1] < model.nbody - 1
    ):
        raise ValueError("source/model must share one free root and all target hinges")
    ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in names])
    reordered = joints[:, [source_names.index(n) for n in names]]
    limited = model.jnt_limited[ids].astype(bool)
    ranges, limited_values = model.jnt_range[ids][limited], reordered[:, limited]
    if np.any(limited_values < ranges[:, 0] - 1e-6) or np.any(limited_values > ranges[:, 1] + 1e-6):
        raise ValueError("source trajectory exceeds target joint limits")
    data = mj.MjData(model)
    max_position_error = max_rotation_error = 0.0
    for frame in np.unique(np.linspace(0, len(joints) - 1, min(9, len(joints)), dtype=int)):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = positions[frame, root_body_index]
        data.qpos[3:7] = quaternions[frame, root_body_index]
        data.qpos[model.jnt_qposadr[ids]] = reordered[frame]
        mj.mj_forward(model, data)
        distances = np.linalg.norm(data.xpos[1:, None] - positions[frame, None], axis=-1)
        dots = np.clip(np.abs(data.xquat[1:] @ quaternions[frame].T), 0, 1)
        angles = 2 * np.arccos(dots)
        rows, columns = linear_sum_assignment(distances + angles)
        max_position_error = max(max_position_error, float(distances[rows, columns].max()))
        max_rotation_error = max(max_rotation_error, float(angles[rows, columns].max()))
    if max_position_error > 1e-4 or max_rotation_error > 1e-4:
        raise ValueError(
            "source root/quaternion/joint semantics do not reproduce target kinematics"
        )
    root = HumanMotion(
        ("root",),
        positions[:, root_body_index : root_body_index + 1],
        quaternions[:, root_body_index : root_body_index + 1],
        source_fps,
    )
    resampled = resample_human_motion(root, target_fps)
    source_times = np.arange(len(joints), dtype=np.float64) / source_fps
    target_times = np.arange(len(resampled.positions), dtype=np.float64) / target_fps
    poses = np.broadcast_to(model.qpos0, (len(target_times), model.nq)).copy()
    poses[:, :3], poses[:, 3:7] = resampled.positions[:, 0], resampled.quaternions[:, 0]
    for index, address in enumerate(model.jnt_qposadr[ids]):
        poses[:, address] = np.interp(target_times, source_times, reordered[:, index])
    provenance = {
        **dict(metadata or {}),
        "source_sha256": file_sha256(source),
        "source_fps": source_fps,
        "target_fps": target_fps,
        "source_frames": len(joints),
        "output_frames": len(poses),
        "source_root_body_index": root_body_index,
        "source_world_convention": "meters; Z-up; wxyz; declared retargeted robot poses",
        "max_source_fk_position_error_m": max_position_error,
        "max_source_fk_rotation_error_rad": max_rotation_error,
        "target_kinematics_sha256": kinematics_sha256(model),
        "target_world_convention": "meters; Z-up; wxyz; body-id axis includes world0",
        "dynamic_feasibility": "not evaluated",
    }
    return export_tracking_motion(
        output,
        model_file=model_file,
        qpos=poses,
        joint_names=names,
        fps=target_fps,
        metadata=provenance,
    )
