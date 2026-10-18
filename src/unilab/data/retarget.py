"""Optional offline GMR adapter at the named-pose/target-model boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from unilab.data.human_motion import HumanMotion, resample_human_motion
from unilab.data.tracking_export import export_tracking_motion


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_g1_joint_mapping(
    source: Any, target: Any, joint_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return qpos addresses only when scalar joint copying preserves kinematics.

    Extra fixed GMR bodies are allowed. Shared physical frames and all 29 hinge
    coordinates must agree; matching strings alone cannot establish this.
    Joint ranges may differ and remain enforced by the solver/target exporter.
    """
    import mujoco as _mj

    mj: Any = _mj

    names = tuple(joint_names)
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("G1 joint mapping requires all 29 unique hinge names")
    maps = []
    body_maps = []
    for label, model in (("source", source), ("target", target)):
        if (
            model.njnt != 30
            or model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE
            or np.any(model.jnt_type[1:] != mj.mjtJoint.mjJNT_HINGE)
            or model.body_parentid[model.jnt_bodyid[0]] != 0
        ):
            raise ValueError(f"{label} model requires one root free joint and exactly 29 hinges")
        joint_map = {
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j): j for j in range(1, model.njnt)
        }
        if set(joint_map) != set(names):
            raise ValueError(f"joint_names must match every {label} hinge exactly once")
        for joint in range(1, model.njnt):
            body = int(model.jnt_bodyid[joint])
            while body != 0:
                if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body) is None:
                    raise ValueError(f"{label} articulated body chains must have named frames")
                body = int(model.body_parentid[body])
        maps.append(joint_map)
        body_maps.append(
            {
                mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b): b
                for b in range(model.nbody)
                if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) is not None
            }
        )
    for name in names:
        a, b = maps[0][name], maps[1][name]
        for field in ("jnt_axis", "jnt_pos"):
            if not np.allclose(
                getattr(source, field)[a], getattr(target, field)[b], atol=1e-9, rtol=0
            ):
                raise ValueError(f"joint {name!r} differs in {field}")
        if not np.isclose(
            source.qpos0[source.jnt_qposadr[a]],
            target.qpos0[target.jnt_qposadr[b]],
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError(f"joint {name!r} differs in reference angle")
        if mj.mj_id2name(source, mj.mjtObj.mjOBJ_BODY, int(source.jnt_bodyid[a])) != mj.mj_id2name(
            target, mj.mjtObj.mjOBJ_BODY, int(target.jnt_bodyid[b])
        ):
            raise ValueError(f"joint {name!r} belongs to a different physical body")
    source_root = mj.mj_id2name(source, mj.mjtObj.mjOBJ_BODY, int(source.jnt_bodyid[0]))
    target_root = mj.mj_id2name(target, mj.mjtObj.mjOBJ_BODY, int(target.jnt_bodyid[0]))
    if source_root is None or source_root != target_root:
        raise ValueError("source and target free joints must describe the same root body frame")
    if not np.allclose(source.jnt_pos[0], target.jnt_pos[0], atol=1e-9, rtol=0):
        raise ValueError("source and target root free-joint positions differ")
    for name in body_maps[0].keys() & body_maps[1].keys():
        a, b = body_maps[0][name], body_maps[1][name]
        source_parent = mj.mj_id2name(source, mj.mjtObj.mjOBJ_BODY, int(source.body_parentid[a]))
        target_parent = mj.mj_id2name(target, mj.mjtObj.mjOBJ_BODY, int(target.body_parentid[b]))
        if source_parent != target_parent or not np.allclose(
            source.body_pos[a], target.body_pos[b], atol=1e-9, rtol=0
        ):
            raise ValueError(f"shared body {name!r} has a different parent or frame position")
        qa, qb = source.body_quat[a], target.body_quat[b]
        if not (np.allclose(qa, qb, atol=1e-9, rtol=0) or np.allclose(qa, -qb, atol=1e-9, rtol=0)):
            raise ValueError(f"shared body {name!r} has a different frame rotation")
    return source.jnt_qposadr[[maps[0][name] for name in names]], target.jnt_qposadr[
        [maps[1][name] for name in names]
    ]


def kinematics_sha256(model: Any) -> str:
    """Hash compiled kinematic declarations, including XML includes, not dynamics.

    This deliberately identifies the body's/joint's compiled coordinate and
    limit layout; it is not a checksum of a complete MuJoCo model binary.
    """
    import mujoco as _mj

    mj: Any = _mj

    payload: dict[str, Any] = {
        "schema": "unilab.compiled_kinematics.v1",
        "body_names": [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)],
        "joint_names": [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)],
    }
    for name in (
        "body_pos",
        "body_quat",
        "body_parentid",
        "jnt_type",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "jnt_limited",
        "jnt_bodyid",
        "jnt_qposadr",
        "qpos0",
    ):
        payload[name] = np.asarray(getattr(model, name)).tolist()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def retarget_to_tracking(
    motion: HumanMotion,
    output: str | Path,
    *,
    model_file: str | Path,
    joint_names: Sequence[str],
    human_height: float,
    target_fps: int = 50,
    source_human: str = "smplx",
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Transfer named joints only; solve with explicit limits, then target FK.

    ``source_human='smplx'`` selects GMR's common named-body matching table,
    not a conversion of parameterized SMPL-H poses into an SMPL-X model.
    ``human_height`` is an explicit scale calibration, never guessed from beta0.
    GMR and Mink are optional preprocessing dependencies, not training runtimes.
    """
    import mink
    import mujoco as _mj
    from general_motion_retargeting import GeneralMotionRetargeting

    mj: Any = _mj

    if isinstance(human_height, bool) or not np.isfinite(human_height) or human_height <= 0:
        raise ValueError("human_height must be a finite positive number in meters")
    sampled = resample_human_motion(motion, target_fps)
    gmr = GeneralMotionRetargeting(
        source_human,
        "unitree_g1",
        actual_human_height=float(human_height),
        solver="daqp",
        verbose=False,
        use_velocity_limit=False,
    )
    required = set(gmr.human_scale_table)
    if not required.issubset(sampled.names):
        raise ValueError(f"human motion lacks GMR frames: {sorted(required - set(sampled.names))}")
    target = mj.MjModel.from_xml_path(str(Path(model_file).resolve(strict=True)))
    source_qadr, target_qadr = validate_g1_joint_mapping(gmr.model, target, joint_names)
    poses = np.broadcast_to(target.qpos0, (len(sampled.positions), target.nq)).copy()
    for i in range(len(poses)):
        gmr.update_targets(sampled.frame(i))
        # Explicit keyword avoids GMR's old sixth-positional-argument mismatch
        # with current Mink (where that argument is safety_break, not limits).
        for enabled, tasks in (
            (gmr.use_ik_match_table1, gmr.tasks1),
            (gmr.use_ik_match_table2, gmr.tasks2),
        ):
            if not enabled:
                continue
            previous = sum(float(np.linalg.norm(t.compute_error(gmr.configuration))) for t in tasks)
            for _ in range(gmr.max_iter + 1):
                velocity = mink.solve_ik(
                    gmr.configuration,
                    tasks,
                    gmr.model.opt.timestep,
                    gmr.solver,
                    damping=gmr.damping,
                    safety_break=True,
                    limits=gmr.ik_limits,
                )
                gmr.configuration.integrate_inplace(velocity, gmr.model.opt.timestep)
                error = sum(
                    float(np.linalg.norm(t.compute_error(gmr.configuration))) for t in tasks
                )
                if previous - error <= 0.001:
                    break
                previous = error
        result = gmr.configuration.data.qpos
        if not np.isfinite(result).all():
            raise ValueError(f"nonfinite GMR result at frame {i}")
        poses[i, :7] = result[:7]
        poses[i, target_qadr] = result[source_qadr]
    provenance = {
        **dict(metadata or {}),
        "source_fps": motion.fps,
        "target_fps": target_fps,
        "source_frames": len(motion.positions),
        "output_frames": len(poses),
        "human_height_m": float(human_height),
        "gmr_matching_table": source_human,
        "target_scene_sha256": file_sha256(model_file),
        "target_kinematics_sha256": kinematics_sha256(target),
        "world_convention": "meters; Z-up; wxyz; body-id axis includes world0",
        "ground_alignment": "unchanged",
        "dynamic_feasibility": "not evaluated",
    }
    return export_tracking_motion(
        output,
        model_file=model_file,
        qpos=poses,
        joint_names=joint_names,
        fps=target_fps,
        metadata=provenance,
    )
