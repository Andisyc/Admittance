"""Named world-frame skeletons reconstructed from explicit SMPL-H parameters.

Only joint kinematics are needed: the linear joint regressor commutes with
shape blending, so joint-space shape directions give the same rest joints as
regressing the fully shaped vertices. Pose-corrective surface displacements
and skinning do not change the kinematic joints. No surface model is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unilab.data.amass import AmassClip
from unilab.utils.rotation import np_quat_apply_batched, np_quat_mul_batched

SMPL_BODY_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


@dataclass(frozen=True)
class HumanMotion:
    """Meter-valued Z-up world positions and wxyz world orientations, [T,J,*].

    The caller must establish the source convention; no axis is inferred from
    motion statistics. GMR consumes this declared convention.
    """

    names: tuple[str, ...]
    positions: np.ndarray
    quaternions: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        p = np.asarray(self.positions, dtype=np.float64)
        q = np.asarray(self.quaternions, dtype=np.float64)
        if (
            not self.names
            or len(set(self.names)) != len(self.names)
            or any(not isinstance(n, str) or not n for n in self.names)
        ):
            raise ValueError("human body names must be unique nonempty strings")
        if p.ndim != 3 or p.shape[0] < 2 or p.shape[1:] != (len(self.names), 3):
            raise ValueError("human positions must be [T>=2,J,3]")
        if q.shape != (*p.shape[:2], 4) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("human positions/quaternions must be finite, with matching axes")
        if not np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-7, rtol=0):
            raise ValueError("human quaternions must be unit wxyz")
        if isinstance(self.fps, bool) or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("human fps must be finite and positive")
        p, q = p.copy(), q.copy()
        p.setflags(write=False)
        q.setflags(write=False)
        object.__setattr__(self, "positions", p)
        object.__setattr__(self, "quaternions", q)
        object.__setattr__(self, "names", tuple(self.names))

    def frame(self, index: int) -> dict[str, list[np.ndarray]]:
        # GMR mutates its input dictionary; never expose owned source arrays.
        return {
            name: [self.positions[index, j].copy(), self.quaternions[index, j].copy()]
            for j, name in enumerate(self.names)
        }


def _numeric(data, name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"body model is missing {name!r}")
    value = data[name]
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"body model {name} must be dense finite numeric data")
    return np.asarray(value, dtype=np.float64)


def reconstruct_smplh(
    clip: AmassClip, model_file: str | Path, *, dmpl_file: str | Path | None = None
) -> HumanMotion:
    """Reconstruct common SMPL-H body frames without truncating shape coefficients.

    ``clip.model_type`` must explicitly be ``smplh`` (metadata or caller
    declaration), packed poses must contain all 52 axis-angle rotations, and
    weights must be matching, dense NPZ files. Nonzero DMPL coefficients require
    their matching ``eigvec`` file. Files with pickle-only sparse objects fail
    rather than enabling arbitrary object loading.
    """
    if clip.model_type != "smplh":
        raise ValueError("SMPL-H reconstruction requires explicit model_type='smplh'")
    if clip.poses.shape[1] != 156 or clip.num_frames < 2:
        raise ValueError("SMPL-H requires [T>=2,156] packed axis-angle poses")
    with np.load(model_file, allow_pickle=False) as model:
        template = _numeric(model, "v_template")
        dirs = _numeric(model, "shapedirs")
        regressor = _numeric(model, "J_regressor")
        tree = _numeric(model, "kintree_table")
        if "gender" in model:
            gender = model["gender"].item()
            gender = gender.decode() if isinstance(gender, bytes) else str(gender)
            if gender.lower() != clip.gender:
                raise ValueError("body model gender conflicts with AMASS gender")
    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError("v_template must be [V,3]")
    if dirs.ndim != 3 or dirs.shape[:2] != template.shape or dirs.shape[2] < clip.betas.size:
        raise ValueError("body shapedirs cannot represent every supplied beta coefficient")
    if regressor.shape != (52, len(template)) or tree.shape != (2, 52):
        raise ValueError("expected a matching 52-joint SMPL-H model")
    if not np.array_equal(tree, np.floor(tree)) or len(set(tree[1])) != 52:
        raise ValueError("invalid body-model joint identifiers")
    indices = {int(identity): i for i, identity in enumerate(tree[1])}
    parents = [-1]
    for j in range(1, 52):
        parent = indices.get(int(tree[0, j]), -1)
        if not 0 <= parent < j:
            raise ValueError("body-model joints must be in parent-before-child order")
        parents.append(parent)
    # J(v_template + shapedirs*beta + dmpldirs*dmpl) in joint space.
    rest = regressor @ template
    shape_joints = np.einsum("jv,vck->jck", regressor, dirs[:, :, : clip.betas.size])
    rest = rest + np.einsum("jck,k->jc", shape_joints, clip.betas)
    rest = np.broadcast_to(rest, (clip.num_frames, 52, 3)).copy()
    if clip.dmpls is not None and np.any(clip.dmpls != 0):
        if dmpl_file is None:
            raise ValueError("nonzero AMASS DMPL coefficients require matching dmpl_file")
        with np.load(dmpl_file, allow_pickle=False) as data:
            dynamic_dirs = _numeric(data, "eigvec")
        if (
            dynamic_dirs.ndim != 3
            or dynamic_dirs.shape[:2] != template.shape
            or dynamic_dirs.shape[2] < clip.dmpls.shape[1]
        ):
            raise ValueError("DMPL eigvec cannot represent every supplied coefficient")
        dynamic_joints = np.einsum(
            "jv,vck->jck", regressor, dynamic_dirs[:, :, : clip.dmpls.shape[1]]
        )
        rest += np.einsum("jck,tk->tjc", dynamic_joints, clip.dmpls)
    pose = np.asarray(clip.poses, dtype=np.float64).reshape(clip.num_frames, 52, 3)
    angle = np.linalg.norm(pose, axis=-1, keepdims=True)
    local = np.concatenate(
        (np.cos(angle / 2), pose * (0.5 * np.sinc(angle / (2 * np.pi)))), axis=-1
    )
    pos = np.empty((clip.num_frames, 22, 3))
    rot = np.empty((clip.num_frames, 22, 4))
    pos[:, 0], rot[:, 0] = rest[:, 0] + clip.trans, local[:, 0]
    for j in range(1, 22):
        parent = parents[j]
        pos[:, j] = pos[:, parent] + np_quat_apply_batched(
            rot[:, parent], rest[:, j] - rest[:, parent]
        )
        rot[:, j] = np_quat_mul_batched(rot[:, parent], local[:, j])
    return HumanMotion(SMPL_BODY_NAMES, pos, rot, clip.fps)


def resample_human_motion(motion: HumanMotion, fps: int) -> HumanMotion:
    """Linear position / quaternion SLERP on an explicit grid, no extrapolation."""
    from scipy.spatial.transform import Rotation, Slerp

    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("target fps must be a positive integer")
    source_time = np.arange(len(motion.positions), dtype=np.float64) / motion.fps
    target_time = np.arange(int(np.floor(source_time[-1] * fps + 1e-9)) + 1) / fps
    if len(target_time) < 2:
        raise ValueError("clip is too short for target frame rate")
    p = np.empty((len(target_time), len(motion.names), 3))
    q = np.empty((len(target_time), len(motion.names), 4))
    for j in range(len(motion.names)):
        for axis in range(3):
            p[:, j, axis] = np.interp(target_time, source_time, motion.positions[:, j, axis])
        # scipy's default ordering is xyzw; keep both conversions explicit.
        rotations = Rotation.from_quat(motion.quaternions[:, j, [1, 2, 3, 0]])
        q[:, j] = Slerp(source_time, rotations)(target_time).as_quat()[:, [3, 0, 1, 2]]
    return HumanMotion(motion.names, p, q, float(fps))
