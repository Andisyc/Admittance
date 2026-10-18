"""Independent kinematic fixtures, not evidence of licensed model availability."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from unilab.data.amass import load_amass_clip
from unilab.data.human_motion import (
    HumanMotion,
    reconstruct_smplh,
    resample_human_motion,
)


@pytest.fixture
def human_case(tmp_path):
    rng = np.random.default_rng(121)
    parents = np.array(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        + [20] * 15
        + [21] * 15
    )
    # Nontrivial dense regression detects accidental assumptions that joints
    # simply equal vertices, and coefficient 15/DMPL 7 must not disappear.
    regressor = rng.uniform(size=(52, 57))
    regressor /= regressor.sum(axis=1, keepdims=True)
    weights = {
        "v_template": rng.normal(size=(57, 3)),
        "shapedirs": rng.normal(scale=0.03, size=(57, 3, 16)),
        "J_regressor": regressor,
        "kintree_table": np.vstack((parents, np.arange(52))),
        "gender": np.array("male"),
    }
    dynamic = rng.normal(scale=0.02, size=(57, 3, 8))
    fields = {
        "poses": rng.normal(scale=0.25, size=(4, 156)),
        "trans": np.array([[1, -2, 3], [1.1, -1.8, 3.2], [1.2, -1.7, 3.3], [1.3, -1.5, 3.4]]),
        "betas": np.linspace(0.1, 1.6, 16),
        "dmpls": rng.normal(size=(4, 8)),
        "gender": np.array("male"),
        "surface_model_type": np.array("smplh"),
        "mocap_framerate": np.array(120.0),
    }
    model_file, dmpl_file, clip_file = (tmp_path / n for n in ("model.npz", "dmpl.npz", "clip.npz"))
    np.savez(model_file, **weights)
    np.savez(dmpl_file, eigvec=dynamic)
    np.savez(clip_file, **fields)
    return load_amass_clip(clip_file), model_file, dmpl_file, weights, dynamic, parents


def test_dense_smplh_all_shape_dynamic_coefficients_and_global_frames(human_case):
    clip, model, dmpl, weights, dynamic, parents = human_case
    result = reconstruct_smplh(clip, model, dmpl_file=dmpl)
    positions, rotations = [], []
    for frame in range(clip.num_frames):
        # Oracle first constructs the FULL shaped/dynamic mesh, then regresses
        # joints. It does not use production's joint-space einsum or quaternion FK.
        vertices = weights["v_template"].copy()
        for k, beta in enumerate(clip.betas):
            vertices += weights["shapedirs"][:, :, k] * beta
        for k, coefficient in enumerate(clip.dmpls[frame]):
            vertices += dynamic[:, :, k] * coefficient
        rest = weights["J_regressor"] @ vertices
        # SciPy's native backend may request a writable buffer; the AMASS
        # reader deliberately keeps its owned arrays immutable.
        local = Rotation.from_rotvec(clip.poses[frame].reshape(52, 3).copy()).as_matrix()
        p, r = np.empty((22, 3)), np.empty((22, 3, 3))
        p[0], r[0] = rest[0] + clip.trans[frame], local[0]
        for joint in range(1, 22):
            parent = parents[joint]
            p[joint] = p[parent] + r[parent] @ (rest[joint] - rest[parent])
            r[joint] = r[parent] @ local[joint]
        positions.append(p)
        rotations.append(r)
    np.testing.assert_allclose(result.positions, positions, atol=2e-14)
    actual = Rotation.from_quat(result.quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]])
    np.testing.assert_allclose(actual.as_matrix().reshape(4, 22, 3, 3), rotations, atol=2e-14)
    assert result.fps == 120
    assert result.names[0] == "pelvis" and result.names[21] == "right_wrist"
    assert not result.positions.flags.writeable
    frame = result.frame(0)
    frame["pelvis"][0][:] = 999
    frame["pelvis"][1][:] = 999
    np.testing.assert_allclose(result.positions, positions, atol=2e-14)
    np.testing.assert_allclose(np.linalg.norm(result.quaternions, axis=-1), 1)


@pytest.mark.parametrize("model_type", ["unknown", "smpl", "smplx"])
def test_reconstruction_does_not_guess_model_type(human_case, model_type):
    clip, model, dmpl, *_ = human_case
    with pytest.raises(ValueError, match="explicit model_type"):
        reconstruct_smplh(replace(clip, model_type=model_type), model, dmpl_file=dmpl)


def test_nonzero_dmpl_requires_weights_but_zero_dmpl_does_not(human_case):
    clip, model, *_ = human_case
    with pytest.raises(ValueError, match="nonzero.*DMPL"):
        reconstruct_smplh(clip, model)
    zero = replace(clip, dmpls=np.zeros_like(clip.dmpls))
    absent = replace(clip, dmpls=None)
    np.testing.assert_array_equal(
        reconstruct_smplh(zero, model).positions, reconstruct_smplh(absent, model).positions
    )


@pytest.mark.parametrize("kind", ["missing", "too_few_coefficients", "wrong_vertices"])
def test_dmpl_model_requires_every_dynamic_coefficient(human_case, tmp_path, kind):
    clip, model, _, _, dynamic, _ = human_case
    bad = tmp_path / "bad_dynamic.npz"
    if kind == "missing":
        np.savez(bad, unrelated=np.zeros(1))
    else:
        truncated = dynamic[:, :, :7] if kind == "too_few_coefficients" else dynamic[:-1]
        np.savez(bad, eigvec=truncated)
    with pytest.raises(ValueError, match="eigvec|coefficient"):
        reconstruct_smplh(clip, model, dmpl_file=bad)


@pytest.mark.parametrize("frames,width", [(1, 156), (4, 72), (4, 165)])
def test_reconstruction_rejects_incomplete_packed_pose(human_case, frames, width):
    clip, model, dmpl, *_ = human_case
    malformed = replace(clip, poses=np.zeros((frames, width)))
    with pytest.raises(ValueError, match="packed axis-angle"):
        reconstruct_smplh(malformed, model, dmpl_file=dmpl)


@pytest.mark.parametrize("missing", ["v_template", "shapedirs", "J_regressor", "kintree_table"])
def test_missing_dense_model_fields_are_rejected(human_case, tmp_path, missing):
    clip, _, dmpl, weights, *_ = human_case
    bad = tmp_path / "missing.npz"
    np.savez(bad, **{key: value for key, value in weights.items() if key != missing})
    with pytest.raises(ValueError, match="missing"):
        reconstruct_smplh(clip, bad, dmpl_file=dmpl)


@pytest.mark.parametrize(
    "kind", ["ten_betas", "wrong_joints", "wrong_gender", "object", "bad_tree"]
)
def test_incompatible_model_does_not_silently_truncate(human_case, tmp_path, kind):
    clip, _, dmpl, weights, *_ = human_case
    weights = dict(weights)
    if kind == "ten_betas":
        weights["shapedirs"] = weights["shapedirs"][:, :, :10]
    elif kind == "wrong_joints":
        weights["J_regressor"] = weights["J_regressor"][:24]
    elif kind == "wrong_gender":
        weights["gender"] = np.array("female")
    elif kind == "object":
        weights["J_regressor"] = np.array({"unsafe": "sparse object"}, dtype=object)
    else:
        weights["kintree_table"] = weights["kintree_table"].copy()
        weights["kintree_table"][0, 1] = 2
    bad = tmp_path / "wrong.npz"
    np.savez(bad, **weights)
    with pytest.raises(ValueError):
        reconstruct_smplh(clip, bad, dmpl_file=dmpl)


@pytest.mark.parametrize("source_fps", [60, 120])
@pytest.mark.parametrize("extra_frames", [0, 1])
def test_fifty_hz_uses_time_grid_and_slerp_not_integer_frame_skip(source_fps, extra_frames):
    # Include both endpoint-on-grid and endpoint-between-grid cases.
    times = np.arange(source_fps + extra_frames + 1) / source_fps
    positions = np.stack((times, 2 * times - 1, -times), axis=-1)[:, None]
    initial = Rotation.from_euler("xyz", [0.4, -0.2, 0.7])
    rotations = Rotation.from_rotvec(times[:, None] * np.array([0.2, -0.3, 0.4])) * initial
    q = rotations.as_quat()[:, [3, 0, 1, 2]][:, None]
    q[::2] *= -1  # equivalent rotations with alternating quaternion signs
    source = HumanMotion(("pelvis",), positions, q, float(source_fps))
    result = resample_human_motion(source, 50)
    expected_time = np.arange(51) / 50
    assert len(result.positions) == 51
    assert result.fps == 50
    np.testing.assert_allclose(result.positions[:, 0, 0], expected_time, atol=1e-14)
    expected = Rotation.from_rotvec(expected_time[:, None] * np.array([0.2, -0.3, 0.4])) * initial
    actual = Rotation.from_quat(result.quaternions[:, 0, [1, 2, 3, 0]])
    np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=2e-14)
    assert result.positions[-1, 0, 0] <= times[-1]


@pytest.mark.parametrize("fps", [True, 0, -1, 50.0])
def test_resampling_rejects_invalid_target_fps(fps):
    motion = HumanMotion(("pelvis",), np.zeros((2, 1, 3)), np.tile([1, 0, 0, 0], (2, 1, 1)), 60)
    with pytest.raises(ValueError, match="positive integer"):
        resample_human_motion(motion, fps)


def test_resampling_rejects_clip_shorter_than_two_target_frames():
    motion = HumanMotion(("pelvis",), np.zeros((2, 1, 3)), np.tile([1, 0, 0, 0], (2, 1, 1)), 120)
    with pytest.raises(ValueError, match="too short"):
        resample_human_motion(motion, 50)
