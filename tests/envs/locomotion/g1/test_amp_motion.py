from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from unilab.tasks.locomotion.g1.amp_motion import (
    G1_AMP_EXPERT_MANIFESTS,
    load_g1_amp_expert_transitions,
    resolve_g1_amp_motion_manifest,
)
from unilab.tasks.locomotion.g1.amp_terms import (
    AMP_ANCHOR_BODY_INDEX,
    AMP_ANCHOR_BODY_NAME,
    AMP_BODY_INDICES,
    AMP_BODY_NAMES,
)


def _write_motion(path: Path, offset: float) -> str:
    frames = 3
    pos = np.zeros((frames, 30, 3), dtype=np.float32)
    pos[:, :, 0] = np.arange(frames, dtype=np.float32)[:, None] + offset
    quat = np.zeros((frames, 30, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    linear = np.zeros_like(pos)
    angular = np.zeros_like(pos)
    linear[:, :, 0] = np.arange(frames, dtype=np.float32)[:, None] + offset
    np.savez(
        path,
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=linear,
        body_ang_vel_w=angular,
        fps=np.array(50.0),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_expert_loader_validates_identity_and_never_wraps_clips(tmp_path: Path) -> None:
    entries = []
    for index, offset in enumerate((0.0, 100.0)):
        name = f"walk_forward_test_{index}.npz"
        sha = _write_motion(tmp_path / name, offset)
        entries.append({"file": name, "sha256": sha, "frames": 3, "fps": 50})
    manifest = {
        "schema_version": 1,
        "motion_class": "forward_walk",
        "adjacency": "within_clip_t_to_t_plus_1",
        "source_repository": "AMP_mjlab",
        "source_relative_directory": "src/assets/motions/g1/amp/WalkandRun",
        "body_names": list(AMP_BODY_NAMES),
        "body_indices": list(AMP_BODY_INDICES),
        "anchor_body_name": AMP_ANCHOR_BODY_NAME,
        "anchor_body_index": AMP_ANCHOR_BODY_INDEX,
        "motions": entries,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    resolved = resolve_g1_amp_motion_manifest(path)
    transitions = load_g1_amp_expert_transitions(resolved)
    assert transitions.current.shape == (4, 195)
    assert transitions.next.shape == (4, 195)
    assert transitions.current.dtype == np.float32
    np.testing.assert_array_equal(transitions.motion_index, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(transitions.frame_index, np.array([0, 1, 0, 1]))
    # Velocity features expose adjacency while confirming clip 0 never wraps to clip 1.
    velocity_start = len(AMP_BODY_NAMES) * 9
    np.testing.assert_array_equal(
        transitions.current[:, velocity_start], np.array([0.0, 1.0, 100.0, 101.0])
    )
    np.testing.assert_array_equal(
        transitions.next[:, velocity_start], np.array([1.0, 2.0, 101.0, 102.0])
    )


def test_expert_manifest_identities_keep_amp_mjlab_default_and_offer_regait() -> None:
    assert G1_AMP_EXPERT_MANIFESTS["amp_mjlab"].name == "manifest.json"
    assert G1_AMP_EXPERT_MANIFESTS["amp_mjlab"].parent.name == "amp_walk"
    assert (
        G1_AMP_EXPERT_MANIFESTS["regait_v013_a022"].parent.name
        == "regait_v013_a022_full_horizon_313"
    )
    regait_path = G1_AMP_EXPERT_MANIFESTS["regait_v013_a022"]
    manifest = json.loads(regait_path.read_text(encoding="utf-8"))
    motion = manifest["motions"][0]
    assert motion["candidate_identity"] == (
        "sha256:78e83649d21fbe24852ccb16ead835e4730dec9a9c72d67637794d09a7e00672"
    )
    assert motion["sha256"] == "b036bb0b2ee784344dc082900169b57701242782dff0dcad93930f993e763715"
    assert motion["source_identity"]["sha256"] == (
        "dc1eba2d7f124d99d058562072a2376bcf123ec52b7c2b565abd723c3f688e1f"
    )
    assert not (regait_path.parent / motion["file"]).exists()


def test_v017_manifest_is_separate_from_original_expert_set() -> None:
    baseline = json.loads(G1_AMP_EXPERT_MANIFESTS["amp_mjlab"].read_text())
    assert [entry["file"] for entry in baseline["motions"]] == [
        "walk_forward_loop_002__A022.npz", "walk_forward_loop_002__A024.npz"
    ]
    path = G1_AMP_EXPERT_MANIFESTS["regait_v017_a022"]
    assert path.parent.name == "regait_v017_a022_313"
    generated = json.loads(path.read_text())
    assert len(generated["motions"]) == 1
    motion = generated["motions"][0]
    assert motion["frames"] == 313 and motion["fps"] == 50
    assert motion["candidate_identity"] == (
        "sha256:d41680b08c0a0a436ee3956ecfbd5d3457b7c09bec6ea7cb79799da3dfacdde5"
    )


def test_expert_manifest_hash_mismatch_fails_before_array_loading(tmp_path: Path) -> None:
    name = "walk_forward_bad.npz"
    _write_motion(tmp_path / name, 0.0)
    manifest = {
        "schema_version": 1,
        "motion_class": "forward_walk",
        "adjacency": "within_clip_t_to_t_plus_1",
        "source_repository": "AMP_mjlab",
        "source_relative_directory": "src/assets/motions/g1/amp/WalkandRun",
        "body_names": list(AMP_BODY_NAMES),
        "body_indices": list(AMP_BODY_INDICES),
        "anchor_body_name": AMP_ANCHOR_BODY_NAME,
        "anchor_body_index": AMP_ANCHOR_BODY_INDEX,
        "motions": [{"file": name, "sha256": "0" * 64, "frames": 3, "fps": 50}],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        resolve_g1_amp_motion_manifest(path)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("hash mismatch was accepted")
