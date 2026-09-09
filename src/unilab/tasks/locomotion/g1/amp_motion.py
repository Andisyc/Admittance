"""Cold-path expert motion resolution for the G1 AMP walk task."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.assets.hub import resolve_motion_files
from unilab.tasks.locomotion.g1.amp_terms import (
    AMP_ANCHOR_BODY_INDEX,
    AMP_ANCHOR_BODY_NAME,
    AMP_BODY_INDICES,
    AMP_BODY_NAMES,
    compute_g1_amp_observation_from_ordered,
)

_REQUIRED_ARRAYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
DEFAULT_G1_AMP_MANIFEST = ASSETS_ROOT_PATH / "motions/g1/amp_walk/manifest.json"
REGAIT_V013_A022_MANIFEST = (
    ASSETS_ROOT_PATH / "motions/g1/regait_v013_a022_full_horizon_313/manifest.json"
)
REGAIT_V017_A022_MANIFEST = (
    ASSETS_ROOT_PATH / "motions/g1/regait_v017_a022_313/manifest.json"
)
G1_AMP_EXPERT_MANIFESTS = {
    "amp_mjlab": DEFAULT_G1_AMP_MANIFEST,
    "regait_v013_a022": REGAIT_V013_A022_MANIFEST,
    "regait_v017_a022": REGAIT_V017_A022_MANIFEST,
}


@dataclass(frozen=True)
class ResolvedG1AmpMotion:
    name: str
    path: Path
    sha256: str
    frames: int
    fps: float


@dataclass(frozen=True)
class ResolvedG1AmpMotionManifest:
    path: Path
    motions: tuple[ResolvedG1AmpMotion, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class G1AmpExpertTransitions:
    current: np.ndarray
    next: np.ndarray
    motion_index: np.ndarray
    frame_index: np.ndarray
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "motion_class": "forward_walk",
        "adjacency": "within_clip_t_to_t_plus_1",
        "source_repository": "AMP_mjlab",
        "source_relative_directory": "src/assets/motions/g1/amp/WalkandRun",
        "body_names": list(AMP_BODY_NAMES),
        "body_indices": list(AMP_BODY_INDICES),
        "anchor_body_name": AMP_ANCHOR_BODY_NAME,
        "anchor_body_index": AMP_ANCHOR_BODY_INDEX,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"G1 AMP manifest {field} does not match the frozen contract")


def resolve_g1_amp_motion_manifest(
    manifest_path: str | Path = DEFAULT_G1_AMP_MANIFEST,
) -> ResolvedG1AmpMotionManifest:
    """Resolve and identity-check all expert files before runtime allocation."""
    selected = G1_AMP_EXPERT_MANIFESTS.get(str(manifest_path), manifest_path)
    path = Path(selected).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"G1 AMP manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("G1 AMP manifest root must be an object")
    _validate_manifest_contract(manifest)
    entries = manifest.get("motions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("G1 AMP manifest motions must be a non-empty list")

    relative_dir = (
        path.parent.relative_to(ASSETS_ROOT_PATH) if path.is_relative_to(ASSETS_ROOT_PATH) else None
    )
    resolved: list[ResolvedG1AmpMotion] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("G1 AMP manifest motion entries must be objects")
        filename = entry.get("file")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.startswith("walk_forward_")
            or not filename.endswith(".npz")
        ):
            raise ValueError(f"invalid forward-walk expert filename: {filename!r}")
        if filename in seen:
            raise ValueError(f"duplicate G1 AMP expert file: {filename}")
        seen.add(filename)
        candidate = path.parent / filename
        request = str(relative_dir / filename) if relative_dir is not None else str(candidate)
        resolved_motion = resolve_motion_files(request)
        if not isinstance(resolved_motion, str):
            raise TypeError(f"motion resolver returned multiple files for {filename}")
        motion_path = Path(cast(str, resolved_motion)).resolve()
        expected_sha = entry.get("sha256")
        actual_sha = _sha256(motion_path)
        if not isinstance(expected_sha, str) or actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha}, got {actual_sha}"
            )
        frames = entry.get("frames")
        fps = entry.get("fps")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 2:
            raise ValueError(f"invalid frame count for {filename}: {frames!r}")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) <= 0.0:
            raise ValueError(f"invalid FPS for {filename}: {fps!r}")
        resolved.append(ResolvedG1AmpMotion(filename, motion_path, actual_sha, frames, float(fps)))
    provenance = {
        "schema_version": 1,
        "motion_class": "forward_walk",
        "adjacency": "within_clip_t_to_t_plus_1",
        "source_repository": "AMP_mjlab",
        "source_relative_directory": "src/assets/motions/g1/amp/WalkandRun",
        "body_names": list(AMP_BODY_NAMES),
        "body_indices": list(AMP_BODY_INDICES),
        "anchor_body_name": AMP_ANCHOR_BODY_NAME,
        "anchor_body_index": AMP_ANCHOR_BODY_INDEX,
        "motions": [
            {
                "file": motion.name,
                "sha256": motion.sha256,
                "frames": motion.frames,
                "fps": motion.fps,
            }
            for motion in resolved
        ],
    }
    return ResolvedG1AmpMotionManifest(path, tuple(resolved), provenance)


def load_g1_amp_expert_transitions(
    manifest: ResolvedG1AmpMotionManifest | str | Path = DEFAULT_G1_AMP_MANIFEST,
) -> G1AmpExpertTransitions:
    """Load finite adjacent expert pairs without wrapping clip boundaries."""
    resolved = (
        manifest
        if isinstance(manifest, ResolvedG1AmpMotionManifest)
        else resolve_g1_amp_motion_manifest(manifest)
    )
    features: list[np.ndarray] = []
    for motion in resolved.motions:
        with np.load(motion.path, allow_pickle=False) as source:
            missing = (set(_REQUIRED_ARRAYS) | {"fps"}) - set(source.files)
            if missing:
                raise ValueError(f"motion {motion.name} is missing arrays: {sorted(missing)}")
            arrays = {name: np.asarray(source[name]) for name in _REQUIRED_ARRAYS}
            frames = arrays["body_pos_w"].shape[0]
            if frames != motion.frames:
                raise ValueError(
                    f"frame count mismatch for {motion.name}: expected {motion.frames}, got {frames}"
                )
            fps = np.asarray(source["fps"]).reshape(-1)
            if fps.size != 1 or float(fps[0]) != motion.fps:
                raise ValueError(f"FPS mismatch for {motion.name}")
        features.append(compute_g1_amp_observation_from_ordered(**arrays))

    counts = np.asarray([feature.shape[0] - 1 for feature in features], dtype=np.int64)
    current = np.ascontiguousarray(np.concatenate([feature[:-1] for feature in features]))
    next_state = np.ascontiguousarray(np.concatenate([feature[1:] for feature in features]))
    motion_index = np.repeat(np.arange(len(features), dtype=np.int64), counts)
    frame_index = np.concatenate([np.arange(count, dtype=np.int64) for count in counts])
    return G1AmpExpertTransitions(
        current=current,
        next=next_state,
        motion_index=motion_index,
        frame_index=frame_index,
        provenance=resolved.provenance,
    )


__all__ = [
    "DEFAULT_G1_AMP_MANIFEST",
    "G1_AMP_EXPERT_MANIFESTS",
    "G1AmpExpertTransitions",
    "ResolvedG1AmpMotion",
    "ResolvedG1AmpMotionManifest",
    "REGAIT_V013_A022_MANIFEST",
    "REGAIT_V017_A022_MANIFEST",
    "load_g1_amp_expert_transitions",
    "resolve_g1_amp_motion_manifest",
]
