"""Read original AMASS packed-axis-angle NPZ clips and sample temporal windows.

The supported layout contains ``poses[T,D]``, ``trans[T,3]``, ``betas[B]``,
``gender`` and ``mocap_framerate`` (or ``mocap_frame_rate``). ``D`` is not used
to guess a body-model variant. Optional DMPL parameters and declared model
metadata are retained. Split-pose layouts require an explicit future adapter.

Arrays retain their floating-point precision and coordinate conventions. No
body-model weights, simulator, retargeting, normalization or resampling is used.
Window lengths and strides count original frames, so equal-length windows at
different frame rates have different durations. Split files before windowing.
"""

from __future__ import annotations

import math
import operator
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _paths(paths: str | Path | Iterable[str | Path]) -> tuple[Path, ...]:
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    result = tuple(Path(path).expanduser().resolve(strict=True) for path in paths)
    if not result:
        raise ValueError("At least one AMASS NPZ path is required")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate AMASS paths, including aliases, are not allowed")
    for path in result:
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise ValueError(f"Expected an AMASS NPZ file, got {path}")
    return result


def _text(value: np.ndarray, name: str) -> str:
    if value.size != 1 or value.dtype.kind not in "SU":
        raise ValueError(f"{name} must be one text value, without pickled objects")
    item = value.reshape(-1)[0]
    text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
    if not text.strip():
        raise ValueError(f"{name} must not be empty")
    return text.strip().lower()


def _array(value: np.ndarray, name: str) -> np.ndarray:
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    result = value.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AmassClip:
    """One validated clip; arrays are read-only and detached from the NPZ file."""

    path: Path
    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    gender: str
    fps: float
    fps_field: str
    model_type: str
    dmpls: np.ndarray | None
    source_fields: tuple[str, ...]

    @property
    def num_frames(self) -> int:
        return int(self.poses.shape[0])

    @property
    def duration_seconds(self) -> float:
        """Time from the first recorded frame to the last: (T-1)/fps."""
        return (self.num_frames - 1) / self.fps


def load_amass_clip(path: str | Path, *, model_type: str | None = None) -> AmassClip:
    """Read a packed-pose AMASS clip, rejecting robot NPZs and malformed files.

    ``model_type`` may explicitly supply missing model metadata. A conflicting
    declaration is rejected. Without either declaration it remains ``unknown``;
    reconstructing body geometry requires resolving that metadata separately.
    Unrelated archive members (e.g. object-valued annotations) are never loaded.
    """
    source = _paths(path)[0]
    if model_type is not None:
        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("model_type must be nonempty text or None")
        model_type = model_type.strip().lower()
    with np.load(source, allow_pickle=False) as data:
        required = {"poses", "trans", "betas", "gender"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"{source}: missing packed-pose AMASS fields {missing}; "
                "robot NPZs and split-pose layouts are not supported by this reader"
            )
        fps_fields = [key for key in ("mocap_framerate", "mocap_frame_rate") if key in data]
        if not fps_fields:
            raise ValueError(f"{source}: missing mocap_framerate/mocap_frame_rate")
        rates = []
        for key in fps_fields:
            value = data[key]
            if value.size != 1 or value.dtype.kind not in "fiu":
                raise ValueError(f"{source}: {key} must be one positive numeric frame rate")
            rate = float(value.reshape(-1)[0])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"{source}: {key} must be finite and positive")
            rates.append(rate)
        if any(rate != rates[0] for rate in rates):
            raise ValueError(f"{source}: conflicting frame rates {rates}")

        poses = _array(data["poses"], "poses")
        trans = _array(data["trans"], "trans")
        betas = _array(data["betas"], "betas")
        if poses.ndim != 2 or poses.shape[0] < 1 or poses.shape[1] < 3 or poses.shape[1] % 3:
            raise ValueError(f"{source}: poses must have shape [T>0, 3*J>0]")
        if trans.shape != (poses.shape[0], 3):
            raise ValueError(f"{source}: trans must have shape [{poses.shape[0]}, 3]")
        if betas.ndim != 1 or betas.size == 0:
            raise ValueError(f"{source}: betas must be a nonempty per-clip vector")
        gender = _text(data["gender"], "gender")
        if gender not in {"male", "female", "neutral"}:
            raise ValueError(f"{source}: unsupported gender metadata {gender!r}")
        declared = (
            _text(data["surface_model_type"], "surface_model_type")
            if "surface_model_type" in data
            else None
        )
        if declared is not None and model_type is not None and declared != model_type:
            raise ValueError(f"{source}: conflicting model_type {model_type!r} and {declared!r}")
        dmpls = _array(data["dmpls"], "dmpls") if "dmpls" in data else None
        if dmpls is not None and (
            dmpls.ndim != 2 or dmpls.shape[0] != poses.shape[0] or dmpls.shape[1] == 0
        ):
            raise ValueError(f"{source}: dmpls must have shape [T, K>0]")
        return AmassClip(
            source,
            poses,
            trans,
            betas,
            gender,
            rates[0],
            fps_fields[0],
            declared or model_type or "unknown",
            dmpls,
            tuple(data.files),
        )


def split_amass_files(
    paths: str | Path | Iterable[str | Path],
    *,
    validation_fraction: float = 0.1,
    seed: int = 0,
    groups: Mapping[str | Path, str] | None = None,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Deterministically split unique files before generating any windows.

    Without ``groups`` this is clip-level isolation. To isolate subjects, supply
    a complete path-to-subject mapping from the dataset's documented identity.
    No subject identity is guessed from directory names. The validation fraction
    applies to groups when supplied, not to frames or individual clips.
    """
    files = tuple(sorted(_paths(paths)))
    if len(files) < 2:
        raise ValueError("At least two clips are needed for nonempty train/validation splits")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, Real)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    membership = {path: str(path) for path in files}
    if groups is not None:
        membership = {}
        for raw_path, group in groups.items():
            path = Path(raw_path).expanduser().resolve(strict=True)
            if path in membership or not isinstance(group, str) or not group.strip():
                raise ValueError("groups must use unique paths and nonempty text group IDs")
            membership[path] = group
        if set(membership) != set(files):
            raise ValueError("groups must cover exactly the supplied AMASS paths")
    identities = sorted(set(membership.values()))
    if len(identities) < 2:
        raise ValueError("At least two groups are needed for a disjoint split")
    count = min(len(identities) - 1, max(1, math.ceil(len(identities) * validation_fraction)))
    indices = np.random.default_rng(seed).permutation(len(identities))
    held_out = {identities[int(i)] for i in indices[:count]}
    validation = tuple(path for path in files if membership[path] in held_out)
    train = tuple(path for path in files if membership[path] not in held_out)
    return train, validation


def _layout(clip: AmassClip) -> tuple[int, int, str, int | None]:
    return (
        clip.poses.shape[1],
        clip.betas.size,
        clip.model_type,
        None if clip.dmpls is None else clip.dmpls.shape[1],
    )


class AmassWindowDataset(Dataset[dict[str, Any]]):
    """Map-style dataset of full windows, with at most one decoded clip cached.

    Construction validates each clip sequentially; it does not retain the full
    corpus in memory. Clips must share a parameter layout, but may have distinct
    frame rates (retained per sample). Short clips are rejected explicitly.
    Supply already separated train/validation paths. Files must remain unchanged
    during this dataset's lifetime. Returned arrays are independent writable
    copies, so Torch's default DataLoader collation is safe.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        *,
        window_length: int,
        stride: int = 1,
        model_type: str | None = None,
    ) -> None:
        self.paths = _paths(paths)
        self.window_length = _positive_int(window_length, "window_length")
        self.stride = _positive_int(stride, "stride")
        self.model_type = model_type
        self._ends: list[int] = []
        self._frame_counts: list[int] = []
        self._fps: list[float] = []
        layout = None
        total = 0
        for path in self.paths:
            clip = load_amass_clip(path, model_type=model_type)
            if layout is not None and _layout(clip) != layout:
                raise ValueError(f"{path}: incompatible AMASS parameter layout in one dataset")
            layout = _layout(clip)
            if clip.num_frames < self.window_length:
                raise ValueError(f"{path}: clip is shorter than window_length={self.window_length}")
            total += 1 + (clip.num_frames - self.window_length) // self.stride
            self._ends.append(total)
            self._frame_counts.append(clip.num_frames)
            self._fps.append(clip.fps)
        self._layout = layout
        self._cached_index: int | None = None
        self._cached_clip: AmassClip | None = None

    def __len__(self) -> int:
        return self._ends[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        clip_index = bisect_right(self._ends, index)
        if self._cached_index != clip_index:
            clip = load_amass_clip(self.paths[clip_index], model_type=self.model_type)
            if (
                clip.num_frames != self._frame_counts[clip_index]
                or clip.fps != self._fps[clip_index]
                or _layout(clip) != self._layout
            ):
                raise ValueError(f"{clip.path}: clip metadata changed after dataset construction")
            self._cached_clip = clip
            self._cached_index = clip_index
        assert self._cached_clip is not None
        clip = self._cached_clip
        offset = self._ends[clip_index - 1] if clip_index else 0
        start = (index - offset) * self.stride
        stop = start + self.window_length
        result = {
            "poses": clip.poses[start:stop].copy(),
            "trans": clip.trans[start:stop].copy(),
            "betas": clip.betas.copy(),
            "gender": clip.gender,
            "fps": clip.fps,
            "model_type": clip.model_type,
            "clip_id": str(clip.path),
            "start_frame": start,
            "stop_frame": stop,
            "frame_times": np.arange(start, stop, dtype=np.float64) / clip.fps,
        }
        if clip.dmpls is not None:
            result["dmpls"] = clip.dmpls[start:stop].copy()
        return result
