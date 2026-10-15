"""Raw AMASS semantics and temporal boundaries, independent of private datasets."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from unilab.data import AmassWindowDataset, load_amass_clip, split_amass_files


def _fields(frames: int = 9, offset: float = 0.0) -> dict:
    marker = np.arange(frames, dtype=np.float64)[:, None] + offset
    return {
        "poses": np.repeat(marker, 156, axis=1),
        "trans": np.repeat(marker, 3, axis=1),
        "betas": np.arange(16, dtype=np.float64),
        "gender": np.array("male"),
        "mocap_framerate": np.array(120.0),
        "dmpls": np.repeat(marker, 8, axis=1),
    }


def _write(path: Path, fields: dict | None = None, *, offset: float = 0.0) -> Path:
    np.savez(path, **(_fields(offset=offset) if fields is None else fields))
    return path


def test_reader_preserves_parameters_precision_and_unknown_model(tmp_path):
    fields = _fields()
    # Extra object annotations must not force execution of a pickle loader.
    fields["labels"] = np.array({"unused": "annotation"}, dtype=object)
    path = _write(tmp_path / "clip.npz", fields)
    clip = load_amass_clip(path)
    assert clip.path == path.resolve()
    assert clip.num_frames == 9
    assert clip.fps == 120
    assert clip.duration_seconds == pytest.approx(8 / 120)
    assert clip.model_type == "unknown"  # 156 columns are not a model declaration.
    assert clip.gender == "male"
    assert "labels" in clip.source_fields
    for key in ("poses", "trans", "betas", "dmpls"):
        actual = getattr(clip, key)
        np.testing.assert_array_equal(actual, fields[key])
        assert actual.dtype == np.float64
        assert not actual.flags.writeable
    path.unlink()
    assert clip.poses[8, 0] == 8  # no dependence on an open NPZ handle


@pytest.mark.parametrize("key", ["mocap_framerate", "mocap_frame_rate"])
def test_frame_rate_keys_and_explicit_metadata(tmp_path, key):
    fields = _fields()
    del fields["mocap_framerate"]
    fields[key] = np.array([59.94])
    fields["gender"] = np.array(b"female")
    fields["surface_model_type"] = np.array("smplh")
    clip = load_amass_clip(_write(tmp_path / "clip.npz", fields), model_type="smplh")
    assert clip.fps == 59.94
    assert clip.fps_field == key
    assert clip.gender == "female"
    assert clip.model_type == "smplh"
    with pytest.raises(ValueError, match="conflicting model_type"):
        load_amass_clip(clip.path, model_type="smplx")


def test_missing_model_can_be_declared_without_body_weights(tmp_path):
    path = _write(tmp_path / "clip.npz")
    assert load_amass_clip(path, model_type="smplh").model_type == "smplh"


@pytest.mark.parametrize("missing", ["poses", "trans", "betas", "gender", "mocap_framerate"])
def test_reader_rejects_missing_fields(tmp_path, missing):
    fields = _fields()
    del fields[missing]
    with pytest.raises(ValueError, match="missing"):
        load_amass_clip(_write(tmp_path / "bad.npz", fields))


@pytest.mark.parametrize("fps", [0, -1, np.nan, np.inf, True, "120", [30, 60]])
def test_reader_rejects_invalid_fps(tmp_path, fps):
    fields = _fields()
    fields["mocap_framerate"] = np.asarray(fps)
    with pytest.raises(ValueError, match="frame rate|finite and positive"):
        load_amass_clip(_write(tmp_path / "bad.npz", fields))


def test_reader_rejects_conflicting_fps_aliases(tmp_path):
    fields = _fields()
    fields["mocap_frame_rate"] = np.array(60.0)
    with pytest.raises(ValueError, match="conflicting frame rates"):
        load_amass_clip(_write(tmp_path / "bad.npz", fields))


@pytest.mark.parametrize("key", ["poses", "trans", "betas", "dmpls"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_reader_rejects_nonfinite_parameters(tmp_path, key, invalid):
    fields = _fields()
    fields[key].flat[0] = invalid
    with pytest.raises(ValueError, match="finite floating-point"):
        load_amass_clip(_write(tmp_path / "bad.npz", fields))


@pytest.mark.parametrize(
    "key,value",
    [
        ("poses", np.zeros((9, 155))),
        ("poses", np.zeros((0, 156))),
        ("poses", np.zeros((9, 52, 3))),
        ("poses", np.zeros((9, 156), dtype=np.int32)),
        ("trans", np.zeros((8, 3))),
        ("trans", np.zeros((9, 4))),
        ("betas", np.zeros((9, 16))),
        ("betas", np.zeros(0)),
        ("dmpls", np.zeros((8, 8))),
        ("dmpls", np.zeros((9, 0))),
        ("gender", np.array("unspecified")),
        ("gender", np.array(["male", "female"])),
    ],
)
def test_reader_rejects_invalid_shapes_and_metadata(tmp_path, key, value):
    fields = _fields()
    fields[key] = value
    with pytest.raises(ValueError):
        load_amass_clip(_write(tmp_path / "bad.npz", fields))


def test_robot_npz_is_not_an_amass_clip(tmp_path):
    with pytest.raises(ValueError, match="packed-pose AMASS"):
        load_amass_clip(_write(tmp_path / "robot.npz", {"joint_pos": np.zeros((9, 29))}))


def test_windows_preserve_clip_boundaries_time_and_order(tmp_path):
    first = _write(tmp_path / "a.npz")
    fields = _fields(frames=7, offset=1000)
    fields["mocap_framerate"] = np.array(60.0)
    second = _write(tmp_path / "b.npz", fields)
    dataset = AmassWindowDataset([first, second], window_length=4, stride=2)
    assert len(dataset) == 5  # a: starts 0,2,4; b: starts 0,2
    for i, (start, base, path, fps) in enumerate(
        [
            (0, 0, first, 120),
            (2, 0, first, 120),
            (4, 0, first, 120),
            (0, 1000, second, 60),
            (2, 1000, second, 60),
        ]
    ):
        sample = dataset[i]
        np.testing.assert_array_equal(sample["poses"][:, 0], np.arange(start, start + 4) + base)
        np.testing.assert_array_equal(sample["trans"][:, 0], sample["poses"][:, 0])
        np.testing.assert_array_equal(sample["dmpls"][:, 0], sample["poses"][:, 0])
        np.testing.assert_allclose(sample["frame_times"], np.arange(start, start + 4) / fps)
        assert sample["clip_id"] == str(path.resolve())
        assert sample["start_frame"] == start
        assert sample["stop_frame"] == start + 4
        assert sample["fps"] == fps
        assert sample["betas"].shape == (16,)
    np.testing.assert_array_equal(dataset[-1]["poses"], dataset[4]["poses"])
    for invalid in (-6, 5):
        with pytest.raises(IndexError):
            dataset[invalid]


def test_samples_and_batches_do_not_mutate_the_cached_clip(tmp_path):
    dataset = AmassWindowDataset(_write(tmp_path / "clip.npz"), window_length=4)
    changed = dataset[0]
    changed["poses"][:] = -999
    changed["betas"][:] = -999
    assert dataset[0]["poses"][0, 0] == 0
    assert dataset[0]["betas"][0] == 0
    batch = next(iter(DataLoader(dataset, batch_size=3)))
    assert batch["poses"].shape == (3, 4, 156)
    assert batch["trans"].shape == (3, 4, 3)
    assert batch["betas"].shape == (3, 16)
    assert batch["dmpls"].shape == (3, 4, 8)
    assert batch["poses"].dtype == torch.float64
    torch.testing.assert_close(batch["poses"][:, 0, 0], torch.arange(3, dtype=torch.float64))
    # Dataset serialization remains usable by DataLoader workers.
    restored = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(restored[2]["poses"], dataset[2]["poses"])


@pytest.mark.parametrize("argument", ["window_length", "stride"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_dataset_rejects_invalid_window_configuration(tmp_path, argument, value):
    kwargs = {"window_length": 4, "stride": 1, argument: value}
    with pytest.raises(ValueError, match="positive integer"):
        AmassWindowDataset(_write(tmp_path / "clip.npz"), **kwargs)


def test_dataset_rejects_short_clips_and_mixed_layouts(tmp_path):
    first = _write(tmp_path / "a.npz")
    with pytest.raises(ValueError, match="shorter"):
        AmassWindowDataset(first, window_length=10)
    fields = _fields()
    fields["poses"] = np.zeros((9, 72))
    second = _write(tmp_path / "b.npz", fields)
    with pytest.raises(ValueError, match="incompatible"):
        AmassWindowDataset([first, second], window_length=4)


def test_dmpls_are_optional_but_not_silently_filled(tmp_path):
    fields = _fields()
    del fields["dmpls"]
    first = _write(tmp_path / "a.npz", fields)
    sample = AmassWindowDataset(first, window_length=4)[0]
    assert "dmpls" not in sample
    second = _write(tmp_path / "b.npz")
    with pytest.raises(ValueError, match="incompatible"):
        AmassWindowDataset([first, second], window_length=4)


def test_split_is_deterministic_disjoint_and_precedes_windowing(tmp_path):
    paths = [_write(tmp_path / f"clip{i}.npz", offset=100 * i) for i in range(5)]
    train, validation = split_amass_files(paths, validation_fraction=0.4, seed=42)
    assert (train, validation) == split_amass_files(
        reversed(paths), validation_fraction=0.4, seed=42
    )
    assert len(train) == 3 and len(validation) == 2
    assert not set(train).intersection(validation)
    assert set(train + validation) == set(paths)
    train_data = AmassWindowDataset(train, window_length=4)
    validation_data = AmassWindowDataset(validation, window_length=4)
    assert {train_data[i]["clip_id"] for i in range(len(train_data))}.isdisjoint(
        validation_data[i]["clip_id"] for i in range(len(validation_data))
    )


def test_aliases_empty_inputs_and_invalid_splits_are_rejected(tmp_path):
    first = _write(tmp_path / "a.npz")
    alias = tmp_path / "alias.npz"
    alias.symlink_to(first)
    with pytest.raises(ValueError, match="Duplicate"):
        split_amass_files([first, alias])
    with pytest.raises(ValueError, match="At least one"):
        AmassWindowDataset([], window_length=4)
    with pytest.raises(ValueError, match="At least two"):
        split_amass_files([first])
    second = _write(tmp_path / "b.npz")
    for fraction in (0, 1, np.nan, True, "0.1"):
        with pytest.raises(ValueError, match="validation_fraction"):
            split_amass_files([first, second], validation_fraction=fraction)


def test_explicit_subject_groups_are_kept_together(tmp_path):
    paths = [_write(tmp_path / f"clip{i}.npz") for i in range(6)]
    groups = {path: f"CMU/subject{i // 2}" for i, path in enumerate(paths)}
    train, validation = split_amass_files(paths, validation_fraction=0.3, groups=groups)
    assert len(validation) == 2
    assert {groups[path] for path in train}.isdisjoint(groups[path] for path in validation)
    with pytest.raises(ValueError, match="exactly"):
        split_amass_files(paths, groups={paths[0]: "subject1"})
    with pytest.raises(ValueError, match="two groups"):
        split_amass_files(paths, groups={path: "same_subject" for path in paths})
    with pytest.raises(ValueError, match="nonempty"):
        split_amass_files(paths, groups={path: "" for path in paths})


def test_import_does_not_initialize_robot_tasks_or_training_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import unilab.data; "
            "assert not any(x == 'unisim' or x == 'uni_rl' or "
            "x.startswith('unilab.tasks') for x in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
