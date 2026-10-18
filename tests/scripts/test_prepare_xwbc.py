"""Data-only archive extraction must not escape or replace verified source files."""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pytest
from scripts.motion.prepare_xwbc import (
    ARCHIVE_SHA256,
    REPOSITORY,
    REVISION,
    _convert,
    _extract_cmu,
)

from unilab.data.retarget import file_sha256


def _archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for name, payload, symlink in entries:
            entry = tarfile.TarInfo(name)
            if symlink:
                entry.type, entry.linkname = tarfile.SYMTYPE, "/tmp/elsewhere"
                archive.addfile(entry)
            else:
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))


def test_extracts_only_cmu_counts_collection_and_verifies_existing(tmp_path):
    archive, destination = tmp_path / "data.tar.gz", tmp_path / "raw"
    name = "unitree_g1_29dof/CMU__01__01_01_stageii.npz"
    _archive(archive, [(name, b"one", False), ("unitree_g1_29dof/KIT__1__walk.npz", b"two", False)])
    paths, count = _extract_cmu(archive, destination)
    assert count == 2 and len(paths) == 1 and paths[0].read_bytes() == b"one"
    assert _extract_cmu(archive, destination) == (paths, count)
    paths[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        _extract_cmu(archive, destination)
    assert paths[0].read_bytes() == b"changed"


@pytest.mark.parametrize(
    "name,symlink",
    [
        ("../escape.npz", False),
        ("/tmp/escape.npz", False),
        ("unitree_g1_29dof/link.npz", True),
        ("unitree_g1_29dof/code.py", False),
    ],
)
def test_rejects_non_data_and_escaping_members(tmp_path, name, symlink):
    archive = tmp_path / "data.tar.gz"
    _archive(archive, [(name, b"payload", symlink)])
    with pytest.raises(ValueError, match="unexpected archive member"):
        _extract_cmu(archive, tmp_path / "raw")
    assert not (tmp_path / "escape.npz").exists()


@pytest.mark.parametrize("requested_names", [("hip", "knee"), ("knee", "hip")])
def test_reused_output_must_preserve_requested_joint_order(tmp_path, requested_names):
    source, output = tmp_path / "CMU__01__walk.npz", tmp_path / "target.npz"
    source.write_bytes(b"pinned input")
    metadata = {
        "source_dataset": "AMASS retargeted by X-WBC; CMU",
        "repository": REPOSITORY,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "source_member": f"unitree_g1_29dof/{source.name}",
        "preprocessor_sha256": "implementation",
        "source_sha256": file_sha256(source),
    }
    np.savez(
        output,
        metadata_json=np.array(json.dumps(metadata)),
        joint_names=np.array(["hip", "knee"]),
        joint_pos=np.zeros((64, 2)),
    )
    job = (source, output, tmp_path / "unused.xml", requested_names, "implementation")
    if requested_names == ("hip", "knee"):
        assert _convert(job)["status"] == "accepted"
    else:
        with pytest.raises(ValueError, match="different joint order"):
            _convert(job)
    with np.load(output, allow_pickle=False) as arrays:
        assert arrays["joint_names"].tolist() == ["hip", "knee"]
