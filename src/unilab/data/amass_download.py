"""Prepare a pinned AMASS CMU subset on the machine that will train it.

Network access occurs only when ``prepare_amass_subset`` is explicitly called.
The public mirror is provenance, not a replacement for the AMASS source license.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from unilab.data.amass import load_amass_clip, split_amass_files

REPOSITORY = "realdream-ai/AMASS"
REVISION = "d38490c0e60dcefae81461fc5f4765fe66727831"
_HOST = "https://huggingface.co"
_CLIP_PATH = re.compile(r"raw/CMU/(\d+)/\d+_\d+_poses\.npz")


def _source_tree() -> list[dict[str, Any]]:
    url = f"{_HOST}/api/datasets/{REPOSITORY}/tree/{REVISION}/raw/CMU?recursive=true&limit=1000"
    entries = []
    visited = set()
    while url:
        if url in visited or not url.startswith(f"{_HOST}/api/datasets/{REPOSITORY}/tree/"):
            raise ValueError("Unexpected source-tree pagination URL")
        visited.add(url)
        with urlopen(url, timeout=60) as response:
            page = json.load(response)
            if not isinstance(page, list):
                raise ValueError("Expected a dataset-tree list")
            entries.extend(page)
            links = response.headers.get("Link", "")
        next_page = re.search(r'<([^>]+)>;\s*rel="next"', links)
        url = next_page.group(1) if next_page else ""
    return entries


def _select(tree: list[dict[str, Any]], clip_count: int) -> list[dict[str, Any]]:
    if (
        isinstance(clip_count, bool)
        or not isinstance(clip_count, int)
        or not 1 <= clip_count <= 384
    ):
        raise ValueError("clip_count must be between 1 and 384")
    subjects = defaultdict(list)
    for entry in tree:
        source = entry.get("path", "")
        match = _CLIP_PATH.fullmatch(source)
        size = entry.get("size", 0)
        digest = entry.get("lfs", {}).get("oid", "")
        if (
            match is not None
            and entry.get("type") == "file"
            and isinstance(size, int)
            and 0 < size <= 4 * 1024**2
            and re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            subjects[match.group(1)].append(
                {"source_path": source, "sha256": digest, "bytes": size, "subject": match[1]}
            )
    for candidates in subjects.values():
        candidates.sort(key=lambda item: hashlib.sha256(item["source_path"].encode()).hexdigest())
    selected = []
    total = 0
    for offset in range(max(map(len, subjects.values()), default=0)):
        for subject in sorted(subjects):
            if offset >= len(subjects[subject]):
                continue
            item = subjects[subject][offset]
            if total + item["bytes"] > 512 * 1024**2:
                continue
            selected.append(item)
            total += item["bytes"]
            if len(selected) == clip_count:
                return selected
    raise ValueError(f"Source does not provide {clip_count} eligible clips within the byte budget")


def _matches(path: Path, entry: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return bool(digest.hexdigest() == entry["sha256"])


def _download(entry: dict[str, Any], root: Path) -> dict[str, Any]:
    source = entry["source_path"]
    if _CLIP_PATH.fullmatch(source) is None:
        raise ValueError("Unexpected AMASS source path")
    destination = (root / source).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError("AMASS output escapes the destination directory")
    url = f"{_HOST}/datasets/{REPOSITORY}/resolve/{REVISION}/{source}"
    if destination.exists() and not _matches(destination, entry):
        raise ValueError(f"Existing file fails source hash/size verification: {destination}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".part", delete=False
            ) as stream:
                temporary = Path(stream.name)
                with urlopen(url, timeout=60) as response:
                    received = 0
                    while block := response.read(1024 * 1024):
                        received += len(block)
                        if received > entry["bytes"]:
                            raise ValueError(f"Source exceeds expected size: {source}")
                        stream.write(block)
            if not _matches(temporary, entry):
                raise ValueError(f"Downloaded file fails source hash/size verification: {source}")
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    clip = load_amass_clip(destination)
    return {
        **entry,
        "local_path": source,
        "source_url": url,
        "frames": clip.num_frames,
        "fps": clip.fps,
        "model_type": clip.model_type,
    }


def prepare_amass_subset(output_dir: str | Path, *, workers: int = 4) -> dict[str, Any]:
    """Download at most 384 clips / 512 MiB; resume verified files on rerun.

    Creates source provenance and an illustrative 64-frame/stride-16 split.
    This function does not construct a model, optimizer, simulator or training run.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    root = Path(output_dir).expanduser().resolve()
    selected = _select(_source_tree(), 384)
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(lambda entry: _download(entry, root), selected))
    paths = [root / item["local_path"] for item in files]
    groups = {root / item["local_path"]: f"CMU/{item['subject']}" for item in files}
    train, validation = split_amass_files(paths, groups=groups, seed=20261015)
    frames = {root / item["local_path"]: item["frames"] for item in files}

    def eligible(subset: tuple[Path, ...]) -> list[str]:
        return [str(path.relative_to(root)) for path in subset if frames[path] >= 64]

    summary = {
        "clips": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "frames": sum(item["frames"] for item in files),
        "subjects": len(set(groups.values())),
    }
    manifest = {
        "schema": "amass-remote-subset/v1",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license_url": "https://amass.is.tue.mpg.de/license.html",
        "selection": "subject round-robin; SHA256(path) per subject; <=4 MiB/clip, <=512 MiB total",
        "files": files,
        "summary": summary,
    }
    splits = {
        "source_manifest": "manifest.json",
        "source_revision": REVISION,
        "seed": 20261015,
        "validation_group_fraction": 0.1,
        "grouping": "explicit CMU subject from source path",
        "window_length": 64,
        "stride": 16,
        "window_note": "Data preview only; S4 training hyperparameters remain undecided.",
        "train": eligible(train),
        "validation": eligible(validation),
        "excluded_short": [str(path.relative_to(root)) for path in paths if frames[path] < 64],
    }
    for name, value in (("manifest.json", manifest), ("splits.json", splits)):
        temporary_json = root / f".{name}.part"
        temporary_json.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary_json.replace(root / name)
    return summary
