"""Prepare the pinned X-WBC G1 AMASS archive with target-model FK, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import yaml

from unilab.data.amass import split_amass_files
from unilab.data.retarget import file_sha256, kinematics_sha256
from unilab.data.robot_motion import prepare_retargeted_motion

REPOSITORY = "TonyZhang1999/xwbc-retarget-data"
REVISION = "b4e169f460bd68b27e04e75c99be6e0affbfe046"
ARCHIVE_SHA256 = "2ed1c6270a6c2d060300cdb8c607a31d2b118f131d6555e6f9dede4321bd1738"


def _extract_cmu(archive: Path, destination: Path) -> tuple[list[Path], int]:
    """Extract data-only members atomically; preserve and verify existing inputs."""
    paths = []
    total_clips = 0
    with tarfile.open(archive, mode="r|gz") as stream:
        for member in stream:
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or len(name.parts) != 2
                or name.parts[0] != "unitree_g1_29dof"
                or name.suffix != ".npz"
                or member.size > 256 * 1024**2
            ):
                raise ValueError(f"unexpected archive member: {member.name}")
            total_clips += 1
            if not re.fullmatch(r"CMU__\d+__[^/]+\.npz", name.name):
                continue
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable archive member: {member.name}")
            payload = source.read()
            target = destination / name.name
            digest = hashlib.sha256(payload).hexdigest()
            if target.exists():
                if file_sha256(target) != digest:
                    raise ValueError(f"existing source differs from pinned archive: {target}")
            else:
                destination.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with NamedTemporaryFile(dir=destination, suffix=".part", delete=False) as f:
                        temporary = Path(f.name)
                        f.write(payload)
                    os.link(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            paths.append(target)
    return paths, total_clips


def _convert(job):
    source, output, model, names, implementation_hash = job
    metadata = {
        "source_dataset": "AMASS retargeted by X-WBC; CMU",
        "repository": REPOSITORY,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "source_member": f"unitree_g1_29dof/{source.name}",
        "preprocessor_sha256": implementation_hash,
    }
    if output.exists():
        with np.load(output, allow_pickle=False) as arrays:
            saved = json.loads(arrays["metadata_json"].item())
            if any(saved.get(k) != v for k, v in metadata.items()):
                raise ValueError(f"existing output has different provenance: {output}")
            if saved.get("source_sha256") != file_sha256(source):
                raise ValueError(f"existing output has different source: {output}")
            if arrays["joint_names"].tolist() != list(names):
                raise ValueError(f"existing output has different joint order: {output}")
    else:
        try:
            prepare_retargeted_motion(
                source,
                output,
                model_file=model,
                joint_names=names,
                root_body_index=0,
                target_fps=50,
                metadata=metadata,
            )
        except ValueError as exc:
            return {"source": source.name, "status": "rejected", "reason": str(exc)}
    with np.load(output, allow_pickle=False) as arrays:
        frames = len(arrays["joint_pos"])
    if frames < 64:
        return {"source": source.name, "status": "excluded_short", "frames": frames}
    return {
        "source": source.name,
        "status": "accepted",
        "path": str(output),
        "frames": frames,
        "group": f"CMU/{source.name.split('__')[1]}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--task-config", type=Path, default=Path("src/unilab/conf/appo/task/anybody_g1/mujoco.yaml")
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if file_sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("archive does not match the pinned source SHA256")
    print("Archive SHA256 verified; extracting all CMU clips", flush=True)
    paths, archive_clips = _extract_cmu(args.archive, args.source_dir.resolve())
    if not paths:
        raise ValueError("pinned archive contains no CMU clips")
    cfg = yaml.safe_load(args.task_config.read_text())
    model = Path(cfg["env"]["scene"]["model_file"]).resolve()
    names = cfg["env"]["scene"]["entities"]["robot"]["joint_names"]
    implementation = hashlib.sha256()
    for owner in (
        Path(__file__),
        args.task_config,
        Path("src/unilab/data/robot_motion.py"),
        Path("src/unilab/data/tracking_export.py"),
        Path("src/unilab/data/human_motion.py"),
        Path("src/unilab/data/retarget.py"),
        model,
    ):
        implementation.update(owner.read_bytes())
    import mujoco as _mj

    mj: Any = _mj
    implementation.update(kinematics_sha256(mj.MjModel.from_xml_path(str(model))).encode())
    output_dir = args.output_dir.resolve()
    jobs = [(p, output_dir / p.name, model, names, implementation.hexdigest()) for p in paths]
    print(
        json.dumps(
            {"archive_clips": archive_clips, "cmu_clips": len(paths), "workers": args.workers}
        ),
        flush=True,
    )
    records = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for result in pool.map(_convert, jobs, chunksize=1):
            records.append(result)
            if len(records) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "processed": len(records),
                            "accepted": sum(r["status"] == "accepted" for r in records),
                        }
                    ),
                    flush=True,
                )
    accepted = [r for r in records if r["status"] == "accepted"]
    if len(accepted) < 2:
        raise ValueError("fewer than two eligible clips; refusing to create a training split")
    groups: dict[str | Path, str] = {Path(r["path"]): r["group"] for r in accepted}
    train, validation = split_amass_files(list(groups), groups=groups, seed=20261015)
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_clips": archive_clips,
        "prepared_subset": "all CMU clips passing explicit target-model checks",
        "preprocessor_sha256": implementation.hexdigest(),
        "records": records,
        "summary": {
            "source_clips": len(paths),
            "accepted_clips": len(accepted),
            "frames": sum(r["frames"] for r in accepted),
            "hours": sum(r["frames"] for r in accepted) / 50 / 3600,
            "groups": len(set(groups.values())),
            "train_clips": len(train),
            "validation_clips": len(validation),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("manifest.json", manifest),
        (
            "splits.json",
            {
                "seed": 20261015,
                "grouping": "CMU subject token in source filenames",
                "train": list(map(str, train)),
                "validation": list(map(str, validation)),
            },
        ),
    ):
        temporary = output_dir / f".{name}.part"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(output_dir / name)
    # Supply a file-backed Hydra list instead of exceeding the OS argument limit.
    # Network/optimizer/device choices remain in the existing task owner and CLI.
    run_config = output_dir / "run-config"
    run_config.mkdir(exist_ok=True)
    formal = {
        "defaults": ["config", "_self_"],
        "env": {"commands": {"motion": {"params": {"motion_file": list(map(str, train))}}}},
    }
    temporary = run_config / ".formal.yaml.part"
    temporary.write_text(yaml.safe_dump(formal, sort_keys=False))
    temporary.replace(run_config / "formal.yaml")
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
