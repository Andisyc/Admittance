"""Offline packed SMPL-H AMASS -> named body poses -> G1 tracking NPZ; no training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from unilab.data.amass import load_amass_clip
from unilab.data.human_motion import reconstruct_smplh
from unilab.data.retarget import file_sha256, retarget_to_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-type", required=True, choices=["smplh"])
    parser.add_argument(
        "--source-up-axis",
        required=True,
        choices=["z"],
        help="Explicit source world-axis declaration; other conventions need an adapter",
    )
    parser.add_argument("--body-model", required=True, help="Matching gender/shape SMPL-H NPZ")
    parser.add_argument("--dmpl-model", help="Matching DMPL NPZ containing eigvec")
    parser.add_argument(
        "--human-height", type=float, required=True, help="Explicit height in meters"
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--task-config", default="src/unilab/conf/appo/task/anybody_g1/mujoco.yaml")
    args = parser.parse_args()
    clip = load_amass_clip(args.input, model_type=args.model_type)
    skeleton = reconstruct_smplh(clip, args.body_model, dmpl_file=args.dmpl_model)
    config = yaml.safe_load(Path(args.task_config).read_text())
    scene = config["env"]["scene"]
    provenance = {
        "source_dataset": "AMASS",
        "source_file": str(clip.path),
        "source_sha256": file_sha256(clip.path),
        "declared_model_type": clip.model_type,
        "declared_source_up_axis": args.source_up_axis,
        "body_model_sha256": file_sha256(args.body_model),
    }
    if args.dmpl_model:
        provenance["dmpl_model_sha256"] = file_sha256(args.dmpl_model)
    output = retarget_to_tracking(
        skeleton,
        args.output,
        model_file=scene["model_file"],
        joint_names=scene["entities"]["robot"]["joint_names"],
        human_height=args.human_height,
        target_fps=args.fps,
        metadata=provenance,
    )
    print(json.dumps({"output": str(output), "training": False}, indent=2))


if __name__ == "__main__":
    main()
