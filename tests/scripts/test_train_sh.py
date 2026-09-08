from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SH = ROOT / "train.sh"


def _run_train(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(TRAIN_SH), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_train_sh_dry_run_uses_official_distill_route() -> None:
    result = _run_train(
        "--dry-run",
        "teacher.checkpoint_path=/tmp/model_5000.pt",
    )

    assert result.returncode == 0, result.stderr
    assert "[train.sh] command: uv run --no-sync train" in result.stdout
    assert "--algo distill" in result.stdout
    assert "--task g1_walk_height" in result.stdout
    assert "--sim mujoco" in result.stdout
    assert "teacher.checkpoint_path=/tmp/model_5000.pt" in result.stdout


def test_train_sh_help_describes_current_owner_entrypoint() -> None:
    result = _run_train("--help")

    assert result.returncode == 0, result.stderr
    assert (
        "uv run --no-sync train --algo distill --task g1_walk_height --sim mujoco" in result.stdout
    )
    assert "does not implement AMP-teacher" in result.stdout
