from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_SH = ROOT / "start.sh"


def _run_start(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(START_SH), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_start_sh_dry_run_uses_official_eval_route() -> None:
    result = _run_start(
        "--dry-run",
        "--algo",
        "appo",
        "--task",
        "g1_amp_walk",
        "--sim",
        "mujoco",
        "--load-run",
        "amp-run-1",
        "training.play_steps=12",
    )

    assert result.returncode == 0, result.stderr
    assert "[start.sh] command: uv run --no-sync eval" in result.stdout
    assert "--algo appo" in result.stdout
    assert "--task g1_amp_walk" in result.stdout
    assert "--sim mujoco" in result.stdout
    assert "--render-mode interactive" in result.stdout
    assert "--load-run amp-run-1" in result.stdout
    assert "training.play_steps=12" in result.stdout


def test_start_sh_dry_run_translates_checkpoint_path() -> None:
    result = _run_start(
        "--dry-run",
        "--algo",
        "appo",
        "--task",
        "g1_amp_walk",
        "--checkpoint-path",
        "/tmp/model_50.pt",
    )

    assert result.returncode == 0, result.stderr
    assert "algo.load_run=/tmp/model_50.pt" in result.stdout


def test_start_sh_distill_route_maps_student_checkpoint_and_default_task() -> None:
    result = _run_start(
        "--dry-run",
        "--algo",
        "distill",
        "--checkpoint-path",
        "/tmp/student_0008.pt",
    )

    assert result.returncode == 0, result.stderr
    assert "[start.sh] command: uv run --no-sync eval" in result.stdout
    assert "--task g1_walk_height" in result.stdout
    assert "student.initial_checkpoint_path=/tmp/student_0008.pt" in result.stdout


def test_start_sh_distill_route_requires_direct_checkpoint() -> None:
    result = _run_start("--dry-run", "--algo", "distill")

    assert result.returncode == 2
    assert "--algo distill requires --checkpoint-path" in result.stderr
    assert "[start.sh] command:" not in result.stdout
