from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from uni_rl.algos.distill.teacher import SACTeacherSpec

from unilab import cli
from unilab.scripts.g1_distill import (
    checkpoint_identity,
    load_identified_sac_teacher,
    resolve_g1_distill_composition,
    resolve_teacher_checkpoint_identity,
)
from unilab.scripts.train_distill import _loss_type

ROOT = Path(__file__).resolve().parents[2]


def _compose_distill():
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT / "src/unilab/conf/distill"),
        version_base="1.3",
    ):
        return compose("config")


def test_g1_walk_height_distill_composes_picklable_sac_environment_binding() -> None:
    cfg = _compose_distill()
    binding = resolve_g1_distill_composition(cfg, root_dir=ROOT)

    assert binding.task_name == "G1WalkHeight"
    assert binding.sim_backend == "mujoco"
    assert binding.teacher_spec["obs_dim"] == 99
    assert binding.teacher_spec["action_dim"] == 29
    assert binding.student_spec["obs_dim"] == 99
    assert binding.student_spec["action_dim"] == 29
    assert "height" in binding.env_cfg_overrides["commands"]
    assert "height_command" in binding.env_cfg_overrides["observations"]["policy"]["terms"]
    assert "track_base_height_exp_smooth" in binding.env_cfg_overrides["rewards"]

    restored = pickle.loads(pickle.dumps(binding))
    assert restored.env_factory.func.__name__ == "make_registry_env"
    assert restored.env_factory.args == ("G1WalkHeight", "mujoco")
    assert restored.env_cfg_overrides == binding.env_cfg_overrides


def test_teacher_checkpoint_identity_is_fail_closed_and_content_addressed(tmp_path: Path) -> None:
    cfg = _compose_distill()
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"frozen-sac-teacher")
    cfg.teacher.checkpoint_path = checkpoint.name

    identity = resolve_teacher_checkpoint_identity(cfg, root_dir=tmp_path)
    assert identity == checkpoint_identity(checkpoint)
    assert identity.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert identity.size_bytes == len(b"frozen-sac-teacher")

    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="teacher checkpoint does not exist"):
        resolve_teacher_checkpoint_identity(cfg, root_dir=tmp_path)


def test_g1_walk_height_distill_rejects_dimension_drift() -> None:
    cfg = _compose_distill()
    cfg.teacher.obs_dim = 98
    with pytest.raises(ValueError, match="owner contract mismatch"):
        resolve_g1_distill_composition(cfg, root_dir=ROOT)


def test_distill_is_discovered_by_the_official_training_cli() -> None:
    assert "distill" in cli.available_algos()
    command = cli.build_command(
        mode="train",
        algo="distill",
        task="g1_walk_height",
        sim="mujoco",
        overrides=["teacher.checkpoint_path=/tmp/model_5000.pt"],
    )
    assert command[1].endswith("/src/unilab/scripts/train_distill.py")
    assert "task=g1_walk_height/mujoco" in command


def test_distill_official_eval_cli_uses_interactive_playback_owner() -> None:
    command = cli.build_command(
        mode="eval",
        algo="distill",
        task="g1_walk_height",
        sim="mujoco",
        render_mode="interactive",
        overrides=["student.initial_checkpoint_path=/tmp/student_0008.pt"],
    )

    assert command[1].endswith("/src/unilab/scripts/play_interactive.py")
    assert command[command.index("--algo") + 1] == "distill"
    assert "student.initial_checkpoint_path=/tmp/student_0008.pt" in command
    assert "interactive.action_mode=policy" in command


def test_teacher_identity_is_rechecked_before_worker_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"first")
    expected = checkpoint_identity(checkpoint)
    checkpoint.write_bytes(b"replaced")
    monkeypatch.setattr(
        "unilab.scripts.g1_distill.load_sac_teacher_policy",
        lambda *_args, **_kwargs: object(),
    )
    with pytest.raises(ValueError, match="identity changed"):
        load_identified_sac_teacher(
            SACTeacherSpec(checkpoint),
            expected=expected,
            device="cpu",
        )


def test_distill_loss_type_fails_closed() -> None:
    assert _loss_type("mse") == "mse"
    with pytest.raises(ValueError, match="unsupported distillation loss_type"):
        _loss_type("cosine")
