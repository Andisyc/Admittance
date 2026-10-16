"""Thin UniLab entrypoint for single-teacher G1 behavior distillation."""

from __future__ import annotations

import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.base.registry import ensure_registries
from unilab.rl.algos.distill.checkpoint import (
    load_distillation_checkpoint,
    save_distillation_checkpoint,
)
from unilab.rl.algos.distill.collector import DistillationCollectorSpec
from unilab.rl.algos.distill.persistent_runtime import PersistentDistillationRuntime
from unilab.rl.algos.distill.persistent_service import SingleTeacherDaggerService
from unilab.rl.algos.distill.teacher import SACTeacherSpec
from unilab.rl.algos.distill.trainer import BehaviorDistillationTrainer
from unilab.rl.algos.distill.workflow import (
    SingleScenarioDaggerConfig,
    SingleScenarioDaggerRunResult,
    run_single_scenario_dagger,
)
from unilab.scripts.g1_distill import (
    build_g1_distill_student,
    load_g1_distill_student,
    load_identified_sac_teacher,
    resolve_g1_distill_composition,
    resolve_teacher_checkpoint_identity,
)
from unilab.training import get_log_root
from unilab.training.experiment import write_run_config_snapshot
from unilab.utils.seed import apply_configured_training_seed

ROOT_DIR = Path(__file__).resolve().parents[3]


def _select_device(configured: Any) -> str:
    if configured not in (None, ""):
        return str(configured)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _loss_type(value: Any) -> Literal["mse", "huber", "l1"]:
    selected = str(value)
    if selected not in {"mse", "huber", "l1"}:
        raise ValueError(f"unsupported distillation loss_type: {selected!r}")
    return cast(Literal["mse", "huber", "l1"], selected)


def _run_dir(cfg: DictConfig, *, root_dir: Path) -> Path:
    configured = OmegaConf.select(cfg, "training.log_dir")
    if configured not in (None, ""):
        path = Path(str(configured)).expanduser()
        return path if path.is_absolute() else root_dir / path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return get_log_root(root_dir, cfg) / str(cfg.training.task_name) / f"{timestamp}_mujoco"


def _initial_student_checkpoint(
    cfg: DictConfig,
    *,
    student: torch.nn.Module,
    run_dir: Path,
    device: str,
) -> Path:
    configured = OmegaConf.select(cfg, "student.initial_checkpoint_path")
    if configured not in (None, ""):
        path = Path(str(configured)).expanduser().resolve()
        load_distillation_checkpoint(student, path, device=device)
        return path
    path = run_dir / "student_0000.pt"
    save_distillation_checkpoint(
        path,
        student=student,
        agent_steps=0,
        distill_runtime_cfg={"task": "G1WalkHeight", "scenario": "g1_walk_height"},
    )
    return path


def run_g1_distillation(
    cfg: DictConfig,
    *,
    root_dir: str | Path = ROOT_DIR,
) -> SingleScenarioDaggerRunResult:
    """Assemble UniLab owners and execute the package-neutral DAgger workflow."""
    if bool(OmegaConf.select(cfg, "training.play_only", default=False)):
        raise ValueError("distill playback is not part of the single-task training route")

    root = Path(root_dir).resolve()
    seed_info = apply_configured_training_seed(cfg)
    learner_device = _select_device(OmegaConf.select(cfg, "training.device"))
    collector_device = _select_device(
        OmegaConf.select(cfg, "training.collector_device", default=learner_device) or learner_device
    )
    composition = resolve_g1_distill_composition(cfg, root_dir=root)
    teacher_identity = resolve_teacher_checkpoint_identity(cfg, root_dir=root)
    run_dir = _run_dir(cfg, root_dir=root)
    run_dir.mkdir(parents=True, exist_ok=True)

    student_spec = dict(composition.student_spec)
    teacher_cfg = composition.teacher_spec
    teacher_spec = SACTeacherSpec(
        checkpoint_path=teacher_identity.path,
        obs_dim=int(teacher_cfg["obs_dim"]),
        action_dim=int(teacher_cfg["action_dim"]),
        actor_hidden_dim=int(teacher_cfg["actor_hidden_dim"]),
        use_layer_norm=bool(teacher_cfg["use_layer_norm"]),
        obs_normalization=bool(teacher_cfg["obs_normalization"]),
    )
    student = build_g1_distill_student(student_spec, device=learner_device)
    trainer = BehaviorDistillationTrainer(
        student=student,
        teacher=None,
        optimizer=torch.optim.Adam(student.parameters(), lr=float(cfg.algo.learning_rate)),
        loss_type=_loss_type(cfg.algo.loss_type),
        max_grad_norm=float(cfg.algo.max_grad_norm),
    )
    initial_checkpoint = _initial_student_checkpoint(
        cfg,
        student=student,
        run_dir=run_dir,
        device=learner_device,
    )
    collector_spec = DistillationCollectorSpec(
        num_samples=int(cfg.algo.dagger_samples_per_iteration),
        student_obs_dim=int(student_spec["obs_dim"]),
        teacher_obs_dim=int(teacher_cfg["obs_dim"]),
        action_dim=int(student_spec["action_dim"]),
        student_obs_key="obs",
        teacher_obs_key="obs",
        workflow_scenario="g1_walk_height",
    )
    runtime = PersistentDistillationRuntime(
        student_loader=partial(
            load_g1_distill_student,
            student_spec=student_spec,
            device="cpu",
        ),
        worker_factory=SingleTeacherDaggerService,
        worker_kwargs={
            "env_factory": composition.env_factory,
            "student_factory": partial(
                build_g1_distill_student,
                student_spec,
                device=collector_device,
            ),
            "teacher_factory": partial(
                load_identified_sac_teacher,
                teacher_spec,
                expected=teacher_identity,
                device=collector_device,
            ),
            "collector_spec": collector_spec,
            "num_envs": composition.num_envs,
            "env_cfg_override": composition.env_cfg_overrides,
            "collection_metadata": {
                "teacher_checkpoint_path": teacher_identity.path,
                "teacher_checkpoint_sha256": teacher_identity.sha256,
                "teacher_checkpoint_size_bytes": teacher_identity.size_bytes,
            },
        },
        request_timeout_seconds=float(cfg.algo.request_timeout_seconds),
    )
    runtime_cfg = {
        "scenario": "g1_walk_height",
        "learner_device": learner_device,
        "collector_device": collector_device,
        "num_envs": composition.num_envs,
        "seed": seed_info.effective_seed,
    }
    teacher_metadata = {
        "checkpoint_path": teacher_identity.path,
        "checkpoint_sha256": teacher_identity.sha256,
        "checkpoint_size_bytes": teacher_identity.size_bytes,
        "spec": teacher_cfg,
    }
    write_run_config_snapshot(
        run_dir,
        run_metadata={
            "algo": "distill",
            "task": composition.task_name,
            "sim_backend": composition.sim_backend,
            "teacher": teacher_metadata,
            **runtime_cfg,
        },
        full_cfg=cfg,
        filename="distill_run_config.json",
        trailing_newline=True,
    )
    try:
        return run_single_scenario_dagger(
            runtime,
            trainer,
            initial_checkpoint=initial_checkpoint,
            output_dir=run_dir / "artifacts",
            config=SingleScenarioDaggerConfig(
                scenario="g1_walk_height",
                iterations=int(cfg.algo.dagger_iterations),
                num_samples=int(cfg.algo.dagger_samples_per_iteration),
                batch_size=int(cfg.algo.dagger_batch_size),
                max_updates=int(cfg.algo.dagger_updates_per_iteration),
                student_obs_dim=int(student_spec["obs_dim"]),
                teacher_obs_dim=int(teacher_cfg["obs_dim"]),
                action_dim=int(student_spec["action_dim"]),
                seed=int(cfg.algo.seed),
            ),
            teacher_metadata=teacher_metadata,
            distill_runtime_cfg=runtime_cfg,
        )
    finally:
        runtime.close()


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    result = run_g1_distillation(cfg)
    print(f"Distillation complete: {result.final_checkpoint_path}")


if __name__ == "__main__":
    main()
