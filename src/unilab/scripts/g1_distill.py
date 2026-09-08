"""UniLab-owned composition for G1 behavior distillation.

This module resolves task/configuration ownership and produces package-neutral,
picklable environment inputs. Training, collection, IPC, and process lifecycle
remain owned by ``uni_rl``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch import nn
from uni_rl.algos.distill.checkpoint import load_distillation_checkpoint
from uni_rl.algos.distill.models import MLPStudentPolicy
from uni_rl.algos.distill.teacher import SACTeacherSpec, load_sac_teacher_policy
from uni_rl.env_contract import EnvFactory

from unilab.base.config_adapter import BackendAdapter
from unilab.base.env_factory import registry_env_factory
from unilab.utils.checkpoint import resolve_task_checkpoint_path

_PACKAGE_CONF_ROOT = Path(__file__).resolve().parents[1] / "conf"
_TASK_NAME = "G1WalkHeight"
_SIM_BACKEND = "mujoco"
_TEACHER_TASK = "sac/g1_walk_height/mujoco"
_TEACHER_OBS_DIM = 99
_ACTION_DIM = 29


@dataclass(frozen=True)
class CheckpointIdentity:
    """Stable filesystem identity for one frozen teacher checkpoint."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class G1DistillComposition:
    """UniLab inputs consumed by a future package-neutral distill runtime."""

    task_name: str
    sim_backend: str
    num_envs: int
    env_factory: EnvFactory
    env_cfg_overrides: dict[str, Any]
    teacher_spec: dict[str, Any]
    student_spec: dict[str, Any]


def build_g1_distill_student(
    student_spec: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build the only student architecture admitted by the phase-2B route."""
    if student_spec.get("model_type") != "mlp":
        raise ValueError("G1WalkHeight distillation currently supports only model_type='mlp'")
    model = MLPStudentPolicy(
        obs_dim=int(student_spec["obs_dim"]),
        action_dim=int(student_spec["action_dim"]),
        hidden_dims=tuple(int(value) for value in student_spec["hidden_dims"]),
        activation=str(student_spec["activation"]),
        squash_action=bool(student_spec["squash_action"]),
    )
    return model.to(device)


def load_g1_distill_student(
    checkpoint_path: str | Path,
    *,
    student_spec: dict[str, Any],
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Rebuild and load a versioned G1 student checkpoint."""
    model = build_g1_distill_student(student_spec, device=device)
    load_distillation_checkpoint(model, checkpoint_path, device=device)
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: str | Path) -> CheckpointIdentity:
    """Validate and identify a checkpoint without loading or mutating it."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"teacher checkpoint does not exist: {resolved}")
    return CheckpointIdentity(
        path=str(resolved),
        sha256=_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def load_identified_sac_teacher(
    spec: SACTeacherSpec,
    *,
    expected: CheckpointIdentity,
    device: str,
) -> nn.Module:
    """Reject a teacher file replaced after parent-side composition."""
    observed = checkpoint_identity(spec.checkpoint_path)
    if observed != expected:
        raise ValueError(
            "SAC teacher checkpoint identity changed before worker load: "
            f"expected={expected}, observed={observed}"
        )
    return load_sac_teacher_policy(spec, device=device)


def _compose_sac_teacher_owner() -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str((_PACKAGE_CONF_ROOT / "sac").resolve()),
        version_base="1.3",
    ):
        return compose("config", overrides=["task=g1_walk_height/mujoco"])


def _plain_mapping(cfg: DictConfig, path: str) -> dict[str, Any]:
    value = OmegaConf.select(cfg, path)
    payload = OmegaConf.to_container(value, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must resolve to a mapping")
    return {str(key): item for key, item in payload.items()}


def _require_g1_walk_height_contract(cfg: DictConfig) -> None:
    expected = {
        "training.task_name": _TASK_NAME,
        "training.sim_backend": _SIM_BACKEND,
        "teacher.algo_family": "sac",
        "teacher.algo_type": "sac",
        "teacher.algo_log_name": "fast_sac",
        "teacher.task": _TEACHER_TASK,
        "teacher.task_name": _TASK_NAME,
        "teacher.obs_dim": _TEACHER_OBS_DIM,
        "teacher.action_dim": _ACTION_DIM,
        "student.obs_dim": _TEACHER_OBS_DIM,
        "student.action_dim": _ACTION_DIM,
    }
    mismatches = {
        path: {"expected": value, "observed": OmegaConf.select(cfg, path)}
        for path, value in expected.items()
        if OmegaConf.select(cfg, path) != value
    }
    if mismatches:
        raise ValueError(f"G1WalkHeight distill owner contract mismatch: {mismatches}")


def resolve_g1_distill_composition(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
) -> G1DistillComposition:
    """Resolve the single-task G1 environment binding from its SAC owner."""
    _require_g1_walk_height_contract(cfg)
    teacher_owner = _compose_sac_teacher_owner()
    if (
        str(teacher_owner.training.task_name) != _TASK_NAME
        or str(teacher_owner.training.sim_backend) != _SIM_BACKEND
    ):
        raise ValueError("SAC G1WalkHeight owner resolved to an unexpected task or backend")

    project_root = Path.cwd() if root_dir is None else Path(root_dir)
    env_cfg_overrides = BackendAdapter(
        teacher_owner,
        root_dir=project_root,
        algo_name="sac",
    ).build_task_env_cfg_override()
    return G1DistillComposition(
        task_name=_TASK_NAME,
        sim_backend=_SIM_BACKEND,
        num_envs=int(cfg.algo.num_envs),
        env_factory=registry_env_factory(_TASK_NAME, _SIM_BACKEND),
        env_cfg_overrides=env_cfg_overrides,
        teacher_spec=_plain_mapping(cfg, "teacher"),
        student_spec=_plain_mapping(cfg, "student"),
    )


def resolve_teacher_checkpoint_identity(
    cfg: DictConfig,
    *,
    root_dir: str | Path | None = None,
) -> CheckpointIdentity:
    """Resolve the configured SAC teacher checkpoint and return its identity."""
    _require_g1_walk_height_contract(cfg)
    root = Path.cwd() if root_dir is None else Path(root_dir)
    explicit = OmegaConf.select(cfg, "teacher.checkpoint_path")
    if explicit not in (None, ""):
        path = Path(str(explicit)).expanduser()
        return checkpoint_identity(path if path.is_absolute() else root / path)

    selector = OmegaConf.select(cfg, "teacher.checkpoint")
    checkpoint = None if selector in (None, "", -1, "-1") else str(selector)
    log_root = OmegaConf.select(
        cfg,
        "teacher.log_root",
        default=OmegaConf.select(cfg, "training.log_root"),
    )
    resolved_checkpoint, _run_dir = resolve_task_checkpoint_path(
        root,
        task_name=_TASK_NAME,
        load_run=str(OmegaConf.select(cfg, "teacher.load_run")),
        algo_log_name=str(OmegaConf.select(cfg, "teacher.algo_log_name")),
        checkpoint=checkpoint,
        suffix=".pt",
        log_root=log_root,
    )
    if resolved_checkpoint is None:
        raise FileNotFoundError("could not resolve a G1WalkHeight SAC teacher checkpoint")
    return checkpoint_identity(resolved_checkpoint)


__all__ = [
    "CheckpointIdentity",
    "G1DistillComposition",
    "build_g1_distill_student",
    "checkpoint_identity",
    "load_identified_sac_teacher",
    "load_g1_distill_student",
    "resolve_g1_distill_composition",
    "resolve_teacher_checkpoint_identity",
]
