"""Generic student-visited, teacher-labelled DAgger collection.

Adapted from ReGait's collect_distillation_dataset_from_env: both policies see
the same pre-step state, teacher targets are detached, and the student alone
chooses environment actions. Task projection is deliberately not inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from unilab.rl.env_contract import EnvProtocol

from .data import DistillationTensorDataset
from .trainer import validate_matrix


@dataclass(frozen=True)
class DistillationCollectorSpec:
    num_samples: int
    student_obs_dim: int
    teacher_obs_dim: int
    action_dim: int
    student_obs_key: str = "obs"
    teacher_obs_key: str = "obs"
    workflow_scenario: str = "default"

    def __post_init__(self) -> None:
        dimensions = (self.num_samples, self.student_obs_dim, self.teacher_obs_dim, self.action_dim)
        if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in dimensions):
            raise ValueError("collector sample count and dimensions must be positive integers")
        if not self.student_obs_key or not self.teacher_obs_key or not self.workflow_scenario:
            raise ValueError("collector keys and workflow_scenario must be nonempty")


def _observations(
    obs: dict[str, np.ndarray], key: str, *, rows: int, width: int, device: torch.device
) -> torch.Tensor:
    if not isinstance(obs, dict) or key not in obs:
        raise ValueError(f"collector observation group is missing: {key}")
    array = np.asarray(obs[key])
    if array.shape != (rows, width):
        raise ValueError(f"collector observation shape mismatch: {key}={array.shape}")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f"collector observation must be finite floating-point: {key}")
    return torch.as_tensor(array, device=device, dtype=torch.float32)


@torch.no_grad()
def collect_distillation_dataset_from_env(
    env: EnvProtocol,
    student: nn.Module,
    teacher: nn.Module,
    spec: DistillationCollectorSpec,
    *,
    metadata: dict[str, Any] | None = None,
) -> DistillationTensorDataset:
    """Collect exactly num_samples rows, respecting vector-env reset semantics."""
    rows = int(env.num_envs)
    if rows <= 0:
        raise ValueError("collector requires a positive environment count")
    student.eval()
    teacher.eval()
    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device
    state = env.state if env.state is not None else env.init_state()
    obs = state.obs
    observations: list[torch.Tensor] = []
    teacher_observations: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    remaining = spec.num_samples
    while remaining:
        student_obs = _observations(
            obs, spec.student_obs_key, rows=rows, width=spec.student_obs_dim, device=student_device
        )
        teacher_obs = _observations(
            obs, spec.teacher_obs_key, rows=rows, width=spec.teacher_obs_dim, device=teacher_device
        )
        action = student(student_obs)
        target = teacher(teacher_obs)
        validate_matrix(action, "collector student action", width=spec.action_dim)
        validate_matrix(target, "collector teacher action", width=spec.action_dim)
        if action.shape[0] != rows or target.shape[0] != rows:
            raise ValueError("collector policy action row count mismatch")
        take = min(rows, remaining)
        # Copy before step: environments may reuse observation storage in place.
        observations.append(student_obs[:take].detach().cpu().clone())
        teacher_observations.append(teacher_obs[:take].detach().cpu().clone())
        labels.append(target[:take].detach().cpu().clone())
        state = env.step(action.detach().cpu().numpy())
        obs = state.obs
        terminated = np.asarray(state.terminated)
        truncated = np.asarray(state.truncated)
        if terminated.shape != (rows,) or truncated.shape != (rows,):
            raise ValueError("collector termination row count mismatch")
        if terminated.dtype != np.bool_ or truncated.dtype != np.bool_:
            raise ValueError("collector termination flags must be boolean")
        done = np.logical_or(terminated, truncated)
        if done.any() and state.final_observation is None:
            # Envs without autoreset supply no final_observation; explicitly reset
            # just their done rows, preserving all other environment histories.
            indices = np.flatnonzero(done)
            reset_obs, _ = env.reset(indices)
            obs = {key: np.asarray(value).copy() for key, value in obs.items()}
            for key, value in reset_obs.items():
                if key not in obs:
                    raise ValueError(f"reset introduced an unexpected observation group: {key}")
                array = np.asarray(value)
                if array.shape == obs[key].shape:
                    obs[key][indices] = array[indices]
                elif array.shape == (len(indices), *obs[key].shape[1:]):
                    obs[key][indices] = array
                else:
                    raise ValueError(f"reset observation shape mismatch: {key}={array.shape}")
        remaining -= take
    return DistillationTensorDataset(
        student_obs=torch.cat(observations),
        teacher_obs=torch.cat(teacher_observations),
        teacher_actions=torch.cat(labels),
        metadata={**dict(metadata or {}), "workflow_scenario": spec.workflow_scenario},
    )
