"""Aligned cached teacher labels for the single-teacher DAgger slice.

Preserves ReGait's student_obs/teacher_obs/teacher_actions dataset contract;
task-specific role, projection and MoE metadata stay outside this runtime.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .trainer import DistillationBatch, validate_matrix


@dataclass(frozen=True)
class DistillationTensorDataset:
    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    teacher_actions: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("student_obs", "teacher_obs", "teacher_actions"):
            validate_matrix(getattr(self, name), name)
        rows = self.student_obs.shape[0]
        if self.teacher_obs.shape[0] != rows or self.teacher_actions.shape[0] != rows:
            raise ValueError("distillation dataset row alignment mismatch")
        tensors = (self.student_obs, self.teacher_obs, self.teacher_actions)
        if len({(t.device, t.dtype) for t in tensors}) != 1:
            raise ValueError("distillation dataset tensors must share dtype and device")

    @property
    def num_samples(self) -> int:
        return int(self.student_obs.shape[0])

    def batch(self, indices: torch.Tensor, *, device: str | torch.device) -> DistillationBatch:
        return DistillationBatch(
            student_obs=self.student_obs[indices].to(device),
            teacher_obs=self.teacher_obs[indices].to(device),
            teacher_actions=self.teacher_actions[indices].to(device),
        )


def save_distillation_dataset(path: str | Path, dataset: DistillationTensorDataset) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    payload = {
        "schema": "unilab.distillation_dataset",
        "schema_version": 1,
        "student_obs": dataset.student_obs.detach().cpu(),
        "teacher_obs": dataset.teacher_obs.detach().cpu(),
        "teacher_actions": dataset.teacher_actions.detach().cpu(),
        "metadata": dataset.metadata,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_distillation_dataset(path: str | Path) -> DistillationTensorDataset:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "unilab.distillation_dataset":
        raise ValueError("unsupported distillation dataset schema")
    if payload.get("schema_version") != 1 or not isinstance(payload.get("metadata"), dict):
        raise ValueError("unsupported distillation dataset version/metadata")
    required = ("student_obs", "teacher_obs", "teacher_actions")
    if not all(name in payload for name in required):
        raise ValueError("distillation dataset is missing aligned tensors")
    return DistillationTensorDataset(
        student_obs=payload["student_obs"],
        teacher_obs=payload["teacher_obs"],
        teacher_actions=payload["teacher_actions"],
        metadata=payload["metadata"],
    )
