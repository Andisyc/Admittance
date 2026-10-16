"""Single-teacher slice of ReGait's behavior-distillation learner.

Source: Andisyc/ReGait@06c12ee605795daa14817c9db7644ba75a453075,
src/unilab/algos/torch/distill/trainer.py. No task or simulator imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


def validate_matrix(value: torch.Tensor, name: str, *, width: int | None = None) -> None:
    """Reject malformed supervision before changing learner state."""
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] <= 0:
        raise ValueError(f"{name} must be a nonempty rank-2 tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating-point values")
    if value.shape[1] <= 0 or (width is not None and value.shape[1] != width):
        raise ValueError(f"{name} width mismatch: expected {width}, got {value.shape[1]}")


@dataclass(frozen=True)
class DistillationBatch:
    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    teacher_actions: torch.Tensor | None = None


@dataclass(frozen=True)
class BehaviorDistillationStats:
    loss: float
    student_grad_norm: float
    update_count: int
    student_action_shape: tuple[int, ...]
    teacher_action_shape: tuple[int, ...]
    teacher_action_requires_grad: bool
    teacher_action_source: str


class BehaviorDistillationTrainer:
    """Match frozen teacher actions on student-visited observations."""

    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        loss_type: Literal["mse", "huber", "l1"] = "mse",
        max_grad_norm: float | None = None,
    ) -> None:
        if loss_type not in {"mse", "huber", "l1"}:
            raise ValueError(f"Unsupported behavior distillation loss: {loss_type!r}")
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive")
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.max_grad_norm = max_grad_norm
        self.update_count = 0
        if teacher is not None:
            teacher.eval()
            teacher.requires_grad_(False)

    def _loss(self, student_action: torch.Tensor, teacher_action: torch.Tensor) -> torch.Tensor:
        validate_matrix(student_action, "student_action")
        validate_matrix(teacher_action, "teacher_action")
        if student_action.shape != teacher_action.shape:
            raise ValueError("student/teacher action shape mismatch")
        if self.loss_type == "mse":
            return F.mse_loss(student_action, teacher_action)
        if self.loss_type == "huber":
            return F.smooth_l1_loss(student_action, teacher_action)
        return F.l1_loss(student_action, teacher_action)

    def update(self, batch: DistillationBatch) -> BehaviorDistillationStats:
        validate_matrix(batch.student_obs, "student_obs")
        validate_matrix(batch.teacher_obs, "teacher_obs")
        if batch.student_obs.shape[0] != batch.teacher_obs.shape[0]:
            raise ValueError("student/teacher batch size mismatch")
        if batch.teacher_actions is None:
            if self.teacher is None:
                raise ValueError("cached teacher_actions are required when teacher=None")
            self.teacher.eval()
            with torch.no_grad():
                target = self.teacher(batch.teacher_obs)
                if isinstance(target, tuple):
                    target = target[0]
            source = "teacher"
        else:
            target = batch.teacher_actions
            source = "cached"
        validate_matrix(target, "teacher_actions")
        if target.shape[0] != batch.student_obs.shape[0]:
            raise ValueError("student/teacher action batch size mismatch")
        target = target.detach()
        self.student.train()
        action = self.student(batch.student_obs)
        loss = self._loss(action, target)
        if not bool(torch.isfinite(loss)):
            raise ValueError("distillation loss is non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for p in self.student.parameters() if p.grad is not None]
        if not parameters:
            raise ValueError("student produced no parameter gradients")
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            raise ValueError("student gradients are non-finite")
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        grad_norm = math.sqrt(
            sum(float(gradient.detach().square().sum()) for gradient in gradients)
        )
        self.optimizer.step()
        self.update_count += 1
        return BehaviorDistillationStats(
            loss=float(loss.detach()),
            student_grad_norm=grad_norm,
            update_count=self.update_count,
            student_action_shape=tuple(action.shape),
            teacher_action_shape=tuple(target.shape),
            teacher_action_requires_grad=target.requires_grad,
            teacher_action_source=source,
        )
