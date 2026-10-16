from __future__ import annotations

import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn


def _cpu_checkpoint_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_checkpoint_value(item) for item in value)
    if isinstance(value, list):
        return [_cpu_checkpoint_value(item) for item in value]
    return value


def validate_module_state(module: nn.Module, state: Any, *, label: str) -> None:
    """Preflight complete tensor structure before a module can be mutated."""
    expected = module.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError(f"{label} checkpoint state key mismatch")
    for name, target in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape:
            raise ValueError(f"{label} checkpoint shape mismatch at {name}")
        if value.dtype != target.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} checkpoint dtype/non-finite value at {name}")


def _validate_optimizer_state(optimizer: torch.optim.Optimizer, state: Any) -> None:
    """Validate serialized parameter ownership and moments without loading them.

    Adam checkpoints must carry complete moments for every initialized
    parameter. Other optimizers retain their existing public state format;
    tensor buffers must match their parameter and every scalar must be finite.
    """
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state must contain state and param_groups")
    groups = state["param_groups"]
    reference = optimizer.state_dict()["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(reference):
        raise ValueError("checkpoint optimizer parameter groups mismatch")
    parameters: dict[int, torch.Tensor] = {}
    options: dict[int, Mapping[str, Any]] = {}
    for group, expected, live_group in zip(groups, reference, optimizer.param_groups, strict=True):
        if not isinstance(group, Mapping) or set(group) != set(expected):
            raise ValueError("checkpoint optimizer parameter group fields mismatch")
        identifiers = group["params"]
        if (
            not isinstance(identifiers, list)
            or any(isinstance(ident, bool) or not isinstance(ident, int) for ident in identifiers)
            or identifiers != expected["params"]
        ):
            raise ValueError("checkpoint optimizer parameter ids/order mismatch")
        for name, value in group.items():
            if name == "params":
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"checkpoint optimizer option {name} must be finite scalar")
            elif isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise ValueError(f"checkpoint optimizer option {name} is non-finite")
            elif isinstance(value, (tuple, list)):
                if not all(
                    isinstance(item, (int, float)) and math.isfinite(item) for item in value
                ):
                    raise ValueError(f"checkpoint optimizer option {name} is malformed")
            elif value is not None:
                raise ValueError(f"checkpoint optimizer option {name} has unsupported type")
            if name in {"lr", "eps", "weight_decay"}:
                if isinstance(value, bool) or not isinstance(value, (int, float, torch.Tensor)):
                    raise ValueError(f"checkpoint optimizer option {name} must be numeric")
                scalar = float(value)
                if scalar < 0 or (scalar == 0 and name != "weight_decay"):
                    raise ValueError(f"checkpoint optimizer option {name} is outside its range")
        if "betas" in group:
            betas = group["betas"]
            if (
                not isinstance(betas, (tuple, list))
                or len(betas) != 2
                or any(isinstance(beta, bool) or not 0 <= beta < 1 for beta in betas)
            ):
                raise ValueError("checkpoint optimizer Adam betas must lie in [0, 1)")
        for ident, parameter in zip(identifiers, live_group["params"], strict=True):
            if ident in parameters:
                raise ValueError("checkpoint optimizer parameter ids must be unique")
            parameters[ident] = parameter
            options[ident] = group
    moments = state["state"]
    if not isinstance(moments, Mapping) or any(
        isinstance(ident, bool) or not isinstance(ident, int) or ident not in parameters
        for ident in moments
    ):
        raise ValueError("checkpoint optimizer state contains invalid parameter ids")
    for ident, buffers in moments.items():
        if not isinstance(buffers, Mapping):
            raise ValueError("checkpoint optimizer parameter state must be a mapping")
        group = options[ident]
        if "betas" in group and "amsgrad" in group:
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                required.add("max_exp_avg_sq")
            if set(buffers) != required:
                raise ValueError("checkpoint optimizer Adam moment state is incomplete")
        for name, value in buffers.items():
            if name == "step":
                if isinstance(value, torch.Tensor):
                    if (
                        value.numel() != 1
                        or not bool(torch.isfinite(value).all())
                        or value.item() < 0
                    ):
                        raise ValueError("checkpoint optimizer step must be finite and nonnegative")
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("checkpoint optimizer step must be finite and nonnegative")
                continue
            if not isinstance(value, torch.Tensor) or value.shape != parameters[ident].shape:
                raise ValueError(f"checkpoint optimizer buffer shape mismatch: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"checkpoint optimizer buffer is non-finite: {name}")
            if name in {"exp_avg_sq", "max_exp_avg_sq"} and bool((value < 0).any()):
                raise ValueError(f"checkpoint optimizer second moment is negative: {name}")


def save_distillation_checkpoint(
    path: str | Path,
    *,
    student: nn.Module,
    agent_steps: int,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
) -> None:
    """Save the deployable student and distillation provenance."""

    resolved_path = Path(path)
    if isinstance(agent_steps, bool) or agent_steps < 0:
        raise ValueError("agent_steps must be non-negative")
    validate_module_state(student, student.state_dict(), label="student")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "unilab.distillation",
        "schema_version": 1,
        "student_state_dict": _cpu_checkpoint_value(student.state_dict()),
        "agent_steps": int(agent_steps),
        "teacher_metadata": dict(teacher_metadata or {}),
        "distill_runtime_cfg": dict(distill_runtime_cfg or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = _cpu_checkpoint_value(optimizer.state_dict())
    if obs_normalizer is not None:
        payload["obs_normalizer"] = _cpu_checkpoint_value(obs_normalizer.state_dict())
    tmp_path = resolved_path.with_name(
        f".{resolved_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        torch.save(_cpu_checkpoint_value(payload), tmp_path)
        tmp_path.replace(resolved_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"distillation checkpoint was not saved: {resolved_path}")


def load_distillation_checkpoint(
    student: nn.Module,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a student-only distillation checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    return restore_distillation_checkpoint(
        student, checkpoint, optimizer=optimizer, obs_normalizer=obs_normalizer
    )


def restore_distillation_checkpoint(
    student: nn.Module,
    checkpoint: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    obs_normalizer: nn.Module | None = None,
) -> dict[str, Any]:
    """Preflight a student snapshot before applying any module state."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("distillation checkpoint must be a mapping")
    if checkpoint.get("schema", "unilab.distillation") != "unilab.distillation":
        raise ValueError("unsupported distillation checkpoint schema")
    if checkpoint.get("schema_version", 1) != 1:
        raise ValueError("unsupported distillation checkpoint schema version")
    student_state = checkpoint.get("student_state_dict")
    if student_state is None:
        raise ValueError("Checkpoint does not contain student_state_dict")
    validate_module_state(student, student_state, label="student")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    normalizer_state = checkpoint.get("obs_normalizer")
    if obs_normalizer is not None:
        validate_module_state(obs_normalizer, normalizer_state, label="obs_normalizer")
    if optimizer is not None:
        _validate_optimizer_state(optimizer, optimizer_state)
    student.load_state_dict(student_state, strict=True)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(dict(optimizer_state))

    if obs_normalizer is not None and normalizer_state is not None:
        obs_normalizer.load_state_dict(normalizer_state)

    return cast(dict[str, Any], checkpoint)
