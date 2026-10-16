from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
    normalized = name.lower()
    if normalized == "elu":
        return cast(type[nn.Module], nn.ELU)
    if normalized == "relu":
        return cast(type[nn.Module], nn.ReLU)
    if normalized == "silu":
        return cast(type[nn.Module], nn.SiLU)
    if normalized == "tanh":
        return cast(type[nn.Module], nn.Tanh)
    raise ValueError(f"Unsupported activation for distillation student: {name!r}")


class MLPStudentPolicy(nn.Module):
    """Small single-input student actor used by generic behavior distillation."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        squash_action: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.squash_action = bool(squash_action)
        if self.obs_dim <= 0 or self.action_dim <= 0 or any(int(d) <= 0 for d in hidden_dims):
            raise ValueError("student dimensions must be positive")

        activation_cls = _activation(activation)
        layers: list[nn.Module] = []
        in_dim = self.obs_dim
        for hidden_dim in hidden_dims:
            out_dim = int(hidden_dim)
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_cls())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim < 1 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Student obs dim mismatch: expected final dimension {self.obs_dim}, got {tuple(obs.shape)}"
            )
        action = self.net(obs)
        if self.squash_action:
            action = torch.tanh(action)
        return action

    @torch.no_grad()
    def explore(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self(obs)

    def as_export_module(self) -> nn.Module:
        return self
