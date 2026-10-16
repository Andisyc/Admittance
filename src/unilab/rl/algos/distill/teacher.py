"""Frozen SAC teacher loading adapted from ReGait's teacher owner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unilab.rl.algos.common.actor_factory import build_actor
from unilab.rl.algos.common.normalization import EmpiricalNormalization

from .checkpoint import validate_module_state
from .trainer import validate_matrix


@dataclass(frozen=True)
class SACTeacherSpec:
    checkpoint_path: str | Path
    obs_dim: int = 99
    action_dim: int = 29
    actor_hidden_dim: int = 512
    use_layer_norm: bool = True
    obs_normalization: bool = False

    def __post_init__(self) -> None:
        if min(self.obs_dim, self.action_dim, self.actor_hidden_dim) <= 0:
            raise ValueError("SAC teacher dimensions must be positive")


class LoadedTeacherPolicy(nn.Module):
    """Teacher normalizer and deterministic SAC action, with gradients disabled."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        spec: SACTeacherSpec,
        obs_normalizer: EmpiricalNormalization | None = None,
    ) -> None:
        super().__init__()
        self.actor: Any = actor
        self.obs_dim = spec.obs_dim
        self.action_dim = spec.action_dim
        self.obs_normalizer = obs_normalizer
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        validate_matrix(obs, "teacher_obs", width=self.obs_dim)
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer(obs, update=False)
        action = self.actor.explore(obs, deterministic=True)
        validate_matrix(action, "teacher_action", width=self.action_dim)
        if action.shape[0] != obs.shape[0]:
            raise ValueError("teacher action batch size mismatch")
        return action.detach()


def load_sac_teacher_policy(
    spec: SACTeacherSpec,
    *,
    device: str | torch.device = "cpu",
) -> LoadedTeacherPolicy:
    """Load the SAC ``actor`` and requested normalization, rejecting shape drift."""
    checkpoint: Any = torch.load(spec.checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("actor"), dict):
        raise ValueError("SAC teacher checkpoint does not contain actor")
    actor = build_actor(
        "sac", spec.obs_dim, spec.action_dim, spec.actor_hidden_dim, spec.use_layer_norm, device
    )
    validate_module_state(actor, checkpoint["actor"], label="SAC teacher actor")
    normalizer = None
    if spec.obs_normalization:
        state = checkpoint.get("obs_normalizer")
        if not isinstance(state, dict) or not state:
            raise ValueError("SAC teacher requires checkpoint obs_normalizer")
        normalizer = EmpiricalNormalization(shape=spec.obs_dim, device=device)
        validate_module_state(normalizer, state, label="SAC teacher obs_normalizer")
        normalizer.load_state_dict(state, strict=True)
        normalizer.eval()
    actor.load_state_dict(checkpoint["actor"], strict=True)
    return LoadedTeacherPolicy(actor=actor, spec=spec, obs_normalizer=normalizer)
