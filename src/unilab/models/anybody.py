# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure MUSE-Transformer encoder and state-conditioned decoder.

Adapted from hazel-hammer/Anybody at 0cffa139eab0bc145c9216223e249f03c67a324c,
source/rsl_rl/rsl_rl/modules/latent_bottleneck_muse_transformer.py.
Only the deterministic network is ported: no teacher, distribution, normalizer,
simulator, history storage, or training objective is owned by this module.

The default G1 input is [goal(73), term-major proprio history(450), mask(1)].
History is oldest first within each term. Decoder history is instead flattened
frame first, after reconstructing [..., 5, 90]. Mask is raw 0/1, never normalized.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

_GOAL_TERMS = (58, 6, 3, 3, 3)
_PROPRIO_TERMS = (29, 29, 3, 29)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _term_sizes(values: Sequence[int], name: str) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    return tuple(_positive_int(value, name) for value in values)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _require_tensor_like(tensor: torch.Tensor, weight: torch.Tensor, name: str) -> None:
    if tensor.dtype != weight.dtype or tensor.device != weight.device:
        raise ValueError(
            f"{name} must match model dtype/device {weight.dtype}/{weight.device}, "
            f"got {tensor.dtype}/{tensor.device}"
        )


class MUSETransformerEncoder(nn.Module):
    """Encode a flat observation into raw mu and a normalized deterministic z.

    ``encode`` returns both tensors. Calling the module returns only z.
    Leading batch axes are preserved, including an unbatched 1-D observation.
    The CLS token and proprio tokens remain visible even when goal is masked.
    """

    def __init__(
        self,
        *,
        goal_term_sizes: Sequence[int] = _GOAL_TERMS,
        proprio_term_sizes: Sequence[int] = _PROPRIO_TERMS,
        history_length: int = 5,
        latent_dim: int = 16,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 768,
    ) -> None:
        super().__init__()
        self.goal_term_sizes = _term_sizes(goal_term_sizes, "goal_term_sizes")
        self.proprio_term_sizes = _term_sizes(proprio_term_sizes, "proprio_term_sizes")
        self.history_length = _positive_int(history_length, "history_length")
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.d_model = _positive_int(d_model, "d_model")
        nhead = _positive_int(nhead, "nhead")
        num_layers = _positive_int(num_layers, "num_layers")
        ffn_dim = _positive_int(ffn_dim, "ffn_dim")
        if self.d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.goal_dim = sum(self.goal_term_sizes)
        self.proprio_dim = sum(self.proprio_term_sizes)
        self.obs_dim = self.goal_dim + self.history_length * self.proprio_dim + 1
        self.seq_len = 2 + self.history_length

        self.goal_proj = nn.Linear(self.goal_dim, self.d_model)
        self.proprio_proj = nn.Linear(self.proprio_dim, self.d_model)
        self.proprio_pos_emb = nn.Parameter(torch.zeros(self.history_length, self.d_model))
        nn.init.normal_(self.proprio_pos_emb, std=0.02)
        self.goal_pos_emb = nn.Parameter(torch.zeros(1, self.d_model))
        nn.init.normal_(self.goal_pos_emb, std=0.02)
        self.modality_emb = nn.Parameter(torch.zeros(2, self.d_model))
        nn.init.normal_(self.modality_emb, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm disables nested tensors in the donor too; make that explicit.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.mu_head = nn.Linear(self.d_model, self.latent_dim)

    def split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return goal [..., G], frame-major proprio [..., H, P], mask [..., 1]."""
        if obs.ndim < 1 or obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs must have last dimension {self.obs_dim}, got {tuple(obs.shape)}")
        _require_tensor_like(obs, self.goal_proj.weight, "obs")
        _require_finite(obs, "obs")
        goal = obs[..., : self.goal_dim]
        offset = self.goal_dim
        parts = []
        for size in self.proprio_term_sizes:
            width = self.history_length * size
            parts.append(
                obs[..., offset : offset + width].reshape(
                    *obs.shape[:-1], self.history_length, size
                )
            )
            offset += width
        proprio = torch.cat(parts, dim=-1)
        mask = obs[..., offset : offset + 1]
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("goal_mask must be raw 0 or 1")
        return goal, proprio, mask

    def _encode_parts(
        self, goal: torch.Tensor, proprio: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = goal.shape[:-1]
        goal_flat = goal.reshape(-1, self.goal_dim)
        proprio_flat = proprio.reshape(-1, self.history_length, self.proprio_dim)
        mask_flat = mask.reshape(-1, 1).bool()
        goal_clean = torch.where(mask_flat, torch.zeros_like(goal_flat), goal_flat)
        goal_token = self.goal_proj(goal_clean).unsqueeze(1)
        goal_token = goal_token + self.goal_pos_emb.unsqueeze(0) + self.modality_emb[0]
        proprio_tokens = self.proprio_proj(proprio_flat)
        proprio_tokens = proprio_tokens + self.proprio_pos_emb.unsqueeze(0) + self.modality_emb[1]
        cls = self.cls_token.expand(goal_flat.shape[0], 1, self.d_model)
        tokens = torch.cat((cls, goal_token, proprio_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros_like(mask_flat),
                mask_flat,
                torch.zeros(
                    (goal_flat.shape[0], self.history_length), dtype=torch.bool, device=goal.device
                ),
            ),
            dim=1,
        )
        output = self.transformer(tokens, src_key_padding_mask=padding)
        mu = self.mu_head(output[:, 0]).reshape(*batch_shape, self.latent_dim)
        _require_finite(mu, "encoder mu")
        return mu, F.normalize(mu, p=2.0, dim=-1, eps=1e-8)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_parts(*self.split_obs(obs))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encode(obs)[1]


class MUSEDecoder(nn.Module):
    """Decode z and frame-major proprio history into unscaled joint commands."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        proprio_dim: int = 90,
        history_length: int = 5,
        num_actions: int = 29,
        hidden_dims: Sequence[int] = (1024, 512, 256, 128),
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_int(latent_dim, "latent_dim")
        self.proprio_dim = _positive_int(proprio_dim, "proprio_dim")
        self.history_length = _positive_int(history_length, "history_length")
        self.num_actions = _positive_int(num_actions, "num_actions")
        hidden = _term_sizes(hidden_dims, "hidden_dims")
        self.input_dim = self.latent_dim + self.history_length * self.proprio_dim
        layers: list[nn.Module] = []
        width = self.input_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, next_width), nn.GELU()))
            width = next_width
        layers.append(nn.Linear(width, self.num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, proprio_history: torch.Tensor) -> torch.Tensor:
        if z.ndim < 1 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have last dimension {self.latent_dim}, got {tuple(z.shape)}")
        expected = (*z.shape[:-1], self.history_length, self.proprio_dim)
        if tuple(proprio_history.shape) != expected:
            raise ValueError(
                f"proprio_history must have shape {expected}, got {tuple(proprio_history.shape)}"
            )
        weight = next(self.net.parameters())
        _require_tensor_like(z, weight, "z")
        _require_tensor_like(proprio_history, weight, "proprio_history")
        _require_finite(z, "z")
        _require_finite(proprio_history, "proprio_history")
        flat = proprio_history.reshape(*z.shape[:-1], self.history_length * self.proprio_dim)
        action = self.net(torch.cat((z, flat), dim=-1))
        _require_finite(action, "decoder action")
        return action


class MUSEEncoderDecoder(nn.Module):
    """Compose the pure networks; output raw actions without sampling or scaling."""

    def __init__(
        self,
        *,
        goal_term_sizes: Sequence[int] = _GOAL_TERMS,
        proprio_term_sizes: Sequence[int] = _PROPRIO_TERMS,
        history_length: int = 5,
        latent_dim: int = 16,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 768,
        num_actions: int = 29,
        decoder_hidden_dims: Sequence[int] = (1024, 512, 256, 128),
    ) -> None:
        super().__init__()
        self.encoder = MUSETransformerEncoder(
            goal_term_sizes=goal_term_sizes,
            proprio_term_sizes=proprio_term_sizes,
            history_length=history_length,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
        )
        self.decoder = MUSEDecoder(
            latent_dim=latent_dim,
            proprio_dim=self.encoder.proprio_dim,
            history_length=history_length,
            num_actions=num_actions,
            hidden_dims=decoder_hidden_dims,
        )
        self.obs_dim = self.encoder.obs_dim

    def split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder.split_obs(obs)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode(obs)

    def decode(self, z: torch.Tensor, proprio_history: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, proprio_history)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        goal, proprio, mask = self.split_obs(obs)
        _, z = self.encoder._encode_parts(goal, proprio, mask)
        return self.decode(z, proprio)
