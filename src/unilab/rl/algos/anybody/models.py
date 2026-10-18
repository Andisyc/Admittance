"""Expose the pure MUSE network through APPO's RSL model contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from numbers import Real

from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import resolve_callable
from tensordict import TensorDict
from torch import nn

from unilab.models import MUSEEncoderDecoder


class MUSEActorModel(MLPModel):
    """Use MUSE as the raw Gaussian mean without changing APPO's inference path.

    Both collector inference and learner loss evaluation consume ``mlp`` after
    ``obs_normalizer``. Keep all network processing inside that shared path;
    normalizing the flat observation would corrupt its raw binary goal mask.
    ``hidden_dims`` configures the decoder, whose existing activation is GELU.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: Sequence[int] = (1024, 512, 256, 128),
        activation: str = "gelu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        *,
        history_length: int = 5,
        latent_dim: int = 16,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 768,
        goal_term_sizes: Sequence[int] = (58, 6, 3, 3, 3),
        proprio_term_sizes: Sequence[int] = (29, 29, 3, 29),
    ) -> None:
        if obs_normalization is not False:
            raise ValueError("MUSE requires obs_normalization=False to preserve the raw goal mask")
        if activation != "gelu":
            raise ValueError("MUSE's existing decoder supports only activation='gelu'")
        if list(obs_groups.get(obs_set, ())) != ["policy"]:
            raise ValueError("MUSE requires exactly one observation group named 'policy'")
        if output_dim != 29:
            raise ValueError("The G1 MUSE actor requires output_dim=29")

        dist_cfg = (
            deepcopy(distribution_cfg)
            if distribution_cfg is not None
            else {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
        if resolve_callable(dist_cfg.get("class_name", "")) is not GaussianDistribution:
            raise ValueError(
                "MUSE APPO requires the standard state-independent GaussianDistribution"
            )
        if set(dist_cfg) - {"class_name", "init_std", "std_type"}:
            raise ValueError("MUSE GaussianDistribution accepts only init_std and std_type")
        init_std = dist_cfg.get("init_std", 1.0)
        if (
            isinstance(init_std, bool)
            or not isinstance(init_std, Real)
            or not math.isfinite(init_std)
            or init_std <= 0
        ):
            raise ValueError("MUSE Gaussian init_std must be finite and strictly positive")
        if dist_cfg.get("std_type", "scalar") not in ("scalar", "log"):
            raise ValueError("MUSE Gaussian std_type must be 'scalar' or 'log'")

        # The base class owns distribution/observation protocols. Its temporary
        # tiny MLP is replaced before the actor reaches any optimizer or runtime.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=(1,),
            activation="elu",
            obs_normalization=False,
            distribution_cfg=dist_cfg,
        )
        network = MUSEEncoderDecoder(
            goal_term_sizes=goal_term_sizes,
            proprio_term_sizes=proprio_term_sizes,
            history_length=history_length,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            num_actions=output_dim,
            decoder_hidden_dims=hidden_dims,
        )
        if self.obs_dim != network.obs_dim:
            raise ValueError(
                f"MUSE policy observation width must be {network.obs_dim}, got {self.obs_dim}"
            )
        self.mlp = network

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("MUSE TorchScript export has not been validated yet")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("MUSE ONNX export has not been validated yet")
