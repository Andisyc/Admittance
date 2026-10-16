"""AMP composition over generic APPO; expert assets are resolved by the caller."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from unilab.rl.algos.amp.checkpoint import save_amp_checkpoint_atomic
from unilab.rl.algos.amp.data import AMPExpertTransitions
from unilab.rl.algos.amp.learner import AMPAPPOLearner
from unilab.rl.algos.amp.spec import AMP_OBSERVATION_DIM
from unilab.rl.algos.appo.runner import APPORunner


class AMPAPPORunner(APPORunner):
    def __init__(
        self,
        *args,
        amp_expert_current: np.ndarray,
        amp_expert_next: np.ndarray,
        amp_expert_provenance: dict[str, Any],
        amp_method_fingerprint: dict[str, Any],
        **kwargs,
    ) -> None:
        self.amp_expert = AMPExpertTransitions(
            amp_expert_current, amp_expert_next, amp_expert_provenance
        )
        if not isinstance(amp_method_fingerprint, dict) or not amp_method_fingerprint:
            raise ValueError("AMP requires the effective method fingerprint")
        self.amp_method_fingerprint = copy.deepcopy(amp_method_fingerprint)
        super().__init__(*args, **kwargs)

    def _on_env_probe(self, env: Any) -> None:
        if env.obs_groups_spec.get("amp") != AMP_OBSERVATION_DIM:
            raise ValueError("AMP env probe requires a 195-D 'amp' observation group")

    def _learner_class(self):
        return AMPAPPOLearner

    def _learner_extra_kwargs(self) -> dict[str, Any]:
        cfg = self.rl_cfg.get("amp")
        if not isinstance(cfg, dict) or cfg.get("observation_dim") != AMP_OBSERVATION_DIM:
            raise ValueError("AMP owner configuration requires observation_dim=195")
        fingerprint = self.amp_method_fingerprint
        if fingerprint.get("amp") != cfg or fingerprint.get("expert") != self.amp_expert.provenance:
            raise ValueError("AMP fingerprint does not match the effective config/expert")
        observation = fingerprint.get("observation", {})
        runtime = fingerprint.get("runtime")
        if not isinstance(observation, dict) or not isinstance(runtime, dict):
            raise ValueError("AMP fingerprint requires observation and runtime identities")
        playback_identity = {
            "actor": copy.deepcopy(self.rl_cfg.get("actor", {})),
            "policy_observation_mapping": copy.deepcopy(
                observation.get("policy_observation_mapping")
            ),
            "action_dim": self.action_dim,
            "sim2sim_runtime": copy.deepcopy(runtime),
        }
        return {
            "amp_expert_current": self.amp_expert.current,
            "amp_expert_next": self.amp_expert.next,
            "amp_expert_provenance": self.amp_expert.provenance,
            "amp_method_fingerprint": fingerprint,
            "amp_playback_identity": playback_identity,
            "amp_hidden_dims": cfg.get("hidden_dims", [1024, 512, 256]),
            "amp_reward_coef": cfg.get("reward_coef", 0.1),
            "amp_task_reward_lerp": cfg.get("task_reward_lerp", 0.75),
            "amp_replay_capacity": cfg.get("replay_capacity", 32768),
            "amp_discriminator_batch_size": cfg.get("discriminator_batch_size", 4096),
            "amp_discriminator_updates": cfg.get("discriminator_updates", 1),
            "amp_discriminator_learning_rate": cfg.get("discriminator_learning_rate", 1e-3),
            "amp_gradient_penalty": cfg.get("gradient_penalty", 10.0),
            "amp_normalizer_epsilon": cfg.get("normalizer_epsilon", 1e-5),
            "amp_normalizer_clip": cfg.get("normalizer_clip", 5.0),
            "amp_seed": cfg.get("seed", self.seed if self.seed is not None else 0),
        }

    def _rollout_observation_specs(self) -> dict[str, int]:
        return {
            "amp_state": AMP_OBSERVATION_DIM,
            "amp_next_state": AMP_OBSERVATION_DIM,
            "timeout_bootstrap": 1,
        }

    def _collector_runtime_kwargs(self) -> dict[str, Any]:
        return {
            "rollout_payload_writer": "unilab.rl.algos.amp.worker:write_amp_rollout_payload",
            "rollout_snapshot_keys": ("amp",),
        }

    def _restore_learner_checkpoint(self, learner, checkpoint: dict[str, Any]) -> None:
        learner.load_state_dict(checkpoint)

    def _save_checkpoint(self, learner, checkpoint_path: str) -> None:
        save_amp_checkpoint_atomic(learner.get_state_dict(), checkpoint_path)
