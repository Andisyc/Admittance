"""UniLab-owned composition for the package-neutral AMP APPO runtime."""

from __future__ import annotations

import copy
import importlib.metadata
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from uni_rl.algos.amp.runner import AMPAPPORunner

from unilab.tasks.locomotion.g1.amp_motion import load_g1_amp_expert_transitions
from unilab.tasks.locomotion.g1.amp_terms import (
    AMP_ANCHOR_BODY_INDEX,
    AMP_ANCHOR_BODY_NAME,
    AMP_BODY_INDICES,
    AMP_BODY_NAMES,
    AMP_OBSERVATION_DIM,
)


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "development"


def amp_runtime_identity() -> dict[str, Any]:
    return {
        "sim_backend": "mujoco",
        "unilab": _version("unilab"),
        "unisim_core": _version("unisim-core"),
        "unilab_rl": _version("unilab-rl"),
        "mujoco_uni_runtime": _version("mujoco-uni-runtime"),
    }


def build_amp_playback_identity(rl_cfg: dict[str, Any], *, action_dim: int) -> dict[str, Any]:
    """Build the narrow identity used before loading an AMP actor."""
    context = rl_cfg.get("runtime_context")
    observation = context.get("observation") if isinstance(context, Mapping) else None
    mapping = (
        observation.get("policy_observation_mapping") if isinstance(observation, Mapping) else None
    )
    if not isinstance(mapping, Mapping):
        raise ValueError("AMP playback requires a validated observation runtime_context")
    return {
        "actor": copy.deepcopy(rl_cfg.get("actor", {})),
        "policy_observation_mapping": copy.deepcopy(dict(mapping)),
        "action_dim": int(action_dim),
        "sim2sim_runtime": amp_runtime_identity(),
    }


def resolve_amp_runtime_context(cfg: Any, env_cfg_override: Mapping[str, Any]) -> dict[str, Any]:
    """Validate frozen AMP semantics from the effective Hydra environment owner."""
    if str(cfg.training.task_name) != "G1AMPWalk" or str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("AMP Phase 3 requires task=G1AMPWalk and sim_backend=mujoco")
    env = dict(env_cfg_override)
    if float(env.get("sim_dt", -1.0)) != 0.005 or float(env.get("ctrl_dt", -1.0)) != 0.02:
        raise ValueError("G1AMPWalk requires sim_dt=0.005 and ctrl_dt=0.02")
    commands = env.get("commands")
    twist = commands.get("twist") if isinstance(commands, Mapping) else None
    if not isinstance(twist, Mapping):
        raise ValueError("G1AMPWalk requires the fixed twist command owner")
    expected_twist = {
        "_target_": "unilab.tasks.locomotion.g1.manager_terms.G1VelocityCommandCfg",
        "entity_name": "robot",
        "resampling_time_range": [20.0, 20.0],
        "heading_command": False,
        "heading_control_stiffness": 0.5,
        "rel_standing_envs": 0.0,
        "rel_heading_envs": 0.0,
        "rel_world_envs": 0.0,
        "rel_forward_envs": 0.0,
        "init_velocity_prob": 0.0,
        "planar_dead_zone": 0.0,
        "ranges": {
            "lin_vel_x": [1.0, 1.0],
            "lin_vel_y": [0.0, 0.0],
            "ang_vel_z": [0.0, 0.0],
        },
    }
    if dict(twist) != expected_twist:
        raise ValueError("G1AMPWalk command owner differs from the frozen contract")

    rewards = env.get("rewards")
    if not isinstance(rewards, Mapping):
        raise ValueError("G1AMPWalk requires explicit task rewards")
    expected_rewards = {
        "tracking_lin_vel": {
            "func": "unilab.tasks.locomotion.common.sensor_reward_terms.track_lin_vel",
            "weight": 2.0,
            "params": {
                "tracking_sigma": 0.25,
                "command_name": "twist",
                "sensor_name": "pelvis_local_linvel",
            },
        },
        "tracking_ang_vel": {
            "func": "unilab.tasks.locomotion.common.sensor_reward_terms.track_ang_vel",
            "weight": 0.2,
            "params": {
                "tracking_sigma": 0.25,
                "command_name": "twist",
                "sensor_name": "torso_gyro",
            },
        },
        "lin_vel_z": {
            "func": "unilab.tasks.locomotion.common.sensor_reward_terms.lin_vel_z",
            "weight": -1.0,
            "params": {"sensor_name": "pelvis_local_linvel"},
        },
        "ang_vel_xy": {
            "func": "unilab.tasks.locomotion.common.sensor_reward_terms.ang_vel_xy",
            "weight": -0.25,
            "params": {"sensor_name": "torso_gyro"},
        },
        "orientation": {
            "func": "unilab.tasks.locomotion.common.sensor_reward_terms.orientation",
            "weight": -5.0,
            "params": {"sensor_name": "torso_upvector"},
        },
        "action_rate": {"func": "unilab.envs.mdp.action_rate_l2", "weight": -0.01},
        "alive": {
            "func": "unilab.tasks.locomotion.common.manager_terms.alive",
            "weight": 1.0,
        },
        "self_collisions": {
            "func": "unilab.tasks.locomotion.g1.amp_terms.SelfCollisionPenalty",
            "weight": -0.1,
            "params": {
                "sensor_name": "self_collision",
                "history_length": 4,
                "force_threshold": 10.0,
            },
        },
    }
    if dict(rewards) != expected_rewards:
        raise ValueError("G1AMPWalk reward schema differs from the frozen contract")
    observations = env.get("observations")
    if not isinstance(observations, Mapping):
        raise ValueError("G1AMPWalk requires explicit observation groups")
    for group in ("policy", "critic"):
        group_cfg = observations.get(group)
        terms = group_cfg.get("terms") if isinstance(group_cfg, Mapping) else None
        if not isinstance(terms, Mapping) or "gait_phase" in terms:
            raise ValueError("G1AMPWalk policy and critic must not contain gait_phase")
    mapping = env.get("additional_observation_groups")
    if not isinstance(mapping, Mapping) or dict(mapping) != {"amp": "amp_style"}:
        raise ValueError("G1AMPWalk must publish the amp_style group as observation key 'amp'")
    expected_amp_group = {
        "terms": {
            "state": {
                "func": "unilab.tasks.locomotion.g1.amp_terms.G1AmpObservation",
                "params": {
                    "asset_cfg": {
                        "_target_": "unilab.managers.SceneEntityCfg",
                        "name": "robot",
                        "body_names": [*AMP_BODY_NAMES, AMP_ANCHOR_BODY_NAME],
                        "preserve_order": True,
                    }
                },
            }
        }
    }
    if observations.get("amp_style") != expected_amp_group:
        raise ValueError("G1AMPWalk AMP observation owner differs from the frozen contract")
    scene = env.get("scene")
    fragments = scene.get("fragment_files") if isinstance(scene, Mapping) else None
    if fragments != ["src/unilab/assets/robots/g1/amp_walk_task.xml"]:
        raise ValueError("G1AMPWalk requires the AMP self-collision task fragment")
    return {
        "task": {
            "name": "G1AMPWalk",
            "command": [1.0, 0.0, 0.0],
            "sim_dt": float(env["sim_dt"]),
            "ctrl_dt": float(env["ctrl_dt"]),
            "scene_fragment_files": copy.deepcopy(fragments),
            "command_owner": copy.deepcopy(expected_twist),
            "reward_owner": copy.deepcopy(expected_rewards),
        },
        "observation": {
            "amp_dim": AMP_OBSERVATION_DIM,
            "body_names": list(AMP_BODY_NAMES),
            "body_indices": list(AMP_BODY_INDICES),
            "anchor_body_name": AMP_ANCHOR_BODY_NAME,
            "anchor_body_index": AMP_ANCHOR_BODY_INDEX,
            "groups": copy.deepcopy(dict(observations)),
            "policy_observation_mapping": {
                "obs": str(env.get("policy_observation_group")),
                "critic": str(env.get("critic_observation_group")),
                "amp": str(mapping["amp"]),
            },
        },
    }


def _method_fingerprint(rl_cfg: dict[str, Any], expert: dict[str, Any]) -> dict[str, Any]:
    context = rl_cfg.get("runtime_context")
    if not isinstance(context, Mapping) or set(context) != {"task", "observation"}:
        raise ValueError("AMP runtime requires a validated task/observation runtime_context")
    return {
        "task": copy.deepcopy(context["task"]),
        "observation": copy.deepcopy(context["observation"]),
        "expert": copy.deepcopy(expert),
        "amp": copy.deepcopy(rl_cfg.get("amp", {})),
        "appo": copy.deepcopy(rl_cfg.get("algorithm", {})),
        "actor": copy.deepcopy(rl_cfg.get("actor", {})),
        "critic": copy.deepcopy(rl_cfg.get("critic", rl_cfg.get("actor", {}))),
        "runtime": amp_runtime_identity(),
    }


def resolve_amp_appo_runtime(rl_cfg: dict[str, Any]) -> SimpleNamespace:
    """Resolve mode-scoped AMP runtime data before runner allocation."""
    if rl_cfg.get("runtime_impl") != "amp_appo":
        raise ValueError("AMP resolver requires algo.runtime_impl=amp_appo")
    amp_cfg = rl_cfg.get("amp")
    if not isinstance(amp_cfg, dict) or amp_cfg.get("observation_dim") != AMP_OBSERVATION_DIM:
        raise ValueError(f"AMP owner config must declare observation_dim={AMP_OBSERVATION_DIM}")
    mode = rl_cfg.get("runtime_mode")
    if mode not in {"train", "play"}:
        raise ValueError("AMP resolver requires runtime_mode='train' or 'play'")
    if mode == "play":
        return SimpleNamespace(runner_cls=AMPAPPORunner, play_fn=None, runner_kwargs={})
    expert_manifest = amp_cfg.get("expert_manifest", "amp_mjlab")
    if not isinstance(expert_manifest, str) or not expert_manifest:
        raise ValueError("AMP owner config expert_manifest must be a non-empty string")
    transitions = load_g1_amp_expert_transitions(expert_manifest)
    provenance = copy.deepcopy(transitions.provenance)
    return SimpleNamespace(
        runner_cls=AMPAPPORunner,
        play_fn=None,
        runner_kwargs={
            "amp_expert_current": transitions.current,
            "amp_expert_next": transitions.next,
            "amp_expert_provenance": provenance,
            "amp_method_fingerprint": _method_fingerprint(rl_cfg, provenance),
        },
    )


__all__ = [
    "amp_runtime_identity",
    "build_amp_playback_identity",
    "resolve_amp_runtime_context",
    "resolve_amp_appo_runtime",
]
