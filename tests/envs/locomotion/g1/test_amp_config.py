from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.locomotion.g1.amp_terms import AMP_ANCHOR_BODY_NAME, AMP_BODY_NAMES

ROOT = Path(__file__).parents[4]


def _materialize_amp_owner():
    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/appo"), version_base="1.3"):
        owner = compose("config", overrides=["task=g1_amp_walk/mujoco"])
    override = BackendAdapter(owner, root_dir=ROOT, algo_name="appo").build_task_env_cfg_override()
    cfg = registry.materialize_env_config("G1AMPWalk")
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(cfg, override)
    cfg.validate()
    return owner, cfg, override


def test_amp_owner_is_fixed_forward_walk_only_and_has_no_shape_authority() -> None:
    owner, cfg, _ = _materialize_amp_owner()

    assert owner.training.task_name == "G1AMPWalk"
    assert owner.algo.runtime_resolver == "unilab.scripts.amp_appo:resolve_amp_appo_runtime"
    assert owner.algo.runtime_context_resolver == (
        "unilab.scripts.amp_appo:resolve_amp_runtime_context"
    )
    assert owner.algo.playback_policy_obs_mode == "actor"
    assert owner.algo.amp.expert_manifest == "amp_mjlab"
    assert cfg.sim_dt == 0.005
    assert cfg.ctrl_dt == 0.02
    assert cfg.sim_substeps == 4
    assert cfg.additional_observation_groups == {"amp": "amp_style"}
    assert list(cfg.observations) == ["policy", "critic", "amp_style"]
    assert "gait_phase" not in cfg.observations["policy"].terms
    assert "gait_phase" not in cfg.observations["critic"].terms
    assert set(cfg.rewards) == {
        "tracking_lin_vel",
        "tracking_ang_vel",
        "lin_vel_z",
        "ang_vel_xy",
        "orientation",
        "action_rate",
        "alive",
        "self_collisions",
    }
    assert "pose" not in cfg.rewards
    assert "feet_phase" not in cfg.rewards
    assert "base_height" not in cfg.rewards
    assert "base_height" not in cfg.terminations

    command = cfg.commands["twist"]
    assert tuple(command.ranges.lin_vel_x) == (1.0, 1.0)
    assert tuple(command.ranges.lin_vel_y) == (0.0, 0.0)
    assert tuple(command.ranges.ang_vel_z) == (0.0, 0.0)
    assert command.rel_standing_envs == 0.0
    assert command.heading_command is False

    assert cfg.scene is not None
    assert cfg.scene.fragment_files == ["src/unilab/assets/robots/g1/amp_walk_task.xml"]
    assert tuple(cfg.scene.entities["robot"].body_names or ()) == (
        *AMP_BODY_NAMES,
        AMP_ANCHOR_BODY_NAME,
    )


@pytest.mark.slow
def test_amp_mujoco_registry_env_exposes_195d_group_and_steps_once() -> None:
    pytest.importorskip("mujoco")
    _, _, override = _materialize_amp_owner()
    env = registry.make("G1AMPWalk", num_envs=2, sim_backend="mujoco", env_cfg_override=override)
    try:
        state = env.init_state()
        assert env.obs_groups_spec == {"obs": 96, "critic": 99, "amp": 195}
        assert state.obs["amp"].shape == (2, 195)
        assert state.obs["amp"].dtype == np.float32
        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert state.obs["amp"].shape == (2, 195)
        assert np.isfinite(state.obs["amp"]).all()
    finally:
        env.close()
