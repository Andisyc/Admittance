from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.locomotion.g1.manager_terms import (
    G1HeightCommandCfg,
    track_base_height_exp_smooth,
)

ROOT = Path(__file__).parents[4]


def _materialize(task: str, task_name: str) -> ManagerBasedRlEnvCfg:
    registry.ensure_registries()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/sac"), version_base="1.3"):
        owner = compose("config", overrides=[f"task={task}"])
    override = BackendAdapter(owner, root_dir=ROOT, algo_name="sac").build_task_env_cfg_override()
    cfg = registry.materialize_env_config(task_name)
    assert isinstance(cfg, ManagerBasedRlEnvCfg)
    apply_cfg_overrides(cfg, override)
    cfg.validate()
    return cfg


def test_height_owner_appends_scalar_command_and_reward_without_changing_flat() -> None:
    flat = _materialize("g1_walk_flat/mujoco", "G1WalkFlat")
    height = _materialize("g1_walk_height/mujoco", "G1WalkHeight")

    assert "height" not in flat.commands
    assert "height_command" not in flat.observations["policy"].terms
    assert "track_base_height_exp_smooth" not in flat.rewards

    command = height.commands["height"]
    assert isinstance(command, G1HeightCommandCfg)
    assert tuple(command.height_range) == pytest.approx((0.3, 0.754))
    assert command.default_height == pytest.approx(0.754)
    assert command.random_height_during_walking is True
    assert command.observe_height_command is True
    assert list(height.observations["policy"].terms)[-1] == "height_command"
    assert list(height.observations["critic"].terms)[-1] == "height_command"
    reward = height.rewards["track_base_height_exp_smooth"]
    assert reward.weight == pytest.approx(4.0)
    assert reward.params == {"tracking_sigma": 0.25, "command_name": "height"}


def test_height_command_sampling_and_reward_match_donor_oracle() -> None:
    env = SimpleNamespace(num_envs=64, rng=np.random.default_rng(7))
    command_cfg = G1HeightCommandCfg(
        resampling_time_range=(20.0, 20.0),
        height_range=(0.3, 0.754),
        default_height=0.754,
        random_height_during_walking=True,
        observe_height_command=True,
    )
    command = command_cfg.build(cast(Any, env))
    ids = np.arange(64, dtype=np.int32)
    command.reset(ids)
    assert command.command.shape == (64, 1)
    assert np.all(command.command >= 0.3)
    assert np.all(command.command <= 0.754)
    assert np.ptp(command.command) > 0.1

    target = np.asarray([[0.5], [0.7]], dtype=np.float32)
    root_pos = np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.8]], dtype=np.float32)
    reward_env = SimpleNamespace(
        num_envs=2,
        command_manager=SimpleNamespace(get_command=lambda name: target),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_pos_w=root_pos))},
    )
    actual = track_base_height_exp_smooth(cast(Any, reward_env), 0.25)
    expected = np.exp(-np.asarray([0.0, 0.1**2]) / 0.25)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)


def test_fixed_height_command_uses_donor_default() -> None:
    env = SimpleNamespace(num_envs=3, rng=np.random.default_rng(0))
    cfg = G1HeightCommandCfg(
        resampling_time_range=(20.0, 20.0),
        height_range=(0.3, 0.754),
        default_height=0.754,
        random_height_during_walking=False,
        observe_height_command=True,
    )
    term = cfg.build(cast(Any, env))
    term.reset(np.arange(3, dtype=np.int32))
    np.testing.assert_array_equal(term.command, np.full((3, 1), 0.754, dtype=term.command.dtype))
