"""YAML-to-real-manager MUSE observations without environment or physics stepping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from unilab.base.registry import apply_cfg_overrides
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from unilab.managers import ActionManager, ActionTerm, ActionTermCfg, ObservationManager
from unilab.tasks.motion_tracking.common import anybody_observations, manager_terms

_CONF = Path(__file__).resolve().parents[2] / "src/unilab/conf/appo"


def _compose(*overrides):
    with initialize_config_dir(config_dir=str(_CONF), version_base="1.3"):
        return compose("config", overrides=["task=anybody_g1/mujoco", *overrides])


@dataclass(kw_only=True)
class _RawActionCfg(ActionTermCfg):
    def build(self, env):
        return _RawAction(self, env)


class _RawAction(ActionTerm):
    """Test-only action storage; applying any physical control is forbidden."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = np.zeros((env.num_envs, 29), dtype=np.float32)

    @property
    def action_dim(self):
        return 29

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions

    def apply_actions(self):
        raise AssertionError("observation tests must not apply physical actions")

    def reset(self, env_ids=None):
        self._raw[slice(None) if env_ids is None else env_ids] = 0


class _FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.rng = np.random.default_rng(15)
        self.data = SimpleNamespace(
            default_joint_pos=np.tile(np.linspace(-0.4, 0.2, 29), (2, 1)),
            default_joint_vel=np.tile(np.linspace(0.1, 0.2, 29), (2, 1)),
        )
        self.scene = {"robot": SimpleNamespace(data=self.data)}
        self.action_manager = ActionManager({"joint_pos": _RawActionCfg(entity_name="robot")}, self)
        self.reference_rotation = Rotation.from_euler("xyz", [[0.3, -0.5, 1.1], [-0.7, 0.2, -0.4]])
        self.robot_anchor_rotation = Rotation.from_euler(
            "xyz", [[-0.8, 0.2, -0.6], [0.2, -0.3, 1.4]]
        )
        self.command = SimpleNamespace(
            anchor_quat_w=self.reference_rotation.as_quat()[:, [3, 0, 1, 2]],
            anchor_lin_vel_w=np.array([[0.7, -1.3, 2.1], [-0.8, 0.6, 1.7]]),
        )
        self.set_frame(0)

    def set_frame(self, frame):
        row = np.arange(2)[:, None] * 1000
        joint = np.arange(29)[None] * 0.125
        self.data.joint_pos = self.data.default_joint_pos + row + 10 + frame + joint
        self.data.joint_vel = self.data.default_joint_vel + row + 100 + 2 * frame + joint
        self.data.root_link_ang_vel_b = row + np.array([200, 201, 202])[None] + 3 * frame
        self.data.root_link_lin_vel_b = row + np.array([300, 301, 302])[None] + 4 * frame
        self.action_manager.process_action(
            np.asarray(row + 400 + 5 * frame + joint, dtype=np.float32)
        )
        self.command.robot_joint_pos = self.data.joint_pos
        self.command.robot_joint_vel = self.data.joint_vel
        self.command.joint_pos = self.data.joint_pos + np.linspace(-2, 1, 29)[None] + frame
        self.command.joint_vel = self.data.joint_vel + np.linspace(3, 7, 29)[None] - frame
        relative = self.robot_anchor_rotation.inv() * self.reference_rotation
        self.command.motion_anchor_ori_b = relative.as_matrix()[:, :, :2].reshape(2, 6)
        self.command.motion_anchor_pos_b = self.robot_anchor_rotation.inv().apply(
            np.array([[0.1, 0.7, -0.5], [-0.6, 0.2, 0.9]]) + frame
        )


@pytest.fixture
def observation_case(monkeypatch):
    env = _FakeEnv()
    monkeypatch.setattr(anybody_observations, "_command", lambda _env, _name: env.command)
    monkeypatch.setattr(manager_terms, "_command", lambda _env, _name: env.command)
    cfg = ManagerBasedRlEnvCfg()
    raw = OmegaConf.to_container(_compose().env.observations.actor, resolve=True)
    # Use the production materializer and actual YAML terms, not a second
    # hand-authored set of ObservationTermCfg functions or a mocked manager.
    apply_cfg_overrides(cfg, {"observations": {"actor": raw}})
    manager = ObservationManager({"actor": cfg.observations["actor"]}, env)
    return env, manager


def _proprio(env):
    return (
        env.data.joint_pos - env.data.default_joint_pos,
        env.data.joint_vel - env.data.default_joint_vel,
        env.data.root_link_ang_vel_b,
        env.action_manager.action.copy(),
    )


def _expected_goal(env):
    cmd = env.command
    return np.concatenate(
        (
            cmd.joint_pos - cmd.robot_joint_pos,
            cmd.joint_vel - cmd.robot_joint_vel,
            cmd.motion_anchor_ori_b,
            cmd.motion_anchor_pos_b,
            env.data.root_link_lin_vel_b,
            env.reference_rotation.inv().apply(cmd.anchor_lin_vel_w),
        ),
        axis=-1,
    )


def test_hydra_profile_is_direct_rl_and_requires_motion_file():
    cfg = _compose()
    assert cfg.algo.actor.class_name == "unilab.rl.algos.anybody.MUSEActorModel"
    assert cfg.algo.critic.class_name == "rsl_rl.models.MLPModel"
    assert cfg.algo.actor.obs_normalization is False
    assert cfg.algo.critic.obs_normalization is False
    assert cfg.algo.actor.history_length == 5
    assert cfg.algo.actor.goal_term_sizes == [58, 6, 3, 3, 3]
    assert cfg.algo.actor.proprio_term_sizes == [29, 29, 3, 29]
    assert "teacher" not in cfg.algo
    params = cfg.env.commands.motion.params
    assert OmegaConf.is_missing(params, "motion_file")
    configured = _compose("env.commands.motion.params.motion_file=/remote/motions/converted.npz")
    assert configured.env.commands.motion.params.motion_file == "/remote/motions/converted.npz"


def test_real_manager_matches_524_physical_order_and_reference_velocity_frame(observation_case):
    env, manager = observation_case
    actual = manager.compute(update_history=True)["actor"]
    assert actual.shape == (2, 524)
    np.testing.assert_allclose(actual[:, :73], _expected_goal(env), atol=1e-12)
    offset = 73
    for term in _proprio(env):
        expected = np.repeat(term[:, None], 5, axis=1).reshape(2, -1)
        np.testing.assert_array_equal(actual[:, offset : offset + expected.shape[1]], expected)
        offset += expected.shape[1]
    assert offset == 523
    np.testing.assert_array_equal(actual[:, 523], 0)
    wrong_frame = env.robot_anchor_rotation.inv().apply(env.command.anchor_lin_vel_w)
    assert not np.allclose(actual[:, 70:73], wrong_frame)
    assert not np.allclose(actual[:, 70:73], env.command.anchor_lin_vel_w)


def test_history_term_major_oldest_first_and_repeated_compute_does_not_push(observation_case):
    env, manager = observation_case
    recorded = []
    for frame in range(7):
        env.set_frame(frame)
        recorded.append(tuple(term.copy() for term in _proprio(env)))
        actual = manager.compute(update_history=True)["actor"]
    expected_history = np.concatenate(
        [
            np.stack([frame[term] for frame in recorded[-5:]], axis=1).reshape(2, -1)
            for term in range(4)
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(actual[:, 73:523], expected_history)
    saved = actual.copy()
    env.set_frame(100)
    # Re-reading after a policy observation fetch must return the same cached
    # control-step observation, even if the underlying test state is changed.
    np.testing.assert_array_equal(manager.compute()["actor"], saved)
    np.testing.assert_array_equal(manager.compute()["actor"], saved)
    for term in manager._group_obs_term_history_buffer["actor"].values():
        np.testing.assert_array_equal(term._num_pushes, 7)


def test_partial_reset_backfills_only_reset_row_and_clears_raw_action(observation_case):
    env, manager = observation_case
    for frame in range(6):
        env.set_frame(frame)
        manager.compute(update_history=True)
    histories = manager._group_obs_term_history_buffer["actor"]
    previous = {name: buffer.buffer.copy() for name, buffer in histories.items()}
    ids = np.array([1], dtype=np.int32)
    manager.reset(ids)
    env.set_frame(20)
    env.action_manager.reset(ids)
    current = _proprio(env)
    rows = manager.compute(update_history=True, env_ids=ids)["actor"]
    assert rows.shape == (1, 524)
    np.testing.assert_array_equal(rows[:, 378:523], 0)
    np.testing.assert_array_equal(env.action_manager.action[1], 0)
    assert np.any(env.action_manager.action[0] != 0)
    for (name, buffer), expected in zip(histories.items(), current):
        np.testing.assert_array_equal(buffer.buffer[0], previous[name][0])
        np.testing.assert_array_equal(buffer.buffer[1], np.repeat(expected[1:2], 5, axis=0))
        np.testing.assert_array_equal(buffer._num_pushes, [6, 1])
    np.testing.assert_allclose(rows[:, :73], _expected_goal(env)[ids], atol=1e-12)


def test_yaml_strict_nan_policy_rejects_nonfinite_current_goal(observation_case):
    env, manager = observation_case
    env.command.anchor_lin_vel_w[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        manager.compute(update_history=True)
