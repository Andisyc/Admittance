from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from unilab.managers import ManagerTermBaseCfg
from unilab.tasks.locomotion.g1.amp_terms import (
    AMP_BODY_NAMES,
    AMP_OBSERVATION_DIM,
    SelfCollisionPenalty,
    compute_g1_amp_observation,
    compute_self_collision_cost,
)


def test_amp_observation_has_frozen_shape_order_and_local_velocity() -> None:
    batch = 2
    body_count = len(AMP_BODY_NAMES)
    positions = np.zeros((batch, body_count, 3), dtype=np.float32)
    positions[..., 0] = np.arange(body_count, dtype=np.float32)
    quaternions = np.zeros((batch, body_count, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    linear = np.zeros((batch, body_count, 3), dtype=np.float32)
    angular = np.zeros((batch, body_count, 3), dtype=np.float32)
    linear[..., 1] = np.arange(body_count, dtype=np.float32)
    angular[..., 2] = np.arange(body_count, dtype=np.float32) + 1.0
    anchor_pos = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    anchor_quat = np.zeros((batch, 4), dtype=np.float32)
    anchor_quat[:, 0] = 1.0

    result = compute_g1_amp_observation(
        body_pos_w=positions,
        body_quat_w=quaternions,
        body_lin_vel_w=linear,
        body_ang_vel_w=angular,
        anchor_pos_w=anchor_pos,
        anchor_quat_w=anchor_quat,
    )

    assert result.shape == (batch, AMP_OBSERVATION_DIM)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0, : body_count * 3], (positions[0] - anchor_pos[0]).ravel())
    velocity_start = body_count * (3 + 6)
    np.testing.assert_allclose(
        result[0, velocity_start : velocity_start + body_count * 3], linear[0].ravel()
    )


def test_self_collision_cost_is_symmetric_strict_and_oldest_to_newest() -> None:
    history = np.zeros((2, 4, 6), dtype=np.float32)
    history[0, 0, 0] = 10.0
    history[0, 1, 1] = 10.01
    history[0, 3, 5] = -11.0
    history[1, 2, 2] = 12.0
    np.testing.assert_array_equal(
        compute_self_collision_cost(history, force_threshold=10.0),
        np.array([2.0, 1.0], dtype=np.float32),
    )


def test_self_collision_term_binds_public_history_once() -> None:
    history = np.zeros((2, 4, 3), dtype=np.float32)
    history[1, :, 0] = 11.0

    class View:
        def read(self) -> np.ndarray:
            return history

    calls: list[tuple[tuple[str, ...], int]] = []

    class Scene:
        def bind_sensor_history(self, names, *, history_length):
            calls.append((tuple(names), history_length))
            return View()

    env = SimpleNamespace(num_envs=2, scene=Scene())
    cfg = ManagerTermBaseCfg(
        func=SelfCollisionPenalty,
        params={
            "sensor_name": "self_collision",
            "history_length": 4,
            "force_threshold": 10.0,
        },
    )
    term = SelfCollisionPenalty(cfg, env)
    np.testing.assert_array_equal(term(env), np.array([0.0, 4.0], dtype=np.float32))
    assert calls == [(("self_collision",), 4)]


def test_amp_collision_sensor_is_task_scoped_and_symmetric() -> None:
    path = Path(__file__).parents[4] / "src/unilab/assets/robots/g1/amp_walk_task.xml"
    text = path.read_text(encoding="utf-8")
    assert 'name="self_collision"' in text
    assert 'subtree1="pelvis"' in text
    assert 'subtree2="pelvis"' in text
    assert 'nsample="4"' in text
