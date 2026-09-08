"""G1 AMP observation and symmetric self-collision terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from unilab.managers.manager_base import ManagerTermBase, ManagerTermBaseCfg
from unilab.managers.scene_entity_config import SceneEntityCfg
from unilab.utils.rotation import (
    np_matrix_first_two_cols_from_quat,
    np_quat_apply_batched,
    np_quat_conjugate_batched,
    np_subtract_anchor_frame_transforms,
)

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
AMP_ANCHOR_BODY_NAME = "torso_link"
AMP_BODY_INDICES = (0, 2, 4, 6, 8, 10, 12, 17, 19, 22, 24, 26, 29)
AMP_ANCHOR_BODY_INDEX = 15
AMP_FEATURES_PER_BODY = 15
AMP_OBSERVATION_DIM = len(AMP_BODY_NAMES) * AMP_FEATURES_PER_BODY


def _body_array(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def compute_g1_amp_observation(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
    anchor_pos_w: np.ndarray,
    anchor_quat_w: np.ndarray,
) -> np.ndarray:
    """Build the frozen AMP_mjlab-compatible 195-D G1 style state."""
    body_pos_w = np.asarray(body_pos_w)
    if body_pos_w.ndim != 3:
        raise ValueError(f"body_pos_w must be three-dimensional, got {body_pos_w.shape}")
    batch = body_pos_w.shape[0]
    body_shape = (batch, len(AMP_BODY_NAMES))
    body_pos_w = _body_array("body_pos_w", body_pos_w, (*body_shape, 3))
    body_quat_w = _body_array("body_quat_w", body_quat_w, (*body_shape, 4))
    body_lin_vel_w = _body_array("body_lin_vel_w", body_lin_vel_w, (*body_shape, 3))
    body_ang_vel_w = _body_array("body_ang_vel_w", body_ang_vel_w, (*body_shape, 3))
    anchor_pos_w = _body_array("anchor_pos_w", anchor_pos_w, (batch, 3))
    anchor_quat_w = _body_array("anchor_quat_w", anchor_quat_w, (batch, 4))

    relative_pos, relative_quat = np_subtract_anchor_frame_transforms(
        anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w
    )
    orientation_6d = np_matrix_first_two_cols_from_quat(relative_quat)
    inverse_body_quat = np_quat_conjugate_batched(body_quat_w)
    local_lin_vel = np_quat_apply_batched(inverse_body_quat, body_lin_vel_w)
    local_ang_vel = np_quat_apply_batched(inverse_body_quat, body_ang_vel_w)
    features = np.concatenate(
        (
            relative_pos.reshape(batch, -1),
            orientation_6d.reshape(batch, -1),
            local_lin_vel.reshape(batch, -1),
            local_ang_vel.reshape(batch, -1),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if features.shape != (batch, AMP_OBSERVATION_DIM):
        raise RuntimeError(f"invalid AMP observation shape {features.shape}")
    return np.ascontiguousarray(features)


def compute_g1_amp_observation_from_ordered(
    *,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
) -> np.ndarray:
    """Build AMP state from the donor's ordered full-body G1 arrays."""
    arrays = {
        "body_pos_w": (np.asarray(body_pos_w), 3),
        "body_quat_w": (np.asarray(body_quat_w), 4),
        "body_lin_vel_w": (np.asarray(body_lin_vel_w), 3),
        "body_ang_vel_w": (np.asarray(body_ang_vel_w), 3),
    }
    batch: int | None = None
    minimum_bodies = max(max(AMP_BODY_INDICES), AMP_ANCHOR_BODY_INDEX) + 1
    for name, (value, width) in arrays.items():
        if value.ndim != 3 or value.shape[-1] != width or value.shape[1] < minimum_bodies:
            raise ValueError(
                f"{name} must have shape (batch, at least {minimum_bodies}, {width}), "
                f"got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
        if batch is None:
            batch = value.shape[0]
        elif value.shape[0] != batch:
            raise ValueError(f"{name} batch size does not match body_pos_w")
    indices = np.asarray(AMP_BODY_INDICES, dtype=np.intp)
    return compute_g1_amp_observation(
        body_pos_w=arrays["body_pos_w"][0][:, indices],
        body_quat_w=arrays["body_quat_w"][0][:, indices],
        body_lin_vel_w=arrays["body_lin_vel_w"][0][:, indices],
        body_ang_vel_w=arrays["body_ang_vel_w"][0][:, indices],
        anchor_pos_w=arrays["body_pos_w"][0][:, AMP_ANCHOR_BODY_INDEX],
        anchor_quat_w=arrays["body_quat_w"][0][:, AMP_ANCHOR_BODY_INDEX],
    )


class G1AmpObservation(ManagerTermBase):
    """Read the declared G1 bodies through the entity facade once per phase."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset({"asset_cfg"})

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        asset_cfg = cfg.params.get("asset_cfg")
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError(f"{self.name} asset_cfg must be a SceneEntityCfg")
        expected_names = (*AMP_BODY_NAMES, AMP_ANCHOR_BODY_NAME)
        if tuple(asset_cfg.body_names or ()) != expected_names or not asset_cfg.preserve_order:
            raise ValueError(f"{self.name} requires preserve_order and body_names={expected_names}")
        self._asset_cfg = asset_cfg

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del params
        data = env.scene[self._asset_cfg.name].data
        ids = self._asset_cfg.body_ids
        pos = data.body_link_pos_w[:, ids]
        quat = data.body_link_quat_w[:, ids]
        lin_vel = data.body_link_lin_vel_w[:, ids]
        ang_vel = data.body_link_ang_vel_w[:, ids]
        body_count = len(AMP_BODY_NAMES)
        return compute_g1_amp_observation(
            body_pos_w=pos[:, :body_count],
            body_quat_w=quat[:, :body_count],
            body_lin_vel_w=lin_vel[:, :body_count],
            body_ang_vel_w=ang_vel[:, :body_count],
            anchor_pos_w=pos[:, body_count],
            anchor_quat_w=quat[:, body_count],
        )


def compute_self_collision_cost(force_history: np.ndarray, *, force_threshold: float) -> np.ndarray:
    """Count substeps where any symmetric self-contact slot exceeds the threshold."""
    history = np.asarray(force_history)
    if history.ndim != 3 or history.shape[-1] % 3 != 0:
        raise ValueError(
            "force_history must have shape (num_envs, history_length, 3 * num_slots), "
            f"got {history.shape}"
        )
    if not np.isfinite(history).all():
        raise ValueError("force_history contains NaN or Inf")
    threshold = float(force_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("force_threshold must be finite and non-negative")
    slot_forces = history.reshape(history.shape[0], history.shape[1], -1, 3)
    hit = np.any(np.linalg.norm(slot_forces, axis=-1) > threshold, axis=-1)
    return np.asarray(np.sum(hit, axis=-1), dtype=np.float32)


class SelfCollisionPenalty(ManagerTermBase):
    """AMP-only symmetric self-collision count backed by native sensor history."""

    _allowed_params: ClassVar[frozenset[str]] = frozenset(
        {"sensor_name", "history_length", "force_threshold"}
    )

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        unexpected = set(cfg.params) - self._allowed_params
        if unexpected:
            raise TypeError(f"{self.name} received unsupported parameters: {sorted(unexpected)}")
        sensor_name = cfg.params.get("sensor_name", "self_collision")
        history_length = cfg.params.get("history_length", 4)
        force_threshold = cfg.params.get("force_threshold", 10.0)
        if sensor_name != "self_collision":
            raise ValueError(f"{self.name} requires sensor_name='self_collision'")
        if isinstance(history_length, bool) or int(history_length) != 4:
            raise ValueError(f"{self.name} requires history_length=4")
        if float(force_threshold) != 10.0:
            raise ValueError(f"{self.name} requires force_threshold=10.0")
        self._force_threshold = float(force_threshold)
        self._history = env.scene.bind_sensor_history(
            (sensor_name,), history_length=int(history_length)
        )

    def __call__(self, env: ManagerBasedRlEnv, **params: Any) -> np.ndarray:
        del env, params
        return compute_self_collision_cost(
            self._history.read(), force_threshold=self._force_threshold
        )


__all__ = [
    "AMP_ANCHOR_BODY_NAME",
    "AMP_ANCHOR_BODY_INDEX",
    "AMP_BODY_INDICES",
    "AMP_BODY_NAMES",
    "AMP_OBSERVATION_DIM",
    "G1AmpObservation",
    "SelfCollisionPenalty",
    "compute_g1_amp_observation",
    "compute_g1_amp_observation_from_ordered",
    "compute_self_collision_cost",
]
