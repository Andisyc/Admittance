"""Current goal terms for the opt-in 524-D MUSE direct-RL observation.

The remaining terms and history are composed by ObservationManager, not by
the network or a second history store. See ADR-0009 for the frozen order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from unilab.utils.rotation import np_quat_apply_inverse

from .manager_terms import _command

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


def delta_command(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    command = _command(env, command_name)
    return np.concatenate(
        (command.joint_pos - command.robot_joint_pos, command.joint_vel - command.robot_joint_vel),
        axis=-1,
    )


def reference_anchor_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> np.ndarray:
    """Reference velocity in the REFERENCE anchor frame, matching donor code.

    The donor's function prose says robot base frame; its actual computation
    uses the reference quaternion. Preserve the computation explicitly.
    """
    command = _command(env, command_name)
    return np_quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)


def visible_goal_mask(env: ManagerBasedRlEnv) -> np.ndarray:
    """All goals visible in the initial direct tracking profile: raw zero bit."""
    return np.zeros((env.num_envs, 1), dtype=np.float32)
