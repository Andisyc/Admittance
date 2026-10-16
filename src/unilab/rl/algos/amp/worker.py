"""Collector AMP payload writer; no discriminator or mutable training state."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.rl.algos.amp.transition import resolve_amp_transition_next, validate_amp_observation


def write_amp_rollout_payload(
    *,
    write_buffer: dict[str, np.ndarray],
    step: int,
    current_observation: dict[str, Any],
    state: Any,
) -> None:
    if "amp" not in current_observation or "amp" not in state.obs:
        raise ValueError("AMP collector requires the 'amp' observation group")
    count = write_buffer["amp_state"].shape[0]
    current = validate_amp_observation(current_observation["amp"], count=count, name="current AMP")
    next_state, _ = resolve_amp_transition_next(
        state.obs["amp"],
        done=state.terminated | state.truncated,
        final_observation=state.final_observation,
        info=state.info,
    )
    write_buffer["amp_state"][:, step, :] = current
    write_buffer["amp_next_state"][:, step, :] = next_state
