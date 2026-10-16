"""True terminal AMP transitions, independent of the reset observation carry."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.rl.algos.amp.spec import AMP_OBSERVATION_DIM


def validate_amp_observation(value: Any, *, count: int, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (count, AMP_OBSERVATION_DIM)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite float32 [{count}, 195]")
    return value


def resolve_amp_transition_next(
    actor_next_amp: np.ndarray,
    *,
    done: np.ndarray | None = None,
    final_observation: dict[str, Any] | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(actor_next_amp, np.ndarray) or actor_next_amp.ndim != 2:
        raise ValueError("AMP next observation must have shape [N, 195]")
    count = actor_next_amp.shape[0]
    validate_amp_observation(actor_next_amp, count=count, name="AMP next observation")
    mask = np.zeros(count, dtype=bool) if done is None else np.asarray(done, dtype=bool).ravel()
    if mask.shape != (count,):
        raise ValueError("AMP terminal mask must have shape [N]")
    if not mask.any():
        return actor_next_amp, mask
    final = final_observation
    if final is None and isinstance(info, dict):
        final = info.get("final_observation")
    if not isinstance(final, dict) or "amp" not in final:
        raise ValueError("terminal AMP transition requires final_observation['amp']")
    terminal = validate_amp_observation(final["amp"], count=count, name="terminal AMP")
    result = actor_next_amp.copy()
    result[mask] = terminal[mask]
    return result, mask
