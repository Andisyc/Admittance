"""Optional sensor-history contracts owned by UniLab.

These protocols describe an additional backend capability; they do not add an
implementation to the published UniSim backends. A provider must supply real
physics-step samples in oldest-to-newest order, without exposing native handles
to environment or manager code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SensorHistoryView(Protocol):
    """Backend-owned fixed-depth history retained by a manager term."""

    @property
    def names(self) -> tuple[str, ...]: ...

    @property
    def history_length(self) -> int: ...

    def read(self) -> np.ndarray:
        """Return real samples shaped ``(num_envs, history_length, width)``."""
        ...


@runtime_checkable
class SensorHistoryBackend(Protocol):
    """Optional public capability implemented only by supporting backends."""

    def bind_sensor_history(
        self, names: Sequence[str], *, history_length: int
    ) -> SensorHistoryView:
        """Validate and bind a history view on the backend's cold path."""
        ...
