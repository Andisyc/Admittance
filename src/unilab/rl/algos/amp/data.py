"""Validated expert-transition value object, independent of asset loading."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.rl.algos.amp.spec import AMP_OBSERVATION_DIM


@dataclass(frozen=True)
class AMPExpertTransitions:
    current: np.ndarray
    next: np.ndarray
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        arrays = []
        for name in ("current", "next"):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.ndim != 2
                or value.shape[1] != AMP_OBSERVATION_DIM
                or value.shape[0] == 0
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"AMP expert {name} must be finite nonempty float32 [K, 195]")
            arrays.append(np.array(value, copy=True, order="C"))
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("AMP expert current/next shapes differ")
        if not isinstance(self.provenance, dict) or not self.provenance:
            raise ValueError("AMP expert provenance must be a nonempty mapping")
        for name, value in zip(("current", "next"), arrays):
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))
