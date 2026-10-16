"""The optional history facade delegates real providers and fails closed."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.entity import EntityScene
from unilab.base.sensor_history import SensorHistoryBackend, SensorHistoryView


class _HistoryView:
    names = ("contact",)
    history_length = 4

    def __init__(self) -> None:
        self.read_calls = 0
        self.samples = np.arange(24, dtype=np.float32).reshape(2, 4, 3)

    def read(self) -> np.ndarray:
        self.read_calls += 1
        return self.samples


class _HistoryProvider:
    backend_type = "history-test-provider"

    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[str], int]] = []
        self.view = _HistoryView()

    def bind_sensor_history(
        self, names: Sequence[str], *, history_length: int
    ) -> SensorHistoryView:
        self.calls.append((names, history_length))
        return self.view


def test_history_facade_delegates_without_reading_or_replacing_provider_view() -> None:
    backend = _HistoryProvider()
    scene = SimpleNamespace(_backend=backend)
    names = ["contact"]

    view = EntityScene.bind_sensor_history(scene, names, history_length=4)

    assert isinstance(backend, SensorHistoryBackend)
    assert isinstance(view, SensorHistoryView)
    assert view is backend.view
    assert backend.calls == [(names, 4)]
    assert backend.calls[0][0] is names
    assert backend.view.read_calls == 0
    assert view.read() is backend.view.samples


def test_history_facade_rejects_unsupported_backend_without_sensor_fallback() -> None:
    def forbidden_sensor_read(*args: object) -> np.ndarray:
        pytest.fail("Current sensor values cannot substitute for real history")

    backend = SimpleNamespace(backend_type="mujoco", bind_sensor_data=forbidden_sensor_read)
    scene = SimpleNamespace(_backend=backend)

    assert not isinstance(backend, SensorHistoryBackend)
    with pytest.raises(NotImplementedError, match="'mujoco'.*SensorHistoryBackend"):
        EntityScene.bind_sensor_history(scene, ("contact",), history_length=4)


@pytest.mark.parametrize("error", [KeyError, TypeError, ValueError, NotImplementedError])
def test_history_facade_preserves_provider_failure_and_adds_backend_context(
    error: type[Exception],
) -> None:
    class FailingProvider:
        backend_type = "history-test-provider"

        def bind_sensor_history(self, names: Sequence[str], *, history_length: int):
            raise error("provider validation failed")

    scene = SimpleNamespace(_backend=FailingProvider())
    with pytest.raises(error, match="history-test-provider.*provider validation failed") as exc:
        EntityScene.bind_sensor_history(scene, ("contact",), history_length=4)
    assert isinstance(exc.value.__cause__, error)
