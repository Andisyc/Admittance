"""Training ownership checks without collectors, optimization or simulation."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from unilab.rl.algos.appo.runtime import APPORuntime, resolve_appo_runtime


def _play(**kwargs):
    return None


def test_core_imports_do_not_fall_back_to_external_training_package():
    source = """
import importlib.abc
import inspect
import sys
from pathlib import Path
class RejectExternalTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'uni_rl' or fullname.startswith('uni_rl.'):
            raise AssertionError('External training package accessed: ' + fullname)
sys.meta_path.insert(0, RejectExternalTraining())
import unilab.rl.env_contract as contract
from unilab.rl.ipc.rollout_ring_buffer import RolloutRingBuffer
from unilab.rl.algos.appo.runtime import resolve_appo_runtime
runtime = resolve_appo_runtime({}, default_play_fn=lambda **kw: None)
for value in (contract, RolloutRingBuffer, runtime.runner_cls):
    assert '/src/unilab/rl/' in Path(inspect.getfile(value)).as_posix()
assert runtime.runner_kwargs == {}
assert not any(name == 'uni_rl' or name.startswith('uni_rl.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_runtime_kwargs_are_owned_per_instance():
    first = APPORuntime(runner_cls=object, play_fn=_play)
    second = APPORuntime(runner_cls=object, play_fn=_play)
    first.runner_kwargs["expert"] = "first"
    assert second.runner_kwargs == {}


def test_runtime_resolver_preserves_and_copies_explicit_runner_kwargs(monkeypatch):
    supplied = {"expert": "clip-source"}
    monkeypatch.setattr(
        "rsl_rl.utils.resolve_callable",
        lambda _: lambda cfg: SimpleNamespace(runner_cls=object, runner_kwargs=supplied),
    )
    runtime = resolve_appo_runtime({"runtime_resolver": "example:resolve"}, default_play_fn=_play)
    assert runtime.runner_cls is object
    assert runtime.play_fn is _play
    assert runtime.runner_kwargs == supplied
    supplied["other"] = 2
    assert "other" not in runtime.runner_kwargs


def test_invalid_runtime_kwargs_are_rejected(monkeypatch):
    monkeypatch.setattr(
        "rsl_rl.utils.resolve_callable",
        lambda _: lambda cfg: SimpleNamespace(runner_cls=object, runner_kwargs="not a mapping"),
    )
    with pytest.raises(TypeError, match="runner_kwargs must be a mapping"):
        resolve_appo_runtime({"runtime_resolver": "example:resolve"}, default_play_fn=_play)


@pytest.mark.parametrize(
    "name", ["pyproject.toml", "pyproject.rocm.toml", "uv.lock", "uv.rocm.lock"]
)
def test_project_does_not_declare_external_training_distribution(name):
    project = Path(__file__).resolve().parents[2] / name
    assert '"unilab-rl' not in project.read_text()
