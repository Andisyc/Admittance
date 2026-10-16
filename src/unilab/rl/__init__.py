"""UniLab-owned RL algorithms, collectors and asynchronous training runtimes."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("unilab")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"
