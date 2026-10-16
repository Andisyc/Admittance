"""Shared rollout IPC ring buffer for APPO / async PPO."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")

_FIELD_SHAPES = {
    "obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, od),
    "critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, cd),
    "actions": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, ad),
    "log_probs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "rewards": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "dones": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "truncated": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "last_obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, od),
    "last_critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, cd),
}


class RolloutRingBuffer:
    """Bounded single-producer/single-consumer rollout transport.

    A free slot belongs to its producer until ``signal_write_done`` publishes
    the complete batch. The consumer borrows the published slot through the
    entire copy/staging operation and releases it only with ``advance_read``.
    Published slots are never overwritten: full rings apply backpressure via
    ``wait_for_space``. Direct ``write_buffer`` access while full fails instead
    of returning a view that can corrupt an outstanding consumer lease.

    Views must not be retained or modified after their owner publishes/releases
    the slot. Exactly one producer and one consumer are required; this is not a
    multi-writer or multi-reader queue.
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        critic_dim: int = 0,
        num_slots: int = 4,
        create: bool = True,
        shm_name_prefix: Dict[str, str] | None = None,
        extra_observation_dims: dict[str, int] | None = None,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_dim = critic_dim
        self.num_slots = num_slots
        if isinstance(num_slots, bool) or not isinstance(num_slots, int) or num_slots < 1:
            raise ValueError("rollout ring num_slots must be a positive integer")
        extra_observation_dims = dict(extra_observation_dims or {})
        for key, width in extra_observation_dims.items():
            if key in _FIELD_SHAPES or not isinstance(width, int) or width < 1:
                raise ValueError(f"invalid extra rollout observation {key!r}: {width!r}")

        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}

        fields_to_allocate = {k: v for k, v in _FIELD_SHAPES.items()}
        if critic_dim == 0:
            fields_to_allocate.pop("critic", None)
            fields_to_allocate.pop("last_critic", None)

        field_shapes = {
            key: shape_fn(num_slots, num_envs, num_steps, obs_dim, action_dim, critic_dim)
            for key, shape_fn in fields_to_allocate.items()
        }
        field_shapes.update(
            {
                key: (num_slots, num_envs, num_steps, width)
                for key, width in extra_observation_dims.items()
            }
        )

        try:
            for field, shape in field_shapes.items():
                nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize

                if create:
                    shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
                else:
                    assert shm_name_prefix is not None, "shm_name_prefix required when create=False"
                    shm = shared_memory.SharedMemory(name=shm_name_prefix[field], create=False)

                self._shm_blocks[field] = shm
                self._arrays[field] = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

        except BaseException:
            if create:
                self.cleanup()
            else:
                self.close()
            raise

        if create:
            self._write_ptr = _SPAWN_CTX.Value("l", 0)
            self._read_ptr = _SPAWN_CTX.Value("l", 0)

    @property
    def name(self) -> Dict[str, str]:
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    @property
    def slot_shapes(self) -> Dict[str, tuple[int, ...]]:
        return {field: tuple(arr.shape[1:]) for field, arr in self._arrays.items()}

    def attach_sync_primitives(self, write_ptr, read_ptr) -> None:
        self._write_ptr = write_ptr
        self._read_ptr = read_ptr

    @property
    def write_slot(self) -> int:
        if self.available() >= self.num_slots:
            raise BufferError("rollout ring is full; wait for consumer release before writing")
        return int(self._write_ptr.value) % self.num_slots

    @property
    def write_buffer(self) -> Dict[str, np.ndarray]:
        s = self.write_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def signal_write_done(self) -> None:
        if self.available() >= self.num_slots:
            raise BufferError("cannot publish a rollout into a full ring")
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    def available(self) -> int:
        return int(self._write_ptr.value) - int(self._read_ptr.value)

    def wait_for_space(self, timeout: float = 0.5, *, stop_event=None) -> bool:
        """Wait a bounded interval for a free slot, cancellable during shutdown."""
        import time

        deadline = time.monotonic() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            if self.available() < self.num_slots:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    @property
    def read_slot(self) -> int:
        if self.available() == 0:
            raise BufferError("rollout ring is empty; no published slot to borrow")
        return int(self._read_ptr.value) % self.num_slots

    def read_numpy_views(self) -> dict[str, np.ndarray]:
        """Return shared-memory views for the current read slot.

        The returned arrays are borrowed views. Consumers must copy them into
        owned storage before calling advance_read(). Backpressure protects every
        field in this slot from producer reuse until that release.
        """
        s = self.read_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def copy_read_slot_to_torch(self, destination: dict) -> None:
        import torch

        s = self.read_slot
        for field, arr in self._arrays.items():
            if field not in destination:
                raise KeyError(f"missing destination tensor for rollout field {field!r}")
            dst = destination[field]
            src_view = arr[s]
            if tuple(dst.shape) != tuple(src_view.shape):
                raise ValueError(
                    f"destination shape mismatch for {field!r}: "
                    f"expected {tuple(src_view.shape)}, got {tuple(dst.shape)}"
                )
            if dst.dtype != torch.float32:
                raise TypeError(f"destination tensor for {field!r} must be torch.float32")
            dst.copy_(torch.from_numpy(src_view), non_blocking=False)

    def read_torch(self, device: str) -> dict:
        import torch

        result = {
            field: torch.empty(tuple(arr.shape[1:]), dtype=torch.float32, device=device)
            for field, arr in self._arrays.items()
        }
        self.copy_read_slot_to_torch(result)
        return result

    def advance_read(self) -> None:
        if self.available() == 0:
            raise BufferError("cannot release an empty rollout ring")
        with self._read_ptr.get_lock():
            self._read_ptr.value += 1

    def cleanup(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def close(self) -> None:
        for shm in self._shm_blocks.values():
            try:
                shm.close()
            except Exception:
                pass
