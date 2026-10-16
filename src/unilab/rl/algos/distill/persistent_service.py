"""Worker-local single-teacher service for the persistent DAgger runner.

The service is the generic EnvFactory counterpart of ReGait's resident G1
worker. It retains the same atomic weight-version barrier and owns every
worker-local environment, policy and shared-memory attachment.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from unilab.rl.env_contract import EnvFactory, EnvProtocol
from unilab.rl.ipc import SharedWeightSync

from .async_runtime import DaggerCollectRequest, DaggerCollectResult
from .collector import DistillationCollectorSpec, collect_distillation_dataset_from_env
from .data import save_distillation_dataset


class SingleTeacherDaggerService:
    def __init__(
        self,
        *,
        env_factory: EnvFactory,
        student_factory: Callable[[], nn.Module],
        teacher_factory: Callable[[], nn.Module],
        collector_spec: DistillationCollectorSpec,
        num_envs: int,
        env_cfg_override: Mapping[str, Any] | None,
        weight_sync_name: str,
        weight_sync_lock: Any,
        weight_param_shapes: dict[str, torch.Size],
        collection_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._closed = False
        self.env: EnvProtocol | None = None
        self.weight_sync: SharedWeightSync | None = None
        self.spec = collector_spec
        self.collection_metadata = dict(collection_metadata or {})
        try:
            self.student = student_factory()
            self.teacher = teacher_factory()
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            state = self.student.state_dict()
            if set(state) != set(weight_param_shapes) or any(
                state[name].shape != shape for name, shape in weight_param_shapes.items()
            ):
                raise ValueError("collector student/shared weight shape mismatch")
            self.weight_sync = SharedWeightSync(
                weight_param_shapes,
                create=False,
                shm_name=weight_sync_name,
                lock=weight_sync_lock,
            )
            self.env = env_factory(num_envs, env_cfg_override)
            if self.env.num_envs != num_envs:
                raise ValueError("collector EnvFactory returned an unexpected environment count")
            self.env.init_state()
        except BaseException:
            self.close()
            raise

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        if self._closed or self.env is None or self.weight_sync is None:
            raise RuntimeError("cannot collect from a closed DAgger service")
        if request.scenario != self.spec.workflow_scenario:
            raise ValueError("DAgger request scenario does not match collector spec")
        observed = self.weight_sync.read_weights_into(self.student.state_dict())
        if observed != request.expected_weight_version:
            raise ValueError("persistent DAgger collector weight version mismatch")
        dataset = collect_distillation_dataset_from_env(
            self.env,
            self.student,
            self.teacher,
            self.spec,
            metadata={
                **self.collection_metadata,
                "request_id": request.request_id,
                "iteration": request.iteration,
                "checkpoint_path": request.checkpoint_path,
                "weight_version": observed,
            },
        )
        save_distillation_dataset(request.output_path, dataset)
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=observed,
            num_samples=dataset.num_samples,
            worker_pid=os.getpid(),
            metadata=dataset.metadata,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.env is not None:
                self.env.close()
        finally:
            if self.weight_sync is not None:
                self.weight_sync.close()
