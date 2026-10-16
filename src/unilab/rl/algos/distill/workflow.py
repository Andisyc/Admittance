"""Single-scenario DAgger barriers adapted from ReGait's workflow owner.

Each outer iteration publishes a frozen student, collects teacher labels on
that student's state distribution, aggregates all labelled history, updates
only the learner student, then atomically saves the next published checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from .async_runtime import DaggerCollectRequest, DaggerCollectResult, validate_dagger_collect_result
from .checkpoint import load_distillation_checkpoint, save_distillation_checkpoint
from .data import DistillationTensorDataset, load_distillation_dataset, save_distillation_dataset
from .trainer import BehaviorDistillationTrainer


class DaggerRuntime(Protocol):
    def activate_checkpoint(self, checkpoint_path: Path) -> int: ...

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult: ...


@dataclass(frozen=True)
class SingleScenarioDaggerConfig:
    scenario: str
    iterations: int
    num_samples: int
    batch_size: int
    max_updates: int
    student_obs_dim: int
    teacher_obs_dim: int
    action_dim: int
    seed: int = 0

    def __post_init__(self) -> None:
        fields = (
            self.iterations,
            self.num_samples,
            self.batch_size,
            self.max_updates,
            self.student_obs_dim,
            self.teacher_obs_dim,
            self.action_dim,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in fields
        ):
            raise ValueError("DAgger counts and dimensions must be positive integers")
        if (
            not self.scenario
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("DAgger scenario must be nonempty and seed a nonnegative integer")


@dataclass(frozen=True)
class SingleScenarioDaggerRunResult:
    final_checkpoint_path: Path
    dataset_paths: tuple[Path, ...]
    checkpoint_paths: tuple[Path, ...]
    total_collected_samples: int
    learner_updates: int


def _validate_collected_dataset(
    dataset: DistillationTensorDataset,
    request: DaggerCollectRequest,
    config: SingleScenarioDaggerConfig,
) -> None:
    expected = (
        (config.num_samples, config.student_obs_dim),
        (config.num_samples, config.teacher_obs_dim),
        (config.num_samples, config.action_dim),
    )
    actual = (dataset.student_obs.shape, dataset.teacher_obs.shape, dataset.teacher_actions.shape)
    if actual != expected:
        raise ValueError(f"DAgger collected supervision shape mismatch: {actual} != {expected}")
    identity = {
        "request_id": request.request_id,
        "workflow_scenario": request.scenario,
        "iteration": request.iteration,
        "checkpoint_path": request.checkpoint_path,
        "weight_version": request.expected_weight_version,
    }
    if any(dataset.metadata.get(name) != value for name, value in identity.items()):
        raise ValueError("DAgger dataset metadata does not match collection request")


def run_single_scenario_dagger(
    runtime: DaggerRuntime,
    trainer: BehaviorDistillationTrainer,
    *,
    initial_checkpoint: str | Path,
    output_dir: str | Path,
    config: SingleScenarioDaggerConfig,
    teacher_metadata: Mapping[str, Any] | None = None,
    distill_runtime_cfg: Mapping[str, Any] | None = None,
) -> SingleScenarioDaggerRunResult:
    checkpoint = Path(initial_checkpoint).resolve()
    # The learner and collector must start from identical weights, including when
    # the caller supplies a saved initial student rather than a fresh model.
    snapshot = load_distillation_checkpoint(trainer.student, checkpoint)
    agent_steps = int(snapshot.get("agent_steps", 0))
    if agent_steps < 0:
        raise ValueError("initial checkpoint agent_steps must be nonnegative")
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    device = next(trainer.student.parameters()).device
    datasets: list[DistillationTensorDataset] = []
    dataset_paths: list[Path] = []
    checkpoint_paths: list[Path] = []
    start_updates = trainer.update_count
    for iteration in range(1, config.iterations + 1):
        version = runtime.activate_checkpoint(checkpoint)
        output = target / f"dataset_{iteration:04d}.pt"
        request = DaggerCollectRequest(
            request_id=f"{config.scenario}:{iteration}:{version}",
            scenario=config.scenario,
            iteration=iteration,
            checkpoint_path=str(checkpoint),
            output_path=str(output),
            expected_weight_version=version,
        )
        result = runtime.collect(request)
        validate_dagger_collect_result(request, result)
        if result.num_samples != config.num_samples:
            raise ValueError("DAgger collector returned an unexpected number of samples")
        dataset = load_distillation_dataset(output)
        _validate_collected_dataset(dataset, request, config)
        expected_teacher = {
            f"teacher_{name}": value
            for name, value in dict(teacher_metadata or {}).items()
            if name in {"checkpoint_path", "checkpoint_sha256", "checkpoint_size_bytes"}
        }
        if any(dataset.metadata.get(key) != value for key, value in expected_teacher.items()):
            raise ValueError("DAgger teacher provenance differs from expected metadata")
        datasets.append(dataset)
        dataset_paths.append(output)
        aggregate = DistillationTensorDataset(
            student_obs=torch.cat([item.student_obs for item in datasets]),
            teacher_obs=torch.cat([item.teacher_obs for item in datasets]),
            teacher_actions=torch.cat([item.teacher_actions for item in datasets]),
            metadata={
                "workflow_scenario": config.scenario,
                "source_paths": [str(p) for p in dataset_paths],
            },
        )
        save_distillation_dataset(target / f"aggregate_{iteration:04d}.pt", aggregate)
        # Sample from every previous student distribution, not only the newest
        # rollout. This is dataset aggregation in the original DAgger workflow.
        for _ in range(config.max_updates):
            indices = torch.randint(
                aggregate.num_samples,
                (min(config.batch_size, aggregate.num_samples),),
                generator=generator,
            )
            trainer.update(aggregate.batch(indices, device=device))
        agent_steps += dataset.num_samples
        checkpoint = target / f"student_{iteration:04d}.pt"
        save_distillation_checkpoint(
            checkpoint,
            student=trainer.student,
            optimizer=trainer.optimizer,
            agent_steps=agent_steps,
            teacher_metadata=teacher_metadata,
            distill_runtime_cfg={
                **dict(distill_runtime_cfg or {}),
                "scenario": config.scenario,
                "iteration": iteration,
                "update_count": trainer.update_count,
                "aggregate_samples": aggregate.num_samples,
            },
        )
        checkpoint_paths.append(checkpoint)
    return SingleScenarioDaggerRunResult(
        final_checkpoint_path=checkpoint,
        dataset_paths=tuple(dataset_paths),
        checkpoint_paths=tuple(checkpoint_paths),
        total_collected_samples=sum(item.num_samples for item in datasets),
        learner_updates=trainer.update_count - start_updates,
    )
