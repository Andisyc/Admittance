"""Read-only local contracts; optimization and rollout proofs are remote/slow."""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.rl.algos.common.actor_factory import build_actor
from unilab.rl.algos.common.normalization import EmpiricalNormalization
from unilab.rl.algos.distill.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
    validate_dagger_collect_result,
)
from unilab.rl.algos.distill.checkpoint import (
    load_distillation_checkpoint,
    restore_distillation_checkpoint,
    save_distillation_checkpoint,
)
from unilab.rl.algos.distill.collector import DistillationCollectorSpec
from unilab.rl.algos.distill.data import (
    DistillationTensorDataset,
    load_distillation_dataset,
    save_distillation_dataset,
)
from unilab.rl.algos.distill.models import MLPStudentPolicy
from unilab.rl.algos.distill.persistent_runtime import PersistentDistillationRuntime
from unilab.rl.algos.distill.persistent_service import SingleTeacherDaggerService
from unilab.rl.algos.distill.teacher import SACTeacherSpec, load_sac_teacher_policy
from unilab.rl.algos.distill.trainer import BehaviorDistillationTrainer, DistillationBatch
from unilab.rl.algos.distill.workflow import (
    SingleScenarioDaggerConfig,
    run_single_scenario_dagger,
)


def _student() -> MLPStudentPolicy:
    return MLPStudentPolicy(obs_dim=3, action_dim=2, hidden_dims=(8,), activation="elu")


def _load_student(path: Path) -> MLPStudentPolicy:
    student = _student()
    load_distillation_checkpoint(student, path)
    return student


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in module.state_dict().items()}


def _assert_state(module: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    for key, value in before.items():
        torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("squash", [False, True])
def test_student_forward_matches_saved_weights(tmp_path: Path, squash: bool) -> None:
    student = MLPStudentPolicy(obs_dim=3, action_dim=2, hidden_dims=(8,), squash_action=squash)
    obs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    with torch.no_grad():
        expected = student(obs)
    target = tmp_path / "student.pt"
    save_distillation_checkpoint(target, student=student, agent_steps=17)
    restored = MLPStudentPolicy(obs_dim=3, action_dim=2, hidden_dims=(8,), squash_action=squash)
    info = load_distillation_checkpoint(restored, target)
    assert info["agent_steps"] == 17
    assert info["schema_version"] == 1
    with torch.no_grad():
        torch.testing.assert_close(restored(obs), expected, rtol=0, atol=0)
    if squash:
        assert bool((expected.abs() <= 1).all())
    assert not list(tmp_path.glob(".*.tmp.*"))


@pytest.mark.parametrize("corruption", ["shape", "nonfinite", "keys", "schema"])
def test_checkpoint_preflight_rejects_without_partial_student_mutation(corruption: str) -> None:
    student = _student()
    before = _state(student)
    state = {name: tensor + 1 for name, tensor in before.items()}
    payload = {"student_state_dict": state}
    last = next(reversed(state))
    if corruption == "shape":
        state[last] = torch.zeros(99)
    elif corruption == "nonfinite":
        state[last].fill_(float("nan"))
    elif corruption == "keys":
        del state[last]
    else:
        payload["schema"] = "other"
    with pytest.raises(ValueError):
        restore_distillation_checkpoint(student, payload)
    _assert_state(student, before)


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "student.pt"
    path.write_bytes(b"previous")

    def interrupted(*_args, **_kwargs):
        raise OSError("interrupted save")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        save_distillation_checkpoint(path, student=_student(), agent_steps=0)
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*.tmp.*"))


def _adam_snapshot(student: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Build serialized Adam moments directly; never invoke an optimizer step."""
    state = copy.deepcopy(optimizer.state_dict())
    parameters = list(student.parameters())
    identifiers = state["param_groups"][0]["params"]
    for ident, parameter in zip(identifiers, parameters, strict=True):
        state["state"][ident] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.full_like(parameter, 0.01),
            "exp_avg_sq": torch.full_like(parameter, 0.001),
        }
    return state


@pytest.mark.parametrize(
    "corruption",
    [
        "state_mapping",
        "unknown_id",
        "duplicate_id",
        "missing_group",
        "nonfinite_lr",
        "missing_moment",
        "moment_shape",
        "moment_nonfinite",
        "negative_variance",
        "step_shape",
        "step_nonfinite",
        "step_negative",
        "parameter_state_mapping",
    ],
)
def test_bad_optimizer_checkpoint_rejects_before_any_owner_mutates(corruption: str) -> None:
    student = _student()
    normalizer = EmpiricalNormalization(shape=3, device="cpu")
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    before_student = _state(student)
    before_normalizer = _state(normalizer)
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    opt_state = _adam_snapshot(student, optimizer)
    ident = opt_state["param_groups"][0]["params"][0]
    if corruption == "state_mapping":
        opt_state["state"] = []
    elif corruption == "unknown_id":
        opt_state["state"][999] = opt_state["state"].pop(ident)
    elif corruption == "duplicate_id":
        opt_state["param_groups"][0]["params"][1] = ident
    elif corruption == "missing_group":
        opt_state["param_groups"] = []
    elif corruption == "nonfinite_lr":
        opt_state["param_groups"][0]["lr"] = float("nan")
    elif corruption == "missing_moment":
        del opt_state["state"][ident]["exp_avg_sq"]
    elif corruption == "moment_shape":
        opt_state["state"][ident]["exp_avg"] = torch.ones(99)
    elif corruption == "moment_nonfinite":
        opt_state["state"][ident]["exp_avg"].fill_(float("inf"))
    elif corruption == "negative_variance":
        opt_state["state"][ident]["exp_avg_sq"].fill_(-1)
    elif corruption == "parameter_state_mapping":
        opt_state["state"][ident] = []
    else:
        opt_state["state"][ident]["step"] = {
            "step_shape": torch.ones(2),
            "step_nonfinite": torch.tensor(float("nan")),
            "step_negative": torch.tensor(-1.0),
        }[corruption]
    changed_normalizer = _state(normalizer)
    changed_normalizer["_mean"].fill_(9)
    payload = {
        "student_state_dict": {name: tensor + 1 for name, tensor in before_student.items()},
        "obs_normalizer": changed_normalizer,
        "optimizer_state_dict": opt_state,
    }
    with pytest.raises(ValueError, match="optimizer"):
        restore_distillation_checkpoint(
            student, payload, optimizer=optimizer, obs_normalizer=normalizer
        )
    _assert_state(student, before_student)
    _assert_state(normalizer, before_normalizer)
    assert optimizer.state_dict() == before_optimizer


def test_valid_adam_moments_load_without_executing_optimizer_step() -> None:
    student = _student()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    snapshot = _adam_snapshot(student, optimizer)
    restore_distillation_checkpoint(
        student,
        {"student_state_dict": _state(student), "optimizer_state_dict": snapshot},
        optimizer=optimizer,
    )
    actual = optimizer.state_dict()
    assert actual["param_groups"] == snapshot["param_groups"]
    for ident, buffers in snapshot["state"].items():
        for name, value in buffers.items():
            torch.testing.assert_close(actual["state"][ident][name], value, rtol=0, atol=0)


@pytest.mark.parametrize("normalization", [False, True])
def test_sac_teacher_loads_real_actor_and_freezes_normalizer(
    tmp_path: Path, normalization: bool
) -> None:
    actor = build_actor("sac", 3, 2, 8, False, "cpu")
    norm = EmpiricalNormalization(shape=3, device="cpu")
    path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict(), "obs_normalizer": norm.state_dict()}, path)
    teacher = load_sac_teacher_policy(
        SACTeacherSpec(
            path,
            obs_dim=3,
            action_dim=2,
            actor_hidden_dim=8,
            use_layer_norm=False,
            obs_normalization=normalization,
        )
    )
    before = _state(teacher)
    obs = torch.ones(4, 3, requires_grad=True)
    actual = teacher(obs)
    expected = actor.explore(norm(obs, update=False) if normalization else obs, deterministic=True)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (4, 2) and not actual.requires_grad
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    _assert_state(teacher, before)


def test_sac_teacher_rejects_missing_required_normalization(tmp_path: Path) -> None:
    actor = build_actor("sac", 3, 2, 8, False, "cpu")
    path = tmp_path / "teacher.pt"
    torch.save({"actor": actor.state_dict()}, path)
    with pytest.raises(ValueError, match="requires checkpoint obs_normalizer"):
        load_sac_teacher_policy(SACTeacherSpec(path, 3, 2, 8, False, True))
    with pytest.raises(ValueError, match="shape mismatch"):
        load_sac_teacher_policy(SACTeacherSpec(path, 4, 2, 8, False))


def test_aligned_label_serialization_and_batch(tmp_path: Path) -> None:
    obs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    labels = obs[:, :2] * 2
    dataset = DistillationTensorDataset(obs, obs + 1, labels, {"workflow_scenario": "toy"})
    path = tmp_path / "labels.pt"
    save_distillation_dataset(path, dataset)
    restored = load_distillation_dataset(path)
    batch = restored.batch(torch.tensor([3, 0]), device="cpu")
    torch.testing.assert_close(batch.student_obs, obs[[3, 0]])
    torch.testing.assert_close(batch.teacher_actions, labels[[3, 0]])
    with pytest.raises(ValueError, match="alignment"):
        DistillationTensorDataset(obs, obs, labels[:2])


def test_request_identity_and_weight_barrier_reject_stale_results() -> None:
    request = DaggerCollectRequest("r1", "toy", 1, "/checkpoint", "/dataset", 4)
    result = DaggerCollectResult("r1", "toy", 1, "/checkpoint", "/dataset", 4, 4, 8, 42)
    validate_dagger_collect_result(request, result)
    with pytest.raises(ValueError, match="weight version mismatch"):
        validate_dagger_collect_result(
            request, dataclasses.replace(result, observed_weight_version=3)
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_dagger_collect_result(request, dataclasses.replace(result, iteration=2))


def test_runtime_publishes_checkpoint_without_starting_rollout(tmp_path: Path) -> None:
    first = tmp_path / "student.pt"
    save_distillation_checkpoint(first, student=_student(), agent_steps=0)
    runtime = PersistentDistillationRuntime(
        student_loader=_load_student, worker_factory=SingleTeacherDaggerService
    )
    try:
        assert runtime.activate_checkpoint(first) == 1
        assert runtime._runner is not None
        assert runtime._runner._collector_process is None
        assert runtime.activate_checkpoint(first) == 2
        assert runtime._runner._collector_process is None
    finally:
        runtime.close()
        runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.activate_checkpoint(first)


def test_distill_owner_has_no_external_or_task_imports() -> None:
    owner = Path(__file__).resolve().parents[2] / "src/unilab/rl/algos/distill"
    for source in owner.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(
                name.startswith(("uni_rl", "unilab.tasks", "unilab.base", "unisim"))
                for name in modules
            ), source


@pytest.mark.parametrize("loss_type", ["mse", "huber", "l1"])
def test_remote_only_optimizer_updates_student_and_never_teacher(loss_type: str) -> None:
    student, teacher = _student(), _student()
    teacher_state = _state(teacher)
    student_state = _state(student)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    trainer = BehaviorDistillationTrainer(
        student=student, teacher=teacher, optimizer=optimizer, loss_type=loss_type
    )
    stats = trainer.update(DistillationBatch(torch.ones(4, 3), torch.ones(4, 3)))
    assert stats.update_count == 1 and np.isfinite(stats.loss)
    assert not stats.teacher_action_requires_grad
    assert any(
        not torch.equal(value, student_state[key]) for key, value in student.state_dict().items()
    )
    _assert_state(teacher, teacher_state)
    assert all(p.grad is None for p in teacher.parameters())


class _ToyEnv:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.state = None

    def init_state(self):
        self.state = SimpleNamespace(
            obs={"obs": np.ones((self.num_envs, 3), dtype=np.float32)},
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            final_observation=None,
        )
        return self.state

    def step(self, actions):
        assert actions.shape == (self.num_envs, 2) and np.isfinite(actions).all()
        self.state.obs["obs"] += 0.01
        return self.state

    def close(self):
        pass


def _env_factory(num_envs, _overrides):
    return _ToyEnv(num_envs)


def test_remote_only_spawn_collect_aggregate_update_and_reload(tmp_path: Path) -> None:
    initial = tmp_path / "initial.pt"
    student = _student()
    save_distillation_checkpoint(initial, student=student, agent_steps=0)
    trainer = BehaviorDistillationTrainer(
        student=student, teacher=None, optimizer=torch.optim.Adam(student.parameters(), lr=1e-3)
    )
    runtime = PersistentDistillationRuntime(
        student_loader=_load_student,
        worker_factory=SingleTeacherDaggerService,
        worker_kwargs={
            "env_factory": _env_factory,
            "student_factory": _student,
            "teacher_factory": _student,
            "collector_spec": DistillationCollectorSpec(8, 3, 3, 2, workflow_scenario="toy"),
            "num_envs": 2,
            "env_cfg_override": {},
        },
        request_timeout_seconds=30,
    )
    try:
        result = run_single_scenario_dagger(
            runtime,
            trainer,
            initial_checkpoint=initial,
            output_dir=tmp_path / "run",
            config=SingleScenarioDaggerConfig("toy", 2, 8, 4, 1, 3, 3, 2),
        )
        assert result.total_collected_samples == 16 and result.learner_updates == 2
        assert load_distillation_dataset(tmp_path / "run/aggregate_0002.pt").num_samples == 16
        reloaded = _load_student(result.final_checkpoint_path)
        _assert_state(reloaded, _state(trainer.student))
    finally:
        runtime.close()
