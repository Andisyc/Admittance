"""AMP value/forward/checkpoint contracts; no local optimizer step or rollout."""

from __future__ import annotations

import copy
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from unilab.rl.algos.amp.checkpoint import (
    extract_amp_actor_state,
    save_amp_checkpoint_atomic,
    validate_amp_checkpoint,
)
from unilab.rl.algos.amp.data import AMPExpertTransitions
from unilab.rl.algos.amp.learner import AMPAPPOLearner, AMPDiscriminator
from unilab.rl.algos.amp.runner import AMPAPPORunner
from unilab.rl.algos.amp.transition import resolve_amp_transition_next
from unilab.rl.algos.amp.worker import write_amp_rollout_payload
from unilab.rl.algos.appo.learner import APPOLearner
from unilab.rl.algos.appo.runner import APPORunner
from unilab.rl.algos.appo.staging import RolloutStagingPool
from unilab.rl.algos.appo.worker import snapshot_rollout_observations
from unilab.rl.ipc.rollout_ring_buffer import RolloutRingBuffer


class TinyModel(nn.Module):
    """Small forward-only stand-in exposing the APPO MLP parameter surface."""

    def __init__(self, width: int, output: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(width, 4), nn.ReLU(), nn.Linear(4, output))


@pytest.fixture
def learner():
    return AMPAPPOLearner(
        actor=TinyModel(3, 2),
        critic=TinyModel(3, 1),
        device="cpu",
        enable_compile=False,
        amp_expert_current=np.zeros((5, 195), dtype=np.float32),
        amp_expert_next=np.ones((5, 195), dtype=np.float32),
        amp_expert_provenance={"motions": [{"sha256": "test-only"}]},
        amp_method_fingerprint={"method": "test-only"},
        amp_playback_identity={"actor": "test-only"},
        amp_hidden_dims=[4],
        amp_replay_capacity=8,
        amp_discriminator_batch_size=2,
    )


def batch():
    return {
        "observations": torch.zeros(2, 2, 3),
        "actions": torch.zeros(2, 2, 2),
        "actions_log_prob": torch.zeros(2, 2),
        "rewards": torch.full((2, 2), 6.0),
        "dones": torch.zeros(2, 2),
        "last_obs": torch.zeros(2, 3),
        "amp_state": torch.zeros(2, 2, 195),
        "amp_next_state": torch.ones(2, 2, 195),
        "timeout_bootstrap": torch.full((2, 2, 1), 2.0),
    }


def test_expert_is_owned_and_rejects_invalid_values():
    current = np.zeros((3, 195), dtype=np.float32)
    expert = AMPExpertTransitions(current, current, {"source": "unit-test"})
    current[:] = 5
    assert not expert.current.any()
    assert not expert.current.flags.writeable
    with pytest.raises(ValueError):
        AMPExpertTransitions(current.astype(np.float64), current, {"source": "unit-test"})


@pytest.mark.parametrize("terminal", [False, True])
def test_transition_uses_true_terminal_without_modifying_reset_carry(terminal):
    carry = np.zeros((2, 195), dtype=np.float32)
    final = np.full_like(carry, 7)
    next_state, _ = resolve_amp_transition_next(
        carry, done=np.array([terminal, False]), final_observation={"amp": final}
    )
    assert np.all(next_state[0] == (7 if terminal else 0))
    assert not next_state[1].any()
    assert not carry.any()


@pytest.mark.parametrize("failure", ["missing", "shape", "dtype", "nan"])
def test_invalid_terminal_payload_does_not_write(failure):
    current = np.zeros((2, 195), dtype=np.float32)
    final = current.copy()
    if failure == "shape":
        final = final[:, :194]
    elif failure == "dtype":
        final = final.astype(np.float64)
    elif failure == "nan":
        final[0, 0] = np.nan
    write = {
        name: np.full((2, 1, 195), -1.0, np.float32) for name in ("amp_state", "amp_next_state")
    }
    state = SimpleNamespace(
        obs={"amp": current},
        terminated=np.array([True, False]),
        truncated=np.zeros(2, dtype=bool),
        info={},
        final_observation=None if failure == "missing" else {"amp": final},
    )
    with pytest.raises(ValueError):
        write_amp_rollout_payload(
            write_buffer=write, step=0, current_observation={"amp": current}, state=state
        )
    assert all(np.all(value == -1) for value in write.values())


@pytest.mark.parametrize("logit,expected", [(-1.0, 0.0), (1.0, 0.1), (3.0, 0.0)])
def test_discriminator_forward_reward_is_detached(logit, expected):
    discriminator = AMPDiscriminator(390, [4], reward_coef=0.1)
    with torch.no_grad():
        discriminator.head.weight.zero_()
        discriminator.head.bias.fill_(logit)
    reward, logits = discriminator.predict_reward(torch.zeros(2, 195), torch.zeros(2, 195))
    torch.testing.assert_close(reward, torch.full((2,), expected))
    torch.testing.assert_close(logits, torch.full((2,), logit))
    assert not reward.requires_grad


def test_discriminator_parameters_are_not_in_policy_optimizer(learner):
    policy_ids = {
        id(parameter) for group in learner.optimizer.param_groups for parameter in group["params"]
    }
    assert not policy_ids.intersection(
        id(parameter) for parameter in learner.discriminator.parameters()
    )


def test_style_blending_keeps_timeout_bootstrap_unscaled(learner, monkeypatch):
    monkeypatch.setattr(APPOLearner, "process_batch", lambda self, value: value)
    monkeypatch.setattr(
        learner,
        "predict_amp_reward",
        lambda current, next_state: (torch.ones(2, 2), torch.zeros(2, 2)),
    )
    data = batch()
    learner.process_batch(data)
    torch.testing.assert_close(data["rewards"], torch.full((2, 2), 5.25))
    torch.testing.assert_close(data["_amp_task_rewards"], torch.full((2, 2), 4.0))
    assert learner.amp_replay.size == 0
    assert learner.amp_normalizer.count.item() == 0
    assert learner.amp_order_trace == [("score", 0), ("vtrace", 0)]


@pytest.mark.parametrize("failure", ["nan", "empty", "wrong_width", "wrong_action", "mask"])
def test_invalid_batch_rejected_before_any_learner_mutation(learner, failure):
    data = batch()
    if failure == "nan":
        data["amp_state"][0, 0, 0] = float("nan")
    elif failure == "empty":
        data["rewards"] = torch.zeros(0, 2)
    elif failure == "wrong_width":
        data["amp_next_state"] = torch.zeros(2, 2, 194)
    elif failure == "wrong_action":
        data["actions"] = torch.zeros(2, 2, 3)
    elif failure == "mask":
        data["dones"][0, 0] = 0.5
    rng = learner.amp_generator.get_state().clone()
    normalizer = learner.amp_normalizer.mean.clone()
    with pytest.raises(ValueError):
        learner.process_batch(data)
    assert learner.amp_replay.size == 0
    assert learner.discriminator_version == 0
    assert learner.amp_order_trace == []
    torch.testing.assert_close(rng, learner.amp_generator.get_state())
    torch.testing.assert_close(normalizer, learner.amp_normalizer.mean)


def test_cold_checkpoint_is_complete_and_roundtrips_without_update(learner, tmp_path):
    state = learner.get_state_dict()
    validate_amp_checkpoint(
        state, expected_method_fingerprint=learner.amp_method_fingerprint, expected_state=state
    )
    destination = tmp_path / "model.pt"
    save_amp_checkpoint_atomic(state, destination)
    restored = torch.load(destination, weights_only=True)
    learner.load_state_dict(restored)
    assert restored["schema"] == "unilab.amp-appo/v1"
    assert learner._update_counter == 0
    assert learner.amp_replay.size == 0
    assert all(tensor.device.type == "cpu" for tensor in restored["actor"].values())


@pytest.mark.parametrize(
    "failure", ["legacy", "fingerprint", "target", "replay", "rng", "optimizer"]
)
def test_resume_preflight_happens_before_actor_mutation(learner, failure):
    state = learner.get_state_dict()
    reference = copy.deepcopy(learner.actor.state_dict())
    for value in state["actor"].values():
        value.add_(1)
    if failure == "legacy":
        del state["schema"]
    elif failure == "fingerprint":
        state["method_fingerprint"] = {"changed": True}
    elif failure == "target":
        state["target_actor"] = {}
    elif failure == "replay":
        state["amp"]["replay"]["size"] = 1000
    elif failure == "rng":
        state["amp"]["generator_state"] = torch.zeros(1, dtype=torch.uint8)
    elif failure == "optimizer":
        state["amp"]["discriminator_optimizer"]["param_groups"] = []
    with pytest.raises(ValueError):
        learner.load_state_dict(state)
    for key, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, reference[key])


def test_actor_only_playback_ignores_training_state_and_rejects_identity_drift(learner):
    state = learner.get_state_dict()
    state["amp"] = {"unneeded": "playback never constructs a discriminator"}
    actor = extract_amp_actor_state(
        state,
        expected_playback_identity=learner.amp_playback_identity,
        expected_actor_state=learner.actor.state_dict(),
    )
    assert set(actor) == set(learner.actor.state_dict())
    with pytest.raises(ValueError, match="identity"):
        extract_amp_actor_state(
            state,
            expected_playback_identity={"actor": "changed"},
            expected_actor_state=learner.actor.state_dict(),
        )


def test_atomic_checkpoint_failure_preserves_previous_file(learner, tmp_path, monkeypatch):
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"previous-checkpoint")

    def fail_save(state, stream):
        stream.write(b"incomplete-checkpoint")
        raise OSError("simulated interruption")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interruption"):
        save_amp_checkpoint_atomic(learner.get_state_dict(), destination)
    assert destination.read_bytes() == b"previous-checkpoint"
    assert list(tmp_path.iterdir()) == [destination]


def test_amp_ipc_staging_keeps_pair_identity_and_generic_default_schema():
    generic = RolloutRingBuffer(2, 3, 4, 1)
    amp = RolloutRingBuffer(
        2, 3, 4, 1, extra_observation_dims={"amp_state": 195, "amp_next_state": 195}
    )
    try:
        assert "amp_state" not in generic.slot_shapes
        pool = RolloutStagingPool(capacity=1, num_envs=2, slot_shapes=amp.slot_shapes, device="cpu")
        for value in amp.write_buffer.values():
            value.fill(0)
        amp.write_buffer["amp_state"].fill(3)
        amp.write_buffer["amp_next_state"].fill(7)
        pool.stage_numpy_views(amp.write_buffer)
        data = pool.batch()
        assert data["amp_state"].shape == (3, 2, 195)
        assert torch.all(data["amp_state"] == 3)
        assert torch.all(data["amp_next_state"] == 7)
    finally:
        generic.cleanup()
        amp.cleanup()


def test_probe_env_closes_on_amp_dimension_failure():
    closed = []
    env = SimpleNamespace(obs_groups_spec={"obs": 3, "amp": 194}, close=lambda: closed.append(True))
    with pytest.raises(ValueError, match="195"):
        AMPAPPORunner(
            env_name="unit-test",
            env_cfg_overrides={},
            rl_cfg={},
            env_factory=lambda n, cfg: env,
            device="cpu",
            collector_device="cpu",
            num_envs=1,
            amp_expert_current=np.zeros((1, 195), np.float32),
            amp_expert_next=np.zeros((1, 195), np.float32),
            amp_expert_provenance={"unit": True},
            amp_method_fingerprint={"unit": True},
        )
    assert closed == [True]


def test_snapshot_freezes_only_declared_groups_without_env_step():
    observations = {"obs": np.zeros((2, 3), np.float32), "amp": np.full((2, 195), 3.0, np.float32)}
    assert snapshot_rollout_observations(observations, ()) is observations
    frozen = snapshot_rollout_observations(observations, ("amp",))
    assert set(frozen) == {"amp"}
    observations["amp"].fill(9)
    assert np.all(frozen["amp"] == 3.0)
    assert not np.shares_memory(frozen["amp"], observations["amp"])
    with pytest.raises(ValueError, match="missing"):
        snapshot_rollout_observations(observations, ("missing",))


def _publish_generation_after_consumer_release(names, sync, started, acquired):
    ring = RolloutRingBuffer(
        1,
        2,
        3,
        1,
        num_slots=1,
        create=False,
        shm_name_prefix=names,
        extra_observation_dims={"amp_state": 195, "amp_next_state": 195},
    )
    try:
        ring.attach_sync_primitives(*sync)
        started.set()
        if not ring.wait_for_space(timeout=4.0):
            raise TimeoutError("consumer did not release the borrowed slot")
        acquired.set()
        for value in ring.write_buffer.values():
            value.fill(2.0)
        ring.signal_write_done()
    finally:
        ring.close()


def test_spsc_borrowed_slot_protects_all_fields_until_consumer_release():
    ctx = mp.get_context("spawn")
    ring = RolloutRingBuffer(
        1, 2, 3, 1, num_slots=1, extra_observation_dims={"amp_state": 195, "amp_next_state": 195}
    )
    started, acquired = ctx.Event(), ctx.Event()
    producer = ctx.Process(
        target=_publish_generation_after_consumer_release,
        args=(ring.name, (ring._write_ptr, ring._read_ptr), started, acquired),
    )
    try:
        for value in ring.write_buffer.values():
            value.fill(1.0)
        ring.signal_write_done()
        borrowed = ring.read_numpy_views()
        producer.start()
        assert started.wait(8.0)
        copied = {}
        for field, value in borrowed.items():
            copied[field] = value.copy()
            time.sleep(0.002)
            assert not acquired.is_set()
        assert all(np.all(value == 1.0) for value in copied.values())
        assert int(ring._write_ptr.value) == 1
        ring.advance_read()
        assert acquired.wait(2.0)
        producer.join(timeout=4.0)
        assert producer.exitcode == 0
        assert all(np.all(value == 2.0) for value in ring.read_numpy_views().values())
    finally:
        if producer.pid is not None:
            producer.join(timeout=0.2)
            if producer.is_alive():
                producer.terminate()
                producer.join(timeout=2.0)
        ring.cleanup()


def test_full_ring_rejects_overwrite_and_space_wait_is_cancellable():
    ring = RolloutRingBuffer(1, 1, 1, 1, num_slots=1)
    stopped = threading.Event()
    result = []
    waiter = None
    try:
        with pytest.raises(BufferError, match="empty"):
            ring.read_numpy_views()
        with pytest.raises(BufferError, match="empty"):
            ring.advance_read()
        ring.write_buffer["obs"].fill(8)
        ring.signal_write_done()
        with pytest.raises(BufferError, match="full"):
            _ = ring.write_buffer
        with pytest.raises(BufferError, match="full"):
            ring.signal_write_done()
        assert not ring.wait_for_space(timeout=0)
        waiter = threading.Thread(
            target=lambda: result.append(ring.wait_for_space(timeout=2.0, stop_event=stopped))
        )
        waiter.start()
        stopped.set()
        waiter.join(timeout=0.5)
        assert not waiter.is_alive()
        assert result == [False]
        assert np.all(ring.read_numpy_views()["obs"] == 8)
        assert int(ring._write_ptr.value) == 1
        ring.advance_read()
        assert ring.wait_for_space(timeout=0)
    finally:
        stopped.set()
        if waiter is not None:
            waiter.join(timeout=1.0)
        ring.cleanup()


class RemoteOnlyEnv:
    """Tiny deterministic spawn environment used only by remote CI."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.obs_groups_spec = {"obs": 3, "critic": 3, "amp": 195}
        self.action_space = SimpleNamespace(shape=(2,))
        self.step_index = 0

    def reset(self, indices=None):
        return {
            "obs": np.zeros((self.num_envs, 3), np.float32),
            "critic": np.zeros((self.num_envs, 3), np.float32),
            "amp": np.zeros((self.num_envs, 195), np.float32),
        }, {}

    def step(self, actions):
        self.step_index += 1
        observations = {
            "obs": np.full((self.num_envs, 3), self.step_index, np.float32),
            "critic": np.full((self.num_envs, 3), self.step_index, np.float32),
            "amp": np.full((self.num_envs, 195), 2.0, np.float32),
        }
        final = {key: value.copy() for key, value in observations.items()}
        final["amp"][0] = 7.0
        truncated = np.zeros(self.num_envs, dtype=bool)
        truncated[0] = self.step_index % 2 == 0
        if truncated[0]:
            for value in observations.values():
                value[0] = 0
        return SimpleNamespace(
            obs=observations,
            final_observation=final,
            info={},
            reward=np.arange(self.num_envs, dtype=np.float32) + 1,
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=truncated,
        )

    def close(self):
        pass


def remote_only_env_factory(num_envs, overrides):
    return RemoteOnlyEnv(num_envs)


class RemoteOnlyInPlaceEnv(RemoteOnlyEnv):
    def reset(self, indices=None):
        self.storage, info = super().reset(indices)
        return self.storage, info

    def step(self, actions):
        state = super().step(actions)
        for key, value in state.obs.items():
            np.copyto(self.storage[key], value)
        state.obs = self.storage
        return state


def remote_only_inplace_env_factory(num_envs, overrides):
    return RemoteOnlyInPlaceEnv(num_envs)


@pytest.mark.parametrize("amp_enabled", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_remote_only_appo_spawn_update_checkpoint_and_resume(tmp_path, amp_enabled, in_place):
    """Runs optimizer/collector operations on remote CI, excluded from local checks."""
    cfg = {
        "actor": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [4],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "rsl_rl.models.MLPModel",
            "hidden_dims": [4],
            "activation": "elu",
            "obs_normalization": False,
        },
        "algorithm": {"num_learning_epochs": 1, "num_mini_batches": 1, "enable_compile": False},
    }
    kwargs = {}
    if amp_enabled:
        cfg["amp"] = {
            "observation_dim": 195,
            "hidden_dims": [4],
            "replay_capacity": 16,
            "discriminator_batch_size": 2,
            "discriminator_updates": 1,
        }
        provenance = {"motions": [{"sha256": "remote-synthetic-unit-test"}]}
        kwargs = {
            "amp_expert_current": np.zeros((4, 195), np.float32),
            "amp_expert_next": np.ones((4, 195), np.float32),
            "amp_expert_provenance": provenance,
            "amp_method_fingerprint": {
                "amp": copy.deepcopy(cfg["amp"]),
                "expert": provenance,
                "observation": {"policy_observation_mapping": {"obs": "policy"}},
                "runtime": {"unit-test": "remote-only"},
            },
        }
    runner_type = AMPAPPORunner if amp_enabled else APPORunner
    env_factory = remote_only_inplace_env_factory if in_place else remote_only_env_factory
    runner = runner_type(
        env_name="remote-synthetic",
        env_cfg_overrides={},
        rl_cfg=copy.deepcopy(cfg),
        env_factory=env_factory,
        device="cpu",
        collector_device="cpu",
        num_envs=2,
        steps_per_env=2,
        replay_queue_size=1,
        seed=4,
        **kwargs,
    )
    try:
        runner.learn(max_iterations=1, save_interval=1, log_dir=str(tmp_path))
    finally:
        runner.close()
    state = torch.load(tmp_path / "model_1.pt", weights_only=True)
    assert bool("amp" in state) == amp_enabled
    assert state["optimizer"]["state"]
    if amp_enabled:
        assert state["amp"]["discriminator_version"] == 1
        assert state["update_counter"] == 1
        assert state["amp"]["discriminator_optimizer"]["state"]
        assert state["amp"]["normalizer"]["count"].item() > 0
        assert torch.any(state["amp"]["replay"]["next_states"] == 7.0)
        replay = state["amp"]["replay"]
        current = replay["states"][: replay["size"], 0]
        next_state = replay["next_states"][: replay["size"], 0]
        assert torch.any((current == 0.0) & (next_state == 2.0))
        assert torch.any((current == 2.0) & (next_state == 7.0))
        resumed = runner_type(
            env_name="remote-synthetic",
            env_cfg_overrides={},
            rl_cfg=copy.deepcopy(cfg),
            env_factory=env_factory,
            device="cpu",
            collector_device="cpu",
            num_envs=2,
            steps_per_env=2,
            seed=4,
            **kwargs,
        )
        try:
            restored = resumed._build_learner()
            resumed._restore_learner_checkpoint(restored, state)
            assert restored.discriminator_version == 1
            assert restored._update_counter == 1
            torch.testing.assert_close(
                restored.amp_generator.get_state(), state["amp"]["generator_state"]
            )
            for key, tensor in restored.target_actor.state_dict().items():
                torch.testing.assert_close(tensor, state["target_actor"][key])
        finally:
            resumed.close()
