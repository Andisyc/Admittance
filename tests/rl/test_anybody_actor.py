"""Prove MUSE's collector/learner policy contract without a training run."""

from __future__ import annotations

import copy
import io

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from unilab.models import MUSEEncoderDecoder
from unilab.rl.algos.anybody import MUSEActorModel
from unilab.rl.algos.appo.learner import APPOLearner
from unilab.rl.ipc.weight_sync import SharedWeightSync


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _obs(batch=4, width=524):
    values = torch.randn(batch, width)
    values[:, -1] = torch.arange(batch) % 2
    return TensorDict({"policy": values}, batch_size=[batch])


def _actor(obs=None, **kwargs):
    params = dict(hidden_dims=(32, 16), d_model=16, nhead=4, num_layers=1, ffn_dim=32)
    params.update(kwargs)
    return MUSEActorModel(
        _obs() if obs is None else obs, {"actor": ["policy"]}, "actor", 29, **params
    )


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_collector_and_learner_mean_std_and_logprob_are_identical(std_type):
    obs = _obs()
    actor = _actor(
        obs,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "std_type": std_type,
            "init_std": 0.7,
        },
    )
    actor.eval()
    with torch.no_grad():
        actions = actor(obs, stochastic_output=True)
        collector_mean = actor.output_mean.clone()
        collector_std = actor.output_std.clone()
        collector_logp = actor.get_output_log_prob(actions)
    # Bypass APPOLearner.__init__: this proof creates no optimizer and performs
    # no update. Exercise the actual learner's direct mlp/normalizer path.
    learner = APPOLearner.__new__(APPOLearner)
    learner.actor = actor
    actor.train()
    mean, std = learner._actor_mean_std(obs["policy"])
    torch.testing.assert_close(mean, collector_mean)
    torch.testing.assert_close(std, collector_std)
    torch.testing.assert_close(
        learner._gaussian_log_prob(actions, mean, std), collector_logp, atol=1e-5, rtol=1e-5
    )


def test_raw_mask_is_preserved_and_normalization_is_noop():
    obs = _obs()
    actor = _actor(obs)
    original = obs["policy"].clone()
    assert isinstance(actor.obs_normalizer, nn.Identity)
    assert actor.obs_normalization is False
    assert isinstance(actor.mlp, MUSEEncoderDecoder)
    actor.update_normalization(obs)
    torch.testing.assert_close(actor.get_latent(obs), original, atol=0, rtol=0)
    actor(obs)
    torch.testing.assert_close(obs["policy"], original, atol=0, rtol=0)
    assert actor.mlp.decoder.net[0].out_features == 32
    assert isinstance(actor.mlp.decoder.net[1], nn.GELU)


@pytest.mark.parametrize("invalid_mask", [-1.0, 0.5, 2.0, float("nan")])
def test_invalid_masks_fail_before_policy_output(invalid_mask):
    obs = _obs()
    actor = _actor(obs)
    obs["policy"][:, -1] = invalid_mask
    with pytest.raises(ValueError, match="goal_mask|finite"):
        actor(obs)


def test_state_checkpoint_target_copy_and_weight_broadcast_preserve_policy():
    obs = _obs()
    actor = _actor(
        obs,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.43,
        },
    ).eval()
    target = copy.deepcopy(actor).eval()
    assert next(target.parameters()) is not next(actor.parameters())
    # This is the same actor state entry saved by APPO's checkpoint owner.
    stream = io.BytesIO()
    torch.save({"actor": actor.state_dict()}, stream)
    stream.seek(0)
    restored = _actor(obs).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True)["actor"])
    receiver = _actor(obs).eval()
    sync = SharedWeightSync.from_state_dict(actor.state_dict(), create=True)
    attached = None
    try:
        attached = SharedWeightSync(
            {name: value.shape for name, value in receiver.state_dict().items()},
            create=False,
            shm_name=sync.name,
        )
        receiver_state = dict(receiver.state_dict())
        assert attached.read_weights_into(receiver_state) == sync.version
        receiver.load_state_dict(receiver_state)
        with torch.no_grad():
            expected = actor(obs)
            for other in (target, restored, receiver):
                torch.testing.assert_close(other(obs), expected, atol=0, rtol=0)
                for name, value in actor.state_dict().items():
                    torch.testing.assert_close(other.state_dict()[name], value, atol=0, rtol=0)
    finally:
        if attached is not None:
            attached.close()
        sync.cleanup()


def test_backward_reaches_encoder_and_decoder_without_parameter_update():
    obs = _obs()
    actor = _actor(obs)
    before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    actor(obs).square().mean().backward()
    for module in (actor.mlp.encoder, actor.mlp.decoder):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
        )
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    for name, parameter in actor.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"obs_normalization": True}, "obs_normalization=False"),
        ({"activation": "elu"}, "gelu"),
        (
            {
                "distribution_cfg": {
                    "class_name": "rsl_rl.modules.distribution.HeteroscedasticGaussianDistribution"
                }
            },
            "state-independent",
        ),
        (
            {
                "distribution_cfg": {
                    "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                    "tanh": True,
                }
            },
            "accepts only",
        ),
    ],
)
def test_rejects_incompatible_preprocessing_and_distribution(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _actor(**kwargs)


@pytest.mark.parametrize("init_std", [0, -1, float("nan"), float("inf"), True])
def test_requires_positive_finite_gaussian_std(init_std):
    with pytest.raises(ValueError, match="init_std"):
        _actor(
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "init_std": init_std,
            }
        )


def test_rejects_wrong_groups_dimensions_and_std_parameterization():
    with pytest.raises(ValueError, match="exactly one"):
        MUSEActorModel(_obs(), {"actor": ["policy", "critic"]}, "actor", 29)
    with pytest.raises(ValueError, match="output_dim=29"):
        MUSEActorModel(_obs(), {"actor": ["policy"]}, "actor", 1)
    with pytest.raises(ValueError, match="observation width must be 524"):
        _actor(_obs(width=523))
    with pytest.raises(ValueError, match="std_type"):
        _actor(
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                "std_type": "tanh",
            }
        )


def test_distribution_config_is_not_mutated():
    config = {"class_name": "rsl_rl.modules.distribution.GaussianDistribution", "init_std": 0.3}
    expected = copy.deepcopy(config)
    _actor(distribution_cfg=config)
    assert config == expected


def test_dimension_mapping_used_by_appo_config_is_accepted():
    actor = MUSEActorModel(
        _obs(),
        {"actor": {"policy": 524}},
        "actor",
        29,
        hidden_dims=(32, 16),
        d_model=16,
        num_layers=1,
        ffn_dim=32,
    )
    assert actor.obs_dim == 524


def test_export_is_explicitly_not_yet_supported():
    actor = _actor()
    with pytest.raises(NotImplementedError, match="TorchScript"):
        actor.as_jit()
    with pytest.raises(NotImplementedError, match="ONNX"):
        actor.as_onnx(verbose=False)
