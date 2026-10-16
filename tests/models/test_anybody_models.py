"""Network semantics, independent donor oracle, and gradient authority for S2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from unilab.models import MUSEDecoder, MUSEEncoderDecoder, MUSETransformerEncoder

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/anybody/muse_transformer_v1.npz"


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def golden():
    with np.load(_FIXTURE, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _fill_parameters(module: nn.Module, seed: int) -> None:
    # The oracle generator defines this deterministic weight recipe independently.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in sorted(module.named_parameters()):
            if parameter.ndim == 1 and name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                parameter.normal_(mean=0.0, std=0.04, generator=generator)


@pytest.fixture
def model(golden):
    instance = MUSEEncoderDecoder().eval()
    _fill_parameters(instance.encoder, int(golden["encoder_seed"]))
    _fill_parameters(instance.decoder, int(golden["decoder_seed"]))
    return instance


def test_matches_independent_upstream_mu_latent_history_and_action(model, golden):
    assert str(golden["source_commit"]) == "0cffa139eab0bc145c9216223e249f03c67a324c"
    obs = torch.from_numpy(golden["obs"])
    with torch.no_grad():
        mu, z = model.encode(obs)
        _, history, mask = model.split_obs(obs)
        action = model(obs)
        decoded = model.decode(z, history)
    assert model.obs_dim == 524
    assert model.encoder.seq_len == 7
    assert model.decoder.input_dim == 466
    assert mask.shape == (6, 1)
    for actual, key in ((mu, "mu"), (z, "z"), (history, "proprio"), (action, "action")):
        torch.testing.assert_close(actual, torch.from_numpy(golden[key]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(decoded, action, atol=0, rtol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(z, dim=-1), torch.ones(6))


def test_term_major_observation_becomes_frame_major_decoder_history():
    encoder = MUSETransformerEncoder()
    obs = torch.zeros(1, 524)
    expected = torch.empty(5, 90)
    offset = 73
    column = 0
    for term_id, size in enumerate((29, 29, 3, 29)):
        for frame in range(5):
            values = 1000 * term_id + 100 * frame + torch.arange(size, dtype=torch.float32)
            obs[0, offset + frame * size : offset + (frame + 1) * size] = values
            expected[frame, column : column + size] = values
        offset += 5 * size
        column += size
    _, history, _ = encoder.split_obs(obs)
    torch.testing.assert_close(history[0], expected, atol=0, rtol=0)
    decoder = MUSEDecoder()
    seen = []
    handle = decoder.net[0].register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    try:
        decoder(torch.full((1, 16), -7.0), history)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0][0, :16], torch.full((16,), -7.0), atol=0, rtol=0)
    torch.testing.assert_close(seen[0][0, 16:], expected.flatten(), atol=0, rtol=0)


@pytest.mark.parametrize("mask_value", [0.0, 1.0])
def test_goal_mask_controls_goal_influence_without_hiding_proprio(model, golden, mask_value):
    obs = torch.from_numpy(golden["obs"][:2].copy())
    obs[:, -1] = mask_value
    goal_changed = obs.clone()
    goal_changed[:, :73] += 5.0
    history_changed = obs.clone()
    history_changed[:, 73:523] += 1.0
    with torch.no_grad():
        baseline = model(obs)
        goal_output = model(goal_changed)
        history_output = model(history_changed)
    if mask_value:
        torch.testing.assert_close(goal_output, baseline, atol=0, rtol=0)
    else:
        assert not torch.allclose(goal_output, baseline, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(history_output, baseline, atol=1e-7, rtol=1e-6)


def test_action_gradient_reaches_encoder_attention_and_decoder(model, golden):
    model.train()
    obs = torch.from_numpy(golden["obs"].copy()).requires_grad_()
    action = model(obs)
    # A differentiability probe, not an RL reward or a distillation objective.
    probe = torch.linspace(-1.0, 1.0, 29)
    (action * probe).sum().backward()
    parameters = dict(model.named_parameters())
    for name in (
        "encoder.goal_proj.weight",
        "encoder.proprio_proj.weight",
        "encoder.transformer.layers.0.self_attn.in_proj_weight",
        "encoder.mu_head.weight",
        "decoder.net.0.weight",
        "decoder.net.8.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name
    assert obs.grad is not None and torch.isfinite(obs.grad).all()
    assert torch.count_nonzero(obs.grad[1, :73]) == 0  # Masked goal has no gradient.
    assert torch.count_nonzero(obs.grad[0, :73]) > 0


def test_decoder_uses_external_latent(model, golden):
    obs = torch.from_numpy(golden["obs"])
    _, history, _ = model.split_obs(obs)
    with torch.no_grad():
        _, z = model.encode(obs)
        baseline = model.decode(z, history)
        intervened = model.decode(-z, history)
    assert not torch.allclose(baseline, intervened, atol=1e-7, rtol=1e-6)


def test_leading_batch_axes_unbatched_and_noncontiguous_inputs(model, golden):
    obs = torch.from_numpy(golden["obs"])
    with torch.no_grad():
        expected = model(obs)
        torch.testing.assert_close(model(obs.reshape(2, 3, 524)), expected.reshape(2, 3, 29))
        torch.testing.assert_close(model(obs[0]), expected[0])
        noncontiguous = obs.reshape(2, 3, 524).transpose(0, 1)
        assert not noncontiguous.is_contiguous()
        torch.testing.assert_close(model(noncontiguous), expected.reshape(2, 3, 29).transpose(0, 1))
        # A different row/batch must not alter an earlier row's result.
        model(obs.flip(0))
        torch.testing.assert_close(model(obs[:1]), expected[:1])


def test_zero_latent_remains_finite_and_is_not_claimed_unit_norm(model, golden):
    with torch.no_grad():
        model.encoder.mu_head.weight.zero_()
        model.encoder.mu_head.bias.zero_()
    obs = torch.from_numpy(golden["obs"])
    mu, z = model.encode(obs)
    assert torch.count_nonzero(mu) == 0
    assert torch.count_nonzero(z) == 0
    assert torch.isfinite(model(obs)).all()


@pytest.mark.parametrize("bad_shape", [(), (2, 523), (2, 525)])
def test_observation_layout_rejected(bad_shape):
    with pytest.raises(ValueError, match="last dimension 524"):
        MUSETransformerEncoder().encode(torch.zeros(bad_shape))


@pytest.mark.parametrize("mask", [-1.0, 0.5, 2.0])
def test_non_binary_mask_rejected(mask):
    obs = torch.zeros(2, 524)
    obs[:, -1] = mask
    with pytest.raises(ValueError, match="raw 0 or 1"):
        MUSETransformerEncoder().encode(obs)


@pytest.mark.parametrize("field", [0, 73, 523])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_observation_rejected_even_when_goal_is_masked(field, bad):
    obs = torch.zeros(2, 524)
    obs[:, -1] = 1
    obs[0, field] = bad
    with pytest.raises(ValueError, match="finite"):
        MUSETransformerEncoder().encode(obs)


def test_wrong_dtype_rejected():
    with pytest.raises(ValueError, match="dtype/device"):
        MUSETransformerEncoder().encode(torch.zeros(2, 524, dtype=torch.float64))


@pytest.mark.parametrize(
    "z, history, message",
    [
        (torch.zeros(2, 15), torch.zeros(2, 5, 90), "last dimension 16"),
        (torch.zeros(2, 16), torch.zeros(2, 450), "proprio_history must have shape"),
        (torch.zeros(2, 16), torch.zeros(1, 5, 90), "proprio_history must have shape"),
        (torch.full((2, 16), float("nan")), torch.zeros(2, 5, 90), "finite"),
        (torch.zeros(2, 16), torch.full((2, 5, 90), float("inf")), "finite"),
    ],
)
def test_decoder_invalid_inputs_rejected(z, history, message):
    with pytest.raises(ValueError, match=message):
        MUSEDecoder()(z, history)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_length": 0},
        {"goal_term_sizes": ()},
        {"proprio_term_sizes": (29, 0)},
        {"latent_dim": -1},
        {"latent_dim": True},
        {"d_model": 191},
        {"nhead": 0},
        {"num_layers": 0},
        {"ffn_dim": 0},
    ],
)
def test_invalid_encoder_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        MUSETransformerEncoder(**kwargs)


def test_dimensions_are_configuration_derived():
    model = MUSEEncoderDecoder(
        goal_term_sizes=(4, 2),
        proprio_term_sizes=(2, 2, 3, 2),
        history_length=3,
        latent_dim=4,
        d_model=16,
        nhead=2,
        num_layers=1,
        ffn_dim=32,
        num_actions=2,
        decoder_hidden_dims=(16, 8),
    )
    obs = torch.zeros(2, 6 + 3 * 9 + 1)
    assert model.obs_dim == 34
    assert model.decoder.input_dim == 31
    assert model(obs).shape == (2, 2)
