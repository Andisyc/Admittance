"""Versioned AMP checkpoints with preflight and actor-only playback support."""

from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

SCHEMA = "unilab.amp-appo/v1"


def checkpoint_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: checkpoint_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [checkpoint_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _tensor_state(actual: Any, expected: Any, name: str) -> None:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        raise ValueError(f"AMP checkpoint {name} keys differ")
    for key, target in expected.items():
        tensor = actual[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != target.shape
            or tensor.dtype != target.dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"AMP checkpoint {name}.{key} tensor differs or is non-finite")


def _counter(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"AMP checkpoint {name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"AMP checkpoint {name} exceeds capacity")
    return int(value)


def _optimizer(actual: Any, expected: Any, shapes: list[list[int]], name: str) -> None:
    if not isinstance(actual, dict) or set(actual) != {"param_groups", "state"}:
        raise ValueError(f"AMP checkpoint {name} optimizer state is malformed")
    groups = actual["param_groups"]
    reference_groups = expected["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(reference_groups):
        raise ValueError(f"AMP checkpoint {name} optimizer groups differ")
    ids: list[int] = []
    for group, reference in zip(groups, reference_groups):
        if not isinstance(group, dict) or set(group) != set(reference):
            raise ValueError(f"AMP checkpoint {name} optimizer group fields differ")
        if group["params"] != reference["params"]:
            raise ValueError(f"AMP checkpoint {name} optimizer parameter order differs")
        for key in group:
            if key in {"params", "lr"}:
                continue
            if group[key] != reference[key]:
                raise ValueError(f"AMP checkpoint {name} optimizer option {key} differs")
        lr = group["lr"]
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not 0.0 < lr < float("inf"):
            raise ValueError(f"AMP checkpoint {name} optimizer learning rate is invalid")
        ids.extend(group["params"])
    if len(ids) != len(shapes) or len(set(ids)) != len(ids):
        raise ValueError(f"AMP checkpoint {name} optimizer dimensions differ")
    states = actual["state"]
    if not isinstance(states, dict) or not set(states).issubset(set(ids)):
        raise ValueError(f"AMP checkpoint {name} optimizer has unknown parameters")
    shape_by_id = dict(zip(ids, shapes))
    group_by_id = {ident: group for group in groups for ident in group["params"]}
    for ident, values in states.items():
        if not isinstance(values, dict):
            raise ValueError(f"AMP checkpoint {name} optimizer parameter state is malformed")
        group = group_by_id[ident]
        if "betas" in group and "amsgrad" in group:
            required_keys = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                required_keys.add("max_exp_avg_sq")
            if set(values) != required_keys:
                raise ValueError(f"AMP checkpoint {name} Adam moment state is incomplete")
        for key, tensor in values.items():
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"AMP checkpoint {name} optimizer {key} must be a finite tensor")
            if key == "step":
                if tensor.numel() != 1 or tensor.item() < 0:
                    raise ValueError(f"AMP checkpoint {name} optimizer step is invalid")
            elif list(tensor.shape) != shape_by_id[ident]:
                raise ValueError(f"AMP checkpoint {name} optimizer {key} shape differs")


def validate_amp_checkpoint(
    state: dict[str, Any],
    *,
    expected_method_fingerprint: dict[str, Any],
    expected_state: dict[str, Any],
) -> None:
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError("AMP resume requires a versioned UniLab AMP checkpoint")
    for key in (
        "method_fingerprint",
        "expert_provenance",
        "playback_identity",
        "optimizer_param_shapes",
        "discriminator_param_shapes",
    ):
        expected = (
            expected_method_fingerprint if key == "method_fingerprint" else expected_state[key]
        )
        if state.get(key) != expected:
            raise ValueError(f"AMP checkpoint {key} differs from the effective runtime")
    for key in ("actor", "critic", "target_actor"):
        _tensor_state(state.get(key), expected_state[key], key)
    _counter(state.get("update_counter"), "update_counter")
    amp = state.get("amp")
    reference = expected_state["amp"]
    if not isinstance(amp, dict) or set(amp) != set(reference):
        raise ValueError("AMP checkpoint learner-local state is incomplete")
    for key in ("discriminator", "normalizer"):
        _tensor_state(amp[key], reference[key], key)
    normalizer = amp["normalizer"]
    if normalizer["count"].item() < 0 or (normalizer["variance"] < 0).any():
        raise ValueError("AMP checkpoint normalizer moments are invalid")
    replay = amp["replay"]
    target_replay = reference["replay"]
    if not isinstance(replay, dict) or set(replay) != set(target_replay):
        raise ValueError("AMP checkpoint replay is incomplete")
    if replay["capacity"] != target_replay["capacity"] or replay["width"] != target_replay["width"]:
        raise ValueError("AMP checkpoint replay layout differs")
    _tensor_state(
        {key: replay[key] for key in ("states", "next_states")},
        {key: target_replay[key] for key in ("states", "next_states")},
        "replay",
    )
    capacity = replay["capacity"]
    _counter(replay["size"], "replay size", maximum=capacity)
    _counter(replay["write_position"], "replay write position", maximum=capacity - 1)
    _counter(amp["discriminator_version"], "discriminator version")
    try:
        torch.Generator(device="cpu").set_state(amp["generator_state"].cpu())
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise ValueError("AMP checkpoint RNG state is invalid") from exc
    _optimizer(
        state.get("optimizer"),
        expected_state["optimizer"],
        expected_state["optimizer_param_shapes"],
        "base",
    )
    _optimizer(
        amp["discriminator_optimizer"],
        reference["discriminator_optimizer"],
        expected_state["discriminator_param_shapes"],
        "discriminator",
    )


def load_amp_resume_state(learner: Any, state: dict[str, Any]) -> None:
    learner.load_state_dict(state)


def extract_amp_actor_state(
    state: dict[str, Any],
    *,
    expected_playback_identity: dict[str, Any],
    expected_actor_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError("AMP playback requires a versioned UniLab AMP checkpoint")
    if state.get("playback_identity") != expected_playback_identity:
        raise ValueError("AMP playback actor/policy/runtime identity differs")
    _tensor_state(state.get("actor"), expected_actor_state, "actor")
    return cast(dict[str, torch.Tensor], state["actor"])


def save_amp_checkpoint_atomic(state: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".amp-", delete=False
        ) as stream:
            temporary = stream.name
            torch.save(checkpoint_to_cpu(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
