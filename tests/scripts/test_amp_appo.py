from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.scripts import amp_appo


def _effective_amp_owner():
    from hydra import compose, initialize_config_dir

    from unilab.base.config_adapter import BackendAdapter

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / "src/unilab/conf/appo")):
        cfg = compose("config", overrides=["task=g1_amp_walk/mujoco"])
    env = BackendAdapter(cfg, root_dir=root, algo_name="appo").build_task_env_cfg_override()
    return cfg, env


def test_amp_runtime_resolves_package_neutral_expert_kwargs(monkeypatch) -> None:
    transitions = SimpleNamespace(
        current=np.zeros((3, 195), dtype=np.float32),
        next=np.ones((3, 195), dtype=np.float32),
        provenance={"schema_version": 1, "motion_class": "forward_walk"},
    )
    selected: list[str] = []

    def load(manifest: str):
        selected.append(manifest)
        return transitions

    monkeypatch.setattr(amp_appo, "load_g1_amp_expert_transitions", load)
    rl_cfg = {
        "runtime_impl": "amp_appo",
        "runtime_mode": "train",
        "amp": {
            "expert_manifest": "regait_v013_a022",
            "observation_dim": 195,
            "reward_coef": 0.1,
        },
        "actor": {"hidden_dims": [8]},
        "critic": {"hidden_dims": [8]},
        "algorithm": {"gamma": 0.99},
        "runtime_context": {
            "task": {"name": "G1AMPWalk", "command": [1.0, 0.0, 0.0]},
            "observation": {
                "amp_dim": 195,
                "policy_observation_mapping": {"amp": "amp_style"},
            },
        },
    }
    runtime = amp_appo.resolve_amp_appo_runtime(rl_cfg)
    assert runtime.runner_cls.__name__ == "AMPAPPORunner"
    assert set(runtime.runner_kwargs) == {
        "amp_expert_current",
        "amp_expert_next",
        "amp_expert_provenance",
        "amp_method_fingerprint",
    }
    assert runtime.runner_kwargs["amp_expert_current"].shape == (3, 195)
    assert selected == ["regait_v013_a022"]
    fingerprint = runtime.runner_kwargs["amp_method_fingerprint"]
    assert fingerprint["task"]["command"] == [1.0, 0.0, 0.0]
    assert fingerprint["observation"]["amp_dim"] == 195
    assert fingerprint["expert"] == transitions.provenance


def test_amp_runtime_rejects_wrong_owner_marker_before_expert_loading(monkeypatch) -> None:
    called = False

    def load(_manifest: str):
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(amp_appo, "load_g1_amp_expert_transitions", load)
    with pytest.raises(ValueError, match="runtime_impl"):
        amp_appo.resolve_amp_appo_runtime({"runtime_impl": "appo", "amp": {"observation_dim": 195}})
    assert called is False


def test_amp_play_runtime_does_not_load_expert(monkeypatch) -> None:
    def load(_manifest: str):
        raise AssertionError("play mode must not load expert motions")

    monkeypatch.setattr(amp_appo, "load_g1_amp_expert_transitions", load)
    runtime = amp_appo.resolve_amp_appo_runtime(
        {
            "runtime_impl": "amp_appo",
            "runtime_mode": "play",
            "amp": {"observation_dim": 195},
        }
    )
    assert runtime.runner_kwargs == {}


def test_amp_runtime_context_rejects_effective_reward_override() -> None:
    cfg, env = _effective_amp_owner()
    env["rewards"]["tracking_lin_vel"]["weight"] = 99.0
    with pytest.raises(ValueError, match="reward schema"):
        amp_appo.resolve_amp_runtime_context(cfg, env)


def test_amp_runtime_context_rejects_command_semantic_override() -> None:
    cfg, env = _effective_amp_owner()
    env["commands"]["twist"]["planar_dead_zone"] = 0.2
    with pytest.raises(ValueError, match="command owner"):
        amp_appo.resolve_amp_runtime_context(cfg, env)


def test_amp_runtime_context_rejects_body_order_override() -> None:
    cfg, env = _effective_amp_owner()
    asset_cfg = env["observations"]["amp_style"]["terms"]["state"]["params"]["asset_cfg"]
    asset_cfg["body_names"][0:2] = reversed(asset_cfg["body_names"][0:2])
    with pytest.raises(ValueError, match="AMP observation owner"):
        amp_appo.resolve_amp_runtime_context(cfg, env)
