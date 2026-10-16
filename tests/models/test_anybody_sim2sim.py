"""Network-structure changes must remain visible to saved run contracts."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.utils.sim2sim import DENYLIST, extract_contract_snapshot


def _cfg():
    root = Path(__file__).resolve().parents[2] / "src/unilab/conf/appo"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        return compose(config_name="config", overrides=["task=anybody_g1/mujoco"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_name", "rsl_rl.models.MLPModel"),
        ("activation", "elu"),
        ("history_length", 4),
        ("latent_dim", 32),
        ("goal_term_sizes", [29, 29, 6, 3, 3, 3]),
        ("proprio_term_sizes", [29, 3, 29, 29]),
        ("d_model", 128),
        ("nhead", 8),
        ("num_layers", 3),
        ("ffn_dim", 512),
        ("obs_normalization", True),
        ("distribution_cfg.init_std", 0.9),
    ],
)
def test_actor_contract_captures_semantic_and_network_changes(field, value):
    cfg = _cfg()
    baseline = extract_contract_snapshot(cfg)
    OmegaConf.update(cfg, f"algo.actor.{field}", value)
    changed = extract_contract_snapshot(cfg)
    owner = f"algo.actor.{field.split('.')[0]}"
    assert owner in DENYLIST
    assert changed[owner] != baseline[owner]


def test_muse_profile_requires_real_motion_path_and_defers_export():
    cfg = _cfg()
    assert OmegaConf.is_missing(cfg.env.commands.motion.params, "motion_file")
    assert cfg.algo.actor.class_name == "unilab.rl.algos.anybody.MUSEActorModel"
    assert cfg.algo.critic.class_name == "rsl_rl.models.MLPModel"
    assert cfg.algo.actor.obs_normalization is False
    assert cfg.algo.algorithm.enable_compile is False
    assert cfg.training.no_play is True
    assert cfg.training.export_policy is False


def test_public_cli_routes_opt_in_profile_without_starting_a_process():
    from unilab.cli import build_command

    command = build_command(
        mode="train",
        algo="appo",
        task="anybody_g1",
        sim="mujoco",
        overrides=["training.no_play=true", "training.export_policy=false"],
    )
    assert "task=anybody_g1/mujoco" in command
    assert any(part.endswith("train_appo.py") for part in command)


def test_support_matrix_does_not_claim_training_acceptance_for_new_profile():
    from scripts.tools.support_matrix import EvidenceLevel, build_support_rows

    row = next(
        r
        for r in build_support_rows(Path(__file__).resolve().parents[2])
        if r.entrypoint_label == "APPO (torch)" and r.task_slug == "anybody_g1"
    )
    assert row.cells["mujoco"].level == EvidenceLevel.CONFIGURED
