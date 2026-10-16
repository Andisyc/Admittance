"""Generate a small numeric oracle from the unmodified upstream MUSE source.

Producer: hazel-hammer/Anybody@0cffa139eab0bc145c9216223e249f03c67a324c,
source/rsl_rl/rsl_rl/modules/latent_bottleneck_muse_transformer.py.
Run explicitly with --upstream-source PATH. Normal tests only read the NPZ;
they need neither a network connection nor the upstream checkout. This script
never imports the port under test. It evaluates the original encoder and the
original decoder construction/forward statements, without importing IsaacLab
or the donor's entire rsl_rl package.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from torch import nn

SOURCE_COMMIT = "0cffa139eab0bc145c9216223e249f03c67a324c"
SOURCE_SHA256 = "6c72578427a6680851b7f47e1ca94fe422908fb9bf11538f670eb922f69f817d"
ENCODER_SEED = 7125
DECODER_SEED = 8125


def fill_parameters(module: nn.Module, seed: int) -> None:
    """Fixed CPU weights; each component uses its own generator and sort order."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in sorted(module.named_parameters()):
            if parameter.ndim == 1 and name.endswith("weight"):
                parameter.fill_(1.0)  # LayerNorm gains.
            else:
                parameter.normal_(mean=0.0, std=0.04, generator=generator)


def generate(source_path: Path, output: Path) -> None:
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the pinned, unmodified Anybody MUSE source")
    tree = ast.parse(source)
    finite_tree = ast.parse((source_path.parent.parent / "utils/finite_checks.py").read_text())
    selected = [
        node
        for node in finite_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "require_all_finite"
    ]
    selected.extend(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name
        in {"_muse_log_sigma_bounds", "_split_history_terms", "_MUSETransformerEncoder"}
    )
    outer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LatentBottleneckMUSETransformer"
    )
    selected.extend(
        node
        for node in outer.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_maybe_normalize_latent", "_shape_proprio_for_decoder", "_decode"}
    )
    namespace = {"torch": torch, "nn": nn, "math": math, "Sequence": Sequence}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
    encoder = namespace["_MUSETransformerEncoder"](
        goal_term_sizes=(58, 6, 3, 3, 3),
        proprio_term_sizes=(29, 29, 3, 29),
        mask_term_size=1,
        history_length=5,
        latent_dim=16,
        d_model=192,
        nhead=4,
        num_layers=2,
        ffn_dim=768,
        activation="gelu",
        latent_sigma_min=0.001,
        latent_sigma_max=1.0,
        fixed_encoder_std=0.3,
        predict_sigma=False,
        dropout=0.0,
    ).eval()

    constructor = next(
        node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    start = next(
        i
        for i, node in enumerate(constructor.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "act_cls" for t in node.targets)
    )
    end = next(
        i
        for i, node in enumerate(constructor.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "decoder" for t in node.targets)
    )
    holder = SimpleNamespace(num_actions=29, decoder_one_step_proprio=False, latent_normalize=True)
    namespace.update(
        self=holder,
        activation="gelu",
        latent_dim=16,
        decoder_proprio_dim=450,
        decoder_hidden_dims=(1024, 512, 256, 128),
    )
    exec(
        compile(
            ast.Module(body=constructor.body[start : end + 1], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    holder._shape_proprio_for_decoder = MethodType(namespace["_shape_proprio_for_decoder"], holder)
    fill_parameters(encoder, ENCODER_SEED)
    fill_parameters(holder.decoder, DECODER_SEED)
    obs = torch.sin(torch.arange(6 * 524, dtype=torch.float32).reshape(6, 524) * 0.013)
    obs[:, -1] = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)
    with torch.no_grad():
        mu, _, proprio = encoder.encode(obs)
        z = namespace["_maybe_normalize_latent"](holder, mu)
        action = namespace["_decode"](holder, z, proprio)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source_commit=np.asarray(SOURCE_COMMIT),
        source_sha256=np.asarray(SOURCE_SHA256),
        producer_torch=np.asarray(torch.__version__),
        encoder_seed=np.asarray(ENCODER_SEED),
        decoder_seed=np.asarray(DECODER_SEED),
        obs=obs.numpy(),
        mu=mu.numpy(),
        z=z.numpy(),
        proprio=proprio.numpy(),
        action=action.numpy(),
    )
    print(f"Wrote independent upstream reference: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("muse_transformer_v1.npz")
    )
    args = parser.parse_args()
    generate(args.upstream_source, args.output)
