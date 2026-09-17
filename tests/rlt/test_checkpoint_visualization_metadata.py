from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from rlt_so101_dual.core.config import RLTConfig
from rlt_so101_dual.core.critic import TwinCritic
from rlt_so101_dual.core.trainer import offline_rl_loop
from tests.rlt.helpers import fill_buffer, make_test_algorithm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlt_so101_dual.experimental.visualize_rlt_warmup_compare import (
    load_critic_from_ckpt,
    normalize_proprio,
    resolve_rl_model_paths,
)


def test_offline_rl_loop_checkpoint_includes_actor_critic_metadata(tmp_path):
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()
    cfg.offline_rl.num_gradient_steps = 2
    cfg.offline_rl.save_every = 1

    offline_rl_loop(algorithm, cfg, fill_buffer(), save_dir=str(tmp_path), metadata={"tag": "unit-test"})

    ckpt = torch.load(tmp_path / "rl_checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["metadata"]["tag"] == "unit-test"
    assert ckpt["metadata"]["actor"]["hidden_dim"] == cfg.actor.hidden_dim
    assert ckpt["metadata"]["actor"]["activation"] == cfg.actor.activation
    assert ckpt["metadata"]["critic"]["hidden_dim"] == cfg.critic.hidden_dim
    assert ckpt["metadata"]["critic"]["activation"] == cfg.critic.activation


def test_resolve_rl_model_paths_prefers_checkpoint_sibling_metrics(tmp_path):
    ac_ckpt = tmp_path / "rl_checkpoint.pt"
    ac_ckpt.write_bytes(b"checkpoint")
    sibling_metrics = tmp_path / "metrics.json"
    sibling_metrics.write_text("{}")
    rl_token = tmp_path / "demo_adapt_checkpoint.pt"
    rl_token.write_bytes(b"rl")
    config_path = tmp_path / "pi05_rlt.yaml"
    config_path.write_text("seed: 0\n")

    args = Namespace(
        no_rl=False,
        rl_token_ckpt=str(rl_token),
        ac_ckpt=str(ac_ckpt),
        rl_config=str(config_path),
        ac_metrics="",
        hf_repo="unused",
        rl_vla_model="unused-vla",
        rl_token_path_in_repo="",
        ac_path_in_repo="",
        rl_config_path_in_repo="",
        ac_metrics_path_in_repo="",
    )

    paths = resolve_rl_model_paths(args)
    assert paths.metrics_path == str(sibling_metrics)


def test_load_critic_from_ckpt_uses_checkpoint_metadata_activation():
    cfg = RLTConfig()
    cfg.critic.activation = "relu"
    state_dim = 7
    chunk_dim = 6
    critic = TwinCritic(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
        hidden_dim=8,
        num_layers=2,
        activation="silu",
        layer_norm=False,
        residual=True,
    )
    ckpt = {
        "critic_state_dict": critic.state_dict(),
        "metadata": {
            "critic": {
                "hidden_dim": 8,
                "num_layers": 2,
                "activation": "silu",
                "layer_norm": False,
                "residual": True,
            }
        },
    }

    loaded = load_critic_from_ckpt(ckpt, cfg, state_dim, chunk_dim, "cpu")
    assert loaded.q1.net.blocks[0][1].__class__.__name__ == "SiLU"


def test_normalize_proprio_matches_training_range():
    proprio = torch.tensor([[2.0, 5.0, 12.0]], dtype=torch.float32)
    q01 = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float32)
    q99 = torch.tensor([4.0, 9.0, 14.0], dtype=torch.float32)

    normalized = normalize_proprio(proprio, q01, q99)

    expected = torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(normalized, expected)
