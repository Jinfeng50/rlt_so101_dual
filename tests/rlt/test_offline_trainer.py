from __future__ import annotations

import math

import pytest
import torch

from rlt_so101_dual.core.trainer import offline_rl_loop
from tests.rlt.helpers import make_test_algorithm, fill_buffer


def test_offline_rl_loop_runs():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()
    buf = fill_buffer()

    metrics = offline_rl_loop(algorithm, cfg, buf)

    expected_critic = cfg.offline_rl.num_gradient_steps * cfg.training.utd_ratio
    assert len(metrics.critic_losses) == expected_critic
    assert len(metrics.actor_losses) > 0
    assert all(not math.isnan(l) for l in metrics.critic_losses)
    assert all(not math.isnan(l) for l in metrics.actor_losses)


def test_offline_rl_loop_frozen_params():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    vla_params_before = {n: p.data.clone() for n, p in algorithm.policy.vla.named_parameters()}
    enc_params_before = {n: p.data.clone() for n, p in algorithm.policy.rl_token.encoder.named_parameters()}
    rl_embed_before = algorithm.policy.rl_token.rl_token_embed.data.clone()

    buf = fill_buffer()
    offline_rl_loop(algorithm, cfg, buf)

    for name, p in algorithm.policy.vla.named_parameters():
        assert torch.equal(p.data, vla_params_before[name]), f"VLA param {name} changed"
    for name, p in algorithm.policy.rl_token.encoder.named_parameters():
        assert torch.equal(p.data, enc_params_before[name]), f"Encoder param {name} changed"
    assert torch.equal(algorithm.policy.rl_token.rl_token_embed.data, rl_embed_before)


def test_offline_rl_loop_actor_critic_update():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    actor_params_before = {n: p.data.clone() for n, p in algorithm.policy.actor.named_parameters()}
    critic_params_before = {n: p.data.clone() for n, p in algorithm.critic.named_parameters()}

    buf = fill_buffer()
    offline_rl_loop(algorithm, cfg, buf)

    actor_changed = any(
        not torch.equal(p.data, actor_params_before[n])
        for n, p in algorithm.policy.actor.named_parameters()
    )
    critic_changed = any(
        not torch.equal(p.data, critic_params_before[n])
        for n, p in algorithm.critic.named_parameters()
    )
    assert actor_changed, "Actor params did not change after training"
    assert critic_changed, "Critic params did not change after training"


def test_offline_rl_loop_with_val_buffer():
    algorithm, cfg = make_test_algorithm()
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()
    train_buf = fill_buffer()
    val_buf = fill_buffer(20)

    metrics = offline_rl_loop(algorithm, cfg, train_buf, val_buffer=val_buf)

    expected_critic = cfg.offline_rl.num_gradient_steps * cfg.training.utd_ratio
    assert len(metrics.critic_losses) == expected_critic


@pytest.mark.parametrize("utd", [1, 5])
def test_offline_rl_loop_honors_utd_ratio(utd):
    """With utd=k: critic_update is called k*num_gradient_steps times but
    soft_update_target is called num_gradient_steps times (bundled per outer
    step, so target rate is independent of UTD). Actor fires every
    `actor_update_interval` critic updates, totaling (k*steps)//interval."""
    algorithm, cfg = make_test_algorithm(num_gradient_steps=4, actor_update_interval=2)
    cfg.training.utd_ratio = utd
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    counts = {"critic": 0, "actor": 0, "soft": 0}

    def _critic(batch, opt, gamma, C):
        counts["critic"] += 1
        return 0.0

    def _actor(batch, opt, beta):
        counts["actor"] += 1
        return 0.0

    def _soft(tau):
        counts["soft"] += 1

    algorithm.critic_update = _critic
    algorithm.actor_update = _actor
    algorithm.soft_update_target = _soft

    buf = fill_buffer()
    offline_rl_loop(algorithm, cfg, buf)

    num_steps = cfg.offline_rl.num_gradient_steps
    actor_interval = cfg.training.actor_update_interval
    assert counts["critic"] == num_steps * utd
    assert counts["soft"] == num_steps
    assert counts["actor"] == (num_steps * utd) // actor_interval
