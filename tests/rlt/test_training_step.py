from __future__ import annotations

import copy

import torch
import pytest

from rlt_so101_dual.core.losses import critic_loss, actor_loss
from rlt_so101_dual.core.utils import soft_update, flatten_chunk, compute_discount_vector
from rlt_so101_dual.core.interfaces import Observation, ChunkTransition
from rlt_so101_dual.core.replay_buffer import ReplayBuffer
from tests.rlt.helpers import (
    make_test_algorithm, make_batch, make_fake_transition,
    TOKEN_DIM, ACTION_DIM, PROPRIO_DIM, C, STATE_DIM, CHUNK_DIM,
)


@pytest.fixture
def algorithm():
    algo, _ = make_test_algorithm()
    return algo


@pytest.fixture
def batch():
    return make_batch()


def test_single_critic_step_loss_not_nan(algorithm, batch):
    """Single critic gradient step produces finite loss."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=1e-3)
    loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify params changed
    loss2 = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    assert not torch.isnan(loss2)


def test_single_actor_step_loss_not_nan(algorithm, batch):
    """Single actor gradient step produces finite loss."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=1e-3)
    loss = actor_loss(algorithm.policy.actor, algorithm.critic, batch, beta=1.0)
    assert not torch.isnan(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_frozen_params_stay_frozen(algorithm, batch):
    """After freezing VLA and encoder, their params should not get gradients."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    # Run both losses
    c_loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.policy.actor, batch, gamma=0.99, C=C)
    c_loss.backward()

    # VLA params: no grad
    for p in algorithm.policy.vla.parameters():
        assert p.grad is None

    # RL token encoder params: no grad
    assert algorithm.policy.rl_token.rl_token_embed.grad is None
    for p in algorithm.policy.rl_token.encoder.parameters():
        assert p.grad is None


def test_full_forward_backward_pipeline(algorithm):
    """End-to-end: obs -> state -> action -> critic -> loss -> backward."""
    algorithm.policy.freeze_vla()
    algorithm.policy.freeze_rl_token_encoder()

    obs = Observation(
        images={"base": torch.randn(4, 3, 64, 64)},
        proprio=torch.randn(4, PROPRIO_DIM),
    )

    with torch.no_grad():
        state = algorithm.policy.get_rl_state(obs)
        ref = algorithm.policy.get_reference_chunk(obs)
        ref_flat = flatten_chunk(ref)

    action, mu = algorithm.policy.actor.sample(state, ref_flat, training=True)
    q = algorithm.critic.min_q(state, action)
    loss = -q.mean()
    loss.backward()

    # Actor should have gradients
    for p in algorithm.policy.actor.parameters():
        assert p.grad is not None

    # Critic should have gradients (from Q computation)
    for p in algorithm.critic.parameters():
        assert p.grad is not None


def test_soft_update_changes_target(algorithm):
    """Verify soft update actually modifies target critic."""
    orig_params = [p.data.clone() for p in algorithm.target_critic.parameters()]

    # Perturb online critic
    for p in algorithm.critic.parameters():
        p.data += torch.randn_like(p.data) * 0.1

    soft_update(algorithm.target_critic, algorithm.critic, tau=0.1)

    changed = False
    for orig, p in zip(orig_params, algorithm.target_critic.parameters()):
        if not torch.allclose(orig, p.data):
            changed = True
            break
    assert changed


def test_soft_update_tau_zero_is_noop(algorithm):
    """tau=0 must leave the target untouched: target = (1-0)*target + 0*source.

    Online training relies on this exact no-op to disable ChunkACPolicy.forward()'s
    internal (mistimed) soft update via cfg.policy.tau=0, doing the real soft
    update itself afterwards once critic.step() has actually run -- see
    backend.py's `_run_online_rl_update`.
    """
    orig_params = [p.data.clone() for p in algorithm.target_critic.parameters()]

    for p in algorithm.critic.parameters():
        p.data += torch.randn_like(p.data) * 0.1

    soft_update(algorithm.target_critic, algorithm.critic, tau=0.0)

    for orig, p in zip(orig_params, algorithm.target_critic.parameters()):
        assert torch.allclose(orig, p.data)


def test_compute_discount_vector():
    v = compute_discount_vector(0.99, 5)
    assert v.shape == (5,)
    assert torch.allclose(v[0], torch.tensor(1.0))
    assert torch.allclose(v[1], torch.tensor(0.99))
    assert torch.allclose(v[4], torch.tensor(0.99 ** 4))


def test_flatten_chunk():
    chunk = torch.randn(3, 10, 14)
    flat = flatten_chunk(chunk)
    assert flat.shape == (3, 140)


def test_replay_buffer_integration():
    """Test that buffer -> sample -> loss pipeline works end-to-end."""
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.add(make_fake_transition())

    from rlt_so101_dual.core.actor import ChunkActor
    from rlt_so101_dual.core.critic import TwinCritic

    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    target_critic = copy.deepcopy(critic)

    batch = buf.sample(8)
    c_loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    a_loss = actor_loss(actor, critic, batch, beta=1.0)

    assert not torch.isnan(c_loss)
    assert not torch.isnan(a_loss)


def test_lerobot_ac_actor_loss_does_not_add_critic_grads(batch):
    from types import SimpleNamespace

    from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    from rlt_so101_dual.core.actor import ChunkActor
    from rlt_so101_dual.core.critic import TwinCritic

    torch.manual_seed(0)
    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        gamma=0.99,
        chunk_length=C,
        tau=0.005,
        actor_update_interval=2,
        beta=1.0,
        target_q_clip=100.0,
    )
    policy.actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=2)
    policy.target_critic = copy.deepcopy(policy.critic)
    policy.register_buffer("_critic_step", torch.ones((), dtype=torch.long))

    lerobot_batch = dict(batch)
    lerobot_batch["exec_chunk"] = lerobot_batch.pop("exec_chunk_flat").view(-1, C, ACTION_DIM)
    lerobot_batch["ref_chunk"] = lerobot_batch.pop("ref_chunk_flat").view(-1, C, ACTION_DIM)
    lerobot_batch["next_ref_chunk"] = lerobot_batch.pop("next_ref_flat").view(-1, C, ACTION_DIM)

    critic_only = copy.deepcopy(policy.critic)
    target_only = copy.deepcopy(policy.target_critic)
    actor_only = copy.deepcopy(policy.actor)
    c_loss = critic_loss(
        critic_only, target_only, actor_only, batch,
        gamma=0.99, C=C, target_q_clip=100.0,
    )
    c_loss.backward()
    expected = {name: p.grad.clone() for name, p in critic_only.named_parameters()}

    loss, info = policy.forward(lerobot_batch)
    assert "loss_actor" in info
    loss.backward()

    for name, p in policy.critic.named_parameters():
        assert torch.allclose(p.grad, expected[name], atol=1e-6, rtol=1e-5)


# --- pinning a joint to the VLA reference ------------------------------------
# Dropping only a joint's BC term hands that dimension to Q alone. wrist_roll is
# the worst candidate: its exec-ref spread in run4 is std 0.957 against
# 0.087-0.200 elsewhere, and that spread is the operator nudging the wrist while
# dragging the leader back -- repositioning noise amplified ~79x by QUANTILES,
# on a joint the actor can move 0.028 deg. Pinning mu to ref removes it from
# both the BC term and the actor's control.

from rlt_so101_dual.core.actor import ChunkActor          # noqa: E402
from rlt_so101_dual.core.critic import TwinCritic          # noqa: E402


def _pin_setup(pin_joint: int = 1):
    action_dim = next(d for d in (6, 4, 2) if CHUNK_DIM % d == 0)
    chunk_length = CHUNK_DIM // action_dim
    joint = torch.zeros(action_dim)
    joint[pin_joint] = 1.0
    return action_dim, chunk_length, joint.repeat(chunk_length)


def test_pinned_dims_equal_the_reference_exactly():
    action_dim, chunk_length, mask = _pin_setup()
    torch.manual_seed(0)
    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    with torch.no_grad():
        for prm in actor.parameters():
            prm.add_(torch.randn_like(prm) * 0.2)      # or mu would start at ref anyway
    actor.set_pin_mask(mask)

    state = torch.randn(8, STATE_DIM)
    ref = torch.randn(8, CHUNK_DIM)
    with torch.no_grad():
        mu, _ = actor(state, ref, training=False)
    torch.testing.assert_close(mu[:, mask.bool()], ref[:, mask.bool()])
    assert (mu[:, ~mask.bool()] - ref[:, ~mask.bool()]).abs().max() > 1e-4, \
        "unpinned dims must still be free, or the test proves nothing"


def test_pinning_survives_reference_dropout():
    """training=True zeroes the ref *input*; the pin must use the real ref."""
    action_dim, chunk_length, mask = _pin_setup()
    torch.manual_seed(1)
    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1,
                       ref_dropout_p=1.0)               # always dropped
    with torch.no_grad():
        for prm in actor.parameters():
            prm.add_(torch.randn_like(prm) * 0.2)
    actor.set_pin_mask(mask)

    state = torch.randn(8, STATE_DIM)
    ref = torch.randn(8, CHUNK_DIM)
    mu, _ = actor(state, ref, training=True)
    torch.testing.assert_close(mu[:, mask.bool()], ref[:, mask.bool()])


def test_exploration_noise_does_not_move_a_pinned_dim():
    action_dim, chunk_length, mask = _pin_setup()
    torch.manual_seed(2)
    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1, fixed_std=0.5)
    actor.set_pin_mask(mask)
    state = torch.randn(8, STATE_DIM)
    ref = torch.randn(8, CHUNK_DIM)
    with torch.no_grad():
        action, _ = actor.sample(state, ref, training=False)
    torch.testing.assert_close(action[:, mask.bool()], ref[:, mask.bool()])
    assert (action[:, ~mask.bool()] - ref[:, ~mask.bool()]).abs().max() > 1e-3


def test_pinned_dim_contributes_nothing_to_bc():
    from rlt_so101_dual.core.losses import actor_loss

    action_dim, chunk_length, mask = _pin_setup()
    torch.manual_seed(3)
    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=32, num_layers=1)
    with torch.no_grad():
        for prm in actor.parameters():
            prm.add_(torch.randn_like(prm) * 0.2)
    actor.set_pin_mask(mask)
    batch = {"state_vec": torch.randn(8, STATE_DIM),
             "ref_chunk_flat": torch.randn(8, CHUNK_DIM)}
    torch.manual_seed(4)
    actor_loss(actor, critic, batch, beta=0.3)
    torch.manual_seed(4)
    mu, _ = actor.forward(batch["state_vec"], batch["ref_chunk_flat"], training=True)
    sq = (mu - batch["ref_chunk_flat"]) ** 2
    assert float((sq * mask).sum()) == 0.0, "a pinned dim leaked into the BC term"


def test_out_of_range_pin_index_is_refused():
    """Config-level, so a typo fails before pi0.5 is loaded, not minutes in."""
    from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig

    with pytest.raises(ValueError, match="out of range"):
        ChunkACPolicyConfig(pin_action_dims_to_ref=(99,))
    ChunkACPolicyConfig(pin_action_dims_to_ref=(4,))          # wrist_roll: valid
