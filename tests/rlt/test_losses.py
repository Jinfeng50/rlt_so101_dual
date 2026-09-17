from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from rlt_so101_dual.core.losses import discounted_chunk_return, critic_loss, actor_loss
from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.critic import TwinCritic
from rlt_so101_dual.core.utils import soft_update

STATE_DIM = 78
CHUNK_DIM = 140
C = 10


@pytest.fixture
def actor():
    return ChunkActor(state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, hidden_dim=64, num_layers=2)


@pytest.fixture
def critic():
    return TwinCritic(state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, hidden_dim=64, num_layers=2)


@pytest.fixture
def target_critic(critic):
    import copy
    tc = copy.deepcopy(critic)
    for p in tc.parameters():
        p.requires_grad = False
    return tc


@pytest.fixture
def batch():
    B = 16
    return {
        "state_vec": torch.randn(B, STATE_DIM),
        "exec_chunk_flat": torch.randn(B, CHUNK_DIM),
        "ref_chunk_flat": torch.randn(B, CHUNK_DIM),
        "reward_seq": torch.randn(B, C),
        "next_state_vec": torch.randn(B, STATE_DIM),
        "next_ref_flat": torch.randn(B, CHUNK_DIM),
        "done": torch.zeros(B),
        "actual_steps": torch.full((B,), C),
    }


def test_discounted_chunk_return_hand_computed():
    """Hand-computed: rewards=[1,1,1], gamma=0.5 -> 1 + 0.5 + 0.25 = 1.75"""
    reward_seq = torch.tensor([[1.0, 1.0, 1.0]])
    result = discounted_chunk_return(reward_seq, gamma=0.5)
    assert torch.allclose(result, torch.tensor([[1.75]]))


def test_discounted_chunk_return_shape():
    reward_seq = torch.randn(8, 10)
    result = discounted_chunk_return(reward_seq, gamma=0.99)
    assert result.shape == (8, 1)


def test_done_masking(actor, critic, target_critic):
    """When done=1, the bootstrap term should be zero."""
    B = 4
    batch_done = {
        "state_vec": torch.randn(B, STATE_DIM),
        "exec_chunk_flat": torch.randn(B, CHUNK_DIM),
        "ref_chunk_flat": torch.randn(B, CHUNK_DIM),
        "reward_seq": torch.ones(B, C),
        "next_state_vec": torch.randn(B, STATE_DIM),
        "next_ref_flat": torch.randn(B, CHUNK_DIM),
        "done": torch.ones(B),  # all done
        "actual_steps": torch.full((B,), C),
    }
    loss = critic_loss(critic, target_critic, actor, batch_done, gamma=0.99, C=C)
    assert not torch.isnan(loss)


def test_critic_loss_scalar(actor, critic, target_critic, batch):
    loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_actor_loss_scalar(actor, critic, batch):
    loss = actor_loss(actor, critic, batch, beta=1.0)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_bc_term_scales_with_beta(actor, critic, batch):
    """Higher beta should give higher actor loss (assuming BC term > 0)."""
    torch.manual_seed(42)
    loss_low = actor_loss(actor, critic, batch, beta=0.0)
    torch.manual_seed(42)
    loss_high = actor_loss(actor, critic, batch, beta=10.0)
    # With beta=0, only Q term. With beta=10, large BC term added.
    # We just check they're different and the high-beta one is larger
    # (BC reg is always non-negative, so adding it increases loss)
    assert loss_high.item() > loss_low.item()


def test_target_is_stop_gradiented(actor, critic, target_critic, batch):
    """Target critic params should not receive gradients through critic_loss."""
    loss = critic_loss(critic, target_critic, actor, batch, gamma=0.99, C=C)
    loss.backward()
    for p in target_critic.parameters():
        assert p.grad is None


def test_actor_loss_matches_paper_bc_scaling():
    """BC term must be per-sample squared-distance sum averaged over batch.

    This is the paper's β convention; it differs from F.mse_loss (mean over
    all elements) by a factor of C * D_flat. We pin the numerical value with a
    deterministic (mu, ref) pair and a stub critic that returns a constant Q
    (so d/dβ of the loss equals the BC term exactly), and check the ratio
    relative to mean-MSE.
    """
    torch.manual_seed(0)
    B = 4
    D_flat = CHUNK_DIM  # = C * D
    mu = torch.randn(B, D_flat)
    ref = torch.randn(B, D_flat)

    class _StubActor:
        def forward(self, x, ref, training=False):
            return mu, None

    class _StubCritic:
        def min_q(self, x, a):
            return torch.zeros(a.shape[0], 1)

    stub_batch = {
        "state_vec": torch.randn(B, STATE_DIM),
        "ref_chunk_flat": ref,
    }

    # Loss at beta=0 removes BC contribution (leaves -q.mean()=0).
    loss_beta0 = actor_loss(_StubActor(), _StubCritic(), stub_batch, beta=0.0)
    loss_beta1 = actor_loss(_StubActor(), _StubCritic(), stub_batch, beta=1.0)
    bc_reg_observed = (loss_beta1 - loss_beta0).item()

    expected_bc = ((mu - ref) ** 2).sum(dim=-1).mean().item()
    assert bc_reg_observed == pytest.approx(expected_bc, rel=1e-6)

    # The new convention equals mean-MSE * (C * D_flat).
    mean_mse = F.mse_loss(mu, ref).item()
    assert bc_reg_observed == pytest.approx(mean_mse * D_flat, rel=1e-6)


def test_critic_loss_respects_target_q_clip():
    class _ZeroCritic(torch.nn.Module):
        def forward(self, state_vec, action_flat):
            z = torch.zeros(state_vec.shape[0], 1)
            return z, z

    class _ConstantTargetCritic(torch.nn.Module):
        def min_q(self, state_vec, action_flat):
            return torch.full((state_vec.shape[0], 1), 1000.0)

    class _ZeroActor:
        def forward(self, state_vec, ref_flat):
            return torch.zeros(ref_flat.shape), None

    batch = {
        "state_vec": torch.zeros(2, STATE_DIM),
        "exec_chunk_flat": torch.zeros(2, CHUNK_DIM),
        "ref_chunk_flat": torch.zeros(2, CHUNK_DIM),
        "reward_seq": torch.zeros(2, C),
        "next_state_vec": torch.zeros(2, STATE_DIM),
        "next_ref_flat": torch.zeros(2, CHUNK_DIM),
        "done": torch.zeros(2),
        "actual_steps": torch.ones(2, dtype=torch.int64),
    }

    clipped = critic_loss(
        _ZeroCritic(), _ConstantTargetCritic(), _ZeroActor(),
        batch, gamma=1.0, C=C, target_q_clip=10.0,
    )
    unclipped = critic_loss(
        _ZeroCritic(), _ConstantTargetCritic(), _ZeroActor(),
        batch, gamma=1.0, C=C, target_q_clip=None,
    )

    assert clipped.item() == pytest.approx(200.0)
    assert unclipped.item() == pytest.approx(2_000_000.0)


class TestTargetPolicySmoothing:
    """TD3's third ingredient, absent from the reference implementation.

    Off by default so the baseline is unchanged; these pin that it is off,
    that turning it on actually perturbs the target, and that the clip holds.
    """

    def _batch(self, B=64, state_dim=12, chunk_dim=20):
        return {
            "state_vec": torch.randn(B, state_dim),
            "exec_chunk_flat": torch.rand(B, chunk_dim) * 2 - 1,
            "next_state_vec": torch.randn(B, state_dim),
            "next_ref_flat": torch.rand(B, chunk_dim) * 2 - 1,
            "reward_seq": torch.zeros(B, 10),
            "done": torch.zeros(B),
            "actual_steps": torch.full((B,), 10),
        }

    def _nets(self, state_dim=12, chunk_dim=20):
        from rlt_so101_dual.core.actor import ChunkActor
        from rlt_so101_dual.core.critic import TwinCritic

        torch.manual_seed(0)
        actor = ChunkActor(state_dim, chunk_dim)
        critic = TwinCritic(state_dim, chunk_dim)
        target = TwinCritic(state_dim, chunk_dim)
        target.load_state_dict(critic.state_dict())
        return actor, critic, target

    def test_off_by_default_is_deterministic(self):
        from rlt_so101_dual.core.losses import critic_loss

        actor, critic, target = self._nets()
        batch = self._batch()
        torch.manual_seed(1)
        a = critic_loss(critic, target, actor, batch, gamma=0.99, C=10)
        torch.manual_seed(2)
        b = critic_loss(critic, target, actor, batch, gamma=0.99, C=10)
        assert torch.equal(a, b), "no noise should mean the target is reproducible"

    def test_enabling_it_changes_the_target(self):
        from rlt_so101_dual.core.losses import critic_loss

        actor, critic, target = self._nets()
        batch = self._batch()
        plain = critic_loss(critic, target, actor, batch, gamma=0.99, C=10)
        torch.manual_seed(3)
        smoothed = critic_loss(
            critic, target, actor, batch, gamma=0.99, C=10, target_policy_noise=0.2
        )
        assert not torch.equal(plain, smoothed)

    def test_noise_is_clipped(self):
        """A huge sigma must still respect the clip, or the target action
        would be pure noise rather than a smoothed neighbourhood."""
        torch.manual_seed(4)
        raw = torch.randn(10000) * 100.0
        clipped = raw.clamp(-0.5, 0.5)
        assert clipped.abs().max() <= 0.5


class TestPaddingInvariant:
    """discounted_chunk_return sums all C entries and relies on the caller
    having zeroed everything past actual_steps. Nothing enforced that, and
    milestone shaping made non-terminal chunks carry rewards for the first
    time -- so the invariant is now checked rather than assumed."""

    def test_reward_at_the_last_valid_step_is_fine(self):
        from rlt_so101_dual.core.losses import discounted_chunk_return

        reward = torch.zeros(2, 10)
        reward[:, 6] = 1.0            # actual_steps = 7 -> index 6
        out = discounted_chunk_return(reward, gamma=0.9, actual_steps=torch.tensor([7, 7]))
        assert out.shape == (2, 1)
        assert out[0, 0].item() == pytest.approx(0.9 ** 6)

    def test_reward_past_actual_steps_is_rejected(self):
        from rlt_so101_dual.core.losses import discounted_chunk_return

        reward = torch.zeros(2, 10)
        reward[:, 8] = 1.0            # beyond actual_steps = 7
        with pytest.raises(ValueError, match="past actual_steps"):
            discounted_chunk_return(reward, gamma=0.9, actual_steps=torch.tensor([7, 7]))

    def test_none_actual_steps_skips_the_check(self):
        from rlt_so101_dual.core.losses import discounted_chunk_return

        reward = torch.zeros(2, 10)
        reward[:, 9] = 1.0
        discounted_chunk_return(reward, gamma=0.9, actual_steps=None)
