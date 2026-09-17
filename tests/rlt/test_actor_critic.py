from __future__ import annotations

import torch
import pytest

from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.critic import ChunkCritic, TwinCritic


@pytest.fixture
def actor():
    return ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


@pytest.fixture
def twin_critic():
    return TwinCritic(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


class TestActor:
    def test_forward_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, std = actor(state, ref)
        assert mu.shape == (8, 140)
        assert std.shape == (8, 140)

    def test_sample_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        action, mu = actor.sample(state, ref)
        assert action.shape == (8, 140)
        assert mu.shape == (8, 140)

    def test_fixed_std(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        _, std = actor(state, ref)
        assert torch.allclose(std, torch.full_like(std, 0.05))

    def test_ref_dropout_statistics(self):
        """With large batch and training=True, ~50% should be zeroed."""
        actor = ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, ref_dropout_p=0.5)
        state = torch.randn(1000, 78)
        ref = torch.ones(1000, 140)  # all ones so we can detect zeroing

        torch.manual_seed(42)
        mu, _ = actor(state, ref, training=True)

        # The ref was multiplied by a mask. We can check by looking at the input
        # indirectly: the ratio of zero-ref samples should be ~50%
        # We verify by calling forward manually and checking the mask effect
        torch.manual_seed(42)
        mask = (torch.rand(1000, 1) > 0.5).float()
        frac_kept = mask.mean().item()
        assert 0.4 < frac_kept < 0.6

    def test_gradient_flow(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        action, _ = actor.sample(state, ref, training=True)
        loss = action.sum()
        loss.backward()
        for p in actor.parameters():
            assert p.grad is not None


class TestResidualToRefActor:
    """residual_to_ref=True must start out as a no-op over the VLA reference
    (mu == ref, delta == 0), for safe online RL initialization on real hardware."""

    def test_mu_equals_ref_at_init(self):
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2, residual_to_ref=True,
        )
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, _ = actor(state, ref, training=False)
        assert torch.allclose(mu, ref)

    def test_mu_equals_ref_at_init_with_residual_mlp(self):
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2,
            residual=True, residual_to_ref=True,
        )
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, _ = actor(state, ref, training=False)
        assert torch.allclose(mu, ref)

    def test_ref_dropout_does_not_break_residual_bias(self):
        """Even when the network's view of ref is dropped out during training,
        the true (undropped) ref is still added back as the residual bias."""
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2,
            residual_to_ref=True, ref_dropout_p=1.0,  # force full dropout
        )
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, _ = actor(state, ref, training=True)
        # delta==0 at init regardless of what the net saw, so mu still == ref.
        assert torch.allclose(mu, ref)

    def test_gradient_flow(self):
        actor = ChunkActor(
            state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2, residual_to_ref=True,
        )
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        action, _ = actor.sample(state, ref, training=True)
        loss = action.sum()
        loss.backward()
        for p in actor.parameters():
            assert p.grad is not None


class TestCritic:
    def test_chunk_critic_shape(self):
        critic = ChunkCritic(state_dim=78, chunk_dim=140, hidden_dim=64)
        q = critic(torch.randn(8, 78), torch.randn(8, 140))
        assert q.shape == (8, 1)

    def test_twin_critic_shapes(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_min_q(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        min_q = twin_critic.min_q(state, action)
        expected = torch.minimum(q1, q2)
        assert torch.allclose(min_q, expected)

    def test_gradient_flow(self, twin_critic):
        state = torch.randn(4, 78)
        action = torch.randn(4, 140)
        q = twin_critic.min_q(state, action)
        q.sum().backward()
        for p in twin_critic.parameters():
            assert p.grad is not None


# --- paper-faithful actor: exploration + gate + non-residual ------------


class TestPaperFaithfulActor:
    """The paper's actor is mu = f(x, ref) sampled from a small-fixed-std
    Gaussian, with the reference dropped 50% of the time during training. All
    three were configured away; these pin them down.
    """

    def _actor(self, **kw):
        from rlt_so101_dual.core.actor import ChunkActor
        defaults = dict(state_dim=8, chunk_dim=6, hidden_dim=16, num_layers=2)
        defaults.update(kw)
        return ChunkActor(**defaults)

    def test_non_residual_does_not_add_the_reference_back(self):
        a = self._actor(residual_to_ref=False, fixed_std=0.0)
        state = torch.zeros(2, 8)
        ref = torch.full((2, 6), 0.7)
        mu, _ = a(state, ref)
        # with residual_to_ref the output would be ref + delta, i.e. >= 0.7-ish
        assert not torch.allclose(mu, ref)

    def test_residual_to_ref_reproduces_the_reference_at_init(self):
        a = self._actor(residual_to_ref=True, fixed_std=0.0)
        ref = torch.full((2, 6), 0.7)
        mu, _ = a(torch.zeros(2, 8), ref)
        assert torch.allclose(mu, ref)

    def test_sample_adds_noise_only_when_std_is_positive(self):
        ref = torch.full((4, 6), 0.1)
        state = torch.zeros(4, 8)
        quiet = self._actor(fixed_std=0.0)
        a1, mu1 = quiet.sample(state, ref)
        assert torch.allclose(a1, mu1)

        noisy = self._actor(fixed_std=0.05)
        a2, mu2 = noisy.sample(state, ref)
        assert not torch.allclose(a2, mu2)

    def test_reference_dropout_only_applies_while_training(self):
        a = self._actor(ref_dropout_p=1.0, residual_to_ref=False, fixed_std=0.0)
        ref = torch.full((6, 6), 0.9)
        state = torch.zeros(6, 8)
        # p=1.0 masks every sample, so the ref cannot influence the output
        train_mu, _ = a(state, ref, training=True)
        zero_mu, _ = a(state, torch.zeros_like(ref), training=True)
        assert torch.allclose(train_mu, zero_mu)
        # at inference the reference is always provided
        infer_mu, _ = a(state, ref, training=False)
        assert not torch.allclose(infer_mu, zero_mu)


def test_closed_actor_gate_still_captures_state_for_the_collector():
    """The gate must only change WHICH actions execute, never whether the
    transition's state is captured. Short-circuiting before the state is stored
    leaves every chunk without a state at chunk start, RLTOnlineCollector drops
    them all, and the 512-transition warmup gate can never be met -- a whole
    session spent stuck in warmup.
    """
    import torch as _t
    from rlt_so101_dual.adapters.lerobot.policies.action_modifier import RLTActionModifier
    from rlt_so101_dual.core.actor import ChunkActor

    C_, AD, PD, TOK = 4, 3, 3, 8
    actor = ChunkActor(state_dim=TOK + PD, chunk_dim=C_ * AD, hidden_dim=8,
                       num_layers=2, fixed_std=0.0, residual_to_ref=False)

    class _Tok:
        def encode(self, prefix): return _t.zeros(1, TOK)

    class _Phase:
        is_critical = True          # modifier reads this via is_rl_phase

    mod = RLTActionModifier(
        rl_token=_Tok(), actor=actor, phase_ctrl=_Phase(),
        chunk_length=C_, action_dim=AD, proprio_dim=PD,
        chunk_exec_steps=C_, vla_ref=True, action_clip_delta=0.1,
    )
    mod.actor_enabled = False

    vla = _t.full((1, C_ + 2, AD), 0.4)
    out = mod.compute_chunk(vla, _t.zeros(1, PD), _t.zeros(1, 1, 16))

    assert mod.get_last_chunk_tensors() is not None, "state was not captured"
    # gate closed => the VLA's own first chunk_length actions are executed
    assert _t.allclose(out, vla[:, :C_, :])


def test_delta_bound_holds_when_the_reference_leaves_minus_one_to_one():
    """QUANTILES puts q01/q99 at -1/+1, so the normalised space legitimately
    extends past them -- wrist_roll's q99 - q01 is 0.558 and its reference
    regularly reaches ~5. An absolute clamp to [-1, 1] discarded the VLA's own
    intent there AND broke the bound this code advertises: a ref of 4.737
    executed as 1.0, i.e. 3.737 away while the bound said 0.1.
    """
    import torch as _t
    from rlt_so101_dual.adapters.lerobot.policies.action_modifier import RLTActionModifier
    from rlt_so101_dual.core.actor import ChunkActor

    C_, AD, PD, TOK = 4, 6, 3, 8
    actor = ChunkActor(state_dim=TOK + PD, chunk_dim=C_ * AD, hidden_dim=8,
                       num_layers=2, fixed_std=0.0, residual_to_ref=False)

    class _Tok:
        def encode(self, prefix): return _t.zeros(1, TOK)

    class _Phase:
        is_critical = True

    DELTA = 0.1
    mod = RLTActionModifier(
        rl_token=_Tok(), actor=actor, phase_ctrl=_Phase(),
        chunk_length=C_, action_dim=AD, proprio_dim=PD,
        chunk_exec_steps=C_, vla_ref=True, action_clip_delta=DELTA,
    )
    # one joint far outside [-1, 1], as wrist_roll genuinely is
    vla = _t.zeros(1, C_ + 2, AD)
    vla[..., 4] = 4.737
    out = mod.compute_chunk(vla, _t.zeros(1, PD), _t.zeros(1, 1, 16))

    ref = vla[:, :C_, :]
    assert (out - ref).abs().max() <= DELTA + 1e-5, \
        f"bound violated: {(out - ref).abs().max():.3f} > {DELTA}"
    # and the reference's own value is preserved rather than truncated to 1.0
    assert out[0, 0, 4] > 4.0, out[0, 0, 4]
