"""Episode-count boundaries in the online RL schedule.

`recorded_episodes` was the outer loop's 0-based index and the update ran
before it was incremented, so `--warmup-episodes 5` was only satisfied on the
sixth episode. Every downstream boundary -- when critic-only starts and ends,
when the actor unfreezes, when beta starts annealing, the step_NNNNNN
checkpoint names -- keys off the same number, so the boundary is pinned here
rather than left to reading the call chain.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlt_so101_dual.adapters.lerobot.record import online_trainer as online_trainer_module
from rlt_so101_dual.adapters.lerobot.record.online_trainer import OnlineRLTrainer
from rlt_so101_dual.core import shape_contract as sc
from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.critic import TwinCritic
from rlt_so101_dual.core.interfaces import ChunkTransition

STATE_DIM, CHUNK_DIM, C, ACTION_DIM = 24, 12, 4, 3


class _StubPolicy(torch.nn.Module):
    """Enough of ChunkACPolicy for the trainer: real actor/critic parameters
    so 'did the actor optimiser actually step' is a real question, and a
    forward() that honours actor_update_interval the way the real one does."""

    def __init__(self, policy_cfg):
        super().__init__()
        self.cfg = policy_cfg
        self.actor = ChunkActor(STATE_DIM, CHUNK_DIM)
        self.critic = TwinCritic(STATE_DIM, CHUNK_DIM)
        self.target_critic = TwinCritic(STATE_DIM, CHUNK_DIM)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self._critic_step = torch.zeros((), dtype=torch.long)
        self.saved_to: list[str] = []

    def forward(self, batch):
        # The trainer hands over unflattened (B, C, action_dim) chunks and
        # expects the policy to flatten them, same as ChunkACPolicy does.
        self._critic_step += 1
        exec_flat = batch["exec_chunk"].flatten(start_dim=-2)
        ref_flat = batch["ref_chunk"].flatten(start_dim=-2)
        q1, _ = self.critic(batch["state_vec"], exec_flat)
        loss = q1.pow(2).mean()
        # Same gating rule as ChunkACPolicy: the actor only contributes to the
        # loss every actor_update_interval critic steps.
        if int(self._critic_step.item()) % self.cfg.actor_update_interval == 0:
            mu, _ = self.actor(batch["state_vec"], ref_flat, training=True)
            loss = loss + mu.pow(2).mean()
        return loss, {"loss_critic": loss.detach()}

    def save_pretrained(self, path):
        self.saved_to.append(str(path))


def _policy_cfg():
    return SimpleNamespace(
        chunk_length=C, action_dim=ACTION_DIM, beta=0.3, tau=0.005,
        actor_update_interval=2, gamma=0.99, utd_ratio=1,
        # The resume check reads the shape contract and both backbone paths off
        # this config, so the fake has to carry them too -- a fake that omits
        # them would let the check pass on None == None.
        proprio_dim=sc.PROPRIO_DIM, rl_token_dim=sc.RL_TOKEN_DIM,
        vla_pretrained_path="", rl_token_pretrained_path="",
    )


def _online_cfg(tmp_path, **overrides):
    cfg = SimpleNamespace(
        replay_capacity=10_000, lr_actor=1e-3, lr_critic=1e-3,
        milestone_reward=0.3, terminal_reward=1.0, time_decay=0.98,
        warmup_episodes=5, critic_only_episodes=10,
        min_warmup_transitions=0, min_warmup_successes=0, min_warmup_failures=0,
        batch_size=4, utd_ratio=1, max_updates_per_episode=4,
        use_stratified_sampling=False, save_dir=str(tmp_path),
        save_every_episodes=5, wandb=False,
        beta_final=None, beta_anneal_episodes=10,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(autouse=True)
def _stub_action_quantiles(monkeypatch):
    """OnlineRLTrainer now refuses to construct without ACTION q01/q99.

    That check belongs at construction rather than at the first chunk close --
    otherwise the arm has already executed a chunk before the run dies. These
    tests exercise the warmup / critic-only / resume boundaries, not
    normalisation, so hand them stub quantiles.
    """
    monkeypatch.setattr(
        online_trainer_module, "_action_quantiles",
        lambda _path: (torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM)),
    )


def _trainer(tmp_path, **overrides):
    policy_cfg = _policy_cfg()
    policy = _StubPolicy(policy_cfg)
    return OnlineRLTrainer(policy, _online_cfg(tmp_path, **overrides), policy_cfg)


def _transition(episode_id: int, done: bool, success: bool, intervention: bool = False):
    reward = torch.zeros(C)
    if done and success:
        reward[C - 1] = 1.0
    return ChunkTransition(
        state_vec=torch.randn(STATE_DIM), exec_chunk=torch.randn(C, ACTION_DIM),
        ref_chunk=torch.randn(C, ACTION_DIM), reward_seq=reward,
        next_state_vec=torch.randn(STATE_DIM), next_ref_chunk=torch.randn(C, ACTION_DIM),
        done=torch.tensor(float(done)), intervention=torch.tensor(float(intervention)),
        actual_steps=torch.tensor(C), episode_id=torch.tensor(episode_id),
        outcome=torch.tensor(float(success)) if done else torch.tensor(-1.0),
    )


def _fill_episode(trainer, episode_id: int, success: bool, n: int = 4):
    for i in range(n):
        trainer.replay_buffer.add(_transition(episode_id, done=(i == n - 1), success=success))


# --- the boundary itself ---------------------------------------------------


class TestWarmupEpisodeBoundary:
    def test_satisfied_exactly_when_the_nth_episode_completes(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=5)
        _fill_episode(t, 0, success=True)
        assert t.warmup_satisfied(4) is False, "four completed is not yet five"
        assert t.warmup_completed_at_episode is None
        assert t.warmup_satisfied(5) is True, "the fifth completing must satisfy it"
        assert t.warmup_completed_at_episode == 5

    def test_warmup_of_one_is_satisfied_by_the_first_episode(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=1)
        _fill_episode(t, 0, success=True)
        assert t.warmup_satisfied(1) is True

    def test_satisfaction_is_sticky(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=5)
        _fill_episode(t, 0, success=True)
        t.warmup_satisfied(5)
        # Even with the buffer emptied, warmup does not un-satisfy.
        t.replay_buffer.buffer.clear()
        assert t.warmup_satisfied(6) is True
        assert t.warmup_completed_at_episode == 5


class TestWarmupGatesAllBind:
    """The episode count is only one of four gates; the other three must
    still be able to hold warmup back on their own."""

    def test_transition_gate_holds_past_the_episode_count(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=2, min_warmup_transitions=100)
        _fill_episode(t, 0, success=True)
        assert t.warmup_satisfied(5) is False
        assert t.warmup_completed_at_episode is None

    def test_success_gate_holds(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=1, min_warmup_successes=2, min_warmup_failures=0)
        _fill_episode(t, 0, success=True)
        assert t.warmup_satisfied(5) is False
        _fill_episode(t, 1, success=True)
        assert t.warmup_satisfied(5) is True

    def test_failure_gate_holds(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=1, min_warmup_successes=0, min_warmup_failures=2)
        _fill_episode(t, 0, success=False)
        assert t.warmup_satisfied(5) is False
        _fill_episode(t, 1, success=False)
        assert t.warmup_satisfied(5) is True

    def test_all_four_together(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=3, min_warmup_transitions=8,
                     min_warmup_successes=1, min_warmup_failures=1)
        _fill_episode(t, 0, success=True)          # 4 transitions, 1 success
        assert t.warmup_satisfied(3) is False      # transitions and failures short
        _fill_episode(t, 1, success=False)         # 8 transitions, 1 of each
        assert t.warmup_satisfied(2) is False      # episode count short
        assert t.warmup_satisfied(3) is True


# --- critic-only window ----------------------------------------------------


def _actor_snapshot(trainer):
    return [p.detach().clone() for p in trainer.policy.actor.parameters()]


def _actor_moved(trainer, before) -> bool:
    return any(not torch.equal(a, b) for a, b in zip(before, trainer.policy.actor.parameters()))


class TestCriticOnlyWindow:
    """Asserts the actor's parameters actually stay put, rather than only
    checking that actor_update_interval was set -- the interval is the
    mechanism, not the thing that matters."""

    def _run_episode(self, t, completed: int, episode_id: int, success: bool = True):
        before_total = t.replay_buffer.total_added
        _fill_episode(t, episode_id, success=success)
        return t.maybe_update(completed, before_total)

    def test_actor_is_frozen_for_exactly_critic_only_episodes(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=1, critic_only_episodes=3,
                     min_warmup_successes=0, min_warmup_failures=0)
        # Episode 1 satisfies warmup; the window then covers completed 1,2,3.
        frozen_at, moved_at = [], []
        for completed in range(1, 7):
            before = _actor_snapshot(t)
            self._run_episode(t, completed, episode_id=completed)
            (moved_at if _actor_moved(t, before) else frozen_at).append(completed)
        assert frozen_at == [1, 2, 3], f"critic-only should cover exactly 3 cycles, got {frozen_at}"
        assert moved_at[0] == 4, "the actor must unfreeze on the cycle after the window"

    def test_window_is_anchored_to_actual_warmup_not_the_configured_episode(self, tmp_path):
        """Warmup can be held past --warmup-episodes by the other gates; the
        window has to start when it actually completed."""
        t = _trainer(tmp_path, warmup_episodes=2, critic_only_episodes=2,
                     min_warmup_transitions=12)
        for completed in (1, 2, 3):
            self._run_episode(t, completed, episode_id=completed)
        assert t.warmup_completed_at_episode == 3
        before = _actor_snapshot(t)
        self._run_episode(t, 4, episode_id=4)
        assert not _actor_moved(t, before), "still inside the window anchored at 3"
        before = _actor_snapshot(t)
        self._run_episode(t, 5, episode_id=5)
        assert _actor_moved(t, before)


class TestBetaSchedule:
    def test_annealing_starts_after_the_critic_only_window(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=1, critic_only_episodes=3,
                     beta_final=0.1, beta_anneal_episodes=4)
        t.warmup_completed_at_episode = 1
        critic_only_until = 1 + 3
        assert t._scheduled_beta(4, critic_only_until) == pytest.approx(0.3)
        assert t._scheduled_beta(6, critic_only_until) == pytest.approx(0.2)
        assert t._scheduled_beta(8, critic_only_until) == pytest.approx(0.1)


# --- checkpoints and resume ------------------------------------------------


class TestCheckpointNaming:
    def test_no_double_increment_in_step_names(self, tmp_path):
        """maybe_update() used to add 1 to the caller's index for the periodic
        save. The caller now passes the completed count, so adding it again
        would name the fifth episode's checkpoint step_000006."""
        t = _trainer(tmp_path, warmup_episodes=1, save_every_episodes=5,
                     min_warmup_successes=0, min_warmup_failures=0)
        for completed in range(1, 6):
            before_total = t.replay_buffer.total_added
            _fill_episode(t, completed, success=True)
            t.maybe_update(completed, before_total)
        assert t.policy.saved_to, "a periodic checkpoint should have been written"
        assert t.policy.saved_to[-1].endswith("step_000005")

    def test_latest_state_records_the_completed_count(self, tmp_path):
        t = _trainer(tmp_path, save_every_episodes=100)
        t.save_latest_state(7)
        state = torch.load(tmp_path / "latest_online_state.pt", weights_only=False)
        assert state["recorded_episodes"] == 7


class TestResumeBoundaries:
    def test_warmup_anchor_and_window_survive_a_round_trip(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=2, critic_only_episodes=4,
                     min_warmup_successes=0, min_warmup_failures=0)
        for completed in (1, 2, 3):
            before_total = t.replay_buffer.total_added
            _fill_episode(t, completed, success=True)
            t.maybe_update(completed, before_total)
        assert t.warmup_completed_at_episode == 2
        t.save_latest_state(3)

        resumed = _trainer(tmp_path, warmup_episodes=2, critic_only_episodes=4,
                           min_warmup_successes=0, min_warmup_failures=0)
        completed = resumed.load_latest_state(str(tmp_path / "latest_online_state.pt"))
        assert completed == 3, "the loop resumes from the completed count"
        assert resumed.warmup_completed_at_episode == 2, "the anchor must not drift"

        # The window is 2..5; the actor unfreezes on 6 in both.
        window_before = [c for c in range(1, 9) if c < t.warmup_completed_at_episode + 4]
        window_after = [c for c in range(1, 9) if c < resumed.warmup_completed_at_episode + 4]
        assert window_before == window_after == [1, 2, 3, 4, 5]

    def test_resume_does_not_replay_warmup(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=2, min_warmup_successes=0, min_warmup_failures=0)
        _fill_episode(t, 1, success=True)
        t.maybe_update(2, 0)
        t.save_latest_state(2)

        resumed = _trainer(tmp_path, warmup_episodes=2, min_warmup_successes=0, min_warmup_failures=0)
        resumed.load_latest_state(str(tmp_path / "latest_online_state.pt"))
        # Buffer came back with it, so warmup stays satisfied at the same anchor.
        assert resumed.warmup_satisfied(3) is True
        assert resumed.warmup_completed_at_episode == 2


class TestDiscardedEpisodesDoNotAdvance:
    def test_an_episode_with_no_critical_phase_does_not_satisfy_warmup(self, tmp_path):
        """maybe_update() returns before touching warmup when the episode
        added no transitions, so an episode where the operator never entered
        the critical phase cannot set the anchor."""
        t = _trainer(tmp_path, warmup_episodes=1, min_warmup_successes=0, min_warmup_failures=0)
        assert t.maybe_update(1, t.replay_buffer.total_added) is None
        assert t.warmup_completed_at_episode is None

    def test_rolled_back_transitions_leave_the_buffer(self, tmp_path):
        t = _trainer(tmp_path, warmup_episodes=1)
        baseline = t.start_episode(0)
        _fill_episode(t, 0, success=True)
        assert len(t.replay_buffer) == 4
        t.discard_episode(baseline)
        assert len(t.replay_buffer) == 0
        assert t.maybe_update(1, t.replay_buffer.total_added) is None
        assert t.warmup_completed_at_episode is None


# --- resume refuses a snapshot that means something else -----------------


class TestResumeCompatibility:
    """Every field below loads without error if unchecked: actor/critic tensors
    keep their shapes no matter which VLA produced the RL tokens they were
    trained against, and the buffered rewards are already baked at the old
    time_decay. The run would look healthy and optimise a different problem.
    """

    def _saved(self, tmp_path, **policy_overrides):
        trainer = _trainer(tmp_path)
        for key, value in policy_overrides.items():
            setattr(trainer.policy_cfg, key, value)
        trainer.task = "insert the bolt"
        _fill_episode(trainer, 0, success=True)
        trainer.save_latest_state(1)
        return Path(tmp_path) / "latest_online_state.pt"

    def _resume(self, tmp_path, path, *, policy=None, online=None, task="insert the bolt"):
        fresh = _trainer(tmp_path)
        fresh.task = task
        for key, value in (policy or {}).items():
            setattr(fresh.policy_cfg, key, value)
        for key, value in (online or {}).items():
            setattr(fresh.cfg, key, value)
        return fresh.load_latest_state(path)

    def test_identical_config_resumes(self, tmp_path):
        path = self._saved(tmp_path)
        assert self._resume(tmp_path, path) == 1

    def test_changed_gamma_is_refused(self, tmp_path):
        path = self._saved(tmp_path)
        with pytest.raises(ValueError, match="gamma"):
            self._resume(tmp_path, path, policy={"gamma": 0.999})

    def test_changed_time_decay_is_refused(self, tmp_path):
        path = self._saved(tmp_path)
        with pytest.raises(ValueError, match="time_decay"):
            self._resume(tmp_path, path, online={"time_decay": 0.97})

    def test_changed_task_is_refused(self, tmp_path):
        path = self._saved(tmp_path)
        with pytest.raises(ValueError, match="task"):
            self._resume(tmp_path, path, task="insert the other bolt")

    def test_changed_shape_contract_is_refused(self, tmp_path):
        path = self._saved(tmp_path)
        with pytest.raises(ValueError, match="shape_contract"):
            self._resume(tmp_path, path, policy={"proprio_dim": 7})

    def test_different_vla_checkpoint_is_refused(self, tmp_path):
        """Same tensor shapes, different RL-token semantics."""
        ckpt = tmp_path / "vla_a"
        ckpt.mkdir()
        (ckpt / "train_config.json").write_text(
            '{"job_name": "vla_ft", "steps": 70000, "dataset": {"repo_id": "local/bolt"}}'
        )
        other = tmp_path / "vla_b"
        other.mkdir()
        (other / "train_config.json").write_text(
            '{"job_name": "vla_ft", "steps": 50000, "dataset": {"repo_id": "local/bolt"}}'
        )

        path = self._saved(tmp_path, vla_pretrained_path=str(ckpt))
        with pytest.raises(ValueError, match="vla_identity"):
            self._resume(tmp_path, path, policy={"vla_pretrained_path": str(other)})

    def test_snapshot_without_version_is_refused(self, tmp_path):
        path = self._saved(tmp_path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        del state["snapshot_version"]
        torch.save(state, path)

        with pytest.raises(ValueError, match="snapshot_version"):
            self._resume(tmp_path, path)

    def test_advisory_change_warns_but_resumes(self, tmp_path, caplog):
        path = self._saved(tmp_path)
        with caplog.at_level("WARNING"):
            assert self._resume(tmp_path, path, online={"lr_actor": 5e-4}) == 1
        assert "lr_actor" in caplog.text


# --- the actor must not drive until it has been trained ------------------


class _GatePolicy(SimpleNamespace):
    """Minimal stand-in exposing the gate the trainer drives."""
    def __init__(self, cfg):
        super().__init__(config=cfg, gate=[])
        self.actor = ChunkActor(state_dim=STATE_DIM, chunk_dim=C * ACTION_DIM,
                                hidden_dim=8, num_layers=2)
        self.critic = TwinCritic(state_dim=STATE_DIM, chunk_dim=C * ACTION_DIM,
                                 hidden_dim=8, num_layers=2)
        self.target_critic = TwinCritic(state_dim=STATE_DIM, chunk_dim=C * ACTION_DIM,
                                        hidden_dim=8, num_layers=2)
        self._critic_step = 0
    def set_actor_gate(self, enabled):
        self.gate.append(enabled)


def test_actor_gate_closed_at_construction(tmp_path):
    """The paper bootstraps by executing the VLA's own actions. A non-residual
    actor at init outputs zeros, not the reference, so the handover has to be
    explicit rather than implied by a zero-init residual head.
    """
    cfg = _policy_cfg()
    pol = _GatePolicy(cfg)
    OnlineRLTrainer(pol, _online_cfg(tmp_path), cfg)
    assert pol.gate == [False]


def test_actor_gate_opens_exactly_when_the_critic_only_window_closes(tmp_path):
    """The gate and the actor freeze are driven by one condition, so they cannot
    drift apart: the arm is handed over at the same episode the actor starts
    being trained.
    """
    tr = _trainer(tmp_path, critic_only_episodes=3)
    seen = []
    tr.policy.set_actor_gate = seen.append          # stub has none; attach after init
    tr.warmup_completed_at_episode = 5

    for ep in (5, 6, 7, 8, 9):
        seen.clear()
        _fill_episode(tr, ep, success=True)
        tr.maybe_update(ep, tr.replay_buffer.total_added - 4)
        assert seen, f"ep {ep}: maybe_update did not touch the gate"
        # critic_only_until = 5 + 3 = 8
        assert seen[-1] is (ep >= 8), f"ep {ep}: gate={seen[-1]}"


def test_resume_reopens_the_actor_gate_before_the_first_episode(tmp_path):
    """__init__ closes the gate and maybe_update() reopens it -- but that does not
    run until AFTER the first resumed episode. Without reopening here, that
    episode would execute VLA actions while its transitions were recorded as the
    actor's.
    """
    tr = _trainer(tmp_path, critic_only_episodes=3)
    tr.warmup_completed_at_episode = 5
    tr.task = None
    _fill_episode(tr, 5, success=True)
    tr.save_latest_state(20)                      # 20 >= 5 + 3, so the actor was live

    fresh = _trainer(tmp_path, critic_only_episodes=3)
    fresh.task = None
    seen = []
    fresh.policy.set_actor_gate = seen.append
    fresh.load_latest_state(Path(tmp_path) / "latest_online_state.pt")
    assert seen and seen[-1] is True, f"gate not reopened on resume: {seen}"


def test_resume_keeps_the_gate_closed_when_still_in_critic_only(tmp_path):
    tr = _trainer(tmp_path, critic_only_episodes=10)
    tr.warmup_completed_at_episode = 5
    tr.task = None
    _fill_episode(tr, 5, success=True)
    tr.save_latest_state(8)                       # 8 < 5 + 10, actor not live yet

    fresh = _trainer(tmp_path, critic_only_episodes=10)
    fresh.task = None
    seen = []
    fresh.policy.set_actor_gate = seen.append
    fresh.load_latest_state(Path(tmp_path) / "latest_online_state.pt")
    assert seen and seen[-1] is False, f"gate wrongly opened: {seen}"


# --- evaluation mode ----------------------------------------------------------

def test_max_updates_zero_is_accepted_as_evaluation_mode():
    """0 must be legal: it is the only way to actually freeze a resumed policy.

    Driving the learning rate to ~0 does not work -- Adam's load_state_dict
    restores the snapshot's param_groups, so a resumed run silently goes back to
    the original lr (OnlineRLTrainer._report_effective_lr says so, and EVAL1's
    Arm A took ~25 updates at 3e-5 because of it).
    """
    from rlt_so101_dual.adapters.lerobot.record.backend import OnlineRLConfig

    OnlineRLConfig(max_updates_per_episode=0)          # must not raise
    OnlineRLConfig(max_updates_per_episode=1)


def test_max_updates_zero_runs_no_update_at_all(tmp_path):
    t = _trainer(tmp_path, warmup_episodes=1, critic_only_episodes=0,
                 min_warmup_transitions=1, min_warmup_successes=1,
                 min_warmup_failures=1, max_updates_per_episode=0)
    _fill_episode(t, 0, success=True)
    _fill_episode(t, 1, success=False)
    before = t.policy.actor.state_dict()
    before = {k: v.clone() for k, v in before.items()}
    stats = t.maybe_update(2, 0)
    assert stats is not None, "the gate/bookkeeping path must still run"
    assert stats.get("actual_updates", 0) == 0
    after = t.policy.actor.state_dict()
    worst = max((before[k] - after[k]).abs().max().item() for k in before)
    assert worst == 0.0, f"actor moved by {worst} with max_updates_per_episode=0"


# --- grasp-stage failures must be recordable ----------------------------------

def test_online_rl_no_longer_requires_skip_prefix_recording(tmp_path):
    """A grasp failure has no critical phase, so with prefix frames dropped it
    has zero frames and gets discarded -- s/f registers the outcome but the
    episode never reaches the dataset. Measured across one evaluation round the
    resulting discard rate differed between the two arms by 22 points, which
    makes their success rates incomparable. skip_prefix_recording only gates
    dataset.add_frame; the replay buffer sees every frame either way.
    """
    import re
    src = open("src/rlt_so101_dual/adapters/lerobot/record/backend.py").read()
    guard = re.search(
        r"if not self\.rlt\.rl_phase_key_toggles_critical_phase[^:]*:", src)
    assert guard is not None, "the phase-key guard disappeared"
    assert "not self.rlt.skip_prefix_recording" not in guard.group(0), \
        "online_rl must not require skip_prefix_recording any more"


def test_cli_exposes_skip_prefix_recording():
    """Default stays True so training behaviour is unchanged; evaluation runs
    pass --no-skip-prefix-recording so grasp failures land in the denominator."""
    from rlt_so101_dual.adapters.lerobot.record import online_cli
    src = open(online_cli.__file__).read()
    assert "--skip-prefix-recording" in src
    assert "BooleanOptionalAction" in src
    assert "args.skip_prefix_recording" in src
