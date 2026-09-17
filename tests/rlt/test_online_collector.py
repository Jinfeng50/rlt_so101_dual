from __future__ import annotations

import torch

# q01=-1, q99=+1 makes 2*(x - q01)/(q99 - q01) - 1 the identity, so these tests
# keep exercising what they always did while satisfying the collector's
# requirement that action stats be present (missing stats now fail fast --
# without them the critic and the actor end up in different action spaces).
def _identity_stats(dim: int):
    """q01=-1, q99=+1 makes 2*(x - q01)/(q99 - q01) - 1 the identity."""
    return torch.full((dim,), -1.0), torch.full((dim,), 1.0)
import pytest

from rlt_so101_dual.adapters.lerobot.online_collector import RLTOnlineCollector
from rlt_so101_dual.adapters.lerobot.record.annotations import SOURCE_RL
from rlt_so101_dual.core.replay_buffer import ReplayBuffer

CHUNK_LENGTH = 4
ACTION_DIM = 12
STATE_DIM = 10


def _make_collector(**kwargs) -> tuple[RLTOnlineCollector, ReplayBuffer]:
    """Collector with shaping off by default, so the tests below describe the
    plain sparse-terminal behaviour; the shaping tests opt in explicitly."""
    buffer = ReplayBuffer(capacity=100)
    kwargs.setdefault("milestone_reward", 0.0)
    kwargs.setdefault("time_decay", 1.0)
    q01, q99 = _identity_stats(ACTION_DIM)
    kwargs.setdefault("action_q01", q01)
    kwargs.setdefault("action_q99", q99)
    collector = RLTOnlineCollector(
        replay_buffer=buffer, chunk_length=CHUNK_LENGTH, action_dim=ACTION_DIM, **kwargs,
    )
    collector.start_episode(episode_id=0)
    return collector, buffer


def _feed_frame(collector: RLTOnlineCollector, chunk_idx: int) -> None:
    is_chunk_start = chunk_idx == 0
    state = torch.randn(STATE_DIM) if is_chunk_start else None
    ref = torch.randn(CHUNK_LENGTH, ACTION_DIM) if is_chunk_start else None
    collector.on_frame(
        action=torch.randn(ACTION_DIM),
        state_vec=state,
        ref_chunk=ref,
        source_type=SOURCE_RL,
        is_critical=1.0,
    )


class TestFlushEpisodeReward:
    def test_success_sets_terminal_reward_partial_chunk(self):
        """Episode ends mid-chunk (not an exact multiple of C): flush_episode
        emits the terminal transition itself."""
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):  # one short of a full chunk
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)

        assert len(buffer) == 1
        transition = buffer.buffer[0]
        actual = int(transition.actual_steps.item())
        assert actual == CHUNK_LENGTH - 1
        assert transition.done.item() == 1.0
        assert transition.reward_seq[actual - 1].item() == 1.0
        assert transition.reward_seq[:actual - 1].sum().item() == 0.0

    def test_failure_leaves_reward_zero_partial_chunk(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)

        transition = buffer.buffer[0]
        assert transition.done.item() == 1.0
        assert transition.reward_seq.sum().item() == 0.0

    def test_success_patches_prev_transition_exact_multiple(self):
        """Episode length is an exact multiple of C: the last chunk was already
        emitted (staged, not yet committed to the buffer) with done=False by
        on_frame; flush_episode must patch it and commit it."""
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):  # exactly one full chunk, already emitted (staged)
            _feed_frame(collector, i)
        assert len(buffer) == 0  # staged, not committed until flush_episode
        assert collector._episode_staging[0].done.item() == 0.0

        collector.flush_episode(episode_success=True)

        assert len(buffer) == 1  # no extra transition emitted
        transition = buffer.buffer[0]
        assert transition.done.item() == 1.0
        actual = int(transition.actual_steps.item())
        assert actual == CHUNK_LENGTH
        assert transition.reward_seq[actual - 1].item() == 1.0

    def test_failure_patches_prev_transition_exact_multiple(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)

        transition = buffer.buffer[0]
        assert transition.done.item() == 1.0
        assert transition.reward_seq.sum().item() == 0.0

    def test_non_terminal_chunks_keep_zero_reward(self):
        """Only the terminal transition's reward should ever be nonzero."""
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):  # first full chunk (non-terminal)
            _feed_frame(collector, i)
        for i in range(CHUNK_LENGTH - 1):  # second, partial, terminal chunk
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)

        assert len(buffer) == 2
        first, second = buffer.buffer[0], buffer.buffer[1]
        assert first.done.item() == 0.0
        assert first.reward_seq.sum().item() == 0.0
        assert second.done.item() == 1.0
        actual = int(second.actual_steps.item())
        assert second.reward_seq[actual - 1].item() == 1.0


class TestEpisodeStaging:
    """Transitions must stay out of the global replay buffer until
    flush_episode() commits them, so a rerecorded/discarded/never-labeled
    episode (flush_episode never called) can't leave dangling, unlabeled,
    no-valid-next-state transitions in the buffer forever."""

    def test_unflushed_episode_never_reaches_global_buffer(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):  # a full non-terminal chunk gets staged
            _feed_frame(collector, i)
        for i in range(CHUNK_LENGTH - 2):  # partial second chunk, never completes
            _feed_frame(collector, i)
        # Episode gets rerecorded/discarded: caller never calls flush_episode().
        assert len(buffer) == 0

    def test_next_start_episode_drops_unflushed_staging(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH):
            _feed_frame(collector, i)
        assert len(collector._episode_staging) == 1
        # Rerecord: start the retry without ever flushing the bad attempt.
        collector.start_episode(episode_id=0)
        assert collector._episode_staging == []
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        # Only the retry's transition made it to the buffer.
        assert len(buffer) == 1


class TestFlushedGuard:
    """After flush_episode() (critical phase resolved), the recorded episode
    may keep going (e.g. VLA autonomously finishing a subsequent step) --
    on_frame() must ignore all of that so the RL reward reflects only what
    the actor actually controlled, not whatever happens afterward."""

    def test_on_frame_is_noop_after_flush(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert len(buffer) == 1

        # Episode keeps recording under VLA afterward -- fed to on_frame as
        # usual by loop.py, but must not affect the buffer at all.
        for i in range(CHUNK_LENGTH * 3):
            _feed_frame(collector, i % CHUNK_LENGTH)
        assert len(buffer) == 1
        assert collector._episode_staging == []

    def test_next_start_episode_clears_flushed_guard(self):
        collector, buffer = _make_collector()
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert collector._flushed is True

        collector.start_episode(episode_id=1)
        assert collector._flushed is False
        for i in range(CHUNK_LENGTH - 1):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)
        assert len(buffer) == 2


class TestMilestoneAndTimeDecay:
    """Sub-goal bonus and the speed incentive.

    The task is 'pick up the bolt, then insert it into the sleeve': the pickup
    is the milestone, the insertion is the terminal reward.
    """

    def _run(self, collector, n_chunks: int, milestone_after: int | None = None) -> float:
        """Feed n_chunks worth of frames, optionally pressing the milestone key
        after `milestone_after` complete chunks. Returns the milestone bonus."""
        bonus = 0.0
        for c in range(n_chunks):
            if milestone_after is not None and c == milestone_after:
                bonus = collector.mark_milestone()
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        if milestone_after is not None and milestone_after >= n_chunks:
            bonus = collector.mark_milestone()
        return bonus

    def test_decay_of_one_reproduces_plain_sparse_reward(self):
        collector, buffer = _make_collector(terminal_reward=1.0, time_decay=1.0)
        self._run(collector, 5)
        collector.flush_episode(episode_success=True)
        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) == pytest.approx(1.0)

    def test_terminal_reward_shrinks_with_chunks_closed(self):
        rewards = []
        for n_chunks in (2, 10):
            collector, buffer = _make_collector(terminal_reward=1.0, time_decay=0.9)
            self._run(collector, n_chunks)
            collector.flush_episode(episode_success=True)
            rewards.append(sum(t.reward_seq.sum().item() for t in buffer.buffer))
        assert rewards[0] == pytest.approx(0.9 ** 2)
        assert rewards[1] == pytest.approx(0.9 ** 10)
        assert rewards[1] < rewards[0], "finishing later must be worth less"

    def test_decay_counts_chunks_not_frames(self):
        """A per-frame exponent over the same span would collapse to ~0."""
        collector, buffer = _make_collector(terminal_reward=1.0, time_decay=0.9)
        self._run(collector, 3)
        collector.flush_episode(episode_success=True)
        total = sum(t.reward_seq.sum().item() for t in buffer.buffer)
        assert total == pytest.approx(0.9 ** 3)
        assert total != pytest.approx(0.9 ** (3 * CHUNK_LENGTH))

    def test_milestone_lands_on_the_next_chunk_to_close(self):
        collector, buffer = _make_collector(milestone_reward=0.3, time_decay=1.0)
        self._run(collector, 4, milestone_after=2)
        collector.flush_episode(episode_success=False)
        # Pressed after 2 closed chunks, so it belongs to the third.
        per_chunk = [t.reward_seq.sum().item() for t in buffer.buffer]
        assert per_chunk[2] == pytest.approx(0.3)
        assert sum(per_chunk) == pytest.approx(0.3)

    def test_milestone_fires_at_most_once(self):
        collector, buffer = _make_collector(milestone_reward=0.3, time_decay=1.0)
        self._run(collector, 2, milestone_after=0)
        assert collector.mark_milestone() == 0.0
        assert collector.mark_milestone() == 0.0
        collector.flush_episode(episode_success=False)
        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) == pytest.approx(0.3)

    def test_milestone_decays_too(self):
        collector, _ = _make_collector(milestone_reward=1.0, time_decay=0.5)
        early = collector.mark_milestone()
        assert early == pytest.approx(1.0)

        collector2, _ = _make_collector(milestone_reward=1.0, time_decay=0.5)
        self._run(collector2, 3)
        assert collector2.mark_milestone() == pytest.approx(0.5 ** 3)

    def test_milestone_and_terminal_add_up(self):
        collector, buffer = _make_collector(
            milestone_reward=0.3, terminal_reward=1.0, time_decay=1.0
        )
        self._run(collector, 3, milestone_after=1)
        collector.flush_episode(episode_success=True)
        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) == pytest.approx(1.3)

    def test_milestone_ignored_once_the_attempt_is_over(self):
        collector, _ = _make_collector(milestone_reward=0.3)
        self._run(collector, 2)
        collector.flush_episode(episode_success=True)
        assert collector.mark_milestone() == 0.0

    def test_begin_attempt_restarts_the_clock(self):
        """The VLA prefix before the operator enters the critical phase is of
        arbitrary length and must not count toward the decay."""
        collector, buffer = _make_collector(terminal_reward=1.0, time_decay=0.9)
        self._run(collector, 6)          # VLA prefix
        collector.begin_attempt()
        self._run(collector, 2)          # the actual attempt
        collector.flush_episode(episode_success=True)
        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) == pytest.approx(0.9 ** 2)


class TestOutcomeLabel:
    def test_milestone_on_a_failed_attempt_is_not_a_success(self):
        """The bug this field exists to prevent: a positive shaping reward on
        an attempt that failed used to be read back as a success by
        stratified sampling."""
        collector, buffer = _make_collector(milestone_reward=0.3, time_decay=1.0)
        for c in range(2):
            if c == 1:
                collector.mark_milestone()
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)

        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) > 0
        assert buffer.episode_outcomes() == {0: "failure"}
        assert buffer.count_outcomes() == (0, 1)

    def test_every_transition_of_the_attempt_carries_the_outcome(self):
        collector, buffer = _make_collector()
        for _ in range(3):
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert all(t.outcome.item() == 1.0 for t in buffer.buffer)
        assert buffer.episode_outcomes() == {0: "success"}


class TestAttemptStats:
    """Reward accounting published for logging. Without it a run gives no way
    to tell whether the milestone key was pressed or what the decay cost."""

    def test_stats_describe_the_attempt(self):
        collector, _ = _make_collector(
            milestone_reward=0.4, terminal_reward=1.0, time_decay=1.0
        )
        for c in range(3):
            if c == 1:
                collector.mark_milestone()
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)

        stats = collector.last_attempt_stats
        assert stats["chunks"] == 3
        assert stats["milestone_awarded"] == 1.0
        assert stats["milestone_bonus"] == pytest.approx(0.4)
        assert stats["terminal_reward"] == pytest.approx(1.0)
        assert stats["total_reward"] == pytest.approx(1.4)
        assert stats["success"] == 1.0

    def test_chunks_is_the_decay_exponent(self):
        collector, _ = _make_collector(terminal_reward=1.0, time_decay=0.9)
        for _ in range(4):
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        stats = collector.last_attempt_stats
        assert stats["terminal_reward"] == pytest.approx(0.9 ** stats["chunks"])

    def test_failed_attempt_still_reports_its_milestone(self):
        collector, _ = _make_collector(milestone_reward=0.3, time_decay=1.0)
        collector.mark_milestone()
        for i in range(CHUNK_LENGTH):
            _feed_frame(collector, i)
        collector.flush_episode(episode_success=False)
        stats = collector.last_attempt_stats
        assert stats["success"] == 0.0
        assert stats["terminal_reward"] == 0.0
        assert stats["total_reward"] == pytest.approx(0.3)


class TestRetryAfterFailedAttempt:
    """Resolving an attempt with `u` and starting another with `r` is the
    normal retry flow, and it used to lose everything silently.

    begin_attempt() reset only the decay clock, so the retry's frames were
    still blocked by the _flushed guard, its milestone key was a no-op, and
    its terminal reward landed on the *previous* attempt's last transition --
    relabelling a failed attempt as successful.
    """

    def _run(self, collector, n_chunks):
        for _ in range(n_chunks):
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)

    def test_retry_after_failure_is_collected(self):
        collector, buffer = _make_collector(
            milestone_reward=0.3, terminal_reward=1.0, time_decay=1.0
        )
        collector.begin_attempt()
        self._run(collector, 2)
        collector.flush_episode(episode_success=False)
        after_first = len(buffer)
        assert after_first == 2

        collector.begin_attempt()
        self._run(collector, 3)
        assert collector.mark_milestone() == pytest.approx(0.3)
        collector.flush_episode(episode_success=True)

        assert len(buffer) > after_first, "the retry's transitions must be collected"
        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) == pytest.approx(1.3)

    def test_failed_attempt_is_not_relabelled_by_the_retry(self):
        collector, buffer = _make_collector(terminal_reward=1.0, time_decay=1.0)
        collector.begin_attempt()
        self._run(collector, 2)
        collector.flush_episode(episode_success=False)
        first_attempt = list(buffer.buffer)
        assert all(t.reward_seq.sum().item() == 0.0 for t in first_attempt)

        collector.begin_attempt()
        self._run(collector, 2)
        collector.flush_episode(episode_success=True)

        assert all(t.reward_seq.sum().item() == 0.0 for t in first_attempt), \
            "the retry's reward must not be written onto the failed attempt"

    def test_each_attempt_gets_its_own_terminal_transition(self):
        collector, buffer = _make_collector()
        for success in (False, True):
            collector.begin_attempt()
            self._run(collector, 2)
            collector.flush_episode(episode_success=success)
        assert sum(1 for t in buffer.buffer if t.done.item() == 1.0) == 2

    def test_vla_prefix_is_dropped_when_the_attempt_starts(self):
        """Chunks before `r` are VLA, not something the actor controlled."""
        collector, buffer = _make_collector()
        self._run(collector, 4)          # VLA prefix
        collector.begin_attempt()
        self._run(collector, 2)          # the attempt
        collector.flush_episode(episode_success=True)
        assert len(buffer) == 2

    def test_double_resolve_is_ignored(self):
        collector, buffer = _make_collector(terminal_reward=1.0, time_decay=1.0)
        collector.begin_attempt()
        self._run(collector, 2)
        collector.flush_episode(episode_success=False)
        collector.flush_episode(episode_success=True)   # stray second press
        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) == 0.0


class TestMilestoneAtChunkBoundary:
    def test_bonus_pressed_after_the_last_chunk_still_lands(self):
        collector, buffer = _make_collector(
            milestone_reward=0.3, terminal_reward=1.0, time_decay=1.0
        )
        for _ in range(2):
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        # Pressed with no frames accumulated: no later chunk will close.
        assert collector.mark_milestone() == pytest.approx(0.3)
        collector.flush_episode(episode_success=True)
        assert sum(t.reward_seq.sum().item() for t in buffer.buffer) == pytest.approx(1.3)

    def test_stats_report_what_was_actually_awarded(self):
        collector, _ = _make_collector(milestone_reward=0.3, time_decay=1.0)
        for i in range(CHUNK_LENGTH):
            _feed_frame(collector, i)
        collector.mark_milestone()
        collector.flush_episode(episode_success=True)
        stats = collector.last_attempt_stats
        assert stats["milestone_bonus"] == pytest.approx(0.3)
        assert stats["total_reward"] == pytest.approx(stats["milestone_bonus"] + stats["terminal_reward"])


class TestDecayValuesAtRealChunkCounts:
    """Pins the numbers the documentation quotes.

    The 5 places that described the decay all carried "50-300 chunks, 0.995
    spans 0.78 to 0.22" -- an upstream bimanual figure copied over without
    converting. One chunk here is chunk_length/fps = 10/30 = 0.333 s, so a
    5-20 s critical phase is 15-60 chunks, where 0.995 only spans 0.93 to
    0.74 and expresses almost no preference for finishing sooner.
    """

    @staticmethod
    def _chunks_for(seconds: float) -> int:
        from rlt_so101_dual.core import shape_contract as sc

        return round(seconds * sc.FPS / sc.CHUNK_LENGTH)

    def test_one_chunk_is_a_third_of_a_second(self):
        from rlt_so101_dual.core import shape_contract as sc

        assert sc.CHUNK_LENGTH / sc.FPS == pytest.approx(1 / 3, abs=1e-3)
        assert self._chunks_for(5) == 15
        assert self._chunks_for(8) == 24
        assert self._chunks_for(20) == 60

    @pytest.mark.parametrize(
        "seconds, expected",
        [(5, 0.739), (8, 0.616), (20, 0.298)],
    )
    def test_terminal_reward_at_the_default_decay(self, seconds, expected):
        n = self._chunks_for(seconds)
        collector, buffer = _make_collector(terminal_reward=1.0, time_decay=0.98)
        for _ in range(n):
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        total = sum(t.reward_seq.sum().item() for t in buffer.buffer)
        assert total == pytest.approx(expected, abs=1e-3)

    def test_the_upstream_default_would_barely_discriminate(self):
        """0.995 over 5 s versus 20 s differs by 25%; 0.98 differs by 2.5x."""
        weak = 0.995 ** self._chunks_for(5) / 0.995 ** self._chunks_for(20)
        ours = 0.98 ** self._chunks_for(5) / 0.98 ** self._chunks_for(20)
        assert weak == pytest.approx(1.25, abs=0.02)
        assert ours == pytest.approx(2.48, abs=0.02)

    def test_milestone_decays_on_the_same_clock(self):
        n = self._chunks_for(3)  # sub-goal reached at about 3 s
        collector, _ = _make_collector(milestone_reward=0.3, time_decay=0.98)
        for _ in range(n):
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        assert collector.mark_milestone() == pytest.approx(0.3 * 0.98 ** n, abs=1e-4)

    def test_terminal_still_outweighs_the_milestone(self):
        """The sub-goal bonus must not compete with finishing the task."""
        milestone = 0.3 * 0.98 ** self._chunks_for(3)
        terminal = 1.0 * 0.98 ** self._chunks_for(8)
        assert terminal > 2 * milestone


class TestWarmupTransitionConversion:
    def test_documented_seconds_conversion(self):
        """--min-warmup-transitions help quotes transitions * chunk_length / fps."""
        from rlt_so101_dual.core import shape_contract as sc

        assert 512 * sc.CHUNK_LENGTH / sc.FPS == pytest.approx(170.7, abs=0.5)
        assert 1000 * sc.CHUNK_LENGTH / sc.FPS == pytest.approx(333.3, abs=0.5)

    def test_a_full_attempt_yields_one_transition_per_chunk(self):
        """The conversion is only valid because the collector emits exactly
        one transition per closed chunk."""
        collector, buffer = _make_collector()
        for _ in range(7):
            for i in range(CHUNK_LENGTH):
                _feed_frame(collector, i)
        collector.flush_episode(episode_success=True)
        assert len(buffer) == 7


# --- a takeover's BC anchor has to live in the actor's space --------------


class TestHumanAnchorUnits:
    """exec_chunk arrives in robot units (loop.py builds it from the
    post-processor's output) while mu and every other ref are normalised.
    Storing the takeover's action raw made the BC target ~20x too large on
    exactly the transitions stratified sampling pins at 20% of every batch.
    """

    def _collector(self, buf, **kw):
        from rlt_so101_dual.adapters.lerobot.online_collector import RLTOnlineCollector
        q01, q99 = _identity_stats(3)
        kw.setdefault("action_q01", q01)
        kw.setdefault("action_q99", q99)
        return RLTOnlineCollector(replay_buffer=buf, chunk_length=2, action_dim=3,
                                  milestone_reward=0.0, terminal_reward=1.0,
                                  time_decay=1.0, **kw)

    def _run(self, col, human):
        from rlt_so101_dual.adapters.lerobot.online_collector import SOURCE_HUMAN
        col.start_episode(0)
        col.begin_attempt()
        src = float(SOURCE_HUMAN) if human else 1.0
        for _ in range(2):
            col.on_frame(action=torch.full((3,), 30.0), state_vec=torch.zeros(4),
                         ref_chunk=torch.zeros(2, 3), source_type=src, is_critical=1.0)
        col.flush_episode(True)

    def test_exec_chunk_is_stored_in_the_actors_space(self):
        """The critic trains on exec_chunk and actor_loss queries it with mu.
        If exec stays in robot units the two are an order of magnitude apart and
        the Q term carries no usable gradient -- online RL silently degrades to
        pure BC, which is what three runs measured.
        """
        from rlt_so101_dual.core.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        col = self._collector(buf, action_q01=torch.zeros(3),
                              action_q99=torch.full((3,), 60.0))
        self._run(col, human=False)          # 30.0 robot units -> 0.0 normalised
        ex = buf.buffer[-1].exec_chunk
        assert ex.abs().max() <= 1.0 + 1e-5, f"robot units reached the buffer: {ex}"
        assert torch.allclose(ex, torch.zeros_like(ex), atol=1e-5), ex

    def test_missing_stats_fail_fast_instead_of_degrading(self):
        """Without stats there is no way to reach the actor's space, and carrying
        on silently is how three runs trained a critic in robot units while
        actor_loss queried it in [-1, 1]. Refuse instead.
        """
        from rlt_so101_dual.core.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        col = self._collector(buf, action_q01=None, action_q99=None)
        with pytest.raises(ValueError, match="q01/q99"):
            self._run(col, human=False)

    def test_human_anchor_is_normalised_when_stats_are_available(self):
        from rlt_so101_dual.core.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        col = self._collector(buf,
                              action_q01=torch.zeros(3), action_q99=torch.full((3,), 60.0))
        self._run(col, human=True)
        ref = buf.buffer[-1].ref_chunk
        # 2*(30 - 0)/60 - 1 == 0
        assert torch.allclose(ref, torch.zeros_like(ref), atol=1e-5), ref
        assert float(buf.buffer[-1].intervention) == 1.0

    def test_non_human_chunks_keep_the_vla_reference(self):
        from rlt_so101_dual.core.replay_buffer import ReplayBuffer
        buf = ReplayBuffer(capacity=10)
        col = self._collector(buf,
                              action_q01=torch.zeros(3), action_q99=torch.full((3,), 60.0))
        self._run(col, human=False)
        assert torch.allclose(buf.buffer[-1].ref_chunk, torch.zeros(2, 3))
        assert float(buf.buffer[-1].intervention) == 0.0


def test_short_chunk_padding_is_zero_in_the_normalised_space():
    """Padding used to be added before normalising, so robot-unit zeros went
    through the transform and became a fabricated action -- -19.5 for
    wrist_roll. losses.py masks only the reward with actual_steps, never the
    action tensors, and 91% of the reward-carrying done=1 transitions are short
    chunks, so that junk landed on the transitions that matter most.
    """
    # deliberately not centred on 0, so a robot-unit zero does NOT normalise to 0
    q01 = torch.full((ACTION_DIM,), 10.0)
    q99 = torch.full((ACTION_DIM,), 50.0)
    collector, buffer = _make_collector(action_q01=q01, action_q99=q99)
    collector.begin_attempt()
    # one real frame, then resolve -> a chunk of length 1 padded to CHUNK_LENGTH
    collector.on_frame(action=torch.full((ACTION_DIM,), 30.0),
                       state_vec=torch.zeros(STATE_DIM),
                       ref_chunk=torch.zeros(CHUNK_LENGTH, ACTION_DIM),
                       source_type=float(SOURCE_RL), is_critical=1.0)
    collector.flush_episode(True)

    t = buffer.buffer[-1]
    assert int(t.actual_steps) == 1
    assert torch.allclose(t.exec_chunk[1:], torch.zeros_like(t.exec_chunk[1:])), \
        f"padding is not zero in the normalised space: {t.exec_chunk[1:]}"
