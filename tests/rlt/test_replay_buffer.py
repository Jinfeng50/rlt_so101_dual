"""Stratified sampling must not manufacture data out of a short pool."""

from __future__ import annotations

import torch

from rlt_so101_dual.core.interfaces import ChunkTransition
from rlt_so101_dual.core.replay_buffer import ReplayBuffer
from tests.rlt.helpers import ACTION_DIM, C, STATE_DIM


def _transition(episode_id: int, *, done: bool, success: bool, intervention: bool = False):
    reward = torch.zeros(C)
    if done and success:
        reward[C - 1] = 1.0
    return ChunkTransition(
        state_vec=torch.randn(STATE_DIM),
        exec_chunk=torch.randn(C, ACTION_DIM),
        ref_chunk=torch.randn(C, ACTION_DIM),
        reward_seq=reward,
        next_state_vec=torch.randn(STATE_DIM),
        next_ref_chunk=torch.randn(C, ACTION_DIM),
        done=torch.tensor(float(done)),
        intervention=torch.tensor(float(intervention)),
        actual_steps=torch.tensor(C),
        episode_id=torch.tensor(episode_id),
        outcome=torch.tensor(float(success)) if done else torch.tensor(-1.0),
    )


def test_stratified_never_duplicates_a_short_pool():
    """A small stratum must not be blown up to fill its quota.

    run5 held 33 intervention transitions -- 1.7% of the buffer -- against a
    fixed 20% quota, so every batch drew ~51 times from those 33 and the batch
    BC term rose 3.5x for 36 episodes. The paper's intervention mechanism is the
    BC-anchor swap (Algorithm 1 line 11), not a fixed share of every batch; the
    share is this project's own addition with no ablation behind it.
    """
    # 800 entries so the 500-wide "recent" window starts at 300 and cannot also
    # pull the 4 intervention transitions -- otherwise their rows could come
    # from two strata and the count would not isolate the one under test.
    buf = ReplayBuffer(capacity=2000)
    for i in range(800):
        buf.add(_transition(i, done=(i % 20 == 19), success=(i % 3 != 0),
                            intervention=(i < 4)))          # 4 of 800

    batch = buf.sample_stratified(256)
    ids = batch["episode_id"].flatten().tolist()
    n_int = sum(1 for e in ids if e < 4)
    assert n_int <= 4, f"drew {n_int} rows from an intervention pool of 4"
    assert batch["state_vec"].shape[0] == 256, "the batch must still be full"


def test_no_transition_is_drawn_twice_by_the_same_stratum():
    """Strata may overlap (a success is also "recent"), so a transition can
    appear once per stratum -- four times at most. What must not happen is one
    stratum drawing the same entry over and over to fill its quota."""
    import collections

    buf = ReplayBuffer(capacity=1000)
    for i in range(30):
        buf.add(_transition(i, done=True, success=True))
    batch = buf.sample_stratified(256)
    counts = collections.Counter(batch["episode_id"].flatten().tolist())
    assert max(counts.values()) <= 4, f"an entry appeared {max(counts.values())} times"


def test_stratified_still_reaches_the_reward_carrying_strata():
    """Capping must not starve the strata the stratification exists for."""
    buf = ReplayBuffer(capacity=1000)
    for i in range(400):
        buf.add(_transition(i, done=(i % 10 == 9), success=(i % 20 < 10)))
    batch = buf.sample_stratified(256)
    assert int((batch["reward_seq"].sum(dim=-1) > 0).sum()) > 0
