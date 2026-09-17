"""BC-regularisation schedule.

beta was a fixed number with no way to change it during a run. A large beta
early keeps the policy anchored to the VLA while the critic's Q estimates are
still noisy; the same value later is what stops the actor from ever departing
from the reference.
"""

from types import SimpleNamespace

import pytest

from rlt_so101_dual.adapters.lerobot.record.online_trainer import OnlineRLTrainer


def _schedule(beta=0.5, beta_final=None, anneal=10):
    trainer = OnlineRLTrainer.__new__(OnlineRLTrainer)
    trainer.beta_start = beta
    trainer.beta_final = beta_final
    trainer.beta_anneal_episodes = anneal
    return trainer


def test_no_final_value_keeps_beta_fixed():
    """The default must reproduce the paper, which has no schedule."""
    t = _schedule(beta=0.3, beta_final=None)
    assert [t._scheduled_beta(e, 5) for e in (0, 5, 50, 500)] == [0.3] * 4


def test_beta_holds_through_the_critic_only_window():
    t = _schedule(beta=0.5, beta_final=0.1, anneal=10)
    # The actor is frozen until critic_only_until, so the value cannot matter
    # yet and must not have started decaying.
    assert t._scheduled_beta(0, 20) == 0.5
    assert t._scheduled_beta(20, 20) == 0.5


def test_beta_anneals_linearly_then_holds():
    t = _schedule(beta=0.5, beta_final=0.1, anneal=10)
    assert t._scheduled_beta(25, 20) == pytest.approx(0.3)     # halfway
    assert t._scheduled_beta(30, 20) == pytest.approx(0.1)     # done
    assert t._scheduled_beta(200, 20) == pytest.approx(0.1)    # stays

    values = [t._scheduled_beta(e, 20) for e in range(20, 31)]
    assert values == sorted(values, reverse=True)


def test_annealing_upward_also_works():
    t = _schedule(beta=0.1, beta_final=0.5, anneal=4)
    assert t._scheduled_beta(22, 20) == pytest.approx(0.3)
    assert t._scheduled_beta(24, 20) == pytest.approx(0.5)
