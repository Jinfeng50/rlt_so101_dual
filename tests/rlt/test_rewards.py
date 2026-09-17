from __future__ import annotations

import pytest
import torch

from rlt_so101_dual.core.rewards import build_reward_seq


C = 6


def test_terminal_reward_placed_at_last_step():
    reward = build_reward_seq(chunk_length=C, is_terminal_chunk=True, episode_success=True)
    assert reward.shape == (C,)
    assert reward[-1].item() == pytest.approx(1.0)
    assert reward[:-1].abs().sum().item() == 0.0


def test_no_reward_when_not_terminal_chunk():
    reward = build_reward_seq(chunk_length=C, is_terminal_chunk=False, episode_success=True)
    assert reward.abs().sum().item() == 0.0


def test_no_reward_on_failure():
    reward = build_reward_seq(chunk_length=C, is_terminal_chunk=True, episode_success=False)
    assert reward.abs().sum().item() == 0.0


def test_actual_steps_places_reward_at_last_valid_step():
    reward = build_reward_seq(
        chunk_length=C, is_terminal_chunk=True, episode_success=True, actual_steps=3,
    )
    assert reward[2].item() == pytest.approx(1.0)
    assert reward[:2].abs().sum().item() == 0.0
    assert reward[3:].abs().sum().item() == 0.0


def test_tensor_actual_steps():
    r_int = build_reward_seq(
        chunk_length=C, is_terminal_chunk=True, episode_success=True, actual_steps=4,
    )
    r_tensor = build_reward_seq(
        chunk_length=C, is_terminal_chunk=True, episode_success=True,
        actual_steps=torch.tensor(4),
    )
    assert torch.allclose(r_int, r_tensor)


def test_actual_steps_zero_yields_no_reward():
    reward = build_reward_seq(
        chunk_length=C, is_terminal_chunk=True, episode_success=True, actual_steps=0,
    )
    assert reward.abs().sum().item() == 0.0
