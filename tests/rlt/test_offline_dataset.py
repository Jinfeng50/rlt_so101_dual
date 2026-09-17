from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from rlt_so101_dual.adapters.lerobot.demo_loader import rlt_demo_collate
from rlt_so101_dual.core.interfaces import ChunkTransition
from rlt_so101_dual.adapters.lerobot.offline_dataset import (
    build_overlap_frame_indices,
    build_transitions_from_demos,
    split_episode_indices,
    save_transition_cache,
    load_transition_cache,
)


def test_split_episode_indices_counts():
    splits = split_episode_indices(100, train_ratio=0.8, val_ratio=0.1, seed=42)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 100


def test_split_episode_indices_no_overlap():
    splits = split_episode_indices(50, train_ratio=0.7, val_ratio=0.15, seed=0)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_split_episode_indices_deterministic():
    s1 = split_episode_indices(80, seed=123)
    s2 = split_episode_indices(80, seed=123)
    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["test"] == s2["test"]


def _make_transitions(n: int, state_dim: int = 10, action_dim: int = 4, C: int = 3):
    transitions = []
    for _ in range(n):
        transitions.append(ChunkTransition(
            state_vec=torch.randn(state_dim),
            exec_chunk=torch.randn(C, action_dim),
            ref_chunk=torch.randn(C, action_dim),
            reward_seq=torch.randn(C),
            next_state_vec=torch.randn(state_dim),
            next_ref_chunk=torch.randn(C, action_dim),
            done=torch.tensor(0.0),
            intervention=torch.tensor(0.0),
            actual_steps=torch.tensor(C),
        ))
    return transitions


def test_save_load_transition_cache(tmp_path):
    transitions = _make_transitions(15)
    save_transition_cache(transitions, str(tmp_path), split="train")
    # These fixtures use deliberately small dimensions, so the contract check
    # has to be opted out of; test_cache_from_another_robot_is_rejected below
    # covers the check itself.
    buf = load_transition_cache(str(tmp_path), split="train", capacity=100, strict=False)
    assert len(buf) == 15

    # Verify round-trip preserves tensor values
    batch = buf.sample(15)
    original_states = torch.stack([t.state_vec for t in transitions])
    loaded_states = batch["state_vec"]
    # Each loaded state should match one original (order may differ due to sampling)
    for i in range(15):
        found = any(
            torch.allclose(loaded_states[i], original_states[j], atol=1e-6)
            for j in range(15)
        )
        assert found, f"Loaded state {i} not found in originals"


def test_build_transitions_basic():
    """build_transitions_from_demos requires a demo loader and agent.

    We test the internal _encoded_to_transitions helper directly since
    build_transitions_from_demos depends on a real DataLoader with Observation.
    """
    from rlt_so101_dual.adapters.lerobot.offline_dataset import _encoded_to_transitions

    state_dim, action_dim, C = 10, 4, 3
    encoded = []
    frame_indices = list(range(5))
    for _ in frame_indices:
        s = torch.randn(state_dim)
        r = torch.randn(C, action_dim)
        e = torch.randn(C, action_dim)
        encoded.append((s, r, e))

    transitions = _encoded_to_transitions(
        encoded=encoded,
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=C,
        stride=1,
    )
    assert len(transitions) == 2

    # Transition starting at t=1 should be terminal because next_state is x_{1+C}=x_4.
    assert transitions[-1].done.item() == 1.0
    for t in transitions[:-1]:
        assert t.done.item() == 0.0

    # next_state must be x_{t+C}, not the next sampled anchor.
    assert torch.allclose(transitions[0].next_state_vec, encoded[3][0])
    assert torch.allclose(transitions[1].next_state_vec, encoded[4][0])

    # Sparse binary reward only appears on the terminal chunk, at the final step.
    assert torch.equal(transitions[0].reward_seq, torch.zeros(C))
    assert torch.equal(transitions[1].reward_seq, torch.tensor([0.0, 0.0, 1.0]))

    # Shapes
    for t in transitions:
        assert t.state_vec.shape == (state_dim,)
        assert t.exec_chunk.shape == (C, action_dim)
        assert t.actual_steps.item() == C


def test_build_transitions_stride_uses_c_step_bootstrap():
    from rlt_so101_dual.adapters.lerobot.offline_dataset import _encoded_to_transitions

    state_dim, action_dim, C, stride = 8, 3, 4, 2
    frame_indices = [0, 2, 4, 6, 8]
    encoded = []
    for _ in frame_indices:
        encoded.append((
            torch.randn(state_dim),
            torch.randn(C, action_dim),
            torch.randn(C, action_dim),
        ))

    transitions = _encoded_to_transitions(
        encoded=encoded,
        frame_indices=frame_indices,
        episode_last_frame=8,
        chunk_length=C,
        stride=stride,
    )

    assert len(transitions) == 3
    assert torch.allclose(transitions[0].next_state_vec, encoded[2][0])  # x_{0+4}
    assert torch.allclose(transitions[1].next_state_vec, encoded[3][0])  # x_{2+4}
    assert torch.allclose(transitions[2].next_state_vec, encoded[4][0])  # x_{4+4}
    assert all(t.actual_steps.item() == C for t in transitions)
    assert transitions[-1].done.item() == 1.0


def test_build_overlap_frame_indices_keeps_terminal_anchor():
    indices = build_overlap_frame_indices(
        episode_start=0,
        episode_stop=14,
        chunk_length=10,
        stride=2,
    )

    assert indices == [0, 2, 3, 4, 6, 8, 10, 12, 13]


def test_encoded_to_transitions_respects_episode_success_flag():
    """When episode_success=False, terminal chunk reward must sum to 0."""
    from rlt_so101_dual.adapters.lerobot.offline_dataset import _encoded_to_transitions

    state_dim, action_dim, C = 10, 4, 3
    frame_indices = list(range(5))
    encoded = []
    for _ in frame_indices:
        encoded.append((
            torch.randn(state_dim),
            torch.randn(C, action_dim),
            torch.randn(C, action_dim),
        ))

    failed = _encoded_to_transitions(
        encoded=encoded,
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=C,
        stride=1,
        episode_success=False,
    )
    assert failed[-1].done.item() == 1.0
    assert failed[-1].reward_seq.sum().item() == 0.0
    for t in failed:
        assert t.reward_seq.sum().item() == 0.0

    succeeded = _encoded_to_transitions(
        encoded=encoded,
        frame_indices=frame_indices,
        episode_last_frame=4,
        chunk_length=C,
        stride=1,
        episode_success=True,
    )
    assert succeeded[-1].done.item() == 1.0
    assert torch.equal(succeeded[-1].reward_seq, torch.tensor([0.0, 0.0, 1.0]))


def test_encoded_to_transitions_accepts_irregular_terminal_anchor():
    from rlt_so101_dual.adapters.lerobot.offline_dataset import _encoded_to_transitions

    state_dim, action_dim, C, stride = 6, 2, 10, 2
    frame_indices = [0, 2, 3, 4, 6, 8, 10, 12, 13]
    encoded = []
    for _ in frame_indices:
        encoded.append((
            torch.randn(state_dim),
            torch.randn(C, action_dim),
            torch.randn(C, action_dim),
        ))

    transitions = _encoded_to_transitions(
        encoded=encoded,
        frame_indices=frame_indices,
        episode_last_frame=13,
        chunk_length=C,
        stride=stride,
    )

    assert [frame_indices[i] for i in range(len(frame_indices)) if frame_indices[i] + C <= 13] == [0, 2, 3]
    assert len(transitions) == 3
    assert torch.allclose(transitions[-1].next_state_vec, encoded[8][0])  # x_{3+10}=x_13
    assert transitions[-1].done.item() == 1.0


class _DemoBatchDataset(Dataset):
    def __init__(self, num_frames: int, chunk_length: int, proprio_dim: int, action_dim: int):
        self._items = []
        for frame_idx in range(num_frames):
            self._items.append({
                "proprio": torch.full((proprio_dim,), float(frame_idx)),
                "expert_actions": torch.full((chunk_length, action_dim), float(frame_idx)),
            })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._items[idx]


class _FakePolicy:
    def __init__(self, chunk_length: int, action_dim: int):
        self.chunk_length = chunk_length
        self.action_dim = action_dim

    def eval(self) -> None:
        return None

    def encode_observation(self, obs):
        state_vec = obs.proprio.float()
        ref_chunk = state_vec[:, :1].unsqueeze(-1).repeat(1, self.chunk_length, self.action_dim)
        return state_vec, ref_chunk


def _build_demo_transitions_for_episode_success(episode_success: bool):
    chunk_length = 3
    action_dim = 2
    dataset = _DemoBatchDataset(num_frames=5, chunk_length=chunk_length, proprio_dim=4, action_dim=action_dim)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=rlt_demo_collate,
        num_workers=0,
        drop_last=False,
    )
    return build_transitions_from_demos(
        policy=_FakePolicy(chunk_length=chunk_length, action_dim=action_dim),
        demo_loader=loader,
        frame_indices=[0, 1, 2, 3, 4],
        episode_last_frame=4,
        chunk_length=chunk_length,
        device="cpu",
        episode_id=3,
        episode_success=episode_success,
    )


def test_build_transitions_from_demos_terminal_failure_has_zero_reward():
    transitions = _build_demo_transitions_for_episode_success(episode_success=False)
    assert transitions[-1].done.item() == 1.0
    assert transitions[-1].reward_seq.sum().item() == pytest.approx(0.0)


def test_build_transitions_from_demos_terminal_success_rewards_last_step():
    transitions = _build_demo_transitions_for_episode_success(episode_success=True)
    assert transitions[-1].done.item() == 1.0
    assert torch.equal(transitions[-1].reward_seq, torch.tensor([0.0, 0.0, 1.0]))


def test_cache_from_another_robot_is_rejected(tmp_path):
    """A cache is a bare list of tensors with no dimension metadata, so one
    built for the bimanual robot used to load without complaint and only
    surface once a batch reached the networks -- or never, if it broadcast."""
    import pytest

    from rlt_so101_dual.core import shape_contract as sc

    bimanual = [
        ChunkTransition(
            state_vec=torch.zeros(sc.RL_TOKEN_DIM + 12),
            exec_chunk=torch.zeros(10, 12),
            ref_chunk=torch.zeros(10, 12),
            reward_seq=torch.zeros(10),
            next_state_vec=torch.zeros(sc.RL_TOKEN_DIM + 12),
            next_ref_chunk=torch.zeros(10, 12),
            done=torch.tensor(0.0),
            intervention=torch.tensor(0.0),
            actual_steps=torch.tensor(10),
        )
        for _ in range(3)
    ]
    save_transition_cache(bimanual, str(tmp_path), split="train")

    with pytest.raises(ValueError, match="built for a different robot"):
        load_transition_cache(str(tmp_path), split="train")

    buf = load_transition_cache(str(tmp_path), split="train", strict=False)
    assert len(buf) == 3


def test_contract_cache_loads_clean(tmp_path):
    from rlt_so101_dual.core import shape_contract as sc

    ours = [
        ChunkTransition(
            state_vec=torch.zeros(sc.state_vec_dim()),
            exec_chunk=torch.zeros(sc.CHUNK_LENGTH, sc.ACTION_DIM),
            ref_chunk=torch.zeros(sc.CHUNK_LENGTH, sc.ACTION_DIM),
            reward_seq=torch.zeros(sc.CHUNK_LENGTH),
            next_state_vec=torch.zeros(sc.state_vec_dim()),
            next_ref_chunk=torch.zeros(sc.CHUNK_LENGTH, sc.ACTION_DIM),
            done=torch.tensor(0.0),
            intervention=torch.tensor(0.0),
            actual_steps=torch.tensor(sc.CHUNK_LENGTH),
        )
        for _ in range(3)
    ]
    save_transition_cache(ours, str(tmp_path), split="train")
    assert len(load_transition_cache(str(tmp_path), split="train")) == 3
