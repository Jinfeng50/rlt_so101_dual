from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_transition_cache_v2_passes_video_backend(monkeypatch, tmp_path):
    module = pytest.importorskip("rlt_so101_dual.cli.build_transition_cache_v2")

    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.num_episodes = 0
            self.meta = SimpleNamespace(episodes=None)

    class FakePolicy:
        config = SimpleNamespace(
            action_dim=12,
            chunk_size=50,
            image_only=False,
            proprio_dim=12,
            token_pool_size=0,
            vla_pretrained_path=None,
        )
        _num_image_tokens = 0
        _pi05 = object()
        rl_token = object()

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeCapture:
        def __init__(self, **kwargs):
            pass

        def attach(self, pi05):
            pass

        def detach(self):
            pass

    monkeypatch.setattr(module, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(module.RLTokenPolicy, "from_pretrained", lambda path: FakePolicy())
    monkeypatch.setattr(module, "PrefixOutputCapture", FakeCapture)
    monkeypatch.setattr(module, "make_rlt_token_pre_post_processors", lambda config: (object(), object()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_transition_cache_v2.py",
            "--demo-dataset-repo-id",
            "local/demo",
            "--demo-dataset-root",
            "/tmp/demo",
            "--rl-token-policy-path",
            "/tmp/rl-token",
            "--vla-pretrained-path",
            "/tmp/vla",
            "--output-dir",
            str(tmp_path),
            "--max-episodes",
            "0",
            "--video-backend",
            "video_reader",
        ],
    )

    module.main()

    assert captured["video_backend"] == "video_reader"


# --- chunk-level TD semantics -------------------------------------------------
# build_overlap_frame_indices returns a MIXED anchor set: stride-sampled starts,
# the terminal anchor, and the x_{t+C} bootstrap states. Treating adjacent
# encoded frames as a transition gives (x_0, a_0:9, x_2) instead of
# (x_0, a_0:9, x_10) -- a next_state two control steps away discounted by
# gamma^C, with the terminal flag on the wrong chunk.

C = 10
FRAMES = [0, 2, 4, 6, 8, 10, 12]   # stride 2, episode_last_frame = 12
LAST = 12
D = 6


def _fakes(with_action=True):
    import torch

    n = max(FRAMES) + 1
    demo = torch.arange(n * 50 * D, dtype=torch.float32).reshape(n, 50, D) * 0.001

    class FakeDataset:
        def __len__(self):
            return n

        def __getitem__(self, i):
            item = {
                "observation.state": torch.full((D,), float(i)),   # marks the frame
                "index": torch.tensor(i),
            }
            if with_action:
                item["action"] = demo[i]
            return item

    def preprocessor(batch):
        out = {"observation.state": batch["observation.state"]}
        if "action" in batch:
            out["action"] = batch["action"]
        return out

    class FakePi05:
        def predict_action_chunk(self, pre):
            b = pre["observation.state"].shape[0]
            return torch.full((b, 50, D), -7.0)

    class FakeRLToken:
        def encode(self, prefix):
            return torch.zeros(prefix.shape[0], 8)

    class FakeCapture:
        def consume(self):
            return torch.zeros(len(FRAMES), 1)

    return FakeDataset(), preprocessor, FakePi05(), FakeRLToken(), FakeCapture(), demo


def _run(module):
    ds, pre, pi, tok, cap, demo = _fakes()
    out = module._encode_episode(
        pi, tok, pre, cap, ds, list(FRAMES),
        chunk_length=C, action_dim=D, proprio_dim=D,
        batch_size=len(FRAMES), num_workers=0, device="cpu",
        empty_cache_every=10_000, task_str="t", ep_id=3, episode_success=True,
        episode_last_frame=LAST, stride=2,
    )
    return out, demo


def test_next_state_is_x_t_plus_C_not_the_next_anchor():
    import torch

    module = pytest.importorskip("rlt_so101_dual.cli.build_transition_cache_v2")
    out, _ = _run(module)

    # Only frames 0 and 2 satisfy start + C <= 12; 4..12 are bootstrap-only.
    assert len(out) == 2

    def frame_of(state_vec):
        return int(state_vec[-D:][0].item())   # proprio was filled with the frame index

    assert frame_of(out[0]["state_vec"]) == 0
    assert frame_of(out[0]["next_state_vec"]) == 0 + C, "must bootstrap from x_10, not x_2"
    assert frame_of(out[1]["state_vec"]) == 2
    assert frame_of(out[1]["next_state_vec"]) == 2 + C

    # Terminal is the chunk whose next_state is the episode's last frame.
    assert out[0]["done"].item() == 0.0
    assert out[1]["done"].item() == 1.0
    assert out[1]["reward_seq"].sum().item() == pytest.approx(1.0)
    assert out[0]["reward_seq"].sum().item() == 0.0
    for tr in out:
        assert int(tr["actual_steps"]) == C
        assert int(tr["episode_id"]) == 3


def test_exec_chunk_is_the_demo_action_not_the_vla_reference():
    """exec_chunk and ref_chunk must be different tensors.

    They used to be the same one, which leaves the critic with zero action
    variation: Q(s,a) becomes V(s) and the -Q term can never prefer one action
    over another. The demonstrator's action is what was actually executed, and
    the preprocessor already puts it in the same QUANTILES-normalised space as
    the (un-postprocessed) VLA chunk.
    """
    import torch

    module = pytest.importorskip("rlt_so101_dual.cli.build_transition_cache_v2")
    out, demo = _run(module)

    for tr, start in zip(out, (0, 2)):
        torch.testing.assert_close(tr["exec_chunk"], demo[start, :C, :D])
        torch.testing.assert_close(tr["ref_chunk"], torch.full((C, D), -7.0))
        assert not torch.equal(tr["exec_chunk"], tr["ref_chunk"])


def test_missing_action_key_fails_loudly():
    """Silently duplicating ref_chunk is what produced the degenerate cache."""
    module = pytest.importorskip("rlt_so101_dual.cli.build_transition_cache_v2")
    ds, _, pi, tok, cap, _ = _fakes(with_action=False)

    with pytest.raises(KeyError, match="delta_timestamps"):
        module._encode_episode(
            pi, tok, lambda b: {"observation.state": b["observation.state"]},
            cap, ds, list(FRAMES),
            chunk_length=C, action_dim=D, proprio_dim=D,
            batch_size=len(FRAMES), num_workers=0, device="cpu",
            empty_cache_every=10_000, task_str="t", ep_id=0, episode_success=True,
            episode_last_frame=LAST, stride=2,
        )
