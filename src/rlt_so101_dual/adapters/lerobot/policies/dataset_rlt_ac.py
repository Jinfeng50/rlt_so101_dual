from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class _EpisodeIndexShim:
    """Minimum LeRobotDataset.meta.episodes shape used by EpisodeAwareSampler."""

    dataset_from_index: list[int]
    dataset_to_index: list[int]


@dataclass
class _MetaShim:
    """Minimum LeRobotDataset.meta surface used by lerobot-train.

    lerobot-train pulls dataset_stats off `meta.stats` and frame ranges off
    `meta.episodes`. Other attributes are duck-typed where possible.
    """

    stats: dict[str, dict[str, torch.Tensor]]
    episodes: _EpisodeIndexShim
    fps: int = 30
    # ChunkTransitionDataset feeds precomputed transitions; policy.input_features /
    # output_features are set by ChunkACPolicyConfig.validate_features, not derived
    # from the dataset. So expose an empty features dict.
    features: dict = field(default_factory=dict)


class ChunkTransitionDataset(Dataset):
    """Dataset over precomputed chunk transitions for ChunkACPolicy training.

    The cache directory must contain `chunk_transitions_{split}.pt`, a Python
    list of dicts with the fields produced by `build_transition_cache.py`:
      state_vec      (state_dim,)
      exec_chunk     (C, action_dim)
      ref_chunk      (C, action_dim)
      reward_seq     (C,)
      next_state_vec (state_dim,)
      next_ref_chunk (C, action_dim)
      done           ()
      intervention   ()
      actual_steps   ()
      (optional) source, episode_id, is_critical

    The samples are returned exactly as stored (dicts of tensors); the
    flattening for `exec_chunk_flat` / `ref_chunk_flat` / `next_ref_flat` is
    deferred to ChunkACPolicy.forward.
    """

    def __init__(self, cache_dir: str | pathlib.Path, split: str = "train"):
        cache = pathlib.Path(cache_dir)
        path = cache / f"chunk_transitions_{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No transition cache at {path}")
        self._transitions: list[dict[str, torch.Tensor]] = torch.load(
            path, weights_only=False, map_location="cpu"
        )
        if not self._transitions:
            raise ValueError(f"empty cache at {path}")
        self.num_frames = len(self._transitions)
        self.num_episodes = 1  # cache is flattened; treat as one mega-episode.
        self.fps = 30

        # Build minimal meta shim. ACTION/STATE stats are absent (the cache is
        # already encoded); pi05 preprocessor pipeline at deploy-time will use
        # the stats embedded in the policy's own saved processor JSONs.
        self.meta = _MetaShim(
            stats={},
            episodes=_EpisodeIndexShim(
                dataset_from_index=[0],
                dataset_to_index=[self.num_frames],
            ),
            fps=self.fps,
        )
        # Inferred from the first sample.
        sample = self._transitions[0]
        self.features: dict[str, dict[str, Any]] = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                self.features[k] = {"shape": tuple(v.shape), "dtype": str(v.dtype)}

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._transitions[idx]
