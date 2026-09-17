from __future__ import annotations

import pytest

from rlt_so101_dual.adapters.lerobot.demo_loader import RLTDemoDataset


class _FakeEpisodes:
    """Minimal stand-in for LeRobotDatasetMetadata.episodes.

    Supports ``in`` (via ``column_names``) and ``__getitem__`` column access,
    matching the calls made by ``RLTDemoDataset.get_episode_success``.
    """

    def __init__(self, labels: list):
        self._labels = list(labels)
        self.column_names = ["episode_success"]

    def __getitem__(self, column: str):
        if column != "episode_success":
            raise KeyError(column)
        return self._labels


class _FakeInnerDataset:
    class _Meta:
        def __init__(self, episodes):
            self.episodes = episodes

    def __init__(self, labels):
        self.meta = self._Meta(_FakeEpisodes(labels))


def _make_loader(labels: list) -> RLTDemoDataset:
    """Bypass RLTDemoDataset.__init__ (which requires a real LeRobotDataset)."""
    loader = RLTDemoDataset.__new__(RLTDemoDataset)
    loader._dataset = _FakeInnerDataset(labels)
    return loader


def test_get_episode_success_canonical_strings():
    loader = _make_loader(["success", "failure", "success"])
    assert loader.get_episode_success(0) is True
    assert loader.get_episode_success(1) is False
    assert loader.get_episode_success(2) is True


def test_get_episode_success_accepts_bool_and_numeric():
    loader = _make_loader([True, False, 1, 0])
    assert loader.get_episode_success(0) is True
    assert loader.get_episode_success(1) is False
    assert loader.get_episode_success(2) is True
    assert loader.get_episode_success(3) is False


def test_get_episode_success_rejects_unknown_value():
    loader = _make_loader(["partial", "success"])
    with pytest.raises(ValueError, match="episode 0"):
        loader.get_episode_success(0)


def test_get_episode_success_missing_column_raises():
    """Strict: datasets without episode_success must be relabeled before use."""

    class _EmptyEpisodes:
        def __getitem__(self, column: str):
            raise KeyError(column)

    loader = RLTDemoDataset.__new__(RLTDemoDataset)
    loader._dataset = type("_", (), {"meta": type("__", (), {"episodes": _EmptyEpisodes()})()})()
    with pytest.raises(KeyError):
        loader.get_episode_success(0)
