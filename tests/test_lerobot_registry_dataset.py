from types import SimpleNamespace

import torch

import rlt_so101_dual.adapters.lerobot.registry as registry
from rlt_so101_dual.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset


def test_register_loads_chunk_transition_dataset_from_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        [
            {
                "state_vec": torch.zeros(3),
                "exec_chunk": torch.zeros(2, 1),
            }
        ],
        cache_dir / "chunk_transitions_train.pt",
    )

    import lerobot.datasets.factory as dataset_factory

    registry._REGISTERED = False
    registry.register()

    cfg = SimpleNamespace(dataset=SimpleNamespace(repo_id=str(cache_dir)))
    dataset = dataset_factory.make_dataset(cfg)

    assert isinstance(dataset, ChunkTransitionDataset)
    assert len(dataset) == 1
    assert dataset[0]["state_vec"].shape == (3,)
