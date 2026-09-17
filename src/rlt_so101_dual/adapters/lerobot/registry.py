"""Register RLT policies with LeRobot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REGISTERED = False

def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import lerobot.datasets.factory as dataset_factory
    import lerobot.policies.factory as factory

    from lerobot.configs.policies import PreTrainedConfig

    from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt import RLTPretrainedConfig
    from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
    from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig

    PreTrainedConfig._choice_registry["rlt"] = RLTPretrainedConfig
    PreTrainedConfig._choice_registry["rlt_ac"] = ChunkACPolicyConfig
    PreTrainedConfig._choice_registry["rlt_token"] = RLTokenPolicyConfig

    original_get_policy_class = factory.get_policy_class
    original_make_policy_config = factory.make_policy_config
    original_make_pre_post_processors = factory.make_pre_post_processors
    original_make_dataset = dataset_factory.make_dataset

    def get_policy_class(name: str):
        if name == "rlt_token":
            from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy

            return RLTokenPolicy
        if name == "rlt_ac":
            from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy

            return ChunkACPolicy
        if name == "rlt":
            from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt import RLTPretrainedPolicy

            return RLTPretrainedPolicy
        return original_get_policy_class(name)

    def make_policy_config(policy_type: str, **kwargs: Any):
        if policy_type == "rlt_token":
            return RLTokenPolicyConfig(**kwargs)
        if policy_type == "rlt_ac":
            return ChunkACPolicyConfig(**kwargs)
        if policy_type == "rlt":
            return RLTPretrainedConfig(**kwargs)
        return original_make_policy_config(policy_type, **kwargs)

    def make_pre_post_processors(policy_cfg, *args: Any, **kwargs: Any):
        if isinstance(policy_cfg, RLTokenPolicyConfig):
            from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors

            return make_rlt_token_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        if isinstance(policy_cfg, ChunkACPolicyConfig):
            from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_ac import make_rlt_ac_pre_post_processors

            return make_rlt_ac_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
        return original_make_pre_post_processors(policy_cfg, *args, **kwargs)

    def make_dataset(cfg):
        repo_id = getattr(cfg.dataset, "repo_id", None)
        cache_dir = Path(repo_id) if isinstance(repo_id, str) else None
        if cache_dir is not None and (cache_dir / "chunk_transitions_train.pt").exists():
            from rlt_so101_dual.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset

            return ChunkTransitionDataset(cache_dir, split="train")
        return original_make_dataset(cfg)

    factory.get_policy_class = get_policy_class
    factory.make_policy_config = make_policy_config
    factory.make_pre_post_processors = make_pre_post_processors
    dataset_factory.make_dataset = make_dataset
    _REGISTERED = True
