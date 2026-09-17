from __future__ import annotations

from typing import Any

import torch

from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_common import load_sft_pi05_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_rlt_token_pre_post_processors(
    config: RLTokenPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Reuse SFT pi05's preprocessor so RL Token training sees the deploy normalization.

    Anchoring to the SFT ckpt instead of auto-computing stats from the RL Token
    training dataset eliminates the QUANTILES drift that previously caused a
    train/deploy distribution shift on observation.state.

    ``dataset_stats`` is part of the factory protocol but unused — the SFT
    stats are authoritative.
    """
    del dataset_stats
    return load_sft_pi05_processors(config.vla_pretrained_path, config.tokenizer_path)
