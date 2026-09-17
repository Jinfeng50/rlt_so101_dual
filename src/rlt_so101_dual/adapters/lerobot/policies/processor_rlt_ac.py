from __future__ import annotations

from typing import Any

import torch

from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_common import load_sft_pi05_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME


class _BypassOnChunkTransitionPipeline(PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]):
    """Short-circuit on precomputed chunk-transition training batches.

    ChunkTransitionDataset emits dicts already encoded through pi05 + RL Token;
    those dicts lack the raw observation keys the pi05 steps expect. The
    ``state_vec`` sentinel lets training bypass every step without losing the
    saved pi05 JSON that ``lerobot-record`` reloads at deploy. At deploy,
    ``PolicyProcessorPipeline.from_pretrained`` returns a vanilla pipeline (this
    subclass is not preserved), so the full pi05 step list runs.
    """

    def __call__(self, data: Any) -> Any:
        if isinstance(data, dict) and "state_vec" in data:
            return data
        return super().__call__(data)


def make_rlt_ac_pre_post_processors(
    config: ChunkACPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Deploy-parity AC pre/post-processor pair.

    The saved preprocessor JSON is byte-identical to the SFT pi05's (modulo
    the portable tokenizer override) so ``lerobot-record --policy.path=<ac>``
    reproduces SFT QUANTILES normalization on raw observations. ``dataset_stats``
    is part of the factory protocol but ignored — the SFT stats are authoritative.
    """
    del dataset_stats
    pi05_pre, pi05_post = load_sft_pi05_processors(config.vla_pretrained_path, config.tokenizer_path)
    bypass_pre = _BypassOnChunkTransitionPipeline(
        steps=list(pi05_pre.steps),
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    return bypass_pre, pi05_post
