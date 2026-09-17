from __future__ import annotations

import os
import pathlib

import pytest
import torch

from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_ac import (
    _BypassOnChunkTransitionPipeline,
    make_rlt_ac_pre_post_processors,
)
from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_common import load_sft_pi05_processors
from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors

_SFT_PI05_DIR_ENV = os.environ.get("RLT_SO101_DUAL_TEST_SFT_PI05_DIR") or os.environ.get(
    "RLT_SINGLE_TEST_SFT_PI05_DIR"
)
_SFT_PI05_DIR = pathlib.Path(_SFT_PI05_DIR_ENV) if _SFT_PI05_DIR_ENV else None

_HAS_SFT = (
    _SFT_PI05_DIR is not None
    and _SFT_PI05_DIR.exists()
    and (_SFT_PI05_DIR / "policy_preprocessor.json").exists()
    and (_SFT_PI05_DIR / "policy_postprocessor.json").exists()
)

requires_sft = pytest.mark.skipif(
    not _HAS_SFT,
    reason="Set RLT_SO101_DUAL_TEST_SFT_PI05_DIR to a pi0.5 checkpoint with processor files",
)


def test_load_sft_pi05_processors_rejects_empty_path():
    with pytest.raises(ValueError, match="vla_pretrained_path"):
        load_sft_pi05_processors("")


@requires_sft
def test_load_sft_pi05_processors_returns_pi05_steps():
    pre, post = load_sft_pi05_processors(str(_SFT_PI05_DIR))
    step_names = [type(s).__name__ for s in pre.steps]
    assert "NormalizerProcessorStep" in step_names
    assert "TokenizerProcessorStep" in step_names
    assert "Pi05PrepareStateTokenizerProcessorStep" in step_names
    assert "DeviceProcessorStep" in step_names
    assert "UnnormalizerProcessorStep" in [type(s).__name__ for s in post.steps]


@requires_sft
def test_load_sft_pi05_processors_overrides_tokenizer():
    pre, _ = load_sft_pi05_processors(str(_SFT_PI05_DIR))
    tok = next(s for s in pre.steps if type(s).__name__ == "TokenizerProcessorStep")
    assert getattr(tok, "tokenizer_name", None) == "google/paligemma-3b-pt-224"


@requires_sft
def test_rlt_token_factory_yields_pi05_pipeline():
    cfg = RLTokenPolicyConfig(vla_pretrained_path=str(_SFT_PI05_DIR))
    cfg.validate_features()
    pre, post = make_rlt_token_pre_post_processors(cfg)
    assert [type(s).__name__ for s in pre.steps] == [
        "RenameObservationsProcessorStep",
        "AddBatchDimensionProcessorStep",
        "NormalizerProcessorStep",
        "Pi05PrepareStateTokenizerProcessorStep",
        "TokenizerProcessorStep",
        "DeviceProcessorStep",
    ]


@requires_sft
def test_rlt_ac_factory_returns_bypass_pipeline_with_pi05_steps():
    cfg = ChunkACPolicyConfig(
        vla_pretrained_path=str(_SFT_PI05_DIR), rl_token_pretrained_path="/dummy"
    )
    cfg.validate_features()
    pre, post = make_rlt_ac_pre_post_processors(cfg)
    assert isinstance(pre, _BypassOnChunkTransitionPipeline)
    assert "NormalizerProcessorStep" in [type(s).__name__ for s in pre.steps]


@requires_sft
def test_bypass_short_circuits_on_state_vec_dict():
    cfg = ChunkACPolicyConfig(
        vla_pretrained_path=str(_SFT_PI05_DIR), rl_token_pretrained_path="/dummy"
    )
    cfg.validate_features()
    pre, _ = make_rlt_ac_pre_post_processors(cfg)
    payload = {
        "state_vec": torch.zeros(4, 2060),
        "exec_chunk": torch.zeros(4, 10, 12),
    }
    out = pre(payload)
    # Identity short-circuit; same dict reference, untouched.
    assert out is payload
    assert "state_vec" in out and out["state_vec"].shape == (4, 2060)


def test_bypass_subclass_directly():
    # Construct an empty subclass instance; verify the bypass behaves
    # without needing an SFT pi05 ckpt.
    pipeline = _BypassOnChunkTransitionPipeline(steps=[], name="test_bypass")
    payload = {"state_vec": torch.zeros(2, 3)}
    out = pipeline(payload)
    assert out is payload


def test_bypass_non_state_vec_dict_goes_through_steps():
    # Without state_vec, the parent __call__ runs. With zero steps, the
    # parent class simply returns the to_output(to_transition(data)).
    # We don't try to assert that here (it depends on lerobot internals);
    # we only assert no crash for the empty-step path.
    pipeline = _BypassOnChunkTransitionPipeline(steps=[], name="test_bypass")
    payload = {"observation.state": torch.zeros(1, 12)}
    # Should not short-circuit; parent class will call to_transition/to_output.
    # Either succeeds (returns something) or raises a specific lerobot error —
    # both are valid; we only check the bypass branch is NOT taken (out is not payload).
    try:
        out = pipeline(payload)
        assert out is not payload, "non-state_vec payload should not bypass"
    except (KeyError, AttributeError):
        # parent class raised because it couldn't find expected keys — that
        # confirms we did NOT bypass.
        pass
