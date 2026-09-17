from __future__ import annotations

import torch

from rlt_so101_dual.core.utils import postprocess_prefix_tokens


B, D = 2, 16
NUM_IMG = 768
NUM_LANG = 200
TOTAL = NUM_IMG + NUM_LANG  # 968 — matches pi0.5 default with 3 cameras at 224/14


def _make_tokens() -> torch.Tensor:
    return torch.randn(B, TOTAL, D)


def test_image_only_false_pool_zero_passthrough():
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=NUM_IMG, pool_size=0)
    assert out.shape == (B, TOTAL, D)
    assert torch.equal(out, tokens)


def test_image_only_false_pool_64_pools_full_sequence():
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=NUM_IMG, pool_size=64)
    assert out.shape == (B, 64, D)


def test_image_only_true_pool_zero_slices_to_image():
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=True, num_image_tokens=NUM_IMG, pool_size=0)
    assert out.shape == (B, NUM_IMG, D)
    assert torch.equal(out, tokens[:, :NUM_IMG, :])


def test_image_only_true_pool_64_excludes_language_tokens():
    """Set language tokens to NaN; output must remain finite, proving they were dropped."""
    tokens = torch.zeros(B, TOTAL, D)
    tokens[:, :NUM_IMG, :] = torch.randn(B, NUM_IMG, D)
    tokens[:, NUM_IMG:, :] = float("nan")

    out = postprocess_prefix_tokens(tokens, image_only=True, num_image_tokens=NUM_IMG, pool_size=64)
    assert out.shape == (B, 64, D)
    assert torch.isfinite(out).all()


def test_image_only_false_pool_64_does_include_language_tokens():
    """Sanity check the inverse: without image_only, NaN language tokens must contaminate output."""
    tokens = torch.zeros(B, TOTAL, D)
    tokens[:, :NUM_IMG, :] = torch.randn(B, NUM_IMG, D)
    tokens[:, NUM_IMG:, :] = float("nan")

    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=NUM_IMG, pool_size=64)
    assert out.shape == (B, 64, D)
    assert torch.isnan(out).any()


def test_pool_skipped_when_seq_already_short():
    """Guard: if sequence is shorter than pool_size, no pooling happens."""
    tokens = torch.randn(B, 32, D)
    out = postprocess_prefix_tokens(tokens, image_only=False, num_image_tokens=0, pool_size=64)
    assert out.shape == (B, 32, D)
    assert torch.equal(out, tokens)


def test_image_only_then_pool_skipped_when_image_count_below_pool():
    """When image_only narrows the sequence below pool_size, pooling is bypassed."""
    tokens = _make_tokens()
    out = postprocess_prefix_tokens(tokens, image_only=True, num_image_tokens=32, pool_size=64)
    assert out.shape == (B, 32, D)
    assert torch.equal(out, tokens[:, :32, :])
