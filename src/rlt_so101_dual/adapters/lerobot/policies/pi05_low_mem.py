"""Low-RAM PI05 loader for 32GB-class capture PCs.

Stock ``PI05Policy.from_pretrained`` peaks ~26GB RSS (fp32 construct + full
``load_file`` dict) and gets OOM-killed on a 30GB machine.

This helper:
  1. Builds the module under ``accelerate.init_empty_weights`` (meta tensors)
  2. Streams safetensors into real CPU/CUDA storage one tensor at a time
  3. Recomputes non-persistent buffers (RoPE ``inv_freq``, ``position_ids``)
"""

from __future__ import annotations

import gc
import logging
import resource
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


def _rss_gb() -> float:
    # Linux: ru_maxrss is KiB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _remap_state_dict_keys(model: Any, original_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Match PI05Policy.from_pretrained key remapping (prefix + openpi fixes)."""
    fixed = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)
    remapped: dict[str, torch.Tensor] = {}
    for key, value in fixed.items():
        if not key.startswith("model."):
            remapped[f"model.{key}"] = value
        else:
            remapped[key] = value
    return remapped


def _restore_integer_index_buffers(model: Any) -> None:
    """Rebuild ``position_ids`` if still meta or wrongly cast away from Long."""
    fixed = 0
    for module in model.modules():
        buf = getattr(module, "position_ids", None)
        if not isinstance(buf, torch.Tensor):
            continue
        needs = buf.device.type == "meta" or buf.dtype not in (torch.int32, torch.int64)
        if not needs:
            continue
        n = int(buf.shape[-1])
        device = torch.device("cpu") if buf.device.type == "meta" else buf.device
        restored = torch.arange(n, device=device, dtype=torch.long).expand(buf.shape).contiguous()
        module.register_buffer("position_ids", restored, persistent=False)
        fixed += 1
    if fixed:
        log.info("restored %d position_ids buffer(s) to torch.long", fixed)


def _recompute_rotary_buffers(model: Any, device: str | torch.device) -> None:
    """Meta init + zero-fill leaves RoPE ``inv_freq`` as zeros → garbage attention / wild arms."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    device = torch.device(device)
    fixed = 0
    for module in model.modules():
        if not hasattr(module, "inv_freq") or not hasattr(module, "config"):
            continue
        if not hasattr(module, "rope_type") and not (
            hasattr(module.config, "rope_parameters") and module.config.rope_parameters
        ):
            continue
        rope_type = getattr(module, "rope_type", None)
        if rope_type is None:
            rope_type = module.config.rope_parameters.get("rope_type", "default")
        if rope_type == "default" and hasattr(module, "compute_default_rope_parameters"):
            rope_init_fn = module.compute_default_rope_parameters
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, attention_scaling = rope_init_fn(module.config, device)
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        module.attention_scaling = attention_scaling
        fixed += 1
    if fixed:
        log.info("recomputed %d RoPE inv_freq buffer(s) on %s", fixed, device)


def _tie_paligemma_embeddings(model: Any) -> None:
    """Ensure language ``embed_tokens`` shares storage with ``lm_head`` when tied."""
    try:
        lm = model.model.paligemma_with_expert.paligemma
        embed = lm.model.language_model.embed_tokens
        head = lm.lm_head
    except AttributeError:
        return
    if embed is None or head is None:
        return
    if embed.weight.data_ptr() == head.weight.data_ptr():
        return
    # Prefer the non-zero / loaded head; SFT usually stores both or lm_head.
    if head.weight.float().abs().mean().item() >= embed.weight.float().abs().mean().item():
        embed.weight = head.weight
    else:
        head.weight = embed.weight
    log.info("tied paligemma lm_head ↔ embed_tokens")


def _materialize_from_safetensors(
    model: Any,
    safetensors_path: Path,
    *,
    device: str,
    strict: bool,
) -> None:
    """Replace meta parameters with real tensors loaded from the checkpoint."""
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as fh:
        file_keys = list(fh.keys())

    # One file tensor may map to multiple model keys (tied lm_head / embed_tokens).
    # Previously we skipped any remap with len!=1, which zeroed lm_head and broke VLA.
    file_key_by_model_key: dict[str, str] = {}
    for fk in file_keys:
        remapped = _remap_state_dict_keys(model, {fk: torch.empty(0)})
        for mk in remapped:
            file_key_by_model_key[mk] = fk

    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    expected = set(named_params) | set(named_buffers)

    loaded = 0
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as fh:
        # Load each unique file key once, assign to every model key that points at it.
        model_keys_by_file: dict[str, list[str]] = {}
        for mk, fk in file_key_by_model_key.items():
            model_keys_by_file.setdefault(fk, []).append(mk)

        for file_key, model_keys in model_keys_by_file.items():
            tensor = fh.get_tensor(file_key)
            for model_key in model_keys:
                if model_key not in named_params and model_key not in named_buffers:
                    continue
                set_module_tensor_to_device(
                    model,
                    model_key,
                    device,
                    value=tensor,
                    dtype=tensor.dtype,
                )
                loaded += 1
            del tensor

    not_in_ckpt = sorted(k for k in expected if k not in file_key_by_model_key)
    # Skip buffers we recompute explicitly after load.
    skip_zero = {
        k
        for k in not_in_ckpt
        if k.endswith("inv_freq")
        or k.endswith("original_inv_freq")
        or k.endswith("position_ids")
    }
    for model_key in not_in_ckpt:
        if model_key in skip_zero:
            continue
        ref = named_params.get(model_key, named_buffers.get(model_key))
        if ref is None:
            continue
        value = torch.zeros(ref.shape, dtype=ref.dtype if ref.dtype.is_floating_point else torch.float32)
        if not ref.dtype.is_floating_point:
            # Should not happen for remaining keys; keep dtype.
            value = torch.zeros(ref.shape, dtype=ref.dtype)
        set_module_tensor_to_device(model, model_key, device, value=value, dtype=value.dtype)
        del value

    if strict and (set(not_in_ckpt) - skip_zero):
        raise RuntimeError(
            f"strict load failed: missing_in_ckpt={sorted(set(not_in_ckpt) - skip_zero)}"
        )
    log.info(
        "meta→device load: wrote %d tensor assignments (+%d deferred recomputes) → %s from %s",
        loaded,
        len(skip_zero),
        device,
        safetensors_path,
    )


def load_pi05_policy_low_mem(
    pretrained_path: str | Path,
    *,
    config: Any,
    revision: str | None = None,
    strict: bool = False,
) -> Any:
    """Build + load a PI05Policy with a much lower CPU peak than stock from_pretrained."""
    from accelerate import init_empty_weights
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    del revision  # unused; kept for call-site parity with from_pretrained

    pretrained_path = Path(pretrained_path)
    weight_path = pretrained_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing {weight_path}")

    target_device = str(getattr(config, "device", "cuda") or "cuda")
    # PI05Policy.__init__ ends with ``self.model.to(config.device)``. Under
    # init_empty_weights that .to() must stay on meta (not cpu), otherwise it
    # tries to copy meta→cpu and crashes / materialises.
    config.device = "meta"
    if getattr(config, "gradient_checkpointing", False):
        config.gradient_checkpointing = False

    log.info(
        "Loading PI05 (low-mem/meta) from %s | peak RSS before construct=%.1f GiB",
        pretrained_path,
        _rss_gb(),
    )
    print(
        "The PI05 model is a direct port of the OpenPI implementation.\n"
        "RLT low-mem loader: meta init + streamed safetensors (avoids ~26GB RSS spike)."
    )

    with init_empty_weights():
        model = PI05Policy(config)
    config.device = target_device
    log.info("PI05 meta-constructed | peak RSS=%.1f GiB", _rss_gb())

    _materialize_from_safetensors(model, weight_path, device="cpu", strict=strict)
    gc.collect()
    log.info("PI05 weights on CPU | peak RSS=%.1f GiB", _rss_gb())

    # Do NOT call to_bfloat16_for_selected_params() after loading: that helper
    # does module.to(bf16) then promotes vision back to fp32, which quantizes
    # already-correct checkpoint weights and hurts deploy accuracy. Stock
    # from_pretrained only runs it on random init, then overwrites from disk.
    _restore_integer_index_buffers(model)
    _recompute_rotary_buffers(model, "cpu")
    _tie_paligemma_embeddings(model)

    if target_device != "cpu":
        model.to(target_device)
        config.device = target_device
        _restore_integer_index_buffers(model)
        _recompute_rotary_buffers(model, target_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("PI05 moved to %s | peak RSS=%.1f GiB", target_device, _rss_gb())

    model.eval()
    return model