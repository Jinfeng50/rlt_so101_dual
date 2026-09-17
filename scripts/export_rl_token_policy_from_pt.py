#!/usr/bin/env python
"""Export demo_adapt_checkpoint.pt to an online-ready RLTokenPolicy directory.

Does not load π0.5 (avoids OOM on small GPUs). Writes:
  config.json + model.safetensors (keys prefixed with rl_token.)

Usage:
  python scripts/export_rl_token_policy_from_pt.py \\
    --pt outputs/rl_token_run/demo_adapt_checkpoint.pt \\
    --vla-path outputs/pi05_sft_run/checkpoints/020000/pretrained_model \\
    --out outputs/rl_token_policy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Path to demo_adapt_checkpoint.pt")
    parser.add_argument(
        "--vla-path",
        required=True,
        help="SFT pi0.5 pretrained_model dir (written into config.vla_pretrained_path).",
    )
    parser.add_argument("--out", required=True, help="Output RLTokenPolicy directory.")
    parser.add_argument("--device-in-config", default="cuda", help="device field stored in config.json")
    args = parser.parse_args()

    from rlt_so101_dual.adapters.lerobot import register
    from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig

    register()

    pt = Path(args.pt).expanduser().resolve()
    vla = Path(args.vla_path).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not pt.is_file():
        raise FileNotFoundError(pt)
    if not (vla / "config.json").is_file():
        raise FileNotFoundError(f"VLA dir missing config.json: {vla}")

    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    if "rl_token_state_dict" not in ckpt:
        raise KeyError(f"{pt} has no rl_token_state_dict; keys={list(ckpt.keys())}")

    sd = ckpt["rl_token_state_dict"]
    meta = ckpt.get("metadata") or {}
    num_rl_tokens = int(meta.get("num_rl_tokens") or 1)

    # Infer architecture from weights (match modeling_rlt._infer_rl_token_arch_from_ckpt).
    ff_dim = 4096
    enc_layers = 3
    dec_layers = 3
    for k, v in sd.items():
        if k.startswith("encoder.") and k.endswith(".linear1.weight"):
            ff_dim = int(v.shape[0])
            break
    enc_max = dec_max = -1
    for k in sd:
        if k.startswith("encoder.layers."):
            enc_max = max(enc_max, int(k.split(".")[2]))
        elif k.startswith("decoder.layers."):
            dec_max = max(dec_max, int(k.split(".")[2]))
    if enc_max >= 0:
        enc_layers = enc_max + 1
    if dec_max >= 0:
        dec_layers = dec_max + 1
    if "rl_token_embed" in sd:
        num_rl_tokens = int(sd["rl_token_embed"].shape[1])

    token_dim = int(sd["rl_token_embed"].shape[-1]) if "rl_token_embed" in sd else 2048

    cfg = RLTokenPolicyConfig(
        vla_pretrained_path=str(vla),
        device=args.device_in_config,
        vla_dtype="bfloat16",
        rl_token_dim=token_dim,
        rl_token_nhead=8,
        rl_token_enc_layers=enc_layers,
        rl_token_dec_layers=dec_layers,
        rl_token_ff_dim=ff_dim,
        rl_token_num_rl_tokens=num_rl_tokens,
        vla_ft_weight=0.0,
    )
    cfg._save_pretrained(out)

    # Policy safetensors keys are submodule-prefixed: rl_token.*
    prefixed = {f"rl_token.{k}": v.contiguous() for k, v in sd.items()}
    save_file(prefixed, str(out / "model.safetensors"))

    print(f"saved: {out}")
    print(f"  step={ckpt.get('step')} num_rl_tokens={num_rl_tokens} "
          f"enc={enc_layers} dec={dec_layers} ff={ff_dim} dim={token_dim}")
    print(f"  files: {sorted(p.name for p in out.iterdir())}")
    print(f"  n_tensors={len(prefixed)}")


if __name__ == "__main__":
    main()
