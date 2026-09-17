from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass
class RLModelPaths:
    vla_model: str
    rl_token_ckpt: str
    ac_ckpt: str
    config_path: str
    metrics_path: str | None


def resolve_hf_file(repo_id: str, path_in_repo: str) -> str:
    return hf_hub_download(repo_id=repo_id, filename=path_in_repo)


def resolve_local_or_hf(
    local_path: str,
    repo_id: str,
    path_in_repo: str,
    label: str,
) -> str:
    if local_path:
        return local_path
    if repo_id and path_in_repo:
        return resolve_hf_file(repo_id, path_in_repo)
    raise ValueError(
        f"{label} is required. Pass a local path or set --hf-repo with the matching repo-relative path."
    )


def resolve_rl_model_paths(args: argparse.Namespace) -> RLModelPaths:
    if args.no_rl:
        raise ValueError("RL model resolution requested while --no-rl is set")

    if not args.rl_vla_model:
        raise ValueError("--rl-vla-model is required unless --no-rl is set")
    rl_token_ckpt = resolve_local_or_hf(
        args.rl_token_ckpt,
        args.hf_repo,
        args.rl_token_path_in_repo,
        "--rl-token-ckpt",
    )
    ac_ckpt = resolve_local_or_hf(
        args.ac_ckpt,
        args.hf_repo,
        args.ac_path_in_repo,
        "--ac-ckpt",
    )
    config_path = resolve_local_or_hf(
        args.rl_config,
        args.hf_repo,
        args.rl_config_path_in_repo,
        "--rl-config",
    )
    metrics_path = args.ac_metrics or None
    if metrics_path is None:
        sibling_metrics = Path(ac_ckpt).parent / "metrics.json"
        if sibling_metrics.exists():
            metrics_path = str(sibling_metrics)
    if metrics_path is None and args.hf_repo and args.ac_metrics_path_in_repo:
        metrics_path = resolve_hf_file(args.hf_repo, args.ac_metrics_path_in_repo)

    return RLModelPaths(
        vla_model=args.rl_vla_model,
        rl_token_ckpt=rl_token_ckpt,
        ac_ckpt=ac_ckpt,
        config_path=config_path,
        metrics_path=metrics_path,
    )
