"""CLI: build transition cache for offline RL."""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import random
import sys
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rlt_so101_dual.adapters.lerobot.policies.action_modifier import PrefixOutputCapture
from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
from rlt_so101_dual.adapters.lerobot.policies.processor_rlt_token import make_rlt_token_pre_post_processors
from rlt_so101_dual.adapters.lerobot.offline_dataset import (
    _encoded_to_transitions,
    build_overlap_frame_indices,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo-dataset-repo-id", required=True)
    p.add_argument("--demo-dataset-root", required=True)
    p.add_argument("--rl-token-policy-path", required=True)
    p.add_argument("--vla-pretrained-path", required=True,
                   help="SFT pi05 ckpt dir — preprocessor source. Must match deploy.")
    p.add_argument("--tokenizer-path", default=None,
                   help="PaliGemma tokenizer repo id or local snapshot path for the SFT preprocessor.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-instruction", default="screw")
    p.add_argument("--chunk-length", type=int, default=10)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap on episodes to process (debug).")
    p.add_argument("--video-backend", default="pyav",
                   help="Video decoder backend passed to LeRobotDataset.")
    p.add_argument("--tolerance-s", type=float, default=0.04,
                   help="Timestamp tolerance passed to LeRobotDataset video decoding.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--empty-cache-every", type=int, default=4,
                   help="Call torch.cuda.empty_cache() every N batches.")
    return p.parse_args()

def _log(msg: str) -> None:
    """Unbuffered timestamped log line."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def _get_episode_success(dataset: LeRobotDataset, episode_idx: int) -> bool:
    """Per-episode success flag, matching RLTDemoDataset.get_episode_success semantics."""
    raw = dataset.meta.episodes["episode_success"][episode_idx]
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized == "success":
            return True
        if normalized == "failure":
            return False
    elif isinstance(raw, bool):
        return raw
    elif isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(raw)
    raise ValueError(f"Unrecognized episode_success value for episode {episode_idx}: {raw!r}")

def _encode_episode(
    pi05,
    rl_token,
    preprocessor,
    capture: PrefixOutputCapture,
    dataset: LeRobotDataset,
    frame_indices: list[int],
    chunk_length: int,
    action_dim: int,
    proprio_dim: int,
    batch_size: int,
    num_workers: int,
    device: str,
    empty_cache_every: int,
    task_str: str,
    ep_id: int,
    episode_success: bool,
    episode_last_frame: int,
    stride: int,
) -> list[dict[str, Tensor]]:
    """Encode every frame in `frame_indices`; build chunk-level TD transitions.

    The transition semantics are the paper's, and they are *not* "adjacent
    encoded frames": with a stride-2 anchor pattern, frame i+1 in the encoded
    array is x_{t+2}, while a chunk transition whose action is a_t:t+C-1 must
    bootstrap from x_{t+C}. Building (x_0, a_0:9, x_2) instead of
    (x_0, a_0:9, x_10) gives the critic a next_state two control steps away
    while discounting it by gamma^C, and mislabels which chunk is terminal.

    build_overlap_frame_indices returns a *mixed* set -- start anchors, the
    terminal anchor, and the bootstrap states x_{t+C} those anchors need -- so
    not every encoded frame is a transition start. _encoded_to_transitions is
    the repo's existing implementation of that mapping; delegate rather than
    reimplement, so the offline and online paths cannot drift apart again.
    """
    out: list[dict[str, Tensor]] = []
    if not frame_indices:
        return out

    loader = DataLoader(
        Subset(dataset, frame_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=False,
    )

    state_vecs: list[Tensor] = []
    ref_chunks: list[Tensor] = []
    exec_chunks: list[Tensor] = []
    t_ep = time.time()
    for batch_i, batch in enumerate(loader):
        t_b = time.time()
        if "task" not in batch:
            batch["task"] = [task_str] * batch["observation.state"].shape[0]
        pre = preprocessor(batch)
        with torch.no_grad():
            vla_chunk = pi05.predict_action_chunk(pre)
            prefix = capture.consume()
            z = rl_token.encode(prefix.to(torch.float32))
        if z.dim() == 3:
            z = z.mean(dim=1)
        proprio = pre["observation.state"][:, :proprio_dim].detach().to("cpu")
        state_vec = torch.cat([z.detach().to("cpu"), proprio], dim=-1)
        ref_chunk = vla_chunk[:, :chunk_length, :action_dim].detach().to("cpu")
        # The executed action is the demonstrator's, not the VLA's prediction.
        # Storing ref_chunk for both leaves the critic with zero action
        # variation, so Q(s,a) collapses to V(s) and -Q can never prefer one
        # action over another -- the exact failure rlt-so101-dual-critic-health
        # reports as DEGENERATE. Both live in the same QUANTILES-normalised
        # space: the preprocessor normalises "action", and vla_chunk skips the
        # postprocessor that would un-normalise it.
        if "action" not in pre:
            raise KeyError(
                "preprocessor output has no 'action'; the demo dataset must be built with "
                "delta_timestamps={'action': ...} so each frame carries its action chunk. "
                "Without it exec_chunk would have to duplicate ref_chunk, which trains a "
                "degenerate critic."
            )
        exec_chunk = pre["action"][:, :chunk_length, :action_dim].detach().to("cpu")
        state_vecs.append(state_vec)
        ref_chunks.append(ref_chunk)
        exec_chunks.append(exec_chunk)
        del vla_chunk, prefix, z, pre
        if (batch_i + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()
        if batch_i == 0 or (batch_i + 1) % 4 == 0:
            elapsed = time.time() - t_b
            cum = time.time() - t_ep
            _log(f"    ep{ep_id} batch {batch_i+1}/{len(loader)} bs={batch_size} dt={elapsed:.2f}s cum={cum:.1f}s")

    state_vecs_t = torch.cat(state_vecs, dim=0)
    ref_chunks_t = torch.cat(ref_chunks, dim=0)
    exec_chunks_t = torch.cat(exec_chunks, dim=0)

    encoded = [
        (state_vecs_t[i], ref_chunks_t[i], exec_chunks_t[i])
        for i in range(state_vecs_t.shape[0])
    ]
    transitions = _encoded_to_transitions(
        encoded,
        frame_indices,
        episode_last_frame=episode_last_frame,
        chunk_length=chunk_length,
        stride=stride,
        episode_success=episode_success,
        source=0,
        episode_id=ep_id,
        is_critical=1.0,
    )
    # ChunkTransitionDataset stores plain dicts, so unpack the dataclass by
    # field rather than dataclasses.asdict(), which deep-copies every tensor.
    return [{f.name: getattr(t, f.name) for f in dataclasses.fields(t)} for t in transitions]

def _save_partial(out_dir: pathlib.Path, split: str, transitions: list, label: str) -> None:
    path = out_dir / f"chunk_transitions_{split}.pt"
    tmp = out_dir / f".chunk_transitions_{split}.tmp.pt"
    torch.save(transitions, tmp)
    tmp.replace(path)
    _log(f"  [{label}] checkpointed {len(transitions)} transitions -> {path.name}")

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"args: {vars(args)}")
    _log(f"load RLTokenPolicy from {args.rl_token_policy_path}")
    RLTokenPolicyConfig.ensure_registered()
    policy = RLTokenPolicy.from_pretrained(args.rl_token_policy_path).to(args.device).eval()
    cfg = policy.config
    # Override the policy's recorded vla path so the preprocessor we load is the
    # SFT pi05's, even if the RL Token ckpt was trained against a different one.
    cfg.vla_pretrained_path = args.vla_pretrained_path
    if args.tokenizer_path is not None:
        cfg.tokenizer_path = args.tokenizer_path

    _log(f"load preprocessor from SFT pi05 dir {args.vla_pretrained_path}")
    preprocessor, _ = make_rlt_token_pre_post_processors(config=cfg)

    _log(f"load dataset {args.demo_dataset_repo_id} root={args.demo_dataset_root}")
    # The VLA horizon's worth of actions is loaded; only the first
    # args.chunk_length are stored. LeRobot pads the tail of a short window and
    # flags it, but _encoded_to_transitions drops any anchor with
    # start + C > episode_last_frame, so no padded step reaches a transition.
    delta = {"action": [i / 30.0 for i in range(cfg.chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=args.demo_dataset_repo_id,
        root=args.demo_dataset_root,
        delta_timestamps=delta,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )
    n_episodes = dataset.num_episodes
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)
    _log(f"episodes: {n_episodes} of {dataset.num_episodes}; batch_size={args.batch_size} num_workers={args.num_workers}")

    pi05 = policy._pi05
    rl_token = policy.rl_token

    capture = PrefixOutputCapture(
        token_pool_size=cfg.token_pool_size,
        image_only=cfg.image_only,
        num_image_tokens=policy._num_image_tokens,
    )
    capture.attach(policy._pi05)
    try:
        ep_indices = list(range(n_episodes))
        random.shuffle(ep_indices)
        n_train = int(args.train_ratio * n_episodes)
        train_eps = ep_indices[:n_train]
        val_eps = ep_indices[n_train:]
        _log(f"split: train={len(train_eps)} val={len(val_eps)}")

        t_start = time.time()
        for split_name, eps in (("train", train_eps), ("val", val_eps)):
            all_tx: list[dict[str, Tensor]] = []
            for k, ep_id in enumerate(eps):
                ep_meta = dataset.meta.episodes
                ep_from = int(ep_meta["dataset_from_index"][ep_id])
                ep_to = int(ep_meta["dataset_to_index"][ep_id])
                # C, not the VLA horizon: the anchor pattern must include the
                # x_{t+C} bootstrap states and the terminal anchor for the RL
                # chunk length actually stored in the transitions.
                frame_indices = build_overlap_frame_indices(
                    episode_start=ep_from,
                    episode_stop=ep_to,
                    chunk_length=args.chunk_length,
                    stride=args.frame_stride,
                )
                episode_success = _get_episode_success(dataset, ep_id)
                _log(f"  [{split_name}] ep {k+1}/{len(eps)} id={ep_id} frames={ep_to-ep_from} chunks={len(frame_indices)} success={episode_success} (total transitions={len(all_tx)}, wall={time.time()-t_start:.0f}s)")
                ep_tx = _encode_episode(
                    pi05=pi05,
                    rl_token=rl_token,
                    preprocessor=preprocessor,
                    capture=capture,
                    dataset=dataset,
                    frame_indices=frame_indices,
                    chunk_length=args.chunk_length,
                    action_dim=cfg.action_dim,
                    proprio_dim=cfg.proprio_dim,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                    empty_cache_every=args.empty_cache_every,
                    task_str=args.task_instruction,
                    ep_id=ep_id,
                    episode_success=episode_success,
                    episode_last_frame=ep_to - 1,
                    stride=args.frame_stride,
                )
                all_tx.extend(ep_tx)
                if (k + 1) % 5 == 0 or (k + 1) == len(eps):
                    _save_partial(out_dir, split_name, all_tx, f"{split_name} ep {k+1}/{len(eps)}")
    finally:
        capture.detach()

    _log(f"done, total wall {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
