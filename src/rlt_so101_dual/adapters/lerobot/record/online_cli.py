"""CLI: rlt-so101-dual-online-train (sync online RL on SO101 dual-arm)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from rlt_so101_dual.core import shape_contract as sc
from rlt_so101_dual.adapters.lerobot.record.common import (
    build_dataset_argv,
    build_robot_argv,
    build_teleop_argv,
    configure_logging,
    load_robot_setup,
    preflight_motor_connections,
    remove_existing_dataset,
    resolve_run_paths,
    set_offline_env,
    stage_follower_calibrations,
    stage_leader_calibrations,
)
from rlt_so101_dual.adapters.lerobot.record.runner import prepare_lerobot_runtime

DEFAULT_DATASET_TAG = "online_rl"
# lerobot's sanity_check_dataset_name() requires the dataset repo_id to start
# with "eval_" whenever a `policy` config is passed to record() (its
# convention: policy-driven recording == evaluating that policy) -- online RL
# training always passes `--policy.type=rlt_ac`, so the actual dataset leaf
# name (see resolve_run_paths()'s `dataset_prefix` param) must satisfy this,
# even though this data is used for training (via the replay buffer), not
# pure evaluation. DEFAULT_DATASET_TAG above still names the per-run output
# folder (via --dataset-tag) and is unaffected.
DEFAULT_DATASET_NAME_PREFIX = "eval_online_rl"

def _parse_dims(raw: str) -> list[int]:
    """Parse "4" / "4,5" into [4, 5]; empty string into []."""
    return [int(x) for x in raw.replace(" ", "").split(",") if x]

def build_online_train_argv(args: argparse.Namespace, setup, paths, cal_dir: str, teleop_argv: list[str]) -> list[str]:
    argv = [
        "online_train",
        *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
        *teleop_argv,
        # Fresh policy construction: no --policy.path, so make_policy() builds
        # a brand-new ChunkACPolicy (random actor/critic init) instead of
        # loading a checkpoint. VLA + RL token are frozen pretrained backbones.
        "--policy.type=rlt_ac",
        f"--policy.vla_pretrained_path={args.vla_path}",
        f"--policy.rl_token_pretrained_path={args.rl_token_path}",
        f"--policy.tokenizer_path={args.tokenizer_path}",
        "--policy.phase_mode=manual",
        f"--policy.chunk_exec_steps={args.chunk_exec_steps}",
        f"--policy.vla_reuse_horizon={args.vla_reuse_horizon}",
        f"--policy.chunk_length={args.chunk_length}",
        f"--policy.action_dim={args.action_dim}",
        f"--policy.proprio_dim={args.proprio_dim}",
        f"--policy.actor_residual_to_ref={'true' if args.actor_residual_to_ref else 'false'}",
        # With residual_to_ref, mu = ref + delta unconditionally -- the output
        # always gets the true (undropped) ref added back as a bias, even on
        # a dropout-masked training sample. So input-side reference dropout
        # can no longer force "independent action generation" the way it does
        # for the paper's non-residual actor (mu = net(state, ref) directly,
        # no ref shortcut); it would just be extra training noise with none
        # of its intended effect. Force 0 whenever residual_to_ref is on.
        f"--policy.actor_ref_dropout_p="
        f"{0.0 if args.actor_residual_to_ref else args.actor_ref_dropout_p}",
        f"--policy.gamma={args.gamma}",
        f"--policy.beta={args.beta}",
        f"--policy.pin_action_dims_to_ref={_parse_dims(args.pin_action_dims_to_ref)}",
        f"--policy.tau={args.tau}",
        f"--policy.utd_ratio={args.utd_ratio}",
        f"--policy.actor_update_interval={args.actor_update_interval}",
        f"--policy.actor_hidden_dim={args.actor_hidden_dim}",
        f"--policy.actor_num_layers={args.actor_num_layers}",
        f"--policy.actor_fixed_std={args.actor_fixed_std}",
        f"--policy.actor_activation={args.actor_activation}",
        f"--policy.actor_residual={'true' if args.actor_residual else 'false'}",
        f"--policy.critic_hidden_dim={args.critic_hidden_dim}",
        f"--policy.critic_num_layers={args.critic_num_layers}",
        f"--policy.critic_activation={args.critic_activation}",
        f"--policy.critic_residual={'true' if args.critic_residual else 'false'}",
        f"--policy.critic_layer_norm={'true' if args.critic_layer_norm else 'false'}",
        f"--policy.target_policy_noise={args.target_policy_noise}",
        f"--policy.target_noise_clip={args.target_noise_clip}",
        "--policy.device=cuda",
        *build_dataset_argv(
            dataset_name=paths.dataset_name,
            dataset_root=paths.dataset_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            fps=args.fps,
            vcodec=args.vcodec,
        ),
        # Policyless (pure teleop, no VLA/RL) reset window between episodes --
        # NOT skipped for online training (unlike plain --only-critical data
        # collection): after a failed/aborted critical-phase attempt the robot
        # can be left in an out-of-distribution pose (e.g. pin half-inserted
        # at a bad angle), and letting the frozen VLA immediately resume
        # autonomous control from there, with no human in the loop yet, is
        # not something it was ever trained to recover from. This window
        # gives you `reset_time_s` seconds of pure leader-arm teleop to
        # physically reset the scene before the next episode's recording (and
        # possible autonomous critical-phase attempt) begins.
        f"--dataset.reset_time_s={args.reset_time_s}",
        # rlt_toggle_key starts the critical-phase attempt; the next press
        # ends it as success immediately, or the u key ends
        # it as failure immediately -- either way hands control back to VLA
        # and flushes that reward into the online replay buffer right there
        # (see loop.py) -- it does NOT end the recorded episode. The episode
        # keeps recording (e.g. VLA autonomously finishing a subsequent step
        # like placing the object) until you press the whole-episode outcome
        # key (episode_success_key/episode_failure_key, s/f by default) once
        # that's done.
        "--rlt.enable=true",
        f"--rlt.rl_phase_key={args.rlt_toggle_key}",
        f"--rlt.milestone_key={args.milestone_key}",
        f"--rlt.intervention_action_blend_time_s={args.intervention_blend_time_s}",
        f"--rlt.skip_prefix_recording={'true' if args.skip_prefix_recording else 'false'}",
        "--rlt.rl_phase_key_toggles_critical_phase=true",
        "--rlt.start_in_teleop=false",
        # v1 online training does not support the RTC runtime.
        "--rlt.rtc_enabled=false",
        "--enable_episode_outcome_labeling=true",
        "--require_episode_success_label=true",
        "--intervention_state_machine_enabled=true",
        f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        f"--play_sounds={'true' if args.play_sounds else 'false'}",
        f"--estop_key={args.estop_key}",
        # Online RL training loop (see backend.OnlineRLConfig).
        "--online_rl.enable=true",
        f"--online_rl.warmup_episodes={args.warmup_episodes}",
        f"--online_rl.critic_only_episodes={args.critic_only_episodes}",
        f"--online_rl.min_warmup_transitions={args.min_warmup_transitions}",
        f"--online_rl.min_warmup_successes={args.min_warmup_successes}",
        f"--online_rl.min_warmup_failures={args.min_warmup_failures}",
        f"--online_rl.max_updates_per_episode={args.max_updates_per_episode}",
        f"--online_rl.use_stratified_sampling={'true' if args.stratified_sampling else 'false'}",
        f"--online_rl.replay_capacity={args.replay_capacity}",
        *( [f"--online_rl.beta_final={args.beta_final}"] if args.beta_final is not None else [] ),
        f"--online_rl.beta_anneal_episodes={args.beta_anneal_episodes}",
        f"--online_rl.terminal_reward={args.terminal_reward}",
        f"--online_rl.milestone_reward={args.milestone_reward}",
        f"--online_rl.time_decay={args.time_decay}",
        f"--online_rl.batch_size={args.batch_size}",
        f"--online_rl.lr_actor={args.lr_actor}",
        f"--online_rl.lr_critic={args.lr_critic}",
        f"--online_rl.save_dir={args.save_dir}",
        f"--online_rl.save_every_episodes={args.save_every_episodes}",
        f"--online_rl.go_home_time_s={args.go_home_time_s}",
        f"--online_rl.go_home_gripper_value={args.go_home_gripper_value}",
        f"--online_rl.wandb={'true' if args.wandb else 'false'}",
        f"--online_rl.wandb_project={args.wandb_project}",
    ]
    if args.actor_action_clip_delta is not None:
        argv.append(f"--policy.actor_action_clip_delta={args.actor_action_clip_delta}")
    if args.go_home_positions is not None:
        argv.append(f"--online_rl.go_home_positions={args.go_home_positions}")
    if args.wandb_entity is not None:
        argv.append(f"--online_rl.wandb_entity={args.wandb_entity}")
    if args.wandb_run_name is not None:
        argv.append(f"--online_rl.wandb_run_name={args.wandb_run_name}")
    if args.wandb_run_id is not None:
        argv.append(f"--online_rl.wandb_run_id={args.wandb_run_id}")
    if args.wandb_resume is not None:
        argv.append(f"--online_rl.wandb_resume={args.wandb_resume}")
    if args.resume_from is not None:
        argv.append(f"--online_rl.resume_from={args.resume_from}")
    return argv

def print_online_train_summary(args: argparse.Namespace, paths) -> None:
    print("\nOnline RL training (synchronous, real hardware)")
    print(f"Dataset (raw episodes, for record-keeping): {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Checkpoints: {args.save_dir}")
    print(f"VLA: {args.vla_path}")
    print(f"RL token: {args.rl_token_path}")
    print(
        f"RL: warmup_episodes={args.warmup_episodes} (+min_transitions={args.min_warmup_transitions} "
        f"min_successes={args.min_warmup_successes} min_failures={args.min_warmup_failures}) "
        f"critic_only_episodes={args.critic_only_episodes} batch_size={args.batch_size} "
        f"lr_actor={args.lr_actor} lr_critic={args.lr_critic} utd_ratio={args.utd_ratio} "
        f"max_updates_per_episode={args.max_updates_per_episode} "
        f"stratified_sampling={args.stratified_sampling} save_every={args.save_every_episodes}"
    )
    print(
        f"Actor: hidden_dim={args.actor_hidden_dim} num_layers={args.actor_num_layers} "
        f"residual_to_ref={args.actor_residual_to_ref} "
        f"fixed_std={args.actor_fixed_std} "
        f"(ref_dropout="
        f"{0.0 if args.actor_residual_to_ref else args.actor_ref_dropout_p}) | "
        f"Critic: hidden_dim={args.critic_hidden_dim} "
        f"num_layers={args.critic_num_layers}"
    )
    # gamma/beta/time_decay are the three parameters with evidence behind their
    # values (README / --help), and --gamma's default is the wrong value for
    # 30 fps, so a run has to carry it explicitly. Nothing else surfaces them:
    # the only validation is a range check, and the wandb config does not
    # record them -- so without this line a forgotten `--gamma 0.999` stays
    # invisible until someone reads a saved checkpoint's config.
    print(
        f"Reward/credit: gamma={args.gamma} beta={args.beta}"
        f"{'' if args.beta_final is None else f'->{args.beta_final} over {args.beta_anneal_episodes} episodes'} "
        f"time_decay={args.time_decay} terminal_reward={args.terminal_reward} "
        f"milestone_reward={args.milestone_reward}"
    )
    print(
        f"Deploy (VLA-parity): chunk_exec_steps={args.chunk_exec_steps} "
        f"(bare π0.5 uses n_action_steps={sc.VLA_HORIZON}) "
        f"vla_reuse_horizon={args.vla_reuse_horizon} "
        f"{'(legacy: re-run π0.5 every chunk — preferred)' if args.vla_reuse_horizon <= args.chunk_length else '(WARNING: open-loop VLA reuse trades accuracy for FPS)'} "
        f"vcodec={args.vcodec}"
    )
    print(
        f"Safety: actor_action_clip_delta={args.actor_action_clip_delta} "
        f"estop_key={args.estop_key} (grabs manual control -- not a hardware E-stop; "
        "stay near the leader arm / power cutoff)"
    )
    print(
        f"Controls (critical attempt): press {args.rlt_toggle_key} to START RL, "
        f"press {args.rlt_toggle_key} again to END as SUCCESS "
        f"(or u=END as FAILURE). Then s/f ends the whole recorded episode. "
        f"{args.teleop_toggle_key}/{args.estop_key}=manual takeover"
    )
    print(
        f"Go-home: {args.go_home_time_s}s follower ramp AFTER s/f, then auto-sync leaders "
        f"to the same park before the reset teleop window — "
        f"{'DISABLED until you pass --go-home-positions (safe)' if args.go_home_positions in (None, '', '{}') else 'using --go-home-positions'}. "
        f"gripper default={args.go_home_gripper_value}. "
        "Record park pose: rlt-so101-dual-record-pose --setup-json configs/hardware/so101_dual_manifest.json"
    )
    print(
        f"Reset window: {args.reset_time_s}s pure teleop after every episode "
        "(success or failure) before the next one starts -- use it to physically "
        "reposition task objects (go-home does not move them)."
    )
    if args.wandb:
        print(
            f"Wandb: project={args.wandb_project} entity={args.wandb_entity} "
            f"run_name={args.wandb_run_name} run_id={args.wandb_run_id} "
            f"resume={args.wandb_resume}"
        )
    if args.resume_from:
        print(f"Resuming online-RL state from: {args.resume_from}")

def run_online_train(args: argparse.Namespace) -> None:
    set_offline_env()
    # Capture PCs (≈30GiB RAM) OOM-kill π0.5 if Hub/dynamo also wake up mid-load.
    import os

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.wandb_resume is not None and args.wandb_run_id is None:
        raise ValueError("--wandb-resume requires --wandb-run-id (the original W&B run ID).")
    if args.vla_reuse_horizon % args.chunk_length != 0:
        raise ValueError(
            f"--vla-reuse-horizon={args.vla_reuse_horizon} must be a multiple of "
            f"--chunk-length={args.chunk_length}"
        )
    if args.vla_reuse_horizon < args.chunk_length:
        raise ValueError(
            f"--vla-reuse-horizon={args.vla_reuse_horizon} must be >= "
            f"--chunk-length={args.chunk_length}"
        )
    if args.vla_reuse_horizon > args.chunk_length:
        print(
            f"WARNING: --vla-reuse-horizon={args.vla_reuse_horizon} > chunk_length="
            f"{args.chunk_length} trades VLA accuracy for FPS. Prefer the default "
            f"{args.chunk_length} unless you explicitly want open-loop reuse.",
            flush=True,
        )
    if args.actor_residual_to_ref and args.actor_ref_dropout_p > 0.0:
        print(
            f"WARNING: --actor-ref-dropout-p={args.actor_ref_dropout_p} is ignored "
            "with --actor-residual-to-ref (forced to 0 in policy argv).",
            flush=True,
        )
    # Paper-first defaults: absolute actor + fixed_std exploration + sparse reward.
    # SO101 safety overrides below are intentional A/B knobs — warn when used.
    if args.actor_residual_to_ref:
        print(
            "WARNING: --actor-residual-to-ref departs from the paper (mu = f(x, ref)). "
            "Use only as an SO101 safety A/B; default is --no-actor-residual-to-ref.",
            flush=True,
        )
    if args.actor_fixed_std <= 0.0:
        print(
            "WARNING: --actor-fixed-std=0 disables the paper's Gaussian exploration "
            "(sample every RL chunk). Prefer the default 0.05 for paper-aligned runs.",
            flush=True,
        )
    if args.actor_action_clip_delta is not None:
        print(
            f"WARNING: --actor-action-clip-delta={args.actor_action_clip_delta} is an "
            "SO101 safety clamp not in the paper; omit it for paper-aligned runs.",
            flush=True,
        )
    if args.stratified_sampling:
        print(
            "WARNING: --stratified-sampling is an engineering addition; the paper path "
            "is uniform replay plus intervention→BC-anchor replacement. "
            "Default is --no-stratified-sampling.",
            flush=True,
        )
    if args.milestone_reward > 0.0 or args.time_decay != 1.0:
        print(
            f"WARNING: milestone_reward={args.milestone_reward} time_decay={args.time_decay} "
            "add reward shaping beyond the paper's sparse terminal reward "
            "(defaults: milestone=0, time_decay=1).",
            flush=True,
        )
    if args.critic_layer_norm:
        print(
            "WARNING: --critic-layer-norm is from RLPD, not the RLT paper. "
            "Default is --no-critic-layer-norm.",
            flush=True,
        )
    # Every key bound during an online run, configurable or fixed. A collision
    # is silent at runtime -- two bindings map to one physical key and one
    # of them simply never fires -- so it has to be caught here.
    bound = {
        "--teleop-toggle-key": args.teleop_toggle_key,
        "--estop-key": args.estop_key,
        "--rlt-toggle-key": args.rlt_toggle_key,
        "--milestone-key": args.milestone_key,
        "fixed failure key": "u",
        "fixed episode-success key": "s",
        "fixed episode-failure key": "f",
    }
    seen: dict[str, str] = {}
    for label, key in bound.items():
        normalized = " " if key == "space" else key
        if normalized in seen:
            raise ValueError(f"{label} conflicts with {seen[normalized]} on key {key!r}.")
        seen[normalized] = label

    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, DEFAULT_DATASET_NAME_PREFIX)
    if args.save_dir is None:
        if args.resume_from is not None:
            raise ValueError(
                "--resume-from requires --save-dir pointing at the same directory the "
                "resumed run used (so new checkpoints land alongside its history) -- pass "
                "it explicitly instead of relying on the auto-generated default."
            )
        # Mirrors the dataset run folder's own <MMDD>_<tag>/<prefix>_<HHMMSS> timestamp so
        # a session's checkpoints and its raw dataset are easy to correlate, and so
        # back-to-back fresh runs never collide/overwrite each other's checkpoints.
        args.save_dir = str(Path("outputs/online_rl") / paths.day_dir.name / paths.dataset_root.name)
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)
    teleop_argv = build_teleop_argv(setup.leaders, no_teleop=False)
    if not teleop_argv:
        raise ValueError(
            "Online training requires leader teleop arms (mid-chunk human intervention "
            "is part of the training loop)."
        )

    leader_cal_dir = None
    with TemporaryDirectory(prefix="online-train-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        if args.preflight:
            preflight_motor_connections(
                setup.followers, setup.leaders, cal_dir,
                leader_cal_dir.name if leader_cal_dir is not None else None,
            )
        sys.argv = build_online_train_argv(args, setup, paths, cal_dir, teleop_argv)
        print_online_train_summary(args, paths)
        if args.dry_run:
            print("\nDry run argv:")
            print(" ".join(sys.argv))
            return

        prepare_lerobot_runtime(
            intervention_toggle_key=args.teleop_toggle_key,
            # Deliberately NOT True here -- see the comment on
            # --dataset.reset_time_s in build_online_train_argv(). Plain
            # --only-critical data collection skips this because it's
            # replaying a fixed, already-trained checkpoint; online training
            # needs the human to get a guaranteed reset window between every
            # episode regardless of how the previous one ended.
            skip_policyless_reset_loop=False,
            background_episode_video_encoding=True,
        )
        from rlt_so101_dual.adapters.lerobot.record.backend import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronous online RL training on real hardware")
    parser.add_argument("--vla-path", required=True, help="Frozen pi0.5 VLA checkpoint (from demo adaptation).")
    parser.add_argument("--rl-token-path", required=True, help="Frozen RL token checkpoint (from demo adaptation).")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--task",
        required=True,
        help="Task instruction string; must match collect / SFT / RL-token verbatim.",
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument(
        "--reset-time-s", type=int, default=15,
        help="Pure-teleop window between episodes to physically reset the scene "
        "(no VLA/RL action sent during this time). Runs regardless of whether the "
        "previous episode succeeded or failed.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG)
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="Video codec: h264 (CPU, default for online — frees GPU for π0.5), "
        "auto (prefer NVENC), or h264_nvenc.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--intervention-blend-time-s", type=float, default=0.3,
        help="Smooth both transitions across a human intervention -- takeover (space "
        "pressed, blends from the last policy action to teleop) and release (space "
        "released, blends from the last teleop position back to the freshly "
        "recomputed policy action) -- over this many seconds, instead of jumping "
        "instantly. 0 disables both blends.",
    )
    parser.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rlt-toggle-key", default="r")
    parser.add_argument("--teleop-toggle-key", default="space")
    parser.add_argument("--estop-key", default="x")
    parser.add_argument(
        "--preflight", action=argparse.BooleanOptionalAction, default=True,
        help="Connect/torque-check follower+leader arms before loading the policy.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)

    # Policy shape (must match the loaded VLA + RL token checkpoints).
    parser.add_argument("--chunk-length", type=int, default=sc.CHUNK_LENGTH)
    parser.add_argument(
        "--chunk-exec-steps",
        type=int,
        default=sc.VLA_HORIZON,
        help="VLA-phase open-loop steps per π0.5 forward (must be <= VLA horizon). "
        f"Default {sc.VLA_HORIZON} matches bare π0.5 n_action_steps / first Stage-B eval. "
        "Smaller values re-plan more often (was briefly 25 for speed experiments).",
    )
    parser.add_argument(
        "--vla-reuse-horizon",
        type=int,
        default=10,
        help="RL-phase: control steps covered by ONE frozen-VLA forward "
        "(multiple of --chunk-length). Default 10 = accuracy-first (re-run π0.5 "
        "every chunk). Values >10 trade accuracy for FPS — not recommended as default.",
    )
    parser.add_argument("--action-dim", type=int, default=sc.ACTION_DIM)
    parser.add_argument("--proprio-dim", type=int, default=sc.PROPRIO_DIM)

    # Reward shaping.
    parser.add_argument(
        "--terminal-reward", type=float, default=1.0,
        help="Reward for a successful critical-phase attempt (paper sparse terminal).",
    )
    parser.add_argument(
        "--milestone-reward", type=float, default=0.0,
        help="Optional one-off sub-goal bonus (NOT in the paper; default 0). "
             "Released by --milestone-key at most once per attempt. Keep well below "
             "--terminal-reward if enabled. 0.0 disables the bonus and unbinds the key.",
    )
    parser.add_argument(
        "--time-decay", type=float, default=1.0,
        help="Optional speed incentive: rewards *= this ** (chunks closed). "
             "Paper path is sparse terminal only — default 1.0 disables shaping. "
             "Deliberately separate from --gamma. Try 0.98 only as an SO101 A/B.",
    )
    parser.add_argument(
        "--milestone-key", default="m",
        help="Key that marks the sub-goal reached. No-op outside the critical phase.",
    )

    # TD3+BC hyperparameters.
    # No default on purpose. 0.99 is the wrong value at 30 fps (half-life 2.3 s
    # against a 5-20 s critical phase) and the only validation downstream is a
    # 0 < gamma <= 1 range check, which 0.99 passes -- so a forgotten flag used
    # to start a full run that trained on the wrong discount without erroring.
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument(
        "--skip-prefix-recording", action=argparse.BooleanOptionalAction, default=True,
        help="Drop frames before the first critical-phase entry from the dataset. True keeps "
             "datasets small, but an episode that fails during the grasp -- s/f pressed without "
             "ever pressing r -- then has zero frames and is discarded, so grasp failures never "
             "enter the data and every success rate is conditioned on reaching the critical "
             "phase. Pass --no-skip-prefix-recording for evaluation runs where the denominator "
             "has to include them. Only dataset writes are affected; the replay buffer sees "
             "every frame regardless.")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument(
        "--pin-action-dims-to-ref", default="",
        help="Comma-separated 0-based joint indices the actor is pinned to the VLA "
             "reference on (mu == ref, so the joint leaves both the BC term and the "
             "actor's control). Use for a joint the task never moves: wrist_roll "
             "(index 4) has a 0.558 deg q99-q01 in these demos, so QUANTILES "
             "amplifies it ~79x -- it carried 52%% of the BC penalty while moving "
             "0.019 deg, and its exec-ref spread is repositioning noise, not action "
             "variation. Empty by default -- this changes the objective.")
    parser.add_argument(
        "--beta-final", type=float, default=None,
        help="Anneal the BC regulariser from --beta down to this over "
             "--beta-anneal-episodes, starting when the critic-only window ends. "
             "Unset (default) keeps beta fixed, matching the paper. A large beta early "
             "keeps the policy near the VLA while Q is still noisy; relaxing it later "
             "lets the actor depart. Try --beta 0.5 --beta-final 0.1 if the actor never "
             "moves away from the reference.",
    )
    parser.add_argument("--beta-anneal-episodes", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument(
        "--target-policy-noise", type=float, default=0.0,
        help="TD3 target policy smoothing: std of the clipped Gaussian noise added to "
             "the target action. 0.0 (default) reproduces the reference implementation, "
             "which omits it; TD3 itself uses 0.2. Training-only -- it never affects "
             "what the robot executes, so it is a safe A/B once you have a baseline.",
    )
    parser.add_argument("--target-noise-clip", type=float, default=0.5,
                        help="Clip bound for --target-policy-noise.")
    # Gradient updates per NEW transition this episode added, capped by
    # --max-updates-per-episode. Default 5 matches the paper / offline UTD.
    parser.add_argument("--utd-ratio", type=int, default=5)
    parser.add_argument("--max-updates-per-episode", type=int, default=200)
    parser.add_argument("--actor-update-interval", type=int, default=2)

    # Network size. 3x512 is the paper's spec for its harder tasks (screw
    # installation); 2x256 is what it used for the simpler ones.
    # (the paper's complex-task tier: 3 layers, hidden_dim 512), appropriate
    # for multi-step, dexterous, contact-rich tasks like bimanual insertion.
    parser.add_argument("--actor-hidden-dim", type=int, default=512)
    parser.add_argument("--actor-num-layers", type=int, default=3)
    parser.add_argument("--actor-activation", default="relu")
    parser.add_argument(
        "--actor-residual", action=argparse.BooleanOptionalAction, default=False,
        help="Use ResidualMLP for the actor's internal trunk (skip connections). "
             "Paper path uses a plain MLP (default off). Not the same as "
             "--actor-residual-to-ref.",
    )
    parser.add_argument(
        "--actor-residual-to-ref", action=argparse.BooleanOptionalAction, default=False,
        help="If set: mu = ref + delta with zero-init last layer (SO101 safety A/B). "
             "Paper default OFF: mu = f(x, ref) as an absolute policy. "
             "Pass --actor-residual-to-ref only when you want the untrained actor ≈ VLA.",
    )
    parser.add_argument(
        "--actor-ref-dropout-p", type=float, default=0.5,
        help="Probability of zeroing the reference chunk fed to the actor during "
             "training (paper-style). Forced to 0 when --actor-residual-to-ref is on.",
    )
    # Paper: Gaussian policy with a small fixed std; compute_chunk() samples
    # whenever fixed_std > 0. On SO101 this can look twitchy — pass
    # --actor-fixed-std 0 for a deterministic SO101 A/B.
    parser.add_argument("--actor-fixed-std", type=float, default=0.05)
    parser.add_argument("--critic-hidden-dim", type=int, default=512)
    parser.add_argument("--critic-num-layers", type=int, default=3)
    parser.add_argument("--critic-activation", default="relu")
    parser.add_argument("--critic-residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--critic-layer-norm", action=argparse.BooleanOptionalAction, default=False,
        help="LayerNorm in the critic. Not from the RLT paper (comes from RLPD). "
             "Default off for paper alignment; pass --critic-layer-norm for high-UTD A/B.",
    )

    # Optional SO101 safety clamp (not in the paper). Omit / leave unset for
    # paper-aligned runs. Pass e.g. --actor-action-clip-delta 0.05 on hardware.
    parser.add_argument(
        "--actor-action-clip-delta", type=float, default=None,
        help="Optional clamp on |mu - VLA ref| in normalised space during RL phase. "
        "Unset (default) = no clamp, matching the paper. There is no hardware E-stop.",
    )

    # Online RL loop.
    parser.add_argument("--warmup-episodes", type=int, default=5)
    parser.add_argument(
        "--critic-only-episodes", type=int, default=10,
        help="Episodes after warmup where only the critic updates (actor frozen at its "
        "zero-init-residual, VLA-equivalent behavior) so the critic isn't acting on a "
        "random value estimate when the actor starts moving.",
    )
    parser.add_argument(
        "--min-warmup-transitions", type=int, default=512,
        help="Warmup also requires at least this many transitions in the buffer, not "
        "just --warmup-episodes worth of episodes. One transition per closed chunk, so "
        "approximate_seconds = transitions * chunk_length / fps -- 512 * 10/30 is about "
        "171 s of critical-phase data. That is approximate: the terminal transition of "
        "each attempt may be a partial chunk, so the exact figure is "
        "sum(actual_steps) / fps. Sized against --batch-size (256) rather than wall "
        "clock: below roughly 2x the batch, every batch is mostly resampling the same "
        "transitions and the critic overfits them.",
    )
    parser.add_argument(
        "--min-warmup-successes", type=int, default=3,
        help="Warmup also requires at least this many successful episodes in the buffer.",
    )
    parser.add_argument(
        "--min-warmup-failures", type=int, default=3,
        help="Warmup also requires at least this many failed episodes in the buffer "
        "(a critic that has only ever seen success, or only failure, can't discriminate).",
    )
    parser.add_argument(
        "--stratified-sampling", action=argparse.BooleanOptionalAction, default=False,
        help="If set: stratified success/failure/intervention/recent batches (engineering). "
        "Paper path is uniform sampling + intervention→BC-anchor replacement (default off).",
    )
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    # Actor lr well below critic lr: the critic should adapt quickly, the
    # actor -- which directly drives the robot -- should not.
    parser.add_argument("--lr-actor", type=float, default=3e-5)
    parser.add_argument("--lr-critic", type=float, default=1e-4)
    parser.add_argument(
        "--save-dir", default=None,
        help="Where to write step_NNNNNN checkpoints, selectable "
        "step_NNNNNN/online_state.pt training snapshots, and latest_online_state.pt. If omitted, "
        "auto-generated under outputs/online_rl/<MMDD>_<dataset-tag>/<HHMMSS>/, timestamped "
        "the same way as the raw dataset folder so a fresh session never collides with a "
        "previous one. Required (not auto-generated) when --resume-from is set, so new "
        "checkpoints land alongside the resumed run's history instead of a disconnected "
        "new folder -- pass the same --save-dir the original run used.",
    )
    parser.add_argument("--save-every-episodes", type=int, default=5)
    parser.add_argument(
        "--go-home-time-s", type=float, default=3.0,
        help="After each episode ends (s/f), ramp the follower back to the calibrated "
        "middle position (all non-gripper joints = 0 degrees) over this many seconds, "
        "before the teleop reset window. 0 disables this step.",
    )
    parser.add_argument(
        "--go-home-gripper-value", type=float, default=100.0,
        help="Gripper target (0-100) during go-home. VERIFY which end means 'open' for "
        "your specific hardware (mounting-dependent) before relying on this.",
    )
    parser.add_argument(
        "--go-home-positions",
        default="{}",
        help="Per-joint go-home targets as raw motor ticks, as a JSON object -- paste the "
        "POS column straight from lerobot-calibrate's \"recording positions\" screen, no "
        "manual conversion needed. Defaults to '{}', i.e. the calibrated midpoint (0 "
        "degrees) for every joint, which is safe but rarely where you want the arm to "
        "park. Record your own pose after calibrating this robot and pass it here. "
        "Joint names must match the robot's bare SO101 names (shoulder_pan.pos, ...); "
        "names that do not match any joint are rejected rather than ignored. Gripper "
        "joints listed here override --go-home-gripper-value.",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Explicit path to a complete online_state.pt snapshot from a previous run "
        "(e.g. outputs/pin_insert_online_rl/step_000100/online_state.pt). "
        "latest_online_state.pt is also accepted for crash recovery, but is not required. "
        "Restores actor/critic/"
        "target_critic weights, optimizer momentum, the full replay buffer, and the "
        "warmup/critic-only anchor, then resumes the episode counter from there -- "
        "--num-episodes is a total target inclusive of the resumed count, not "
        "'N more episodes'. Recording still starts a fresh video dataset; only the "
        "online-RL training state is carried over. Omit to start a fresh session "
        "(default).",
    )
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False,
        help="Log actor/critic loss and replay-buffer/warmup progress to Weights & Biases, "
        "one point per recorded episode. Requires `pip install rlt-so101-dual[wandb]` and "
        "`wandb login` beforehand. No model weights/checkpoints are uploaded.",
    )
    parser.add_argument("--wandb-project", default="rlt-so101-dual")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-run-id", default=None,
        help="Stable W&B run ID (not the display name). Reuse the original ID to append "
        "metrics to the same run.",
    )
    parser.add_argument(
        "--wandb-resume", choices=["allow", "must", "never", "auto"], default=None,
        help="W&B resume policy. For an intentional continuation, pass --wandb-run-id "
        "<original-id> --wandb-resume must.",
    )
    return parser

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_online_train(args)

if __name__ == "__main__":
    main()
