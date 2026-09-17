"""CLI: rlt-so101-dual-record."""

from __future__ import annotations

import argparse
import json

from rlt_so101_dual.adapters.lerobot.record.runner import run_collect, run_full, run_live, run_segment

DEFAULT_COLLECT_DATASET_TAG = "so101_dual_collect"

def add_common_record_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-source", choices=["vla", "teleop"], required=True)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--vla-path", default=None)
    parser.add_argument("--rl-token-path", default=None)
    parser.add_argument(
        "--task",
        required=True,
        help="Task instruction string; must match SFT / RL-token / online verbatim.",
    )
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--reset-time-s", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument(
        "--vcodec",
        default="auto",
        help="Video codec: auto (prefer NVENC), h264_nvenc, or h264 (CPU). "
        "auto offloads save/encode to GPU when available.",
    )
    parser.add_argument("--no-teleop", action="store_true", default=False)
    parser.add_argument("--default-episode-success", choices=["success", "failure"], default=None)
    parser.add_argument(
        "--discard-unlabeled-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Discard and rerecord an episode when it ends without an explicit success/failure key.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--dry-run", action="store_true", default=False)

def add_rtc_args(
    parser: argparse.ArgumentParser,
    *,
    execution_horizon_default: int = 10,
    vla_execution_horizon_default: int | None = None,
    action_queue_default: int | None = None,
) -> None:
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=execution_horizon_default)
    parser.add_argument("--vla-rtc-execution-horizon", type=int, default=vla_execution_horizon_default)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        default="EXP",
        choices=["EXP", "LINEAR", "ONES", "ZEROS"],
    )
    parser.add_argument(
        "--rtc-action-queue-size-to-get-new-actions",
        type=int,
        default=action_queue_default,
    )

def add_default_collect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--vla-path", default=None)
    parser.add_argument("--rl-token-path", default=None)
    parser.add_argument(
        "--task",
        required=True,
        help="Task instruction string; must match SFT / RL-token / online verbatim.",
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--episode-time-s", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--setup-json", default=None)
    parser.add_argument("--dataset-tag", default=DEFAULT_COLLECT_DATASET_TAG)
    parser.add_argument(
        "--vcodec",
        default="auto",
        help="Video codec: auto (prefer NVENC), h264_nvenc, or h264 (CPU).",
    )
    parser.add_argument("--no-teleop", action="store_true", default=False)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rlt-toggle-key", default="r")
    parser.add_argument("--teleop-toggle-key", default="space")
    parser.add_argument("--default-episode-success", choices=["success", "failure"], default=None)
    parser.add_argument(
        "--discard-unlabeled-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--start-with-teleop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start each episode in teleop instead of VLA.",
    )
    parser.add_argument(
        "--only-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record only the RLT critical segment. The first RLT key press starts "
            "recording and enters RLT; the next RLT key press saves the segment. "
            "The default records the full trajectory immediately and uses the RLT key "
            "as the full-episode outcome key."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", default=False)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified real-robot recording entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect",
        help="Run the default VLA-RLT-VLA real-robot data collection script.",
    )
    add_default_collect_args(collect)
    add_rtc_args(
        collect,
        vla_execution_horizon_default=25,
        action_queue_default=30,
    )
    collect.set_defaults(func=run_collect)

    segment = subparsers.add_parser(
        "segment",
        help="Record only the key segment. Success/failure labels apply to the segment.",
    )
    add_common_record_args(segment)
    add_rtc_args(segment)
    segment.add_argument("--critical-source", choices=["rlt", "vla"], required=True)
    segment.add_argument("--vla-ref", action=argparse.BooleanOptionalAction, default=True)
    segment.add_argument("--chunk-exec-steps", type=int, default=50)
    segment.add_argument("--intervention-action-blend-time-s", type=float, default=0.4)
    segment.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    segment.set_defaults(func=run_segment)

    full = subparsers.add_parser(
        "full",
        help="Record the full trajectory. Success/failure labels apply to the trajectory.",
    )
    add_common_record_args(full)
    add_rtc_args(full)
    full.add_argument("--phase-mode", default=None)
    full.add_argument("--chunk-exec-steps", type=int, default=None)
    full.add_argument("--pedal-outcome", action=argparse.BooleanOptionalAction, default=False)
    full.add_argument("--episode-outcome-key", default="r")
    full.add_argument(
        "--split-critical-phase", action=argparse.BooleanOptionalAction, default=False,
        help="Toggle just the critical sub-phase with --rlt-toggle-key (r) instead of using "
        "it as the whole-episode outcome key -- VLA drives the rest of the episode before "
        "and after it, and the human separately labels the whole episode's outcome with "
        "s/f. This is what actually lets --phase-mode manual do anything when loading a "
        "trained rlt_ac --policy-path for evaluation; --episode-outcome-key/--pedal-outcome "
        "are ignored when this is set. Forces --phase-mode manual (default when omitted). "
        "Off by default -- full's existing behavior is unchanged.",
    )
    full.add_argument("--rlt-toggle-key", default="r", help="Only used with --split-critical-phase.")
    full.add_argument(
        "--teleop-toggle-key", default="space",
        help="Only used with --split-critical-phase (grabs manual control as a safety net).",
    )
    full.add_argument("--estop-key", default="x", help="Only used with --split-critical-phase.")
    full.add_argument(
        "--rename-map", type=json.loads, default=None,
        help="Dataset->policy observation key rename map (JSON dict), applied before "
        "make_policy's feature check. Omit it to decide from the checkpoint's own "
        "config.json: a plain pi0.5 VLA expects base_0_rgb/left_wrist_0_rgb and gets "
        "shape_contract.PI05_CAMERA_MAP; an rlt_ac checkpoint expects the raw "
        "left_wrist/right_wrist/right_front aliases and gets no rename. "
        "Pass '{}' to force no rename.",
    )
    full.set_defaults(func=run_full)

    live = subparsers.add_parser("live", help="Run policy live on the robot without saving a dataset.")
    live.add_argument("--policy-path", required=True)
    live.add_argument("--eval-script", required=True)
    live.add_argument("--vla-path", default=None)
    live.add_argument("--rl-token-path", default=None)
    live.add_argument("--phase-mode", default="always_vla")
    live.add_argument("--chunk-exec-steps", type=int, default=50)
    live.add_argument(
        "--task",
        required=True,
        help="Task instruction string; must match SFT / RL-token / online verbatim.",
    )
    live.add_argument("--duration", type=float, default=120.0)
    live.add_argument("--fps", type=float, default=30.0)
    live.add_argument("--setup-json", default=None)
    live.add_argument("--dry-run", action="store_true", default=False)
    add_rtc_args(live)
    live.set_defaults(func=run_live)
    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"segment", "full"} and args.dataset_tag is None:
        args.dataset_tag = f"{args.initial_source}_{args.command}"
    args.func(args)

if __name__ == "__main__":
    main()
