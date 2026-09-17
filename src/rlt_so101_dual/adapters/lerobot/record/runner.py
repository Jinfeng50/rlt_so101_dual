from __future__ import annotations

import argparse
import logging
import runpy
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rlt_so101_dual.adapters.lerobot import register
from rlt_so101_dual.adapters.lerobot.record.common import (
    build_dataset_argv,
    build_policy_overrides,
    build_robot_argv,
    build_rtc_argv,
    build_teleop_argv,
    configure_logging,
    load_robot_setup,
    preflight_motor_connections,
    remove_existing_dataset,
    resolve_camera_rename_map,
    resolve_run_paths,
    set_offline_env,
    stage_follower_calibrations,
    stage_leader_calibrations,
)

log = logging.getLogger(__name__)


def prepare_lerobot_runtime(
    *,
    episode_outcome_key: str | None = None,
    episode_outcome_failure_key: str | None = "u",
    intervention_toggle_key: str | None = None,
    skip_policyless_reset_loop: bool = False,
    background_episode_video_encoding: bool = False,
) -> None:
    from rlt_so101_dual.adapters.lerobot.record.streaming_temps import patch_streaming_temp_dir_names

    _patch_record_keyboard_listener()
    patch_streaming_temp_dir_names()
    if episode_outcome_key is not None:
        _patch_episode_outcome_listener(
            episode_outcome_key,
            failure_key=episode_outcome_failure_key,
        )
    register()
    if intervention_toggle_key is not None:
        _patch_record_intervention_toggle_key(intervention_toggle_key)
    if skip_policyless_reset_loop:
        _patch_skip_policyless_reset_loop()
    if background_episode_video_encoding:
        _patch_append_safe_episode_metadata_flush()
        _patch_save_episode_extra_metadata()


class _CompositeListener:
    def __init__(self, *listeners):
        self._listeners = [listener for listener in listeners if listener is not None]

    def stop(self) -> None:
        for listener in self._listeners:
            stop = getattr(listener, "stop", None)
            if callable(stop):
                stop()


def _event_key_name(key: str | None) -> str | None:
    if key is None:
        return None
    normalized = key.lower()
    return "space" if normalized == " " else normalized


def _keyboard_key_arg(key: str) -> str:
    normalized = _event_key_name(key)
    if normalized is None:
        raise ValueError("key must not be None")
    return " " if normalized == "space" else normalized


def _pynput_key_name(key: Any, keyboard_module: Any) -> str | None:
    if key == keyboard_module.Key.space:
        return "space"
    if key == keyboard_module.Key.enter:
        return "enter"
    if hasattr(key, "char") and key.char:
        return key.char.lower()
    return None


class _EpisodeOutcomeRouter:
    """Route the success and failure keys to an immediate episode outcome."""

    def __init__(
        self,
        events: dict[str, Any],
        outcome_key: str,
        failure_key: str | None = None,
    ) -> None:
        self._events = events
        self._outcome_key = _event_key_name(outcome_key)
        self._failure_key = _event_key_name(failure_key)

    def on_press(self, key_name: str) -> None:
        normalized_key = _event_key_name(key_name)
        if self._failure_key is not None and normalized_key == self._failure_key:
            self._mark_failure()
            return
        if normalized_key != self._outcome_key:
            return
        self._mark_success()

    def _mark_success(self) -> None:
        self._events["episode_outcome"] = "success"
        self._events["exit_early"] = True
        logging.info("Episode outcome marked as success")

    def _mark_failure(self) -> None:
        self._events["episode_outcome"] = "failure"
        self._events["exit_early"] = True
        logging.info("Episode outcome marked as failure")


PEDAL_TOGGLE_COOLDOWN_S = 0.5


def _start_record_event_pedal_listener(
    events: dict[str, Any],
    key_bindings: dict[str | None, str | None],
    outcome_router: _EpisodeOutcomeRouter | None = None,
):
    from rlt_so101_dual.adapters.lerobot.record.pedal_listener import PedalListener

    event_by_key: dict[str, str] = {}
    for key, event_name in key_bindings.items():
        key_name = _event_key_name(key)
        if key_name is not None and event_name is not None and key_name not in event_by_key:
            event_by_key[key_name] = event_name

    if outcome_router is None and not event_by_key:
        logging.info("No pedal key bindings configured; pedal listener skipped")
        return None

    cooldown_events = {"toggle_intervention", "toggle_critical_phase"}
    last_event_times: dict[str, float] = {}

    def on_press(key_name: str) -> None:
        normalized_key = _event_key_name(key_name)
        if normalized_key is None:
            return
        if outcome_router is not None:
            outcome_router.on_press(normalized_key)
        event_name = event_by_key.get(normalized_key)
        if event_name is None:
            return
        if event_name in cooldown_events:
            now = time.monotonic()
            if now - last_event_times.get(event_name, 0.0) < PEDAL_TOGGLE_COOLDOWN_S:
                return
            last_event_times[event_name] = now
        logging.info("Pedal '%s' pressed -> events[%s] = True", normalized_key, event_name)
        events[event_name] = True

    listener = PedalListener(on_press=on_press)
    if listener.start():
        return listener
    return None


def _start_episode_outcome_key_listener(
    outcome_key: str,
    router: _EpisodeOutcomeRouter,
    failure_key: str | None = None,
):
    import lerobot.utils.control_utils as control_utils

    if control_utils.is_headless():
        return None
    from pynput import keyboard

    normalized_outcome_key = _event_key_name(outcome_key)
    normalized_failure_key = _event_key_name(failure_key)

    def on_press(key: Any) -> None:
        key_name = _pynput_key_name(key, keyboard)
        if key_name == normalized_outcome_key or (
            normalized_failure_key is not None and key_name == normalized_failure_key
        ):
            router.on_press(key_name)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _start_record_event_keyboard_listener(events: dict[str, Any], key_bindings: dict[str | None, str | None]):
    import lerobot.utils.control_utils as control_utils

    if control_utils.is_headless():
        return None
    from pynput import keyboard

    event_by_key = {
        _event_key_name(key): event_name
        for key, event_name in key_bindings.items()
        if _event_key_name(key) is not None and event_name is not None
    }

    def on_press(key: Any) -> None:
        if not _hotkeys_armed(events):
            return
        key_name = _pynput_key_name(key, keyboard)
        event_name = event_by_key.get(key_name)
        if event_name is None:
            return
        if not _debounce_allow(events, event_name):
            return
        events[event_name] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _ensure_record_events(events: dict[str, Any]) -> None:
    events.setdefault("episode_outcome", None)
    events.setdefault("exit_early", False)
    events.setdefault("rerecord_episode", False)
    events.setdefault("stop_recording", False)
    # Wall-clock (perf_counter) until which outcome / stop / intervention
    # hotkeys are ignored — set at each episode start so a leftover key
    # repeat or accidental tap right after "Recording episode N" cannot
    # immediately end or intervene.
    events.setdefault("ignore_hotkeys_until", 0.0)
    events.setdefault("_hotkey_last_fire", {})
    for event_name in [
        "toggle_intervention",
        "toggle_critical_phase",
        "cp_mark_success",
        "cp_mark_failure",
        "start_rl_phase",
        "mark_rl_phase_failure",
        "mark_rl_milestone",
        "end_phase_success",
        "end_phase_failure",
    ]:
        events.setdefault(event_name, False)


# Seconds after episode start before s/f/←/→/space/Esc are accepted.
HOTKEY_ARM_DELAY_S = 2.0
# Per-logical-action debounce (global pynput often doubles).
HOTKEY_DEBOUNCE_S = 0.45


def _hotkeys_armed(events: dict[str, Any]) -> bool:
    return time.perf_counter() >= float(events.get("ignore_hotkeys_until") or 0.0)


def _debounce_allow(events: dict[str, Any], action: str, cooldown_s: float = HOTKEY_DEBOUNCE_S) -> bool:
    last = events.setdefault("_hotkey_last_fire", {})
    now = time.perf_counter()
    if now - float(last.get(action, 0.0)) < cooldown_s:
        return False
    last[action] = now
    return True


def _start_demo_arrow_keyboard_listener(events: dict[str, Any]):
    """Demo / teleop record flow keys (overrides LeRobot default arrows).

    * ``←`` discard / rerecord current episode
    * ``→`` save current episode as success
    * ``↑`` stop the whole recording session
    * ``Esc`` also stops (backup)
    """
    import lerobot.utils.control_utils as control_utils

    if control_utils.is_headless():
        return None
    from pynput import keyboard

    def on_press(key: Any) -> None:
        try:
            if not _hotkeys_armed(events):
                return
            if key == keyboard.Key.left:
                if not _debounce_allow(events, "rerecord"):
                    return
                print("Left arrow: discard / rerecord this episode")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.right:
                if not _debounce_allow(events, "episode_success"):
                    return
                print("Right arrow: save episode as success")
                events["episode_outcome"] = "success"
                events["exit_early"] = True
            elif key == keyboard.Key.up:
                if not _debounce_allow(events, "stop_recording"):
                    return
                print("Up arrow: stop recording session")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                # Esc is easy to brush; require a slightly longer debounce and
                # the arming delay so an accidental tap mid-episode is less likely.
                if not _debounce_allow(events, "stop_recording", cooldown_s=1.0):
                    return
                print("Escape: stop recording session")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as exc:
            logging.warning("Error handling demo arrow key: %s", exc)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _patch_record_keyboard_listener() -> None:
    """Demo arrow keys + RLT character bindings on top of LeRobot record."""
    import lerobot.utils.control_utils as control_utils

    if getattr(control_utils.init_keyboard_listener, "_rlt_so101_dual_record_keys", False):
        return

    def init_keyboard_listener(*args, **kwargs):
        # Built as pairs first so a collision can be reported. Two physical
        # keys sharing one event is intentional (the takeover pair); two events
        # claiming one physical key means one of them is being dropped, and
        # dropping it silently is how s/f stopped labelling episodes under
        # --split-critical-phase.
        _binding_pairs = [
            (kwargs.pop("intervention_toggle_key", None), "toggle_intervention"),
            # Safety hotkey: reuses the same event as intervention_toggle_key
            # so pressing either grabs manual control (dict values need not
            # be unique -- two physical keys, one logical event).
            (kwargs.pop("estop_key", None), "toggle_intervention"),
            (kwargs.pop("critical_phase_toggle_key", None), "toggle_critical_phase"),
            (kwargs.pop("episode_success_key", None), "episode_success"),
            (kwargs.pop("episode_failure_key", None), "episode_failure"),
            (kwargs.pop("cp_success_key", None), "cp_mark_success"),
            (kwargs.pop("cp_failure_key", None), "cp_mark_failure"),
            (kwargs.pop("rl_phase_key", None), "start_rl_phase"),
            (kwargs.pop("rl_phase_failure_key", None), "mark_rl_phase_failure"),
            (kwargs.pop("milestone_key", None), "mark_rl_milestone"),
            (kwargs.pop("end_success_key", None), "end_phase_success"),
            (kwargs.pop("end_failure_key", None), "end_phase_failure"),
        ]
        key_bindings = {}
        for _key, _event in _binding_pairs:
            if _key is None:
                continue
            if _key in key_bindings and key_bindings[_key] != _event:
                logging.warning(
                    "Key %r is claimed by both %r and %r; %r wins and %r is dropped.",
                    _key, key_bindings[_key], _event, _event, key_bindings[_key],
                )
            key_bindings[_key] = _event

        # Do not use upstream arrow/Esc listener: right used to mean "exit
        # unlabeled", which conflicts with "→ = save success".
        events: dict[str, Any] = {}
        _ensure_record_events(events)
        arrow_listener = _start_demo_arrow_keyboard_listener(events)

        # Episode outcomes carry a string value; the other bindings are flags.
        # Keep optional letter keys (s/f, …) for online / segment paths.
        outcome_bindings = {
            key: event.removeprefix("episode_")
            for key, event in key_bindings.items()
            if event in {"episode_success", "episode_failure"}
        }
        regular_bindings = {
            key: event
            for key, event in key_bindings.items()
            if event not in {"episode_success", "episode_failure"}
        }

        record_listener = _start_record_event_keyboard_listener(events, regular_bindings)
        outcome_listener = _start_episode_outcome_keyboard_listener(events, outcome_bindings)
        return _CompositeListener(arrow_listener, record_listener, outcome_listener), events

    init_keyboard_listener._rlt_so101_dual_record_keys = True
    control_utils.init_keyboard_listener = init_keyboard_listener


def _start_episode_outcome_keyboard_listener(events: dict[str, Any], key_bindings: dict[str | None, str]):
    import lerobot.utils.control_utils as control_utils

    if control_utils.is_headless() or not any(key is not None for key in key_bindings):
        return None
    from pynput import keyboard

    outcomes = {_event_key_name(key): value for key, value in key_bindings.items() if key is not None}

    def on_press(key: Any) -> None:
        if not _hotkeys_armed(events):
            return
        outcome = outcomes.get(_pynput_key_name(key, keyboard))
        if outcome is None:
            return
        if not _debounce_allow(events, f"episode_{outcome}"):
            return
        events["episode_outcome"] = outcome
        events["exit_early"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _patch_episode_outcome_listener(
    outcome_key: str,
    failure_key: str | None = "u",
) -> None:
    import lerobot.utils.control_utils as control_utils

    if getattr(control_utils.init_keyboard_listener, "_rlt_so101_dual_episode_outcome", False):
        return

    original_init_keyboard_listener = control_utils.init_keyboard_listener

    def init_keyboard_listener(*args, **kwargs):
        intervention_toggle_key = kwargs.pop("intervention_toggle_key", None)
        critical_phase_toggle_key = kwargs.pop("critical_phase_toggle_key", None)
        kwargs.pop("episode_success_key", None)
        kwargs.pop("episode_failure_key", None)
        cp_success_key = kwargs.pop("cp_success_key", None)
        cp_failure_key = kwargs.pop("cp_failure_key", None)
        rl_phase_key = kwargs.pop("rl_phase_key", None)
        end_success_key = kwargs.pop("end_success_key", None)
        end_failure_key = kwargs.pop("end_failure_key", None)
        keyboard_listener, events = original_init_keyboard_listener()
        _ensure_record_events(events)
        router = _EpisodeOutcomeRouter(events, outcome_key, failure_key=failure_key)
        record_key_bindings = {
            intervention_toggle_key: "toggle_intervention",
            critical_phase_toggle_key: "toggle_critical_phase",
            cp_success_key: "cp_mark_success",
            cp_failure_key: "cp_mark_failure",
            rl_phase_key: "start_rl_phase",
            end_success_key: "end_phase_success",
            end_failure_key: "end_phase_failure",
        }
        pedal_listener = _start_record_event_pedal_listener(events, record_key_bindings, router)
        extra_keyboard_listener = _start_episode_outcome_key_listener(outcome_key, router, failure_key=failure_key)
        record_event_listener = _start_record_event_keyboard_listener(events, record_key_bindings)
        return (
            _CompositeListener(keyboard_listener, pedal_listener, extra_keyboard_listener, record_event_listener),
            events,
        )

    init_keyboard_listener._rlt_so101_dual_episode_outcome = True
    control_utils.init_keyboard_listener = init_keyboard_listener


def _patch_record_intervention_toggle_key(toggle_key: str) -> None:
    from rlt_so101_dual.adapters.lerobot.record import backend

    if getattr(backend.init_keyboard_listener, "_rlt_so101_dual_intervention_key", None):
        return

    original_init_keyboard_listener = backend.init_keyboard_listener
    keyboard_toggle_key = _keyboard_key_arg(toggle_key)

    def init_keyboard_listener(*args, **kwargs):
        kwargs["intervention_toggle_key"] = keyboard_toggle_key
        return original_init_keyboard_listener(*args, **kwargs)

    init_keyboard_listener._rlt_so101_dual_intervention_key = _event_key_name(toggle_key)
    backend.init_keyboard_listener = init_keyboard_listener


def _patch_skip_policyless_reset_loop() -> None:
    from rlt_so101_dual.adapters.lerobot.record import backend

    if getattr(backend.record_loop, "_rlt_so101_dual_skip_policyless_reset_loop", False):
        return

    original_record_loop = backend.record_loop

    def record_loop(*args, **kwargs):
        if kwargs.get("policy") is None and kwargs.get("dataset") is None:
            logging.info("Skipping policyless between-episode reset loop.")
            return None
        return original_record_loop(*args, **kwargs)

    record_loop._rlt_so101_dual_skip_policyless_reset_loop = True
    backend.record_loop = record_loop


def _metadata_buffer_to_pydict(metadata_buffer: list[dict], numpy_module: Any) -> dict:
    combined_dict = {}
    for episode_dict in metadata_buffer:
        for key, value in episode_dict.items():
            if key not in combined_dict:
                combined_dict[key] = []
            val = value[0] if isinstance(value, list) else value
            combined_dict[key].append(val.tolist() if isinstance(val, numpy_module.ndarray) else val)
    return combined_dict


def _patch_append_safe_episode_metadata_flush() -> None:
    import lerobot.datasets.dataset_metadata as dataset_metadata

    metadata_cls = dataset_metadata.LeRobotDatasetMetadata
    if getattr(metadata_cls._flush_metadata_buffer, "_rlt_so101_dual_append_safe", False):
        return

    original_flush_metadata_buffer = metadata_cls._flush_metadata_buffer

    def _flush_metadata_buffer(self) -> None:
        metadata_buffer = getattr(self, "_metadata_buffer", None)
        if not metadata_buffer:
            return
        if getattr(self, "_pq_writer", None) is not None:
            return original_flush_metadata_buffer(self)

        first_ep = metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]
        path = self.root / dataset_metadata.DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        if not Path(path).exists():
            return original_flush_metadata_buffer(self)

        incoming_dict = _metadata_buffer_to_pydict(metadata_buffer, dataset_metadata.np)
        incoming_table = dataset_metadata.pa.Table.from_pydict(incoming_dict)
        existing_df = dataset_metadata.pd.read_parquet(path)
        incoming_df = incoming_table.to_pandas()
        combined_df = dataset_metadata.pd.concat([existing_df, incoming_df], ignore_index=True)
        combined_df.to_parquet(path)
        self.latest_episode = metadata_buffer[-1]
        metadata_buffer.clear()

    _flush_metadata_buffer._rlt_so101_dual_append_safe = True
    metadata_cls._flush_metadata_buffer = _flush_metadata_buffer


def _patch_save_episode_extra_metadata() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset.save_episode, "_rlt_so101_dual_save_episode_patch", False):
        return

    original_save_episode = LeRobotDataset.save_episode

    def save_episode_with_extra_metadata(dataset, args, kwargs, extra_episode_metadata):
        if extra_episode_metadata is None:
            return original_save_episode(dataset, *args, **kwargs)

        original_meta_save_episode = dataset.meta.save_episode

        def meta_save_episode(episode_index, episode_length, episode_tasks, episode_stats, episode_metadata):
            merged_episode_metadata = dict(episode_metadata)
            merged_episode_metadata.update(extra_episode_metadata)
            return original_meta_save_episode(
                episode_index,
                episode_length,
                episode_tasks,
                episode_stats,
                merged_episode_metadata,
            )

        dataset.meta.save_episode = meta_save_episode
        try:
            return original_save_episode(dataset, *args, **kwargs)
        finally:
            dataset.meta.save_episode = original_meta_save_episode

    def save_episode(self, *args, **kwargs):
        extra_episode_metadata = kwargs.pop("extra_episode_metadata", None)
        return save_episode_with_extra_metadata(self, args, kwargs, extra_episode_metadata)

    save_episode._rlt_so101_dual_save_episode_patch = True
    LeRobotDataset.save_episode = save_episode


def _validate_distinct_keys(**keys: str | None) -> None:
    seen: dict[str, str] = {}
    for label, key in keys.items():
        normalized = _event_key_name(key)
        if normalized is None:
            continue
        if normalized in seen:
            raise ValueError(f"{label} conflicts with {seen[normalized]} on key {normalized!r}")
        seen[normalized] = label


def _collect_external_episode_outcome_key(args: argparse.Namespace) -> str | None:
    if args.only_critical:
        return None
    return args.rlt_toggle_key


def _episode_outcome_argv(
    enabled: bool,
    default_episode_success: str | None = None,
    discard_unlabeled_episodes: bool = False,
) -> list[str]:
    if not enabled:
        return []
    argv = [
        "--enable_episode_outcome_labeling=true",
        "--require_episode_success_label=true",
    ]
    if default_episode_success is not None:
        argv.append(f"--default_episode_success={default_episode_success}")
    if discard_unlabeled_episodes:
        argv.append("--discard_unlabeled_episodes=true")
    return argv


def _collect_rlt_phase_argv(args: argparse.Namespace) -> list[str]:
    common = [
        "--rlt.enable=true",
        f"--rlt.rl_phase_key={_keyboard_key_arg(args.rlt_toggle_key)}",
    ]
    if args.only_critical:
        return [
            *common,
            "--rlt.skip_prefix_recording=true",
            "--rlt.rl_phase_key_toggles_episode=true",
            f"--rlt.start_in_teleop={'true' if args.start_with_teleop else 'false'}",
        ]
    return [
        *common,
        f"--rlt.start_in_teleop={'true' if args.start_with_teleop else 'false'}",
    ]


def run_collect(args: argparse.Namespace) -> None:
    set_offline_env()
    episode_outcome_key = _collect_external_episode_outcome_key(args)
    validation_keys = {"teleop_toggle_key": args.teleop_toggle_key, "failure_key": "u"}
    if args.only_critical:
        validation_keys["rlt_toggle_key"] = args.rlt_toggle_key
    else:
        validation_keys["episode_outcome_key"] = episode_outcome_key
    _validate_distinct_keys(**validation_keys)
    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, "eval_so101_dual_collect")
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)
    teleop_argv = build_teleop_argv(setup.leaders, args.no_teleop)

    if args.policy_path is None:
        raise ValueError("default collection requires --policy-path")
    if not teleop_argv:
        raise ValueError("default VLA-RLT-VLA collection requires leader teleop arms")

    leader_cal_dir = None
    with TemporaryDirectory(prefix="record-collect-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        sys.argv = build_default_collect_record_argv(args, setup, paths, cal_dir, teleop_argv)
        print_collect_summary(args, paths)
        if args.dry_run:
            print("\nDry run argv:")
            print(" ".join(sys.argv))
            return

        prepare_lerobot_runtime(
            episode_outcome_key=episode_outcome_key,
            intervention_toggle_key=args.teleop_toggle_key,
            skip_policyless_reset_loop=args.only_critical and not args.start_with_teleop,
            background_episode_video_encoding=True,
        )
        from rlt_so101_dual.adapters.lerobot.record.backend import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def build_default_collect_record_argv(
    args: argparse.Namespace,
    setup,
    paths,
    cal_dir: str,
    teleop_argv: list[str],
) -> list[str]:
    argv = [
        "record_collect",
        *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
        *teleop_argv,
        *build_policy_overrides(
            policy_path=args.policy_path,
            vla_path=args.vla_path,
            rl_token_path=args.rl_token_path,
            phase_mode="manual",
        ),
        *build_dataset_argv(
            dataset_name=paths.dataset_name,
            dataset_root=paths.dataset_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            fps=args.fps,
            vcodec=args.vcodec,
        ),
        *_collect_rlt_phase_argv(args),
        *build_rtc_argv(
            enabled=args.rtc,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            vla_execution_horizon=args.vla_rtc_execution_horizon,
            action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
        ),
        *_episode_outcome_argv(
            args.only_critical or _collect_external_episode_outcome_key(args) is not None,
            getattr(args, "default_episode_success", None),
            getattr(args, "discard_unlabeled_episodes", False),
        ),
        "--intervention_state_machine_enabled=true",
        f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        f"--play_sounds={'true' if args.play_sounds else 'false'}",
    ]
    return argv


def print_collect_summary(args: argparse.Namespace, paths) -> None:
    vla_horizon = args.vla_rtc_execution_horizon or args.rtc_execution_horizon
    print("\nDefault VLA-RLT-VLA collection")
    print(f"Dataset: {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Log: {paths.log_file}")
    print(f"Policy: {args.policy_path}")
    print(f"VLA: {args.vla_path}")
    print(f"RL token: {args.rl_token_path}")
    print(
        "RTC: "
        f"enabled={args.rtc} rlt_horizon={args.rtc_execution_horizon} "
        f"vla_horizon={vla_horizon} guidance={args.rtc_max_guidance_weight} "
        f"schedule={args.rtc_prefix_attention_schedule} "
        f"refill_threshold={args.rtc_action_queue_size_to_get_new_actions}"
    )
    if args.only_critical:
        print(
            f"Recording mode: RLT critical segment only. {args.rlt_toggle_key} enters RLT and starts recording; "
            f"the next {args.rlt_toggle_key} saves success immediately; u saves failure immediately."
        )
    else:
        print(
            f"Recording mode: full trajectory. Recording starts immediately; "
            f"{args.rlt_toggle_key} saves success immediately; "
            f"u saves failure immediately."
        )
    print(f"Episode starts with: {'teleop' if args.start_with_teleop else 'VLA'}")
    if args.only_critical:
        rlt_control = f"{args.rlt_toggle_key}=start/end RLT critical recording, u=mark failure"
    else:
        rlt_control = f"{args.rlt_toggle_key}=save full-episode outcome, u=mark failure"
    print(
        "Controls: "
        f"{rlt_control}, "
        f"{args.teleop_toggle_key}=toggle teleop intervention"
    )


def run_segment(args: argparse.Namespace) -> None:
    set_offline_env()
    setup = load_robot_setup(args.setup_json)
    paths = resolve_run_paths(setup.setup, args.dataset_tag, f"eval_{args.critical_source}_segment")
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)

    if args.critical_source in {"rlt", "vla"} and args.policy_path is None:
        raise ValueError("segment recording with critical-source rlt/vla requires --policy-path")
    if args.rtc and not args.vla_ref and args.critical_source == "rlt":
        raise ValueError("RTC RLT recording requires --vla-ref; --no-vla-ref hides the guided reference")

    teleop_argv = build_teleop_argv(setup.leaders, args.no_teleop)
    if args.initial_source == "teleop" and not teleop_argv:
        raise ValueError("--initial-source teleop requires leader teleop arms")

    leader_cal_dir = None
    with TemporaryDirectory(prefix="record-segment-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        if not args.dry_run and args.preflight:
            preflight_motor_connections(
                setup.followers,
                setup.leaders if teleop_argv else [],
                cal_dir,
                leader_cal_dir.name if leader_cal_dir is not None else None,
            )

        sys.argv = build_segment_record_argv(args, setup, paths, cal_dir, teleop_argv)
        print_segment_summary(args, paths)
        if args.dry_run:
            print("\nDry run argv:")
            print(" ".join(sys.argv))
            return

        prepare_lerobot_runtime(background_episode_video_encoding=True)
        from rlt_so101_dual.adapters.lerobot.record.backend import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def build_segment_record_argv(args, setup, paths, cal_dir: str, teleop_argv: list[str]) -> list[str]:
    argv = [
        "record_segment",
        *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
        *teleop_argv,
        *build_segment_policy_argv(args),
        *build_dataset_argv(
            dataset_name=paths.dataset_name,
            dataset_root=paths.dataset_root,
            task=args.task,
            num_episodes=args.num_episodes,
            episode_time_s=args.episode_time_s,
            fps=args.fps,
            vcodec=args.vcodec,
        ),
        *build_reset_time_argv(args),
        "--rlt.enable=true",
        "--rlt.skip_prefix_recording=true",
        "--rlt.rl_phase_key_toggles_episode=true",
        f"--rlt.start_in_teleop={'true' if args.initial_source == 'teleop' else 'false'}",
        f"--rlt.intervention_action_blend_time_s={args.intervention_action_blend_time_s}",
        *build_rtc_argv(
            enabled=args.rtc,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            vla_execution_horizon=args.vla_rtc_execution_horizon,
            action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
        ),
        "--enable_episode_outcome_labeling=true",
        "--intervention_state_machine_enabled=true",
        (
            "--policy_sync_to_teleop="
            f"{'true' if teleop_argv and args.critical_source in {'rlt', 'vla'} else 'false'}"
        ),
        f"--vla_ref={'true' if args.vla_ref else 'false'}",
        "--play_sounds=true",
    ]
    return argv


def build_segment_policy_argv(args: argparse.Namespace) -> list[str]:
    if args.critical_source == "vla":
        return build_policy_overrides(
            policy_path=args.policy_path,
            vla_path=args.vla_path,
            rl_token_path=args.rl_token_path,
            phase_mode="always_vla",
            chunk_exec_steps=args.chunk_exec_steps,
        )
    return build_policy_overrides(
        policy_path=args.policy_path,
        vla_path=args.vla_path,
        rl_token_path=args.rl_token_path,
    )


def build_reset_time_argv(args: argparse.Namespace) -> list[str]:
    if args.reset_time_s is None:
        return []
    return [f"--dataset.reset_time_s={args.reset_time_s}"]


def print_segment_summary(args: argparse.Namespace, paths) -> None:
    vla_horizon = args.vla_rtc_execution_horizon or args.rtc_execution_horizon
    print(f"\nDataset: {paths.dataset_name} -> {paths.dataset_root}")
    print(f"Log: {paths.log_file}")
    print(f"Initial source: {args.initial_source}")
    print(f"Critical source: {args.critical_source}")
    print(
        "RTC: "
        f"enabled={args.rtc} rlt_horizon={args.rtc_execution_horizon} "
        f"vla_horizon={vla_horizon} "
        f"guidance={args.rtc_max_guidance_weight} "
        f"schedule={args.rtc_prefix_attention_schedule} "
        f"refill_threshold={args.rtc_action_queue_size_to_get_new_actions or 'chunk_length-1'}"
    )
    if args.policy_path is not None:
        print(f"Policy: {args.policy_path}")
    print(
        "Segment outcome mode: r starts the recorded critical segment; "
        "r ends it as success immediately; "
        "u marks failure immediately."
    )


def run_full(args: argparse.Namespace) -> None:
    """Record the full trajectory. Default behavior (--split-critical-phase
    off) is unchanged: --episode-outcome-key (r by default) is the single
    whole-episode outcome key, and --phase-mode controls the policy's phase
    for the whole episode with no mid-episode toggle.

    --split-critical-phase switches to a second, independent control scheme
    (matching rlt-so101-dual-online-train's): --rlt-toggle-key (r) toggles ONLY the
    critical sub-phase -- VLA drives the rest of the episode before and
    after it -- and the whole episode's outcome is labeled separately with
    s/f. This is what actually lets --phase-mode manual do anything when
    loading a trained rlt_ac --policy-path for evaluation: without it,
    nothing ever calls the policy's PhaseController.trigger_critical(), so a
    loaded checkpoint's actor is never invoked even though phase_mode=manual
    was requested (see backend.py's rlt_active/rl_phase_key_binding gating).
    online_rl stays disabled either way (default) -- pure inference, no
    reward flush, no replay buffer, no gradient updates.
    """
    set_offline_env()
    split = args.split_critical_phase
    if split:
        if args.phase_mode not in (None, "manual"):
            raise ValueError("--split-critical-phase requires --phase-mode manual (or omit --phase-mode).")
        _validate_distinct_keys(
            rlt_toggle_key=args.rlt_toggle_key,
            teleop_toggle_key=args.teleop_toggle_key,
            estop_key=args.estop_key,
            failure_key="u",
        )
        if args.reset_time_s is None:
            args.reset_time_s = 15
        # full's own add_rtc_args(full) call (build_parser()) doesn't set
        # these two -- only apply the validated defaults the standalone eval
        # command used to have when the user hasn't overridden them, without
        # changing add_rtc_args' shared defaults for non-split full users.
        if args.vla_rtc_execution_horizon is None:
            args.vla_rtc_execution_horizon = 25
        if args.rtc_action_queue_size_to_get_new_actions is None:
            args.rtc_action_queue_size_to_get_new_actions = 30

    setup = load_robot_setup(args.setup_json)
    # LeRobot reserves dataset names beginning with ``eval_`` for policy
    # rollouts.  A teleoperation-only recording has no policy, so using that
    # prefix makes LeRobot's dataset-name sanity check reject the run.
    # Parent folder already carries date + --dataset-tag; leaf is just a clock.
    # Keep eval_* for policy rollouts (LeRobot rejects non-eval names with a policy).
    dataset_prefix = "session" if args.initial_source == "teleop" else "eval_vla_full"
    paths = resolve_run_paths(setup.setup, args.dataset_tag, dataset_prefix)
    configure_logging(paths.log_file, args.log_level)
    remove_existing_dataset(paths.dataset_root)
    teleop_argv = build_teleop_argv(setup.leaders, args.no_teleop)

    if args.initial_source == "vla" and args.policy_path is None:
        raise ValueError("full recording with --initial-source vla requires --policy-path")
    if args.initial_source == "teleop" and not teleop_argv:
        raise ValueError("full recording with --initial-source teleop requires leader teleop arms")
    if split and args.policy_path is None:
        raise ValueError("--split-critical-phase requires --policy-path (a trained rlt_ac checkpoint).")

    leader_cal_dir = None
    with TemporaryDirectory(prefix="record-full-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        leader_cal_dir = stage_leader_calibrations(setup.leaders, teleop_argv)
        sys.argv = [
            "record_full",
            *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
            *teleop_argv,
            *build_policy_overrides(
                policy_path=args.policy_path,
                vla_path=args.vla_path,
                rl_token_path=args.rl_token_path,
                phase_mode="manual" if split else args.phase_mode,
                chunk_exec_steps=args.chunk_exec_steps,
            ),
            *build_dataset_argv(
                dataset_name=paths.dataset_name,
                dataset_root=paths.dataset_root,
                task=args.task,
                num_episodes=args.num_episodes,
                episode_time_s=args.episode_time_s,
                fps=args.fps,
                vcodec=args.vcodec,
                rename_map=resolve_camera_rename_map(args.policy_path, args.rename_map),
            ),
            *build_reset_time_argv(args),
            *build_rtc_argv(
                enabled=args.rtc,
                execution_horizon=args.rtc_execution_horizon,
                max_guidance_weight=args.rtc_max_guidance_weight,
                prefix_attention_schedule=args.rtc_prefix_attention_schedule,
                vla_execution_horizon=args.vla_rtc_execution_horizon,
                action_queue_size_to_get_new_actions=args.rtc_action_queue_size_to_get_new_actions,
            ),
        ]
        if split:
            sys.argv += [
                # rl_phase_key toggles ONLY the critical sub-phase
                # (rl_phase_key_toggles_episode left false) -- the whole
                # episode keeps recording under VLA before and after it,
                # ended separately by episode_success_key/episode_failure_key
                # (s/f defaults, see RecordConfig). skip_prefix_recording=
                # false (unlike online-train) so the full task video is kept
                # for review, not just the critical clip.
                "--rlt.enable=true",
                f"--rlt.rl_phase_key={_keyboard_key_arg(args.rlt_toggle_key)}",
                "--rlt.rl_phase_key_toggles_critical_phase=true",
                "--rlt.rl_phase_key_toggles_episode=false",
                "--rlt.skip_prefix_recording=false",
                "--rlt.start_in_teleop=false",
                *_episode_outcome_argv(True, args.default_episode_success, args.discard_unlabeled_episodes),
                "--intervention_state_machine_enabled=true",
                f"--policy_sync_to_teleop={'true' if teleop_argv else 'false'}",
                "--play_sounds=true",
                f"--estop_key={args.estop_key}",
            ]
        else:
            sys.argv += [
                *_episode_outcome_argv(
                    True,
                    args.default_episode_success,
                    args.discard_unlabeled_episodes,
                ),
                "--intervention_state_machine_enabled=true",
                f"--policy_sync_to_teleop={'true' if teleop_argv and args.initial_source == 'vla' else 'false'}",
                "--play_sounds=true",
            ]
        if args.dry_run:
            print(" ".join(sys.argv))
            return
        if split:
            prepare_lerobot_runtime(
                intervention_toggle_key=args.teleop_toggle_key,
                skip_policyless_reset_loop=False,
                background_episode_video_encoding=True,
            )
        else:
            prepare_lerobot_runtime(
                episode_outcome_key=args.episode_outcome_key if args.pedal_outcome else None,
                background_episode_video_encoding=True,
            )
        from rlt_so101_dual.adapters.lerobot.record.backend import record

        record()

    if leader_cal_dir is not None:
        leader_cal_dir.cleanup()


def run_live(args: argparse.Namespace) -> None:
    set_offline_env()
    setup = load_robot_setup(args.setup_json)
    eval_script = Path(args.eval_script).expanduser()
    if not eval_script.exists():
        raise FileNotFoundError(f"RTC eval script not found: {eval_script}")

    with TemporaryDirectory(prefix="record-live-cal-") as cal_dir:
        stage_follower_calibrations(setup.followers, cal_dir)
        sys.argv = [
            "eval_with_real_robot",
            *build_policy_overrides(
                policy_path=args.policy_path,
                vla_path=args.vla_path,
                rl_token_path=args.rl_token_path,
                phase_mode=args.phase_mode,
                chunk_exec_steps=args.chunk_exec_steps,
            ),
            "--policy.device=cuda",
            "--device=cuda",
            f"--rtc.enabled={'true' if args.rtc else 'false'}",
            f"--rtc.execution_horizon={args.rtc_execution_horizon}",
            f"--rtc.max_guidance_weight={args.rtc_max_guidance_weight}",
            f"--rtc.prefix_attention_schedule={args.rtc_prefix_attention_schedule}",
            *build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir),
            f"--task={args.task}",
            f"--duration={args.duration}",
            f"--fps={args.fps}",
        ]
        if args.dry_run:
            print(" ".join(sys.argv))
            return
        prepare_lerobot_runtime()
        runpy.run_path(str(eval_script), run_name="__main__")
