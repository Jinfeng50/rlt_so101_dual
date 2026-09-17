# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core recording loop used by `lerobot_record.py`."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import torch

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from rlt_so101_dual.adapters.lerobot.record.hil import (
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
    _capture_policy_runtime_state,
    set_teleop_manual_control as apply_teleop_manual_control,
    _predict_policy_action_with_acp_inference,
)
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from rlt_so101_dual.adapters.lerobot.record.annotations import (
    COLLECTOR_HUMAN,
    COLLECTOR_POLICY,
    EPISODE_FAILURE,
    EPISODE_SUCCESS,
    PHASE_CRITICAL,
    PHASE_PREFIX,
    SOURCE_HUMAN,
    SOURCE_VLA,
    resolve_collector_policy_id,
    resolve_rlt_collector_policy_id,
)
from rlt_so101_dual.adapters.lerobot.record.frame_delivery import (
    SIDECAR_NAME,
    FrameDeliveryCounter,
    persist as persist_camera_delivery,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import log_rerun_data

T = TypeVar("T")


def _clone_robot_action(action: RobotAction) -> RobotAction:
    cloned: RobotAction = {}
    for key, value in action.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def _blend_robot_actions(
    action_feature_names: list[str],
    start_action: RobotAction,
    target_action: RobotAction,
    alpha: float,
) -> RobotAction:
    clipped_alpha = min(max(alpha, 0.0), 1.0)
    blended: RobotAction = {}
    for name in action_feature_names:
        start_value = start_action.get(name)
        target_value = target_action.get(name)
        if start_value is None:
            blended[name] = target_value
            continue
        if target_value is None:
            blended[name] = start_value
            continue

        start_array = np.asarray(start_value, dtype=np.float32)
        target_array = np.asarray(target_value, dtype=np.float32)
        blended_value = (1.0 - clipped_alpha) * start_array + clipped_alpha * target_array
        if blended_value.shape == ():
            blended[name] = float(blended_value)
        else:
            blended[name] = blended_value.astype(np.float32)
    return blended


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


def episode_frame_count(dataset: LeRobotDataset | None) -> int:
    """Rows written so far in the episode currently being recorded.

    The buffer lives on `dataset.writer`, not on the dataset. Reading
    `dataset.episode_buffer` raises AttributeError -- it did, at the end of the
    first episode of a session, and ended the run. The eight call sites that
    used to spell it out inline were all on the online-RL path (critical-phase
    marks and human takeovers), so plain teleop recording never reached them
    and the breakage stayed invisible.

    Returns 0 rather than raising when there is no dataset or no buffer yet:
    every caller is annotating a frame index, and a wrong mark is worth less
    than a lost episode.
    """
    writer = getattr(dataset, "writer", None) if dataset is not None else None
    buffer = getattr(writer, "episode_buffer", None)
    if not buffer:
        return 0
    try:
        return int(buffer["size"])
    except (KeyError, TypeError, ValueError):
        return 0


def _validate_policy_image_features(
    policy: PreTrainedPolicy,
    dataset_features: dict[str, dict],
    rename_map: dict[str, str] | None = None,
) -> None:
    """Check that the dataset provides at least one image feature the policy expects.

    Raises a clear error if ALL images are missing - the most common cause is
    `--dataset.video=false` which silently drops all image features from the
    dataset due to an upstream lerobot limitation in
    `aggregate_pipeline_dataset_features`.

    A *partial* mismatch (e.g. the dataset has 2 of pi0.5's 3 camera slots) is
    not an error: `PI05Policy._preprocess_images` pads any policy-expected key
    absent from the observation batch with an all -1 image and a zero mask, so
    single/dual-camera rigs are expected to be missing some slots by design
    (see shape_contract.PI05_CAMERA_MAP). Only a *complete* absence of images
    is the actual `--dataset.video=false` symptom this check exists to catch.

    `rename_map` is the same `--dataset.rename_map` applied at inference time
    by the `rename_observations_processor` (e.g. raw `observation.images.right_front`
    -> pi0.5's `observation.images.base_0_rgb`) -- without it, a dataset that
    is actually complete reads as missing every renamed camera.
    """
    policy_image_keys = [
        k for k, ft in policy.config.input_features.items()
        if ft.type.value == "VISUAL"
    ]
    if not policy_image_keys:
        return

    rename_map = rename_map or {}
    ds_image_keys = [
        rename_map.get(k, k) for k, ft in dataset_features.items()
        if ft.get("dtype") in ("image", "video")
    ]
    missing = [k for k in policy_image_keys if k not in ds_image_keys]
    if not missing:
        return
    overlap = [k for k in policy_image_keys if k in ds_image_keys]
    if overlap:
        # Partial coverage, and at least one of the policy's expected keys
        # was actually found -- the legitimate missing-camera-slot case.
        return

    # Reaching here means zero overlap between what the policy expects and
    # what the (renamed) dataset provides -- either --dataset.video=false
    # (ds_image_keys empty) or a rename_map that doesn't actually produce any
    # of the policy's expected names (wrong direction, typo'd target, etc.).
    # `ds_image_keys` alone being non-empty is NOT sufficient to skip this:
    # it only proves the dataset has *some* images, not that renaming lines
    # any of them up with what the policy will actually read.
    hint = (
        "This usually means --dataset.video=false was set, which disables ALL "
        "image features in the dataset (upstream lerobot limitation), or "
        "--dataset.rename_map doesn't map anything to a name the policy expects. "
        "Set --dataset.video=true (the default) to fix the former."
        "\n  Check camera naming - BiSOFollower auto-prepends left_/right_ "
        "to each arm's camera names."
    )
    raise ValueError(
        f"Policy expects image features {missing} but they are not in "
        f"the dataset features (dataset provides: {ds_image_keys}). {hint}"
    )


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    policy_sync_executor: PolicySyncDualArmExecutor | None = None,
    intervention_state_machine_enabled: bool = True,
    collector_policy_id_policy: int = COLLECTOR_POLICY,
    collector_policy_id_human: int = COLLECTOR_HUMAN,
    acp_inference: ACPInferenceConfig | None = None,
    communication_retry_timeout_s: float = 2.0,
    communication_retry_interval_s: float = 0.1,
    rlt_online_collector: Any | None = None,
    critical_phase_tracker: Any | None = None,
    rlt_intervention_tracker: Any | None = None,
    skip_prefix_recording: bool = False,
    rl_phase_key_toggles_episode: bool = False,
    rl_phase_key_toggles_critical_phase: bool = False,
    start_in_teleop: bool = False,
    intervention_action_blend_time_s: float = 0.0,
    rename_map: dict[str, str] | None = None,
):
    if intervention_action_blend_time_s < 0:
        raise ValueError("intervention_action_blend_time_s must be >= 0")
    if acp_inference is None:
        acp_inference = ACPInferenceConfig()

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if dataset is None and policy is not None:
        raise ValueError("Policy-driven recording requires a dataset for feature mapping.")

    # Early check: verify dataset features include all image features the policy expects.
    if policy is not None and dataset is not None:
        _validate_policy_image_features(policy, dataset.features, rename_map=rename_map)

    action_feature_names = dataset.features[ACTION]["names"] if dataset is not None else None
    if action_feature_names is None:
        if hasattr(robot.action_features, "keys"):
            action_feature_names = list(robot.action_features.keys())
        else:
            action_feature_names = list(robot.action_features)
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    has_teleop = isinstance(teleop, (Teleoperator, list))
    # Duck-type RLT phase control: if policy has set_rl_mode, it's an RLT policy
    rlt = policy if policy is not None and hasattr(policy, "set_rl_mode") else None
    has_autonomous_source = policy is not None
    intervention_enabled = intervention_state_machine_enabled and has_autonomous_source and has_teleop
    # start_in_teleop: episode begins in human-teleop mode (no policy actions
    # are sent to the robot) until the user presses r to enter RL. Used by
    # the wo_prefix HIL recorder where VLA should never drive.
    if start_in_teleop and intervention_enabled:
        intervention_state = INTERVENTION_STATE_ACTIVE
    else:
        intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    last_policy_action_for_blend: RobotAction | None = None
    intervention_blend_start_t: float | None = None
    intervention_blend_start_action: RobotAction | None = None
    # Symmetric counterpart of the takeover blend above, but for the release
    # (space let go): without this, the first post-release frame jumps
    # straight to a freshly recomputed policy action with no transition from
    # wherever the leader arm was just released, which reads as a sudden
    # snap. Blends FROM that last teleop position TOWARD the policy's output
    # over the same intervention_action_blend_time_s window.
    # release_blend_pending=True means "armed but not yet ticking" -- the
    # clock only starts on the first post-release frame that actually has a
    # fresh action to blend toward (see _apply_release_blend), not at the
    # release keypress itself.
    release_blend_pending = False
    release_blend_start_t: float | None = None
    release_blend_start_action: RobotAction | None = None
    teleop_fallback_warned = False

    teleop_arm_for_mode_switch: Any | None = None
    if isinstance(teleop, Teleoperator):
        teleop_arm_for_mode_switch = teleop
    elif isinstance(teleop, list):
        teleop_arm_for_mode_switch = teleop_arm

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_arm_for_mode_switch is not None:
            apply_teleop_manual_control(teleop_arm_for_mode_switch, enabled)

    if policy is None:
        # During reset/teleop-only loops keep leader backdrivable for manual dragging.
        set_teleop_manual_control(True)

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    cond_policy_runtime_state: dict[str, Any] | None = None
    uncond_policy_runtime_state: dict[str, Any] | None = None
    if policy is not None and acp_inference.enable and acp_inference.use_cfg:
        cond_policy_runtime_state = _capture_policy_runtime_state(policy)
        uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

    if intervention_enabled:
        if intervention_state == INTERVENTION_STATE_ACTIVE:
            # start_in_teleop mode: leader is backdrivable, follower mirrors leader.
            set_teleop_manual_control(True)
        else:
            # S0: policy drives both arms, teleop arm should accept feedback commands.
            set_teleop_manual_control(False)

    def run_with_connection_retry(action_name: str, fn: Callable[[], T]) -> T:
        timeout_s = max(communication_retry_timeout_s, 0.0)
        interval_s = max(communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0
        first_error: ConnectionError | None = None

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                    # INFO on purpose: a recovered Feetech blip is normal and
                    # must not look like a session failure in the console.
                    logging.info(
                        "%s OK after %d bus retry(ies) in %.2fs — keep recording.",
                        action_name,
                        attempts - 1,
                        elapsed_s,
                    )
                return result
            except ConnectionError as error:
                if first_error is None:
                    first_error = error
                    logging.info(
                        "%s: brief Feetech bus blip; auto-retry up to %.1fs (not a crash).",
                        action_name,
                        timeout_s,
                    )

                if timeout_s <= 0.0:
                    raise

                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise

                sleep_s = interval_s if interval_s > 0.0 else remaining_s
                time.sleep(min(sleep_s, remaining_s))

    def build_action_tensor(values: RobotAction) -> torch.Tensor:
        return torch.tensor(
            [float(np.asarray(values[name]).reshape(-1)[0]) for name in action_feature_names],
            dtype=torch.float32,
        )

    # Open sidecar JSONL for crash-recovery (state/action per frame)
    _recovery_fh = None
    _frame_counter = 0
    if dataset is not None and hasattr(dataset, "root") and dataset.root is not None:
        _recovery_path = dataset.root / "recovery_frames.jsonl"
        _recovery_path.parent.mkdir(parents=True, exist_ok=True)
        _recovery_fh = open(_recovery_path, "a")  # noqa: SIM115

    def _is_image_key(key: str) -> bool:
        return "image" in key or (dataset is not None and key in dataset.features
                                  and dataset.features[key].get("dtype") in ("video", "image"))

    timestamp = 0
    start_episode_t = time.perf_counter()
    prev_phase = PHASE_PREFIX
    rl_phase_started = False
    final_outcome: str | None = None
    _frame_idx = 0
    # empty_cache is costly and stalls the control loop; online on 16GB cards
    # already pressures timing — defrag far less often than every few seconds.
    _cuda_cleanup_interval = 2000

    def get_episode_frame_index() -> int:
        return episode_frame_count(dataset)

    def _start_intervention() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        intervention_state = INTERVENTION_STATE_ACTIVE
        set_teleop_manual_control(True)
        if rlt_intervention_tracker is not None:
            rlt_intervention_tracker.start(get_episode_frame_index())
        if intervention_action_blend_time_s > 0 and last_policy_action_for_blend is not None:
            intervention_blend_start_t = time.perf_counter()
            intervention_blend_start_action = _clone_robot_action(last_policy_action_for_blend)
            logging.info("Intervention action blend started for %.2fs.", intervention_action_blend_time_s)
        else:
            intervention_blend_start_t = None
            intervention_blend_start_action = None
        if rlt is not None:
            rlt.interrupt_chunk()
            log_say("intervene", play_sounds=True)
        logging.info("Intervention enabled (S1): teleop actions now override policy execution.")

    def _reset_policy_after_intervention_release() -> None:
        nonlocal cond_policy_runtime_state, uncond_policy_runtime_state
        if policy is None or preprocessor is None or postprocessor is None:
            return
        # Prefer interrupt_chunk over policy.reset(): reset() clears the
        # last-chunk state/ref cache the online collector needs, and flips
        # phase to VLA before set_rl_mode() puts it back — a needless wipe.
        if hasattr(policy, "interrupt_chunk"):
            policy.interrupt_chunk()
        else:
            policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        if acp_inference.enable and acp_inference.use_cfg:
            cond_policy_runtime_state = _capture_policy_runtime_state(policy)
            uncond_policy_runtime_state = _capture_policy_runtime_state(policy)
        logging.info("Policy cache reset on release: next policy action is recomputed.")

    def _release_intervention() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        nonlocal release_blend_pending, release_blend_start_t, release_blend_start_action
        if rlt_intervention_tracker is not None:
            rlt_intervention_tracker.stop(get_episode_frame_index())
        intervention_state = INTERVENTION_STATE_RELEASE
        intervention_blend_start_t = None
        intervention_blend_start_action = None
        if intervention_action_blend_time_s > 0 and last_teleop_action is not None:
            # Capture the start position now, but don't start the clock yet
            # (see _apply_release_blend): with a remote policy, the first
            # post-release action can take a real network round trip + a
            # full re-inference (interrupt_chunk() just cleared the queue),
            # which can itself exceed the blend window -- starting the timer
            # here would let that wait alone burn through the whole blend
            # before it's ever visually applied.
            release_blend_pending = True
            release_blend_start_action = _clone_robot_action(last_teleop_action)
            release_blend_start_t = None
            logging.info("Release action blend armed for %.2fs.", intervention_action_blend_time_s)
        else:
            release_blend_pending = False
            release_blend_start_t = None
            release_blend_start_action = None
        set_teleop_manual_control(False)
        _reset_policy_after_intervention_release()
        if rlt is not None:
            if rl_phase_started:
                rlt.set_rl_mode()
            else:
                rlt.interrupt_chunk()
            log_say("resume", play_sounds=True)
            logging.info("RLT chunk interrupted on release: next action recomputed.")
        logging.info("Intervention release requested (S2): returning control to policy.")

    def _handle_intervention_toggle() -> None:
        if not events.get("toggle_intervention", False):
            return
        events["toggle_intervention"] = False
        if not intervention_enabled:
            logging.info("Intervention toggle ignored because policy+teleop are not both active.")
            return
        if intervention_state == INTERVENTION_STATE_POLICY:
            _start_intervention()
            return
        _release_intervention()

    def _handle_critical_phase_events() -> None:
        if events.get("toggle_critical_phase", False):
            events["toggle_critical_phase"] = False
            if rlt is not None:
                rlt.trigger_critical_phase()
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.toggle(get_episode_frame_index())
                if critical_phase_tracker.is_active:
                    from lerobot.utils.audio_feedback import say_start
                    say_start()
        if events.get("cp_mark_success", False):
            events["cp_mark_success"] = False
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_success(get_episode_frame_index())
                from lerobot.utils.audio_feedback import say_success
                say_success()
        if events.get("cp_mark_failure", False):
            events["cp_mark_failure"] = False
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_failure(get_episode_frame_index())
                from lerobot.utils.audio_feedback import say_failure
                say_failure()

    def _mark_rl_phase_failure(toggles_episode: bool, toggles_cp: bool) -> None:
        nonlocal final_outcome, rl_phase_started
        if toggles_episode:
            final_outcome = EPISODE_FAILURE
            events["exit_early"] = True
        elif toggles_cp:
            if rlt is not None:
                rlt.set_vla_mode()
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_failure(get_episode_frame_index())
            if rlt_online_collector is not None:
                # Reward the critical phase independently of the full episode.
                rlt_online_collector.flush_episode(False)
            rl_phase_started = False
        log_say("failure", play_sounds=True)
        logging.info("RL phase ended via failure key (failure)")

    def _mark_rl_phase_success(toggles_episode: bool, toggles_cp: bool) -> None:
        nonlocal final_outcome, rl_phase_started
        if toggles_episode:
            final_outcome = EPISODE_SUCCESS
            events["exit_early"] = True
        elif toggles_cp:
            if rlt is not None:
                rlt.set_vla_mode()
            if critical_phase_tracker is not None and dataset is not None:
                critical_phase_tracker.mark_success(get_episode_frame_index())
            if rlt_online_collector is not None:
                # Flush at critical-phase end; the later VLA tail belongs only
                # to the recorded dataset, not this actor-controlled reward.
                rlt_online_collector.flush_episode(True)
            rl_phase_started = False
        log_say("success", play_sounds=True)
        logging.info("RL phase ended via success key (success)")

    def _start_rl_phase_from_key() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        nonlocal rl_phase_started
        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            intervention_state = INTERVENTION_STATE_RELEASE
            intervention_blend_start_t = None
            intervention_blend_start_action = None
            set_teleop_manual_control(False)
        if rlt is not None:
            rlt.set_rl_mode()
        if critical_phase_tracker is not None and dataset is not None:
            critical_phase_tracker.toggle(get_episode_frame_index())
        rl_phase_started = True
        if rlt_online_collector is not None:
            # start_episode() ran before the VLA prefix, which is of arbitrary
            # length. Re-anchor here so the time decay measures the attempt the
            # actor actually controls, not how long the operator let the VLA
            # run first.
            rlt_online_collector.begin_attempt()
        log_say("RL start", play_sounds=True)
        logging.info("RL phase started (r key)")

    def _handle_rl_phase_start_event() -> None:
        if not events.get("start_rl_phase", False):
            return
        events["start_rl_phase"] = False
        r_key_active = rlt is not None or rl_phase_key_toggles_episode or rl_phase_key_toggles_critical_phase
        active_intervention = intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE
        if rl_phase_started and active_intervention:
            logging.info("Ignoring r key: human intervention is active")
            return
        if not r_key_active:
            return
        toggles_episode = rl_phase_key_toggles_episode
        toggles_cp = rl_phase_key_toggles_critical_phase
        if (toggles_episode or toggles_cp) and rl_phase_started:
            _mark_rl_phase_success(toggles_episode, toggles_cp)
            return
        _start_rl_phase_from_key()

    def _handle_rl_milestone_event() -> None:
        if not events.get("mark_rl_milestone", False):
            return
        events["mark_rl_milestone"] = False
        if not rl_phase_started or rlt_online_collector is None:
            return
        bonus = rlt_online_collector.mark_milestone()
        if bonus > 0:
            log_say("milestone", play_sounds=True)
            logging.info("Milestone marked, bonus %.4f", bonus)

    def _handle_rl_phase_failure_event() -> None:
        if not events.get("mark_rl_phase_failure", False):
            return
        events["mark_rl_phase_failure"] = False
        if not rl_phase_started:
            return
        toggles_episode = rl_phase_key_toggles_episode
        toggles_cp = rl_phase_key_toggles_critical_phase
        if not (toggles_episode or toggles_cp):
            return
        _mark_rl_phase_failure(toggles_episode, toggles_cp)

    def _release_active_intervention_after_phase_end() -> None:
        nonlocal intervention_state, intervention_blend_start_t, intervention_blend_start_action
        if not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE):
            return
        if rlt_intervention_tracker is not None:
            rlt_intervention_tracker.stop(get_episode_frame_index())
        intervention_state = INTERVENTION_STATE_RELEASE
        intervention_blend_start_t = None
        intervention_blend_start_action = None
        set_teleop_manual_control(False)

    def _handle_end_phase_event(event_name: str, outcome: str) -> None:
        if not events.get(event_name, False):
            return
        events[event_name] = False
        if rlt is None:
            return
        rlt.set_vla_mode()
        if critical_phase_tracker is not None and dataset is not None:
            marker = critical_phase_tracker.mark_success if outcome == EPISODE_SUCCESS else critical_phase_tracker.mark_failure
            marker(get_episode_frame_index())
        if rlt_online_collector is not None:
            rlt_online_collector.flush_episode(outcome == EPISODE_SUCCESS)
        _release_active_intervention_after_phase_end()
        log_say(outcome, play_sounds=True)
        logging.info("RL phase ended (%s)", outcome)
        # s/f are bound both as phase-end keys and whole-episode outcome keys.
        # Record the outcome here, in the same listener path that ends the
        # frame loop. Relying on the separate outcome listener races with the
        # backend: exit_early can be observed before that listener writes the
        # label, causing the completed episode to abort the entire recording
        # run with "Missing episode_success label" instead of advancing to the
        # next episode.
        events["episode_outcome"] = outcome
        events["exit_early"] = True

    def _select_action_values(
        act_processed_policy: RobotAction | None,
        act_processed_teleop: RobotAction | None,
    ) -> tuple[float, RobotAction]:
        nonlocal teleop_fallback_warned
        if not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE):
            action = act_processed_policy if act_processed_policy is not None else act_processed_teleop
            return 0.0, action
        if act_processed_teleop is not None:
            return 1.0, act_processed_teleop
        if last_teleop_action is not None:
            if not teleop_fallback_warned:
                logging.warning("Intervention is active but no fresh teleop action is available; reusing last teleop action.")
                teleop_fallback_warned = True
            return 1.0, last_teleop_action
        if act_processed_policy is not None:
            if not teleop_fallback_warned:
                logging.warning("Intervention is active but teleop action is unavailable; falling back to policy action.")
                teleop_fallback_warned = True
            return 1.0, act_processed_policy
        if not teleop_fallback_warned:
            logging.warning("Intervention is active but no teleop/policy action is available; sending zero action.")
            teleop_fallback_warned = True
        return 1.0, zero_policy_action

    def _apply_intervention_blend(is_intervention: float, action_values: RobotAction) -> RobotAction:
        nonlocal intervention_blend_start_t, intervention_blend_start_action
        if not is_intervention or intervention_blend_start_t is None or intervention_blend_start_action is None:
            return action_values
        elapsed_s = time.perf_counter() - intervention_blend_start_t
        alpha = min(elapsed_s / intervention_action_blend_time_s, 1.0)
        if alpha < 1.0:
            return _blend_robot_actions(action_feature_names, intervention_blend_start_action, action_values, alpha)
        intervention_blend_start_t = None
        intervention_blend_start_action = None
        return action_values

    def _apply_release_blend(is_intervention: float, action_values: RobotAction) -> RobotAction:
        """Symmetric counterpart of _apply_intervention_blend(): smooths the
        first `intervention_action_blend_time_s` seconds AFTER intervention
        ends, blending from the leader's last commanded position toward the
        freshly recomputed policy action -- see release_blend_start_t/action's
        setup in _release_intervention()."""
        nonlocal release_blend_pending, release_blend_start_t, release_blend_start_action
        if is_intervention or release_blend_start_action is None:
            return action_values
        if release_blend_pending:
            # First post-release frame that actually has a fresh action_values
            # to blend toward -- start the clock now (not at the release
            # keypress), so a slow first remote inference call doesn't burn
            # through the blend window before it's ever visually applied.
            release_blend_start_t = time.perf_counter()
            release_blend_pending = False
        elapsed_s = time.perf_counter() - release_blend_start_t
        alpha = min(elapsed_s / intervention_action_blend_time_s, 1.0)
        if alpha < 1.0:
            return _blend_robot_actions(action_feature_names, release_blend_start_action, action_values, alpha)
        release_blend_start_t = None
        release_blend_start_action = None
        return action_values

    def _write_recovery_row(frame: dict[str, Any]) -> None:
        nonlocal _frame_counter
        if _recovery_fh is None:
            return
        recovery_row = {}
        for key, value in frame.items():
            if _is_image_key(key) or key == "task":
                continue
            if isinstance(value, np.ndarray):
                recovery_row[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool)):
                recovery_row[key] = value
        _recovery_fh.write(json.dumps(recovery_row) + "\n")
        _recovery_fh.flush()
        _frame_counter += 1

    def _collector_policy_code(
        is_intervention: float,
        selected_from_policy: bool,
        rlt_source: float,
    ) -> int:
        if rlt is not None:
            return resolve_rlt_collector_policy_id(
                is_intervention=bool(is_intervention),
                source_type=rlt_source,
            )
        return resolve_collector_policy_id(
            intervention_enabled=intervention_enabled,
            is_intervention=bool(is_intervention),
            selected_from_policy=selected_from_policy,
            policy_id=collector_policy_id_policy,
            human_id=collector_policy_id_human,
        )

    # Per-frame timing instrumentation -> /tmp/frame_timing.csv
    _perf_fh = open("/tmp/frame_timing.csv", "w")  # noqa: SIM115
    _perf_fh.write("frame,total_ms,obs_ms,infer_ms,send_ms,dataset_ms,sleep_ms\n")

    # How often the loop was handed the same camera buffer twice. `read_latest()`
    # is non-blocking and does not copy, so a camera that misses the deadline
    # contributes its previous frame again and nothing downstream notices -- the
    # frame count, the declared fps and the loop rate all stay correct while the
    # pixels go stale. See frame_delivery.py for why this is compared by object
    # identity on the observation rather than by timestamp or by pixels.
    #
    # This replaced a counter that compared `camera.latest_timestamp`, sampled
    # after get_observation() had already returned -- a different moment from
    # the frame it wrote, so it could call a round stale when two different
    # frames were written. Both ran on the same three episodes before it was
    # removed: they agreed exactly on top (21 and 21) and the timestamp one
    # over-reported wrist by 2 of 326.
    _delivery = FrameDeliveryCounter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        _t_infer = 0.0  # only set on inference frames
        _t_send = 0.0
        _t_dataset = 0.0

        if events["exit_early"]:
            events["exit_early"] = False
            break

        _handle_intervention_toggle()
        _handle_critical_phase_events()
        _handle_rl_phase_start_event()
        _handle_rl_milestone_event()
        _handle_rl_phase_failure_event()
        _handle_end_phase_event("end_phase_success", EPISODE_SUCCESS)
        _handle_end_phase_event("end_phase_failure", EPISODE_FAILURE)

        # Get robot observation (retry Feetech bus blips; bare get_observation
        # used to abort the whole session on a single "no status packet").
        _t0 = time.perf_counter()
        obs = run_with_connection_retry("robot.get_observation", robot.get_observation)
        _t_obs = (time.perf_counter() - _t0) * 1000

        # On `obs`, not on obs_processed or on the camera object: `obs` holds the
        # array read_latest() returned, which is what reaches the dataset.
        _delivery.observe(obs, getattr(robot, "cameras", {}))

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from policy and/or teleop
        act_processed_policy: RobotAction | None = None
        act_processed_teleop: RobotAction | None = None
        if (
            policy is not None
            and preprocessor is not None
            and postprocessor is not None
            and not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)
        ):
            _t0 = time.perf_counter()
            policy_action = _predict_policy_action_with_acp_inference(
                observation_frame=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
                acp_inference=acp_inference,
                cond_runtime_state=cond_policy_runtime_state,
                uncond_runtime_state=uncond_policy_runtime_state,
            )
            _t_infer = (time.perf_counter() - _t0) * 1000
            act_processed_policy = make_robot_action(policy_action, dataset.features)

        if isinstance(teleop, Teleoperator):
            act = run_with_connection_retry("teleop.get_action", teleop.get_action)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))

        elif isinstance(teleop, list):
            arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))

        if act_processed_policy is None and act_processed_teleop is None:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            # `continue` skips the loop tail, and the tail is where the frame
            # budget is honoured and `timestamp` advances. Without these two
            # lines the policyless reset window spun near 1 kHz and `timestamp`
            # stayed 0, so `while timestamp < control_time_s` never ended:
            # --reset-time-s was a no-op and the window only closed when the
            # operator happened to press a key (measured 28 s and 37 s against
            # a configured 6 s, and 28k log lines for one reset).
            precise_sleep(max(1 / fps - (time.perf_counter() - start_loop_t), 0.0))
            timestamp = time.perf_counter() - start_episode_t
            continue

        if act_processed_teleop is not None:
            last_teleop_action = act_processed_teleop
            teleop_fallback_warned = False

        policy_action_for_storage = (
            act_processed_policy if act_processed_policy is not None else zero_policy_action
        )

        is_intervention, action_values = _select_action_values(act_processed_policy, act_processed_teleop)
        action_values = _apply_intervention_blend(is_intervention, action_values)
        action_values = _apply_release_blend(is_intervention, action_values)

        # Applies a pipeline to the action, default is IdentityProcessor
        robot_action_to_send = robot_action_processor((action_values, obs))

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        selected_from_policy = act_processed_policy is not None and action_values is act_processed_policy
        if selected_from_policy:
            last_policy_action_for_blend = _clone_robot_action(action_values)
        _t0 = time.perf_counter()
        if policy_sync_executor is not None and selected_from_policy:
            _sent_action = run_with_connection_retry(
                "policy_sync_executor.send_action",
                lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                    robot_action_to_send
                ),
            )
        else:
            _sent_action = run_with_connection_retry(
                "robot.send_action",
                lambda robot_action_to_send=robot_action_to_send: robot.send_action(robot_action_to_send),
            )
        _t_send = (time.perf_counter() - _t0) * 1000

        # Compute RLT metadata for both dataset writing and online collector.
        # Only pop metadata when policy action was actually executed (not during intervention)
        # to keep _meta_queue in sync with _action_queue.
        rlt_meta = None
        if rlt is not None and not is_intervention:
            rlt_meta = rlt.pop_step_metadata()

        if rlt is None and skip_prefix_recording:
            # Pure-teleop mode with r-key-driven episode boundaries: derive
            # the phase gate from rl_phase_started so skip_prefix_recording
            # drops pre-r frames even though no rlt policy is emitting
            # per-step phase metadata.
            rlt_phase = PHASE_CRITICAL if rl_phase_started else PHASE_PREFIX
        else:
            rlt_phase = rlt_meta.phase if rlt_meta is not None else prev_phase
        rlt_source = SOURCE_HUMAN if is_intervention else (rlt_meta.source_type if rlt_meta else SOURCE_VLA)
        rlt_is_critical = float(rlt_phase == PHASE_CRITICAL)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            policy_action_frame = build_dataset_frame(
                dataset.features, policy_action_for_storage, prefix="complementary_info.policy_action"
            )
            frame = {**observation_frame, **action_frame, **policy_action_frame, "task": single_task}

            if "complementary_info.is_intervention" in dataset.features:
                frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
            if "complementary_info.state" in dataset.features:
                frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
            if "complementary_info.collector_policy_id" in dataset.features:
                collector_code = _collector_policy_code(is_intervention, selected_from_policy, rlt_source)
                frame["complementary_info.collector_policy_id"] = np.array([collector_code], dtype=np.int64)
            if "complementary_info.phase" in dataset.features:
                frame["complementary_info.phase"] = np.array([rlt_phase], dtype=np.float32)
            prev_phase = rlt_phase
            skip_frame = skip_prefix_recording and rlt_phase == PHASE_PREFIX
            if not skip_frame:
                _t0 = time.perf_counter()
                dataset.add_frame(frame)
                _t_dataset = (time.perf_counter() - _t0) * 1000

                _write_recovery_row(frame)

        if rlt_online_collector is not None:
            action_tensor = build_action_tensor(action_values)
            # Peek (non-consuming): the most recent real VLA/RL-token encoding,
            # possibly several frames old if human intervention is active (policy
            # inference -- and therefore compute_chunk() -- is skipped while
            # intervention is active, see the `not is_intervention` guard above).
            # RLTOnlineCollector only actually uses this on its own chunk-start
            # frame, so getting the same value across several frames is fine.
            last_chunk_tensors = rlt.get_last_chunk_tensors() if rlt is not None else None
            state_vec_for_collector, ref_chunk_for_collector = (
                last_chunk_tensors if last_chunk_tensors is not None else (None, None)
            )
            rlt_online_collector.on_frame(
                action=action_tensor,
                state_vec=state_vec_for_collector,
                ref_chunk=ref_chunk_for_collector,
                source_type=rlt_source,
                is_critical=rlt_is_critical,
            )

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        if intervention_state == INTERVENTION_STATE_RELEASE:
            intervention_state = INTERVENTION_STATE_POLICY

        # Periodically defragment CUDA allocator and trigger Python GC to prevent
        # progressive inference slowdown from allocator fragmentation + GC pressure.
        # (KV cache alloc/dealloc every n_action_steps fragments the CUDA free list;
        # episode_buffer accumulates ~20 numpy arrays/frame -> 250K+ objects by 10 min.)
        _frame_idx += 1
        if policy is not None and _frame_idx % _cuda_cleanup_interval == 0:
            torch.cuda.empty_cache()

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        _t_total = (time.perf_counter() - start_loop_t) * 1000
        _t_sleep = _t_total - dt_s * 1000
        _perf_fh.write(
            f"{_frame_idx},{_t_total:.1f},{_t_obs:.1f},{_t_infer:.1f},"
            f"{_t_send:.1f},{_t_dataset:.1f},{_t_sleep:.1f}\n"
        )
        if _frame_idx % 100 == 0:
            _perf_fh.flush()

        timestamp = time.perf_counter() - start_episode_t

    # Finalize toggle-mode episode end: stop intervention, switch RLT back to
    # VLA mode, and tag both the critical phase interval and the episode with
    # the resolved success/failure outcome.
    if final_outcome is not None:
        if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
            if rlt_intervention_tracker is not None:
                rlt_intervention_tracker.stop(get_episode_frame_index())
            intervention_state = INTERVENTION_STATE_RELEASE
            set_teleop_manual_control(False)
        if rlt is not None:
            rlt.set_vla_mode()
        if critical_phase_tracker is not None and dataset is not None:
            ep_size = get_episode_frame_index()
            if final_outcome == EPISODE_SUCCESS:
                critical_phase_tracker.mark_success(ep_size)
            else:
                critical_phase_tracker.mark_failure(ep_size)
        events["episode_outcome"] = final_outcome

    # Close timing file
    if _perf_fh is not None:
        _perf_fh.close()
        logging.info("[Timing] Wrote %d frame timings to /tmp/frame_timing.csv", _frame_idx)

    if _delivery.samples and _delivery.cameras:
        elapsed = time.perf_counter() - start_episode_t
        logging.info("[Camera delivery] loop sampled %d times in %.1fs (%.1f Hz)  |  %s",
                     _delivery.samples, elapsed, _delivery.samples / elapsed,
                     _delivery.summary_line())

        # The terminal line survives nowhere but the scrollback, and the run you
        # most want to compare against is always the one you closed. Raw counts
        # go next to the episodes they describe, so a threshold decided later
        # can be applied to sessions recorded before it existed.
        _record = persist_camera_delivery(_delivery, dataset, elapsed)
        if _record["episode_index"] is None:
            logging.info("[Camera delivery] episode index unavailable; row still written")
        if dataset is None or getattr(dataset, "root", None) is None:
            logging.info("[Camera delivery] no dataset root; not persisting %s", SIDECAR_NAME)

        for name in _delivery.cameras:
            repeats = _delivery.repeated_buffers[name]
            if repeats > _delivery.samples * 0.1:
                logging.warning(
                    "[Camera delivery] %s handed the loop a repeated buffer %.0f%% of the "
                    "time. The dataset will store those repeats as distinct frames.",
                    name, 100 * repeats / _delivery.samples)

    # Close sidecar file
    if _recovery_fh is not None:
        logging.info("[Recovery] Wrote %d frames to recovery_frames.jsonl", _frame_counter)
        _recovery_fh.close()
