"""Recording / online-RL backend loop."""

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import torch

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from rlt_so101_dual.adapters.lerobot.record.common import load_dataset_stats_from_pretrained
from rlt_so101_dual.adapters.lerobot.record.hil import (
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
    _capture_policy_runtime_state,  # noqa: F401
    _predict_policy_action_with_acp_inference,  # noqa: F401
    ramp_teleop_to_action,
)
from rlt_so101_dual.adapters.lerobot.record.loop import record_loop
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.import_utils import register_third_party_plugins
from rlt_so101_dual.adapters.lerobot.record.annotations import (
    COLLECTOR_HUMAN,
    COLLECTOR_POLICY,
    RLT_COLLECTOR_POLICY_ID_TO_NAME,
    infer_collector_policy_version,
    normalize_episode_success_label,
    resolve_episode_success_label,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun

@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to >=1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1
    # Video codec for encoding videos. Options: 'h264', 'hevc', 'libsvtav1'.
    # Use 'h264' for faster encoding on systems where AV1 encoding is CPU-heavy.
    vcodec: str = "libsvtav1"
    # Encode videos in real time during capture instead of writing PNGs first.
    # This keeps save_episode() from blocking the foreground recording loop.
    streaming_encoding: bool = True
    # Maximum number of frames to buffer per camera when using streaming encoding.
    encoder_queue_maxsize: int = 30
    # Number of threads per encoder instance. None uses the codec default.
    encoder_threads: int | None = None
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")

@dataclass
class RLTRecordConfig:
    enable: bool = False
    critical_phase_toggle_key: str = "p"
    default_reset_mode: str = "full"
    # RLT deploy settings (used when enable=True)
    vla_model: str = ""
    rl_token_ckpt: str = ""
    ac_ckpt: str = ""
    task_instruction: str = ""
    phase_mode: str = "manual"
    device: str = "cuda"
    chunk_length: int = 10
    chunk_exec_steps: int = 50
    token_pool_size: int = 64
    image_only: bool = False
    deterministic: bool = True
    actor_hidden_dim: int = 256
    actor_num_layers: int = 3
    actor_residual: bool = True
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    # Keyboard keys for RLT HIL mode
    rl_phase_key: str = "r"
    # Marks the active RL phase/episode as a failure immediately, at any
    # point while it is running.
    rl_phase_failure_key: str = "u"
    # Marks the sub-goal reached (e.g. the bolt is now in the gripper). Only
    # meaningful once the critical phase has started; a no-op otherwise.
    milestone_key: str = "m"
    end_success_key: str = "s"
    end_failure_key: str = "f"
    # wo_prefix mode: drop frames captured during PHASE_PREFIX (before RL phase
    # starts). Used when the dataset should only contain the RL-driven segment.
    skip_prefix_recording: bool = False
    # wo_prefix mode: rl_phase_key toggles - first press starts the episode (and
    # RL phase), second press ends the episode (sets exit_early) and marks it
    # as success immediately; pressing rl_phase_failure_key at any point
    # marks it as failure instead.
    rl_phase_key_toggles_episode: bool = False
    # With-prefix mode: rl_phase_key toggles the critical phase only - first
    # press starts RL, second press ends RL and marks the critical phase as
    # success immediately; pressing rl_phase_failure_key at any point marks
    # it as failure instead. Episode
    # keeps going in VLA mode afterwards.
    rl_phase_key_toggles_critical_phase: bool = False
    # wo_prefix mode: start each episode in human-teleop state (leader drives
    # follower, no policy actions sent) until the user presses the rl_phase_key
    # to enter RL. Required for pure RL-only HIL recording where VLA should
    # never drive the robot.
    start_in_teleop: bool = False
    # Blend follower commands from the last policy action to teleop during
    # SPACE handoff, reducing one-frame jumps when the leader is released.
    intervention_action_blend_time_s: float = 0.0
    # RTC deploy settings for rlt_ac. Disabled by default so existing record
    # scripts keep the synchronous chunk queue.
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 10
    vla_rtc_execution_horizon: int | None = None
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "EXP"
    rtc_action_queue_size_to_get_new_actions: int | None = None

@dataclass
class OnlineRLConfig:
    """Synchronous online RL training on real hardware (RLT Algorithm 1).

    Requires rlt.rl_phase_key_toggles_critical_phase=true (not
    rl_phase_key_toggles_episode): the RL-phase key ends the critical-phase
    ATTEMPT, not necessarily the whole recorded episode -- the dataset
    episode may keep recording afterward (e.g. VLA autonomously finishing a
    subsequent step), ended later by the usual episode-outcome keys. The
    online replay buffer is flushed (see RLTOnlineCollector.flush_episode(),
    called from loop.py at the exact moment the critical phase resolves) with
    the CRITICAL PHASE's own success/failure -- not the whole episode's --
    so the reward reflects only what the actor actually controlled. After
    each recorded episode returns, min(new_transitions_this_cycle *
    utd_ratio, max_updates_per_episode) gradient steps run on `policy` in
    place before the next rollout episode -- no checkpoint save/reload
    between episodes.
    """

    enable: bool = False
    # Episodes to collect (VLA-reference rollout via the zero-init residual
    # actor) before any gradient update runs, matching the paper's warmup
    # phase of pre-filling the replay buffer.
    warmup_episodes: int = 5
    # Episodes immediately after warmup during which only the critic updates
    # (actor stays frozen at its zero-init-residual, VLA-equivalent behavior).
    # Lets the critic form a non-random value estimate before the actor starts
    # moving away from the safe VLA-equivalent starting point. Real-hardware
    # sample budgets can't afford the thousands of critic-only steps common in
    # sim (that alone would be hundreds of robot episodes); this is a much
    # smaller, real-hardware-sized version of the same idea.
    critic_only_episodes: int = 10
    replay_capacity: int = 20_000
    # Optional BC-regularisation schedule. None keeps beta fixed, which is
    # what the reference implementation does and what the paper describes.
    # Set a final value to anneal beta linearly from --beta down to it over
    # beta_anneal_episodes, starting when the critic-only window ends.
    beta_final: float | None = None
    beta_anneal_episodes: int = 50
    # --- Reward shaping -------------------------------------------------
    # Awarded on success at the end of the critical-phase attempt (paper sparse).
    terminal_reward: float = 1.0
    # Optional sub-goal bonus (NOT in the paper). 0.0 disables the key.
    milestone_reward: float = 0.0
    # Optional speed incentive: rewards *= time_decay ** chunks_closed.
    # Paper path: 1.0 (no shaping). Separate from gamma on purpose.
    time_decay: float = 1.0
    batch_size: int = 256
    # Actor lr is kept well below critic lr: the critic needs to adapt
    # quickly to new transitions, while the actor -- which directly drives
    # the robot -- should change slowly and conservatively online.
    lr_actor: float = 3e-5
    lr_critic: float = 1e-4
    # Gradient updates per episode = min(new_transitions * cfg.policy.utd_ratio,
    # max_updates_per_episode) -- scaled by how much data the episode actually
    # added, not a fixed count regardless of episode length.
    max_updates_per_episode: int = 200
    # Warmup ends only once ALL of these hold (not just warmup_episodes):
    # episode count alone doesn't say whether the critic has seen both
    # outcomes, and a VLA that always succeeds (or always fails) in its first
    # few episodes gives the critic nothing to discriminate.
    # One transition per closed chunk, so this is roughly
    # transitions * chunk_length / fps seconds of critical-phase data --
    # 512 * 10/30 is about 171 s. Sized against batch_size (256) rather than
    # wall clock: a buffer smaller than about 2x the batch means every batch
    # is mostly resampling the same transitions.
    min_warmup_transitions: int = 512
    min_warmup_successes: int = 3
    min_warmup_failures: int = 3
    # Stratified batches (success/failure/intervention/recent) instead of
    # uniform sampling -- under sparse terminal-only reward, a uniform batch
    # from a mostly-zero-reward buffer can end up with few or no positive
    # examples.
    use_stratified_sampling: bool = False
    # Directory to save periodic online-training checkpoints. Required when enable=true.
    save_dir: str | None = None
    save_every_episodes: int = 5
    # Path to a latest_online_state.pt written by a previous run's
    # save_latest_state() (crash recovery / continuing a session after a
    # stop). Restores actor/critic/target_critic weights, optimizer
    # momentum, the full replay buffer, and the warmup/critic-only anchor,
    # and resumes the episode counter from where it left off. None (default)
    # starts a fresh session as before. This is unrelated to the top-level
    # `resume` field, which is LeRobot's own dataset-append resume.
    resume_from: str | None = None
    # Log actor/critic training curves (loss, buffer growth, warmup progress)
    # to Weights & Biases -- one point per recorded episode. No model
    # weights/checkpoints are ever uploaded, only scalars. Requires the
    # `wandb` extra (`pip install rlt-so101-dual[wandb]`) and `wandb login` done
    # beforehand; if the import fails this is treated the same as wandb=False
    # (a warning is logged, training is not blocked on it).
    wandb: bool = False
    wandb_project: str = "rlt-so101-dual"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    # Stable W&B run identity is separate from the display name. Pass both
    # fields when resuming if the metrics must continue on the same run.
    wandb_run_id: str | None = None
    wandb_resume: str | None = None
    # After each recorded episode ends (s/f pressed), before the teleop reset
    # window, smoothly ramp the follower back to the calibrated middle
    # position (all non-gripper joints = 0 degrees -- exactly the pose set by
    # hand during lerobot-calibrate's homing step) over this many seconds.
    # 0 disables this step (robot stays wherever the episode left it).
    go_home_time_s: float = 3.0
    # Gripper target during go-home (0-100 range, no "middle" concept for an
    # open/close range). VERIFY which end means "open" for your specific
    # hardware (mounting-dependent, not fixed by lerobot) before relying on
    # this -- sending the wrong direction closes the gripper instead of
    # opening it.
    go_home_gripper_value: float = 100.0
    # Per-joint go-home targets as RAW motor ticks -- i.e. paste the POS
    # column straight from lerobot-calibrate's "recording positions" screen,
    # no manual conversion needed. Keyed by action-feature name, e.g.
    # "shoulder_pan.pos" (single arm) or "left_shoulder_pan.pos" /
    # "right_shoulder_pan.pos" (bimanual). Any joint not listed here (e.g.
    # wrist_roll, which lerobot-calibrate excludes from ROM recording) falls
    # back to the calibrated-midpoint default (0 degrees), same as before.
    # Overrides go_home_gripper_value for any gripper joint it lists.
    go_home_positions: dict[str, float] | None = None

@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False
    # In policy mode, broadcast the same robot action to the teleop arm via `teleop.send_feedback`.
    policy_sync_to_teleop: bool = False
    # Use parallel dispatch to reduce action broadcast latency when syncing policy to teleop.
    policy_sync_parallel: bool = True
    # Enable S0/S1/S2 intervention state machine when policy + teleop are both available.
    intervention_state_machine_enabled: bool = True
    # Keyboard key used to toggle entering/leaving intervention.
    intervention_toggle_key: str = "i"
    # Safety hotkey: bound to the same toggle-intervention event as
    # `intervention_toggle_key`, giving a second always-available way to grab
    # manual control (e.g. during autonomous RL-phase rollout in online
    # training). Not a hardware E-stop -- physical supervision is still
    # required.
    estop_key: str = "x"
    # Pure-teleop mode: r key starts an episode (entering critical phase),
    # second r press ends the episode and marks it success; u marks it as
    # failure. No VLA,
    # no RL inference, no SPACE intervention - teleop drives the entire
    # time, r is the only episode-control input. Requires policy to be
    # None (teleop-only) and reuses the same underlying state machine as
    # the rlt wo_prefix recorder.
    teleop_r_key_episodes: bool = False
    # Whether to capture episode-level success/failure labels from keyboard.
    enable_episode_outcome_labeling: bool = False
    # Keyboard key to mark the current episode as success and end it.
    episode_success_key: str = "s"
    # Keyboard key to mark the current episode as failure and end it.
    episode_failure_key: str = "f"
    # Optional fallback label used when no explicit success/failure key was pressed.
    default_episode_success: str | None = None
    # Discard an episode instead of saving or raising when it has no outcome label.
    discard_unlabeled_episodes: bool = False
    # If True, require explicit or default episode labels before saving.
    require_episode_success_label: bool = False
    # Unified schema always records step-level collector source ids.
    enable_collector_policy_id: bool = True
    # Numeric code used when the executed action comes from the primary policy.
    collector_policy_id_policy: int = COLLECTOR_POLICY
    # Numeric code used when the executed action comes from human teleoperation.
    collector_policy_id_human: int = COLLECTOR_HUMAN
    # ACP inference controls for policy-driven recording.
    acp_inference: ACPInferenceConfig = field(default_factory=ACPInferenceConfig)
    # Retry timeout for transient communication errors (seconds). Set to 0 to fail immediately.
    # Demo dual-arm sessions drop Feetech packets around episode boundaries; 5s covers short bus blips.
    communication_retry_timeout_s: float = 5.0
    # Sleep interval between communication retries (seconds).
    communication_retry_interval_s: float = 0.15
    # Enable critical phase labeling via keyboard toggle during recording.
    enable_critical_phase_labeling: bool = False
    # Keyboard key for toggling critical phase marking. Default is space.
    critical_phase_toggle_key: str = " "
    rlt: RLTRecordConfig = field(default_factory=RLTRecordConfig)
    online_rl: OnlineRLConfig = field(default_factory=OnlineRLConfig)
    # Path to a JSON file with robot + camera config (e.g. roboclaw setup.json).
    # When set, overrides robot port and camera CLI args.
    robot_config_file: str | None = None
    # If False, the rlt_ac actor receives a zeroed VLA reference chunk at
    # inference (mirrors training ref-dropout). RL phase only; VLA passthrough
    # and non-rlt_ac policies are unaffected.
    vla_ref: bool = True

    def __post_init__(self):
        if self.robot_config_file is not None:
            from rlt_so101_dual.adapters.lerobot.record.robot_config import load_robot_config_from_json

            self.robot = load_robot_config_from_json(self.robot_config_file)

        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")

        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")

            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        # When RLT is enabled, translate RLT config into a standard RLT policy config
        if self.rlt.enable and self.rlt.vla_model and self.policy is None:
            from rlt_so101_dual.adapters.lerobot.policies.configuration_rlt import RLTPretrainedConfig

            self.policy = RLTPretrainedConfig(
                vla_pretrained_path=self.rlt.vla_model,
                rl_token_ckpt_path=self.rlt.rl_token_ckpt,
                ac_ckpt_path=self.rlt.ac_ckpt,
                task_instruction=self.rlt.task_instruction or self.dataset.single_task,
                phase_mode=self.rlt.phase_mode,
                device=self.rlt.device,
                chunk_length=self.rlt.chunk_length,
                chunk_exec_steps=self.rlt.chunk_exec_steps,
                token_pool_size=self.rlt.token_pool_size,
                image_only=self.rlt.image_only,
                deterministic=self.rlt.deterministic,
                actor_hidden_dim=self.rlt.actor_hidden_dim,
                actor_num_layers=self.rlt.actor_num_layers,
                actor_residual=self.rlt.actor_residual,
                actor_activation=self.rlt.actor_activation,
                actor_layer_norm=self.rlt.actor_layer_norm,
            )
            # preprocessor/postprocessor come from VLA model directory
            self.policy.pretrained_path = self.rlt.vla_model

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator, or enable RLT to control the robot")
        if not self.intervention_toggle_key or len(self.intervention_toggle_key) != 1:
            raise ValueError("`intervention_toggle_key` must be a single character.")
        if not self.estop_key or len(self.estop_key) != 1:
            raise ValueError("`estop_key` must be a single character.")
        if self.estop_key.lower() == self.intervention_toggle_key.lower():
            raise ValueError("`estop_key` must differ from `intervention_toggle_key`.")

        if self.enable_episode_outcome_labeling:
            label_key_bindings = {
                "episode_success_key": self.episode_success_key,
                "episode_failure_key": self.episode_failure_key,
            }
            for key_name, key_value in label_key_bindings.items():
                if not key_value or len(key_value) != 1:
                    raise ValueError(f"`{key_name}` must be a single character.")

            normalized_keys = [
                self.intervention_toggle_key.lower(),
                self.estop_key.lower(),
                self.episode_success_key.lower(),
                self.episode_failure_key.lower(),
            ]
            if len(set(normalized_keys)) != len(normalized_keys):
                raise ValueError(
                    "`intervention_toggle_key`, `estop_key`, `episode_success_key`, and "
                    "`episode_failure_key` must be distinct."
                )

        if self.rlt.enable:
            if not self.rlt.critical_phase_toggle_key or len(self.rlt.critical_phase_toggle_key) != 1:
                raise ValueError("`rlt.critical_phase_toggle_key` must be a single character.")
            if not self.rlt.rl_phase_key or len(self.rlt.rl_phase_key) != 1:
                raise ValueError("`rlt.rl_phase_key` must be a single character.")
            if not self.rlt.rl_phase_failure_key or len(self.rlt.rl_phase_failure_key) != 1:
                raise ValueError("`rlt.rl_phase_failure_key` must be a single character.")

            reserved_keys = [
                self.rlt.critical_phase_toggle_key.lower(),
                self.rlt.rl_phase_key.lower(),
                self.rlt.rl_phase_failure_key.lower(),
                self.intervention_toggle_key.lower(),
                self.estop_key.lower(),
            ]
            if self.enable_episode_outcome_labeling:
                reserved_keys.append(self.episode_success_key.lower())
                reserved_keys.append(self.episode_failure_key.lower())
            if len(set(reserved_keys)) != len(reserved_keys):
                raise ValueError(
                    "RLT phase keys must not collide with intervention or episode outcome keys."
                )

        if self.default_episode_success is not None:
            self.default_episode_success = normalize_episode_success_label(self.default_episode_success)

        if not self.enable_collector_policy_id:
            raise ValueError("`enable_collector_policy_id` must stay true for the unified recording schema.")
        if self.collector_policy_id_human < 0:
            raise ValueError("`collector_policy_id_human` must be >= 0.")
        if self.collector_policy_id_policy < 0:
            raise ValueError("`collector_policy_id_policy` must be >= 0.")
        if self.collector_policy_id_human == self.collector_policy_id_policy:
            raise ValueError("`collector_policy_id_human` and `collector_policy_id_policy` must be distinct.")
        if self.acp_inference.use_cfg and not self.acp_inference.enable:
            raise ValueError("`acp_inference.use_cfg=true` requires `acp_inference.enable=true`.")
        if self.acp_inference.cfg_beta < 0:
            raise ValueError("`acp_inference.cfg_beta` must be >= 0.")
        if self.communication_retry_timeout_s < 0:
            raise ValueError("`communication_retry_timeout_s` must be >= 0.")
        if self.communication_retry_interval_s <= 0:
            raise ValueError("`communication_retry_interval_s` must be > 0.")
        if self.rlt.intervention_action_blend_time_s < 0:
            raise ValueError("`rlt.intervention_action_blend_time_s` must be >= 0.")

        if self.online_rl.enable:
            if self.policy is None or getattr(self.policy, "type", None) != "rlt_ac":
                raise ValueError("`online_rl.enable=true` requires an `rlt_ac`-type `policy`.")
            if not self.enable_episode_outcome_labeling:
                raise ValueError(
                    "`online_rl.enable=true` requires `enable_episode_outcome_labeling=true` "
                    "(episode success/failure is the sparse reward signal)."
                )
            if not self.rlt.rl_phase_key_toggles_critical_phase:
                raise ValueError(
                    "`online_rl.enable=true` requires `rlt.rl_phase_key_toggles_critical_phase=true` "
                    "(the RL-phase key ends the critical-phase attempt and hands back to VLA, without "
                    "ending the whole recorded episode)."
                )
            # skip_prefix_recording is NOT required. It only gates dataset.add_frame;
            # rlt_online_collector.on_frame sits outside that gate, so the replay
            # buffer is unaffected either way. Requiring it made an episode that
            # fails before the critical phase unrecordable: s/f registers the
            # outcome, but with every pre-critical frame dropped the episode has
            # zero frames and gets discarded. Grasp-stage failures then never
            # reach the dataset at all, so every success rate is silently
            # conditioned on having reached the critical phase -- and the discard
            # rate differed between arms by 22 points in one evaluation round.
            if not self.online_rl.save_dir:
                raise ValueError("`online_rl.enable=true` requires `online_rl.save_dir` to be set.")
            if self.online_rl.warmup_episodes < 0:
                raise ValueError("`online_rl.warmup_episodes` must be >= 0.")
            if self.online_rl.critic_only_episodes < 0:
                raise ValueError("`online_rl.critic_only_episodes` must be >= 0.")
            if self.online_rl.min_warmup_transitions < 0:
                raise ValueError("`online_rl.min_warmup_transitions` must be >= 0.")
            if self.online_rl.min_warmup_successes < 0:
                raise ValueError("`online_rl.min_warmup_successes` must be >= 0.")
            if self.online_rl.min_warmup_failures < 0:
                raise ValueError("`online_rl.min_warmup_failures` must be >= 0.")
            # 0 is evaluation mode: num_updates = min(requested, 0) = 0, so no
            # gradient step runs at all and the policy is genuinely frozen. The
            # alternative -- driving the learning rate to ~0 -- does not work on
            # a resumed run, because Adam's load_state_dict restores the
            # snapshot's param_groups and silently puts the old lr back
            # (see OnlineRLTrainer._report_effective_lr).
            if self.online_rl.max_updates_per_episode < 0:
                raise ValueError(
                    "`online_rl.max_updates_per_episode` must be >= 0 "
                    "(0 = run the policy without updating it)."
                )
            if self.online_rl.batch_size <= 0:
                raise ValueError("`online_rl.batch_size` must be > 0.")
            if self.online_rl.replay_capacity < self.online_rl.batch_size:
                raise ValueError("`online_rl.replay_capacity` must be >= `online_rl.batch_size`.")
            if self.online_rl.save_every_episodes <= 0:
                raise ValueError(
                    "`online_rl.save_every_episodes` must be > 0 (used as a modulo divisor)."
                )
            if self.online_rl.lr_actor <= 0 or self.online_rl.lr_critic <= 0:
                raise ValueError("`online_rl.lr_actor` and `online_rl.lr_critic` must be > 0.")
            if not (0 < self.policy.gamma <= 1):
                raise ValueError("`policy.gamma` must be in (0, 1].")
            if self.policy.beta < 0:
                raise ValueError("`policy.beta` must be >= 0.")
            if not (0 <= self.policy.tau <= 1):
                raise ValueError("`policy.tau` must be in [0, 1].")
            if self.policy.utd_ratio <= 0:
                raise ValueError("`policy.utd_ratio` must be > 0.")
            if self.policy.actor_update_interval <= 0:
                raise ValueError("`policy.actor_update_interval` must be > 0.")
            if self.policy.actor_action_clip_delta is not None and self.policy.actor_action_clip_delta < 0:
                raise ValueError("`policy.actor_action_clip_delta` must be >= 0 when set.")
            if self.online_rl.go_home_time_s < 0:
                raise ValueError("`online_rl.go_home_time_s` must be >= 0.")
            if not (0 <= self.online_rl.go_home_gripper_value <= 100):
                raise ValueError("`online_rl.go_home_gripper_value` must be in [0, 100].")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

def _ensure_human_inloop_compatible_features(
    dataset_features: dict[str, dict],
    *,
    action_feature_names: list[str],
) -> None:
    # Unified annotation schema shared by future recorded datasets.
    dataset_features["complementary_info.policy_action"] = {
        "dtype": "float32",
        "shape": (len(action_feature_names),),
        "names": action_feature_names,
    }
    dataset_features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["is_intervention"],
    }
    dataset_features["complementary_info.state"] = {
        "dtype": "float32",
        "shape": (1,),
        "names": ["state"],
    }
    dataset_features["complementary_info.phase"] = {"dtype": "float32", "shape": (1,), "names": ["phase"]}

def _add_collector_policy_id_feature(dataset_features: dict[str, dict]) -> None:
    dataset_features["complementary_info.collector_policy_id"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": ["collector_policy_id"],
    }

def _build_collector_policy_id_codebook(cfg: RecordConfig) -> dict[str, str]:
    if cfg.rlt.enable:
        return {str(code): name for code, name in RLT_COLLECTOR_POLICY_ID_TO_NAME.items()}
    if cfg.policy is None:
        return {str(cfg.collector_policy_id_human): "human"}
    return {
        str(cfg.collector_policy_id_human): "human",
        str(cfg.collector_policy_id_policy): infer_collector_policy_version(cfg.policy),
    }

def _write_schema_metadata(
    dataset: LeRobotDataset,
    *,
    collector_policy_id_codebook: dict[str, str],
    include_rlt_episode_metadata: bool,
) -> None:
    collector_info = dataset.meta.info["features"].get("complementary_info.collector_policy_id")
    if collector_info is None:
        return
    collector_info["info"] = {"codebook": collector_policy_id_codebook}
    if include_rlt_episode_metadata:
        dataset.meta.info["rlt_episode_metadata_fields"] = {
            "rl_intervals": "List of {start_frame, end_frame, outcome} for each RL phase.",
            "human_intervention_intervals": "List of {start_frame, end_frame} for each human intervention segment.",
        }
    dataset.meta.info["recording_schema_version"] = 2
    write_info(dataset.meta.info, dataset.root)

def _configure_rlt_record_policy(policy, cfg: RecordConfig) -> None:
    from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy

    if not isinstance(policy, ChunkACPolicy):
        return
    policy.vla_ref = cfg.vla_ref
    logging.info("rlt_ac vla_ref=%s (False => zeroed VLA reference chunk)", cfg.vla_ref)
    if not cfg.rlt.rtc_enabled:
        return

    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    schedule = RTCAttentionSchedule(cfg.rlt.rtc_prefix_attention_schedule)
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=cfg.rlt.rtc_execution_horizon,
        max_guidance_weight=cfg.rlt.rtc_max_guidance_weight,
        prefix_attention_schedule=schedule,
    )
    vla_rtc_config = None
    if cfg.rlt.vla_rtc_execution_horizon is not None:
        vla_rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=cfg.rlt.vla_rtc_execution_horizon,
            max_guidance_weight=cfg.rlt.rtc_max_guidance_weight,
            prefix_attention_schedule=schedule,
        )
    policy.configure_rtc(
        rtc_config,
        fps=cfg.dataset.fps,
        action_queue_size_to_get_new_actions=cfg.rlt.rtc_action_queue_size_to_get_new_actions,
        vla_rtc_config=vla_rtc_config,
    )
    logging.info(
        "rlt_ac RTC enabled: rlt_horizon=%d vla_horizon=%d guidance=%.3f schedule=%s refill_threshold=%s",
        cfg.rlt.rtc_execution_horizon,
        cfg.rlt.vla_rtc_execution_horizon or cfg.rlt.rtc_execution_horizon,
        cfg.rlt.rtc_max_guidance_weight,
        cfg.rlt.rtc_prefix_attention_schedule,
        cfg.rlt.rtc_action_queue_size_to_get_new_actions,
    )

def _raw_ticks_to_normalized(robot, action_name: str, raw_value: float) -> float:
    """Convert a raw motor tick (``rlt-so101-dual-record-pose`` / calibrate POS)
    into the units ``robot.send_action()`` expects for that motor.

    Followers here use ``use_degrees=True``, so body joints are degrees — not
    the [-100, 100] range. Always emitting RANGE_M100_100 made a recorded park
    pose command wildly wrong targets.
    """
    motor_name = action_name.removesuffix(".pos")
    arm = robot
    for prefix, attr in (("left_", "left_arm"), ("right_", "right_arm")):
        if motor_name.startswith(prefix) and hasattr(robot, attr):
            arm = getattr(robot, attr)
            motor_name = motor_name[len(prefix) :]
            break
    motor_id = arm.bus.motors[motor_name].id
    return float(arm.bus._normalize({motor_id: int(raw_value)})[motor_id])

def _build_phase_trackers(auto_save_path, want_intervention_tracker: bool):
    """Build the interval trackers, or degrade loudly if they do not exist here.

    `lerobot.utils.critical_phase_tracker` lives in the upstream project's own
    lerobot fork, not in the lerobot 0.5.1 this project pins -- the class is not
    in the environment and not in this repo. The import therefore sat behind a
    condition that never became true, so it was never executed, until 741c744
    made the condition reachable and turned dead code into a crash on the first
    real `--split-critical-phase` run.

    Returning (None, None) restores exactly the behaviour every recording so far
    has had: `r`/`u` still drive the phase state machine (the marks are already
    guarded by `is not None`), the reward still flushes, and phase outcomes are
    still recoverable from the log -- only the frame-indexed JSON is missing.
    The warning is deliberately loud: nothing else would tell the operator that
    the artifact they expect is not going to appear.
    """
    try:
        from lerobot.utils.critical_phase_tracker import (  # noqa: PLC0415
            CriticalPhaseTracker,
            EpisodeIntervalTracker,
        )
    except ModuleNotFoundError:
        logging.warning(
            "lerobot.utils.critical_phase_tracker is not available in this lerobot "
            "(it belongs to the upstream fork), so %s will NOT be written. Critical-phase "
            "outcomes still reach the log as 'RL phase started' / 'RL phase ended via ...' "
            "and can be recovered from there; frame indices cannot.",
            auto_save_path,
        )
        return None, None
    tracker = CriticalPhaseTracker(auto_save_path=auto_save_path)
    intervention = EpisodeIntervalTracker(label="Human intervention") if want_intervention_tracker else None
    return tracker, intervention

def _should_track_critical_phase(cfg, teleop_r_key_mode: bool) -> bool:
    """Whether to build the CriticalPhaseTracker that writes
    `critical_phase_intervals.json`.

    The r key toggling a critical phase is itself the reason to record where
    those phases were: `rl_phase_key_toggles_critical_phase` is set by both
    `full --split-critical-phase` and `rlt-so101-dual-online-train`, and in both the
    r/u presses already call the tracker's toggle/mark_success/mark_failure --
    they were simply being skipped by the `is not None` guard because nothing
    built one.

    Before this clause the tracker needed `enable_critical_phase_labeling` or
    `rlt.vla_model`, and **nothing in the repo ever sets either**: online_cli
    passes `--policy.vla_pretrained_path`, not `--rlt.vla_model`. So the file was
    never written on any path. The 30-episode baseline of 2026-08-20 logged 25
    critical phases and produced no intervals file; its per-phase numbers had to
    be reconstructed from log lines, which carry outcomes but no frame indices.

    Deliberately not exposing `enable_critical_phase_labeling` as a flag
    instead: it also binds `critical_phase_toggle_key` and, whenever
    `rlt.enable` is false, binds cp_success/cp_failure onto s/f -- which would
    take those keys away from the episode-outcome listener again, the exact
    regression 9db8a9e fixed.
    """
    return bool(
        cfg.enable_critical_phase_labeling
        or (cfg.rlt.enable and cfg.rlt.vla_model)
        or (cfg.rlt.enable and cfg.rlt.rl_phase_key_toggles_critical_phase)
        or teleop_r_key_mode
    )

@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    if cfg.require_episode_success_label and not cfg.enable_episode_outcome_labeling:
        raise ValueError(
            "`require_episode_success_label=true` requires `enable_episode_outcome_labeling=true`."
        )
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    action_names = dataset_features[ACTION]["names"]
    action_names = list(robot.action_features) if action_names is None else list(action_names)
    _ensure_human_inloop_compatible_features(dataset_features, action_feature_names=action_names)
    if cfg.enable_collector_policy_id:
        _add_collector_policy_id_feature(dataset_features)

    dataset = None
    listener = None
    policy_sync_executor = None
    critical_phase_tracker = None
    intervention_tracker = None
    online_trainer = None
    online_rl_resume_episodes = None

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Create empty dataset or load existing saved episodes
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
        _write_schema_metadata(
            dataset,
            collector_policy_id_codebook=_build_collector_policy_id_codebook(cfg),
            include_rlt_episode_metadata=cfg.rlt.enable,
        )

        # Load pretrained policy
        policy = (
            None
            if cfg.policy is None
            else make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.dataset.rename_map)
        )
        _configure_rlt_record_policy(policy, cfg)

        online_collector = None
        if cfg.online_rl.enable:
            from rlt_so101_dual.adapters.lerobot.record.online_trainer import OnlineRLTrainer

            online_trainer = OnlineRLTrainer(
                policy, cfg.online_rl, cfg.policy, task=cfg.dataset.single_task
            )
            online_collector = online_trainer.collector
            if cfg.online_rl.resume_from is not None:
                online_rl_resume_episodes = online_trainer.load_latest_state(cfg.online_rl.resume_from)

        preprocessor = None
        postprocessor = None
        if cfg.acp_inference.enable and cfg.policy is None:
            raise ValueError("`acp_inference.enable=true` requires `policy` to be set.")
        if cfg.policy is not None:
            dataset_stats = rename_stats(dataset.meta.stats, cfg.dataset.rename_map)
            if not dataset_stats and getattr(cfg.policy, "vla_pretrained_path", None):
                # Fresh rlt_ac policy (online RL): no --policy.path checkpoint
                # of its own, and this dataset has zero episodes, so the
                # usual dataset_stats source is empty -- fall back to the
                # REAL normalization the frozen VLA was actually trained
                # with, from its own checkpoint. See
                # load_dataset_stats_from_pretrained()'s docstring for why
                # this matters (un-normalized state/action reads as the
                # VLA "acting randomly").
                dataset_stats = load_dataset_stats_from_pretrained(cfg.policy.vla_pretrained_path)
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=dataset_stats,
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        collector_policy_id_policy = cfg.collector_policy_id_policy
        collector_policy_id_human = cfg.collector_policy_id_human

        robot.connect()
        if teleop is not None:
            teleop.connect()
        on_record_connected = getattr(cfg, "_on_record_connected", None)
        if callable(on_record_connected):
            on_record_connected(robot, teleop)

        if cfg.policy_sync_to_teleop:
            if cfg.policy is None:
                raise ValueError("`policy_sync_to_teleop=true` requires `policy` to be set.")
            if teleop is None or isinstance(teleop, list):
                raise ValueError(
                    "`policy_sync_to_teleop=true` requires exactly one teleoperator with send_feedback support."
                )
            policy_sync_executor = PolicySyncDualArmExecutor(
                robot=robot,
                teleop=teleop,
                parallel_dispatch=cfg.policy_sync_parallel,
            )

        critical_phase_tracker = None
        teleop_r_key_mode = cfg.teleop_r_key_episodes and policy is None
        if _should_track_critical_phase(cfg, teleop_r_key_mode):
            critical_phase_tracker, intervention_tracker = _build_phase_trackers(
                dataset.root / "critical_phase_intervals.json", cfg.rlt.enable
            )

        # RLT policy is now a standard PreTrainedPolicy - no separate instantiation needed

        cp_key = cfg.critical_phase_toggle_key if cfg.enable_critical_phase_labeling else None
        if cp_key is None and cfg.rlt.enable:
            cp_key = cfg.rlt.critical_phase_toggle_key

        # RLT HIL mode: use SPACE for intervention, r/s/f for phase control
        rlt_hil_mode = cfg.rlt.enable and policy is not None and teleop is not None
        rlt_active = cfg.rlt.enable and policy is not None
        rlt_key_controls_phase = (
            cfg.rlt.rl_phase_key_toggles_episode or cfg.rlt.rl_phase_key_toggles_critical_phase
        )
        if rlt_active and rlt_key_controls_phase:
            rl_phase_key_binding = cfg.rlt.rl_phase_key
            rl_phase_failure_key_binding = cfg.rlt.rl_phase_failure_key
        elif teleop_r_key_mode:
            rl_phase_key_binding = "r"
            rl_phase_failure_key_binding = "u"
        else:
            rl_phase_key_binding = None
            rl_phase_failure_key_binding = None
        # Only bind the milestone key when a bonus can actually be awarded:
        # it needs online RL collecting transitions, and a nonzero reward.
        milestone_key_binding = (
            cfg.rlt.milestone_key
            if cfg.online_rl.enable and cfg.online_rl.milestone_reward > 0.0
            else None
        )
        # In teleop_r_key_mode the r key is the single episode-control input;
        # unbind s/f so the user cannot accidentally end an episode out of
        # the r/u outcome state machine.
        bind_ep_outcome_keys = cfg.enable_episode_outcome_labeling and not teleop_r_key_mode
        listener, events = init_keyboard_listener(
            intervention_toggle_key=(
                (" " if rlt_hil_mode else cfg.intervention_toggle_key) if policy is not None else None
            ),
            estop_key=cfg.estop_key if policy is not None else None,
            critical_phase_toggle_key=cp_key if not rlt_active else None,
            episode_success_key=cfg.episode_success_key if bind_ep_outcome_keys else None,
            episode_failure_key=cfg.episode_failure_key if bind_ep_outcome_keys else None,
            cp_success_key="s" if cfg.enable_critical_phase_labeling and not rlt_active else None,
            cp_failure_key="f" if cfg.enable_critical_phase_labeling and not rlt_active else None,
            rl_phase_key=rl_phase_key_binding,
            rl_phase_failure_key=rl_phase_failure_key_binding,
            milestone_key=milestone_key_binding,
            # In critical-phase-toggle mode (`full --split-critical-phase`) the
            # r/u pair already resolves the phase -- loop.py's
            # _handle_rl_phase_start_event marks success on the second r press.
            # Binding end_success_key/end_failure_key on top of that is not just
            # redundant, it silently steals the episode label: they default to
            # the same s/f as episode_success_key/episode_failure_key, and
            # runner.py's key_bindings is a dict keyed by the character, so the
            # later entry wins and the episode-outcome listener ends up with
            # nothing bound. The episode then cannot be labelled or ended early
            # -- it runs to episode_time_s and silently takes
            # default_episode_success. Measured: 3299 frames (110 s of a 120 s
            # limit) and episode_success='failure' after three r/s pairs that
            # had each marked the critical phase a success.
            #
            # rl_phase_key_toggles_episode mode (collect/segment, runner.py:535
            # and :737 -- not online RL, which uses the critical-phase toggle) is
            # untouched: there
            # _mark_rl_phase_success(toggles_episode=True) sets both the outcome
            # and exit_early, so s/f really do end the episode.
            end_success_key=(
                cfg.rlt.end_success_key
                if rlt_active and not cfg.rlt.rl_phase_key_toggles_critical_phase
                else None
            ),
            end_failure_key=(
                cfg.rlt.end_failure_key
                if rlt_active and not cfg.rlt.rl_phase_key_toggles_critical_phase
                else None
            ),
        )

        def _warmup_rlt_path() -> None:
            if not (rlt_active and policy is not None and preprocessor is not None and postprocessor is not None):
                return
            log_say("Warming up RL path", cfg.play_sounds)
            warmup_obs = robot.get_observation()
            warmup_obs_processed = robot_observation_processor(warmup_obs)
            warmup_frame = build_dataset_frame(dataset.features, warmup_obs_processed, prefix=OBS_STR)
            if hasattr(policy, "set_rl_mode"):
                policy.set_rl_mode()
            _predict_policy_action_with_acp_inference(
                observation_frame=warmup_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=cfg.dataset.single_task,
                robot_type=robot.robot_type,
                acp_inference=cfg.acp_inference,
            )
            # Warming up the RL path deliberately primes the actor's compute
            # path once (see set_rl_mode() above) -- but nothing sent a robot
            # action while doing so, and _reset_policy_for_episode() below is
            # relied on to leave the policy back in VLA phase before the first
            # real episode starts. Don't rely on that alone: explicitly force
            # VLA mode back here too, so a bug/edge case in reset()'s phase
            # handling can't leave a session starting in critical phase with
            # no r ever pressed.
            if hasattr(policy, "set_vla_mode"):
                policy.set_vla_mode()
            if hasattr(policy, "pop_step_metadata"):
                # Don't just trust set_vla_mode() succeeded silently -- run one
                # more prediction and directly read back the phase it actually
                # computed under, so a bug here is impossible to misread (this
                # log line either says phase=0.0 or it says phase=1.0, no
                # inference required from downstream RLT_ACTOR prints).
                _predict_policy_action_with_acp_inference(
                    observation_frame=warmup_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=cfg.dataset.single_task,
                    robot_type=robot.robot_type,
                    acp_inference=cfg.acp_inference,
                )
                confirm_meta = policy.pop_step_metadata()
                if confirm_meta is not None:
                    logging.info(
                        "Post-warmup phase check: phase=%s (0.0=VLA, 1.0=critical) source=%s "
                        "(should be 0.0/0.0 here -- if not, VLA mode is not taking effect)",
                        confirm_meta.phase, confirm_meta.source_type,
                    )
            log_say("Ready", cfg.play_sounds)

        def _start_episode_trackers() -> None:
            if critical_phase_tracker is not None:
                critical_phase_tracker.on_episode_start(dataset.num_episodes)
            if intervention_tracker is not None:
                intervention_tracker.on_episode_start(dataset.num_episodes)

        def _reset_policy_for_episode() -> None:
            if policy is not None and hasattr(policy, "set_rl_mode"):
                policy.reset()
                # reset() already resets the phase controller to VLA (for
                # phase_mode="manual", see ChunkACPolicy.reset()) -- this is
                # a belt-and-suspenders explicit call on top, not a
                # workaround for a known bug there. Every episode must start
                # in VLA phase regardless of how the previous one ended.
                if hasattr(policy, "set_vla_mode"):
                    policy.set_vla_mode()

        def _record_episode() -> None:
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                display_compressed_images=display_compressed_images,
                policy_sync_executor=policy_sync_executor,
                intervention_state_machine_enabled=cfg.intervention_state_machine_enabled,
                collector_policy_id_policy=collector_policy_id_policy,
                collector_policy_id_human=collector_policy_id_human,
                acp_inference=cfg.acp_inference,
                communication_retry_timeout_s=cfg.communication_retry_timeout_s,
                communication_retry_interval_s=cfg.communication_retry_interval_s,
                rename_map=cfg.dataset.rename_map,
                critical_phase_tracker=critical_phase_tracker,
                rlt_intervention_tracker=intervention_tracker,
                skip_prefix_recording=cfg.rlt.skip_prefix_recording or teleop_r_key_mode,
                rl_phase_key_toggles_episode=cfg.rlt.rl_phase_key_toggles_episode or teleop_r_key_mode,
                rl_phase_key_toggles_critical_phase=cfg.rlt.rl_phase_key_toggles_critical_phase,
                start_in_teleop=cfg.rlt.start_in_teleop,
                intervention_action_blend_time_s=cfg.rlt.intervention_action_blend_time_s,
                rlt_online_collector=online_collector,
            )

        def _current_episode_frame_count() -> int:
            writer = getattr(dataset, "writer", None)
            episode_buffer = getattr(writer, "episode_buffer", None)
            return episode_buffer["size"] if episode_buffer else 0

        def _finish_episode_trackers() -> None:
            ep_frames = _current_episode_frame_count()
            if critical_phase_tracker is not None:
                critical_phase_tracker.on_episode_end(ep_frames)
            if intervention_tracker is not None:
                intervention_tracker.on_episode_end(ep_frames)

        def _resolve_current_episode_success() -> str | None:
            if not cfg.enable_episode_outcome_labeling:
                return None
            episode_success = resolve_episode_success_label(
                explicit_label=events.get("episode_outcome"),
                default_label=cfg.default_episode_success,
                require_label=cfg.require_episode_success_label and not cfg.discard_unlabeled_episodes,
            )
            if events.get("episode_outcome") is None and episode_success is not None:
                logging.warning(
                    "Episode %s has no explicit success/failure label, defaulting to '%s'.",
                    dataset.num_episodes,
                    episode_success,
                )
            return episode_success

        def _notify_episode_outcome(episode_success: str | None) -> None:
            on_episode_outcome = getattr(cfg, "_on_record_episode_outcome", None)
            if callable(on_episode_outcome):
                on_episode_outcome(robot, teleop, episode_success)

        def _should_run_reset_loop(recorded_episodes: int, *, was_rerecord: bool = False) -> bool:
            # Call after _finish_recorded_episode so encoder threads are already
            # stopped. was_rerecord must be captured before finish clears the flag.
            next_episode_needed = recorded_episodes < cfg.dataset.num_episodes
            return (
                not events["stop_recording"]
                and not cfg.rlt.start_in_teleop
                and not teleop_r_key_mode
                and (was_rerecord or next_episode_needed)
            )

        def _run_go_home_if_needed() -> None:
            """After the recorded episode ends (s/f pressed), before the
            teleop reset window, smoothly ramp the follower back to the
            go-home position (see OnlineRLConfig.go_home_positions /
            go_home_time_s).

            Requires an explicit ``go_home_positions`` dict. The old fallback
            (every non-gripper joint → 0°) is the calibrated *mid-range*
            pose, which on SO101 dual arms looks like both arms stretched
            straight out — then the reset teleop snaps them back to the
            leaders. That is unsafe and confusing; refuse to invent a park
            pose.
            """
            if not cfg.online_rl.enable or cfg.online_rl.go_home_time_s <= 0:
                return
            raw_targets = cfg.online_rl.go_home_positions or {}
            if not raw_targets:
                logging.warning(
                    "Skipping go-home after s/f: --go-home-positions was not set. "
                    "Defaulting to 0° (calibrated midpoint) stretches both SO101 arms "
                    "out; record a park pose with "
                    "`rlt-so101-dual-record-pose --setup-json configs/hardware/so101_dual_manifest.json` "
                    "and pass the printed JSON, or set --go-home-time-s 0 to silence this."
                )
                return
            try:
                import time as _time

                from lerobot.utils.robot_utils import precise_sleep

                start_action = robot.get_observation()
                action_names = list(robot.action_features.keys())
                # Names that match no joint used to be dropped in silence, so a
                # go-home pose recorded on a different robot degraded into
                # "0 degrees everywhere" -- a pose the arm would ramp to at full
                # speed without warning.
                unknown = sorted(set(raw_targets) - set(action_names))
                if unknown:
                    raise ValueError(
                        f"go_home_positions names {unknown} match no joint on this robot. "
                        f"Known joints: {action_names}."
                    )
                home_action = {
                    name: (
                        _raw_ticks_to_normalized(robot, name, raw_targets[name])
                        if name in raw_targets
                        else cfg.online_rl.go_home_gripper_value
                        if name.endswith("gripper.pos")
                        else float(start_action[name])
                    )
                    for name in action_names
                }
                steps = max(1, round(cfg.online_rl.go_home_time_s * cfg.dataset.fps))
                logging.info(
                    "Go-home: ramping %d joints over %.1fs (%d steps) to recorded park pose "
                    "(per-step motion still limited by max_relative_target).",
                    len(action_names),
                    cfg.online_rl.go_home_time_s,
                    steps,
                )
                for i in range(1, steps + 1):
                    step_t = _time.perf_counter()
                    alpha = i / steps
                    blended = {
                        name: start_action[name] + (home_action[name] - start_action[name]) * alpha
                        for name in action_names
                    }
                    robot.send_action(blended)
                    precise_sleep(max(1 / cfg.dataset.fps - (_time.perf_counter() - step_t), 0.0))
                # Leaders were NOT driven during the follower ramp above (policy
                # sync only runs in the policy control loop). If we open the
                # reset teleop window now, followers chase stale leader poses
                # and undo go-home. Sync leaders to the same park pose first.
                if teleop is not None and not isinstance(teleop, list):
                    try:
                        # Soften PSU load: pause after follower motion, then
                        # move leaders alone toward the same park.
                        _time.sleep(1.0)
                        leader_steps = max(1, round(min(cfg.online_rl.go_home_time_s, 3.0) * cfg.dataset.fps))
                        ramp_teleop_to_action(
                            teleop,
                            home_action,
                            steps=leader_steps,
                            fps=float(cfg.dataset.fps),
                            reason="Go-home leader sync",
                        )
                    except Exception:
                        logging.exception(
                            "Leader go-home sync failed; reset teleop may snap followers "
                            "toward wherever the leaders currently sit — align leaders by hand if needed."
                        )
            except ValueError:
                # Misconfiguration, not a transient hardware fault -- surface it
                # instead of parking the arm somewhere unintended every episode.
                raise
            except Exception:
                logging.exception("Go-home ramp failed; leaving robot where the episode left it.")

        def _sync_leaders_to_followers_before_reset() -> None:
            """Match leaders to current follower pose before unlocking teleop."""
            if teleop is None or isinstance(teleop, list):
                return
            try:
                obs = robot.get_observation()
                action_names = list(robot.action_features.keys())
                target = {name: float(obs[name]) for name in action_names if name in obs}
                if not target:
                    return
                steps = max(1, round(1.5 * cfg.dataset.fps))
                ramp_teleop_to_action(
                    teleop,
                    target,
                    steps=steps,
                    fps=float(cfg.dataset.fps),
                    reason="Pre-reset leader↔follower align",
                )
            except Exception:
                logging.exception(
                    "Pre-reset leader sync failed; followers may jump when teleop unlocks."
                )

        def _run_reset_loop_if_needed(recorded_episodes: int, *, was_rerecord: bool = False) -> None:
            if not _should_run_reset_loop(recorded_episodes, was_rerecord=was_rerecord):
                return
            log_say("Reset the environment", cfg.play_sounds)
            if robot.name == "unitree_g1":
                robot.reset()
            # Go-home (followers) can brown out a shared PSU; give buses a moment
            # before unlocking leaders for the teleop reset window.
            time.sleep(1.0)
            # Align leaders to wherever followers are *now* (park if go-home ran,
            # else end-of-episode pose) so unlocking teleop does not yank the
            # followers across a large leader/follower gap.
            _sync_leaders_to_followers_before_reset()
            try:
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    policy_sync_executor=policy_sync_executor,
                    intervention_state_machine_enabled=cfg.intervention_state_machine_enabled,
                    collector_policy_id_policy=collector_policy_id_policy,
                    collector_policy_id_human=collector_policy_id_human,
                    acp_inference=cfg.acp_inference,
                    communication_retry_timeout_s=cfg.communication_retry_timeout_s,
                    communication_retry_interval_s=cfg.communication_retry_interval_s,
                )
            except Exception:
                # Episode data + online RL state are already saved above. A bus
                # fault here must not discard that work or kill the session.
                logging.exception(
                    "Reset teleop window failed (often leader bus Input voltage error). "
                    "Episode is kept; physically reset the scene, power-cycle arms if "
                    "needed, then continue or restart online-train with --resume-from."
                )
                return
            # Brief bus settle before the next episode's sync_read / encoder
            # start. After episode 1+ the save path also remuxes growing mp4s
            # (heavy CPU/IO); give the Feetech bus longer to recover.
            settle_s = 3.0 if recorded_episodes >= 2 else 1.5
            logging.info("Settling follower bus for %.1fs before next episode.", settle_s)
            time.sleep(settle_s)

        def _discard_rerecord_episode() -> bool:
            if not events["rerecord_episode"]:
                return False
            log_say("Re-record episode", cfg.play_sounds)
            if critical_phase_tracker is not None:
                critical_phase_tracker.discard_episode(dataset.num_episodes)
            if intervention_tracker is not None:
                intervention_tracker.discard_episode(dataset.num_episodes)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            events["episode_outcome"] = None
            dataset.clear_episode_buffer()
            return True

        def _extra_episode_metadata(episode_success: str | None) -> dict | None:
            extra_episode_metadata = {}
            if cfg.enable_episode_outcome_labeling:
                extra_episode_metadata["episode_success"] = episode_success
            if cfg.rlt.enable:
                episode_idx = dataset.num_episodes
                extra_episode_metadata["rl_intervals"] = (
                    critical_phase_tracker.serialize_episode_intervals(episode_idx)
                    if critical_phase_tracker is not None
                    else []
                )
                extra_episode_metadata["human_intervention_intervals"] = (
                    intervention_tracker.serialize_episode_intervals(episode_idx)
                    if intervention_tracker is not None
                    else []
                )
            return extra_episode_metadata or None

        def _report_episode(frames: int, episode_success: str | None) -> None:
            """One line per kept episode: what it cost, and what the session
            is averaging. The running average is the number to plan from --
            the reset window and go-home are not in episode_time_s but are
            most of the wall clock."""
            nonlocal kept_episodes, kept_frames, kept_recorded_s
            kept_episodes += 1
            kept_frames += frames
            kept_recorded_s += episode_recorded_s
            wall_s = time.perf_counter() - session_started_at
            rate = frames / episode_recorded_s if episode_recorded_s > 0 else 0.0
            logging.info(
                "Episode %d kept: %d frames, %.1fs recorded (%.1f fps), outcome %s"
                "  |  session: %d kept, %d frames, %.1f min recorded, %.1f min wall, "
                "%.1f min/episode",
                dataset.num_episodes - 1, frames, episode_recorded_s, rate,
                episode_success or "unlabeled",
                kept_episodes, kept_frames, kept_recorded_s / 60, wall_s / 60,
                wall_s / 60 / max(kept_episodes, 1),
            )
            if abs(rate - cfg.dataset.fps) > cfg.dataset.fps * 0.05:
                logging.warning(
                    "Recorded at %.1f fps but the dataset is declared %d fps. A camera "
                    "delivering fewer frames than it advertises is the usual cause; "
                    "check with rlt-so101-dual-preflight.",
                    rate, cfg.dataset.fps,
                )

        def _finish_recorded_episode(recorded_episodes: int, episode_success: str | None) -> int:
            if _discard_rerecord_episode():
                return recorded_episodes
            if cfg.discard_unlabeled_episodes and episode_success is None:
                logging.info("Discarding unlabeled episode %s.", dataset.num_episodes)
                dataset.clear_episode_buffer()
                return recorded_episodes
            if _current_episode_frame_count() == 0:
                # Can happen with rlt.skip_prefix_recording=true if the
                # episode-outcome key (s/f) is pressed before ever entering
                # critical phase (r): every frame so far was PHASE_PREFIX and
                # none were added to the dataset, so save_episode() would
                # crash on lerobot's "must add_frame before add_episode"
                # check. Treat it the same as a labeled-but-empty attempt --
                # discard, don't count it, and don't crash the session.
                logging.warning(
                    "Discarding episode %s: outcome '%s' was recorded but it has zero frames "
                    "(likely s/f pressed before entering critical phase with r).",
                    dataset.num_episodes, episode_success,
                )
                dataset.clear_episode_buffer()
                return recorded_episodes
            frames = _current_episode_frame_count()
            dataset.save_episode(extra_episode_metadata=_extra_episode_metadata(episode_success))
            _report_episode(frames, episode_success)
            return recorded_episodes + 1

        def _run_online_rl_update(completed_episodes: int, buffer_total_added_before: int) -> None:
            """flush_episode() itself already happened (if at all) inside
            loop.py, at the moment the critical phase resolved -- NOT here,
            and NOT keyed off the whole recorded episode's outcome label (see
            OnlineRLConfig's docstring). online_trainer.maybe_update() just
            checks whether that happened (via the total_added delta) and, if
            so, trains -- see OnlineRLTrainer for the actual TD3+BC logic.

            If this whole episode is being rerecorded (left arrow), roll back
            instead: whatever flush_episode() already committed to the buffer
            this cycle is undone here, before _finish_recorded_episode()'s
            _discard_rerecord_episode() clears the raw dataset episode below
            -- otherwise the discarded attempt's transitions (and any
            gradient step already taken on them) would silently survive in
            the replay buffer even though the episode itself was thrown out."""
            if online_trainer is None:
                return
            if events["rerecord_episode"]:
                online_trainer.discard_episode(buffer_total_added_before)
                return
            online_trainer.maybe_update(completed_episodes, buffer_total_added_before)

        def _save_online_rl_latest_state(completed_episodes: int) -> None:
            if online_trainer is None:
                return
            online_trainer.save_latest_state(completed_episodes)

        with VideoEncodingManager(dataset):
            _warmup_rlt_path()
            # online_rl_resume_episodes carries over the online-RL episode
            # counter from a resumed session (see online_rl.resume_from
            # above) -- dataset.num_episodes (this run's own, freshly created
            # video dataset) intentionally starts at 0 regardless; the two
            # counters are independent and only coincide in a fresh session.
            # --dataset.num_episodes is then a total target inclusive of the
            # resumed count, not "N more episodes".
            recorded_episodes = online_rl_resume_episodes if online_rl_resume_episodes is not None else 0
            # Per-episode timing, so a session can be planned from what the
            # first few episodes actually cost rather than from the nominal
            # episode_time_s. `recorded_s` is time spent recording; the wall
            # clock also carries go-home and the reset window, which together
            # are usually the larger half.
            session_started_at = time.perf_counter()
            episode_recorded_s = 0.0
            kept_episodes = 0
            kept_frames = 0
            kept_recorded_s = 0.0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                events["episode_outcome"] = None
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
                # Ignore s/f/←/→/space/Esc for a short arming window so a key
                # repeat or accidental tap cannot end the episode in <2s.
                # Keep in sync with runner.HOTKEY_ARM_DELAY_S.
                events["ignore_hotkeys_until"] = time.perf_counter() + 2.0
                # Probe + retry before spinning up new encoders. A bare one-shot
                # get_observation after remux often fails once; wait out the
                # same communication_retry window so episode start is quiet.
                _probe_deadline = time.perf_counter() + max(cfg.communication_retry_timeout_s, 0.0)
                _probe_attempts = 0
                while True:
                    _probe_attempts += 1
                    try:
                        robot.get_observation()
                        if _probe_attempts > 1:
                            logging.warning(
                                "Follower bus probe recovered after %d tries before episode %s.",
                                _probe_attempts - 1,
                                dataset.num_episodes,
                            )
                        break
                    except ConnectionError as exc:
                        if cfg.communication_retry_timeout_s <= 0 or time.perf_counter() >= _probe_deadline:
                            logging.error(
                                "Follower bus still dead after %.1fs before episode %s: %s. "
                                "Power-cycle the left follower, run preflight, then resume.",
                                cfg.communication_retry_timeout_s,
                                dataset.num_episodes,
                                exc,
                            )
                            raise
                        if _probe_attempts == 1:
                            logging.warning(
                                "Follower bus probe failed before episode %s; retrying for %.1fs (%s)",
                                dataset.num_episodes,
                                cfg.communication_retry_timeout_s,
                                exc,
                            )
                        time.sleep(cfg.communication_retry_interval_s)
                _start_episode_trackers()
                _reset_policy_for_episode()
                # Captured before _record_episode(): most of an episode's
                # transitions are added live, frame-by-frame, during rollout
                # (RLTOnlineCollector.on_frame's non-terminal _emit_transition
                # calls) -- only the terminal one is added later by
                # flush_episode() inside _run_online_rl_update. Baselining
                # total_added here (not inside _run_online_rl_update, after
                # rollout already happened) is what makes the delta actually
                # cover the whole episode.
                # recorded_episodes, not dataset.num_episodes: the replay
                # buffer's ChunkTransition.episode_id must stay monotonic
                # across a resume_from (see online_rl.resume_from above) --
                # dataset.num_episodes restarts at 0 for this run's own fresh
                # video dataset, which would collide with episode_ids already
                # present in a resumed replay buffer and corrupt
                # episode_outcomes()'s per-episode success/failure grouping.
                # In a non-resumed run the two counters are always equal here
                # (both only advance together via _finish_recorded_episode()),
                # so this is a no-op change for that case.
                buffer_total_added_before = (
                    online_trainer.start_episode(recorded_episodes) if online_trainer is not None else 0
                )
                _episode_started_at = time.perf_counter()
                _record_episode()
                episode_recorded_s = time.perf_counter() - _episode_started_at
                _finish_episode_trackers()
                episode_success = (
                    None if events["rerecord_episode"] else _resolve_current_episode_success()
                )
                _notify_episode_outcome(episode_success)
                # +1: the episode just recorded counts as completed, so
                # --warmup-episodes N is satisfied when the Nth finishes
                # rather than the (N+1)th. The periodic step_NNNNNN saves
                # inside maybe_update() use the same number, so this must
                # not be added twice.
                _run_online_rl_update(recorded_episodes + 1, buffer_total_added_before)
                # Safety net: a gradient update (and the warmup-satisfied
                # transition, if this was the episode that crossed it) has
                # already happened in memory at this point, but go-home and
                # the reset window below are real robot motion that can take
                # 15-20+ seconds and are exactly where a hardware fault (e.g.
                # a motor bus voltage error) can kill the process -- without
                # this, that update is silently lost: --resume-from would
                # restart from the save made after the *previous* episode,
                # re-doing warmup satisfaction and this update's work.
                # completed_episodes uses +1 (not the actual, possibly-
                # rerecorded final count from _finish_recorded_episode below,
                # which needs go-home/reset to have already happened) so a
                # resume from *this* save starts the next new episode at an
                # id that doesn't collide with the one just flushed into the
                # buffer above -- the trade-off is that a rerecord after this
                # point makes this interim save overcount by one episode
                # versus the authoritative save at the end of this loop body,
                # which harmlessly just gets overwritten by that later save
                # in the common (non-crash) case.
                _save_online_rl_latest_state(recorded_episodes + 1)
                # Finish/discard (flush or cancel streaming encoders) *before*
                # go-home / reset teleop. Running reset while encoder threads
                # from the just-ended episode were still alive correlated with
                # Feetech left-arm bus dropouts on both → save and ← rerecord.
                was_rerecord = bool(events.get("rerecord_episode"))
                recorded_episodes = _finish_recorded_episode(recorded_episodes, episode_success)
                _save_online_rl_latest_state(recorded_episodes)
                _run_go_home_if_needed()
                _run_reset_loop_if_needed(recorded_episodes, was_rerecord=was_rerecord)
    finally:
        def _save_critical_phase_intervals() -> None:
            if critical_phase_tracker is None or len(critical_phase_tracker) == 0:
                return
            import json

            intervals = critical_phase_tracker.get_intervals()
            logging.info("Critical phase labeling: %d intervals recorded.", len(intervals))
            for ep_idx, start, end, outcome in intervals:
                outcome_str = f" [{outcome}]" if outcome else ""
                logging.info("  Episode %d: frames %d-%d (%d frames)%s", ep_idx, start, end, end - start, outcome_str)
            if dataset is None:
                return
            intervals_file = dataset.root / "critical_phase_intervals.json"
            with open(intervals_file, "w") as f:
                json.dump(
                    [
                        {"episode_index": ep, "start_frame": s, "end_frame": e, "outcome": o}
                        for ep, s, e, o in intervals
                    ],
                    f,
                    indent=2,
                )
            logging.info("Critical phase intervals saved to %s", intervals_file)

        def _finalize_dataset() -> None:
            if not dataset:
                return
            dataset.finalize()
            logging.info(
                "To inspect the recorded dataset, run:\n  lerobot-dataset-report --dataset %s",
                dataset.repo_id,
            )

        def _shutdown_policy_sync() -> None:
            if policy_sync_executor is not None:
                policy_sync_executor.shutdown()

        def _close_online_trainer() -> None:
            if online_trainer is not None:
                online_trainer.close()

        def _disconnect_devices() -> None:
            if robot.is_connected:
                robot.disconnect()
            if teleop and teleop.is_connected:
                teleop.disconnect()

        def _stop_listener() -> None:
            if listener and hasattr(listener, "stop"):
                listener.stop()

        def _push_dataset_to_hub() -> None:
            if not cfg.dataset.push_to_hub:
                return
            if dataset is not None:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
                return
            logging.warning(
                "`dataset.push_to_hub=true` was requested, but dataset was not initialized due to an earlier error."
            )

        log_say("Stop recording", cfg.play_sounds, blocking=True)
        _finalize_dataset()
        _save_critical_phase_intervals()
        _shutdown_policy_sync()
        _close_online_trainer()
        _disconnect_devices()
        _stop_listener()
        _push_dataset_to_hub()
        log_say("Exiting", cfg.play_sounds)
    return dataset

def main():
    register_third_party_plugins()
    record()

if __name__ == "__main__":
    main()
