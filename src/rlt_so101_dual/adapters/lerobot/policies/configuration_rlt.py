from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from rlt_so101_dual.core import shape_contract as sc

# Single-arm SO101 joint keys (6 DOF). Sourced from the shape contract so
# the dataset, the policy config and the robot can never disagree.
_DEFAULT_PROPRIO_KEYS: list[str] = list(sc.JOINT_NAMES)

_DEFAULT_CAMERA_KEYS: list[str] = list(sc.CAMERA_KEYS)

_DEFAULT_ACTION_KEYS: list[str] = list(_DEFAULT_PROPRIO_KEYS)  # same as proprio


@dataclass
class RLTPretrainedConfig(PreTrainedConfig):
    """Configuration for RLT (RL Token) policy.

    RLT wraps a frozen VLA backbone (pi0.5) with a lightweight RL head
    (RL Token encoder + ChunkActor) trained via offline RL.
    """

    # --- VLA backbone ---
    vla_pretrained_path: str = "lerobot/pi05_base"
    vla_revision: str | None = None
    task_instruction: str = ""
    token_pool_size: int = 64
    image_only: bool = False  # if true, drop language tokens before RL token encode

    # --- Checkpoint paths (loaded during __init__) ---
    rl_token_ckpt_path: str = ""
    ac_ckpt_path: str = ""

    # --- RL Token encoder architecture ---
    rl_token_dim: int = 2048
    rl_token_nhead: int = 8
    rl_token_enc_layers: int = 3
    rl_token_dec_layers: int = 3
    rl_token_ff_dim: int = 4096
    rl_token_num_rl_tokens: int = 4

    # --- Actor architecture ---
    actor_hidden_dim: int = 256
    actor_num_layers: int = 3
    actor_fixed_std: float = 0.05
    actor_ref_dropout_p: float = 0.5
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = True

    # --- Deployment ---
    chunk_length: int = sc.CHUNK_LENGTH
    chunk_exec_steps: int = sc.VLA_HORIZON  # VLA phase: match bare π0.5 n_action_steps
    action_dim: int = sc.ACTION_DIM
    proprio_dim: int = sc.PROPRIO_DIM
    phase_mode: str = "always_rl"
    deterministic: bool = True

    # --- Observation mapping ---
    camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERA_KEYS))
    proprio_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_PROPRIO_KEYS))
    action_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_ACTION_KEYS))

    # --- Normalization (RLT does not normalize; VLA handles its own) ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    @classmethod
    def ensure_registered(cls) -> None:
        import lerobot.policies  # noqa: F401

        PreTrainedConfig._choice_registry["rlt"] = cls

    def __post_init__(self) -> None:
        self.ensure_registered()
        super().__post_init__()
        if self.phase_mode not in ("always_rl", "always_vla", "manual"):
            raise ValueError(
                f"phase_mode must be 'always_rl', 'always_vla', or 'manual', got '{self.phase_mode}'"
            )

    @property
    def type(self) -> str:
        return "rlt"

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.proprio_dim,),
            )
        for cam_key in self.camera_keys:
            img_key = f"{OBS_IMAGES}.{cam_key}"
            if img_key not in self.input_features:
                self.input_features[img_key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, 224, 224),
                )
        if not self.output_features:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.action_dim,),
            )

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=3e-4, weight_decay=0.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_length))

    @property
    def reward_delta_indices(self) -> None:
        return None
