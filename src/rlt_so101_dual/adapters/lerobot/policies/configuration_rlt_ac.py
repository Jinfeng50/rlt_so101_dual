from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from rlt_so101_dual.core import shape_contract as sc

_DEFAULT_CAMERA_KEYS: list[str] = list(sc.CAMERA_KEYS)


@dataclass
class ChunkACPolicyConfig(PreTrainedConfig):
    """Config for the chunk-level TD3+BC actor-critic policy on top of RL Token.

    Train via lerobot-train CLI with a ChunkTransitionDataset; deploy via
    make_policy + lerobot-record. forward(batch) returns a single scalar TD3+BC
    loss combining critic MSE + (every actor_update_interval steps) the actor
    Q-maximization + BC reg. Target critic is soft-updated with tau after every
    critic step. UTD ratio is achieved by setting outer --steps=outer*utd
    (each forward() == one critic update; actor update gated by counter).
    """

    # --- VLA + RL Token backbones (frozen, not serialized) ---
    vla_pretrained_path: str = "lerobot/pi05_base"
    vla_revision: str | None = None
    vla_dtype: str = "bfloat16"
    rl_token_pretrained_path: str = ""

    # --- RL Token arch (must match the loaded RLTokenPolicy ckpt) ---
    rl_token_dim: int = sc.RL_TOKEN_DIM
    rl_token_num_rl_tokens: int = 1
    token_pool_size: int = 0
    image_only: bool = False
    active_camera_indices: list[int] | None = None
    num_per_camera: int = 0

    # --- Actor ---
    actor_hidden_dim: int = 512
    actor_num_layers: int = 3
    actor_fixed_std: float = 0.05
    actor_ref_dropout_p: float = 0.5
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = False
    # Paper path: absolute mu = f(x, ref). residual_to_ref=True is an SO101
    # safety A/B (mu = ref + delta, zero-init last layer ≈ VLA at init).
    actor_residual_to_ref: bool = False
    # Optional |mu - ref| clamp in normalised space. None = no clamp (paper).
    actor_action_clip_delta: float | None = None

    # --- Critic + target ---
    critic_hidden_dim: int = 512
    critic_num_layers: int = 3
    critic_activation: str = "relu"
    # LayerNorm is from RLPD, not the RLT paper. Default off for paper alignment.
    critic_layer_norm: bool = False
    critic_residual: bool = False

    # --- TD3+BC hyperparams ---
    gamma: float = 0.99
    beta: float = 0.3
    # Action dims (0-based, within action_dim) the actor is pinned to the VLA
    # reference on: mu == ref there, so the joint leaves both the BC term and
    # the actor's control. Empty by default -- this changes the objective.
    pin_action_dims_to_ref: tuple[int, ...] = ()
    tau: float = 0.005
    utd_ratio: int = 5
    actor_update_interval: int = 2
    target_q_clip: float = 100.0
    # TD3 target policy smoothing. The paper says its critic follows TD3, and
    # smoothing is one of TD3's three core ingredients, but the reference
    # implementation leaves it out -- so this defaults off to keep the
    # baseline unchanged rather than silently altering it. TD3's own values
    # are 0.2 / 0.5. Worth an A/B once a baseline exists: it is training-only,
    # so it cannot affect what the robot executes.
    target_policy_noise: float = 0.0
    target_noise_clip: float = 0.5

    # --- Shapes ---
    chunk_length: int = sc.CHUNK_LENGTH
    action_dim: int = sc.ACTION_DIM
    proprio_dim: int = sc.PROPRIO_DIM

    # --- Deploy ---
    # Match bare π0.5 deploy (n_action_steps = VLA_HORIZON = 50). A shorter
    # open-loop window (e.g. 25) re-plans more often and was measured to feel
    # less stable than the first Stage-B / SFT eval on this cell.
    chunk_exec_steps: int = sc.VLA_HORIZON
    # How many RL control steps to cover with ONE frozen-VLA forward.
    # Must be a positive multiple of chunk_length and <= VLA horizon (50).
    # 10 = legacy (re-run π0.5 every chunk). 30–40 ≈ 3–4× fewer VLA calls on
    # 16GB cards; each window still refreshes proprio + actor and keeps C=10
    # transitions for online RL. Visual prefix is held from the VLA call.
    vla_reuse_horizon: int = 10
    phase_mode: str = "always_rl"
    deterministic: bool = True

    # --- Observation mapping (for deploy preprocessor) ---
    camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERA_KEYS))

    # --- Normalization MATCHES PI05 (deploy parity) ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # --- pi05 proxy fields ---
    max_state_dim: int = sc.PI05_MAX_DIM
    max_action_dim: int = sc.PI05_MAX_DIM
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    tokenizer_path: str | None = None

    @classmethod
    def ensure_registered(cls) -> None:
        import lerobot.policies  # noqa: F401

        PreTrainedConfig._choice_registry["rlt_ac"] = cls

    def __post_init__(self) -> None:
        self.ensure_registered()
        super().__post_init__()
        if self.phase_mode not in ("always_rl", "always_vla", "manual"):
            raise ValueError(
                f"phase_mode must be 'always_rl', 'always_vla', or 'manual', got {self.phase_mode!r}"
            )
        # Check here, not in the policy: the policy validates only after loading
        # the RL-token checkpoint, so a typo would surface minutes into a run.
        bad = [d for d in (self.pin_action_dims_to_ref or ()) if not 0 <= d < self.action_dim]
        if bad:
            raise ValueError(
                f"pin_action_dims_to_ref {bad} out of range for action_dim={self.action_dim}; "
                "indices are 0-based per joint, not per flattened chunk step."
            )
        if self.pin_action_dims_to_ref and not getattr(self, "vla_ref", True):
            raise ValueError(
                "pin_action_dims_to_ref with vla_ref=False is not meaningful: the deploy "
                "path zeroes ref_flat before calling the actor, so a pinned joint would be "
                "held at 0 instead of the VLA reference."
            )
        if self.vla_reuse_horizon < self.chunk_length:
            raise ValueError(
                f"vla_reuse_horizon={self.vla_reuse_horizon} must be >= chunk_length="
                f"{self.chunk_length}"
            )
        if self.vla_reuse_horizon % self.chunk_length != 0:
            raise ValueError(
                f"vla_reuse_horizon={self.vla_reuse_horizon} must be a multiple of "
                f"chunk_length={self.chunk_length}"
            )

    @property
    def type(self) -> str:
        return "rlt_ac"

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.proprio_dim,)
            )
        for cam_key in self.camera_keys:
            img_key = f"{OBS_IMAGES}.{cam_key}"
            if img_key not in self.input_features:
                self.input_features[img_key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, *self.image_resolution)
                )
        if not self.output_features:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.action_dim,)
            )

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=3e-4, weight_decay=0.0, grad_clip_norm=1.0)

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
