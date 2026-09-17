from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rlt_so101_dual.core import shape_contract


@dataclass
class RLTokenConfig:
    token_dim: int = shape_contract.RL_TOKEN_DIM
    nhead: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    ff_dim: int | None = None  # defaults to 4 * token_dim if None
    num_rl_tokens: int = 1  # number of RL tokens (>1 reduces compression ratio)

    def __post_init__(self):
        if self.ff_dim is None:
            self.ff_dim = 4 * self.token_dim


@dataclass
class ActorConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    fixed_std: float = 0.05
    lr: float = 3e-4
    ref_dropout_p: float = 0.5
    activation: str = "relu"
    layer_norm: bool = False
    residual: bool = False
    residual_to_ref: bool = False


@dataclass
class CriticConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    lr: float = 3e-4
    activation: str = "relu"
    layer_norm: bool = False
    residual: bool = False


@dataclass
class DemoAdaptConfig:
    steps: int = 5000
    batch_size: int = 32
    lr: float = 1e-4
    vla_ft_weight: float = 1.0
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    gamma: float = 0.99
    beta: float = 1.0
    tau: float = 0.005
    batch_size: int = 256
    utd_ratio: int = 5
    actor_update_interval: int = 2


@dataclass
class ReplayConfig:
    capacity: int = 200_000


@dataclass
class CollectorConfig:
    warmup_steps: int = 5000
    total_env_steps: int = 100_000
    chunk_subsample_stride: int = 2


@dataclass
class OfflineRLConfig:
    """Configuration for offline RL training on demo data."""

    num_gradient_steps: int = 100_000
    eval_every: int = 5000
    save_every: int = 10000
    log_every: int = 100
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    frame_stride: int = 2
    cache_dir: str | None = None
    demo_adapt_checkpoint: str | None = None


@dataclass
class RLTConfig:
    seed: int = 0
    control_hz: int = shape_contract.FPS
    action_dim: int = shape_contract.ACTION_DIM
    proprio_dim: int = shape_contract.PROPRIO_DIM
    vla_horizon: int = shape_contract.VLA_HORIZON
    chunk_length: int = shape_contract.CHUNK_LENGTH
    cameras: list[str] = field(default_factory=lambda: list(shape_contract.CAMERA_KEYS))

    rl_token: RLTokenConfig = field(default_factory=RLTokenConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    demo_adaptation: DemoAdaptConfig = field(default_factory=DemoAdaptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    offline_rl: OfflineRLConfig = field(default_factory=OfflineRLConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RLTConfig:
        """Load config from a YAML file, using defaults for missing fields."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        sub_configs = {}
        for key, subcls in [
            ("rl_token", RLTokenConfig),
            ("actor", ActorConfig),
            ("critic", CriticConfig),
            ("demo_adaptation", DemoAdaptConfig),
            ("training", TrainingConfig),
            ("replay", ReplayConfig),
            ("collector", CollectorConfig),
            ("offline_rl", OfflineRLConfig),
        ]:
            if key in raw:
                sub_configs[key] = subcls(**raw.pop(key))

        return cls(**raw, **sub_configs)

    @classmethod
    def default(cls) -> RLTConfig:
        """The shape contract's values.

        This used to read a shipped base.yaml, which was a second place the
        dimensions lived and which still carried the upstream bimanual 14-DoF
        numbers. There is now one source: shape_contract, via the field
        defaults above.
        """
        return cls()
