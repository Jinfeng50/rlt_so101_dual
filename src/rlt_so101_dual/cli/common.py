from __future__ import annotations

import logging
import sys
from pathlib import Path


from rlt_so101_dual.core import shape_contract as sc


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CAMERAS = list(sc.CAMERA_KEYS)
DEFAULT_ACTION_DIM = sc.ACTION_DIM
DEFAULT_PROPRIO_DIM = sc.PROPRIO_DIM
DEFAULT_VLA_HORIZON = sc.VLA_HORIZON
DEFAULT_CHUNK_LENGTH = sc.CHUNK_LENGTH

_SHAPE_FIELDS = ("action_dim", "proprio_dim", "vla_horizon", "chunk_length", "cameras")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def configure_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def load_training_config(config_path: str | None):
    """Load an RLTConfig, letting the YAML win over the built-in defaults.

    This used to overwrite the five shape fields unconditionally right after
    parsing, which made `action_dim: 6` in a YAML file a no-op with no
    warning. Whatever ends up in effect is now logged every run, so a wrong
    number shows up in the first ten lines of output instead of as a shape
    error thirty minutes into training.
    """
    from rlt_so101_dual.core.config import RLTConfig

    config = RLTConfig.from_yaml(config_path) if config_path else RLTConfig()
    log = logging.getLogger(__name__)
    origin = f"{config_path}" if config_path else "built-in defaults"
    log.info("Effective shape config (from %s):", origin)
    for name in _SHAPE_FIELDS:
        log.info("  %-13s = %s", name, getattr(config, name))

    # A YAML is allowed to override the contract -- that is the point of the
    # fix above -- but disagreeing with it silently is how the bimanual
    # numbers survived. Say so.
    expected = {
        "action_dim": sc.ACTION_DIM,
        "proprio_dim": sc.PROPRIO_DIM,
        "vla_horizon": sc.VLA_HORIZON,
        "chunk_length": sc.CHUNK_LENGTH,
        "cameras": list(sc.CAMERA_KEYS),
    }
    for name, want in expected.items():
        got = getattr(config, name)
        if got != want:
            log.warning(
                "  %s = %s does NOT match the bimanual SO101 contract (%s). "
                "Intentional overrides are fine; an unintentional one produces "
                "networks of the wrong width with no later error.",
                name, got, want,
            )
    return config


def build_pi05_policy(
    config,
    model_path: str,
    task_instruction: str,
    device: str,
    token_pool_size: int,
    dtype: str,
    rl_token_checkpoint: str | None = None,
    vla_cache_dir: str | None = None,
    image_only: bool = False,
    active_cameras: list[str] | None = None,
    tokenizer_path: str | None = None,
):
    from rlt_so101_dual.adapters.lerobot.pi05_adapter import Pi05VLAAdapter
    from rlt_so101_dual.core.policy import RLTPolicy
    import torch

    vla = Pi05VLAAdapter(
        model_path=model_path,
        actual_action_dim=config.action_dim,
        actual_proprio_dim=config.proprio_dim,
        task_instruction=task_instruction,
        dtype=dtype,
        device=device,
        cache_dir=vla_cache_dir,
        token_pool_size=token_pool_size,
        image_only=image_only,
        active_cameras=active_cameras,
        tokenizer_path=tokenizer_path,
    )
    policy = RLTPolicy(config, vla).to(device)
    if rl_token_checkpoint is not None:
        checkpoint = torch.load(rl_token_checkpoint, map_location=device, weights_only=False)
        policy.rl_token.load_state_dict(checkpoint["rl_token_state_dict"], strict=False)
    return policy
