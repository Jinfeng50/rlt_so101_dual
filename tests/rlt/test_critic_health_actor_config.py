"""critic-health must not guess how the actor forms mu.

`residual_to_ref` picks between `mu = delta` and `mu = ref + delta`. It is not
a parameter, so it leaves no trace in the state_dict. The tool used to hardcode
True, which made it report |mu - ref| = 1.25 and "BC-DOMINATED" for a run whose
real values were 0.05 and "healthy" -- the two configurations describe different
functions, so every downstream metric was wrong.
"""

from __future__ import annotations

import json

import pytest
import torch

from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.critic import TwinCritic
from rlt_so101_dual.diagnostics.critic_action_sensitivity import load_run

STATE_DIM, CHUNK_DIM, HIDDEN, LAYERS = 16, 8, 32, 1


def _write_run(tmp_path, *, residual_to_ref: bool, with_config: bool = True,
               activation: str = "relu"):
    """A minimal run directory: step_000002/{online_state.pt,config.json}."""
    torch.manual_seed(0)
    actor = ChunkActor(STATE_DIM, CHUNK_DIM, hidden_dim=HIDDEN, num_layers=LAYERS,
                       residual_to_ref=residual_to_ref, activation=activation)
    # A zero-init output layer would make mu == ref for the residual variant and
    # hide the very difference this test is about, so perturb it.
    with torch.no_grad():
        for param in actor.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    critic = TwinCritic(STATE_DIM, CHUNK_DIM, hidden_dim=HIDDEN, num_layers=LAYERS,
                        activation=activation)

    step = tmp_path / "step_000002"
    step.mkdir(parents=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "replay_buffer": ["one transition is enough; load_run only checks emptiness"],
            "recorded_episodes": 2,
        },
        step / "online_state.pt",
    )
    if with_config:
        (step / "config.json").write_text(json.dumps({
            "actor_hidden_dim": HIDDEN,
            "actor_num_layers": LAYERS,
            "actor_activation": activation,
            "actor_fixed_std": 0.05,
            "actor_ref_dropout_p": 0.5,
            "actor_residual_to_ref": residual_to_ref,
            "critic_activation": activation,
            "actor_action_clip_delta": 0.1,
            "pin_action_dims_to_ref": [],
        }))
    return step / "online_state.pt", actor


@pytest.mark.parametrize("residual_to_ref", [False, True])
def test_reads_residual_to_ref_from_config(tmp_path, residual_to_ref):
    path, saved = _write_run(tmp_path, residual_to_ref=residual_to_ref)
    run = load_run(path, device="cpu")

    assert run["residual_to_ref"] is residual_to_ref
    assert run["config_source"].endswith("config.json")

    # Same weights and inputs must reproduce the saved actor's mu exactly.
    state = torch.randn(4, STATE_DIM)
    ref = torch.randn(4, CHUNK_DIM)
    with torch.no_grad():
        expected, _ = saved(state, ref, training=False)
        got, _ = run["actor"](state, ref, training=False)
    torch.testing.assert_close(got, expected)


def test_the_two_variants_are_different_functions(tmp_path):
    """Guards the test above: if mu were the same either way it would prove nothing."""
    path, _ = _write_run(tmp_path, residual_to_ref=False)
    as_delta = load_run(path, device="cpu", residual_to_ref=False)["actor"]
    as_residual = load_run(path, device="cpu", residual_to_ref=True)["actor"]

    state = torch.randn(4, STATE_DIM)
    ref = torch.randn(4, CHUNK_DIM)
    with torch.no_grad():
        a, _ = as_delta(state, ref, training=False)
        b, _ = as_residual(state, ref, training=False)
    assert (a - b).abs().max() > 1e-3


def test_explicit_override_wins_over_config(tmp_path):
    path, _ = _write_run(tmp_path, residual_to_ref=False)
    run = load_run(path, device="cpu", residual_to_ref=True)
    assert run["residual_to_ref"] is True
    assert run["config_source"] == "--actor-residual-to-ref"


def test_missing_config_fails_loudly(tmp_path):
    """Every field that leaves no trace in the weights must be named."""
    path, _ = _write_run(tmp_path, residual_to_ref=False, with_config=False)
    with pytest.raises(SystemExit) as e:
        load_run(path, device="cpu")
    msg = str(e.value)
    for flag in ("--actor-residual-to-ref", "--actor-activation", "--critic-activation"):
        assert flag in msg


def test_explicit_residual_override_still_reads_activation_from_config(tmp_path):
    """An override names one field; it must not silently default the others.

    Passing --actor-residual-to-ref used to skip the config read entirely, so
    activation fell back to ReLU -- rebuilding a different function for any run
    trained with gelu/tanh.
    """
    path, _ = _write_run(tmp_path, residual_to_ref=False, activation="gelu")
    run = load_run(path, device="cpu", residual_to_ref=True)
    assert run["residual_to_ref"] is True
    assert run["config_source"] == "--actor-residual-to-ref"
    assert run["actor_activation"] == "gelu"
    assert run["critic_activation"] == "gelu"


def test_missing_config_with_all_overrides_succeeds(tmp_path):
    path, _ = _write_run(tmp_path, residual_to_ref=False, with_config=False)
    run = load_run(path, device="cpu", residual_to_ref=False,
                   actor_activation="relu", critic_activation="relu",
                   pin_action_dims_to_ref=())
    assert run["residual_to_ref"] is False
    assert run["clip_delta"] is None


def test_finds_config_for_snapshot_at_run_root(tmp_path):
    """latest_online_state.pt sits at the run root, not inside a step_ dir."""
    step_path, _ = _write_run(tmp_path, residual_to_ref=False)
    latest = tmp_path / "latest_online_state.pt"
    latest.write_bytes(step_path.read_bytes())
    assert load_run(latest, device="cpu")["residual_to_ref"] is False


def test_accepts_a_str_path(tmp_path):
    """torch.load takes str, so load_run has always tolerated one."""
    path, _ = _write_run(tmp_path, residual_to_ref=False)
    assert load_run(str(path), device="cpu")["residual_to_ref"] is False


def test_a_config_predating_the_pin_field_means_no_pinning(tmp_path):
    """run4/run5 configs have no pin key; absent must mean (), not a hard error."""
    import json

    path, _ = _write_run(tmp_path, residual_to_ref=False)
    cfg_path = path.parent / "config.json"
    data = json.loads(cfg_path.read_text())
    del data["pin_action_dims_to_ref"]
    cfg_path.write_text(json.dumps(data))

    run = load_run(path, device="cpu")
    assert run["pinned_dims"] == ()


def test_no_config_at_all_still_demands_the_pin_field(tmp_path):
    path, _ = _write_run(tmp_path, residual_to_ref=False, with_config=False)
    with pytest.raises(SystemExit, match="--pin-action-dims-to-ref"):
        load_run(path, device="cpu", residual_to_ref=False,
                 actor_activation="relu", critic_activation="relu")
