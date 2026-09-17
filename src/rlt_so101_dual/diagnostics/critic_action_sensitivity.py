"""Measure whether the critic still depends on the action, or has collapsed to V(s).

Why this matters: the actor is trained with `-Q(s, mu) + beta*||mu - ref||^2`.
If Q barely changes when the action changes, the first term contributes almost
no gradient and only the BC regulariser is left -- the actor converges to the
VLA reference and online RL silently does nothing. Sparse rewards plus a large
beta make this failure mode easy to hit, and it looks exactly like "RL just
isn't helping yet" from the outside.

Loads only the actor and critic out of an online_state.pt (or a transition
cache plus a saved policy dir). pi0.5 is never instantiated, so this runs on
CPU in a few seconds.

Usage:
    python diagnostics/critic_action_sensitivity.py \
        --online-state outputs/online_rl/<run>/latest_online_state.pt

    # compare several checkpoints over the course of a run
    python diagnostics/critic_action_sensitivity.py \
        --online-state outputs/.../step_000020/online_state.pt \
                       outputs/.../step_000060/online_state.pt \
                       outputs/.../latest_online_state.pt \
        --plot outputs/critic_health.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rlt_so101_dual.core import shape_contract as sc
from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.critic import TwinCritic
from rlt_so101_dual.core.utils import infer_actor_architecture

# Below this the BC term produces an order of magnitude more gradient than Q,
# so the actor is effectively doing behaviour cloning. A flag, not a proof.
GRAD_BALANCE_FLOOR = 0.1


def _infer_critic_kwargs(sd: dict[str, torch.Tensor]) -> dict:
    """Mirror infer_actor_architecture for TwinCritic (prefix q1./q2.)."""
    q1 = {k[len("q1.") :]: v for k, v in sd.items() if k.startswith("q1.")}
    kwargs = infer_actor_architecture(q1)
    for dead in ("fixed_std", "ref_dropout_p"):
        kwargs.pop(dead, None)
    return kwargs


def _dims(actor_sd: dict[str, torch.Tensor]) -> tuple[int, int]:
    """Recover (state_dim, chunk_dim) from the actor's first and last layers."""
    first = actor_sd.get("net.input_proj.weight")
    if first is None:
        first = actor_sd[sorted(k for k in actor_sd if k.endswith(".weight") and actor_sd[k].ndim == 2)[0]]
    out = actor_sd.get("net.output_proj.weight")
    if out is None:
        out = actor_sd[sorted(k for k in actor_sd if k.endswith(".weight") and actor_sd[k].ndim == 2)[-1]]
    chunk_dim = out.shape[0]
    state_dim = first.shape[1] - chunk_dim
    return state_dim, chunk_dim


def _actor_runtime_config(path: Path) -> tuple[dict | None, Path | None]:
    """Read the fields the weights cannot reveal, or (None, None) if absent.

    `residual_to_ref` decides whether mu is `delta` or `ref + delta`. It is not
    a parameter and leaves no trace in the state_dict, so a guess here makes
    every number below describe a different function than the one that was
    trained: with the wrong value this tool reported |mu - ref| = 1.32 and
    "BC-DOMINATED" for a run whose real values are 0.056 and "healthy".
    `activation` is invisible for the same reason. Read them from the run's own
    config.json and refuse to guess.
    """
    # load_run is documented to take a Path but torch.load accepts str, so
    # callers have always been able to pass one.
    path = Path(path)
    run_dir = path.parent.parent if path.parent.name.startswith("step_") else path.parent
    candidates = [path.parent / "config.json", *sorted(run_dir.glob("step_*/config.json"), reverse=True)]
    for cfg in candidates:
        if not cfg.is_file():
            continue
        try:
            data = json.loads(cfg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "actor_residual_to_ref" in data:
            return data, cfg
    return None, None


# Fields that leave no trace in the weights. Guessing any of them rebuilds a
# different function than the one that was trained, silently.
_UNRECOVERABLE = (
    ("residual_to_ref", "actor_residual_to_ref", "--actor-residual-to-ref"),
    ("actor_activation", "actor_activation", "--actor-activation"),
    ("critic_activation", "critic_activation", "--critic-activation"),
)

# Added after run4/run5, so a config from those runs simply lacks the key -- and
# absent then unambiguously means "no pinning", the feature did not exist. Demand
# it only when there is no config at all; the fields above have always been
# written, so a config missing *those* is a config from something else.
_UNRECOVERABLE_WITH_DEFAULT = (
    ("pin_action_dims_to_ref", "pin_action_dims_to_ref", "--pin-action-dims-to-ref", ()),
)


def load_run(
    path: Path,
    device: str,
    residual_to_ref: bool | None = None,
    actor_activation: str | None = None,
    critic_activation: str | None = None,
    pin_action_dims_to_ref: tuple[int, ...] | None = None,
) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    for key in ("actor_state_dict", "critic_state_dict", "replay_buffer"):
        if key not in state:
            raise SystemExit(
                f"{path} has no '{key}'. Expected an online_state.pt written by "
                "rlt-so101-dual-online-train (latest_online_state.pt or "
                "step_NNNNNN/online_state.pt)."
            )
    actor_sd, critic_sd = state["actor_state_dict"], state["critic_state_dict"]
    state_dim, chunk_dim = _dims(actor_sd)

    # infer_actor_architecture recovers only what the weights determine
    # (hidden_dim, num_layers, layer_norm, residual); the rest comes from the
    # run's config or an explicit override -- never from a default.
    kwargs = infer_actor_architecture(actor_sd)
    cfg, cfg_path = _actor_runtime_config(path)

    # An explicit CLI value overrides only the field it names; everything else
    # still comes from the config. Passing --actor-residual-to-ref used to skip
    # the config entirely, silently falling back to ReLU.
    explicit = {"residual_to_ref": residual_to_ref,
                "actor_activation": actor_activation,
                "critic_activation": critic_activation,
                "pin_action_dims_to_ref": pin_action_dims_to_ref}
    resolved: dict = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for name, cfg_key, flag in _UNRECOVERABLE:
        if explicit[name] is not None:
            resolved[name], sources[name] = explicit[name], flag
        elif cfg is not None and cfg_key in cfg:
            resolved[name], sources[name] = cfg[cfg_key], str(cfg_path)
        else:
            missing.append(flag)
    for name, cfg_key, flag, default in _UNRECOVERABLE_WITH_DEFAULT:
        if explicit[name] is not None:
            resolved[name], sources[name] = explicit[name], flag
        elif cfg is not None:
            resolved[name], sources[name] = cfg.get(cfg_key, default), str(cfg_path)
        else:
            missing.append(flag)
    if missing:
        raise SystemExit(
            f"Cannot determine {', '.join(m.lstrip('-') for m in missing)} for {path}.\n"
            "These leave no trace in the weights, and a wrong value makes every metric this "
            "tool prints describe a different function than the one that was trained.\n"
            "Either point at a snapshot whose run carries config.json, or pass "
            f"{' '.join(missing)} explicitly."
        )

    residual_to_ref = bool(resolved["residual_to_ref"])
    kwargs["activation"] = resolved["actor_activation"]
    if cfg is not None:
        for cfg_key, kwarg in (("actor_fixed_std", "fixed_std"),
                               ("actor_ref_dropout_p", "ref_dropout_p")):
            if cfg_key in cfg:
                kwargs[kwarg] = cfg[cfg_key]
    clip_delta = None if cfg is None else cfg.get("actor_action_clip_delta")

    actor = ChunkActor(state_dim, chunk_dim, residual_to_ref=residual_to_ref, **kwargs)
    # Pinned joints leave the weights untouched, so this is another field the
    # state_dict cannot reveal. Without it the tool reports a |mu - ref| and a
    # BC gradient for a joint the run had frozen -- and wrist_roll alone carried
    # 52% of the unpinned penalty, so the error is not small.
    pinned = tuple(resolved["pin_action_dims_to_ref"] or ())
    if pinned:
        action_dim = chunk_dim // kwargs.get("chunk_length", 0) if kwargs.get("chunk_length") \
            else sc.ACTION_DIM
        joint = torch.zeros(action_dim)
        joint[list(pinned)] = 1.0
        actor.set_pin_mask(joint.repeat(chunk_dim // action_dim))
    actor.load_state_dict(actor_sd, strict=False)
    critic_kwargs = _infer_critic_kwargs(critic_sd)
    critic_kwargs["activation"] = resolved["critic_activation"]
    critic = TwinCritic(state_dim, chunk_dim, **critic_kwargs)
    critic.load_state_dict(critic_sd)
    actor.eval().to(device)
    critic.eval().to(device)

    buf = state["replay_buffer"]
    if not buf:
        raise SystemExit(f"{path} has an empty replay buffer; nothing to evaluate.")
    return {
        "path": path,
        "actor": actor,
        "critic": critic,
        "buffer": buf,
        "state_dim": state_dim,
        "chunk_dim": chunk_dim,
        "episodes": state.get("recorded_episodes", "?"),
        "residual_to_ref": residual_to_ref,
        "config_source": sources["residual_to_ref"],
        "actor_activation": resolved["actor_activation"],
        "critic_activation": resolved["critic_activation"],
        "clip_delta": clip_delta,
        "pinned_dims": pinned,
    }


def analyse(run: dict, n: int, n_actions: int, device: str, seed: int, beta: float) -> dict:
    clip_delta = run.get("clip_delta")
    g = torch.Generator(device="cpu").manual_seed(seed)
    buf = run["buffer"]
    idx = torch.randperm(len(buf), generator=g)[:n].tolist()
    batch = [buf[i] for i in idx]

    s = torch.stack([t.state_vec for t in batch]).to(device).float()
    a = torch.stack([t.exec_chunk.reshape(-1) for t in batch]).to(device).float()
    ref = torch.stack([t.ref_chunk.reshape(-1) for t in batch]).to(device).float()
    critic, actor = run["critic"], run["actor"]

    # --- The headline number ------------------------------------------------
    # The actor minimises  -Q(s, mu) + beta*||mu - ref||^2, so what decides
    # whether RL does anything is which of the two terms produces the larger
    # gradient at mu. Measure them directly rather than inferring from Q
    # statistics: this is the quantity the optimiser actually sees.
    mu, _ = actor(s, ref, training=False)

    # The headline ratio is taken on the ACTOR'S PARAMETERS at training=True --
    # what the optimiser actually receives. Norms in action space cannot stand in
    # for it: a pinned joint has zero derivative to every parameter, but dQ/da is
    # still nonzero there, so an action-space ratio counts that joint in the
    # numerator while mu == ref keeps it out of the denominator. wrist_roll is
    # both the pinned joint and the one carrying the most (repositioning-noise)
    # variation, so the inflation is not small. Going through the parameters also
    # picks up reference dropout and the network Jacobian for free.
    params = [prm for prm in actor.parameters() if prm.requires_grad]
    torch.manual_seed(seed)
    mu_train, _ = actor(s, ref, training=True)
    q_loss = -critic.min_q(s, mu_train).mean()
    bc_loss = beta * ((mu_train - ref) ** 2).sum(dim=-1).mean()
    g_q = torch.autograd.grad(q_loss, params, retain_graph=True, allow_unused=True)
    g_bc = torch.autograd.grad(bc_loss, params, allow_unused=True)
    _n = lambda gs: torch.sqrt(sum((g ** 2).sum() for g in gs if g is not None)).item()
    param_grad_q, param_grad_bc = _n(g_q), _n(g_bc)

    grad_bc_train = (2.0 * beta * (mu_train.detach() - ref)).norm(dim=-1)
    mu_q = mu.detach().clone().requires_grad_(True)
    critic.min_q(s, mu_q).sum().backward()
    grad_q = mu_q.grad.norm(dim=-1)
    # d/dmu of beta*sum((mu-ref)^2) = 2*beta*(mu-ref)
    grad_bc = (2.0 * beta * (mu.detach() - ref)).norm(dim=-1)
    # At zero-init mu == ref exactly, so the BC gradient is 0 and the ratio is
    # undefined rather than infinitely good. Report it as such.
    at_init = bool(param_grad_bc < 1e-12)
    balance = float("nan") if at_init else param_grad_q / param_grad_bc

    with torch.no_grad():
        q = critic.min_q(s, a).squeeze(-1)

        # --- Action sensitivity, reported two ways --------------------------
        # Absolute mean|dQ| is directly comparable to the reward scale (sparse
        # +1, so a well-trained Q lives roughly in [0, 1]).
        # The ratio compares it against perturbing the *state* by the same
        # relative amount, which removes the state_dim >> action_dim bias that
        # makes any wide-state critic look action-insensitive.
        s_scale = s.std()
        a_scale = a.std().clamp_min(1e-6)
        sensitivity, sensitivity_ratio = {}, {}
        for eps in (0.01, 0.05, 0.1, 0.3):
            da = torch.randn(a.shape, generator=g).to(device) * eps * a_scale
            ds = torch.randn(s.shape, generator=g).to(device) * eps * s_scale
            # No clamp to [-1, 1]. Normalised actions legitimately leave that
            # box -- wrist_roll's q99-q01 is 0.558 deg, so its normalised ref
            # runs to 6.0 -- and clamping turns a small local perturbation into
            # a jump from 6.0 to 1.0, measuring a cross-distribution change
            # instead of action sensitivity.
            dq_a = (critic.min_q(s, a + da).squeeze(-1) - q).abs().mean()
            dq_s = (critic.min_q(s + ds, a).squeeze(-1) - q).abs().mean()
            sensitivity[eps] = dq_a.item()
            sensitivity_ratio[eps] = (dq_a / dq_s.clamp_min(1e-12)).item()

        # --- Can the critic rank alternatives at a fixed state? -------------
        alt = torch.stack(
            [
                critic.min_q(
                    s, a + torch.randn(a.shape, generator=g).to(device) * 0.1 * a_scale
                ).squeeze(-1)
                for _ in range(n_actions)
            ]
        )
        spread = (alt.max(dim=0).values - alt.min(dim=0).values).mean().item()

        # --- Does the actor find anything the critic prefers to the VLA? ----
        # Two different questions, kept apart:
        #   objective: what actor_loss actually maximises -- Q at the raw mu.
        #   executed : what reaches the robot -- ref + clamp(mu - ref, +-delta).
        # Reporting only one of them, and clamping mu to [-1, 1] on the way,
        # queried a support domain that matches neither.
        q_ref = critic.min_q(s, ref).squeeze(-1)
        adv = critic.min_q(s, mu.detach()).squeeze(-1) - q_ref
        if clip_delta is None:
            adv_exec = None
        else:
            executed = ref + (mu.detach() - ref).clamp(-clip_delta, clip_delta)
            adv_exec = critic.min_q(s, executed).squeeze(-1) - q_ref

    return {
        "n": len(batch),
        "episodes": run["episodes"],
        "residual_to_ref": run["residual_to_ref"],
        "config_source": run["config_source"],
        "actor_activation": run["actor_activation"],
        "critic_activation": run["critic_activation"],
        "pinned_dims": run.get("pinned_dims", ()),
        "q_mean": q.mean().item(),
        "q_std": q.std().item(),
        "q_min": q.min().item(),
        "q_max": q.max().item(),
        "sensitivity": sensitivity,
        "sensitivity_ratio": sensitivity_ratio,
        "spread": spread,
        "n_actions": n_actions,
        "grad_balance": balance,
        "at_init": at_init,
        "grad_q": grad_q.median().item(),
        "grad_bc": grad_bc.median().item(),
        "param_grad_q": param_grad_q,
        "param_grad_bc": param_grad_bc,
        "grad_bc_train": grad_bc_train.median().item(),
        "actor_delta_train": (mu_train.detach() - ref).abs().mean().item(),
        "adv_mean": adv.mean().item(),
        "adv_frac_positive": (adv > 0).float().mean().item(),
        "clip_delta": clip_delta,
        "adv_exec_mean": None if adv_exec is None else adv_exec.mean().item(),
        "adv_exec_frac_positive":
            None if adv_exec is None else (adv_exec > 0).float().mean().item(),
        "actor_delta_clipped":
            None if clip_delta is None
            else (mu.detach() - ref).clamp(-clip_delta, clip_delta).abs().mean().item(),
        "actor_delta": (mu.detach() - ref).abs().mean().item(),
    }


def _sparkline(values: list[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    finite = [v for v in values if v == v]
    if not finite:
        return "n/a"
    lo, hi = min(finite), max(finite)
    span = hi - lo or 1.0
    return "".join(blocks[min(7, int((v - lo) / span * 7))] if v == v else "?" for v in values)


def report(res: dict, label: str, beta: float) -> None:
    print(f"\n=== {label} ===")
    print(f"  transitions sampled : {res['n']}  (after {res['episodes']} episodes)")
    print(f"  actor mu            : {'ref + delta' if res['residual_to_ref'] else 'delta'}"
          f"   (residual_to_ref={res['residual_to_ref']} from {res['config_source']})")
    print(f"  activations         : actor {res['actor_activation']}  critic {res['critic_activation']}")
    if res.get("pinned_dims"):
        print(f"  pinned to ref       : action dims {list(res['pinned_dims'])} "
              f"(mu == ref there, so they contribute nothing below)")
    print(f"  Q over states       : mean {res['q_mean']:+.4f}  std {res['q_std']:.4f}  "
          f"range [{res['q_min']:+.4f}, {res['q_max']:+.4f}]")

    print(f"\n  [1] Gradient balance on the ACTOR'S PARAMETERS   (beta = {beta})")
    print(f"      ||d(-Q)/d(theta)||        : {res['param_grad_q']:.6f}")
    print(f"      ||d(beta*BC)/d(theta)||   : {res['param_grad_bc']:.6f}")
    print("      (training=True: includes ref dropout, the pin Jacobian and the net's)")
    print(f"      for reference, in action space at inference-mode mu:")
    print(f"        ||dQ/da||               : {res['grad_q']:.6f}   (median over states)")
    print(f"        ||d(beta*BC)/da||       : {res['grad_bc']:.6f}   (ref present)")
    print(f"        ||d(beta*BC)/da||       : {res['grad_bc_train']:.6f}   (ref dropout on)")
    if res["at_init"]:
        print("      ratio Q : BC              : n/a  (mu == ref exactly, so the BC gradient "
              "is 0 -- the actor has not been updated yet)")
    else:
        print(f"      ratio Q : BC              : {res['grad_balance']:.4f}"
              f"   <-- below {GRAD_BALANCE_FLOOR} means BC dominates and RL barely moves the actor")

    print("\n  [2] Action sensitivity of Q")
    print("      perturb   mean|dQ|    vs same-size state perturbation")
    for eps in res["sensitivity"]:
        print(f"      {eps:<8}  {res['sensitivity'][eps]:.6f}    {res['sensitivity_ratio'][eps]:.4f}")
    print(f"      trend   : {_sparkline(list(res['sensitivity'].values()))}"
          f"   (should rise with perturbation size)")
    print(f"      Q spread over {res['n_actions']} alternative actions at the same state: {res['spread']:.6f}")

    print(f"\n  [3] Actor advantage, training objective  Q(s,mu) - Q(s,ref) : "
          f"{res['adv_mean']:+.6f}   positive on {res['adv_frac_positive']*100:.1f}% of states")
    print(f"      mean |mu - ref| per dim                                  : {res['actor_delta']:.6f}")
    if res.get("adv_exec_mean") is not None:
        print(f"      as executed (ref +- {res['clip_delta']})  Q(s,a_exec) - Q(s,ref) : "
              f"{res['adv_exec_mean']:+.6f}   positive on "
              f"{res['adv_exec_frac_positive']*100:.1f}% of states")
        print(f"      mean |a_exec - ref| per dim                              : "
              f"{res['actor_delta_clipped']:.6f}"
              f"   ({100*res['actor_delta_clipped']/max(res['clip_delta'],1e-12):.0f}% of the clip)")

    balance = res["grad_balance"]
    print()
    if res["q_std"] < 1e-6 and res["spread"] < 1e-6:
        print("  VERDICT: Q is constant -- the critic has not learned anything yet. "
              "Expected before/just after warmup; re-check after critic-only.")
    elif res["spread"] < 1e-6:
        print("  VERDICT: DEGENERATE. Q varies across states but is identical for every action "
              "at a given state -- the critic is a value function V(s), not a Q function. The "
              "-Q term can never prefer one action over another.")
        print("           Most likely the critic never saw actions that differ from the "
              "reference: check that exec_chunk != ref_chunk in the training data, and raise "
              "gamma so the terminal reward actually reaches the start of the critical phase.")
        print("           An offline cache built before a3e3ebb stored ref_chunk as both, so "
              "any such cache produces exactly this. Rebuild it.")
    elif res["at_init"]:
        print("  VERDICT: the critic separates actions, but the actor is still at its zero-init "
              "(mu == ref exactly). Normal before the first actor update; re-run after a few "
              "actor steps to get a meaningful gradient balance.")
    elif balance < GRAD_BALANCE_FLOOR:
        print(f"  VERDICT: BC-DOMINATED. The Q gradient is {balance:.3f}x the BC gradient, so the "
              f"actor is essentially being trained to copy the VLA and online RL will look like "
              f"it is doing nothing.")
        print("           Either the critic has collapsed toward V(s) (check [2]: if mean|dQ| is "
              "~0 and the trend is flat, it has) or beta is simply too large for the current Q "
              "scale. Lower beta first -- it is the cheaper experiment.")
        print("           If [2] is also flat, raise gamma so the terminal reward actually "
              "reaches the start of the critical phase, and check that exec_chunk really "
              "differs from ref_chunk in the training data.")
    else:
        print(f"  VERDICT: healthy. Q gradient is {balance:.2f}x the BC gradient, and the critic "
              f"separates alternative actions at the same state.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--online-state", nargs="+", required=True, type=Path,
                   help="One or more online_state.pt snapshots, in training order")
    p.add_argument("--n", type=int, default=512, help="Transitions to sample per snapshot")
    p.add_argument("--n-actions", type=int, default=16,
                   help="Alternative actions per state for the variance decomposition")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.3,
                   help="BC regularisation weight the run was trained with; only affects "
                        "the gradient-balance metric")
    p.add_argument("--actor-activation", default=None,
                   help="Overrides the run config. Not recoverable from the weights.")
    p.add_argument("--critic-activation", default=None,
                   help="Overrides the run config. Not recoverable from the weights.")
    p.add_argument("--pin-action-dims-to-ref", default=None,
                   help="Comma-separated joint indices pinned to the reference, or "
                        "'none'. Overrides the run config. Not recoverable from the "
                        "weights, and it changes what the actor computes.")
    p.add_argument("--actor-residual-to-ref", choices=("auto", "true", "false"), default="auto",
                   help="How mu is formed: 'delta' (false) or 'ref + delta' (true). Not "
                        "recoverable from the weights, so 'auto' reads it from the run's "
                        "config.json and fails if it is not there. Override only for a "
                        "snapshot separated from its config.")
    p.add_argument("--plot", type=Path, default=None,
                   help="Write a PNG tracking the metrics across the given snapshots")
    args = p.parse_args()

    print(f"contract: state_dim {sc.state_vec_dim()}  chunk_dim {sc.chunk_flat_dim()}")
    results = []
    for path in args.online_state:
        residual_to_ref = None if args.actor_residual_to_ref == "auto" \
            else args.actor_residual_to_ref == "true"
        pinned_override = None
        if args.pin_action_dims_to_ref is not None:
            raw = args.pin_action_dims_to_ref.strip().lower()
            pinned_override = () if raw in ("", "none") else tuple(
                int(x) for x in raw.replace(" ", "").split(",") if x
            )
        run = load_run(path, args.device, residual_to_ref=residual_to_ref,
                       actor_activation=args.actor_activation,
                       critic_activation=args.critic_activation,
                       pin_action_dims_to_ref=pinned_override)
        if run["state_dim"] != sc.state_vec_dim() or run["chunk_dim"] != sc.chunk_flat_dim():
            print(f"  ! {path.name}: state_dim {run['state_dim']}, chunk_dim {run['chunk_dim']} "
                  f"-- does not match this project's contract")
        res = analyse(run, args.n, args.n_actions, args.device, args.seed, args.beta)
        report(res, path.parent.name or path.name, args.beta)
        results.append((path, res))

    if args.plot and len(results) >= 1:
        _write_plot(results, args.plot)


def _write_plot(results, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not installed; skipping {out}. "
              "Install it with: pip install matplotlib")
        return

    eps_counts = [r[1]["episodes"] for r in results]
    # Episode counts only make a usable axis if they actually differ and
    # increase; otherwise every snapshot lands on the same tick.
    if all(isinstance(e, int) for e in eps_counts) and len(set(eps_counts)) == len(eps_counts):
        xs, xlabel = eps_counts, "episodes"
    else:
        xs, xlabel = list(range(len(results))), "snapshot"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    balances = [r[1]["grad_balance"] for r in results]
    finite = [b for b in balances if b == b]
    axes[0].plot(xs, balances, marker="o", c="tab:blue")
    axes[0].axhline(GRAD_BALANCE_FLOOR, ls="--", c="k", lw=1, label="BC-dominated below")
    if finite:
        axes[0].set_yscale("log")
    for x, b in zip(xs, balances):
        if b != b:  # NaN: actor still at zero-init, BC gradient is exactly 0
            axes[0].annotate("at init", (x, GRAD_BALANCE_FLOOR), ha="center", va="bottom",
                             fontsize=8, color="gray", rotation=90)
    axes[0].set_title("Gradient balance  ||dQ|| / ||d(beta*BC)||")
    axes[0].set_xlabel(xlabel)
    axes[0].legend(fontsize=8)

    for eps in (0.01, 0.05, 0.1, 0.3):
        axes[1].plot(xs, [r[1]["sensitivity"][eps] for r in results], marker="o", label=f"eps={eps}")
    axes[1].set_title("Action sensitivity  mean|dQ|")
    axes[1].set_xlabel(xlabel)
    axes[1].legend(fontsize=8)

    axes[2].plot(xs, [r[1]["adv_mean"] for r in results], marker="o", c="tab:green")
    axes[2].axhline(0, ls="--", c="k", lw=1)
    axes[2].set_title("Actor advantage  Q(s,mu) - Q(s,ref)")
    axes[2].set_xlabel(xlabel)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
