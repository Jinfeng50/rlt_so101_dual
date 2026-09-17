"""Standalone online-RL "brain": replay buffer + TD3+BC gradient updates for
`rlt_ac`, extracted out of `backend.record()`'s closures into their own
testable unit (see `backend.py`, which calls this unchanged).

This is a pure extraction: the control flow and math below are copied
verbatim from the closures that used to live inside `record()`
(`_run_online_rl_update`, `_warmup_satisfied`, `_save_online_rl_latest_state`)
-- only the closure variables became `self` attributes. Do not "clean up" the
ordering here without re-checking against RLT's TD3+BC correctness notes
(critic-before-actor step order, the tau=0/actor_update_interval=10**9
freeze-and-restore trick, warmup anchored to when it was actually satisfied
rather than a fixed offset) -- see the comments inline and on
`OnlineRLConfig` in `backend.py`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from rlt_so101_dual.adapters.lerobot.online_collector import RLTOnlineCollector
from rlt_so101_dual.core import shape_contract as sc
from rlt_so101_dual.core.replay_buffer import ReplayBuffer
from rlt_so101_dual.core.utils import soft_update, unflatten_chunk


def _action_quantiles(vla_path: str | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """ACTION q01/q99 from the VLA checkpoint's own preprocessor, in robot units.

    The collector needs these to express a human takeover's action in the
    normalised space the actor works in. Read from the VLA rather than the
    recording dataset because the online session's dataset starts empty, so its
    own stats do not exist yet -- the same reason
    load_dataset_stats_from_pretrained() exists.

    Returns (None, None) when unavailable; the collector then declines to
    substitute the anchor rather than storing robot units.
    """
    if not vla_path:
        return None, None
    from rlt_so101_dual.adapters.lerobot.record.common import load_dataset_stats_from_pretrained

    stats = load_dataset_stats_from_pretrained(vla_path)
    action = (stats or {}).get("action") or {}
    q01, q99 = action.get("q01"), action.get("q99")
    if q01 is None or q99 is None:
        logging.warning(
            "No ACTION q01/q99 in %s: a human takeover's BC anchor cannot be put in "
            "the actor's space, so takeovers will not steer the actor.", vla_path,
        )
        return None, None
    return (torch.as_tensor(q01, dtype=torch.float32).flatten(),
            torch.as_tensor(q99, dtype=torch.float32).flatten())


SNAPSHOT_VERSION = 1


def _checkpoint_identity(path: str | None) -> dict[str, Any]:
    """job_name / steps / dataset.repo_id from a checkpoint's train_config.json.

    Enough to tell two of our own checkpoints apart -- three of the four fields
    README / docs say to read before trusting any checkpoint -- without
    hashing ~1 GB of weights on every resume.

    An unreadable config answers all-None, which compares equal to another
    unreadable one. That is deliberate: a checkpoint can legitimately arrive
    without its config (the 70000-step VLA did), and refusing to resume forever
    is worse than verifying at the level the files allow.
    """
    ident: dict[str, Any] = {"job_name": None, "steps": None, "repo_id": None}
    if not path:
        return ident
    try:
        raw = json.loads((Path(path).expanduser() / "train_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return ident
    if not isinstance(raw, dict):
        return ident
    ident["job_name"] = raw.get("job_name")
    ident["steps"] = raw.get("steps")
    dataset = raw.get("dataset")
    if isinstance(dataset, dict):
        ident["repo_id"] = dataset.get("repo_id")
    return ident


class OnlineRLTrainer:
    """Owns the replay buffer, collector, optimizers, and warmup/critic-only
    state for one online-RL session, and can run gradient updates on `policy`
    in place. `policy` must already be a live `ChunkACPolicy` on the target
    device -- this class does not build or move it."""

    def __init__(self, policy: Any, online_rl_cfg: Any, policy_cfg: Any, task: str | None = None):
        self.policy = policy
        self.cfg = online_rl_cfg
        self.policy_cfg = policy_cfg
        # Recorded in the resume snapshot: the same weights against a different
        # task string is a different problem, and nothing else in these two
        # configs would catch it.
        self.task = task
        # Paper-faithful bootstrap: the VLA drives until the actor has actually
        # been trained. Closed here and reopened in maybe_update() once the
        # critic-only window ends. Harmless for a residual actor (mu == ref at
        # init anyway); required for a non-residual one, whose untrained output
        # is zeros rather than the reference.
        if hasattr(policy, "set_actor_gate"):
            policy.set_actor_gate(False)

        self.replay_buffer = ReplayBuffer(capacity=online_rl_cfg.replay_capacity)
        q01, q99 = _action_quantiles(getattr(policy_cfg, "vla_pretrained_path", None))
        # Fail here, not on the first chunk close. The collector also refuses to
        # fabricate units, but by then the arm has already executed a chunk --
        # the operator would be standing over a moving robot when it dies.
        if q01 is None or q99 is None:
            raise ValueError(
                "No ACTION q01/q99 available: cannot normalise executed actions into the "
                "policy's space, so the critic would train on robot units while actor_loss "
                f"queries it in [-1, 1]. Checked the preprocessor of "
                f"{getattr(policy_cfg, 'vla_pretrained_path', None)!r}. Point --vla-path at a "
                "checkpoint whose policy_preprocessor carries QUANTILES stats for 'action'."
            )
        self.collector = RLTOnlineCollector(
            replay_buffer=self.replay_buffer,
            chunk_length=policy_cfg.chunk_length,
            action_dim=policy_cfg.action_dim,
            milestone_reward=online_rl_cfg.milestone_reward,
            terminal_reward=online_rl_cfg.terminal_reward,
            time_decay=online_rl_cfg.time_decay,
            action_q01=q01,
            action_q99=q99,
        )
        # Separate optimizers (not just param groups on one Adam): actor lr
        # << critic lr since the critic should adapt quickly while the actor
        # -- which directly drives the robot -- should not, and this also
        # makes actor/critic state separately checkpointable/restartable.
        self.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=online_rl_cfg.lr_actor)
        self.critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=online_rl_cfg.lr_critic)
        # policy_cfg.actor_update_interval is read live by ChunkACPolicy.forward()
        # on every call; remembered here so the critic-only warmup window (below)
        # can temporarily inflate it and then restore the real value.
        self.online_actor_update_interval = policy_cfg.actor_update_interval
        # ChunkACPolicy.forward() soft-updates target_critic BEFORE the
        # caller's optimizer.step() actually applies this iteration's
        # critic gradient -- i.e. one iteration too early, off standard
        # TD3 order. forward() is shared with offline training
        # (lerobot-train), so it isn't touched; instead policy_cfg.tau is
        # temporarily zeroed around each online forward() call (making its
        # internal soft_update(..., tau=0) an exact no-op, since
        # target = (1-tau)*target + tau*source), and the real soft update
        # runs here afterwards, once critic.step() has actually happened.
        self.online_tau = policy_cfg.tau
        # BC-regularisation schedule. A large beta early keeps the policy
        # anchored to the VLA while the critic's Q estimates are still noisy;
        # relaxing it later lets the actor actually depart from the reference.
        # Held at beta_start until the critic-only window ends, since before
        # that the actor is frozen and the value would have no effect anyway.
        self.beta_start = policy_cfg.beta
        self.beta_final = getattr(online_rl_cfg, "beta_final", None)
        self.beta_anneal_episodes = max(1, getattr(online_rl_cfg, "beta_anneal_episodes", 1))
        # Set once, the episode warmup actually finished on -- see
        # warmup_satisfied(). None means warmup hasn't completed yet.
        self.warmup_completed_at_episode: int | None = None

        self.wandb_run = self._init_wandb(online_rl_cfg, policy_cfg)

    def _init_wandb(self, online_rl_cfg: Any, policy_cfg: Any) -> Any | None:
        if not getattr(online_rl_cfg, "wandb", False):
            return None
        try:
            import wandb
        except ImportError:
            logging.warning(
                "online_rl.wandb=true but the `wandb` package is not installed "
                "(pip install rlt-so101-dual[wandb]) -- continuing without wandb logging."
            )
            return None
        return wandb.init(
            project=online_rl_cfg.wandb_project,
            entity=online_rl_cfg.wandb_entity,
            name=online_rl_cfg.wandb_run_name,
            id=online_rl_cfg.wandb_run_id,
            resume=online_rl_cfg.wandb_resume,
            config={
                "warmup_episodes": online_rl_cfg.warmup_episodes,
                "critic_only_episodes": online_rl_cfg.critic_only_episodes,
                "min_warmup_transitions": online_rl_cfg.min_warmup_transitions,
                "min_warmup_successes": online_rl_cfg.min_warmup_successes,
                "min_warmup_failures": online_rl_cfg.min_warmup_failures,
                "replay_capacity": online_rl_cfg.replay_capacity,
                "batch_size": online_rl_cfg.batch_size,
                "lr_actor": online_rl_cfg.lr_actor,
                "lr_critic": online_rl_cfg.lr_critic,
                "utd_ratio": policy_cfg.utd_ratio,
                "max_updates_per_episode": online_rl_cfg.max_updates_per_episode,
                "use_stratified_sampling": online_rl_cfg.use_stratified_sampling,
                "chunk_length": policy_cfg.chunk_length,
                "action_dim": policy_cfg.action_dim,
                "actor_hidden_dim": policy_cfg.actor_hidden_dim,
                "critic_hidden_dim": policy_cfg.critic_hidden_dim,
            },
            settings=wandb.Settings(save_code=False),
        )

    def _log(self, data: dict[str, Any], step: int) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(data, step=step)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def start_episode(self, episode_id: int) -> int:
        """Call at the start of every rollout episode. Returns the replay
        buffer's `total_added` baseline to pass back into `maybe_update()`."""
        self.collector.start_episode(episode_id)
        return self.replay_buffer.total_added

    def discard_episode(self, buffer_total_added_before: int) -> None:
        """Roll back every transition this cycle's critical-phase attempt(s)
        already flushed into the replay buffer -- call this instead of
        maybe_update() when the whole episode is being rerecorded/discarded
        (left arrow), so a redo doesn't leave the discarded attempt's
        transitions (correctly labeled or not) permanently mixed into
        training data."""
        n_removed = self.replay_buffer.rollback(buffer_total_added_before)
        if n_removed:
            logging.info("Online RL: discarded %d transition(s) from rerecorded episode.", n_removed)

    def warmup_satisfied(self, completed_episodes: int) -> bool:
        """`completed_episodes` counts episodes finished *including* this one,
        so it is 1-based and matches what the operator sees.

        It used to be the caller's 0-based loop index, which meant
        --warmup-episodes 5 was only satisfied on the sixth episode.
        """
        if self.warmup_completed_at_episode is not None:
            return True  # sticky: once satisfied, stays satisfied
        if completed_episodes < self.cfg.warmup_episodes:
            return False
        if len(self.replay_buffer) < self.cfg.min_warmup_transitions:
            return False
        successes, failures = self.replay_buffer.count_outcomes()
        if not (successes >= self.cfg.min_warmup_successes and failures >= self.cfg.min_warmup_failures):
            return False
        # Record when warmup actually finished -- the critic-only window
        # below anchors to this, not a fixed offset (see its comment).
        self.warmup_completed_at_episode = completed_episodes
        logging.info(
            "=" * 70 + "\nOnline RL: warmup satisfied after episode %d -- gradient updates "
            "start now (critic-only for the next %d episodes, then actor unfreezes).\n" + "=" * 70,
            completed_episodes, self.cfg.critic_only_episodes,
        )
        return True

    def _scheduled_beta(self, completed_episodes: int, critic_only_until: int) -> float:
        """Linearly anneal beta from beta_start to beta_final once the actor
        starts training. Returns beta_start unchanged when no final value is
        configured, which is the default."""
        if self.beta_final is None:
            return self.beta_start
        elapsed = completed_episodes - critic_only_until
        if elapsed <= 0:
            return self.beta_start
        frac = min(1.0, elapsed / self.beta_anneal_episodes)
        return self.beta_start + (self.beta_final - self.beta_start) * frac

    def maybe_update(self, completed_episodes: int, buffer_total_added_before: int) -> dict | None:
        """Run gradient steps scaled to how much data this cycle actually
        added, in place on `policy` -- the very next rollout episode uses
        the updated weights directly, no checkpoint save/reload.

        The caller (or the collector's flush_episode()) is responsible for
        having already committed this cycle's transitions to the replay
        buffer -- this only checks whether that happened (via the
        total_added delta) and, if so, trains. Returns a stats dict on
        update (with `checkpoint_path` set if a periodic checkpoint was
        saved this call), or None if no update ran.
        """
        # total_added is monotonic (unlike len(), which stops growing once
        # the deque hits capacity and starts evicting) -- see ReplayBuffer.
        new_transitions = self.replay_buffer.total_added - buffer_total_added_before
        if new_transitions == 0:
            return None  # no critical-phase attempt was flushed this cycle

        if not self.warmup_satisfied(completed_episodes):
            successes, failures = self.replay_buffer.count_outcomes()
            logging.info(
                "Online RL warmup not yet satisfied after episode %d: "
                "transitions=%d (need %d), successes=%d (need %d), failures=%d (need %d)",
                completed_episodes, len(self.replay_buffer), self.cfg.min_warmup_transitions,
                successes, self.cfg.min_warmup_successes,
                failures, self.cfg.min_warmup_failures,
            )
            self._log(
                {
                    "online_rl/warmup_satisfied": 0,
                    "online_rl/buffer_transitions": len(self.replay_buffer),
                    "online_rl/buffer_successes": successes,
                    "online_rl/buffer_failures": failures,
                },
                step=completed_episodes,
            )
            return None
        if len(self.replay_buffer) < self.cfg.batch_size:
            return None

        # Critic-only window: freeze actor updates so the critic gets a
        # chance to form a non-random value estimate before the actor
        # starts moving away from its safe, VLA-equivalent starting
        # point. Implemented by inflating actor_update_interval (read
        # live by ChunkACPolicy.forward()) rather than toggling
        # requires_grad, so no actor gradient is even computed. Anchored
        # to warmup_completed_at_episode (set by warmup_satisfied(), which
        # already returned True above) rather than a fixed
        # warmup_episodes + critic_only_episodes offset -- if warmup took
        # longer than warmup_episodes to actually satisfy its
        # transition/success/failure thresholds, the fixed-offset version
        # would already be in the past, skipping critic-only entirely.
        assert self.warmup_completed_at_episode is not None
        critic_only_until = self.warmup_completed_at_episode + self.cfg.critic_only_episodes
        self.policy_cfg.beta = self._scheduled_beta(completed_episodes, critic_only_until)
        actor_live = completed_episodes >= critic_only_until
        self.policy_cfg.actor_update_interval = (
            self.online_actor_update_interval if actor_live else 10**9
        )
        # Hand the arm over at exactly the same moment the actor starts being
        # trained -- one condition, so the gate and the freeze cannot drift apart.
        if hasattr(self.policy, "set_actor_gate"):
            self.policy.set_actor_gate(actor_live)
        # See the comment at online_tau's definition: disable forward()'s
        # own (mistimed) soft update for the duration of this call, then
        # do it correctly ourselves after each real critic.step() below.
        self.policy_cfg.tau = 0.0

        # UTD: gradient updates scaled to how much new data this episode
        # added, not a fixed count regardless of episode length.
        requested_updates = max(new_transitions, 0) * self.policy_cfg.utd_ratio
        num_updates = min(requested_updates, self.cfg.max_updates_per_episode)

        device = next(self.policy.parameters()).device
        self.policy.train()
        start_t = time.perf_counter()
        info = None
        try:
            for _ in range(num_updates):
                # ReplayBuffer.sample()/sample_stratified() return core/losses.py's
                # flat-key format (exec_chunk_flat/ref_chunk_flat/next_ref_flat);
                # ChunkACPolicy's forward()/_coerce_batch expects unflattened
                # (B, C, action_dim) exec_chunk/ref_chunk/next_ref_chunk and
                # flattens internally.
                if self.cfg.use_stratified_sampling:
                    raw = self.replay_buffer.sample_stratified(self.cfg.batch_size)
                else:
                    raw = self.replay_buffer.sample(self.cfg.batch_size)
                chunk_length = self.policy_cfg.chunk_length
                batch = {
                    "state_vec": raw["state_vec"],
                    "exec_chunk": unflatten_chunk(raw["exec_chunk_flat"], chunk_length),
                    "ref_chunk": unflatten_chunk(raw["ref_chunk_flat"], chunk_length),
                    "reward_seq": raw["reward_seq"],
                    "next_state_vec": raw["next_state_vec"],
                    "next_ref_chunk": unflatten_chunk(raw["next_ref_flat"], chunk_length),
                    "done": raw["done"],
                    "actual_steps": raw["actual_steps"],
                }
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, info = self.policy.forward(batch)
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                # Matches ChunkACPolicyConfig.get_optimizer_preset()'s
                # grad_clip_norm=1.0 (applied automatically for offline
                # training by lerobot's generic train loop; this custom
                # online loop bypasses that loop entirely, so it has to be
                # applied explicitly here to get the same protection).
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()
                self.actor_optimizer.step()
                # Correct TD3 order: soft-update target only now, after
                # critic.step() has actually applied this iteration's gradient.
                soft_update(self.policy.target_critic, self.policy.critic, self.online_tau)
        finally:
            # Always restore, even on exception (CUDA OOM, bad sample, NaN
            # loss, ...): policy_cfg is the same object make_policy() gave
            # the live ChunkACPolicy, so a corrupted tau=0 /
            # actor_update_interval=10**9 left behind by an unhandled
            # exception would silently disable target updates / actor
            # training for the rest of the session (and leak into
            # policy.save_pretrained()'s config.json below).
            self.policy_cfg.tau = self.online_tau
            self.policy_cfg.actor_update_interval = self.online_actor_update_interval
            self.policy.eval()
        training_time_s = time.perf_counter() - start_t
        logging.info(
            "Online RL update after episode %d: new_transitions=%d requested_updates=%d "
            "actual_updates=%d effective_utd=%.2f training_time=%.1fs loss=%s",
            completed_episodes, new_transitions, requested_updates, num_updates,
            (num_updates / new_transitions) if new_transitions > 0 else 0.0,
            training_time_s,
            {k: (v.item() if torch.is_tensor(v) else v) for k, v in (info or {}).items()},
        )

        loss_dict = {k: (v.item() if torch.is_tensor(v) else v) for k, v in (info or {}).items()}
        stats: dict[str, Any] = {
            "new_transitions": new_transitions,
            "requested_updates": requested_updates,
            "actual_updates": num_updates,
            "training_time_s": training_time_s,
            "loss": loss_dict,
            "checkpoint_path": None,
        }
        successes, failures = self.replay_buffer.count_outcomes()
        self._log(
            {
                "online_rl/warmup_satisfied": 1,
                "online_rl/critic_only": float(completed_episodes < critic_only_until),
                "online_rl/beta": self.policy_cfg.beta,
                "online_rl/new_transitions": new_transitions,
                "online_rl/actual_updates": num_updates,
                "online_rl/effective_utd": (num_updates / new_transitions) if new_transitions > 0 else 0.0,
                "online_rl/training_time_s": training_time_s,
                "online_rl/buffer_transitions": len(self.replay_buffer),
                "online_rl/buffer_successes": successes,
                "online_rl/buffer_failures": failures,
                **{f"online_rl/{k}": v for k, v in loss_dict.items() if k != "critic_step"},
                # Reward accounting for the attempt just committed. `chunks`
                # is the decay exponent, so watching it fall is how you see
                # the speed incentive working.
                **{f"online_rl/attempt_{k}": v for k, v in self.collector.last_attempt_stats.items()},
            },
            step=completed_episodes,
        )
        if completed_episodes % self.cfg.save_every_episodes == 0:
            save_path = Path(self.cfg.save_dir) / f"step_{completed_episodes:06d}"
            self.policy.save_pretrained(save_path)
            logging.info("Online RL checkpoint saved to %s", save_path)
            stats["checkpoint_path"] = str(save_path)
        return stats

    def _immutable_identity(self) -> dict[str, Any]:
        """Fields a resume must not silently change.

        Each one either redefines the reward the buffer's stored transitions
        were already computed under, or changes what the stored actor/critic
        weights mean. Tensor shapes stay compatible across every one of them,
        so load_state_dict() would accept the mismatch without complaint and
        training would carry on against a quietly different problem -- the
        failure mode this whole check exists for.
        """
        p, c = self.policy_cfg, self.cfg
        return {
            "task": self.task,
            "gamma": p.gamma,
            "time_decay": c.time_decay,
            "milestone_reward": c.milestone_reward,
            "terminal_reward": c.terminal_reward,
            # Objective-changing, and invisible in the weights: resuming without
            # it would silently hand a pinned joint back to the actor.
            "pin_action_dims_to_ref": list(getattr(p, "pin_action_dims_to_ref", ()) or ()),
            "shape_contract": {
                "chunk_length": p.chunk_length,
                "action_dim": p.action_dim,
                "proprio_dim": p.proprio_dim,
                "rl_token_dim": p.rl_token_dim,
                "camera_keys": list(sc.CAMERA_KEYS),
            },
            "vla_identity": _checkpoint_identity(getattr(p, "vla_pretrained_path", None)),
            "rl_token_identity": _checkpoint_identity(getattr(p, "rl_token_pretrained_path", None)),
        }

    def _advisory_config(self) -> dict[str, Any]:
        """Recorded and diffed, but not enforced: changing these mid-run is a
        legitimate thing to want to do. They are reported on resume so the
        change is at least a decision rather than a surprise.
        """
        c, p = self.cfg, self.policy_cfg
        return {
            "lr_actor": c.lr_actor,
            "lr_critic": c.lr_critic,
            "beta": p.beta,
            "tau": p.tau,
            "utd_ratio": getattr(c, "utd_ratio", None),
            "batch_size": getattr(c, "batch_size", None),
            "replay_capacity": self.replay_buffer.capacity,
            "warmup_episodes": getattr(c, "warmup_episodes", None),
            "critic_only_episodes": getattr(c, "critic_only_episodes", None),
            "min_warmup_transitions": getattr(c, "min_warmup_transitions", None),
        }

    def save_latest_state(self, completed_episodes: int) -> None:
        """Overwrite a single, internally-consistent 'latest' snapshot every
        episode -- model weights AND optimizer state AND replay buffer AND
        counters together, not just the optimizer/buffer/counters. The
        periodic step_NNNNNN saves above (policy.save_pretrained(), every
        `save_every_episodes`) run on a DIFFERENT cadence than this
        per-episode save; if this file held only optimizer/buffer/counters,
        a crash between two step_NNNNNN saves would leave weights from
        episode N sitting next to optimizer/buffer state from episode N+2,
        which can't be recombined into a valid training state. Written to a
        temp file then atomically renamed so a crash mid-write never
        leaves a half-written file behind.

        See load_latest_state() for the corresponding resume path.
        """
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "snapshot_version": SNAPSHOT_VERSION,
            "immutable": self._immutable_identity(),
            "advisory": self._advisory_config(),
            "recorded_episodes": completed_episodes,
            "actor_state_dict": self.policy.actor.state_dict(),
            "critic_state_dict": self.policy.critic.state_dict(),
            "target_critic_state_dict": self.policy.target_critic.state_dict(),
            "critic_step": self.policy._critic_step,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer.buffer),
            "replay_buffer_total_added": self.replay_buffer.total_added,
            "replay_capacity": self.replay_buffer.capacity,
            "warmup_completed_at_episode": self.warmup_completed_at_episode,
            "torch_rng_state": torch.get_rng_state(),
        }
        final_path = save_dir / "latest_online_state.pt"
        tmp_path = save_dir / "latest_online_state.pt.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, final_path)
        archive_path = None
        if completed_episodes % self.cfg.save_every_episodes == 0:
            # Keep a selectable, internally-consistent training snapshot next
            # to the inference-only policy checkpoint. A hard link avoids
            # writing the often-large replay buffer twice; later replacement
            # of latest_online_state.pt does not modify this historical inode.
            archive_path = save_dir / f"step_{completed_episodes:06d}" / "online_state.pt"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_tmp = archive_path.with_suffix(".pt.tmp")
            try:
                archive_tmp.unlink(missing_ok=True)
                os.link(final_path, archive_tmp)
            except OSError:
                # Some filesystems do not support hard links. Preserve the
                # same interface with a regular copy in that case.
                shutil.copy2(final_path, archive_tmp)
            os.replace(archive_tmp, archive_path)
        logging.info(
            "Online RL state saved to %s%s (episodes=%d, buffer=%d transitions)",
            final_path,
            f" and selectable snapshot {archive_path}" if archive_path is not None else "",
            completed_episodes,
            len(self.replay_buffer),
        )

    def _reject_incompatible_snapshot(self, path: Path, state: dict) -> None:
        """Refuse a resume whose reward definition, shapes, or backbones differ.

        Everything checked here would otherwise load without error: the actor
        and critic tensors have the same shapes regardless of which VLA produced
        the RL tokens they were trained against, and the replay buffer carries
        rewards already baked at the old time_decay. The result is a run that
        looks healthy and optimises a different objective than the buffer
        describes.
        """
        version = state.get("snapshot_version")
        if version is None:
            raise ValueError(
                f"{path} carries no snapshot_version, so none of the fields that must match "
                "(gamma, reward shaping, shape contract, VLA/RL-token identity) can be checked. "
                "Stage 4 has not started, so no snapshot worth migrating exists yet -- start a "
                "fresh run rather than resuming blind."
            )
        if version != SNAPSHOT_VERSION:
            raise ValueError(
                f"{path} is snapshot_version {version}, this build writes {SNAPSHOT_VERSION}."
            )

        saved = state.get("immutable") or {}
        current = self._immutable_identity()
        mismatches = [
            f"  {key}:\n    snapshot: {saved.get(key)!r}\n    now:      {value!r}"
            for key, value in current.items()
            if saved.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"Refusing to resume {path}: these must match and do not.\n"
                + "\n".join(mismatches)
                + "\n\nThe stored weights and the buffered rewards were produced under the "
                "snapshot values. Either restore those settings, or start a fresh run."
            )

        for key, was in (state.get("advisory") or {}).items():
            now = self._advisory_config().get(key)
            if was != now:
                logging.warning("Resume changes %s: snapshot=%r now=%r (allowed, not enforced).", key, was, now)

    def _report_effective_lr(self) -> None:
        """Adam's param_groups come back from the snapshot, so the lr actually in
        use after a resume is the snapshot's -- not whatever the CLI asked for
        and printed in the startup summary. Report all three so the discrepancy
        cannot hide.
        """
        for name, optimizer, requested in (
            ("actor", self.actor_optimizer, self.cfg.lr_actor),
            ("critic", self.critic_optimizer, self.cfg.lr_critic),
        ):
            effective = [group.get("lr") for group in optimizer.param_groups]
            if any(lr != requested for lr in effective):
                logging.warning(
                    "%s lr after resume: CLI requested %r, optimizer state now %r "
                    "(load_state_dict restored the snapshot's param_groups).",
                    name, requested, effective,
                )

    def load_latest_state(self, path: str | Path) -> int:
        """Restore a snapshot written by save_latest_state(): model weights,
        optimizer momentum, the full replay buffer, and the warmup/critic-only
        anchor -- so training resumes exactly where it left off instead of
        re-entering warmup (or re-freezing the actor for another
        critic_only_episodes window) from the resume point. Does NOT restore
        torch_rng_state onto CUDA generators if the snapshot was saved on a
        different device configuration; this only affects exploration noise
        reproducibility, not correctness.

        Returns the recorded_episodes count to resume the outer episode loop
        from.
        """
        path = Path(path)
        # map_location="cpu", NOT the policy's device: the replay buffer's
        # ChunkTransition tensors must stay on CPU, matching the invariant
        # every other write path relies on (_emit_transition() always calls
        # .cpu() before storing; maybe_update() is what moves a *batch* to
        # device, right before the forward pass). Loading straight onto the
        # policy's device would leave resumed transitions on GPU while
        # newly-collected ones are CPU, and ReplayBuffer._collate()'s
        # torch.stack() crashes the moment a sampled batch mixes both.
        # actor_state_dict/critic_state_dict/optimizer state loaded below are
        # unaffected -- load_state_dict() copies values onto the existing
        # (already correctly-placed) parameters regardless of the source
        # tensors' device.
        #
        # weights_only=False: this snapshot embeds ChunkTransition dataclass
        # instances (the replay buffer) and optimizer state, not just plain
        # tensors -- torch>=2.6 defaults weights_only=True and refuses to
        # unpickle those. Safe here since this is our own save_latest_state()
        # output, not a third-party checkpoint.
        state = torch.load(path, map_location="cpu", weights_only=False)
        self._reject_incompatible_snapshot(path, state)
        self.policy.actor.load_state_dict(state["actor_state_dict"])
        self.policy.critic.load_state_dict(state["critic_state_dict"])
        self.policy.target_critic.load_state_dict(state["target_critic_state_dict"])
        self.policy._critic_step = state["critic_step"]
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.replay_buffer.buffer = deque(state["replay_buffer"], maxlen=self.replay_buffer.capacity)
        self.replay_buffer.total_added = state["replay_buffer_total_added"]
        self.warmup_completed_at_episode = state.get("warmup_completed_at_episode")
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        self._report_effective_lr()
        recorded_episodes = state["recorded_episodes"]
        # Reopen the actor gate here if the resumed run was already past
        # critic-only. __init__ closes it, and maybe_update() -- which is what
        # normally reopens it -- does not run until AFTER the first resumed
        # episode, so without this that episode would execute VLA actions while
        # its transitions were recorded as the actor's.
        if self.warmup_completed_at_episode is not None and hasattr(self.policy, "set_actor_gate"):
            actor_live = recorded_episodes >= (
                self.warmup_completed_at_episode + self.cfg.critic_only_episodes
            )
            self.policy.set_actor_gate(actor_live)
            logging.info("Actor gate after resume: %s", "open" if actor_live else "closed")
        logging.info(
            "Online RL resumed from %s (episodes=%d, buffer=%d transitions, warmup_completed_at_episode=%s)",
            path, recorded_episodes, len(self.replay_buffer), self.warmup_completed_at_episode,
        )
        return recorded_episodes
