from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from rlt_so101_dual.core.actor import ChunkActor
from rlt_so101_dual.core.phase_controller import PhaseController
from rlt_so101_dual.core.rl_token import RLTokenModule
from rlt_so101_dual.core.utils import flatten_chunk, postprocess_prefix_tokens, unflatten_chunk


class PrefixOutputCapture:
    """Capture prefix hidden states from PI05Policy's PaliGemmaWithExpertModel.

    PI05's ``sample_actions`` calls ``paligemma_with_expert.forward()``
    directly (not via ``__call__``), so standard PyTorch forward hooks
    never fire.  Instead we monkey-patch ``forward`` to intercept the
    prefix-only call (``inputs_embeds=[prefix_embs, None]``).

    After capture the raw prefix tokens are optionally sliced to image-only
    (dropping language tokens) and pooled from (B, ~968, 2048) to
    (B, token_pool_size, 2048) via adaptive average pooling.
    """

    def __init__(
        self,
        token_pool_size: int = 64,
        image_only: bool = False,
        num_image_tokens: int = 0,
    ):
        self.token_pool_size = token_pool_size
        self.image_only = image_only
        self.num_image_tokens = num_image_tokens
        self._captured: Tensor | None = None
        self._original_forward = None
        self._target = None

    def attach(self, policy) -> None:
        """Monkey-patch ``forward`` on ``policy.model.paligemma_with_expert``.

        When image_only is true and num_image_tokens is unset (0), derive it from
        the SigLIP vision config so callers don't need to know paligemma internals.
        """
        target = policy.model.paligemma_with_expert
        self._target = target
        self._original_forward = target.forward

        if self.image_only and self.num_image_tokens == 0:
            self.num_image_tokens = self._infer_num_image_tokens(policy)

        capture = self  # closure reference

        def patched_forward(*args, **kwargs):
            result = capture._original_forward(*args, **kwargs)
            outputs, _past_kv = result
            prefix_tokens = outputs[0]
            if prefix_tokens is not None:
                capture._captured = postprocess_prefix_tokens(
                    prefix_tokens.detach().float(),
                    image_only=capture.image_only,
                    num_image_tokens=capture.num_image_tokens,
                    pool_size=capture.token_pool_size,
                )
            return result

        target.forward = patched_forward

    @staticmethod
    def _infer_num_image_tokens(policy) -> int:
        pi05_cfg = policy.config
        vision_cfg = policy.model.paligemma_with_expert.paligemma.config.vision_config
        n_per_cam = (pi05_cfg.image_resolution[0] // vision_cfg.patch_size) ** 2
        return n_per_cam * len(pi05_cfg.image_features)

    def consume(self) -> Tensor:
        """Return and clear the captured prefix tokens.

        Raises AssertionError if no prefix output has been captured yet (i.e.
        the VLA forward pass has not run since the last consume).
        """
        assert self._captured is not None, (
            "No prefix_output captured -- VLA forward not yet called"
        )
        result = self._captured
        self._captured = None
        return result

    def detach(self) -> None:
        """Restore the original forward method."""
        if self._original_forward is not None and self._target is not None:
            self._target.forward = self._original_forward
            self._original_forward = None
            self._target = None


@dataclass
class RLTStepMetadata:
    """Per-step metadata emitted alongside each popped action."""

    phase: float  # 0.0 = VLA, 1.0 = critical/RL
    source_type: float  # 0.0 = VLA action, 1.0 = RL action


class RLTActionModifier(nn.Module):
    """RL Token Encoder + Actor + Phase Controller + Chunk Queue.

    Sits between the VLA action output and the final action used by the robot.
    In VLA phase the VLA chunk is passed through unchanged; in RL phase the
    Actor refines the chunk conditioned on the RL-token state representation.
    """

    def __init__(
        self,
        rl_token: RLTokenModule,
        actor: ChunkActor,
        phase_ctrl: PhaseController,
        chunk_length: int,
        action_dim: int,
        proprio_dim: int,
        chunk_exec_steps: int = 50,
        vla_ref: bool = True,
        action_clip_delta: float | None = None,
    ):
        super().__init__()
        self.rl_token = rl_token
        self.actor = actor
        self.phase_ctrl = phase_ctrl
        self.chunk_length = chunk_length
        self.chunk_exec_steps = chunk_exec_steps
        self.vla_ref = vla_ref
        self.action_clip_delta = action_clip_delta
        # Whether the actor may drive the arm yet. The paper bootstraps online RL
        # by executing the VLA's own actions first and only hands control to the
        # actor once it has been trained; with a non-residual actor (the paper's
        # mu = f(x, ref), not ref + delta) an untrained actor outputs zeros, so
        # the handover has to be explicit rather than relying on a zero-init
        # residual head to make mu == ref. OnlineRLTrainer flips this when the
        # critic-only window closes; True by default so non-online callers
        # (evaluation of an already-trained checkpoint) are unaffected.
        self.actor_enabled = True
        self._cc_log_count = 0  # throttle for the compute_chunk ref diagnostic
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self._action_queue: deque[Tensor] = deque()
        self._step_metadata: deque[RLTStepMetadata] = deque()
        # Last computed RL-phase chunk tensors, for online-RL transition
        # collection (see RLTOnlineCollector). Populated only in RL phase;
        # persists (peek, not pop) until overwritten by the next compute_chunk()
        # or cleared by reset() -- see get_last_chunk_tensors().
        self._last_state_vec: Tensor | None = None
        self._last_ref_chunk: Tensor | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_rl_phase(self) -> bool:
        return self.phase_ctrl.is_critical

    @property
    def needs_new_chunk(self) -> bool:
        return len(self._action_queue) == 0

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_chunk(
        self,
        vla_chunk: Tensor,
        proprio: Tensor,
        prefix_tokens: Tensor,
    ) -> Tensor:
        """Compute action chunk, either VLA pass-through or RL-refined.

        Args:
            vla_chunk: (B, H, action_dim) normalised VLA action chunk.
            proprio: (B, proprio_dim) normalised proprioceptive state.
            prefix_tokens: (B, pool_size, token_dim) pooled VLA prefix tokens.

        Returns:
            VLA phase:  (B, chunk_exec_steps, action_dim)
            RL phase:   (B, chunk_length, action_dim) in [-1, 1].
        """
        phase_val = 1.0 if self.is_rl_phase else 0.0
        source_val = phase_val

        if not self.is_rl_phase:
            # VLA phase: take the first chunk_exec_steps consecutive actions.
            n = min(self.chunk_exec_steps, vla_chunk.shape[1])
            self._enqueue_metadata(phase_val, source_val, n)
            return vla_chunk[:, :n, :]

        # RL phase: take first chunk_length frames as actor reference
        ref_chunk = vla_chunk[:, :self.chunk_length, :]
        z_rl = self.rl_token.encode(prefix_tokens)
        state_vec = torch.cat([z_rl, proprio], dim=-1)
        ref_flat = flatten_chunk(ref_chunk)
        if not self.vla_ref:
            # Hide the VLA reference from the actor: training ref-dropout
            # multiplies ref_chunk_flat by a 0/1 mask, so a dropped sample
            # is exactly an all-zero ref. Reproduce that here.
            ref_flat = torch.zeros_like(ref_flat)
        should_log = self._cc_log_count < 3 or self._cc_log_count % 30 == 0
        # The paper's actor is "a Gaussian policy with a small fixed standard
        # deviation"; exploration comes from sampling it. forward() returns the
        # mean only, so sampling has to be requested explicitly. When
        # action_clip_delta is set (SO101 safety A/B), the bound applies to
        # the sampled action too.
        #
        # Paper-aligned online default: actor.fixed_std > 0 → sample every
        # RL chunk. Pass fixed_std=0 to execute the mean (SO101 anti-twitch A/B).
        if not self.actor_enabled:
            # Warmup / critic-only: the VLA drives, exactly as the paper's
            # bootstrap does. state_vec and ref_chunk are still computed and
            # stored below -- RLTOnlineCollector reads them to build the
            # transition, so short-circuiting here would leave every chunk
            # without a state at chunk start and the 512-transition warmup gate
            # could never be met. Executing ref_chunk reproduces what the old
            # zero-init residual head produced, bit for bit.
            mu = flatten_chunk(ref_chunk)
        elif getattr(self.actor, "fixed_std", 0.0) > 0.0:
            sampled, mu = self.actor.sample(state_vec, ref_flat, training=False)
            mu = sampled
        else:
            mu, _ = self.actor(state_vec, ref_flat, training=False)
        # No absolute [-1, 1] clamp here: QUANTILES maps q01/q99 to -1/+1, so the
        # normalised action space legitimately extends past +-1 -- wrist_roll's
        # q99 - q01 is 0.558 across the demonstrations, and its reference regularly
        # reaches ~5. Clamping discarded the VLA's own intent for that joint (a ref
        # of 4.737 executed as 1.0) and, worse, broke the safety property this code
        # claims: |executed - ref| came out at 3.737 while the bound said 0.1.
        # The delta bound below is the real guarantee and it is relative to ref,
        # which is where safety actually comes from.
        chunk = unflatten_chunk(mu, self.chunk_length)
        if self.action_clip_delta is not None:
            # Safety bound: limit how far the (possibly still-training) RL
            # actor's output may deviate from the VLA reference chunk, on
            # top of the residual-actor zero-init. Guards against a partially
            # trained actor producing large jumps on real hardware.
            delta = (chunk - ref_chunk).clamp(-self.action_clip_delta, self.action_clip_delta)
            chunk = ref_chunk + delta
        self._last_state_vec = state_vec.detach().clone()
        self._last_ref_chunk = ref_chunk.detach().clone()
        if should_log:
            delta = (chunk - ref_chunk).abs()
            # Diagnostic: the VLA ref is only the actor input; the returned
            # chunk below is the RLT actor output that will be executed.
            print(
                f"[RLT source=RLT_ACTOR vla_ref={self.vla_ref}] "
                f"compute_chunk #{self._cc_log_count}: "
                f"vla_ref[0,0,:4]={[round(v, 4) for v in ref_chunk[0, 0, :4].tolist()]}, "
                f"actor_out[0,0,:4]={[round(v, 4) for v in chunk[0, 0, :4].tolist()]}, "
                f"mean_abs_delta={delta.mean().item():.4f}, "
                f"max_abs_delta={delta.max().item():.4f}, "
                f"ref-into-actor abs-sum={ref_flat.abs().sum().item():.4f}",
                flush=True,
            )
        self._cc_log_count += 1

        self._enqueue_metadata(phase_val, source_val, self.chunk_length)
        return chunk

    def _enqueue_metadata(self, phase: float, source: float, count: int) -> None:
        """Enqueue metadata entries for every step in the upcoming chunk."""
        for _ in range(count):
            self._step_metadata.append(RLTStepMetadata(phase=phase, source_type=source))

    def enqueue(self, chunk: Tensor) -> None:
        """Enqueue chunk steps into the action queue.

        Args:
            chunk: (B, C, action_dim).
        """
        self._action_queue.extend(chunk.transpose(0, 1))

    def pop_action(self) -> Tensor:
        """Pop and return the next single-step action from the queue."""
        return self._action_queue.popleft()

    def pop_step_metadata(self) -> RLTStepMetadata | None:
        """Pop and return the next step's metadata, or None if empty."""
        if len(self._step_metadata) == 0:
            return None
        return self._step_metadata.popleft()

    def get_last_chunk_tensors(self) -> tuple[Tensor, Tensor] | None:
        """Peek (not pop) the (state_vec, ref_chunk) from the most recently
        computed RL-phase chunk, or None if none has been computed yet.

        Deliberately non-consuming: policy inference (and therefore
        compute_chunk()) is skipped entirely while human intervention is
        active, so this can go several frames without a fresh value. Callers
        that only care about it at their own chunk-start boundary (e.g.
        RLTOnlineCollector) still get the most recent real VLA/RL-token
        encoding available instead of losing it the instant it's read once.

        Does NOT include the actor's output chunk: RLTOnlineCollector builds
        exec_chunk itself from the actual per-frame executed actions (which
        may differ from the actor's raw output under human intervention), not
        from what the actor originally proposed.
        """
        if self._last_state_vec is None:
            return None
        return (self._last_state_vec, self._last_ref_chunk)

    # ------------------------------------------------------------------
    # Phase control (duck-typed interface for recording_loop)
    # ------------------------------------------------------------------

    def set_rl_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_critical()

    def set_vla_mode(self) -> None:
        self.interrupt_chunk()
        self.phase_ctrl.trigger_vla()

    def trigger_critical_phase(self) -> None:
        """Toggle between VLA and critical phase, clearing queues."""
        self.interrupt_chunk()
        if self.phase_ctrl.is_critical:
            self.phase_ctrl.trigger_vla()
        else:
            self.phase_ctrl.trigger_critical()

    def interrupt_chunk(self) -> None:
        """Clear action and metadata queues (e.g. on phase switch or human
        intervention start). Deliberately does NOT clear the last-chunk
        tensor cache (see get_last_chunk_tensors) -- intervention stops new
        chunks from being computed, so the cache should keep holding the most
        recent real encoding rather than going blank the instant it's read."""
        self._action_queue.clear()
        self._step_metadata.clear()

    def reset(self) -> None:
        """Reset queues and phase controller to initial state.

        Clears the last-chunk tensor cache only on a true episode reset.
        Intervention release must NOT wipe this cache (see interrupt_chunk);
        ChunkACPolicy.reset() is used for both paths historically, so callers
        that only need a queue flush should prefer interrupt_chunk().
        """
        self._action_queue.clear()
        self._step_metadata.clear()
        self._last_state_vec = None
        self._last_ref_chunk = None
        self.phase_ctrl.reset()
