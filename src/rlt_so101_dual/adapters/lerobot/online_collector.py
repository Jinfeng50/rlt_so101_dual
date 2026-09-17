from __future__ import annotations

import logging

import torch

from rlt_so101_dual.core.interfaces import ChunkTransition
from rlt_so101_dual.core.replay_buffer import ReplayBuffer
from rlt_so101_dual.adapters.lerobot.record.annotations import SOURCE_HUMAN


class RLTOnlineCollector:
    """Accumulates per-frame robot data into ChunkTransitions every C frames.

    Transitions stage locally per-episode and only commit to the global
    replay buffer on flush_episode() -- see _episode_staging.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        chunk_length: int,
        action_dim: int,
        milestone_reward: float = 0.0,
        terminal_reward: float = 1.0,
        time_decay: float = 1.0,
        action_q01: torch.Tensor | None = None,
        action_q99: torch.Tensor | None = None,
    ):
        self._buffer = replay_buffer
        self._C = chunk_length
        self._action_dim = action_dim
        self._milestone_reward = milestone_reward
        self._terminal_reward = terminal_reward
        # Speed incentive, applied as time_decay ** chunks_closed. Deliberately
        # a separate knob from gamma: gamma has to stay high for the terminal
        # reward to bootstrap back across a multi-hundred-chunk attempt, so
        # using it to also express "finish sooner" would trade away the
        # long-horizon credit assignment. 1.0 disables the decay.
        #
        # The exponent counts closed chunks, not frames. One chunk is
        # chunk_length/fps = 10/30 = 0.333 s, so a 5-20 s critical phase is
        # 15-60 chunks, over which 0.98 spans 0.74 down to 0.30. 0.995 --
        # the value inherited from the upstream bimanual setup -- would only
        # span 0.93 to 0.74 there, which is not enough of a gradient to
        # express "finish sooner" at all.
        self._time_decay = time_decay
        self._chunks_closed: int = 0
        self._milestone_given: bool = False
        # A milestone keypress lands on the *next* chunk to close rather than
        # cutting a short chunk at the moment of the press, so the chunk
        # boundaries stay aligned with the actor's own chunking.
        self._pending_bonus: float = 0.0
        self._milestone_bonus_awarded: float = 0.0
        # Per-attempt reward accounting, published for logging. Without it
        # there is no way to tell from a training run whether the milestone
        # key was ever pressed or how much the decay ate.
        self.last_attempt_stats: dict[str, float] = {}
        self._frame_actions: list[torch.Tensor] = []
        self._frame_sources: list[float] = []
        self._chunk_state: torch.Tensor | None = None
        self._chunk_ref: torch.Tensor | None = None
        self._chunk_is_critical: float = 0.0
        self._episode_id: int = -1
        self._prev_transition: ChunkTransition | None = None
        # Transitions accumulate here, NOT in the global replay buffer,
        # until flush_episode() commits them. A rerecorded/discarded/never-
        # labeled episode's staged transitions are simply dropped by the next
        # start_episode() call instead of permanently polluting the global
        # buffer with dangling non-terminal transitions (no valid next_state,
        # no outcome label, possibly built from footage the user rejected).
        self._episode_staging: list[ChunkTransition] = []
        # True once flush_episode() has fired for this "sub-episode" (one
        # critical-phase attempt). The recorded dataset episode may continue
        # past that point (e.g. VLA autonomously finishing a subsequent
        # placement step) -- on_frame() ignores those later frames entirely
        # so the RL reward reflects only what the actor actually controlled,
        # not whatever happens afterward. Reset by the next start_episode().
        # QUANTILES stats for the ACTION feature, in robot units. Needed to put
        # a human takeover's action into the same normalised space the actor and
        # its BC target live in -- see _emit_transition().
        self._action_q01 = action_q01
        self._action_q99 = action_q99
        self._flushed: bool = False

    def start_episode(self, episode_id: int) -> None:
        self._episode_id = episode_id
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        self._prev_transition = None
        self._episode_staging = []
        self._flushed = False
        self._reset_attempt_clock()

    def _reset_attempt_clock(self) -> None:
        self._chunks_closed = 0
        self._milestone_given = False
        self._pending_bonus = 0.0
        self._milestone_bonus_awarded = 0.0

    def _to_policy_space(self, action: torch.Tensor) -> torch.Tensor | None:
        """Robot units -> the policy's normalised action space, or None.

        Everything the actor and critic touch lives in the normalised space the
        VLA was trained in, but the actions reaching this collector come from
        loop.py's build_action_tensor(), i.e. the post-processor's output in robot
        units. Storing those raw trained the critic on one action space while
        actor_loss queried it in another, an order of magnitude apart -- so the Q
        term carried no usable gradient and online RL was doing pure BC.

        Normalising what the robot actually executed (rather than reusing the
        chunk the policy proposed) is also the more faithful choice: the
        post-processor may clip to joint limits, and the critic should learn the
        value of the action that happened.

        QUANTILES, matching the policy's own preprocessor:
        2*(x - q01)/(q99 - q01) - 1.
        """
        if self._action_q01 is None or self._action_q99 is None:
            raise ValueError(
                "RLTOnlineCollector has no ACTION q01/q99, so executed actions cannot be "
                "put in the policy's normalised space. Without them the critic trains on "
                "robot units while actor_loss queries it in [-1, 1] -- an order of "
                "magnitude apart -- and the Q term carries no usable gradient, which is "
                "how three runs silently degraded to pure BC. Pass a VLA checkpoint whose "
                "preprocessor carries the stats."
            )
        denom = self._action_q99 - self._action_q01
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        return 2.0 * (action - self._action_q01) / denom - 1.0

    def begin_attempt(self) -> None:
        """Start a critical-phase attempt: re-anchor the clock and the staging.

        Called when the operator enters the critical phase, which can happen
        more than once per recorded episode -- a failed attempt is resolved
        with `u`, then `r` starts another. Everything scoped to one attempt is
        reset here, not just the clock:

        * `_flushed`, or the retry would collect nothing at all and its reward
          would land on the *previous* attempt's last transition, marking a
          failed attempt successful.
        * `_prev_transition`, or the retry's first chunk would write its
          next-state back into the previous attempt's terminal transition.
        * `_episode_staging`, which at this point holds only VLA-prefix chunks
          from before the operator took the critical phase. Those are not
          something the actor controlled, and committing them would file them
          into the success/failure buckets under this attempt's label. The
          class already excludes the tail after an attempt resolves; this
          makes the head symmetric.
        """
        self._reset_attempt_clock()
        self._flushed = False
        self._prev_transition = None
        self._episode_staging = []
        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0

    def mark_milestone(self) -> float:
        """Award the one-off sub-goal bonus. Returns the amount, 0.0 if declined.

        Fires at most once per attempt: a repeatable bonus would let the policy
        farm it by redoing the sub-goal instead of finishing the task.
        """
        if self._flushed or self._milestone_given or self._milestone_reward <= 0.0:
            return 0.0
        self._milestone_given = True
        bonus = self._milestone_reward * (self._time_decay ** self._chunks_closed)
        self._pending_bonus += bonus
        self._milestone_bonus_awarded = bonus
        return bonus

    def on_frame(
        self,
        action: torch.Tensor,
        state_vec: torch.Tensor | None,
        ref_chunk: torch.Tensor | None,
        source_type: float,
        is_critical: float,
    ) -> ChunkTransition | None:
        if self._flushed:
            return None
        # VLA-prefix / post-attempt frames have no RL state/ref. Accumulating
        # them only spams "Dropping a chunk... no state/ref" every C steps and
        # they are discarded by begin_attempt() anyway. Skip until critical.
        if is_critical < 0.5 and state_vec is None:
            return None
        if len(self._frame_actions) == 0:
            # Capture state at chunk start. During intervention state_vec may be None;
            # fall back to the previous transition's state so human chunks are not dropped.
            if state_vec is not None:
                # rlt.get_last_chunk_tensors() returns compute_chunk()'s raw
                # batched tensors ((1, D) / (1, C, action_dim), B is always 1
                # for live single-robot inference) -- squeeze the leading
                # batch dim here so stored state/ref match exec_chunk's
                # unbatched (C, action_dim) shape. Without this, a chunk
                # never touched by human intervention keeps the (1, C, D)
                # shape while an intervened chunk gets (C, D) from
                # exec_chunk.clone() below, and ReplayBuffer._collate()'s
                # torch.stack() crashes the first time a batch mixes both.
                self._chunk_state = state_vec.squeeze(0) if state_vec.dim() > 1 else state_vec
                self._chunk_ref = ref_chunk.squeeze(0) if ref_chunk is not None and ref_chunk.dim() > 2 else ref_chunk
            elif self._prev_transition is not None:
                self._chunk_state = self._prev_transition.next_state_vec
                self._chunk_ref = self._prev_transition.next_ref_chunk
            self._chunk_is_critical = is_critical

        self._frame_actions.append(action.detach().cpu())
        self._frame_sources.append(source_type)

        if len(self._frame_actions) >= self._C:
            return self._emit_transition(done=False)
        return None

    def flush_episode(self, episode_success: bool) -> ChunkTransition | None:
        """Finalize the critical-phase attempt, marking the terminal
        transition done=1 and writing the sparse binary reward (r=1 iff
        success, else 0) onto its last valid timestep, matching the paper's
        terminal-reward-only setup. Commits every transition staged so far to
        the global replay buffer, then ignores any further on_frame() calls
        until the next start_episode() -- call this at the moment the
        critical phase itself resolves (success/failure), not necessarily at
        whole-episode end: the recorded episode may continue afterward (e.g.
        VLA autonomously finishing a subsequent step), but that tail is
        dataset-only and must not leak into the RL reward. For a
        rerecorded/discarded/never-labeled attempt, just don't call this and
        let the next start_episode() drop the staged data instead.
        """
        if self._flushed:
            # A second resolve for the same attempt. Without this guard the
            # terminal reward would be written onto the *previous* attempt's
            # last transition, which is already in the buffer -- silently
            # relabelling a failed attempt as successful.
            logging.warning("flush_episode() called twice for one attempt; ignoring the second.")
            return None

        result: ChunkTransition | None = None
        if self._frame_actions:
            result = self._emit_transition(done=True)
        if result is None and self._prev_transition is not None:
            # Either the attempt ended on an exact multiple of C, or the last
            # partial chunk could not be emitted (no state/ref captured). Both
            # need the previous chunk marked terminal: without a done=1
            # anywhere, the critic bootstraps forever off a self-referencing
            # next_state, and the terminal reward is never written at all.
            self._prev_transition.done = torch.tensor(1.0)
            result = self._prev_transition

        if result is not None and self._pending_bonus:
            # The milestone was pressed after the last chunk had already
            # closed, so no later _emit_transition() ever came to consume it.
            # Land it on the terminal chunk rather than letting it evaporate.
            actual = int(result.actual_steps.item())
            if actual > 0:
                result.reward_seq[actual - 1] += self._pending_bonus
                self._pending_bonus = 0.0

        # Computed after the last chunk closes so its own duration counts
        # toward the decay, the way every earlier chunk did.
        terminal = (
            self._terminal_reward * (self._time_decay ** self._chunks_closed)
            if episode_success
            else 0.0
        )
        awarded_terminal = 0.0
        if result is not None and terminal:
            actual = int(result.actual_steps.item())
            if actual > 0:
                result.reward_seq[actual - 1] += terminal
                awarded_terminal = terminal
        if terminal and awarded_terminal == 0.0:
            logging.error(
                "Attempt resolved as success but no transition could carry the terminal "
                "reward; it has been dropped. The attempt produced no usable chunk."
            )

        self.last_attempt_stats = {
            "chunks": float(self._chunks_closed),
            "milestone_awarded": float(self._milestone_given),
            # A bonus still pending has no chunk to land on and is lost;
            # report what actually reached a transition.
            "milestone_bonus": self._milestone_bonus_awarded - self._pending_bonus,
            "terminal_reward": awarded_terminal,
            "total_reward": float(sum(t.reward_seq.sum().item() for t in self._episode_staging)),
            "success": float(episode_success),
            "interventions": float(
                sum(1 for t in self._episode_staging if t.intervention.item() == 1.0)
            ),
        }

        outcome = torch.tensor(float(episode_success))
        for transition in self._episode_staging:
            # Every transition of the attempt carries the outcome, not just the
            # terminal one, so stratified sampling can bucket the lead-up steps
            # too -- and so a milestone bonus on a failed attempt is never
            # mistaken for success.
            transition.outcome = outcome.clone()
            self._buffer.add(transition)
        self._episode_staging = []
        self._flushed = True
        return result

    def _emit_transition(self, done: bool) -> ChunkTransition | None:
        if self._chunk_state is None or self._chunk_ref is None:
            # No state/ref was captured for this chunk, so it cannot become a
            # transition. Drop it, but say so: silently discarding chunks also
            # stops the decay clock, which inflates every later reward.
            logging.warning(
                "Dropping a chunk of %d frame(s): no state/ref captured at chunk start.",
                len(self._frame_actions),
            )
            self._frame_actions.clear()
            self._frame_sources.clear()
            self._chunk_state = None
            self._chunk_ref = None
            self._chunk_is_critical = 0.0
            return None

        actual = len(self._frame_actions)
        exec_list = self._frame_actions[: self._C]
        exec_chunk = self._to_policy_space(torch.stack(exec_list))
        if actual < self._C:
            # Pad AFTER normalising. Padding first meant the robot-unit zeros went
            # through the transform and became a fabricated action -- for
            # wrist_roll, 2*(0 - q01)/(q99 - q01) - 1 is -19.5. losses.py masks
            # only the reward with actual_steps, never the action tensors, and 91%
            # of the reward-carrying done=1 transitions are short chunks, so that
            # junk landed on precisely the transitions that matter most.
            pad = torch.zeros(self._C - actual, self._action_dim, dtype=exec_chunk.dtype)
            exec_chunk = torch.cat([exec_chunk, pad])

        # Deterministic tie-break: human wins ties (highest priority)
        dominant_source = max(set(self._frame_sources), key=lambda s: (self._frame_sources.count(s), s))
        ref = self._chunk_ref.cpu()
        if dominant_source == SOURCE_HUMAN:
            # A takeover replaces the BC anchor with what the person did, so the
            # actor is pulled toward the demonstration. exec_chunk is already in
            # the actor's space here; when it was not, this stored robot units and
            # made the BC target ~20x too large on exactly the transitions
            # stratified sampling pins at 20% of every batch.
            ref = exec_chunk.clone()

        state = self._chunk_state.cpu()
        reward_seq = torch.zeros(self._C)
        if self._pending_bonus and actual > 0:
            reward_seq[actual - 1] += self._pending_bonus
            self._pending_bonus = 0.0
        transition = ChunkTransition(
            state_vec=state,
            exec_chunk=exec_chunk,
            ref_chunk=ref,
            reward_seq=reward_seq,
            next_state_vec=state,
            next_ref_chunk=ref,
            done=torch.tensor(float(done)),
            intervention=torch.tensor(float(dominant_source == SOURCE_HUMAN)),
            actual_steps=torch.tensor(actual),
            source=torch.tensor(int(dominant_source)),
            episode_id=torch.tensor(self._episode_id),
            is_critical=torch.tensor(self._chunk_is_critical),
        )

        if self._prev_transition is not None:
            self._prev_transition.next_state_vec = state.clone()
            self._prev_transition.next_ref_chunk = ref.clone()

        self._episode_staging.append(transition)
        self._prev_transition = transition
        self._chunks_closed += 1

        self._frame_actions.clear()
        self._frame_sources.clear()
        self._chunk_state = None
        self._chunk_ref = None
        self._chunk_is_critical = 0.0
        return transition
