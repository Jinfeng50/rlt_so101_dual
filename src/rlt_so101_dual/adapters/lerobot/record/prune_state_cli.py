"""Remove specific episode(s) from a saved online-RL state's replay buffer
(latest_online_state.pt) -- e.g. an episode later found to have a bad
outcome label or corrupted data. This only stops those transitions from
being sampled in future training; it does not undo any gradient step already
taken while they were still in the buffer (see the module-level warning
printed at runtime).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def _drop(state: dict, keep) -> tuple[dict, int]:
    buffer = state["replay_buffer"]
    kept = [t for t in buffer if keep(t)]
    n_removed = len(buffer) - len(kept)
    state["replay_buffer"] = kept
    # total_added must stay an accurate count of what's actually in the
    # buffer's history (see ReplayBuffer.rollback()'s same convention) --
    # decrement by what was actually removed, don't just recompute from len().
    state["replay_buffer_total_added"] = state["replay_buffer_total_added"] - n_removed
    return state, n_removed


def prune_episodes(state: dict, episode_ids: set[int]) -> tuple[dict, int]:
    return _drop(state, lambda t: int(t.episode_id.item()) not in episode_ids)


def prune_interventions(state: dict, episode_ids: set[int] | None = None) -> tuple[dict, int]:
    """Drop human-intervention transitions, optionally only in some episodes.

    A terminal transition is never dropped: it carries the attempt's outcome
    and its reward, and removing it would leave the rest of the episode with
    no outcome anchor for stratified sampling.
    """

    def keep(t) -> bool:
        if float(t.intervention.item()) != 1.0:
            return True
        if float(t.done.item()) == 1.0:
            return True
        return episode_ids is not None and int(t.episode_id.item()) not in episode_ids

    return _drop(state, keep)


def summarise(state: dict) -> str:
    """Per-episode transition / intervention counts, so you can see what you
    would be removing before removing it."""
    buffer = state["replay_buffer"]
    if not buffer:
        return "replay buffer is empty"
    stats: dict[int, dict[str, int]] = {}
    outcomes: dict[int, str] = {}
    for t in buffer:
        eid = int(t.episode_id.item())
        row = stats.setdefault(eid, {"n": 0, "intervention": 0})
        row["n"] += 1
        row["intervention"] += int(float(t.intervention.item()) == 1.0)
        if float(t.done.item()) == 1.0:
            explicit = float(getattr(t, "outcome", torch.tensor(-1.0)).item())
            if explicit >= 0.0:
                outcomes[eid] = "success" if explicit >= 0.5 else "failure"
            else:
                outcomes[eid] = "success" if t.reward_seq.sum().item() > 0 else "failure"

    lines = [f"{len(buffer)} transitions across {len(stats)} episode(s)",
             f"{'episode':>8}  {'chunks':>7}  {'interv':>7}  outcome"]
    for eid in sorted(stats):
        row = stats[eid]
        lines.append(f"{eid:>8}  {row['n']:>7}  {row['intervention']:>7}  {outcomes.get(eid, '-')}")
    total_int = sum(r["intervention"] for r in stats.values())
    lines.append(f"{'total':>8}  {len(buffer):>7}  {total_int:>7}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove one or more episode_id's transitions from a saved "
        "online-RL latest_online_state.pt."
    )
    parser.add_argument("--state-path", required=True, help="Path to latest_online_state.pt.")
    parser.add_argument(
        "--episode-id", type=int, action="append", default=None, dest="episode_ids",
        help="Episode id to remove (repeat --episode-id N for multiple). This is the "
        "recorded-episode index shown in training logs, e.g. 'Recording episode 41' -> 41. "
        "With --drop-interventions it narrows the removal to those episodes instead.",
    )
    parser.add_argument(
        "--drop-interventions", action="store_true", default=False,
        help="Remove human-intervention transitions rather than whole episodes -- for "
        "a takeover you decided was a bad demonstration. Terminal transitions are "
        "always kept, since they carry the attempt's outcome. Note that intervention "
        "data is weighted heavily (a fixed 20%% of every stratified batch, and it "
        "replaces the VLA reference in the BC term), so a few bad takeovers are worth "
        "removing.",
    )
    parser.add_argument(
        "--list", action="store_true", default=False,
        help="Print per-episode transition and intervention counts, then exit.",
    )
    parser.add_argument(
        "--in-place", action="store_true", default=False,
        help="Overwrite --state-path (a .bak backup of the original is written first). "
        "Default: write to --output instead, leaving the original untouched.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path when not --in-place. Defaults to <state-path>.pruned.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    state_path = Path(args.state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    buffer_len_before = len(state["replay_buffer"])

    if args.list:
        print(summarise(state))
        return

    if not args.drop_interventions and not args.episode_ids:
        raise SystemExit("Pass --episode-id, or --drop-interventions, or --list.")

    episode_ids = set(args.episode_ids) if args.episode_ids else None
    if args.drop_interventions:
        state, n_removed = prune_interventions(state, episode_ids)
        what = "human-intervention transition(s)"
        scope = f" in episodes {sorted(episode_ids)}" if episode_ids else ""
    else:
        state, n_removed = prune_episodes(state, episode_ids)
        what = "transition(s)"
        scope = f" with episode_id in {sorted(episode_ids)}"

    if n_removed == 0:
        print(f"No {what}{scope} found in {buffer_len_before} total -- "
              "nothing removed, no file written.")
        return

    if args.in_place:
        backup_path = state_path.with_suffix(state_path.suffix + ".bak")
        shutil.copy2(state_path, backup_path)
        out_path = state_path
        print(f"Backed up original ({buffer_len_before} transitions) to {backup_path}")
    else:
        out_path = Path(args.output) if args.output else state_path.with_suffix(state_path.suffix + ".pruned.pt")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(out_path)
    print(
        f"Removed {n_removed} {what}{scope} "
        f"({buffer_len_before} -> {len(state['replay_buffer'])} transitions). Wrote {out_path}\n"
        "Note: gradient steps already taken using this data before now cannot be undone -- "
        "this only prevents future sampling of it."
    )


if __name__ == "__main__":
    main()
