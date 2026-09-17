# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Did the control loop write the same camera frame into the dataset twice?

`SOFollower.get_observation()` calls `camera.read_latest()`, which is
non-blocking and returns `self.latest_frame` *without copying it*. So when the
loop samples faster than a camera produces, the same array object is written
into two consecutive dataset rows, and nothing else in the pipeline notices:
the frame count, the declared fps and the loop rate all stay correct.

Comparing the returned object with ``is`` answers exactly that question, with
no threshold and no race:

* It looks at the array the loop actually wrote, not at a timestamp sampled at
  some other moment. The counter this replaces read ``camera.latest_timestamp``
  *after* ``get_observation()`` had already returned, so the timestamp it
  recorded could belong to a frame that arrived after the one it wrote. It
  could therefore call a round stale when two different frames were written,
  and fresh when the same frame was written twice.
* Holding a reference to the previous frame keeps it alive, so its address
  cannot be recycled under a new array. ``is`` cannot produce a false match.

Do not measure this on the encoded video instead. A duplicate that survives
into an h264 file is not recoverable from the pixels: LeRobot encodes with
``g=2``, so every other frame is an I-frame, and two byte-identical inputs
decode to images differing by tens of grey levels -- more than genuine motion
does in the quiet parts of an episode. See
``tests/rlt/test_h264_roundtrip.py``.

The content hash is a second, narrower question: did the camera hand over a
*different* array holding *identical* bytes? That is a camera or driver
repeat rather than the loop outrunning the camera, so it is counted apart.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SIDECAR_NAME = "camera_delivery.jsonl"


def frame_digest(frame: np.ndarray) -> bytes:
    """Content hash of one frame, over the bytes as laid out in memory."""
    return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=16).digest()


class FrameDeliveryCounter:
    """Per-episode tally of repeated camera buffers.

    Counts are kept raw -- samples and repeats, never percentages -- so that a
    WARN or FAIL threshold chosen later can be applied to sessions recorded
    before it existed, without re-recording them.
    """

    def __init__(self, hash_content: bool = True) -> None:
        self.samples = 0
        self.repeated_buffers: dict[str, int] = {}
        self.hash_repeats: dict[str, int] = {}
        # Which rows carry a repeat, not just how many. Whether repeats cluster
        # at the start of an episode or spread through it decides the shape of
        # any threshold -- a rate, or a count after ignoring the first frames --
        # and that question cannot be answered from a total. Storing it costs a
        # few hundred bytes per episode and cannot be recovered afterwards: the
        # repeats are invisible in the encoded video.
        self.at_frames: dict[str, list[int]] = {}
        self.hash_at_frames: dict[str, list[int]] = {}
        self._hash_content = hash_content
        # Keeping the previous frame alive is what makes `is` trustworthy.
        self._prev_frame: dict[str, Any] = {}
        self._prev_digest: dict[str, bytes] = {}

    def observe(self, obs: Mapping[str, Any], names: Iterable[str]) -> None:
        """Record one loop sample. `names` are the camera keys in `obs`.

        Every loop sample becomes one dataset row, so the sample index is the
        `frame_index` of the row within the episode.
        """
        self.samples += 1
        frame_index = self.samples - 1
        for name in names:
            frame = obs.get(name)
            if frame is None:
                continue
            self.repeated_buffers.setdefault(name, 0)
            self.hash_repeats.setdefault(name, 0)
            self.at_frames.setdefault(name, [])
            self.hash_at_frames.setdefault(name, [])

            prev = self._prev_frame.get(name)
            if prev is not None and frame is prev:
                # Same object: the bytes are identical by definition, so this
                # is not also a hash repeat. Leave the stored digest alone.
                self.repeated_buffers[name] += 1
                self.at_frames[name].append(frame_index)
            elif self._hash_content:
                digest = frame_digest(frame)
                if self._prev_digest.get(name) == digest:
                    self.hash_repeats[name] += 1
                    self.hash_at_frames[name].append(frame_index)
                self._prev_digest[name] = digest
            self._prev_frame[name] = frame

    @property
    def cameras(self) -> list[str]:
        return sorted(self.repeated_buffers)

    def record(self, episode_index: int | None, elapsed_s: float) -> dict[str, Any]:
        """The sidecar row for this episode."""
        return {
            "episode_index": episode_index,
            "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(elapsed_s, 2),
            "loop_samples": self.samples,
            "cameras": {
                name: {
                    "repeated_buffers": self.repeated_buffers[name],
                    "hash_repeats": self.hash_repeats[name],
                    "at_frames": self.at_frames[name],
                    "hash_at_frames": self.hash_at_frames[name],
                }
                for name in self.cameras
            },
        }

    def summary_line(self) -> str:
        """One-line human summary; percentages here, raw counts in the file."""
        parts = []
        for name in self.cameras:
            n = self.repeated_buffers[name]
            pct = 100 * n / self.samples if self.samples else 0.0
            extra = ""
            if self.hash_repeats[name]:
                extra = f" + {self.hash_repeats[name]} byte-identical"
            parts.append(f"{name} {n} repeated buffers ({pct:.1f}%){extra}")
        return "   ".join(parts)


def append_sidecar(path: Path | str, record: Mapping[str, Any]) -> None:
    """Append one episode's row. Never raises: instrumentation must not end a run."""
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(record, sort_keys=False) + "\n")
    except OSError:
        pass


def resolve_episode_index(dataset: Any) -> int | None:
    """Index of the episode that was just recorded, or None if it cannot be read.

    `num_episodes` counts the episodes already saved, and `save_episode()` runs
    after the recording loop returns, so during the loop it is the index of the
    episode in progress. Note that a discarded episode leaves a row whose index
    the retake will reuse; rows are appended in order, so the later row for an
    index is the one that describes kept data.

    Do not reach for `dataset.episode_buffer`: that attribute lives on
    `dataset.writer`, not on `LeRobotDataset`. Reading it from the dataset
    raises AttributeError, which is how the first version of this crashed a
    recording session.
    """
    if dataset is None:
        return None
    try:
        return int(dataset.num_episodes)
    except Exception:  # noqa: BLE001 - see persist(): losing the number beats losing the run
        return None


def persist(counter: FrameDeliveryCounter, dataset: Any, elapsed_s: float) -> dict[str, Any]:
    """Build this episode's row and append it next to the episodes it describes.

    Every failure mode here is swallowed. A missing number is a gap in a
    diagnostic; a raised exception in this path would cost the episode itself,
    and the whole reason this counter exists is that instrumentation had been
    trusted without ever being run.
    """
    record = counter.record(resolve_episode_index(dataset), elapsed_s)
    root = getattr(dataset, "root", None) if dataset is not None else None
    if root is not None:
        append_sidecar(Path(root) / SIDECAR_NAME, record)
    return record
