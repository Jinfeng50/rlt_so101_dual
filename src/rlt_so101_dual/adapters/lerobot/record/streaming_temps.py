"""Human-readable temp dirs for streaming video encode mid-flight.

Upstream LeRobot uses ``tempfile.mkdtemp`` under the session root, which leaves
opaque names like ``tmpi9kc5zxd/``. On a clean finalize those dirs are consumed
into ``videos/``; after Ctrl+C / crash they linger and look like the dataset.

We rename the mid-flight folders (and filenames) to camera aliases so leftovers
are readable: ``left_wrist/``, ``right_wrist/``, ``right_front/``.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
from pathlib import Path

log = logging.getLogger(__name__)

CAMERA_TEMP_DIR_NAMES: dict[str, str] = {
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "right_front": "right_front",
}

_PATCHED = False


def camera_temp_dirname(video_key: str) -> str:
    """Map ``observation.images.left_wrist`` → ``left_wrist``."""
    cam = video_key.rsplit(".", 1)[-1]
    return CAMERA_TEMP_DIR_NAMES.get(cam, cam)


def patch_streaming_temp_dir_names() -> None:
    """Idempotent monkeypatch of ``StreamingVideoEncoder.start_episode``."""
    global _PATCHED
    if _PATCHED:
        return

    from lerobot.datasets.video_utils import StreamingVideoEncoder, _CameraEncoderThread

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        if self._episode_active:
            self.cancel_episode()

        self._dropped_frames.clear()
        base = Path(temp_dir)

        for video_key in video_keys:
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()

            dirname = camera_temp_dirname(video_key)
            temp_video_dir = base / dirname
            if temp_video_dir.exists():
                shutil.rmtree(temp_video_dir)
            temp_video_dir.mkdir(parents=True)
            video_path = temp_video_dir / f"{dirname}.mp4"

            encoder_thread = _CameraEncoderThread(
                video_path=video_path,
                fps=self.fps,
                vcodec=self.vcodec,
                pix_fmt=self.pix_fmt,
                g=self.g,
                crf=self.crf,
                preset=self.preset,
                frame_queue=frame_queue,
                result_queue=result_queue,
                stop_event=stop_event,
                encoder_threads=self.encoder_threads,
            )
            encoder_thread.start()

            self._frame_queues[video_key] = frame_queue
            self._result_queues[video_key] = result_queue
            self._threads[video_key] = encoder_thread
            self._stop_events[video_key] = stop_event
            self._video_paths[video_key] = video_path

        self._episode_active = True

    StreamingVideoEncoder.start_episode = start_episode  # type: ignore[method-assign]
    _PATCHED = True
    log.info(
        "Streaming encode temps use camera dirs: %s",
        ", ".join(f"{k}→{v}" for k, v in CAMERA_TEMP_DIR_NAMES.items()),
    )
