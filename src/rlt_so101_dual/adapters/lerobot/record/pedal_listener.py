"""Linux evdev foot-pedal listener for real-robot recording."""
from __future__ import annotations

import glob
import logging
import os
import select
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

LINTX_PEDAL_GLOB = "/dev/input/by-id/usb-LinTx_LinTx_Keyboard_*-if01-event-kbd"


def discover_pedal_devices() -> tuple[str, ...]:
    """Return stable LinTx foot-pedal event-device symlinks."""
    return tuple(sorted(glob.glob(LINTX_PEDAL_GLOB)))


class PedalListener:
    """Background listener that reports LinTx pedal key-downs as key names."""

    def __init__(
        self,
        on_press: Callable[[str], None],
        devices: tuple[str, ...] | list[str] | None = None,
        key_map: dict[int, str] | None = None,
    ) -> None:
        from evdev import ecodes

        self._on_press = on_press
        self._device_paths = tuple(devices) if devices is not None else discover_pedal_devices()
        self._key_map = key_map or {
            ecodes.KEY_R: "r",
            ecodes.KEY_SPACE: "space",
        }
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._devices = []

    def start(self) -> bool:
        from evdev import InputDevice

        for path in self._device_paths:
            if not os.path.exists(path):
                log.info("Pedal device not found, skipping: %s", path)
                continue
            if not os.access(path, os.R_OK):
                log.warning("Pedal device not readable; check plugdev udev permissions: %s", path)
                continue
            self._devices.append(InputDevice(path))
            log.info("Pedal listener attached: %s", path)

        if not self._devices:
            log.info("No readable pedals found; pedal listener disabled")
            return False

        self._thread = threading.Thread(target=self._run, name="rlt-so101-dual-pedal-listener", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        for device in self._devices:
            device.close()
        self._devices = []

    def _run(self) -> None:
        fd_to_device = {device.fd: device for device in self._devices}
        while not self._stop_event.is_set():
            readable, _, _ = select.select(list(fd_to_device), [], [], 0.2)
            for fd in readable:
                for event in fd_to_device[fd].read():
                    self._handle_event(event)

    def _handle_event(self, event) -> None:
        from evdev import ecodes

        if event.type != ecodes.EV_KEY or event.value != 1:
            return
        key_name = self._key_map.get(event.code)
        if key_name is None:
            return
        self._on_press(key_name)
