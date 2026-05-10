"""Robust OpenCV camera/video opening with diagnostics and warmup."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2

import config as C


LOGGER = logging.getLogger("eduaware.camera")


@dataclass
class CameraDiagnostics:
    source: str
    opened: bool
    backend: str
    width: int
    height: int
    fps: float


class ThreadedCamera:
    """Background frame reader that keeps the latest valid frame available."""

    def __init__(self, src, width: int, height: int):
        self._cap = _open_video_capture(src)
        self._configure(width, height)
        self._ret = False
        self._frame = None
        self._frame_id = 0
        self._dropped_reads = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.wait_until_ready(C.CAMERA_READ_TIMEOUT_S)

    def _configure(self, width: int, height: int):
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._ret = True
                    self._frame = frame
                    self._frame_id += 1
            else:
                with self._lock:
                    self._ret = False
                    self._dropped_reads += 1
                time.sleep(0.01)

    def read(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return self._ret and frame is not None, frame

    def wait_until_ready(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.01)
        return False

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def getBackendName(self):
        try:
            return self._cap.getBackendName()
        except Exception:
            return "unknown"

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def parse_source(src: str):
    return int(src) if str(src).isdigit() else src


def open_capture(src, threaded: bool = True):
    use_thread = threaded and isinstance(src, int)
    if use_thread:
        cap = ThreadedCamera(src, C.CAPTURE_WIDTH, C.CAPTURE_HEIGHT)
    else:
        cap = _open_video_capture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    diagnostics = get_diagnostics(cap, src)
    log_diagnostics(diagnostics)
    return cap, diagnostics


def _open_video_capture(src):
    """Open webcams with DirectShow on Windows and retry slow devices."""
    for attempt in range(1, C.CAMERA_OPEN_RETRIES + 1):
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        if cap.isOpened():
            return cap
        LOGGER.warning("camera open attempt %s/%s failed for %s", attempt, C.CAMERA_OPEN_RETRIES, src)
        cap.release()
        time.sleep(C.CAMERA_RETRY_DELAY_S)
    return cap


def get_diagnostics(cap, src) -> CameraDiagnostics:
    opened = bool(cap.isOpened())
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    backend = cap.getBackendName() if opened and hasattr(cap, "getBackendName") else "unknown"
    return CameraDiagnostics(str(src), opened, backend, width, height, fps)


def log_diagnostics(diagnostics: CameraDiagnostics):
    LOGGER.info("camera opened: %s", diagnostics.opened)
    LOGGER.info("camera source: %s", diagnostics.source)
    LOGGER.info("camera backend: %s", diagnostics.backend)
    LOGGER.info("frame resolution: %sx%s", diagnostics.width, diagnostics.height)
    LOGGER.info("capture fps reported: %.2f", diagnostics.fps)


def wait_for_first_frame(cap, timeout_s: float = C.CAMERA_READ_TIMEOUT_S) -> Optional[object]:
    """Wait for a real frame so slow camera warmup does not exit the program."""
    deadline = time.time() + timeout_s
    last_frame = None
    warmups = 0
    while time.time() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            last_frame = frame
            warmups += 1
            if warmups >= C.CAMERA_WARMUP_FRAMES:
                return last_frame
        time.sleep(0.02)
    return last_frame
