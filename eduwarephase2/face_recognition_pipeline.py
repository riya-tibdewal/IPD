"""Asynchronous identity cache scaffold for tracker-associated face recognition.

This module is intentionally adapter-based: plug an existing recognizer into
`AsyncFaceIdentityManager(recognizer=...)`.  The runtime keeps tracking and
behavior scoring on every frame while identity revalidation happens every N
frames and is cached on the track ID.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

import config as C


class FaceRecognizerAdapter(Protocol):
    def recognize(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, float]:
        ...


@dataclass
class IdentityResult:
    track_id: int
    name: str
    confidence: float
    created_at: float


class AsyncFaceIdentityManager:
    """Associates identities with continuous tracker IDs without blocking FPS."""

    def __init__(self, recognizer: Optional[FaceRecognizerAdapter] = None, max_workers: int = 1):
        self.recognizer = recognizer
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending: dict[int, concurrent.futures.Future] = {}
        self.last_submit: dict[int, float] = {}

    def maybe_submit(self, frame_idx: int, frame: np.ndarray, state, bbox: tuple[int, int, int, int]):
        if self.recognizer is None:
            return
        now = time.time()
        age = now - state.temporal.identity_last_seen
        x1, y1, x2, y2 = bbox
        if state.seen_frames < C.FACE_MIN_TRACK_FRAMES:
            return
        if min(x2 - x1, y2 - y1) < C.FACE_MIN_BOX_SIZE:
            return
        if state.last_factors.get("back_view", 0.0) > 0.50 or state.last_factors.get("head_down", 0.0) > 0.65:
            return
        if now - self.last_submit.get(state.track_id, 0.0) < C.FACE_SUBMIT_COOLDOWN_S:
            return
        should_run = frame_idx % C.FACE_RECOGNITION_EVERY_N == 0
        should_revalidate = age > C.FACE_REVALIDATE_S
        if not (should_run or should_revalidate):
            return
        if state.track_id in self.pending:
            return
        crop_frame = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()
        if crop_frame.size == 0:
            return
        self.last_submit[state.track_id] = now
        self.pending[state.track_id] = self.executor.submit(
            self._recognize_safe,
            state.track_id,
            crop_frame,
            (0, 0, crop_frame.shape[1], crop_frame.shape[0]),
        )

    def collect(self, tracker):
        done_ids = [tid for tid, fut in self.pending.items() if fut.done()]
        for tid in done_ids:
            result = self.pending.pop(tid).result()
            state = tracker.get_or_create(tid)
            if result.confidence >= C.FACE_MIN_CONFIDENCE:
                current_conf = state.temporal.identity_confidence
                is_empty = state.temporal.identity in ("", "Unknown")
                if is_empty or result.name == state.temporal.identity or result.confidence >= current_conf + 0.08:
                    state.temporal.identity = result.name
                    state.temporal.identity_confidence = result.confidence
                    state.temporal.identity_last_seen = result.created_at

    def expire(self, state):
        if state.temporal.identity == "Unknown":
            return
        if time.time() - state.temporal.identity_last_seen > C.FACE_IDENTITY_TTL_S:
            state.temporal.identity_confidence *= 0.98

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _recognize_safe(self, track_id: int, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> IdentityResult:
        try:
            name, confidence = self.recognizer.recognize(frame, bbox)
        except Exception:
            name, confidence = "Unknown", 0.0
        return IdentityResult(track_id, name, confidence, time.time())
