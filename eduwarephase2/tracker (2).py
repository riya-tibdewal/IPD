"""
tracker.py — EduAware Per-Student State Manager
=================================================
Wraps YOLOv8's built-in ByteTrack with a per-student state object that
combines:
  • Temporal smoothing buffer  (temporal_smoothing.StudentBuffer)
  • Attention score registry   (attention_scoring.AttentionRegistry)
  • Wrist history              (for motion-based phone/writing detection)
  • Last-seen timestamp        (stale track pruning)

Why ByteTrack (built into YOLOv8)?
  • Zero extra dependencies
  • Motion-only association → fast on CPU
  • Handles occlusion & re-entry with stable IDs
  • Adequate for a fixed classroom camera where students don't change seats

StudentTracker is a thin coordinator.  It does NOT contain heuristic logic
(that lives in behavior_analysis.py) or display logic (visualization.py).
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from temporal_smoothing import StudentBuffer
from attention_scoring   import AttentionRegistry
import config as C


# ══════════════════════════════════════════════════════════════════════════════
# PER-STUDENT STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackState:
    """
    All mutable state for one tracked student.

    Fields
    ──────
    track_id    : ByteTrack integer ID (stable across frames)
    buffer      : temporal smoothing buffer
    prev_wrists : (2, 2) float32 wrist positions from the last frame,
                  used to compute wrist motion velocity in classify()
    last_seen   : Unix timestamp; tracks older than STALE_TRACK_S are pruned
    """
    track_id:    int
    buffer:      StudentBuffer      = field(default_factory=StudentBuffer)
    prev_wrists: Optional[np.ndarray] = None
    last_seen:   float              = field(default_factory=time.time)

    @property
    def stable_label(self) -> str:
        return self.buffer.stable

    @property
    def confidence(self) -> float:
        return self.buffer.confidence

    @property
    def recent_labels(self):
        """Expose the buffer's internal deque for Talking detection."""
        return self.buffer._buf

    def touch(self):
        """Update last_seen to now (called every frame this track appears)."""
        self.last_seen = time.time()

    def is_stale(self, timeout_s: float = C.STALE_TRACK_S) -> bool:
        return (time.time() - self.last_seen) > timeout_s


# ══════════════════════════════════════════════════════════════════════════════
# TRACKER COORDINATOR
# ══════════════════════════════════════════════════════════════════════════════

class StudentTracker:
    """
    Manages the collection of TrackState objects and the AttentionRegistry.

    Typical frame loop
    ──────────────────
    1. YOLO returns list of (track_id, bbox, kps) tuples
    2. For each tuple:
         state = tracker.get_or_create(track_id)
         raw_label, wrists = classify(kps, state.recent_labels, state.prev_wrists)
         state.prev_wrists = wrists
         stable = state.buffer.push(raw_label)
         state.touch()
         tracker.registry.record(track_id, stable)
    3. tracker.prune_stale()
    """

    def __init__(self):
        self._states:  dict[int, TrackState] = {}
        self.registry: AttentionRegistry     = AttentionRegistry()

    # ──────────────────────────────────────────────────────────────────────
    # State access
    # ──────────────────────────────────────────────────────────────────────

    def get_or_create(self, track_id: int) -> TrackState:
        """Return existing state or create a fresh one for a new track."""
        if track_id not in self._states:
            self._states[track_id] = TrackState(track_id=track_id)
        return self._states[track_id]

    def __len__(self) -> int:
        return len(self._states)

    # ──────────────────────────────────────────────────────────────────────
    # Maintenance
    # ──────────────────────────────────────────────────────────────────────

    def prune_stale(self):
        """Remove tracks not seen for STALE_TRACK_S seconds."""
        stale = [tid for tid, s in self._states.items() if s.is_stale()]
        for tid in stale:
            del self._states[tid]

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    @property
    def active_states(self) -> list[TrackState]:
        return list(self._states.values())
