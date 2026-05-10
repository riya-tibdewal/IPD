"""Per-student tracking state and aggregate attention registry."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import config as C
from scoring import StudentTemporalState


@dataclass
class TrackState:
    track_id: int
    temporal: StudentTemporalState = field(default_factory=StudentTemporalState)
    last_seen: float = field(default_factory=time.time)
    last_bbox: Optional[tuple[int, int, int, int]] = None
    last_reasons: list[str] = field(default_factory=list)
    last_factors: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)
    seen_frames: int = 0
    missed_frames: int = 0
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=12))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_keypoints: object = None

    @property
    def stable_label(self) -> str:
        return self.temporal.stable_label

    @property
    def confidence(self) -> float:
        return self.temporal.confidence

    @property
    def attention_score(self) -> float:
        return self.temporal.attention_score

    @property
    def identity(self) -> str:
        return self.temporal.identity or "Unknown"

    def touch(self, bbox: Optional[tuple[int, int, int, int]] = None):
        self.last_seen = time.time()
        self.seen_frames += 1
        self.missed_frames = 0
        if bbox is not None:
            self.last_bbox = bbox
            self.bbox_history.append(bbox)

    def mark_missed(self):
        self.missed_frames += 1

    def smooth_bbox(self) -> Optional[tuple[int, int, int, int]]:
        if not self.bbox_history:
            return self.last_bbox
        n = len(self.bbox_history)
        total_weight = n * (n + 1) / 2
        vals = [0.0, 0.0, 0.0, 0.0]
        for weight, bbox in enumerate(self.bbox_history, start=1):
            for i, value in enumerate(bbox):
                vals[i] += weight * value
        return tuple(int(v / total_weight) for v in vals)

    def is_stale(self, timeout_s: float = C.STALE_TRACK_S) -> bool:
        return (time.time() - self.last_seen) > timeout_s


@dataclass
class StudentScore:
    track_id: int
    frame_counts: Counter = field(default_factory=Counter)
    attention_sum: float = 0.0
    frames: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def record(self, label: str, attention_score: float):
        self.frame_counts[label] += 1
        self.attention_sum += attention_score
        self.frames += 1
        self.last_seen = time.time()

    @property
    def attention_pct(self) -> float:
        return self.attention_sum / self.frames if self.frames else 0.0

    def dominant_behavior(self) -> str:
        return self.frame_counts.most_common(1)[0][0] if self.frame_counts else "UNKNOWN"

    def summary_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "frames": self.frames,
            "attention_pct": round(self.attention_pct, 1),
            "dominant": self.dominant_behavior(),
            "frame_counts": dict(self.frame_counts),
        }


class AttentionRegistry:
    def __init__(self):
        self._scores: dict[int, StudentScore] = {}

    def record(self, track_id: int, label: str, attention_score: float):
        if track_id not in self._scores:
            self._scores[track_id] = StudentScore(track_id=track_id)
        self._scores[track_id].record(label, attention_score)

    def live_pct(self, track_id: int) -> float:
        score = self._scores.get(track_id)
        return score.attention_pct if score else 0.0

    def to_list(self) -> list[dict]:
        return [score.summary_dict() for score in self._scores.values()]

    def print_final_report(self):
        if not self._scores:
            print("\n[EduAware] No students tracked. Empty report.")
            return
        rows = sorted(self._scores.values(), key=lambda s: s.attention_pct, reverse=True)
        print("\n" + "=" * 76)
        print("  EduAware FINAL ATTENTION REPORT")
        print("=" * 76)
        print(f"  {'Student':<10} {'Attention%':>10} {'Frames':>8}  Dominant behavior")
        print("-" * 76)
        for score in rows:
            print(f"  S{score.track_id:02d}       {score.attention_pct:>8.1f}% {score.frames:>8}  {score.dominant_behavior()}")
        avg = sum(score.attention_pct for score in rows) / len(rows)
        print("-" * 76)
        print(f"  CLASS AVERAGE {avg:>18.1f}%")
        print("=" * 76 + "\n")


class StudentTracker:
    def __init__(self):
        self._states: dict[int, TrackState] = {}
        self.registry = AttentionRegistry()

    def get_or_create(self, track_id: int) -> TrackState:
        if track_id not in self._states:
            self._states[track_id] = TrackState(track_id=track_id)
        return self._states[track_id]

    def prune_stale(self):
        for state in self._states.values():
            state.mark_missed()
        stale_ids = [tid for tid, state in self._states.items() if state.is_stale()]
        for tid in stale_ids:
            del self._states[tid]

    @property
    def active_states(self) -> list[TrackState]:
        return list(self._states.values())
