"""
attention_scoring.py — EduAware Attention Scoring
===================================================
Tracks per-student frame counts for every behavior label and computes:

  Attention % = (Attentive frames + Writing frames) / Total frames × 100

Why include Writing?
  Writing requires focused engagement with course material.
  A student who is writing is actively processing content — this is a
  productive behavior, so it contributes to the attention score.

Why exclude Looking Up / Phone / Talking / Distracted?
  These behaviors indicate disengagement from the primary lecture content.

Live display: shows a running percentage, updated every processed frame.
Final report: printed to console + returned as a dict at video end.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import config as C


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StudentScore:
    """
    Accumulates per-behavior frame counts for one tracked student.

    frame_counts:  { behavior_label: n_frames }
    first_seen:    Unix timestamp when this student was first detected
    last_seen:     Unix timestamp of the most recent processed frame
    """
    track_id:     int
    frame_counts: dict = field(default_factory=dict)
    first_seen:   float = field(default_factory=time.time)
    last_seen:    float = field(default_factory=time.time)

    # ──────────────────────────────────────────────────────────────────────
    # Mutation
    # ──────────────────────────────────────────────────────────────────────

    def record(self, behavior: str):
        """Increment the counter for one behavior label."""
        self.frame_counts[behavior] = self.frame_counts.get(behavior, 0) + 1
        self.last_seen = time.time()

    # ──────────────────────────────────────────────────────────────────────
    # Derived metrics
    # ──────────────────────────────────────────────────────────────────────

    @property
    def total_frames(self) -> int:
        return sum(self.frame_counts.values())

    @property
    def attentive_frames(self) -> int:
        """Frames where the student was actively engaged (attentive or writing)."""
        return sum(
            self.frame_counts.get(b, 0)
            for b in C.ATTENTIVE_BEHAVIORS
        )

    @property
    def attention_pct(self) -> float:
        """Live attention percentage (0–100)."""
        total = self.total_frames
        if total == 0:
            return 0.0
        return (self.attentive_frames / total) * 100.0

    def dominant_behavior(self) -> str:
        """The behavior seen most often overall."""
        if not self.frame_counts:
            return "Unknown"
        return max(self.frame_counts, key=self.frame_counts.get)

    def summary_dict(self) -> dict:
        """Return a serialisable summary for final reporting."""
        return {
            "track_id":       self.track_id,
            "total_frames":   self.total_frames,
            "attentive_pct":  round(self.attention_pct, 1),
            "frame_counts":   dict(self.frame_counts),
            "dominant":       self.dominant_behavior(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class AttentionRegistry:
    """
    Holds a StudentScore for every tracked student.
    Created once at pipeline start; persists until video ends.
    """

    def __init__(self):
        self._scores: dict[int, StudentScore] = {}

    def record(self, track_id: int, stable_label: str):
        """
        Record one stable-label frame for a student.
        Creates a new StudentScore on first encounter.
        """
        if track_id not in self._scores:
            self._scores[track_id] = StudentScore(track_id=track_id)
        self._scores[track_id].record(stable_label)

    def get(self, track_id: int) -> Optional[StudentScore]:
        return self._scores.get(track_id)

    def live_pct(self, track_id: int) -> float:
        sc = self._scores.get(track_id)
        return sc.attention_pct if sc else 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Final report
    # ──────────────────────────────────────────────────────────────────────

    def print_final_report(self):
        """
        Print a formatted final attention table to stdout.
        Call this when the video / session ends.
        """
        if not self._scores:
            print("\n[EduAware] No students tracked — empty report.")
            return

        print("\n" + "═" * 65)
        print("  EduAware — FINAL ATTENTION REPORT")
        print("═" * 65)
        print(f"  {'Student':<12} {'Attention%':>11} {'Attentive':>10} "
              f"{'Total':>7}  {'Dominant Behavior'}")
        print("─" * 65)

        sorted_scores = sorted(
            self._scores.values(),
            key=lambda s: s.attention_pct,
            reverse=True,
        )
        for sc in sorted_scores:
            print(
                f"  S{sc.track_id:02d}         "
                f"  {sc.attention_pct:>8.1f}%"
                f"  {sc.attentive_frames:>9}"
                f"  {sc.total_frames:>6}"
                f"  {sc.dominant_behavior()}"
            )

        # Class average
        avg = sum(s.attention_pct for s in sorted_scores) / len(sorted_scores)
        print("─" * 65)
        print(f"  CLASS AVERAGE               {avg:>8.1f}%")
        print("═" * 65 + "\n")

    def to_list(self) -> list[dict]:
        """Return all summaries as a list of dicts (for export / JSON)."""
        return [s.summary_dict() for s in self._scores.values()]
