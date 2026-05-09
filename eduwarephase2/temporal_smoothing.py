"""
temporal_smoothing.py — EduAware Temporal Smoothing
=====================================================
Single-frame behavior predictions are noisy because keypoint confidence
oscillates due to motion blur, partial occlusion, and codec artifacts.

Solution: a rolling majority-vote buffer per student.
Each StudentBuffer stores the last N raw predictions in a deque.
The stable label only changes when one behavior dominates ≥ SMOOTH_THRESH
fraction of the window.  This eliminates label flickering completely while
still reacting to genuine behavior changes in ~0.5–1.5 seconds.

How it works
────────────
Frames:  Writing Writing Attentive Writing Writing
Counts:  Writing=4, Attentive=1
Fraction: Writing = 4/5 = 0.80  ≥ 0.70 → stable = "Writing"

If no behavior clears the threshold (e.g. 50 / 50 split), the previous
stable label is preserved — never showing a misleading new label.
"""

from collections import deque, Counter
from config import SMOOTH_BUF_LEN, SMOOTH_THRESH


class StudentBuffer:
    """
    Rolling prediction buffer for one tracked student.

    Args:
        buf_len:   Length of the rolling window (default from config)
        threshold: Fraction required for a label to become stable
    """

    def __init__(
        self,
        buf_len:   int   = SMOOTH_BUF_LEN,
        threshold: float = SMOOTH_THRESH,
    ):
        self._buf       = deque(maxlen=buf_len)
        self._threshold = threshold
        self.stable     = "Unknown"   # currently displayed label

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def push(self, raw_label: str) -> str:
        """
        Add a new raw frame prediction and recompute the stable label.

        Returns:
            The current stable label (possibly unchanged).
        """
        self._buf.append(raw_label)
        self._recompute()
        return self.stable

    @property
    def confidence(self) -> float:
        """
        Fraction of recent frames that agree with the current stable label.
        Used for the confidence indicator in the HUD.
        """
        if not self._buf:
            return 0.0
        return Counter(self._buf)[self.stable] / len(self._buf)

    @property
    def frame_count(self) -> int:
        """Number of predictions currently in the buffer."""
        return len(self._buf)

    def reset(self):
        """Clear buffer and reset stable label (e.g. after track loss)."""
        self._buf.clear()
        self.stable = "Unknown"

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _recompute(self):
        if not self._buf:
            return
        top_label, top_count = Counter(self._buf).most_common(1)[0]
        fraction = top_count / len(self._buf)
        if fraction >= self._threshold:
            self.stable = top_label
        # else: keep previous stable label — no flip on ambiguous evidence
