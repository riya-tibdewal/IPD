"""
behavior_analysis.py — EduAware Behavior Heuristics
=====================================================
Classifies student behavior from a single (17, 3) COCO keypoint array.
No neural network required — pure NumPy geometry, <1 ms per person.

All distances are normalized by shoulder width so thresholds are
scale-invariant (work at 2 m or 10 m camera distance).

Heuristic rationale
───────────────────
Behavior          | Geometric cue              | Why it's reliable
──────────────────────────────────────────────────────────────────────────────
Attentive         | Both ears/eyes visible,    | Symmetric facial landmark
                  | nose between ear midpoint  | visibility = frontal face
──────────────────────────────────────────────────────────────────────────────
Looking Up        | Nose Y >> shoulder Y       | Head tilted back raises nose
                  |                            | far above shoulder plane
──────────────────────────────────────────────────────────────────────────────
Distracted (Side) | One ear visible or         | Side-turn occludes one ear
                  | large nose-to-ear offset   |
──────────────────────────────────────────────────────────────────────────────
Talking           | Sustained side-turn        | A glance is brief; talking to
                  | (>50% of recent frames)    | a neighbor is held for seconds
──────────────────────────────────────────────────────────────────────────────
Using Phone       | Wrist elevated to face     | Phone = wrist at face height
                  | AND head pitched down      | + head tilt toward screen
──────────────────────────────────────────────────────────────────────────────
Writing           | Head down + wrist in desk  | Writing = sustained low-motion
                  | zone + low wrist motion    | wrist near desk surface
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from typing import Optional
from collections import deque

import config as C


# ══════════════════════════════════════════════════════════════════════════════
# KEYPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _xy(kps: np.ndarray, idx: int) -> Optional[np.ndarray]:
    """Return (x, y) if confidence ≥ KP_CONF, else None."""
    if kps is None or idx >= len(kps):
        return None
    x, y, conf = kps[idx]
    return np.array([float(x), float(y)]) if conf >= C.KP_CONF else None


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) * 0.5


def _dist(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def _norm_dist(a, b, ref: Optional[float]) -> Optional[float]:
    """Euclidean distance normalized by a reference length."""
    d = _dist(a, b)
    if d is None or ref is None or ref < 1.0:
        return None
    return d / ref


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def classify(
    kps:          np.ndarray,
    recent_labels: deque,
    prev_wrists:  Optional[np.ndarray],
) -> tuple[str, Optional[np.ndarray]]:
    """
    Classify behavior from one frame's keypoints.

    Args:
        kps:           (17, 3) keypoint array from YOLOv8
        recent_labels: deque of recent raw labels (for Talking detection)
        prev_wrists:   previous frame's wrist positions for motion estimate

    Returns:
        (behavior_label, current_wrists)
        current_wrists is (2, 2) float32 or None — stored for next frame.
    """
    # ── Extract relevant landmarks ────────────────────────────────────────
    nose        = _xy(kps, C.KP_NOSE)
    l_eye       = _xy(kps, C.KP_L_EYE)
    r_eye       = _xy(kps, C.KP_R_EYE)
    l_ear       = _xy(kps, C.KP_L_EAR)
    r_ear       = _xy(kps, C.KP_R_EAR)
    l_shoulder  = _xy(kps, C.KP_L_SHOULDER)
    r_shoulder  = _xy(kps, C.KP_R_SHOULDER)
    l_wrist     = _xy(kps, C.KP_L_WRIST)
    r_wrist     = _xy(kps, C.KP_R_WRIST)
    l_hip       = _xy(kps, C.KP_L_HIP)
    r_hip       = _xy(kps, C.KP_R_HIP)

    if nose is None:
        return "Unknown", _pack_wrists(l_wrist, r_wrist)

    # ── Reference scale ───────────────────────────────────────────────────
    # Shoulder width normalizes all distance thresholds for distance/zoom.
    sh_width   = _dist(l_shoulder, r_shoulder)         # None if missing
    sh_mid     = _mid(l_shoulder, r_shoulder)

    # ── Head PITCH (down / up) ────────────────────────────────────────────
    # Measured as: how far is the nose above the shoulder midpoint,
    # in units of shoulder width.
    # Frontal:   nose_above ≈ 1.2–1.8  (nose comfortably above shoulders)
    # Head down: nose_above < 0.4      (nose near / below shoulder level)
    # Head up:   nose_above > 2.2      (nose far above shoulders)
    head_down = False
    head_up   = False
    if sh_mid is not None and sh_width is not None and sh_width > 1.0:
        nose_above = (sh_mid[1] - nose[1]) / sh_width   # positive = nose above
        head_down  = nose_above < 0.40
        head_up    = nose_above > 2.20

    # ── Wrist elevation (relative to shoulders) ───────────────────────────
    # wrist_rel_y < -0.5 → wrist at or above shoulder (= phone-height)
    # wrist_rel_y >  1.0 → wrist below shoulder mid  (= desk-height)
    l_wrist_high = r_wrist_high = False
    l_wrist_low  = r_wrist_low  = False
    if sh_mid is not None and sh_width is not None and sh_width > 1.0:
        for wrist, hi_flag, lo_flag in [
            (l_wrist, "l_wrist_high", "l_wrist_low"),
            (r_wrist, "r_wrist_high", "r_wrist_low"),
        ]:
            if wrist is not None:
                wrel = (wrist[1] - sh_mid[1]) / sh_width
                if wrel < -0.50:
                    if hi_flag == "l_wrist_high": l_wrist_high = True
                    else: r_wrist_high = True
                elif wrel > 1.00:
                    if lo_flag == "l_wrist_low": l_wrist_low = True
                    else: r_wrist_low = True

    # ── Wrist motion (writing vs phone discrimination) ────────────────────
    cur_wrists = _pack_wrists(l_wrist, r_wrist)
    wrist_motion = 0.0
    if cur_wrists is not None and prev_wrists is not None:
        wrist_motion = float(np.mean(
            np.linalg.norm(cur_wrists - prev_wrists, axis=1)
        ))

    # ── YAW proxy ─────────────────────────────────────────────────────────
    both_ears_vis = l_ear is not None and r_ear is not None
    one_ear_only  = (l_ear is not None) ^ (r_ear is not None)

    strong_yaw = False
    if both_ears_vis and sh_width is not None and sh_width > 1.0:
        ear_mid    = _mid(l_ear, r_ear)
        if ear_mid is not None:
            yaw_off  = abs(nose[0] - ear_mid[0]) / sh_width
            strong_yaw = yaw_off > 0.30

    # ══════════════════════════════════════════════════════════════════════
    # PRIORITY DECISION TREE
    # ══════════════════════════════════════════════════════════════════════

    # 1. Phone — wrist at face level AND head pitched down
    if head_down and (l_wrist_high or r_wrist_high):
        return "Using Phone", cur_wrists

    # 2. Writing — head down, wrist at desk level, low motion (<8 px/frame)
    if head_down and (l_wrist_low or r_wrist_low) and wrist_motion < 8.0:
        return "Writing", cur_wrists

    # 3. Looking Up — nose very far above shoulders
    if head_up:
        return "Looking Up", cur_wrists

    # 4. Talking — *sustained* lateral turn (held >50 % of recent window)
    if one_ear_only:
        n = len(recent_labels)
        if n > 0:
            side_count = sum(
                1 for lbl in recent_labels
                if lbl in ("Talking", "Distracted (Side)")
            )
            if side_count / n > 0.50:
                return "Talking", cur_wrists
        return "Distracted (Side)", cur_wrists

    if strong_yaw:
        return "Distracted (Side)", cur_wrists

    # 5. Attentive — symmetric frontal head pose
    if both_ears_vis or (l_eye is not None and r_eye is not None):
        return "Attentive", cur_wrists

    return "Unknown", cur_wrists


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pack_wrists(
    l: Optional[np.ndarray],
    r: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Pack left & right wrist into (2, 2) float32 array, or None."""
    if l is None and r is None:
        return None
    lv = l if l is not None else np.zeros(2, dtype=np.float32)
    rv = r if r is not None else np.zeros(2, dtype=np.float32)
    return np.stack([lv, rv]).astype(np.float32)
