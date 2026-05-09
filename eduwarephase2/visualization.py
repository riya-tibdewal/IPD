"""
visualization.py — EduAware Rendering Module
=============================================
All OpenCV drawing is isolated here.
No logic, no heuristics — pure rendering.

This makes it easy to swap out the UI without touching pipeline code.
"""

import cv2
import numpy as np
from typing import Optional

import config as C
from tracker import TrackState


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (local rendering tweaks)
# ══════════════════════════════════════════════════════════════════════════════

_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_SLIM  = cv2.FONT_HERSHEY_SIMPLEX
_KP_COLOR   = (255, 220,  80)    # keypoint dot color
_PILL_ALPHA = 0.72               # label pill background opacity


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_student(
    frame:      np.ndarray,
    state:      TrackState,
    bbox:       tuple,              # (x1, y1, x2, y2)
    kps:        Optional[np.ndarray],
    attn_pct:   float,
    show_pose:  bool  = True,
    show_debug: bool  = False,
    raw_label:  str   = "",
):
    """
    Render one student's complete overlay on the frame (in-place).

    Layers drawn (bottom to top):
      1. Person bounding box
      2. Upper-body skeleton
      3. Keypoint dots
      4. Semi-transparent label pill  "S03 | Writing | 82%"
      5. (optional) raw unsmoothed label for debugging
    """
    x1, y1, x2, y2 = bbox
    label  = state.stable_label
    color  = C.BEHAVIOR_COLORS.get(label, (140, 140, 140))

    # ── 1. Bounding box ───────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # ── 2 + 3. Skeleton & keypoints ───────────────────────────────────────
    if show_pose and kps is not None:
        _draw_skeleton(frame, kps, color)

    # ── 4. Label pill ─────────────────────────────────────────────────────
    conf_pct  = int(state.confidence * 100)
    pill_text = f"S{state.track_id:02d} | {label} | {attn_pct:.0f}%"
    _draw_pill(frame, pill_text, x1, y1, color)

    # ── 5. Debug: raw (pre-smoothing) label ───────────────────────────────
    if show_debug and raw_label and raw_label != label:
        cv2.putText(
            frame, f"raw: {raw_label}",
            (x1, y2 + 16), _FONT_SLIM, 0.40,
            (200, 200, 80), 1, cv2.LINE_AA,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HUD OVERLAY
# ══════════════════════════════════════════════════════════════════════════════

def draw_hud(
    frame:       np.ndarray,
    fps:         float,
    n_students:  int,
    yolo_conf:   float,
    show_pose:   bool,
    paused:      bool,
    summary:     dict,    # behavior → count
    class_avg:   float,   # class-level attention average
):
    """
    Render the global heads-up display:
      • Top-right: FPS + student count
      • Top-left:  behavior summary + class attention average
      • Bottom:    keyboard legend
      • Centre:    PAUSED banner (when paused)
    """
    H, W = frame.shape[:2]

    # ── Top-right info bar ────────────────────────────────────────────────
    top_right = f"FPS {fps:.1f}  |  Students: {n_students}"
    (tw, _), _ = cv2.getTextSize(top_right, _FONT_SLIM, 0.65, 2)
    cv2.putText(frame, top_right, (W - tw - 10, 30),
                _FONT_SLIM, 0.65, (255, 220, 0), 2, cv2.LINE_AA)

    # ── Top-left: live behavior summary ───────────────────────────────────
    y = 28
    cv2.putText(frame, "Behavior Summary", (10, y),
                _FONT_SLIM, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    y += 20
    for lbl, cnt in sorted(summary.items()):
        clr = C.BEHAVIOR_COLORS.get(lbl, (200, 200, 200))
        cv2.putText(frame, f"  {lbl}: {cnt}",
                    (10, y), _FONT_SLIM, 0.45, clr, 1, cv2.LINE_AA)
        y += 17

    # Class attention average
    y += 4
    avg_color = (50, 210, 50) if class_avg >= 70 else (0, 140, 255)
    cv2.putText(frame, f"Class Attn: {class_avg:.1f}%",
                (10, y), _FONT_SLIM, 0.55, avg_color, 1, cv2.LINE_AA)

    # ── Bottom legend ─────────────────────────────────────────────────────
    pose_str  = "P=pose:ON" if show_pose else "P=pose:OFF"
    legend    = (f"Q=quit  {pose_str}  D=debug  S=screenshot  "
                 f"+/-=conf({yolo_conf:.2f})  SPACE=pause")
    cv2.putText(frame, legend, (10, H - 10),
                _FONT_SLIM, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Paused banner ─────────────────────────────────────────────────────
    if paused:
        cv2.putText(frame, "PAUSED",
                    (W // 2 - 80, H // 2),
                    _FONT, 1.6, (0, 200, 255), 3, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_pill(
    frame: np.ndarray,
    text:  str,
    x1:    int,
    y1:    int,
    color: tuple,
):
    """
    Draw a semi-transparent colored pill above the bounding box.
    The pill background improves readability on any background color.
    """
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.52, 1)
    pad     = 6
    px1, py1 = x1, max(y1 - th - pad * 2, 0)
    px2, py2 = x1 + tw + pad * 2, y1

    # Semi-transparent fill using addWeighted
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), color, -1)
    cv2.addWeighted(overlay, _PILL_ALPHA, frame, 1 - _PILL_ALPHA, 0, frame)

    # White text on top
    cv2.putText(frame, text,
                (px1 + pad, py2 - pad),
                _FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_skeleton(frame: np.ndarray, kps: np.ndarray, color: tuple):
    """Draw upper-body skeleton edges and keypoint circles."""
    # Edges
    for (a, b) in C.UPPER_SKELETON_EDGES:
        if a >= len(kps) or b >= len(kps):
            continue
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca >= C.KP_CONF and cb >= C.KP_CONF:
            cv2.line(frame,
                     (int(xa), int(ya)), (int(xb), int(yb)),
                     color, 2, cv2.LINE_AA)

    # Keypoint dots (upper body only: indices 0–10)
    for x, y, c in kps[:11]:
        if c >= C.KP_CONF:
            cv2.circle(frame, (int(x), int(y)), 4, _KP_COLOR, -1)
