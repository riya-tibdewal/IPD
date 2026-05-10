"""OpenCV overlays for students, skeletons, diagnostics, and HUD."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import config as C


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
KP_COLOR = (255, 230, 90)


def draw_student(
    frame: np.ndarray,
    state,
    bbox: tuple[int, int, int, int],
    kps: Optional[np.ndarray],
    show_pose: bool = True,
    show_debug: bool = False,
    raw_label: str = "",
):
    x1, y1, x2, y2 = bbox
    label = state.stable_label
    color = C.BEHAVIOR_COLORS.get(label, C.BEHAVIOR_COLORS["UNKNOWN"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if show_pose and kps is not None:
        _draw_skeleton(frame, kps, color)

    name = state.identity if state.identity != "Unknown" else f"ID {state.track_id}"
    text = f"{name} | {label} | attn {state.attention_score:.0f}% | conf {state.confidence * 100:.0f}%"
    _draw_label(frame, text, x1, y1, color)

    if show_debug:
        reason = ", ".join(state.last_reasons[:3])
        debug = f"raw={raw_label or label} {reason}"
        cv2.putText(frame, debug, (x1, min(y2 + 18, frame.shape[0] - 8)), FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    n_students: int,
    yolo_conf: float,
    show_pose: bool,
    paused: bool,
    summary: dict,
    class_avg: float,
    diagnostics=None,
    perf: Optional[dict] = None,
):
    h, w = frame.shape[:2]
    cv2.putText(frame, f"FPS {fps:.1f} | Students {n_students}", (w - 260, 28), FONT, 0.62, (255, 230, 60), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Class attention {class_avg:.1f}%", (10, 28), FONT, 0.58, _attention_color(class_avg), 2, cv2.LINE_AA)

    y = 52
    for label, count in sorted(summary.items()):
        color = C.BEHAVIOR_COLORS.get(label, C.BEHAVIOR_COLORS["UNKNOWN"])
        cv2.putText(frame, f"{label}: {count}", (10, y), FONT, 0.45, color, 1, cv2.LINE_AA)
        y += 18

    if diagnostics and y < h - 80:
        cv2.putText(frame, f"{diagnostics.backend} {diagnostics.width}x{diagnostics.height}", (10, y + 6), FONT, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
        y += 22

    if perf and y < h - 95:
        perf_text = (
            f"infer {perf.get('infer_ms', 0.0):.0f}ms | obj {perf.get('object_ms', 0.0):.0f}ms | "
            f"draw {perf.get('render_ms', 0.0):.0f}ms | drop {int(perf.get('dropped', 0))}"
        )
        cv2.putText(frame, perf_text, (10, y + 6), FONT, 0.42, (180, 220, 220), 1, cv2.LINE_AA)

    legend = f"Q quit | P pose {'on' if show_pose else 'off'} | D debug | S screenshot | +/- conf {yolo_conf:.2f} | Space pause"
    cv2.putText(frame, legend, (10, h - 12), FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 90, h // 2), FONT_BOLD, 1.5, (0, 220, 255), 3, cv2.LINE_AA)


def _draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.48, 1)
    pad = 6
    h, w = frame.shape[:2]
    x = max(0, min(x, w - tw - pad * 2 - 1))
    y_top = y - th - pad * 2
    if y_top < 2:
        y_top = min(y + 2, h - th - pad * 2 - 1)
        y_text = y_top + th + pad
    else:
        y_text = y - pad
    x2 = min(x + tw + pad * 2, w - 1)
    y2 = min(y_top + th + pad * 2, h - 1)
    cv2.rectangle(frame, (x, y_top), (x2, y2), color, -1)
    cv2.putText(frame, text, (x + pad, y_text), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_skeleton(frame: np.ndarray, kps: np.ndarray, color: tuple[int, int, int]):
    for a, b in C.UPPER_SKELETON_EDGES:
        if a >= len(kps) or b >= len(kps):
            continue
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca >= C.KP_CONF and cb >= C.KP_CONF:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA)
    for x, y, conf in kps[:13]:
        if conf >= C.KP_CONF:
            cv2.circle(frame, (int(x), int(y)), 3, KP_COLOR, -1)


def _attention_color(value: float):
    if value >= 70:
        return (60, 220, 60)
    if value >= 45:
        return (0, 210, 255)
    return (0, 90, 255)
