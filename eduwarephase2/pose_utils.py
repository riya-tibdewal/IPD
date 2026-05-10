"""
Pose geometry helpers for classroom behavior analysis.

The system uses YOLOv8/COCO 17-point pose keypoints.  These helpers turn noisy
keypoints into normalized posture metrics that can be compared across camera
distance, crop size, and front/side/rear classroom views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config as C


@dataclass
class PoseMetrics:
    valid: bool = False
    view: str = "unknown"
    scale: float = 1.0
    head_down: float = 0.0
    head_up: float = 0.0
    head_side: float = 0.0
    torso_lean: float = 0.0
    shoulder_tilt: float = 0.0
    shoulder_symmetry: float = 0.0
    hands_near_desk: float = 0.0
    hands_near_head: float = 0.0
    hands_close: float = 0.0
    wrist_motion: float = 0.0
    body_motion: float = 0.0
    motion_energy: float = 0.0
    vertical_extension: float = 0.0
    lower_body_visibility: float = 0.0
    leg_visibility: float = 0.0
    bbox_aspect: float = 0.0
    seated_cue: float = 0.0
    slouch: float = 0.0
    face_visibility: float = 0.0
    back_view: float = 0.0
    keypoints: dict[str, Optional[np.ndarray]] = field(default_factory=dict)


def point(kps: Optional[np.ndarray], idx: int, conf: float = None) -> Optional[np.ndarray]:
    if kps is None or idx >= len(kps):
        return None
    conf = C.KP_CONF if conf is None else conf
    x, y, score = kps[idx]
    if score < conf:
        return None
    return np.array([float(x), float(y)], dtype=np.float32)


def midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is None or b is None:
        return None
    return (a + b) * 0.5


def distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def safe01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def score_range(value: Optional[float], good: float, bad: float, invert: bool = False) -> float:
    """Map a scalar to 0..1 between good and bad thresholds."""
    if value is None:
        return 0.0
    if abs(bad - good) < 1e-6:
        return 0.0
    t = (value - good) / (bad - good)
    if invert:
        t = 1.0 - t
    return safe01(t)


def pack_points(*pts: Optional[np.ndarray]) -> Optional[np.ndarray]:
    visible = [p for p in pts if p is not None]
    if not visible:
        return None
    return np.stack(visible).astype(np.float32)


def mean_motion(cur: Optional[np.ndarray], prev: Optional[np.ndarray], scale: float) -> float:
    if cur is None or prev is None or len(cur) != len(prev) or scale <= 1.0:
        return 0.0
    return float(np.mean(np.linalg.norm(cur - prev, axis=1)) / scale)


def extract_pose_metrics(
    kps: Optional[np.ndarray],
    bbox: tuple[int, int, int, int],
    prev_wrists: Optional[np.ndarray] = None,
    prev_body: Optional[np.ndarray] = None,
) -> tuple[PoseMetrics, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return posture metrics plus current wrist/body point packs for history."""
    metrics = PoseMetrics()
    if kps is None:
        return metrics, None, None

    pts = {
        "nose": point(kps, C.KP_NOSE),
        "l_eye": point(kps, C.KP_L_EYE),
        "r_eye": point(kps, C.KP_R_EYE),
        "l_ear": point(kps, C.KP_L_EAR),
        "r_ear": point(kps, C.KP_R_EAR),
        "l_shoulder": point(kps, C.KP_L_SHOULDER),
        "r_shoulder": point(kps, C.KP_R_SHOULDER),
        "l_elbow": point(kps, C.KP_L_ELBOW),
        "r_elbow": point(kps, C.KP_R_ELBOW),
        "l_wrist": point(kps, C.KP_L_WRIST),
        "r_wrist": point(kps, C.KP_R_WRIST),
        "l_hip": point(kps, C.KP_L_HIP),
        "r_hip": point(kps, C.KP_R_HIP),
        "l_knee": point(kps, C.KP_L_KNEE),
        "r_knee": point(kps, C.KP_R_KNEE),
        "l_ankle": point(kps, C.KP_L_ANKLE),
        "r_ankle": point(kps, C.KP_R_ANKLE),
    }
    metrics.keypoints = pts

    x1, y1, x2, y2 = bbox
    box_h = max(float(y2 - y1), 1.0)
    box_w = max(float(x2 - x1), 1.0)
    metrics.bbox_aspect = box_h / box_w
    shoulder_width = distance(pts["l_shoulder"], pts["r_shoulder"])
    hip_width = distance(pts["l_hip"], pts["r_hip"])
    scale = shoulder_width or hip_width or max(box_w * 0.30, 1.0)
    metrics.scale = max(scale, 1.0)

    sh_mid = midpoint(pts["l_shoulder"], pts["r_shoulder"])
    hip_mid = midpoint(pts["l_hip"], pts["r_hip"])
    eye_mid = midpoint(pts["l_eye"], pts["r_eye"])
    ear_mid = midpoint(pts["l_ear"], pts["r_ear"])
    head = pts["nose"]
    if head is None:
        head = eye_mid
    if head is None:
        head = ear_mid

    face_parts = [pts["nose"], pts["l_eye"], pts["r_eye"], pts["l_ear"], pts["r_ear"]]
    face_vis = sum(p is not None for p in face_parts) / len(face_parts)
    metrics.face_visibility = face_vis
    metrics.back_view = 1.0 if face_vis <= 0.20 and sh_mid is not None else 0.0
    metrics.view = "rear" if metrics.back_view else ("front" if face_vis >= 0.60 else "side")

    if sh_mid is not None and head is not None:
        head_above = (sh_mid[1] - head[1]) / metrics.scale
        metrics.head_down = score_range(head_above, C.HEAD_DOWN_START, C.HEAD_DOWN_STRONG)
        metrics.head_up = score_range(head_above, C.HEAD_UP_START, C.HEAD_UP_STRONG)
        if ear_mid is not None:
            metrics.head_side = score_range(abs(head[0] - ear_mid[0]) / metrics.scale, 0.25, 0.65)
        elif (pts["l_ear"] is None) ^ (pts["r_ear"] is None):
            metrics.head_side = 0.70

    if sh_mid is not None and hip_mid is not None:
        dx = abs(sh_mid[0] - hip_mid[0]) / metrics.scale
        dy = abs(hip_mid[1] - sh_mid[1]) / metrics.scale
        metrics.torso_lean = safe01(dx / max(dy, 0.25))
        metrics.vertical_extension = safe01((hip_mid[1] - sh_mid[1]) / max(box_h, 1.0) * 2.2)
        metrics.slouch = score_range((hip_mid[1] - sh_mid[1]) / metrics.scale, 1.0, 0.35)

    lower_body_points = [
        pts["l_hip"], pts["r_hip"], pts["l_knee"], pts["r_knee"],
        pts["l_ankle"], pts["r_ankle"],
    ]
    leg_points = [pts["l_knee"], pts["r_knee"], pts["l_ankle"], pts["r_ankle"]]
    metrics.lower_body_visibility = sum(p is not None for p in lower_body_points) / len(lower_body_points)
    metrics.leg_visibility = sum(p is not None for p in leg_points) / len(leg_points)
    if hip_mid is not None:
        hip_low_in_box = (hip_mid[1] - y1) / box_h
        metrics.seated_cue = safe01((hip_low_in_box - 0.58) / 0.22)

    if pts["l_shoulder"] is not None and pts["r_shoulder"] is not None:
        shoulder_dy = abs(pts["l_shoulder"][1] - pts["r_shoulder"][1]) / metrics.scale
        metrics.shoulder_tilt = safe01(shoulder_dy)
        metrics.shoulder_symmetry = 1.0 - safe01(shoulder_dy / 0.65)

    wrists = [pts["l_wrist"], pts["r_wrist"]]
    elbows = [pts["l_elbow"], pts["r_elbow"]]
    if sh_mid is not None:
        desk_hits = 0
        head_hits = 0
        visible_wrists = 0
        for wrist in wrists:
            if wrist is None:
                continue
            visible_wrists += 1
            rel_y = (wrist[1] - sh_mid[1]) / metrics.scale
            desk_hits += rel_y >= C.DESK_WRIST_REL_Y
            if head is not None:
                head_hits += (distance(wrist, head) or 999.0) / metrics.scale <= C.HAND_HEAD_DIST
        if visible_wrists:
            metrics.hands_near_desk = desk_hits / visible_wrists
            metrics.hands_near_head = head_hits / visible_wrists
        if wrists[0] is not None and wrists[1] is not None:
            metrics.hands_close = score_range(distance(wrists[0], wrists[1]) / metrics.scale, 0.75, 0.20)

    cur_wrists = pack_points(*wrists)
    cur_body = pack_points(
        pts["nose"], pts["l_shoulder"], pts["r_shoulder"],
        pts["l_elbow"], pts["r_elbow"], pts["l_wrist"], pts["r_wrist"],
        pts["l_hip"], pts["r_hip"]
    )
    metrics.wrist_motion = mean_motion(cur_wrists, prev_wrists, metrics.scale)
    metrics.body_motion = mean_motion(cur_body, prev_body, metrics.scale)
    metrics.motion_energy = safe01(
        0.58 * safe01(metrics.wrist_motion / max(C.WRITING_MOTION_MAX, 1e-6))
        + 0.42 * safe01(metrics.body_motion / max(C.EXCESSIVE_MOTION, 1e-6))
    )
    metrics.valid = sh_mid is not None or head is not None
    return metrics, cur_wrists, cur_body
