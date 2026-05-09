"""
config.py — EduAware v2 Global Configuration
=============================================
All tunable constants live here.
Import this module everywhere instead of hard-coding magic numbers.
"""

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO / CAPTURE
# ══════════════════════════════════════════════════════════════════════════════

CAPTURE_WIDTH   = 1280
CAPTURE_HEIGHT  = 720
THREADED_CAP    = True      # background thread for camera reads

# ══════════════════════════════════════════════════════════════════════════════
# YOLO
# ══════════════════════════════════════════════════════════════════════════════

YOLO_MODEL      = "yolov8n-pose.pt"   # n=fastest  s=balanced  m=accurate
YOLO_IMGSZ      = 480                 # inference input size (lower = faster)
YOLO_CONF       = 0.45                # person detection confidence floor
YOLO_IOU        = 0.45                # NMS IoU threshold
USE_GPU         = False               # True → CUDA device 0

# ══════════════════════════════════════════════════════════════════════════════
# TRACKING (ByteTrack via YOLOv8 built-in)
# ══════════════════════════════════════════════════════════════════════════════

TRACK_PERSIST   = True                # keep tracker state between calls

# ══════════════════════════════════════════════════════════════════════════════
# KEYPOINT THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

KP_CONF         = 0.35                # minimum per-keypoint confidence

# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

SMOOTH_BUF_LEN  = 45                  # rolling window size (frames)
SMOOTH_THRESH   = 0.70                # majority fraction to flip stable label

# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

FRAME_SKIP      = 2                   # process 1-in-N frames (1 = every frame)
STALE_TRACK_S   = 5.0                 # seconds before a lost track is removed

# ══════════════════════════════════════════════════════════════════════════════
# ATTENTION SCORING
# ══════════════════════════════════════════════════════════════════════════════

# Behaviors that count as "attentive" for the percentage calculation
ATTENTIVE_BEHAVIORS = {"Attentive", "Writing"}

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

WINDOW_NAME     = "EduAware v2 — Classroom Behavior Analytics"
WINDOW_W        = 1280
WINDOW_H        = 720

# Behavior → BGR color
BEHAVIOR_COLORS = {
    "Attentive":         (50,  210,  50),
    "Writing":           (255, 180,   0),
    "Using Phone":       (0,   60,  220),
    "Talking":           (200,  80, 200),
    "Looking Up":        (0,  200, 255),
    "Distracted (Side)": (0,  120, 255),
    "Unknown":           (140, 140, 140),
}

# ══════════════════════════════════════════════════════════════════════════════
# COCO KEYPOINT INDICES  (YOLOv8-pose 17-point layout)
# ══════════════════════════════════════════════════════════════════════════════

KP_NOSE       = 0
KP_L_EYE      = 1;  KP_R_EYE      = 2
KP_L_EAR      = 3;  KP_R_EAR      = 4
KP_L_SHOULDER = 5;  KP_R_SHOULDER = 6
KP_L_ELBOW    = 7;  KP_R_ELBOW    = 8
KP_L_WRIST    = 9;  KP_R_WRIST    = 10
KP_L_HIP      = 11; KP_R_HIP      = 12
KP_L_KNEE     = 13; KP_R_KNEE     = 14
KP_L_ANKLE    = 15; KP_R_ANKLE    = 16

# Upper-body skeleton edges (drawn per student)
UPPER_SKELETON_EDGES = [
    (KP_NOSE, KP_L_EYE),       (KP_NOSE, KP_R_EYE),
    (KP_L_EYE, KP_L_EAR),      (KP_R_EYE, KP_R_EAR),
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW),(KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW),(KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP),  (KP_R_SHOULDER, KP_R_HIP),
]
