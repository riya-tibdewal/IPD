"""
yolo_face_pipeline.py — EduAware Full Pipeline
================================================
Integrates:
  • YOLOv8-pose  → detects persons + body keypoints (pose)
  • RetinaFace   → detects faces within person crops
  • ArcFace      → extracts face embeddings for recognition
  • FaceDatabase → matches embeddings to known identities

Flow per frame:
    Camera → YOLO (detect persons + poses) → For each person bbox:
        → Crop upper body → RetinaFace (detect face) → ArcFace (embed)
        → DB lookup → Label on screen

Two-stage detection is more reliable than face-only detection in classrooms:
  YOLO handles occlusion/distance, RetinaFace handles precise face alignment.

Controls:
    Q       — quit
    P       — toggle pose skeleton overlay
    F       — toggle face-only mode (skip YOLO, run face detection directly)
    R       — reload face database
    S       — save screenshot
    +/-     — increase/decrease recognition threshold

Usage:
    python yolo_face_pipeline.py
    python yolo_face_pipeline.py --cam 0 --gpu
    python yolo_face_pipeline.py --yolo-model yolov8n-pose.pt   # nano (fastest)
    python yolo_face_pipeline.py --yolo-model yolov8m-pose.pt   # medium (balanced)
"""

import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO

from face_db import FaceDatabase
from face_recognizer import FaceRecognizer, FaceResult


# ──────────────────────────────────────────────────────
# Pose skeleton definition (COCO 17-keypoint format)
# ──────────────────────────────────────────────────────
SKELETON_EDGES = [
    (0,  1), (0,  2),                    # nose → eyes
    (1,  3), (2,  4),                    # eyes → ears
    (5,  6),                             # shoulders
    (5,  7), (7,  9),                    # L arm
    (6,  8), (8, 10),                    # R arm
    (5, 11), (6, 12),                    # torso
    (11, 12),                            # hips
    (11, 13), (13, 15),                  # L leg
    (12, 14), (14, 16),                  # R leg
]
SKELETON_COLOR = (0, 200, 255)
KP_COLOR       = (255, 200, 0)


# ──────────────────────────────────────────────────────
# Attention heuristic (head pose proxy)
# ──────────────────────────────────────────────────────

def estimate_attention(keypoints: np.ndarray) -> str:
    """
    Very lightweight proxy for attention using YOLO keypoints.
    No extra models needed — just geometry.

    Keypoints: shape (17, 3) — (x, y, confidence)
    Returns: "Attentive" | "Distracted" | "Unknown"
    """
    if keypoints is None or len(keypoints) < 7:
        return "Unknown"

    # Nose (0), left-eye (1), right-eye (2), left-ear (3), right-ear (4)
    nose, l_eye, r_eye, l_ear, r_ear = [keypoints[i] for i in range(5)]

    # Use confidence threshold
    def visible(kp, thresh=0.4):
        return kp[2] > thresh

    if not visible(nose):
        return "Unknown"

    # If both ears visible and nose is between them → facing forward (attentive)
    if visible(l_ear) and visible(r_ear):
        ear_mid_x = (l_ear[0] + r_ear[0]) / 2
        nose_x    = nose[0]
        # Nose should be close to the midpoint of ears for frontal pose
        ear_width = abs(l_ear[0] - r_ear[0])
        if abs(nose_x - ear_mid_x) < ear_width * 0.25:
            return "Attentive"
        return "Distracted"

    # Only one ear visible → turned to side
    if visible(l_ear) and not visible(r_ear):
        return "Distracted"
    if visible(r_ear) and not visible(l_ear):
        return "Distracted"

    # Both eyes visible, no ears → could be forward facing
    if visible(l_eye) and visible(r_eye):
        return "Attentive"

    return "Unknown"


ATTENTION_COLORS = {
    "Attentive":  (0, 220, 0),
    "Distracted": (0, 100, 255),
    "Unknown":    (160, 160, 160),
}


# ──────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────

class EduAwarePipeline:
    def __init__(
        self,
        db:          FaceDatabase,
        rec:         FaceRecognizer,
        yolo_model:  str   = "yolov8n-pose.pt",
        face_thresh: float = 0.45,
        use_gpu:     bool  = False,
    ):
        self.db         = db
        self.rec        = rec
        self.face_thresh = face_thresh

        device = "0" if use_gpu else "cpu"
        print(f"[Pipeline] Loading YOLO model '{yolo_model}' on {device} …")
        self.yolo = YOLO(yolo_model)
        print("[Pipeline] YOLO ready ✓")

    def process_frame(
        self,
        frame: np.ndarray,
        draw_pose:      bool = True,
        face_only_mode: bool = False,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Run the full pipeline on one frame.

        Args:
            frame:          BGR OpenCV frame
            draw_pose:      Whether to draw skeleton overlays
            face_only_mode: Skip YOLO, run face detection directly on full frame

        Returns:
            (annotated_frame, list of result dicts per person)
        """
        out     = frame.copy()
        persons = []

        if face_only_mode:
            # ── Face-only mode ─────────────────────────
            face_results = self.rec.recognize(frame)
            for r in face_results:
                persons.append({
                    "face":      r,
                    "attention": "Unknown",
                    "keypoints": None,
                })
            out = self.rec.draw_results(frame, face_results)

        else:
            # ── YOLO → face pipeline ───────────────────
            yolo_results = self.yolo(frame, verbose=False, classes=[0])  # class 0 = person

            for result in yolo_results:
                boxes     = result.boxes
                keypoints = result.keypoints

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf           = float(box.conf[0])
                    if conf < 0.4:
                        continue

                    # ── Keypoints / attention ───────────
                    kps       = None
                    attention = "Unknown"
                    if keypoints is not None and i < len(keypoints.data):
                        kps       = keypoints.data[i].cpu().numpy()  # (17, 3)
                        attention = estimate_attention(kps)

                    # ── Draw YOLO person box ─────────────
                    att_color = ATTENTION_COLORS[attention]
                    cv2.rectangle(out, (x1, y1), (x2, y2), att_color, 2)
                    cv2.putText(
                        out, attention,
                        (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, att_color, 1,
                    )

                    # ── Draw skeleton ────────────────────
                    if draw_pose and kps is not None:
                        self._draw_skeleton(out, kps)

                    # ── Crop face region (upper 40% of person bbox) ──
                    face_h   = int((y2 - y1) * 0.45)
                    face_y2  = min(y1 + face_h, frame.shape[0])
                    face_crop = frame[y1:face_y2, x1:x2]

                    face_result = None
                    if face_crop.size > 0:
                        face_results = self.rec.recognize(face_crop)
                        if face_results:
                            face_result = face_results[0]
                            # Translate bbox back to full-frame coords
                            fr = face_result
                            abs_x1 = x1 + fr.x1
                            abs_y1 = y1 + fr.y1
                            abs_x2 = x1 + fr.x2
                            abs_y2 = y1 + fr.y2

                            # Draw face box + label
                            face_color = (0, 200, 0) if fr.is_known else (0, 0, 220)
                            cv2.rectangle(out, (abs_x1, abs_y1), (abs_x2, abs_y2), face_color, 2)

                            label = fr.label
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                            cv2.rectangle(
                                out,
                                (abs_x1, abs_y1 - th - 6),
                                (abs_x1 + tw + 4, abs_y1),
                                face_color, -1,
                            )
                            cv2.putText(
                                out, label,
                                (abs_x1 + 2, abs_y1 - 3),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA,
                            )

                    persons.append({
                        "face":      face_result,
                        "attention": attention,
                        "keypoints": kps,
                        "bbox":      (x1, y1, x2, y2),
                    })

        # ── HUD ──────────────────────────────────────
        mode_str = "Face-Only Mode" if face_only_mode else "YOLO+Face Mode"
        cv2.putText(
            out, f"Persons: {len(persons)}  [{mode_str}]",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )

        return out, persons

    def _draw_skeleton(self, frame: np.ndarray, kps: np.ndarray):
        """Draw COCO pose skeleton from 17 keypoints."""
        for (a, b) in SKELETON_EDGES:
            if a >= len(kps) or b >= len(kps):
                continue
            xa, ya, ca = kps[a]
            xb, yb, cb = kps[b]
            if ca > 0.4 and cb > 0.4:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), SKELETON_COLOR, 2)

        for kp in kps:
            x, y, c = kp
            if c > 0.4:
                cv2.circle(frame, (int(x), int(y)), 4, KP_COLOR, -1)


# ──────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────

def run(
    cam_id:     int   = 0,
    yolo_model: str   = "yolov8n-pose.pt",
    threshold:  float = 0.45,
    use_gpu:    bool  = False,
    db_path:    str   = "face_database.pkl",
):
    db  = FaceDatabase(db_path)
    rec = FaceRecognizer(db, rec_thresh=threshold, use_gpu=use_gpu)
    pipeline = EduAwarePipeline(db, rec, yolo_model=yolo_model, use_gpu=use_gpu)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera {cam_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[Pipeline] Running. Controls:")
    print("  Q=quit  P=toggle pose  F=toggle face-only  R=reload DB  S=screenshot  +/-=threshold\n")

    show_pose  = True
    face_only  = False
    prev_time  = time.time()
    frame_cnt  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(cam_id, str):
                print("[Info] Video processing complete.")
            else:
                print("[Error] Failed to read from camera.")
            break

        out, persons = pipeline.process_frame(frame, draw_pose=show_pose, face_only_mode=face_only)

        # FPS
        frame_cnt += 1
        fps = frame_cnt / (time.time() - prev_time + 1e-6)
        cv2.putText(
            out, f"FPS: {fps:.1f}",
            (out.shape[1] - 130, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )

        # Controls legend
        cv2.putText(
            out,
            f"Q=quit  P=pose  F=face-only  R=reload  S=shot  +/-=thresh({threshold:.2f})",
            (10, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
        )

        cv2.imshow("EduAware — YOLO + Face Recognition", out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            show_pose = not show_pose
        elif key == ord("f"):
            face_only = not face_only
            print(f"[Pipeline] Face-only mode: {face_only}")
        elif key == ord("r"):
            db._load()
            print("[Pipeline] Database reloaded.")
        elif key == ord("s"):
            from datetime import datetime
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{ts}.jpg"
            cv2.imwrite(path, out)
            print(f"[Pipeline] Screenshot: {path}")
        elif key == ord("+") or key == ord("="):
            threshold = min(threshold + 0.02, 0.99)
            rec.rec_thresh = threshold
            print(f"[Pipeline] Threshold → {threshold:.2f}")
        elif key == ord("-"):
            threshold = max(threshold - 0.02, 0.10)
            rec.rec_thresh = threshold
            print(f"[Pipeline] Threshold → {threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("[Pipeline] Done.")


def build_parser():
    p = argparse.ArgumentParser(description="EduAware — YOLO + Face Recognition Pipeline")
    # Change --cam to --src to support both int (camera) and str (video file)
    p.add_argument("--src", type=str, default="0", 
                   help="Camera index (e.g., 0) or path to video file (e.g., video.mp4)")
    p.add_argument("--yolo-model", type=str, default="yolov8n-pose.pt")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--db", type=str, default="face_database.pkl")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    
    # Check if the source is a digit (camera index) or a file path
    src = int(args.src) if args.src.isdigit() else args.src
    
    run(
        cam_id=src, # This now accepts the file path string or int index
        yolo_model=args.yolo_model,
        threshold=args.threshold,
        use_gpu=args.gpu,
        db_path=args.db,
    )
