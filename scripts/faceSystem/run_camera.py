"""
run_camera.py — EduAware Live Face Recognition
================================================
Runs face recognition on a live webcam feed (no YOLO, just face pipeline).
Use this to test registration and recognition before integrating with YOLO.

Controls:
    Q       — quit
    R       — reload database (pick up new registrations on the fly)
    S       — save a screenshot
    L       — toggle landmarks display
    SPACE   — register currently detected face (quick register)

Usage:
    python run_camera.py
    python run_camera.py --cam 1 --gpu
    python run_camera.py --threshold 0.5   # stricter matching
"""

import cv2
import time
import argparse
import numpy as np
from datetime import datetime

from face_db import FaceDatabase
from face_recognizer import FaceRecognizer


def run(
    cam_id:    int   = 0,
    threshold: float = 0.45,
    use_gpu:   bool  = False,
    db_path:   str   = "face_database.pkl",
):
    db  = FaceDatabase(db_path)
    rec = FaceRecognizer(db, rec_thresh=threshold, use_gpu=use_gpu)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera {cam_id}")
        return

    # Optionally set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[Run] Live face recognition started. Press Q to quit.\n")

    show_landmarks = False
    prev_time      = time.time()
    frame_count    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(cam_id, str):
                print("[Info] Video processing complete.")
            else:
                print("[Error] Failed to read from camera.")
            break

        # ── Run recognition ──────────────────────
        results = rec.recognize(frame)
        out     = rec.draw_results(frame, results, show_landmarks=show_landmarks)

        # ── FPS counter ──────────────────────────
        frame_count += 1
        now  = time.time()
        fps  = frame_count / (now - prev_time + 1e-6)
        cv2.putText(
            out, f"FPS: {fps:.1f}",
            (out.shape[1] - 130, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )

        # ── Controls legend ──────────────────────
        cv2.putText(
            out, "Q=quit  R=reload  S=screenshot  L=landmarks",
            (10, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
        )

        # ── Threshold bar ────────────────────────
        cv2.putText(
            out, f"Thresh: {threshold:.2f}",
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        cv2.imshow("EduAware — Face Recognition", out)

        # ── Key handling ─────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            db._load()
            print("[Run] Database reloaded.")

        elif key == ord("l"):
            show_landmarks = not show_landmarks

        elif key == ord("s"):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{ts}.jpg"
            cv2.imwrite(path, out)
            print(f"[Run] Screenshot saved: {path}")

        elif key == ord(" "):
            # Quick-register the first detected unknown face
            unknowns = [r for r in results if not r.is_known]
            if unknowns:
                face   = unknowns[0]
                name   = input("  Enter name for unknown face: ").strip()
                uid    = input("  Enter ID (roll number): ").strip()
                if name and uid:
                    db.add_identity(uid, name, face.embedding)
                    db.save()
                    print(f"  Registered '{name}' ({uid}) ✓")
            else:
                print("  [Info] No unknown face to register.")

    cap.release()
    cv2.destroyAllWindows()
    print("[Run] Done.")


# Change the build_parser function at the bottom of yolo_face_pipeline.py
def build_parser():
    p = argparse.ArgumentParser(description="EduAware — Live Camera Recognition")
    # Changed from --cam (int) to --src (str) to support file paths
    p.add_argument("--src", type=str, default="0", 
                   help="Camera index (e.g., 0) or path to video file (e.g., video.mp4)")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="Cosine similarity threshold (default: 0.45)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--db", type=str, default="face_database.pkl")
    return p

# Update the main call to handle the source type
if __name__ == "__main__":
    args = build_parser().parse_args()
    
    # Logic to convert "0" to 0 (int) but keep "video.mp4" as a string
    source = int(args.src) if args.src.isdigit() else args.src
    
    run(
        cam_id=source,
        threshold=args.threshold,
        use_gpu=args.gpu,
        db_path=args.db,
    )