"""
main.py — EduAware v2  Entry Point
====================================
Classroom behavior analysis system.
No face recognition — pure upper-body pose heuristics.

Pipeline per processed frame
─────────────────────────────
  ThreadedCamera / VideoCapture
      ↓  frame
  YOLOv8-pose (ByteTrack)
      ↓  [(track_id, bbox, kps), ...]
  behavior_analysis.classify()     ← raw label (1 frame)
      ↓
  StudentBuffer.push()             ← temporal smoothing → stable label
      ↓
  AttentionRegistry.record()       ← accumulate frame counts
      ↓
  visualization.draw_*()           ← render overlays
      ↓
  cv2.imshow()

Usage
─────
  python main.py
  python main.py --src classroom.mp4
  python main.py --src 0 --gpu --skip 1
  python main.py --model yolov8s-pose.pt --imgsz 640
"""

import cv2
import time
import argparse
import threading
import numpy as np
from datetime import datetime

# ── Project modules ───────────────────────────────────────────────────────────
import config as C
from tracker           import StudentTracker
from behavior_analysis import classify
from visualization     import draw_student, draw_hud

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════════
# THREADED CAMERA READER
# ══════════════════════════════════════════════════════════════════════════════
# Reading frames on a background thread decouples I/O from GPU inference.
# Without threading, cap.read() blocks for ~30 ms on USB cameras, stalling
# the YOLO call and halving effective FPS.

class ThreadedCamera:
    """Reads from a VideoCapture on a daemon thread; always provides latest frame."""

    def __init__(self, src, width: int, height: int):
        self._cap = cv2.VideoCapture(src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # drop stale frames fast
        self._ret   = False
        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            with self._lock:
                self._ret, self._frame = ret, frame

    def read(self):
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
            return self._ret, frame

    def isOpened(self): return self._cap.isOpened()

    def release(self):
        self._stop.set()
        self._t.join(timeout=2)
        self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    # ── Override config from CLI args ─────────────────────────────────────
    yolo_model = args.model
    yolo_imgsz = args.imgsz
    yolo_conf  = args.conf
    frame_skip = args.skip
    use_gpu    = args.gpu

    # ── Load YOLO ─────────────────────────────────────────────────────────
    device = "0" if use_gpu else "cpu"
    print(f"[EduAware] Loading {yolo_model} on {device} …")
    yolo = YOLO(yolo_model)
    print("[EduAware] Model ready ✓\n")

    # ── Tracker + registry ────────────────────────────────────────────────
    tracker = StudentTracker()

    # ── Open video source ─────────────────────────────────────────────────
    src = int(args.src) if args.src.isdigit() else args.src
    if C.THREADED_CAP and isinstance(src, int):
        cap = ThreadedCamera(src, C.CAPTURE_WIDTH, C.CAPTURE_HEIGHT)
    else:
        cap = cv2.VideoCapture(src)
        if hasattr(cap, "set"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  C.CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[Error] Cannot open source: {src}")
        return

    # ── Display window ────────────────────────────────────────────────────
    cv2.namedWindow(C.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(C.WINDOW_NAME, C.WINDOW_W, C.WINDOW_H)

    print(f"[EduAware] Running  |  model={yolo_model}  imgsz={yolo_imgsz}  "
          f"conf={yolo_conf}  skip={frame_skip}")
    print("  Controls: Q=quit  P=pose  D=debug  S=screenshot  +/-=conf  SPACE=pause\n")

    # ── State ─────────────────────────────────────────────────────────────
    show_pose  = True
    show_debug = False
    paused     = False
    frame_cnt  = 0
    fps        = 0.0
    fps_alpha  = 0.10     # EMA smoothing coefficient
    prev_time  = time.time()
    last_out   = None     # cached annotated frame shown on skipped frames

    # ══════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[EduAware] Stream ended or read error.")
                break

            frame_cnt += 1

            # ── Pause ─────────────────────────────────────────────────────
            if paused:
                if last_out is not None:
                    cv2.imshow(C.WINDOW_NAME, last_out)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"): break
                if key == ord(" "): paused = False
                continue

            # ── Frame skipping ────────────────────────────────────────────
            # We read every frame (drains the capture buffer, preventing lag)
            # but only run inference on every Nth frame.
            if frame_cnt % frame_skip != 0:
                if last_out is not None:
                    cv2.imshow(C.WINDOW_NAME, last_out)
                cv2.waitKey(1)
                continue

            # ── YOLO inference (ByteTrack enabled by persist=True) ─────────
            results = yolo.track(
                frame,
                imgsz   = yolo_imgsz,
                conf    = yolo_conf,
                iou     = C.YOLO_IOU,
                classes = [0],          # person class only
                persist = True,         # enables ByteTrack across calls
                verbose = False,
                half    = use_gpu,      # FP16 on GPU for extra speed
            )

            # ── Per-frame render target ────────────────────────────────────
            out     = frame.copy()
            summary = {}   # behavior → count  (for HUD)

            # ── Process each detected person ───────────────────────────────
            for result in results:
                boxes    = result.boxes
                kps_data = result.keypoints
                if boxes is None:
                    continue

                for i, box in enumerate(boxes):
                    # ByteTrack ID (falls back to detection index if tracker off)
                    track_id = int(box.id[0]) if box.id is not None else i
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Keypoints for this person
                    kps = None
                    if kps_data is not None and i < len(kps_data.data):
                        kps = kps_data.data[i].cpu().numpy()   # (17, 3)

                    # ── Get / create student state ─────────────────────
                    state = tracker.get_or_create(track_id)

                    # ── Classify behavior ──────────────────────────────
                    raw_label, new_wrists = classify(
                        kps,
                        state.recent_labels,
                        state.prev_wrists,
                    ) if kps is not None else ("Unknown", None)

                    state.prev_wrists = new_wrists

                    # ── Temporal smoothing ─────────────────────────────
                    state.buffer.push(raw_label)
                    state.touch()

                    # ── Record stable label for attention score ────────
                    tracker.registry.record(track_id, state.stable_label)

                    # ── Live attention % ───────────────────────────────
                    attn_pct = tracker.registry.live_pct(track_id)

                    # ── Accumulate summary ─────────────────────────────
                    lbl = state.stable_label
                    summary[lbl] = summary.get(lbl, 0) + 1

                    # ── Draw student overlay ───────────────────────────
                    draw_student(
                        out, state,
                        (x1, y1, x2, y2), kps,
                        attn_pct,
                        show_pose  = show_pose,
                        show_debug = show_debug,
                        raw_label  = raw_label,
                    )

            # ── Prune lost tracks ──────────────────────────────────────────
            tracker.prune_stale()

            # ── FPS (exponential moving average) ──────────────────────────
            now      = time.time()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps      = fps_alpha * inst_fps + (1 - fps_alpha) * fps
            prev_time = now

            # ── Class-level attention average ──────────────────────────────
            all_scores = [
                tracker.registry.live_pct(s.track_id)
                for s in tracker.active_states
            ]
            class_avg = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

            # ── HUD ────────────────────────────────────────────────────────
            draw_hud(
                out, fps, len(all_scores),
                yolo_conf, show_pose, paused,
                summary, class_avg,
            )

            cv2.imshow(C.WINDOW_NAME, out)
            last_out = out

            # ── Key handling ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if   key == ord("q"):  break
            elif key == ord("p"):  show_pose  = not show_pose
            elif key == ord("d"):  show_debug = not show_debug
            elif key == ord(" "): paused = True
            elif key == ord("s"):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"eduaware_{ts}.jpg"
                cv2.imwrite(path, out)
                print(f"[EduAware] Screenshot → {path}")
            elif key in (ord("+"), ord("=")):
                yolo_conf = min(round(yolo_conf + 0.05, 2), 0.95)
                print(f"[EduAware] Conf → {yolo_conf:.2f}")
            elif key == ord("-"):
                yolo_conf = max(round(yolo_conf - 0.05, 2), 0.10)
                print(f"[EduAware] Conf → {yolo_conf:.2f}")

    finally:
        # Always print report even if interrupted with Ctrl-C
        cap.release()
        cv2.destroyAllWindows()
        tracker.registry.print_final_report()
        print("[EduAware] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="EduAware v2 — Classroom Behavior & Attention Analytics"
    )
    p.add_argument("--src",   type=str,   default="0",
                   help="Camera index (0) or video path (class.mp4)")
    p.add_argument("--model", type=str,   default=C.YOLO_MODEL,
                   help="YOLOv8-pose variant (yolov8n/s/m-pose.pt)")
    p.add_argument("--imgsz", type=int,   default=C.YOLO_IMGSZ,
                   help="YOLO inference size (lower = faster)")
    p.add_argument("--conf",  type=float, default=C.YOLO_CONF,
                   help="Person detection confidence threshold")
    p.add_argument("--skip",  type=int,   default=C.FRAME_SKIP,
                   help="Process 1-in-N frames (default 2)")
    p.add_argument("--gpu",   action="store_true",
                   help="Use CUDA GPU")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
