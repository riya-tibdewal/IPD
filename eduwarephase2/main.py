"""EduAware entry point: tracking, pose scoring, identity caching, overlays."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

import config as C
from behavior_classifier import BehaviorClassifier
from camera_manager import open_capture, parse_source, wait_for_first_frame
from face_recognition_pipeline import AsyncFaceIdentityManager
from tracker import StudentTracker
from visualization import draw_hud, draw_student


LOGGER = logging.getLogger("eduaware")


def run(args):
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    device = "0" if args.gpu else "cpu"
    LOGGER.info("loading pose model %s on %s", args.model, device)
    pose_model = YOLO(args.model)
    object_model = None
    if args.objects:
        LOGGER.info("loading object model %s for phone detection", args.object_model)
        object_model = YOLO(args.object_model)

    tracker = StudentTracker()
    classifier = BehaviorClassifier()
    identity_manager = AsyncFaceIdentityManager(recognizer=None)

    src = parse_source(args.src)
    cap, diagnostics = open_capture(src, threaded=C.THREADED_CAP)
    if not diagnostics.opened:
        LOGGER.error("cannot open source: %s", src)
        return

    first_frame = wait_for_first_frame(cap)
    if first_frame is None:
        LOGGER.error("camera opened but no frames arrived within %.1fs", C.CAMERA_READ_TIMEOUT_S)
        cap.release()
        return

    cv2.namedWindow(C.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(C.WINDOW_NAME, C.WINDOW_W, C.WINDOW_H)
    LOGGER.info("running: imgsz=%s conf=%.2f skip=%s objects=%s", args.imgsz, args.conf, args.skip, args.objects)
    LOGGER.info("controls: Q quit, P pose, D debug, S screenshot, +/- confidence, Space pause")

    show_pose = True
    show_debug = args.debug
    paused = False
    frame_idx = 0
    fps = 0.0
    fps_alpha = 0.10
    prev_time = time.time()
    last_out = None
    pending_frame = first_frame
    object_context = {"phone": [], "writing": []}
    cached_tracks = []
    perf = {"infer_ms": 0.0, "object_ms": 0.0, "render_ms": 0.0, "latency_ms": 0.0, "dropped": 0}

    try:
        while True:
            if pending_frame is not None:
                ret, frame = True, pending_frame
                pending_frame = None
            else:
                ret, frame = cap.read()
            if not ret or frame is None:
                LOGGER.warning("stream ended or frame read failed")
                break

            frame_idx += 1

            if paused:
                if last_out is not None:
                    cv2.imshow(C.WINDOW_NAME, last_out)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = False
                continue

            frame_start = time.perf_counter()
            run_inference = frame_idx % max(args.skip, 1) == 0

            if object_model is not None and frame_idx % C.OBJECT_DETECT_EVERY_N == 0:
                t0 = time.perf_counter()
                object_context = detect_object_context(object_model, frame, args.imgsz, args.conf, device, args.gpu)
                perf["object_ms"] = _ema_ms(perf["object_ms"], t0)

            if run_inference:
                t0 = time.perf_counter()
                results = pose_model.track(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=C.YOLO_IOU,
                    classes=[0],
                    persist=True,
                    verbose=False,
                    device=device,
                    half=args.gpu,
                )
                perf["infer_ms"] = _ema_ms(perf["infer_ms"], t0)
                cached_tracks = []
                detected_tracks = []

                for result in results:
                    boxes = result.boxes
                    keypoints = result.keypoints
                    if boxes is None:
                        continue

                    for i, box in enumerate(boxes):
                        track_id = int(box.id[0]) if box.id is not None else i
                        bbox = tuple(map(int, box.xyxy[0]))
                        kps = None
                        if keypoints is not None and i < len(keypoints.data):
                            kps = keypoints.data[i].cpu().numpy()
                        detected_tracks.append((track_id, bbox, kps))

                object_owners = assign_object_context(object_context, detected_tracks)
                for track_id, bbox, kps in detected_tracks:
                    state = tracker.get_or_create(track_id)
                    state.touch(bbox)
                    state.last_keypoints = kps
                    owned_objects = object_owners.get(track_id, {"phone": False, "writing": False})
                    result_state = classifier.classify(
                        kps,
                        bbox,
                        state,
                        phone_detected=owned_objects["phone"],
                        writing_object_detected=owned_objects["writing"],
                    )

                    identity_manager.maybe_submit(frame_idx, frame, state, bbox)
                    identity_manager.expire(state)

                    tracker.registry.record(track_id, result_state.label, result_state.attention_score)
                    cached_tracks.append((state.track_id, result_state.raw_label))
            else:
                perf["dropped"] += 1
                for state in tracker.active_states:
                    if state.last_bbox is not None:
                        tracker.registry.record(state.track_id, state.stable_label, state.attention_score)

            t_render = time.perf_counter()
            out = frame.copy()
            summary = {}
            for state in tracker.active_states:
                if state.last_bbox is None:
                    continue
                summary[state.stable_label] = summary.get(state.stable_label, 0) + 1
                draw_student(
                    out,
                    state,
                    state.smooth_bbox() or state.last_bbox,
                    state.last_keypoints,
                    show_pose=show_pose,
                    show_debug=show_debug,
                    raw_label=max(state.raw_scores, key=state.raw_scores.get) if state.raw_scores else state.stable_label,
                )
            perf["render_ms"] = _ema_ms(perf["render_ms"], t_render)

            identity_manager.collect(tracker)
            tracker.prune_stale()

            now = time.time()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps = fps_alpha * inst_fps + (1.0 - fps_alpha) * fps
            prev_time = now
            all_scores = [state.attention_score for state in tracker.active_states]
            class_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

            perf["latency_ms"] = (time.perf_counter() - frame_start) * 1000.0
            draw_hud(out, fps, len(all_scores), args.conf, show_pose, paused, summary, class_avg, diagnostics, perf)
            cv2.imshow(C.WINDOW_NAME, out)
            last_out = out

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                show_pose = not show_pose
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord(" "):
                paused = True
            elif key == ord("s"):
                path = f"eduaware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(path, out)
                LOGGER.info("screenshot saved: %s", path)
            elif key in (ord("+"), ord("=")):
                args.conf = min(round(args.conf + 0.05, 2), 0.95)
            elif key == ord("-"):
                args.conf = max(round(args.conf - 0.05, 2), 0.10)
    finally:
        identity_manager.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        tracker.registry.print_final_report()
        LOGGER.info("done")


def detect_object_context(model, frame, imgsz: int, conf: float, device: str, use_gpu: bool):
    results = model.predict(frame, imgsz=imgsz, conf=max(conf, 0.25), verbose=False, device=device, half=use_gpu)
    context = {"phone": [], "writing": []}
    names = getattr(model, "names", {})
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = str(names.get(cls_id, cls_id)).lower()
            if name in C.PHONE_CLASSES:
                context["phone"].append(tuple(map(int, box.xyxy[0])))
            elif name in C.WRITING_OBJECT_CLASSES:
                context["writing"].append(tuple(map(int, box.xyxy[0])))
    return context


def _center_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _object_near_student(obj: tuple[int, int, int, int], person: tuple[int, int, int, int]) -> bool:
    if _center_inside(obj, person):
        return True
    x1, y1, x2, y2 = obj
    px1, py1, px2, py2 = person
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    margin_x = (px2 - px1) * 0.18
    margin_y = (py2 - py1) * 0.12
    return px1 - margin_x <= cx <= px2 + margin_x and py1 - margin_y <= cy <= py2 + margin_y


def assign_object_context(object_context: dict, detected_tracks: list[tuple[int, tuple, object]]) -> dict:
    owners = {track_id: {"phone": False, "writing": False} for track_id, _, _ in detected_tracks}
    for kind, boxes in object_context.items():
        for obj_box in boxes:
            best_track = None
            best_score = 0.0
            for track_id, person_box, kps in detected_tracks:
                score = _object_student_score(obj_box, person_box, kps)
                if score > best_score:
                    best_score = score
                    best_track = track_id
            if best_track is not None and best_score >= 0.18:
                owners[best_track][kind] = True
    return owners


def _object_student_score(obj: tuple[int, int, int, int], person: tuple[int, int, int, int], kps) -> float:
    score = 0.0
    if _center_inside(obj, person):
        score += 0.35
    if _object_near_student(obj, person):
        score += 0.20
    score += 0.25 * _iou(obj, person)
    wrist_score = _wrist_object_score(obj, person, kps)
    score += 0.35 * wrist_score
    return score


def _wrist_object_score(obj: tuple[int, int, int, int], person: tuple[int, int, int, int], kps) -> float:
    if kps is None:
        return 0.0
    ox = (obj[0] + obj[2]) * 0.5
    oy = (obj[1] + obj[3]) * 0.5
    px1, py1, px2, py2 = person
    scale = max(px2 - px1, py2 - py1, 1)
    best = 0.0
    for idx in (C.KP_L_WRIST, C.KP_R_WRIST):
        if idx < len(kps) and kps[idx][2] >= C.KP_CONF:
            dx = float(kps[idx][0]) - ox
            dy = float(kps[idx][1]) - oy
            dist = (dx * dx + dy * dy) ** 0.5 / scale
            best = max(best, max(0.0, 1.0 - dist / 0.35))
    return best


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _ema_ms(prev_ms: float, start_time: float, alpha: float = 0.15) -> float:
    current = (time.perf_counter() - start_time) * 1000.0
    return current if prev_ms <= 0 else alpha * current + (1.0 - alpha) * prev_ms


def build_parser():
    parser = argparse.ArgumentParser(description="EduAware classroom behavior analytics")
    parser.add_argument("--src", type=str, default="0", help="camera index or video path")
    parser.add_argument("--model", type=str, default=C.YOLO_MODEL, help="YOLO pose model")
    parser.add_argument("--object-model", type=str, default=C.OBJECT_MODEL, help="YOLO object model for phone detection")
    parser.add_argument("--objects", action="store_true", default=C.ENABLE_OBJECT_DETECTION, help="enable object detection for phones")
    parser.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ)
    parser.add_argument("--conf", type=float, default=C.YOLO_CONF)
    parser.add_argument("--skip", type=int, default=C.FRAME_SKIP)
    parser.add_argument("--gpu", action="store_true", default=C.USE_GPU)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", type=str, default=C.LOG_LEVEL)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
