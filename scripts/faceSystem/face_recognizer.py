"""
face_recognizer.py — EduAware Core Face Recognizer
====================================================
Wraps InsightFace (RetinaFace detector + ArcFace embedder) into a
clean class that detects faces in a frame and returns recognition results.

Model used: buffalo_l  (best accuracy; ~340 MB, auto-downloaded on first run)
Alternative: buffalo_s (faster, lighter; use on CPU-only machines)

Usage (standalone test):
    python face_recognizer.py
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

import insightface
from insightface.app import FaceAnalysis

from face_db import FaceDatabase


# ──────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────

@dataclass
class FaceResult:
    """Holds all information for one detected face in a frame."""
    bbox:        np.ndarray   # [x1, y1, x2, y2]
    landmark:    np.ndarray   # 5-point landmark
    embedding:   np.ndarray   # 512-d ArcFace embedding
    det_score:   float        # RetinaFace detection confidence

    # Recognition results (filled after DB lookup)
    person_id:   Optional[str] = None
    name:        str = "Unknown"
    similarity:  float = 0.0

    @property
    def x1(self): return int(self.bbox[0])
    @property
    def y1(self): return int(self.bbox[1])
    @property
    def x2(self): return int(self.bbox[2])
    @property
    def y2(self): return int(self.bbox[3])

    @property
    def is_known(self) -> bool:
        return self.person_id is not None

    @property
    def label(self) -> str:
        if self.is_known:
            return f"{self.name} ({self.similarity:.2f})"
        return f"Unknown ({self.similarity:.2f})"


# ──────────────────────────────────────────────────────
# Main Recognizer Class
# ──────────────────────────────────────────────────────

class FaceRecognizer:
    """
    Detects faces with RetinaFace, extracts ArcFace embeddings,
    and matches them against a FaceDatabase.

    Args:
        db:            FaceDatabase instance
        model_name:    'buffalo_l' (accurate) or 'buffalo_s' (fast)
        det_size:      Detection input size. Larger = more accurate but slower.
                       Use (320, 320) for speed, (640, 640) for accuracy.
        det_thresh:    RetinaFace detection confidence threshold (0–1)
        rec_thresh:    Cosine similarity threshold for recognition (0–1)
        use_gpu:       True = use GPU (ctx_id=0), False = CPU (ctx_id=-1)
    """

    def __init__(
        self,
        db:          FaceDatabase,
        model_name:  str   = "buffalo_l",
        det_size:    tuple = (640, 640),
        det_thresh:  float = 0.5,
        rec_thresh:  float = 0.45,
        use_gpu:     bool  = False,
    ):
        self.db         = db
        self.rec_thresh = rec_thresh
        self.det_thresh = det_thresh

        ctx_id = 0 if use_gpu else -1  # 0 = first GPU, -1 = CPU

        print(f"[FaceRecognizer] Loading InsightFace model '{model_name}' "
              f"({'GPU' if use_gpu else 'CPU'}) …")

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        print("[FaceRecognizer] Model ready ✓")

    # ──────────────────────────────────────────────
    # Core pipeline
    # ──────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Run face detection on a BGR OpenCV frame.

        Returns:
            List of FaceResult objects (embedding extracted, recognition NOT yet done).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        results = []
        for face in faces:
            emb = face.normed_embedding  # already L2-normalized 512-d vector
            results.append(FaceResult(
                bbox=face.bbox,
                landmark=face.kps,
                embedding=emb,
                det_score=float(face.det_score),
            ))
        return results

    def recognize(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Detect faces AND match each against the database.

        Returns:
            List of FaceResult objects with recognition fields filled.
        """
        results = self.detect(frame)
        for result in results:
            pid, name, score = self.db.find_match(result.embedding, self.rec_thresh)
            result.person_id  = pid
            result.name       = name
            result.similarity = score
            if pid:
                self.db.increment_seen(pid)
        return results

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract an ArcFace embedding from an already-cropped face image.
        Useful when you have face crops from another detector (e.g., YOLO).

        Args:
            face_crop: BGR image of the face (any size, will be resized internally)

        Returns:
            512-d normalized embedding, or None if no face detected.
        """
        results = self.detect(face_crop)
        if results:
            return results[0].embedding
        return None

    # ──────────────────────────────────────────────
    # Drawing utilities
    # ──────────────────────────────────────────────

    def draw_results(
        self,
        frame: np.ndarray,
        results: list[FaceResult],
        show_score: bool = True,
        show_landmarks: bool = False,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on a frame.

        Returns:
            Annotated frame (copy).
        """
        out = frame.copy()

        for r in results:
            color = (0, 200, 0) if r.is_known else (0, 0, 220)  # green / red

            # Bounding box
            cv2.rectangle(out, (r.x1, r.y1), (r.x2, r.y2), color, 2)

            # Label background + text
            label = r.label if show_score else (r.name if r.is_known else "Unknown")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.rectangle(out, (r.x1, r.y1 - th - 8), (r.x1 + tw + 4, r.y1), color, -1)
            cv2.putText(
                out, label,
                (r.x1 + 2, r.y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            # Optional: 5-point landmarks (eyes, nose, mouth corners)
            if show_landmarks and r.landmark is not None:
                for (lx, ly) in r.landmark.astype(int):
                    cv2.circle(out, (lx, ly), 3, (255, 200, 0), -1)

        # HUD: face count
        cv2.putText(
            out, f"Faces: {len(results)}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )

        return out


# ──────────────────────────────────────────────────────
# Standalone test on a static image
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os

    db = FaceDatabase()
    rec = FaceRecognizer(db, use_gpu=False)

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"[Error] Image not found: {img_path}")
            sys.exit(1)
        frame = cv2.imread(img_path)
        results = rec.recognize(frame)
        print(f"Detected {len(results)} face(s):")
        for r in results:
            print(f"  BBox: {r.bbox.astype(int)}, Label: {r.label}, DetScore: {r.det_score:.3f}")
        out = rec.draw_results(frame, results, show_landmarks=True)
        cv2.imshow("Face Recognition Test", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Usage: python face_recognizer.py <image_path>")
        print("Running quick embedding extraction test …")
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = rec.recognize(dummy)
        print(f"Detected {len(results)} face(s) in random noise frame (expected: 0)")
        print("[Test] face_recognizer.py OK ✓")
