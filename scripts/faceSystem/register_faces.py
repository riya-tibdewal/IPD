"""
register_faces.py — EduAware Face Registration Tool
=====================================================
Register known identities into the face database.

Two modes:
  1. FROM FOLDER  — Point to a folder of images organized by person
  2. FROM CAMERA  — Capture face shots live from webcam

Folder structure expected for mode 1:
    faces/
    ├── Mulraj_Gala__60004240032/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── img3.jpg
    ├── Nilish_Shah__60004240234/
    │   └── photo.jpg
    └── ...

    Folder name format: Name__ID  (two underscores between name and ID)
    Spaces in name use single underscore: "Mulraj_Gala" → "Mulraj Gala"

Usage:
    # Register from folder
    python register_faces.py --mode folder --src ./faces

    # Register one person live from camera
    python register_faces.py --mode camera --name "Mulraj Gala" --id 60004240032 --shots 5

    # List all registered identities
    python register_faces.py --list
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm

from face_db import FaceDatabase
from face_recognizer import FaceRecognizer


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def register_from_folder(db: FaceDatabase, rec: FaceRecognizer, src_dir: str):
    """
    Walk a folder tree, detect faces in each image, and register embeddings.

    Each sub-folder = one identity.
    Folder name format: FirstName_LastName__StudentID
    """
    if not os.path.isdir(src_dir):
        print(f"[Error] Folder not found: {src_dir}")
        return

    person_dirs = [
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    ]

    if not person_dirs:
        print(f"[Warning] No sub-folders found in '{src_dir}'.")
        return

    print(f"\nFound {len(person_dirs)} person folder(s). Starting registration …\n")

    for folder_name in sorted(person_dirs):
        # Parse "Mulraj_Gala__60004240032" → name="Mulraj Gala", id="60004240032"
        if "__" in folder_name:
            raw_name, person_id = folder_name.split("__", 1)
            name = raw_name.replace("_", " ")
        else:
            name = folder_name.replace("_", " ")
            person_id = folder_name  # fall back: use folder name as ID

        folder_path = os.path.join(src_dir, folder_name)
        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        ]

        if not image_files:
            print(f"  [Skip] No images found for '{name}'")
            continue

        print(f"  Registering: {name} ({person_id}) — {len(image_files)} image(s)")
        added = 0

        for img_file in tqdm(image_files, desc=f"    {name}", leave=False):
            img_path = os.path.join(folder_path, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"    [Warn] Could not read: {img_file}")
                continue

            faces = rec.detect(frame)
            if not faces:
                print(f"    [Warn] No face found in: {img_file}")
                continue

            # Use the face with the highest detection score
            best = max(faces, key=lambda f: f.det_score)
            db.add_identity(person_id, name, best.embedding)
            added += 1

        print(f"    ✓ {added}/{len(image_files)} shots registered for '{name}'")

    db.save()
    print("\nRegistration complete.")
    db.list_identities()


def register_from_camera(
    db:       FaceDatabase,
    rec:      FaceRecognizer,
    name:     str,
    person_id: str,
    num_shots: int = 5,
    cam_id:   int = 0,
    delay_s:  float = 0.8,
):
    """
    Capture `num_shots` face images from the webcam for one person.

    Press SPACE to manually capture a shot.
    Press Q to quit early.
    """
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[Error] Cannot open camera {cam_id}")
        return

    print(f"\nCapturing {num_shots} shots for: {name} ({person_id})")
    print("Controls: SPACE = capture  |  Q = quit\n")

    captured  = 0
    last_time = 0.0

    while captured < num_shots:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        faces = rec.detect(frame)

        # Draw face boxes (detection only, no DB lookup during registration)
        for f in faces:
            cv2.rectangle(display, (f.x1, f.y1), (f.x2, f.y2), (0, 200, 255), 2)

        # Overlay instructions
        cv2.putText(
            display,
            f"Registering: {name}  [{captured}/{num_shots} captured]",
            (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 2,
        )
        cv2.putText(
            display, "SPACE = capture  |  Q = quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )

        cv2.imshow("EduAware — Register Face", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[Register] Quit early.")
            break

        auto_capture = (time.time() - last_time) > delay_s and faces
        if key == ord(" ") or auto_capture:
            if not faces:
                print("  [Warn] No face detected, try again.")
                continue
            best = max(faces, key=lambda f: f.det_score)
            db.add_identity(person_id, name, best.embedding)
            captured += 1
            last_time = time.time()
            print(f"  ✓ Shot {captured}/{num_shots} captured (score={best.det_score:.3f})")

            # Flash green feedback
            feedback = display.copy()
            cv2.rectangle(feedback, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
            cv2.imshow("EduAware — Register Face", feedback)
            cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()

    if captured > 0:
        db.save()
        print(f"\nRegistered '{name}' with {captured} shot(s) ✓")
    else:
        print("\nNo shots captured. Identity not saved.")


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="EduAware — Face Registration Tool")
    p.add_argument("--mode", choices=["folder", "camera"], default="folder",
                   help="Registration mode")
    p.add_argument("--src",  type=str, default="./faces",
                   help="[folder mode] Path to root folder with person sub-folders")
    p.add_argument("--name", type=str, default="",
                   help="[camera mode] Full name of the person")
    p.add_argument("--id",   type=str, default="",
                   help="[camera mode] Unique ID / roll number")
    p.add_argument("--shots", type=int, default=5,
                   help="[camera mode] Number of shots to capture (default: 5)")
    p.add_argument("--cam",  type=int, default=0,
                   help="Camera index (default: 0)")
    p.add_argument("--db",   type=str, default="face_database.pkl",
                   help="Path to face database file")
    p.add_argument("--gpu",  action="store_true",
                   help="Use GPU for inference")
    p.add_argument("--list", action="store_true",
                   help="List all registered identities and exit")
    p.add_argument("--remove", type=str, default="",
                   help="Remove an identity by ID and exit")
    return p


def main():
    args = build_parser().parse_args()

    db  = FaceDatabase(args.db)
    rec = FaceRecognizer(db, use_gpu=args.gpu)

    if args.list:
        db.list_identities()
        return

    if args.remove:
        db.remove_identity(args.remove)
        db.save()
        return

    if args.mode == "folder":
        register_from_folder(db, rec, args.src)

    elif args.mode == "camera":
        if not args.name or not args.id:
            print("[Error] --name and --id are required for camera mode.")
            print("Example: python register_faces.py --mode camera --name 'Mulraj Gala' --id 60004240032")
            sys.exit(1)
        register_from_camera(
            db, rec,
            name=args.name,
            person_id=args.id,
            num_shots=args.shots,
            cam_id=args.cam,
        )


if __name__ == "__main__":
    main()
