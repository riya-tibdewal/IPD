"""
face_db.py — EduAware Face Database Manager
============================================
Manages a persistent database of face embeddings.
Each identity stores: name, embeddings list, metadata.

Usage (standalone test):
    python face_db.py
"""

import os
import pickle
import numpy as np
from datetime import datetime
from typing import Optional


DB_FILE = "face_database.pkl"


class FaceDatabase:
    """
    Stores and retrieves face embeddings for identity matching.

    Structure:
        {
            "student_001": {
                "name":       "Mulraj Gala",
                "embeddings": [np.array, np.array, ...],  # multiple shots
                "added_on":   "2025-05-09 10:30:00",
                "seen_count": 42
            },
            ...
        }
    """

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.db: dict = {}
        self._load()

    # ──────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────

    def _load(self):
        """Load database from disk if it exists."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.db = pickle.load(f)
            print(f"[FaceDB] Loaded {len(self.db)} identities from '{self.db_path}'")
        else:
            print(f"[FaceDB] No existing database found. Starting fresh.")

    def save(self):
        """Persist database to disk."""
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)
        print(f"[FaceDB] Saved {len(self.db)} identities to '{self.db_path}'")

    # ──────────────────────────────────────────────
    # CRUD
    # ──────────────────────────────────────────────

    def add_identity(
        self,
        person_id: str,
        name: str,
        embedding: np.ndarray,
        overwrite: bool = False,
    ) -> bool:
        """
        Register a new identity or add another embedding to an existing one.

        Args:
            person_id:  Unique ID string, e.g. "60004240032"
            name:       Human-readable label, e.g. "Mulraj Gala"
            embedding:  512-dim ArcFace vector (numpy array)
            overwrite:  If True, replace all existing embeddings for this ID

        Returns:
            True if successfully added.
        """
        embedding = embedding.astype(np.float32)

        if person_id not in self.db or overwrite:
            self.db[person_id] = {
                "name":       name,
                "embeddings": [embedding],
                "added_on":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seen_count": 0,
            }
            print(f"[FaceDB] Registered new identity: '{name}' ({person_id})")
        else:
            # Append additional embedding for same identity (improves matching)
            max_shots = 10  # keep at most 10 reference shots
            self.db[person_id]["embeddings"].append(embedding)
            if len(self.db[person_id]["embeddings"]) > max_shots:
                self.db[person_id]["embeddings"].pop(0)
            print(f"[FaceDB] Added embedding #{len(self.db[person_id]['embeddings'])} for '{name}'")

        return True

    def remove_identity(self, person_id: str) -> bool:
        """Delete an identity from the database."""
        if person_id in self.db:
            name = self.db[person_id]["name"]
            del self.db[person_id]
            print(f"[FaceDB] Removed identity: '{name}' ({person_id})")
            return True
        print(f"[FaceDB] Identity '{person_id}' not found.")
        return False

    def increment_seen(self, person_id: str):
        """Track how many times a person has been seen."""
        if person_id in self.db:
            self.db[person_id]["seen_count"] += 1

    # ──────────────────────────────────────────────
    # Querying
    # ──────────────────────────────────────────────

    def get_mean_embeddings(self) -> dict:
        """
        Return a dict of { person_id -> mean_embedding } for all identities.
        The mean embedding is more robust than individual shots.
        """
        return {
            pid: np.mean(data["embeddings"], axis=0)
            for pid, data in self.db.items()
        }

    def find_match(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.45,
    ) -> tuple[Optional[str], Optional[str], float]:
        """
        Find the best matching identity for a query embedding.

        Args:
            query_embedding: 512-dim ArcFace vector from the unknown face
            threshold:       Cosine similarity threshold (0.45 is a good default)
                             Higher = stricter matching (fewer false positives)

        Returns:
            (person_id, name, similarity_score)
            If no match found: (None, "Unknown", 0.0)
        """
        if not self.db:
            return None, "Unknown", 0.0

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-6)

        best_id    = None
        best_name  = "Unknown"
        best_score = 0.0

        for pid, data in self.db.items():
            # Compare against each stored shot, take the best
            for ref_emb in data["embeddings"]:
                ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-6)
                score = float(np.dot(query_norm, ref_norm))  # cosine similarity
                if score > best_score:
                    best_score = score
                    best_id    = pid
                    best_name  = data["name"]

        if best_score >= threshold:
            return best_id, best_name, best_score
        return None, "Unknown", best_score

    # ──────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────

    def list_identities(self):
        """Print a summary of all stored identities."""
        if not self.db:
            print("[FaceDB] Database is empty.")
            return
        print(f"\n{'─'*55}")
        print(f"  {'ID':<15} {'Name':<20} {'Shots':<6} {'Seen'}")
        print(f"{'─'*55}")
        for pid, data in self.db.items():
            print(
                f"  {pid:<15} {data['name']:<20} "
                f"{len(data['embeddings']):<6} {data['seen_count']}"
            )
        print(f"{'─'*55}\n")

    def __len__(self):
        return len(self.db)


# ──────────────────────────────────────────────────────
# Quick standalone test
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    db = FaceDatabase("test_db.pkl")

    # Simulate adding two identities with random 512-d embeddings
    emb_a = np.random.randn(512).astype(np.float32)
    emb_b = np.random.randn(512).astype(np.float32)

    db.add_identity("S001", "Alice", emb_a)
    db.add_identity("S002", "Bob",   emb_b)
    db.add_identity("S001", "Alice", emb_a + np.random.randn(512) * 0.05)

    db.list_identities()

    # Test matching: a slightly perturbed version of Alice's embedding
    query = emb_a + np.random.randn(512) * 0.08
    pid, name, score = db.find_match(query, threshold=0.40)
    print(f"Query matched → ID: {pid}, Name: {name}, Score: {score:.4f}")

    db.save()
    os.remove("test_db.pkl")
    print("[Test] Passed ✓")
