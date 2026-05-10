"""Compatibility module for the requested face_recognition.py architecture."""

from face_recognition_pipeline import AsyncFaceIdentityManager, FaceRecognizerAdapter, IdentityResult

__all__ = ["AsyncFaceIdentityManager", "FaceRecognizerAdapter", "IdentityResult"]

