"""Temporal behavior scoring engine with hysteresis and explainable factors."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config as C
from pose_utils import PoseMetrics, safe01


BEHAVIORS = ("ATTENTIVE", "WRITING", "USING_PHONE", "SLEEPING", "STANDING", "TALKING", "DISTRACTED", "UNKNOWN")


@dataclass
class BehaviorScore:
    label: str
    confidence: float
    factors: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class FrameAssessment:
    behavior: BehaviorScore
    scores: dict[str, float]
    attention_score: float
    attention_factors: dict[str, float]


class StudentTemporalState:
    """Per-student memory used by the scoring engine."""

    def __init__(self, history_len: int = C.HISTORY_FRAMES):
        self.history = deque(maxlen=history_len)
        self.ema_scores = {label: 0.0 for label in BEHAVIORS}
        self.ema_attention = 0.0
        self.stable_label = "UNKNOWN"
        self.frames_since_switch = 0
        self.strong_candidate = "UNKNOWN"
        self.strong_candidate_frames = 0
        self.prev_wrists = None
        self.prev_body = None
        self.prev_object_owner = {"phone": 0.0, "writing": 0.0}
        self.identity = "Unknown"
        self.identity_confidence = 0.0
        self.identity_last_seen = 0.0

    @property
    def confidence(self) -> float:
        if not self.history:
            return 0.0
        return Counter(item["label"] for item in self.history)[self.stable_label] / len(self.history)

    @property
    def attention_score(self) -> float:
        return self.ema_attention

    def update(self, assessment: FrameAssessment) -> BehaviorScore:
        alpha = C.EMA_ALPHA
        for label, value in assessment.scores.items():
            prev = self.ema_scores.get(label, 0.0)
            label_alpha = min(0.35, alpha * 1.8) if value >= 0.80 else alpha
            self.ema_scores[label] = label_alpha * value + (1.0 - label_alpha) * prev
        attn_alpha = C.ATTENTION_SCORE_ALPHA
        self.ema_attention = attn_alpha * assessment.attention_score + (1.0 - attn_alpha) * self.ema_attention

        winner = max(self.ema_scores, key=self.ema_scores.get)
        winner_score = self.ema_scores[winner]
        current_score = self.ema_scores.get(self.stable_label, 0.0)
        self.frames_since_switch += 1
        if winner == self.strong_candidate:
            self.strong_candidate_frames += 1
        else:
            self.strong_candidate = winner
            self.strong_candidate_frames = 1

        can_switch = self.frames_since_switch >= C.MIN_SWITCH_FRAMES
        strong_evidence = (
            self.strong_candidate_frames >= C.STRONG_EVIDENCE_FRAMES
            and winner_score >= current_score + C.STRONG_EVIDENCE_MARGIN
            and winner_score >= C.MIN_LABEL_CONFIDENCE
        )
        normal_evidence = can_switch and winner_score >= current_score + C.HYSTERESIS_MARGIN
        if self.stable_label == "UNKNOWN" or strong_evidence or normal_evidence:
            self.stable_label = winner
            self.frames_since_switch = 0

        self.history.append({
            "label": assessment.behavior.label,
            "stable": self.stable_label,
            "scores": dict(assessment.scores),
            "attention": assessment.attention_score,
            "metrics": dict(assessment.behavior.factors),
        })

        stable_conf = safe01(max(self.ema_scores.get(self.stable_label, 0.0), self.confidence))
        return BehaviorScore(
            label=self.stable_label,
            confidence=stable_conf,
            factors=assessment.behavior.factors,
            reasons=assessment.behavior.reasons,
        )


class BehaviorScoringEngine:
    """Scores all behavior classes from pose, object detections, and history."""

    def assess(
        self,
        metrics: PoseMetrics,
        state: StudentTemporalState,
        phone_detected: bool = False,
        writing_object_detected: bool = False,
    ) -> FrameAssessment:
        if not metrics.valid:
            scores = {label: 0.0 for label in BEHAVIORS}
            scores["UNKNOWN"] = 1.0
            return FrameAssessment(
                BehaviorScore("UNKNOWN", 1.0, reasons=["pose unavailable"]),
                scores,
                0.0,
                {},
            )

        recent = list(state.history)
        prolonged_down = _recent_mean(recent, "head_down", metrics.head_down)
        side_persistence = _recent_mean(recent, "head_side", metrics.head_side)
        stillness_history = _recent_mean(recent, "stillness", 0.0)
        slouch_history = _recent_mean(recent, "slouch", metrics.slouch)
        writing_object_persistence = _recent_mean(recent, "writing_object", float(writing_object_detected))
        phone_persistence = _recent_mean(recent, "phone_object", float(phone_detected))
        writing_motion_history = _recent_mean(recent, "writing_motion", 0.0)
        standing_history = _recent_mean(recent, "true_standing_pose", 0.0)
        stillness = 1.0 - safe01(max(metrics.body_motion, metrics.wrist_motion) / C.EXCESSIVE_MOTION)
        writing_motion = _band_score(metrics.wrist_motion, C.WRITING_MOTION_MIN, C.WRITING_MOTION_MAX)
        active_wrist = float(C.WRITING_MOTION_MIN <= metrics.wrist_motion <= C.WRITING_MOTION_MAX)
        active_wrist_fraction = _recent_fraction(recent, "active_wrist", 0.5, active_wrist)
        inactive_now = float(metrics.wrist_motion < C.WRITING_MOTION_MIN * 0.55 and metrics.body_motion < C.STILL_MOTION_MAX)
        inactive_fraction = _recent_fraction(recent, "inactive", 0.5, inactive_now)
        head_down_fraction = _recent_fraction(recent, "head_down", 0.60, metrics.head_down)
        motion_energy_history = _recent_mean(recent, "motion_energy", metrics.motion_energy)
        rhythmic_wrist = safe01(0.45 * writing_motion + 0.55 * writing_motion_history)
        excessive_motion = safe01(metrics.body_motion / C.EXCESSIVE_MOTION)
        front_or_back = max(1.0 - metrics.head_side, metrics.back_view * 0.85)
        upright = (1.0 - metrics.slouch) * (1.0 - safe01(metrics.torso_lean))
        desk_focus = max(metrics.hands_near_desk, metrics.head_down * 0.65)
        peer_facing = safe01(metrics.head_side * (1.0 - metrics.head_down))
        tall_box = safe01((metrics.bbox_aspect - 1.65) / 1.10)
        full_body_evidence = max(metrics.leg_visibility, tall_box)
        standing_evidence = max(metrics.vertical_extension, tall_box * metrics.lower_body_visibility) * full_body_evidence * upright
        true_standing_pose = float(
            standing_evidence >= 0.78
            and metrics.leg_visibility >= 0.50
            and metrics.seated_cue < 0.35
            and metrics.head_down < 0.45
        )
        social_motion = safe01(0.45 * side_persistence + 0.25 * peer_facing + 0.20 * excessive_motion + 0.10 * metrics.torso_lean)
        body_engagement = safe01(
            0.34 * motion_energy_history
            + 0.26 * active_wrist_fraction
            + 0.18 * (1.0 - inactive_fraction)
            + 0.12 * upright
            + 0.10 * metrics.hands_near_desk
        )
        writing_context = safe01(
            0.28 * max(float(writing_object_detected), writing_object_persistence)
            + 0.30 * active_wrist_fraction
            + 0.20 * metrics.hands_near_desk
            + 0.10 * metrics.head_down
            + 0.08 * body_engagement
            + 0.06 * (1.0 - social_motion)
        )
        sleep_context = safe01(
            0.24 * head_down_fraction
            + 0.24 * max(metrics.slouch, slouch_history, metrics.torso_lean)
            + 0.24 * inactive_fraction
            + 0.12 * metrics.seated_cue
            + 0.10 * (1.0 - metrics.face_visibility)
            + 0.06 * metrics.hands_near_head
        )
        collapse_context = max(metrics.slouch, slouch_history, metrics.torso_lean, metrics.hands_near_head)

        factors = {
            "head_down": metrics.head_down,
            "head_side": metrics.head_side,
            "upright": upright,
            "desk_hands": metrics.hands_near_desk,
            "hands_close": metrics.hands_close,
            "wrist_motion": metrics.wrist_motion,
            "body_motion": metrics.body_motion,
            "stillness": stillness,
            "slouch": metrics.slouch,
            "motion_energy": metrics.motion_energy,
            "body_engagement": body_engagement,
            "active_wrist": active_wrist,
            "inactive": inactive_now,
            "active_wrist_fraction": active_wrist_fraction,
            "inactive_fraction": inactive_fraction,
            "head_down_fraction": head_down_fraction,
            "back_view": metrics.back_view,
            "vertical_extension": metrics.vertical_extension,
            "lower_body_visibility": metrics.lower_body_visibility,
            "leg_visibility": metrics.leg_visibility,
            "bbox_aspect": metrics.bbox_aspect,
            "peer_facing": peer_facing,
            "seated_cue": metrics.seated_cue,
            "writing_object": float(writing_object_detected),
            "phone_object": float(phone_detected),
            "writing_motion": writing_motion,
            "true_standing_pose": true_standing_pose,
            "social_motion": social_motion,
            "collapse_context": collapse_context,
        }

        scores = {label: 0.0 for label in BEHAVIORS}
        scores["ATTENTIVE"] = safe01(
            0.28 * (1.0 - metrics.head_down)
            + 0.22 * upright
            + 0.20 * front_or_back
            + 0.15 * (1.0 - excessive_motion)
            + 0.15 * metrics.shoulder_symmetry
        )
        scores["ATTENTIVE"] *= 1.0 - 0.75 * peer_facing
        scores["WRITING"] = safe01(
            0.78 * writing_context
            + 0.10 * metrics.head_down
            + 0.08 * body_engagement
            + 0.04 * (1.0 - metrics.slouch)
        )
        scores["USING_PHONE"] = safe01(
            0.36 * max(float(phone_detected), phone_persistence)
            + 0.20 * metrics.head_down
            + 0.17 * metrics.hands_close
            + 0.14 * max(metrics.hands_near_desk, metrics.hands_near_head)
            + 0.17 * safe01(prolonged_down)
        )
        scores["SLEEPING"] = safe01(
            0.24 * head_down_fraction
            + 0.20 * max(metrics.slouch, slouch_history, metrics.torso_lean)
            + 0.22 * inactive_fraction
            + 0.14 * metrics.hands_near_head
            + 0.12 * safe01(prolonged_down)
            + 0.08 * metrics.shoulder_tilt
            + 0.30 * sleep_context
        )
        scores["STANDING"] = safe01(
            0.45 * standing_evidence
            + 0.25 * standing_history
            + 0.30 * _recent_rise(recent, metrics.vertical_extension) * full_body_evidence
            + 0.10 * upright * full_body_evidence
        )
        scores["TALKING"] = safe01(
            0.42 * side_persistence
            + 0.22 * peer_facing
            + 0.14 * metrics.head_side
            + 0.12 * safe01(metrics.body_motion / max(C.EXCESSIVE_MOTION, 1e-6))
            + 0.10 * (1.0 - desk_focus)
        )
        scores["DISTRACTED"] = safe01(
            0.36 * metrics.head_side
            + 0.20 * excessive_motion
            + 0.18 * (1.0 - front_or_back)
            + 0.14 * metrics.head_up
            + 0.10 * metrics.torso_lean
            + 0.10 * (1.0 - desk_focus)
            + 0.20 * peer_facing
        )

        strong_sleep = (
            len(recent) >= C.SLEEP_MIN_HISTORY
            and sleep_context >= 0.70
            and head_down_fraction >= C.SLEEP_HEAD_DOWN_FRAC
            and inactive_fraction >= C.SLEEP_INACTIVE_FRAC
            and active_wrist_fraction < 0.18
            and body_engagement < 0.38
            and collapse_context >= 0.45
        )
        strong_writing = (
            len(recent) >= C.WRITING_MIN_HISTORY
            and writing_context >= 0.68
            and max(float(writing_object_detected), writing_object_persistence) >= 0.35
            and active_wrist_fraction >= C.WRITING_ACTIVE_WRIST_FRAC
            and body_engagement >= C.BODY_ENGAGEMENT_MIN
            and metrics.hands_near_desk >= 0.45
            and social_motion < 0.48
            and not phone_detected
            and not strong_sleep
        )
        true_standing = (
            max(true_standing_pose, standing_history) >= 0.70
            and metrics.leg_visibility >= 0.50
            and metrics.seated_cue < 0.35
            and metrics.head_down < 0.45
        )

        # Behavior competition: strong social/phone/sleep evidence suppresses
        # quiet desk-work defaults, and head-down alone is never writing.
        if not strong_writing:
            scores["WRITING"] *= 0.22
        if active_wrist_fraction >= 0.28 and max(float(writing_object_detected), writing_object_persistence) >= 0.30:
            scores["SLEEPING"] *= 0.18
        if inactive_fraction >= 0.70 and body_engagement < 0.30:
            scores["WRITING"] *= 0.15
        if collapse_context < 0.45:
            scores["SLEEPING"] *= 0.38
        sleep_like = sleep_context >= 0.66 and inactive_fraction >= 0.62 and collapse_context >= 0.45
        if sleep_like:
            scores["TALKING"] *= 0.30
            scores["USING_PHONE"] *= 0.60
            scores["ATTENTIVE"] *= 0.65
        if social_motion >= 0.45 and not sleep_like:
            scores["WRITING"] *= 0.35
            scores["TALKING"] = max(scores["TALKING"], 0.58 + 0.35 * social_motion)
        if scores["USING_PHONE"] >= 0.55:
            scores["ATTENTIVE"] *= 0.55
            scores["WRITING"] *= 0.45

        if strong_sleep:
            scores["SLEEPING"] = max(scores["SLEEPING"], 0.76)
            scores["WRITING"] *= 0.35
            scores["ATTENTIVE"] *= 0.40
            scores["STANDING"] *= 0.10
            scores["TALKING"] *= 0.25
            scores["DISTRACTED"] *= 0.45
            scores["USING_PHONE"] *= 0.45
        elif sleep_context >= 0.62 and inactive_fraction >= 0.60:
            scores["ATTENTIVE"] *= 0.60

        if strong_writing:
            scores["WRITING"] = max(scores["WRITING"], 0.78)
            scores["ATTENTIVE"] *= 0.55
            scores["SLEEPING"] *= 0.45
            scores["STANDING"] *= 0.10
            scores["TALKING"] *= 0.45
            scores["DISTRACTED"] *= 0.70

        if true_standing:
            scores["STANDING"] = max(scores["STANDING"], 0.80)
            scores["WRITING"] *= 0.45
            scores["ATTENTIVE"] *= 0.65
            scores["SLEEPING"] *= 0.40

        if side_persistence >= 0.45 and scores["WRITING"] < 0.50 and not strong_sleep:
            scores["TALKING"] = max(scores["TALKING"], 0.76)
            scores["DISTRACTED"] = min(max(scores["DISTRACTED"], 0.58), scores["TALKING"] - 0.08)

        if true_standing:
            scores["STANDING"] = max(scores["STANDING"], 0.78)
            scores["ATTENTIVE"] *= 0.65

        # Seated classroom crops often show shoulders/hips but no knees/ankles.
        # Do not call those standing unless the box is tall or lower body exists.
        seated_or_desk = max(metrics.seated_cue, metrics.hands_near_desk, metrics.head_down)
        if metrics.leg_visibility < 0.50 and metrics.bbox_aspect < 2.45:
            scores["STANDING"] *= 0.08
        if seated_or_desk > 0.35 and standing_evidence < 0.78:
            scores["STANDING"] *= 0.45
        if scores["WRITING"] >= 0.62:
            scores["STANDING"] *= 0.20
            scores["SLEEPING"] *= 0.55
        if scores["SLEEPING"] >= 0.62:
            scores["STANDING"] *= 0.10
            scores["ATTENTIVE"] *= 0.45

        # Writing and sleeping both involve downward head pose; protect note taking
        # from being mislabeled as sleeping unless stillness/slouch dominate.
        if scores["WRITING"] > 0.48 and metrics.wrist_motion >= C.WRITING_MOTION_MIN:
            scores["SLEEPING"] *= 0.45
            scores["DISTRACTED"] *= 0.75

        if phone_detected:
            scores["USING_PHONE"] = max(scores["USING_PHONE"], 0.72)
        elif scores["WRITING"] > scores["USING_PHONE"] and metrics.hands_near_desk > 0:
            scores["USING_PHONE"] *= 0.70

        if len(recent) < C.SLEEP_MIN_HISTORY:
            scores["SLEEPING"] *= 0.55
        if len(recent) < C.PHONE_MIN_HISTORY and not phone_detected:
            scores["USING_PHONE"] *= 0.65

        max_special = max(scores[b] for b in ("WRITING", "USING_PHONE", "SLEEPING", "STANDING", "TALKING", "DISTRACTED"))
        if max_special < 0.52:
            scores["ATTENTIVE"] = max(scores["ATTENTIVE"], 0.68)

        winner = max(scores, key=scores.get)
        confidence = scores[winner]
        reasons = _reasons(winner, scores, factors, phone_detected)
        attention_factors = {
            "head_pose": safe01((1.0 - metrics.head_side) * (1.0 - metrics.head_up)),
            "posture": safe01(max(upright, scores["WRITING"] * 0.90)),
            "gaze": safe01(max(front_or_back, scores["WRITING"] * 0.85)),
            "movement": safe01(max(1.0 - excessive_motion, writing_motion * 0.90)),
            "task_context": safe01(max(scores["ATTENTIVE"], scores["WRITING"]) - 0.45 * max(scores["USING_PHONE"], scores["SLEEPING"])),
        }
        attention = 100.0 * sum(
            C.ATTENTION_WEIGHTS[name] * attention_factors[name]
            for name in C.ATTENTION_WEIGHTS
        )
        if winner in ("USING_PHONE", "SLEEPING"):
            attention *= 0.35
        elif winner in ("DISTRACTED", "TALKING"):
            attention *= 0.65

        return FrameAssessment(
            behavior=BehaviorScore(winner, confidence, factors=factors, reasons=reasons),
            scores=scores,
            attention_score=safe01(attention / 100.0) * 100.0,
            attention_factors=attention_factors,
        )


def _band_score(value: float, low: float, high: float) -> float:
    if value < low:
        return safe01(value / max(low, 1e-6))
    if value > high:
        return safe01(1.0 - (value - high) / max(high, 1e-6))
    return 1.0


def _recent_mean(history: list[dict], key: str, fallback: float) -> float:
    vals = [item.get("metrics", {}).get(key) for item in history[-C.HISTORY_FRAMES:] if item.get("metrics", {}).get(key) is not None]
    if not vals:
        return fallback
    return float(np.mean(vals + [fallback]))


def _recent_fraction(history: list[dict], key: str, threshold: float, fallback: float) -> float:
    vals = [item.get("metrics", {}).get(key) for item in history[-C.HISTORY_FRAMES:] if item.get("metrics", {}).get(key) is not None]
    vals.append(fallback)
    return sum(1 for value in vals if float(value) >= threshold) / max(len(vals), 1)


def _recent_rise(history: list[dict], current: float) -> float:
    vals = [item.get("metrics", {}).get("vertical_extension") for item in history[-20:] if item.get("metrics", {}).get("vertical_extension") is not None]
    if len(vals) < 5:
        return 0.0
    return safe01(current - float(np.mean(vals[: max(1, len(vals) // 2)])))


def _reasons(label: str, scores: dict[str, float], factors: dict[str, float], phone_detected: bool) -> list[str]:
    reasons = []
    if phone_detected:
        reasons.append("phone object detected")
    for key, text in [
        ("head_down", "head angled down"),
        ("head_side", "frequent side orientation"),
        ("upright", "upright posture"),
        ("desk_hands", "hands near desk"),
        ("hands_close", "hands close together"),
        ("stillness", "low movement"),
        ("back_view", "rear-view shoulder cues"),
        ("vertical_extension", "standing height profile"),
        ("peer_facing", "side-facing peer interaction"),
        ("seated_cue", "seated desk posture"),
    ]:
        if factors.get(key, 0.0) >= 0.55:
            reasons.append(text)
    if not reasons:
        reasons.append(f"{label.lower()} score {scores[label]:.2f}")
    return reasons[:4]
