"""Public behavior-classification facade used by the runtime loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pose_utils import extract_pose_metrics
from scoring import BehaviorScoringEngine, BehaviorScore, FrameAssessment, StudentTemporalState


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    attention_score: float
    reasons: list[str]
    factors: dict[str, float]
    raw_label: str
    raw_scores: dict[str, float]


class BehaviorClassifier:
    """Combines pose metrics, object cues, and temporal state per track."""

    def __init__(self):
        self.engine = BehaviorScoringEngine()

    def classify(
        self,
        kps: Optional[np.ndarray],
        bbox: tuple[int, int, int, int],
        state,
        phone_detected: bool = False,
        writing_object_detected: bool = False,
    ) -> ClassificationResult:
        metrics, wrists, body = extract_pose_metrics(
            kps,
            bbox,
            prev_wrists=state.temporal.prev_wrists,
            prev_body=state.temporal.prev_body,
        )
        state.temporal.prev_wrists = wrists
        state.temporal.prev_body = body

        assessment: FrameAssessment = self.engine.assess(
            metrics,
            state.temporal,
            phone_detected=phone_detected,
            writing_object_detected=writing_object_detected,
        )
        stable: BehaviorScore = state.temporal.update(assessment)
        state.last_reasons = stable.reasons
        state.last_factors = stable.factors
        state.raw_scores = assessment.scores

        return ClassificationResult(
            label=stable.label,
            confidence=stable.confidence,
            attention_score=state.temporal.attention_score,
            reasons=stable.reasons,
            factors=stable.factors,
            raw_label=assessment.behavior.label,
            raw_scores=assessment.scores,
        )


__all__ = ["BehaviorClassifier", "ClassificationResult", "StudentTemporalState"]
