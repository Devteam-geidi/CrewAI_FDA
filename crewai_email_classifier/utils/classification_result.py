# utils/classification_result.py
# -----------------------------------------------------
# 📦 ClassificationResult
#
# A standardized structure returned by all agents to ensure
# consistent downstream logging, routing, fallback handling,
# and Supabase compatibility.
# -----------------------------------------------------

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class ClassificationResult:
    # route: str
    # reason: str
    # confidence: float
    # fallback_route: Optional[str] = None
    def __init__(self, route, reason, confidence):
        self.route = route
        self.reason = reason
        self.confidence = confidence

    # def to_dict(self) -> Dict[str, Any]:
    #     """
    #     Convert to dictionary for JSON logging, Supabase inserts, etc.
    #     """
    #     return asdict(self)

    def to_dict(self):
        return {
            "route": self.route,
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def needs_human_review(cls, reason: str):
        return cls(route="Human_Review", reason=reason, confidence=0.0)
    # def needs_human_review(cls, reason: str, fallback_route: Optional[str] = None):
    #     """
    #     Factory method to quickly return a human-review fallback object.
    #     """
    #     return cls(
    #         route="Needs Human Review",
    #         reason=reason,
    #         confidence=0.0,
    #         fallback_route=fallback_route
    #     )

    # def is_confident(self, threshold: float = 0.6) -> bool:
    #     """
    #     Helper to check if result meets the minimum threshold.
    #     """
    #     return self.confidence >= threshold
