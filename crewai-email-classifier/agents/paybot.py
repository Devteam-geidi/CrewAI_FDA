from agents.base_agent import BaseEmailClassifierAgent
from utils.classification_result import ClassificationResult

class PayBot(BaseEmailClassifierAgent):
    def __init__(self):
        self.name = "PayBot"
        self.route_name = "Finance_Unpaid"
        self.confidence_score = 0.9
        self.unpaid_keywords = [
            "invoice", "tax invoice", "payment due", "new invoice", "due date"
        ]
        self.overdue_keywords = [
            "overdue", "unpaid", "second notice", "past due", "balance due", "reminder"
        ]
        self.sender_indicators = ["accounts@", "billing@", "finance@"]

    def classify(self, subject: str, body: str, sender: str, attachment_text: str) -> ClassificationResult:
        full_text = f"{subject}\n{body}\n{attachment_text}".lower()
        sender_lower = sender.lower()

        sender_match = any(indicator in sender_lower for indicator in self.sender_indicators)
        unpaid_match = any(keyword in full_text for keyword in self.unpaid_keywords)
        overdue_match = any(keyword in full_text for keyword in self.overdue_keywords)

        if overdue_match:
            return ClassificationResult(
                route="Try Fallback Agent Finance_Overdue",
                reason="Contains overdue-related keywords, better handled by Finance_Overdue.",
                confidence=0.0
            )

        if unpaid_match or sender_match:
            return ClassificationResult(
                route=self.route_name,
                reason=f"Matched unpaid keyword or sender pattern for {self.route_name}",
                confidence=self.confidence_score
            )

        return ClassificationResult.needs_human_review(
            reason=f"No matching unpaid or overdue keywords or sender indicators for {self.route_name}"
        )
