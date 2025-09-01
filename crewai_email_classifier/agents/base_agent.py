# agents/base_agent.py

from abc import ABC, abstractmethod
import os
from utils.classification_result import ClassificationResult
from utils.webhook_sender import send_to_n8n


class BaseEmailClassifierAgent(ABC):
    """
    Abstract base class for all email classification agents.
    Subclasses must implement the classify method and return a ClassificationResult.
    """
    # Load webhook URL from environment (configurable per deployment)
    N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "https://geidi.app.n8n.cloud/webhook/forward")

    @property
    def name(self) -> str:
        """Human-readable name for this agent."""
        return self.__class__.__name__

    @abstractmethod
    def classify(self, subject: str, body: str, sender: str, attachment_text: str) -> ClassificationResult:
        """
        Analyze the given email content and return a structured classification result.

        :param subject: Email subject line
        :param body: Full email body (including parsed attachments)
        :param sender: Sender email address
        :param attachment_text: Extracted text from PDF or other attachments
        :return: ClassificationResult(route, confidence, reason[, fallback_route])
        """
        pass

    def classify_and_send(self, subject: str, body: str, sender: str, attachment_text: str) -> ClassificationResult:
        """
        Wrapper: Calls classify(), then automatically forwards the result to n8n.
        """
        result = self.classify(subject, body, sender, attachment_text)

        if isinstance(result, ClassificationResult):
            payload = {
                "agent": self.name,
                "subject": subject,
                "sender": sender,
                "route": result.route,
                "reason": result.reason,
                "confidence": result.confidence,
            }
            send_to_n8n(payload, self.N8N_WEBHOOK_URL)

        return result

    def log_debug_info(self, subject: str, body: str, sender: str, attachment_text: str):
        """
        Optional debug method that agents can call to output diagnostic information.
        """
        print(f"📨 Debug [{self.name}]:")
        print(f"  - Subject: {subject}")
        print(f"  - Sender: {sender}")
        print(f"  - Body Preview: {body[:150]}")
        print(f"  - Attachment Preview: {attachment_text[:150]}")

    def __str__(self):
        return f"<EmailAgent: {self.name}>"
