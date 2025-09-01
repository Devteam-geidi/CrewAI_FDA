from crewai_email_classifier.agents.base_llm_agent import BaseLLMAgent
from crewai_email_classifier.utils.classification_result import ClassificationResult

class LLMEmailAgent(BaseLLMAgent):
    def __init__(self, route_name, prompt_template, confidence=0.8, known_routes=None, model: str | None = None):  # ← NEW
        super().__init__(
            route_name=route_name,
            prompt_template=prompt_template,
            confidence=confidence,
            known_routes=known_routes,
            model=model  # ← NEW
        )

    def classify(self, subject: str, body: str, sender: str, attachment_text: str) -> ClassificationResult:
        return super().classify(subject, body, sender, attachment_text)
