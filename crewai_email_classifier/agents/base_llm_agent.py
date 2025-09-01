# -----------------------------------------------------
# 🤖 BaseLLMAgent
# -----------------------------------------------------

from crewai_email_classifier.utils.env_loader import OPENAI_API_KEY
from crewai_email_classifier.utils.classification_result import ClassificationResult
from crewai_email_classifier.utils.classification_loader import load_known_routes
from openai import OpenAI, OpenAIError
from crewai_email_classifier.agents.base_agent import BaseEmailClassifierAgent
import re
import os  # ← NEW
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HELICONE_API_KEY = os.getenv("HELICONE_API_KEY", "")
HELICONE_BASE_URL = os.getenv("HELICONE_BASE_URL", "https://oai.helicone.ai/v1")

class BaseLLMAgent(BaseEmailClassifierAgent):
    def __init__(self, route_name, prompt_template, confidence=0.8, known_routes=None, model: str | None = None):  # ← NEW param
        self.route_name = route_name
        self.prompt_template = prompt_template
        self.confidence = confidence
        self.known_routes = known_routes or load_known_routes()
        # Default model comes from env, falls back to gpt-4o
        self.model = model or os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")  # ← NEW

    def classify(self, subject: str, body: str, sender: str, attachment_text: str) -> ClassificationResult:
        try:
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=HELICONE_BASE_URL,
                default_headers={
                    "Helicone-Auth": f"Bearer {HELICONE_API_KEY}",
                    "Helicone-User-Id": os.getenv("HELICONE_USER_ID", "finance-mailbot"),
                    "Helicone-Session-Id": self.route_name,
                    "Helicone-Cache-Enabled": os.getenv("HELICONE_CACHE_ENABLED", "false"),
                    "Helicone-Property-App": "email-classifier",
                    "Helicone-Property-Env": os.getenv("APP_ENV", "prod"),
                }
            )

            full_email = f"Subject: {subject}\nFrom: {sender}\nBody: {body}\nAttachment: {attachment_text}"
            prompt = self.prompt_template.format(email_content=full_email)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0
            )

            try:
                content = (
                    response.choices[0].message.content.strip()
                    if response.choices and hasattr(response.choices[0], "message") and response.choices[0].message.content
                    else ""
                )
            except Exception as e:
                print(f"❌ Could not extract content from LLM response: {e}")
                print(f"📭 Raw response: {response}")
                return ClassificationResult.needs_human_review("Malformed response from LLM.")

            print("\n🧠 LLM Raw Response:\n" + "-" * 40)
            print(content or "[Empty response]")
            print("-" * 40)

            if not content:
                return ClassificationResult.needs_human_review("Empty response from LLM.")
            if "---END---" not in content:
                print("⚠️ Missing ---END--- delimiter.")
                return ClassificationResult.needs_human_review("Missing ---END--- marker.")

            route = None
            reason = None
            fallback_route = None

            route_match = re.search(r"(?i)^route\s*:\s*(.+)", content, re.MULTILINE)
            reason_match = re.search(r"(?i)^reason\s*:\s*(.+)", content, re.MULTILINE)

            # 🛠 Extract and normalize reason
            if reason_match:
                reason = reason_match.group(1).strip()
                if reason.endswith("---END---"):
                    reason = reason.replace("---END---", "").strip()
                if not reason:
                    reason = "No justification provided."
            else:
                reason = "No justification provided."

            # 🛠 Extract and handle route
            if route_match:
                raw_route = route_match.group(1).strip()

                # 🔁 Fallback suggestion
                if re.match(r"(?i)^try fallback agent\s+", raw_route):
                    fallback_candidate = re.sub(r"(?i)^try fallback agent\s+", "", raw_route).strip()
                    normalized_fallback = fallback_candidate.replace(" ", "_")
                    if normalized_fallback in self.known_routes:
                        print(f"🔁 Detected fallback route: {normalized_fallback}")
                        return ClassificationResult(
                            route=f"Try Fallback Agent {normalized_fallback}",
                            reason=reason or "LLM requested fallback.",
                            confidence=0.0
                        )
                    else:
                        print(f"⚠️ Unknown fallback route: '{normalized_fallback}' (original: '{fallback_candidate}')")
                        return ClassificationResult.needs_human_review(f"Unknown fallback route: {fallback_candidate}")

                elif raw_route.lower() == "needs human review":
                    route = "Needs Human Review"
                else:
                    normalized = raw_route.replace(" ", "_")
                    if normalized in self.known_routes:
                        route = normalized
                    else:
                        print(f"⚠️ Unknown route: '{raw_route}' → normalized as '{normalized}'")
                        route = "Needs Human Review"
            else:
                print("⚠️ Route line missing or malformed.")
                return ClassificationResult.needs_human_review("Route line missing in LLM response.")

            return ClassificationResult(
                route=route,
                reason=reason,
                confidence=self.confidence if route != "Needs Human Review" else 0.0
            )

        except OpenAIError as e:
            print(f"❌ LLM classification failed: {e}")
            return ClassificationResult.needs_human_review(f"LLM error: {str(e)}")
        except Exception as ex:
            print(f"❌ Unexpected error in LLM agent: {ex}")
            return ClassificationResult.needs_human_review("Unexpected parsing error in LLM agent.")
