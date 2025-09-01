from crewai_email_classifier.utils.classification_result import ClassificationResult
from crewai_email_classifier.utils.classification_loader import load_fallback_groups, load_known_routes
from crewai_email_classifier.agents.agent_factory import AgentFactory

class ManagerAgent:
    def __init__(self, yaml_path="config/classification_rules.yaml"):
        self._name = "ManagerAgent"
        self.agent_factory = AgentFactory(yaml_path=yaml_path)
        self.known_routes = load_known_routes(yaml_path)
        self.fallback_groups = load_fallback_groups("config/classification_fallback_groups.yaml")

    @property
    def name(self):
        return self._name

    def run(self, email_subject, email_body, from_address, attachment_text, used_routes):
        print("🧠 ManagerAgent activated...")
        email_content = f"{email_subject}\n{email_body}\n{attachment_text}\n{from_address}".lower()

        MAX_FALLBACK_ATTEMPTS = 3  # 🔁 Limit to avoid excessive retries
        attempts = 0

        for group_name, routes in self.fallback_groups.items():
            print(f"🔎 Scanning fallback group: {group_name}")
            for route in routes:
                if attempts >= MAX_FALLBACK_ATTEMPTS:
                    print(f"⚠️ Reached max fallback attempts ({MAX_FALLBACK_ATTEMPTS})")
                    break

                if route in used_routes:
                    print(f"⏩ Skipping already used route: {route}")
                    continue

                agent = self.agent_factory.get_agent(route)
                if not agent:
                    print(f"⚠️ No agent found for route: {route}")
                    continue

                print(f"🤖 Trying agent: {route}")
                try:
                    result = agent.classify(
                        subject=email_subject,
                        body=email_body + "\n\n" + attachment_text,
                        sender=from_address,
                        attachment_text=attachment_text
                    )
                except Exception as e:
                    print(f"❌ Error running agent {route}: {e}")
                    used_routes.add(route)
                    continue

                used_routes.add(route)
                attempts += 1  # ✅ Count toward fallback limit

                if result.route != "Needs Human Review":
                    if result.confidence is None:
                        result.confidence = getattr(agent, "default_confidence", 0.9)
                    print(f"✅ ManagerAgent resolved: {result.route} (Confidence: {result.confidence})")
                    return result
                else:
                    print(f"🤔 Agent {route} returned Needs Human Review")

        print("❌ ManagerAgent could not resolve. Returning Needs Human Review.")
        return ClassificationResult(
            route="Needs Human Review",
            reason="ManagerAgent exhausted fallback groups",
            confidence=0.0
        )
