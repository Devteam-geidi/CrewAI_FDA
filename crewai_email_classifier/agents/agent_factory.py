# agents/agent_factory.py
# -----------------------------------------------------
# 📁 AgentFactory
#
# Loads all classification agents defined in the YAML config.
# Supports both LLM-based and static rule-based agents (like PayBot).
# -----------------------------------------------------

from crewai_email_classifier.utils.classification_loader import load_classifications
from crewai_email_classifier.agents.llm_agent import LLMEmailAgent
from crewai_email_classifier.agents.paybot import PayBot

class AgentFactory:
    def __init__(self, yaml_path="config/classification_rules.yaml"):
        self.yaml_path = yaml_path
        self.agents = self._load_agents()

    def _load_agents(self):
        classifications = load_classifications(self.yaml_path)
        agents = {}

        known_routes = {c.get("name") for c in classifications}
        known_routes.add("Needs Human Review")

        for classification in classifications:
            route_name = classification.get("name")
            keywords = classification.get("keywords", [])
            confidence = classification.get("default_confidence", 0.9)
            use_llm = classification.get("use_llm", True)
            model = classification.get("model")  # ← NEW

            prompt_template = classification.get(
                "prompt_template",
                f"Analyze this email. Is it related to '{route_name}'? Email Content:\n{{email_content}}"
            )

            if use_llm:
                print(f"🤖 Loading {route_name} as LLMEmailAgent (model={model or 'gpt-4o'})")  # ← log model
                agent = LLMEmailAgent(
                    route_name=route_name,
                    prompt_template=prompt_template,
                    confidence=confidence,
                    known_routes=known_routes,
                    model=model  # ← NEW
                )
            elif route_name == "Finance_Unpaid":
                print(f"🧾 Loading {route_name} as PayBot (static rule-based agent)")
                agent = PayBot()
            else:
                print(f"⚠️ Unknown static agent for route: {route_name}. Skipping...")
                continue

            agent.keywords = keywords
            agents[route_name] = agent

        return agents

    def get_agent(self, route_name):
        return self.agents.get(route_name)

    def get_all_agents(self):
        return self.agents
