from utils.classification_result import ClassificationResult
from utils.classification_loader import load_fallback_groups, load_known_routes, load_classifications
from agents.agent_factory import AgentFactory
import os, re
from collections import defaultdict

def _word_hit(word: str, text: str) -> bool:
    try:
        return re.search(rf"\b{re.escape(word.lower())}\b", text) is not None
    except re.error:
        return word.lower() in text

class EscalationAgent:
    """
    Improved EscalationAgent:
    - If we have a hint (a route already tried), restrict to that route's group.
    - Otherwise, score groups by keyword hits and search only the top groups.
    """
    def __init__(self, yaml_path="config/classification_rules.yaml"):
        self._name = "EscalationAgent"
        self.agent_factory = AgentFactory(yaml_path=yaml_path)
        self.known_routes = load_known_routes(yaml_path)
        self.fallback_groups = load_fallback_groups("config/classification_fallback_groups.yaml")
        self.classifications = load_classifications(yaml_path)  # for group scoring
        self.max_fallback_attempts = int(os.getenv("MAX_FALLBACK_ATTEMPTS", "3"))
        self.max_groups_to_try = int(os.getenv("ESCALATION_MAX_GROUPS", "1"))  # try top-1 group by default

        # Build: route -> group, and group -> keywords[]
        self.route_to_group = {}
        self.group_keywords = defaultdict(list)
        for c in self.classifications:
            r = c.get("name")
            g = (c.get("group") or "").strip().lower()
            kws = [kw.strip() for kw in c.get("keywords", []) if kw]
            if r and g:
                self.route_to_group[r] = g
                self.group_keywords[g].extend(kws)

    @property
    def name(self):
        return self._name

    def _infer_groups(self, email_subject: str, email_body: str, from_address: str, attachment_text: str, used_routes: set[str]) -> list[str]:
        # 1) If we already tried a route, use that route's group (most precise).
        for r in used_routes:
            g = self.route_to_group.get(r)
            if g:
                return [g]

        # 2) Otherwise, score groups by keyword hits
        text = f"{email_subject}\n{email_body}\n{attachment_text}\n{from_address}".lower()
        scores = []
        for g, kws in self.group_keywords.items():
            score = 0
            for kw in set(kws):
                if _word_hit(kw, text):
                    score += 1
            scores.append((g, score))

        # sort by score desc, keep top-K with score > 0
        scores.sort(key=lambda t: t[1], reverse=True)
        top = [g for g, s in scores if s > 0][: self.max_groups_to_try]

        # 3) If nothing scored, return empty -> we won't escalate pointlessly
        return top

    def run(self, email_subject, email_body, from_address, attachment_text, used_routes):
        print("🧠 EscalationAgent activated...")

        candidate_groups = self._infer_groups(
            email_subject=email_subject,
            email_body=email_body,
            from_address=from_address,
            attachment_text=attachment_text,
            used_routes=used_routes or set()
        )

        if not candidate_groups:
            print("🧭 No promising groups inferred — escalating directly to human review.")
            return ClassificationResult(
                route="Needs Human Review",
                reason="Low signal; no suitable fallback groups inferred",
                confidence=0.0
            )

        attempts = 0
        for group_name in candidate_groups:
            routes = self.fallback_groups.get(group_name, [])
            print(f"🔎 Escalating within group: {group_name} (routes={routes})")

            for route in routes:
                if attempts >= self.max_fallback_attempts:
                    print(f"⚠️ Reached max fallback attempts ({self.max_fallback_attempts})")
                    break
                if route in (used_routes or set()):
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
                    (used_routes or set()).add(route)
                    continue

                (used_routes or set()).add(route)
                attempts += 1

                if result.route != "Needs Human Review":
                    if result.confidence is None:
                        result.confidence = getattr(agent, "default_confidence", 0.9)
                    print(f"✅ EscalationAgent resolved: {result.route} (Confidence: {result.confidence})")
                    return result
                else:
                    print(f"🤔 Agent {route} returned Needs Human Review")

        print("❌ EscalationAgent could not resolve. Returning Needs Human Review.")
        return ClassificationResult(
            route="Needs Human Review",
            reason="EscalationAgent exhausted candidate groups or attempts",
            confidence=0.0
        )
