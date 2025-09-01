from utils.classification_result import ClassificationResult
from utils.classification_loader import load_classifications, load_known_routes
from agents.agent_factory import AgentFactory
from utils.env_loader import OPENAI_API_KEY
from openai import OpenAI
import os, re

def _word_hit(word: str, text: str) -> bool:
    try:
        return re.search(rf"\b{re.escape(word.lower())}\b", text) is not None
    except re.error:
        return word.lower() in text

class CrewManagerAgent:
    """
    Crew-style manager:
      1) TRIAGE: Ask the LLM to choose the best route from candidates.
      2) EXECUTE: Run that route's agent; return its result if resolved.
      3) Otherwise: Needs Human Review.
    """

    def __init__(self, yaml_path="config/classification_rules.yaml"):
        self._name = "ManagerAgent"  # stable label for traces/dashboards
        self.yaml_path = yaml_path
        self.agent_factory = AgentFactory(yaml_path=yaml_path)
        self.classifications = load_classifications(yaml_path)
        self.known_routes = set(load_known_routes(yaml_path))
        self.model = os.getenv("DEFAULT_MANAGER_MODEL", os.getenv("DEFAULT_LLM_MODEL", "gpt-5"))
        self.top_k = int(os.getenv("MANAGER_TOPK", "6"))

    @property
    def name(self):
        return self._name

    # ---- TRIAGE: pick candidate routes via keyword scoring ----
    def _score_candidates(self, subject: str, body: str, sender: str, attachment_text: str, used_routes: set[str]):
        text = f"{subject}\n{body}\n{attachment_text}\n{sender}".lower()

        # Heavier weights for anchor words
        WEIGHTS = {
            "invoice": 3, "overdue": 4, "past due": 4, "second notice": 3,
            "remittance": 2, "payment": 2, "statement": 2,
        }

        scored = []
        for c in self.classifications:
            route = c.get("name")
            if not route or route in (used_routes or set()) or route == "Needs Human Review":
                continue
            kws = [k for k in c.get("keywords", []) if isinstance(k, str) and k.strip()]
            if not kws:
                continue
            score = 0
            hits = []
            for kw in set(kws):
                if _word_hit(kw, text):
                    hits.append(kw)
                    score += WEIGHTS.get(kw.lower(), 1)
            if score > 0:
                scored.append((route, score, hits))

        if not scored:
            # no obvious candidates → offer a small generic set (dedup)
            fallback = [r for r in self.known_routes if r not in (used_routes or set()) and r != "Needs Human Review"]
            # keep order while deduping
            dedup_fallback = list(dict.fromkeys(fallback))
            return dedup_fallback[: self.top_k], []

        scored.sort(key=lambda t: (t[1], len(t[2])), reverse=True)
        routes = [r for (r, _, _) in scored[: self.top_k]]
        # dedupe in case of duplicate rules in YAML
        routes = list(dict.fromkeys(routes))
        return routes, scored[0][2]  # candidates, best hits (for logging)

    # ---- TRIAGE LLM call ----
    def _triage_with_llm(self, subject: str, body: str, sender: str, attachment_text: str,
                         candidates: list[str], banned: set[str]):
        client = OpenAI(api_key=OPENAI_API_KEY)

        full_email = f"Subject: {subject}\nFrom: {sender}\nBody: {body}\nAttachment: {attachment_text}"
        candidate_list = ", ".join(candidates) if candidates else "(none)"
        banned_list = ", ".join(banned) if banned else "(none)"

        prompt = f"""You are the ManagerAgent for an email-classification system.
You must choose exactly ONE route from the allowed list below (or 'Needs Human Review' if appropriate).
Allowed routes: {candidate_list}
Do NOT choose any of these routes: {banned_list}

Pick the *single best* route that should handle this email. If none clearly applies, output 'Needs Human Review'.

Return ONLY in this format:
Route: <one allowed route name or 'Needs Human Review'>
Reason: <concise one-sentence rationale>
Confidence: <0.0-1.0>
---END---

Email:
{full_email}
"""

        try:
            # Some models force default temperature and reject explicit values.
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                # intentionally OMIT temperature to avoid 400 on models that force defaults
            }
            resp = client.chat.completions.create(**params)
            content = (resp.choices[0].message.content or "").strip()
            print("\n🧠 CrewManager TRIAGE Raw:\n" + "-"*40 + f"\n{content}\n" + "-"*40)

            # Parse the strict format
            route_match = re.search(r"Route:\s*(.+)", content)
            reason_match = re.search(r"Reason:\s*(.+)", content)
            conf_match = re.search(r"Confidence:\s*([0-9.]+)", content)
            end_ok = "---END---" in content

            route = (route_match.group(1).strip() if route_match else "Needs Human Review")
            reason = (reason_match.group(1).strip() if reason_match else "No reason provided")
            try:
                conf = float(conf_match.group(1)) if conf_match else 0.0
            except Exception:
                conf = 0.0

            if not end_ok:
                return "Needs Human Review", "Triage format error (missing ---END---)", 0.0

            # Enforce allowed set
            if route not in candidates and route != "Needs Human Review":
                return "Needs Human Review", f"Triage picked unknown/blocked route '{route}'", 0.0

            return route, reason, conf

        except Exception as e:
            print(f"❌ CrewManager triage error: {e}")
            return "Needs Human Review", f"Triage exception: {e}", 0.0

    # ---- PUBLIC: run() ----
    def run(self, email_subject, email_body, from_address, attachment_text, used_routes):
        print("🧑‍✈️ CrewManagerAgent engaged...")
        used_routes = used_routes or set()

        candidates, hits = self._score_candidates(
            subject=email_subject,
            body=email_body,
            sender=from_address,
            attachment_text=attachment_text,
            used_routes=used_routes
        )
        print(f"🗂️ Manager candidates: {candidates} \n(top hits: {hits})")

        if not candidates:
            return ClassificationResult.needs_human_review("No viable candidates for manager triage.")

        route, reason, m_conf = self._triage_with_llm(
            subject=email_subject,
            body=email_body,
            sender=from_address,
            attachment_text=attachment_text,
            candidates=candidates,
            banned=used_routes
        )

        if route == "Needs Human Review":
            return ClassificationResult.needs_human_review(reason or "Manager triage uncertain.")

        # Execute the chosen route's agent
        agent = self.agent_factory.get_agent(route)
        if not agent:
            return ClassificationResult.needs_human_review(f"Manager chose route '{route}' but no agent found.")

        print(f"🧭 Manager selected route: {route} — executing agent...")
        try:
            result = agent.classify(
                subject=email_subject,
                body=email_body,
                sender=from_address,
                attachment_text=attachment_text
            )
        except Exception as e:
            return ClassificationResult.needs_human_review(f"Route agent error for '{route}': {e}")

        # If the route agent resolved, return it. Otherwise, return NHR with manager reasoning.
        if result.route != "Needs Human Review":
            if result.confidence is None:
                result.confidence = max(m_conf, getattr(agent, "default_confidence", 0.8))
            print(f"✅ Manager path resolved via {route} (confidence={result.confidence})")
            return result

        return ClassificationResult.needs_human_review(
            f"Manager chose '{route}' but agent returned NHR: {result.reason}"
        )
