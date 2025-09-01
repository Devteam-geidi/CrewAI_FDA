# 📧 Email Classification API (FastAPI)

from crewai_email_classifier.utils.env_loader import SUPABASE_URL, SUPABASE_EMAILS_URL, SUPABASE_KEY, WEBHOOK_URL
from crewai_email_classifier.agents.agent_factory import AgentFactory

# Manager selection (EscalationAgent default; CrewManagerAgent behind flag)
from crewai_email_classifier.agents.escalation_agent import EscalationAgent
try:
    from crewai_email_classifier.agents.crew_manager_agent import CrewManagerAgent
except Exception:
    CrewManagerAgent = None

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from crewai_email_classifier.utils.classification_result import ClassificationResult
from crewai_email_classifier.utils.thread_utils import get_or_create_thread_id
from crewai_email_classifier.utils.classification_loader import load_classifications  # NEW: per-route thresholds
import requests
import fitz  # PyMuPDF
import os
import re  # NEW: word-boundary + fallback normalization
import traceback
from datetime import datetime
from tempfile import NamedTemporaryFile  # NEW
from contextlib import suppress

# ===============================
# 🔧 Initialization
# ===============================
agent_factory = AgentFactory(yaml_path="config/classification_rules.yaml")

# Build quick lookup for per-route min confidence (optional in YAML)
_route_min_conf = {
    c.get("name"): c.get("min_confidence")
    for c in load_classifications("config/classification_rules.yaml")
    if c.get("name")
}

USE_CREW_MANAGER = os.getenv("USE_CREW_MANAGER") == "1"
manager_agent = (
    CrewManagerAgent(yaml_path="config/classification_rules.yaml")
    if USE_CREW_MANAGER and CrewManagerAgent
    else EscalationAgent(yaml_path="config/classification_rules.yaml")
)

CONFIDENCE_THRESHOLD = 0.6
PDF_MAX_CHARS = 10_000
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "20"))
HTTP_TIMEOUT = (5, 20)  # (connect, read) seconds

app = FastAPI()

# ===============================
# 🔕️ Request Schema
# ===============================
class EmailRequest(BaseModel):
    subject: str
    body: str
    from_address: str
    message_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    attachment_links: Optional[List[str]] = []

# ===============================
# 📄 Extract Text from PDF (robust)
# ===============================
def extract_text_from_pdf(url: str) -> str:
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()

        # Optional content-type sanity check
        ctype = resp.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower():
            print(f"⚠️ Attachment at {url} does not look like a PDF (Content-Type: {ctype})")

        tmp = None
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                tmp = f.name
                f.write(resp.content)

            text_parts = []
            with fitz.open(tmp) as doc:
                page_count = min(len(doc), PDF_MAX_PAGES)
                for i in range(page_count):
                    text_parts.append(doc[i].get_text())
            text = "".join(text_parts).strip() or "[No extractable text found]"

            if len(text) > PDF_MAX_CHARS:
                print(f"⚠️ Attachment too long: truncating to {PDF_MAX_CHARS} characters.")
                text = text[:PDF_MAX_CHARS] + "\n\n[...truncated...]"

            return text
        finally:
            if tmp:
                with suppress(Exception):
                    os.remove(tmp)
    except requests.Timeout:
        print(f"❌ PDF fetch timed out: {url}")
        return "[Error extracting attachment text: timeout]"
    except requests.RequestException as e:
        print(f"❌ Error downloading PDF at {url}: {e}")
        return "[Error extracting attachment text]"
    except Exception as e:
        print(f"❌ PDF parse error for {url}: {e}")
        return "[Error extracting attachment text]"

# ===============================
# 📜 Supabase Logger
# ===============================
def log_to_supabase(subject, body, message_id, in_reply_to, from_address, thread_id, classification_route, reason):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "subject": subject,
        "body_plain": body,
        "message_id": message_id or "",
        "in_reply_to": in_reply_to or "",
        "from_address": from_address,
        "thread_id": thread_id,
        "received_at": datetime.utcnow().isoformat(),
        "classification": classification_route,
        "reason": reason,
        "is_read": False,
        "is_flagged": False,
        "folder": "inbox"
    }

    try:
        response = requests.post(SUPABASE_EMAILS_URL, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
        print(f"📱 Supabase response: {response.status_code} | {response.text}")
        if response.status_code == 201:
            print("📜 Logged email to Supabase [emails table]")
        else:
            print(f"⚠️ Logging failed: {response.status_code}, {response.text}")
    except requests.Timeout:
        print("❌ Supabase logging timeout")
    except Exception as e:
        print(f"❌ Supabase logging error: {e}")

# ===============================
# 🆕 Build Webhook Payload
# ===============================
def build_webhook_payload(email_subject, email_body, from_address, attachment_text,
                          message_id, in_reply_to, route, reason, confidence, trace, action):
    return {
        "email_subject": email_subject,
        "email_body": email_body,
        "from_address": from_address,
        "attachment_text": attachment_text,
        "message_id": message_id,
        "in_reply_to": in_reply_to,
        "classification": {
            "route": route,
            "reason": reason,
            "confidence": confidence,
            "trace": trace,
            "action_taken": action
        },
        "received_at": datetime.utcnow().isoformat()
    }




# ===============================
# 🧐 Base Classifier (keyword with word boundaries)
# ===============================
def base_classifier(subject, body, sender, attachment_text, agents):
    full_text = f"{subject}\n{body}\n{attachment_text}\n{sender}".lower()

    def has_word(word: str) -> bool:
        # \bword\b with simple sanitation; fallback to substring if regex fails
        try:
            return re.search(rf"\b{re.escape(word.lower())}\b", full_text) is not None
        except re.error:
            return word.lower() in full_text

    for route, agent in agents.items():
        for keyword in getattr(agent, "keywords", []):
            if has_word(keyword):
                reason = f"Matched keyword '{keyword}'"
                print(f"🔍 BaseClassifier matched keyword '{keyword}' for route '{route}'")
                return route, reason

    print("🔍 BaseClassifier found no match.")
    return "Needs Human Review", "No keywords matched."

# ===============================
# 📩 Core Email Handler
# ===============================
def handle_email(email_subject, email_body, from_address, attachment_links=None, message_id=None, in_reply_to=None):
    print(f"📨 Received Email: {email_subject}")
    full_body = email_body
    attachment_text = ""
    classification_trace = []
    thread_id = get_or_create_thread_id(subject=email_subject, in_reply_to=in_reply_to)

    if attachment_links:
        print(f"📌 {len(attachment_links)} attachment(s) found.")
        for link in attachment_links:
            print(f"🔗 Downloading {link}")
            extracted_text = extract_text_from_pdf(link)
            attachment_text += f"\n\n[Attachment Text from {link}]:\n{extracted_text}"

    full_body += attachment_text
    all_agents = agent_factory.get_all_agents()
    initial_route, base_reason = base_classifier(email_subject, full_body, from_address, attachment_text, all_agents)
    classification_trace.append({"stage": "BaseClassifier", "route": initial_route, "reason": base_reason})

    result = None
    used_routes = set()  # only add *actual* agents we run (avoid "Needs Human Review")

    def run_agent(agent_obj, stage_name) -> ClassificationResult:
        print(f"🤖 Running Agent: {stage_name}")
        res = agent_obj.classify(
            subject=email_subject,
            body=full_body,
            sender=from_address,
            attachment_text=attachment_text
        )
        classification_trace.append({
            "stage": stage_name,
            "route": res.route,
            "reason": res.reason
        })
        return res

    # Try agent for the base-selected route (if real)
    agent = agent_factory.get_agent(initial_route)
    if agent:
        used_routes.add(initial_route)
        print(f"✅ Loaded agent for route: {initial_route}")
        result = run_agent(agent, f"LLMAgent {initial_route}")

        # 🔴 NEW: if initial agent returns NHR, let the manager triage/resolve
        if str(result.route) == "Needs Human Review":
            manager_result = manager_agent.run(
                email_subject=email_subject,
                email_body=full_body,
                from_address=from_address,
                attachment_text=attachment_text,
                used_routes=used_routes
            )
            classification_trace.append({
                "stage": manager_agent.name,
                "route": manager_result.route,
                "reason": manager_result.reason
            })
            result = manager_result

        if isinstance(result.route, str) and result.route.startswith("Try Fallback Agent"):
            fallback_route = result.route.replace("Try Fallback Agent", "").strip()
            # Normalize spaces → underscores for safety
            fallback_route = re.sub(r"\s+", "_", fallback_route)
            print(f"🔁 Detected fallback route: {fallback_route}")

            fallback_agent = agent_factory.get_agent(fallback_route)
            if fallback_agent:
                used_routes.add(fallback_route)
                fallback_result = run_agent(fallback_agent, f"Fallback {fallback_route}")

                if fallback_result.route != "Needs Human Review":
                    result = fallback_result
                else:
                    manager_result = manager_agent.run(
                        email_subject=email_subject,
                        email_body=full_body,
                        from_address=from_address,
                        attachment_text=attachment_text,
                        used_routes=used_routes
                    )
                    classification_trace.append({
                        "stage": manager_agent.name,
                        "route": manager_result.route,
                        "reason": manager_result.reason
                    })
                    result = manager_result
            else:
                print(f"⚠️ Fallback agent '{fallback_route}' not found. Running {manager_agent.name}...")
                manager_result = manager_agent.run(
                    email_subject=email_subject,
                    email_body=full_body,
                    from_address=from_address,
                    attachment_text=attachment_text,
                    used_routes=used_routes
                )
                classification_trace.append({
                    "stage": manager_agent.name,
                    "route": manager_result.route,
                    "reason": manager_result.reason
                })
                result = manager_result
    else:
        print(f"❗ No agent found for base route '{initial_route}', running {manager_agent.name}...")
        manager_result = manager_agent.run(
            email_subject=email_subject,
            email_body=full_body,
            from_address=from_address,
            attachment_text=attachment_text,
            used_routes=used_routes
        )
        classification_trace.append({
            "stage": manager_agent.name,
            "route": manager_result.route,
            "reason": manager_result.reason
        })
        result = manager_result

    # ==========================
    # Final decision & logging
    # ==========================
    route = result.route
    reason = result.reason
    confidence = float(result.confidence or 0.0)

    # Per-route threshold (falls back to global)
    threshold = _route_min_conf.get(route, CONFIDENCE_THRESHOLD)
    try:
        threshold = float(threshold) if threshold is not None else CONFIDENCE_THRESHOLD
    except Exception:
        threshold = CONFIDENCE_THRESHOLD

    classification_trace.append({"stage": "Final", "route": route, "reason": reason})
    print(f"📬 Final Route: {route}")
    print(f"📈 Confidence: {confidence} (threshold={threshold})")
    print(f"💡 Reason: {reason}")

    action = "auto_routed" if confidence >= threshold else "human_review"

    if action == "auto_routed":
        log_to_supabase(email_subject, full_body, message_id, in_reply_to, from_address, thread_id, route, reason)
    else:
        print("⚠️ Triggering human review...")
        try:
            resp = requests.post(
                WEBHOOK_URL,
                json={
                    "email_subject": email_subject,
                    "email_body": full_body,
                    "reason": reason,
                    "confidence": confidence,
                    "suggested_route": route,
                    "notes": f"Low confidence – requires human review after {initial_route}"
                },
                timeout=HTTP_TIMEOUT
            )
            print(f"📱 Webhook response: {resp.status_code} | {resp.text}")
        except requests.Timeout:
            print("❌ Human-review webhook timeout")
        except Exception as e:
            print(f"❌ Error calling webhook: {e}")
        log_to_supabase(email_subject, full_body, message_id, in_reply_to, from_address, thread_id, route, reason)

    print("🕿️ Classification Trace:")
    for step in classification_trace:
        print(f"  - {step['stage']}: {step['route']} ({step['reason']})")

    try:
        return {
            "confidence": confidence,
            "suggested_route": str(route or "Needs Human Review"),
            "reason": str(reason or "No reason provided"),
            "action_taken": action,
            "classification_trace": classification_trace,
            "base_reason": base_reason
        }
    except Exception as e:
        print(f"❌ Error formatting final JSON response: {e}")
        return {
            "confidence": 0.0,
            "suggested_route": "Needs Human Review",
            "reason": f"Post-processing error: {str(e)}",
            "action_taken": "human_review",
            "classification_trace": classification_trace,
            "base_reason": base_reason
        }

# ===============================
# 🌐 Endpoint
# ===============================
@app.post("/classify")
def classify_email(request: EmailRequest):
    try:
        response = handle_email(
            email_subject=request.subject,
            email_body=request.body,
            from_address=request.from_address,
            attachment_links=request.attachment_links,
            message_id=request.message_id,
            in_reply_to=request.in_reply_to
        )

        if not isinstance(response, dict):
            raise ValueError("Response is not a valid dictionary")

        return JSONResponse(content=response)

    except Exception as e:
        trace = traceback.format_exc()
        print(f"❌ Classification error: {trace}")
        return JSONResponse(status_code=500, content={
            "message": "JSON could not be generated",
            "code": 500,
            "hint": "See server logs for full stack trace",
            "details": trace
        })

@app.get("/ping")
def ping():
    return {"status": "ok"}
