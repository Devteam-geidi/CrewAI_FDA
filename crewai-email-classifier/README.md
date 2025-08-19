📬 CrewAI Email Classifier
A robust, modular email classification system that combines rule-based matching, LLM agents, and a manager layer (fallback escalation or Crew-style delegation) to route inbound emails reliably. Built for production workflows: transparent decisions, easy config, and safe human-review handoffs.

🧠 System Architecture
mermaid
Copy code
flowchart TD
    A[Incoming Email (Webhook/API)] --> B[Rule Matcher]
    B -->|Keyword Match| C[Route LLM Agent]
    C -->|Confidence OK| E[Actions + Supabase Log]
    C -->|Low/Uncertain| F[Manager Layer]
    F -->|Default| F1[EscalationAgent (grouped fallbacks)]
    F -->|Flag ON| F2[CrewManagerAgent (hierarchical manager)]
    F1 --> |Resolved| E
    F2 --> |Resolved| E
    F1 -->|Unresolved| H[Human Review]
    F2 -->|Unresolved| H[Human Review]
✅ Key Features
🔍 Hybrid routing: keyword fast-path → per-route LLM agent

🔁 Grouped fallbacks (EscalationAgent): escalate within route groups first

🧑‍✈️ Manager layer options:

EscalationAgent (default): smart, capped fallbacks by group

CrewManagerAgent (optional): triages candidates, delegates to best route

📦 Supabase logging: route, reason, confidence, thread IDs, timestamps

📎 Attachment parsing: safe PDF text extraction (truncated)

🧵 Threading: subject hash + In-Reply-To handling

🧪 Tests: happy path, overdue fallback, unknown → human review

🧩 New route included: Finance_Remittance for POP / remittance advice

📂 Project Structure
bash
Copy code
crewai-email-classifier/
├── main.py
├── agents/
│   ├── agent_factory.py
│   ├── base_agent.py
│   ├── base_llm_agent.py
│   ├── llm_agent.py
│   ├── paybot.py
│   ├── escalation_agent.py          # Replaces the old “ManagerAgent” behavior
│   └── crew_manager_agent.py        # True Crew-style manager (optional)
├── utils/
│   ├── classification_loader.py
│   ├── classification_result.py
│   ├── env_loader.py
│   └── thread_utils.py
├── config/
│   ├── classification_rules.yaml
│   ├── classification_fallback_groups.yaml
│   └── routes.yaml                  # (optional) per-route webhooks/actions
├── scripts/
│   └── generate_fallback_groups.py
└── tests/
    └── test_pipeline.py
🧩 Agents (at a glance)
LLMEmailAgent — generic per-route LLM with strict output parsing:

vbnet
Copy code
Route: <RouteName | Try Fallback Agent <Route> | Needs Human Review>
Reason: <one sentence>
Confidence: <0.0–1.0>
---END---
PayBot — rule-oriented helper for finance “unpaid”

EscalationAgent — scans same-group routes (skips used), tries up to N attempts

CrewManagerAgent — candidate triage + targeted delegation; runs chosen agent

🔐 Environment
Place in .env (loaded by env_loader.py):

ini
Copy code
SUPABASE_URL=<https://xxx.supabase.co>
SUPABASE_EMAILS_URL=<https://xxx.supabase.co/rest/v1/emails>
SUPABASE_KEY=<service-role-or-client-key>
OPENAI_API_KEY=<your-openai-key>
WEBHOOK_URL=<fallback-human-review-webhook>

# Feature flags (optional)
USE_CREW_MANAGER=1                 # enable CrewManagerAgent
MAX_FALLBACK_ATTEMPTS=3            # cap attempts for EscalationAgent
DEFAULT_MANAGER_MODEL=gpt-5        # model for CrewManager triage (or gpt-4o)
PDF_MAX_PAGES=20                   # safety bound for PDF parsing
Windows PowerShell venv tips:

arduino
Copy code
& ".\.venv\Scripts\Activate.ps1"
If you installed ngrok manually, your exe may be here:
C:\Users\rey\Downloads\ngrok-v3-stable-windows-amd64\ngrok.exe

⚙️ Configuration Files
config/classification_rules.yaml
Defines every route:

name, group, description

keywords (for fast path)

use_llm, model, default_confidence

prompt_template (must emit the strict format above)

(optional) min_confidence (per-route threshold override)

Included Finance routes (sample): Finance_Unpaid, Finance_Paid, Finance_Overdue,
Finance_Synergy_Geidi, Finance_Swoop_Geidi, Finance_Atlassian_Geidi,
Finance_Promotional_Relevant, Finance_Remittance, etc.
System: System_Proofpoint, Email_Failed_Delivery, etc.

config/classification_fallback_groups.yaml
Groups related routes. Example:

yaml
Copy code
finance:
  - Finance_Overdue
  - Finance_Unpaid
  - Finance_Paid
  - Finance_Remittance
# ... other groups ...
Generate/refresh with:

bash
Copy code
python scripts/generate_fallback_groups.py
(Optional) config/routes.yaml — per-route webhooks
yaml
Copy code
routes:
  System_Proofpoint:
    webhook: "https://example.com/hooks/proofpoint"
    method: "POST"
  Finance_Remittance:
    webhook: "https://example.com/hooks/remittance"
    method: "POST"
🚀 Run Locally
bash
Copy code
uvicorn main:app --reload
Expose with ngrok (Windows example)
powershell
Copy code
# start ngrok to your local server
& "C:\Users\rey\Downloads\ngrok-v3-stable-windows-amd64\ngrok.exe" http http://localhost:8000

# get the public https URL
(Invoke-RestMethod 'http://127.0.0.1:4040/api/tunnels').tunnels |
  Where-Object { $_.proto -eq 'https' } |
  Select-Object -ExpandProperty public_url
When calling through ngrok, add the header to skip the warning:
ngrok-skip-browser-warning: 1

🌐 API
POST /classify
Request

json
Copy code
{
  "subject": "Invoice for May",
  "body": "Your invoice is due...",
  "from_address": "billing@example.com",
  "message_id": "optional, but recommended",
  "in_reply_to": "optional",
  "attachment_links": ["https://.../file.pdf"]
}
Response (example)

json
Copy code
{
  "confidence": 0.90,
  "suggested_route": "Finance_Overdue",
  "reason": "LLM/agent justification...",
  "action_taken": "auto_routed",
  "classification_trace": [
    {"stage": "BaseClassifier", "route": "Finance_Unpaid", "reason": "Matched keyword 'invoice'"},
    {"stage": "LLMAgent Finance_Unpaid", "route": "Try Fallback Agent Finance_Overdue", "reason": "Past-due wording"},
    {"stage": "Fallback Finance_Overdue", "route": "Finance_Overdue", "reason": "Clear past-due terms"},
    {"stage": "Final", "route": "Finance_Overdue", "reason": "..."}
  ],
  "base_reason": "Matched keyword 'invoice'"
}
GET /ping
json
Copy code
{ "status": "ok" }
🧭 How the Pipeline Decides
Keyword Fast-Path
Word-boundary search across subject + body + attachments + sender. First matching route wins (unless you enable the scoring variant later).

Route LLM Agent
Each route’s prompt enforces strict output:

Route: one of: the route, Try Fallback Agent <Route>, or Needs Human Review

Reason: one sentence

Confidence: 0.0–1.0

End with ---END---

Fallback / Manager Layer

If agent says Try Fallback Agent X → run X

If still unresolved or no base match:

EscalationAgent (default): tries within the same group, capped attempts

CrewManagerAgent (flagged): LLM triage picks best candidate, executes it

Finalization

Compare confidence vs. per-route min_confidence (or global 0.6)

auto_routed → Supabase insert (+ optional per-route webhooks)

human_review → calls WEBHOOK_URL and logs to Supabase

🧪 Testing
bash
Copy code
pytest -v
Covers:

Invoice → unpaid (high confidence)

Overdue → fallback to overdue

Unknown → human review

Manual smoke via ngrok (PowerShell):

powershell
Copy code
$URL = (Invoke-RestMethod 'http://127.0.0.1:4040/api/tunnels').tunnels |
  Where-Object { $_.proto -eq 'https' } |
  Select-Object -ExpandProperty public_url

$Body = @{
  subject = "Remittance Advice for INV-357932"
  body = "Payment remitted via EFT today. POP attached."
  from_address = "accounts@client.com"
  message_id = "dev-remit-001"
  attachment_links = @()
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "$URL/classify" -Method POST -ContentType "application/json" `
  -Headers @{ "ngrok-skip-browser-warning" = "1" } -Body $Body
Expected: Finance_Remittance or fallback to Finance_Paid/Finance_Unpaid.

🛠 Troubleshooting & Tips
Supabase 409 duplicate key during dev
You’re likely replaying the same message_id. This is harmless; treat as idempotent success or generate unique IDs in tests.

Human-review webhook 404 not registered
Your production workflow isn’t active. Point WEBHOOK_URL to a test endpoint during development, or ignore non-2xx responses.

Ngrok banner HTML instead of JSON
Add header ngrok-skip-browser-warning: 1 in your requests.

PDF extraction
Large PDFs are truncated to PDF_MAX_PAGES and PDF_MAX_CHARS for safety.

🧱 Adding a New Route (Playbook)
Add a block to config/classification_rules.yaml:

name, group, keywords

use_llm: true, model, default_confidence

prompt_template with strict output format

Add it to a group in classification_fallback_groups.yaml

(Optional) Add a per-route webhook in routes.yaml

Restart Uvicorn — AgentFactory auto-loads the new agent

🔒 Security (recommended)
Protect /classify with an API key header (e.g., X-API-Key)

Limit/max size on incoming body and attachment content

Use ngrok authtoken + reserved domain for stable webhooks

Store service keys in .env (never commit)

🗒️ Changelog (recent)
New: Finance_Remittance route for remittance advice / POP

New: CrewManagerAgent (triage + delegate) behind USE_CREW_MANAGER

Change: Replaced legacy “ManagerAgent” with EscalationAgent by default

Enhancement: Word-boundary keyword matcher; safer PDF extraction; optional per-route min_confidence

👩‍💻 Maintainers
System design & orchestration: @Geidi_Dev

Dev agent strategy & YAML prompt architecture: CrewAI





# local setup
add .venv 
: python -m venv .venv
To activate .venv 
: .venv\Scripts\activate.ps1

Install all dependencies listted in requirements.txt

To run
run ngrok
: ngrok http 8000     
then in another terminal under the same root project directory
: uvicorn main:app
