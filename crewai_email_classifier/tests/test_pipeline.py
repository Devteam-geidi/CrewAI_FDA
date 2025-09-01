# tests/test_pipeline.py
# Hermetic tests (no network/LLM/Supabase). Run with: pytest -q

import os
import sys
import types
import pytest

# ----------------------------
# Ensure required env are set
# ----------------------------
os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("SUPABASE_EMAILS_URL", "http://localhost:54321/rest/v1/emails")
os.environ.setdefault("SUPABASE_API_KEY", "test-supabase-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("WEBHOOK_URL", "http://localhost:1234/webhook")
# Optional flags
os.environ.setdefault("USE_CREW_MANAGER", "0")
os.environ.setdefault("MAX_FALLBACK_ATTEMPTS", "3")
os.environ.setdefault("DEFAULT_LLM_MODEL", "gpt-4o")

# -------------------------------------------------------
# Provide lightweight fakes for supabase / postgrest libs
# (prevents ImportError during utils.thread_utils import)
# -------------------------------------------------------
if "supabase" not in sys.modules:
    fake_supabase = types.ModuleType("supabase")

    class _FakeTable:
        def __init__(self, name):
            self.name = name
        def select(self, *args, **kwargs): return self
        def eq(self, *args, **kwargs): return self
        def execute(self): return types.SimpleNamespace(data=[])  # empty results
        def insert(self, *args, **kwargs):
            # return dummy thread id
            return types.SimpleNamespace(data=[{"id": "thread-1"}])

    class _FakeClient:
        def __init__(self, url, key): pass
        def table(self, name): return _FakeTable(name)

    def create_client(url, key): return _FakeClient(url, key)

    fake_supabase.create_client = create_client
    sys.modules["supabase"] = fake_supabase

if "postgrest" not in sys.modules:
    fake_postgrest = types.ModuleType("postgrest")
    fake_exc_mod = types.ModuleType("postgrest.exceptions")
    class _APIError(Exception): pass
    fake_exc_mod.APIError = _APIError
    fake_postgrest.exceptions = fake_exc_mod
    sys.modules["postgrest"] = fake_postgrest
    sys.modules["postgrest.exceptions"] = fake_exc_mod

# -------------------------------------------
# After env + module fakes, import the system
# -------------------------------------------
from main import handle_email  # noqa: E402

# ----------------------------
# Pytest fixtures for mocking
# ----------------------------
@pytest.fixture(autouse=True)
def mock_thread_utils(monkeypatch):
    # Avoid any real DB lookups; return stable thread id
    monkeypatch.setattr(
        "utils.thread_utils.get_or_create_thread_id",
        lambda subject, in_reply_to: "thread-1",
        raising=True,
    )

@pytest.fixture(autouse=True)
def mock_requests(monkeypatch):
    """Mock requests.get/post for PDF fetch, webhook, and Supabase insert."""
    calls = {"get": [], "post": []}

    class FakeResp:
        def __init__(self, status=200, text="ok", content=b"%PDF-1.4 ...", headers=None):
            self.status_code = status
            self.text = text
            self.content = content
            self.headers = headers or {"Content-Type": "application/pdf"}
        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    def fake_get(url, timeout=None, **kwargs):
        calls["get"].append({"url": url, "timeout": timeout})
        # Simulate timeout if URL contains 'timeout.pdf'
        if isinstance(url, str) and url.endswith("timeout.pdf"):
            import requests as _rq
            raise _rq.Timeout("simulated timeout")
        return FakeResp()

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls["post"].append({"url": url, "json": json, "timeout": timeout})
        # Always return 201 for Supabase insert; 200 otherwise
        status = 201 if "rest/v1/emails" in url else 200
        return FakeResp(status=status, text="ok", headers={"Content-Type": "application/json"})

    monkeypatch.setattr("requests.get", fake_get, raising=True)
    monkeypatch.setattr("requests.post", fake_post, raising=True)
    return calls

@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """
    Patch the OpenAI client used by BaseLLMAgent so no real API calls happen.
    The logic below routes based on keywords found in the prompt.
    """
    import agents.base_llm_agent as base_llm_mod

    class _FakeChoicesMessage:
        def __init__(self, content): self.content = content

    class _FakeChoices:
        def __init__(self, content): self.message = _FakeChoicesMessage(content)

    class _FakeResponse:
        def __init__(self, content): self.choices = [_FakeChoices(content)]

    class _FakeCompletions:
        def create(self, model, messages, temperature):
            # messages: [{"role":"system","content": prompt_template_filled}]
            content = messages[0]["content"].lower()
            if "force-missing-end" in content:
                # simulate LLM forgetting the ---END--- marker
                return _FakeResponse("Route: Finance_Unpaid\nReason: Missing end marker")
            if ("overdue" in content) or ("past due" in content) or ("second reminder" in content):
                return _FakeResponse("Route: Finance_Overdue\nReason: Past due wording\n---END---")
            if ("invoice" in content) or ("tax invoice" in content) or ("due date" in content):
                return _FakeResponse("Route: Finance_Unpaid\nReason: Invoice detected\n---END---")
            return _FakeResponse("Route: Needs Human Review\nReason: No strong signal\n---END---")

    class _FakeChat:
        def __init__(self): self.completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, api_key=None): self.chat = _FakeChat()

    monkeypatch.setattr(base_llm_mod, "OpenAI", _FakeOpenAI, raising=True)

# ------------- #
# Test cases    #
# ------------- #

def test_invoice_should_classify_as_unpaid(mock_requests):
    result = handle_email(
        email_subject="Invoice for May",
        email_body="Here’s the tax invoice for this month. Please pay before the due date.",
        from_address="billing@example.com"
    )
    assert isinstance(result, dict)
    assert result["suggested_route"] == "Finance_Unpaid"
    assert result["confidence"] >= 0.6
    print("✅ test_invoice_should_classify_as_unpaid passed.")

def test_overdue_should_trigger_fallback_or_classify_overdue(mock_requests):
    result = handle_email(
        email_subject="2nd Notice: Overdue Payment",
        email_body="Your balance is past due. This is the second reminder.",
        from_address="accounts@client.com"
    )
    assert isinstance(result, dict)
    assert result["suggested_route"] == "Finance_Overdue"
    assert result["confidence"] >= 0.6
    print("✅ test_overdue_should_trigger_fallback passed.")

def test_unknown_email_goes_to_human_review_and_calls_webhook(mock_requests):
    result = handle_email(
        email_subject="Just saying hi!",
        email_body="Hope your week is going great",
        from_address="random@no-match.com"
    )
    assert isinstance(result, dict)
    assert result["suggested_route"] == "Needs Human Review"
    assert result["confidence"] == 0.0

    # Ensure human-review webhook was called at least once
    assert any("/webhook" in c["url"] for c in mock_requests["post"])
    print("✅ test_unknown_email_goes_to_human_review passed.")

def test_llm_missing_end_marker_yields_human_review(monkeypatch):
    """
    Force the LLM to omit the ---END--- marker by injecting a token in the email
    that our fake OpenAI client looks for.
    """
    result = handle_email(
        email_subject="Please classify FORCE-MISSING-END",
        email_body="(body)",
        from_address="qa@example.com"
    )
    assert isinstance(result, dict)
    assert result["suggested_route"] == "Needs Human Review"
    assert result["confidence"] == 0.0
    print("✅ test_llm_missing_end_marker_yields_human_review passed.")

def test_pdf_timeout_does_not_crash_and_still_classifies(mock_requests):
    result = handle_email(
        email_subject="Invoice with attachment",
        email_body="See attached.",
        from_address="billing@example.com",
        attachment_links=["https://example.com/timeout.pdf"]  # triggers Timeout in fake_get
    )
    assert isinstance(result, dict)
    # With our fake LLM logic, "invoice" should push to Unpaid even if attachment times out
    assert result["suggested_route"] in {"Finance_Unpaid", "Needs Human Review"}
    print("✅ test_pdf_timeout_does_not_crash_and_still_classifies passed.")
