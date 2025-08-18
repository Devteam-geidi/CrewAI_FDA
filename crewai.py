# ===================================================
# 1. Imports and Environment Setup
# ===================================================
import os
import math
import time
import logging
import json
import gc
import yaml
import asyncio
import re
import hashlib
import tldextract
from typing import List, Dict, Any, Optional, Union, Tuple

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import functions_framework
from langchain_openai import ChatOpenAI
import openai
from rapidfuzz import fuzz  # pip install rapidfuzz

# ===================================================
# 2. Model Definitions
# ===================================================
class EmailAttachment(BaseModel):
    ID: Optional[str] = Field(None, alias="id")
    Name: str = Field(..., alias="name")
    ContentType: Optional[str] = Field(None, alias="contentType")
    Size: Optional[int] = Field(None, alias="size")
    ContentID: Optional[str] = Field(None, alias="contentId")
    ContentBytes: Optional[str] = Field(None, alias="contentBytes")
    Link: Optional[str] = Field(None, alias="link")

    class Config:
        extra = "ignore"
        allow_population_by_field_name = True

class EmailData(BaseModel):
    Subject: str = Field(default="No Subject")
    WebLink: Optional[str] = Field(default="")
    SenderName: Optional[str] = Field(default="")
    SenderEmail: Optional[str] = Field(default="")
    FromName: Optional[str] = Field(default="")
    FromEmail: str = Field(default="")
    ToRecipients: List[str] = Field(default_factory=list)
    CCRecipients: List[str] = Field(default_factory=list)
    PlainTextEmailBody: str = Field(default="")
    ID: Optional[str] = Field(default="")
    ConversationID: Optional[str] = Field(default="")
    HasAttachments: bool = Field(default=False)
    ReceivedDateTime: Optional[str] = Field(default="")
    Attachments: List[EmailAttachment] = Field(default_factory=list)
    Account: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "Subject": "Meeting Tomorrow",
                "FromEmail": "user@example.com",
                "PlainTextEmailBody": "Hello, let's meet tomorrow at 10 AM.",
                "ToRecipients": ["recipient@example.com"]
            }
        }

class ClassificationRequest(BaseModel):
    emails: List[Union[Dict[str, Any], EmailData]]

class ProcessingResult(BaseModel):
    sub_category: Optional[str] = None
    condition: Optional[str] = None
    next_action: Optional[Union[str, List[str]]] = None
    summary: Optional[str] = None  # New field for summary
    rationale: Optional[str] = None
    raw_output: Optional[str] = None
    error: Optional[str] = None
    
    # Jira-compatible ID fields
    jira_status_id: Optional[int] = None
    jira_entity_id: Optional[int] = None
    jira_task_type_id: Optional[int] = None
    jira_assignee_id: Optional[str] = None
    jira_transition_id: Optional[int] = None

class EmailResult(BaseModel):
    email: Dict[str, Any]
    category: str
    processing_result: ProcessingResult

class ClassificationResponse(BaseModel):
    status: str
    processed: int
    results: List[EmailResult]

class MemoryStatus(BaseModel):
    status: str
    memory_usage_mb: float
    memory_threshold_mb: int
    crews_cache_size: int
    persistent_agents_count: int
    estimated_memory_per_crew_mb: float
    max_cache_size: int
    cached_crews: List[str]
    request_counter: int
    uptime_seconds: int

class OptimizeResponse(BaseModel):
    status: str
    memory_before_mb: float
    memory_after_mb: float
    memory_saved_mb: float
    crews_remaining: int

class ReloadConfigResponse(BaseModel):
    status: str
    message: str
    force_reset: bool
    persistent_agents_count: int
    crews_count: int

# ===================================================
# 3. Application Configuration & Global State
# ===================================================
logging.basicConfig(level=logging.INFO)
app = FastAPI(
    title="Email Classification API",
    description="Service for classifying and processing emails using CrewAI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY! Set it with export OPENAI_API_KEY='your-api-key'")

GITLAB_REPO_ID = os.getenv("GITLAB_REPO_ID", "geidi2%2Fcpi_report")
BRANCH = os.getenv("GITLAB_BRANCH", "finance")
FILE_PATH = os.getenv("GITLAB_FILE_PATH", "finance_process_documentation.yaml")
GITLAB_API_URL = f"https://gitlab.com/api/v4/projects/{GITLAB_REPO_ID}/repository/files/{FILE_PATH}/raw?ref={BRANCH}"
DOMAIN_FILE_PATH = os.getenv("GITLAB_DOMAIN_FILE_PATH", "domain.yaml")
GITLAB_DOMAIN_API_URL = f"https://gitlab.com/api/v4/projects/{GITLAB_REPO_ID}/repository/files/{DOMAIN_FILE_PATH}/raw?ref={BRANCH}"
JIRA_FILE_PATH = os.getenv("GITLAB_JIRA_FILE_PATH", "jira.yaml")
GITLAB_JIRA_API_URL = f"https://gitlab.com/api/v4/projects/{GITLAB_REPO_ID}/repository/files/{JIRA_FILE_PATH}/raw?ref={BRANCH}"
LAST_AGENTS_FILE = "last_agents.json"

# Global state objects
class AppState:
    def __init__(self):
        self.crews_cache = {}
        self.valid_subcategories = set()
        self.valid_actions_reference = set()
        self.persistent_agents = {}
        self.yaml_data_cache = None
        self.domain_yaml_cache = None
        self.START_TIME = time.time()
        self.request_counter = 0
        self.misclassifications = []
        self.last_yaml_hash = None
        self.last_domain_yaml_hash = None
        self.jira_yaml_cache = None
        self.jira_mappings = {}

state = AppState()

# Memory optimization settings
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "10"))
MEMORY_CHECK_INTERVAL = int(os.getenv("MEMORY_CHECK_INTERVAL", "100"))
memory_threshold_mb = int(os.getenv("MEMORY_THRESHOLD_MB", "1024"))
AGENT_REUSE = os.getenv("AGENT_REUSE", "true").lower() == "true"

# ===================================================
# 4. Utility Functions
# ===================================================
def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)

def optimize_memory():
    """Check memory usage and clean up if needed."""
    current_usage = get_memory_usage()
    logging.info(f"Current memory usage: {current_usage:.2f} MB")
    if current_usage > memory_threshold_mb:
        logging.warning(f"Memory usage ({current_usage:.2f} MB) exceeds threshold ({memory_threshold_mb} MB)")
        if len(state.crews_cache) > 1:
            classifier = state.crews_cache.get("classification")
            state.crews_cache.clear()
            if classifier:
                state.crews_cache["classification"] = classifier
        gc.collect()
        logging.info(f"Memory after cleanup: {get_memory_usage():.2f} MB")

def limit_cache_size():
    """Limit the size of crews_cache to MAX_CACHE_SIZE."""
    if len(state.crews_cache) > MAX_CACHE_SIZE:
        logging.info(f"Trimming cache size from {len(state.crews_cache)} to {MAX_CACHE_SIZE}")
        classifier = state.crews_cache.get("classification")
        other_keys = [k for k in state.crews_cache.keys() if k != "classification"]
        keys_to_keep = other_keys[:MAX_CACHE_SIZE-1] if classifier else other_keys[:MAX_CACHE_SIZE]
        new_cache = {}
        if classifier:
            new_cache["classification"] = classifier
        for k in keys_to_keep:
            new_cache[k] = state.crews_cache[k]
        state.crews_cache = new_cache
        gc.collect()

def clean_text(text):
    """Clean text fields."""
    if not text:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ===================================================
# 5. Domain and YAML Configuration Functions
# ===================================================
async def load_yaml_from_gitlab():
    """Fetch and parse the YAML configuration from GitLab asynchronously."""
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(GITLAB_API_URL, headers=headers)
            if response.status_code == 200:
                raw_yaml = response.text
                logging.info("Successfully fetched YAML from GitLab.")
                return yaml.safe_load(raw_yaml), raw_yaml
            else:
                error_msg = f"Failed to fetch YAML: {response.text}"
                logging.error(error_msg)
                raise Exception(error_msg)
    except httpx.ConnectTimeout as e:
        logging.error(f"Timeout connecting to GitLab: {str(e)}")
        raise Exception(f"Timeout connecting to GitLab: {str(e)}")
    except Exception as e:
        logging.error(f"Error fetching YAML from GitLab: {str(e)}")
        raise

async def load_domain_config_from_gitlab():
    """Fetch and parse the domain configuration YAML from GitLab asynchronously."""
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(GITLAB_DOMAIN_API_URL, headers=headers)
            if response.status_code == 200:
                raw_yaml = response.text
                logging.info("Successfully fetched domain config YAML from GitLab.")
                return yaml.safe_load(raw_yaml), raw_yaml
            else:
                raise Exception(f"Failed to fetch domain config YAML: {response.text}")
    except httpx.ConnectTimeout as e:
        logging.error(f"Timeout connecting to GitLab for domain config: {str(e)}")
        raise Exception(f"Timeout connecting to GitLab for domain config: {str(e)}")
    except Exception as e:
        logging.error(f"Error fetching domain config YAML from GitLab: {str(e)}")
        raise

async def load_jira_yaml_from_gitlab():
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(GITLAB_JIRA_API_URL, headers=headers)
        if response.status_code == 200:
            return yaml.safe_load(response.text)
        else:
            raise Exception(f"Failed to fetch jira.yaml: {response.text}")

def extract_domain_config(domain_yaml):
    supplier_mapping = domain_yaml.get('supplier', {})
    client_mapping = domain_yaml.get('client', {})
    internal_domains = domain_yaml.get('internal', {}).get('domains', [])
    return supplier_mapping, client_mapping, internal_domains

def extract_domain_root(email: str) -> str:
    logging.info("Extracting domain from email: %s", email)
    try:
        parts = email.split('@')
        if len(parts) != 2:
            logging.warning("Email format is not valid: %s", email)
            return ""
        extract_result = tldextract.extract(parts[1])
        domain_value = extract_result.domain  # e.g., 'quintis' for 'morne@quintis.com.au'
        logging.info("Extracted domain: %s (from %s)", domain_value, parts[1])
        return domain_value
    except Exception as e:
        logging.error("Error extracting domain from %s: %s", email, e)
        return ""

def extract_jira_mappings(jira_yaml):
    mappings = {
        "status_ids": {item["name"].lower(): item["id"] for item in jira_yaml.get("status_mapping", [])},
        "user_ids": {item["name"].lower(): item["accountId"] for item in jira_yaml.get("user_mapping", [])},
        "task_type_ids": {item["task_type"].lower(): item["id"] for item in jira_yaml.get("task_type_mapping", [])},
        "company_ids": {item["company"].lower(): item["id"] for item in jira_yaml.get("company_mapping", [])},
        "transition_ids": {item["name"].lower(): item["transition_id"] for item in jira_yaml.get("transitions", [])}
    }
    return mappings

async def reload_finance_yaml_if_changed():
    """Reload finance processing YAML from GitLab only if changed."""
    finance_yaml, raw_finance_yaml = await load_yaml_from_gitlab()
    new_finance_hash = compute_hash(raw_finance_yaml)
    
    if state.last_yaml_hash != new_finance_hash:
        logging.info("✅ Finance YAML has changed, reloading...")
        state.last_yaml_hash = new_finance_hash
        state.yaml_data_cache = finance_yaml
        return True
    else:
        logging.info("ℹ️ Finance YAML unchanged")
    return False

async def reload_domain_yaml_if_changed():
    """Reload domain configuration YAML from GitLab only if changed."""
    domain_yaml, raw_domain_yaml = await load_domain_config_from_gitlab()
    new_domain_hash = compute_hash(raw_domain_yaml)

    if state.last_domain_yaml_hash != new_domain_hash:
        logging.info("✅ Domain YAML has changed, reloading...")
        state.last_domain_yaml_hash = new_domain_hash
        state.domain_config = extract_domain_config(domain_yaml)
        return True
    else:
        logging.info("ℹ️ Domain YAML unchanged")
    return False


async def get_domain_config():
    if not hasattr(state, 'domain_config'):
        domain_yaml, _ = await load_domain_config_from_gitlab()
        state.domain_config = extract_domain_config(domain_yaml)
    return state.domain_config

# ===================================================
# 6. Extended Matching Functions
# ===================================================
def match_supplier_or_client_extended(
    sender_email: str,
    subject: str,
    supplier_mapping: Dict[str, Dict[str, Union[str, List[str]]]],
    client_mapping: Dict[str, Dict[str, Union[str, List[str]]]],
    threshold: int = 70
) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to match an incoming sender_email first by exact host/subdomain
    (skipped for generic free-mail domains), then by fuzzy full-address,
    then by local-part, then by subject. Returns (category, matched_name).
    """
    sender_email_lower = sender_email.lower()
    sender_host = sender_email_lower.split("@")[-1]
    subject_lower = subject.lower() if subject else ""

    # --- Step 1: Exact host or subdomain match (skip if generic) ---
    skip_host_match = sender_host in GENERIC_DOMAINS
    if not skip_host_match:
        # Supplier host match
        for name, info in supplier_mapping.items():
            raw = info.get("email")
            emails = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
            for addr in emails:
                host = addr.lower().split("@")[-1]
                if sender_host == host or sender_host.endswith("." + host):
                    logging.info(f"✅ Supplier host match: {sender_host} vs {host} → {name}")
                    return "supplier_emails", name

        # Client host match
        for name, info in client_mapping.items():
            raw = info.get("email")
            emails = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
            for addr in emails:
                host = addr.lower().split("@")[-1]
                if sender_host == host or sender_host.endswith("." + host):
                    logging.info(f"✅ Client host match: {sender_host} vs {host} → {name}")
                    return "client_emails", name

    # --- Step 1.5: Domain-root fuzzy → name fallback ---
    root = extract_domain_root(sender_email_lower)  # e.g. "upwork"
    if root:
        # supplier side
        best_sup, best_sup_score = None, 0
        for name in supplier_mapping:
            score = fuzz.token_sort_ratio(root, name.lower())
            if score > best_sup_score:
                best_sup_score, best_sup = score, name
        if best_sup_score >= threshold:
            logging.info(f"✅ Supplier root-domain match: '{root}' ≈ '{best_sup}' ({best_sup_score})")
            return "supplier_emails", best_sup

        # client side
        best_cli, best_cli_score = None, 0
        for name in client_mapping:
            score = fuzz.token_sort_ratio(root, name.lower())
            if score > best_cli_score:
                best_cli_score, best_cli = score, name
        if best_cli_score >= threshold:
            logging.info(f"✅ Client root-domain match: '{root}' ≈ '{best_cli}' ({best_cli_score})")
            return "client_emails", best_cli

    # --- Step 2: Full-email fuzzy matching ---
    best_supplier_ratio, best_supplier = 0, None
    for name, info in supplier_mapping.items():
        raw = info.get("email")
        emails = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
        for single in emails:
            if isinstance(single, str):
                ratio = fuzz.ratio(sender_email_lower, single.lower())
                if ratio > best_supplier_ratio:
                    best_supplier_ratio, best_supplier = ratio, name
    if best_supplier_ratio >= threshold:
        logging.info(f"✅ Supplier email fuzzy match: {sender_email_lower} matches {best_supplier} ({best_supplier_ratio})")
        return "supplier_emails", best_supplier

    best_client_ratio, best_client = 0, None
    for name, info in client_mapping.items():
        raw = info.get("email")
        emails = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
        for single in emails:
            if isinstance(single, str):
                ratio = fuzz.ratio(sender_email_lower, single.lower())
                if ratio > best_client_ratio:
                    best_client_ratio, best_client = ratio, name
    if best_client_ratio >= threshold:
        logging.info(f"✅ Client email fuzzy match: {sender_email_lower} matches {best_client} ({best_client_ratio})")
        return "client_emails", best_client

    # --- Step 3: Local-part matching ---
    raw_local = sender_email_lower.split("@")[0]
    local_part = re.sub(r"\d+$", "", raw_local)

    supplier_local = [(name, fuzz.partial_ratio(local_part, name.lower())) for name in supplier_mapping]
    if supplier_local:
        best_name, best_score = max(supplier_local, key=lambda x: x[1])
        if best_score >= threshold:
            logging.info(f"✅ Supplier local-part match: '{local_part}' vs '{best_name}' ({best_score})")
            return "supplier_emails", best_name

    client_local = [(name, fuzz.partial_ratio(local_part, name.lower())) for name in client_mapping]
    if client_local:
        best_name, best_score = max(client_local, key=lambda x: x[1])
        if best_score >= threshold:
            logging.info(f"✅ Client local-part match: '{local_part}' vs '{best_name}' ({best_score})")
            return "client_emails", best_name

    # --- Step 4: Subject-based fallback ---
    supplier_subj = [(name, fuzz.partial_ratio(subject_lower, name.lower())) for name in supplier_mapping]
    if supplier_subj:
        best_name, best_score = max(supplier_subj, key=lambda x: x[1])
        if best_score >= threshold:
            logging.info(f"✅ Supplier subject match: '{subject_lower}' vs '{best_name}' ({best_score})")
            return "supplier_emails", best_name

    client_subj = [(name, fuzz.partial_ratio(subject_lower, name.lower())) for name in client_mapping]
    if client_subj:
        best_name, best_score = max(client_subj, key=lambda x: x[1])
        if best_score >= threshold:
            logging.info(f"✅ Client subject match: '{subject_lower}' vs '{best_name}' ({best_score})")
            return "client_emails", best_name

    # --- No match found ---
    logging.info("❌ No supplier or client match found")
    return None, None

def match_against_names(value: str, name_list: List[str], threshold: int = 80) -> bool:
    """
    Improved matching that handles domain suffixes appropriately and prevents
    false matches based on common domain extensions like .com
    """
    if not value or not name_list:
        return False
        
    value_lower = value.lower()
    
    # For email addresses, extract and properly handle domain parts
    if '@' in value_lower:
        # Extract domain components
        domain_part = value_lower.split('@')[1]
        domain_root = extract_domain_root(value_lower)
        
        # Common TLDs that should not influence matching
        common_tlds = ['.com', '.org', '.net', '.co', '.io', '.edu', '.gov']
        
        for candidate in name_list:
            candidate_lower = candidate.lower()
            
            # Direct domain equality
            if domain_part == candidate_lower or domain_root == candidate_lower:
                logging.info(f"Exact domain match: '{domain_part}/{domain_root}' equals '{candidate_lower}'")
                return True
                
            # Clean candidate and domain for matching by removing common TLDs
            clean_candidate = candidate_lower
            clean_domain = domain_root
            
            for tld in common_tlds:
                if clean_candidate.endswith(tld):
                    clean_candidate = clean_candidate[:-len(tld)]
                if clean_domain.endswith(tld):
                    clean_domain = clean_domain[:-len(tld)]
            
            # Now match on cleaned values
            if clean_domain == clean_candidate and len(clean_candidate) > 3:
                logging.info(f"Cleaned domain match: '{clean_domain}' equals '{clean_candidate}'")
                return True
            
            # Domain contains candidate (e.g., "quintis" in "quintis.com.au")
            if clean_candidate in clean_domain and len(clean_candidate) > 3:
                logging.info(f"Domain contains name: '{clean_domain}' contains '{clean_candidate}'")
                return True
                
        # For fuzzy matching, use token_sort_ratio which is less affected by word order and suffixes
        for candidate in name_list:
            candidate_lower = candidate.lower()
            # Token sort ratio helps with reordered words and is less influenced by suffixes
            token_ratio = fuzz.token_sort_ratio(domain_root, candidate_lower)
            
            if token_ratio >= threshold:
                # Validate this isn't just matching on common TLDs
                common_tld_match = False
                for tld in common_tlds:
                    if tld in domain_part and tld in candidate_lower:
                        # If both have same TLD, check if that's the main reason for match
                        if fuzz.ratio(domain_root.replace(tld, ''), candidate_lower.replace(tld, '')) < threshold:
                            common_tld_match = True
                
                if not common_tld_match:
                    logging.info(f"Domain fuzzy match: '{domain_root}' matches '{candidate_lower}' with token_sort_ratio {token_ratio}")
                    return True
                else:
                    logging.info(f"Rejected TLD-based match: '{domain_root}' vs '{candidate_lower}' (matching mainly on common TLD)")
    else:
        # For non-email values, use standard matching
        for candidate in name_list:
            candidate_lower = candidate.lower()
            
            # Exact match
            if value_lower == candidate_lower:
                logging.info(f"Exact string match: '{value_lower}' equals '{candidate_lower}'")
                return True
                
            # Use token_sort_ratio for more resilient matching
            token_ratio = fuzz.token_sort_ratio(value_lower, candidate_lower)
            if token_ratio >= threshold:
                logging.info(f"Token ratio match: '{value_lower}' matches '{candidate_lower}' with ratio {token_ratio}")
                return True
                
    logging.info(f"No match found for '{value_lower}' against name list")
    return False

def get_best_match(text: str, names: List[str], threshold: int = 75) -> str:
    best_match = None
    best_score = threshold
    for name in names:
        score = fuzz.ratio(text.lower(), name.lower())
        if score > best_score:
            best_score = score
            best_match = name
    return best_match if best_match is not None else "Unknown"

# ===================================================
# 7. Email Classification and Processing Functions
# ===================================================
def extract_invoice_number(subject: str, body: str = "") -> str:
    """
    Try to pull an invoice number first from the subject;
    if nothing matches, look in the body. Supports:
      - Invoice #1234
      - Invoice 1234
      - Tax Invoice: 1234
      - INV-1234
      - Invoice (#1234)
      - Any standalone 5+ digit number
    """
    patterns = [
        r'Invoice\s*\(\s*#?([\w\-\/]+)\s*\)',   # Invoice (#1234) or Invoice (1234)
        r'Invoice\s*#?\s*([\w\-\/]+)',          # Invoice #1234 or Invoice 1234
        r'INV[-_ ]?(\d+)',                      # INV-1234 or INV 1234
        r'Tax\s*Invoice[:\-]?\s*([\w\-\/]+)',   # Tax Invoice: 1234
        r'\b(\d{5,})\b'                          # Any standalone 5+ digit number
    ]

    def _search(text: str) -> Optional[str]:
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    # 1) try subject
    invoice = _search(subject or "")
    if invoice:
        return invoice

    # 2) fallback to body
    invoice = _search(body or "")
    if invoice:
        return invoice

    return "N/A"

def build_substitution_context(email_meta: dict, category: str, matched_name: str = None) -> dict:
    context = {
        "email_subject": email_meta.get("Subject", ""),
        "email_attachment": (email_meta.get("Attachments", [{}])[0].get("name") if email_meta.get("Attachments") else ""),
        "entity_name": email_meta.get("FromEmail", "").split("@")[1] if "@" in email_meta.get("FromEmail", "") else "unknown",
        "invoice_number": extract_invoice_number(
           email_meta.get("Subject", ""),
           email_meta.get("PlainTextEmailBody", "")
        ),
        "invoice_attachment": email_meta.get("InvoiceAttachment", ""),
        "invoice_due_date": email_meta.get("InvoiceDueDate", "N/A"),
        "soa_attachment": email_meta.get("SOAAttachment", ""),
        "soa_due_date": email_meta.get("SOADueDate", "N/A"),
        "dispute_file": email_meta.get("DisputeFile", ""),
        "assignee": email_meta.get("Assignee", "default_assignee"),
        "supplier_or_client_owner": "N/A",
        "assigned_person": email_meta.get("AssignedPerson", "N/A"),
        "entity_folder": email_meta.get("EntityFolder", "Default Folder"),
        "responsible_person": email_meta.get("ResponsiblePerson", "N/A"),
        "supplier_name": "Unknown Supplier",  # default fallback
        "client_name": "Unknown Client",
        "supplier_owner": "N/A",
        "client_owner": "N/A",
    }

    if category == "supplier_emails":
        sender_name = email_meta.get("SenderName", "").strip()
        from_name = email_meta.get("FromName", "").strip()
        default_from = (
            email_meta.get("FromEmail", "").split('@')[1].split('.')[0].strip()
            if "@" in email_meta.get("FromEmail", "")
            else ""
        )

        # ✅ Prefer matched_name from YAML match
        supplier_name = matched_name or sender_name or from_name or default_from or "Unknown Supplier"
        logging.info(f"[Context] Supplier Name resolved as: '{supplier_name}' (matched_name: '{matched_name}', sender_name: '{sender_name}', from_name: '{from_name}', fallback: '{default_from}')")
        context["supplier_name"] = supplier_name

    elif category == "client_emails":
        sender_name = email_meta.get("SenderName", "").strip()
        from_name = email_meta.get("FromName", "").strip()
        default_from = (
            email_meta.get("FromEmail", "").split('@')[1].split('.')[0].strip()
            if "@" in email_meta.get("FromEmail", "")
            else ""
        )

        # ✅ Prefer matched_name from YAML match
        client_name = matched_name or sender_name or from_name or default_from or "Unknown Client"
        logging.info(f"[Context] Client Name resolved as: '{client_name}' (matched_name: '{matched_name}', sender_name: '{sender_name}', from_name: '{from_name}', fallback: '{default_from}')")
        context["client_name"] = client_name

    return context


def substitute_placeholders(template: str, context: dict) -> str:
    try:
        template_fixed = re.sub(r'{{\s*(\w+)\s*}}', r'{\1}', template)
        return template_fixed.format(**context)
    except Exception as e:
        logging.error(f"Error substituting placeholders: {e}")
        return template

def parse_output(output_str, valid_subcats):
    try:
        json_match = re.search(r'(\{.*\})', output_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Sanitize common mistakes
            json_str = json_str.replace('\n', ' ').replace('\\', '')
            result = json.loads(json_str)
            
            # Check if result is a dictionary
            if not isinstance(result, dict):
                logging.warning(f"Parsed result is not a dictionary: {type(result)}")
                return {"raw_output": output_str, "rationale": f"Expected dict, got {type(result).__name__}"}
                
            return result
        else:
            start = output_str.find('{')
            end = output_str.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = output_str[start:end]
                logging.info(f"Manually extracted JSON string: {json_str}")
                result = json.loads(json_str)
                
                # Check if result is a dictionary
                if not isinstance(result, dict):
                    logging.warning(f"Manually extracted result is not a dictionary: {type(result)}")
                    return {"raw_output": output_str, "rationale": f"Expected dict, got {type(result).__name__}"}
                    
                return result
            return {"raw_output": output_str, "rationale": "Could not parse JSON."}
    except Exception as e:
        logging.error(f"Parsing error: {str(e)}")
        return {"error": str(e), "raw_output": output_str, "rationale": "Parsing failed."}

def extract_email_metadata(email):
    try:
        # Check if email is a dictionary
        if not isinstance(email, dict):
            logging.error(f"extract_email_metadata - Email is not a dictionary: {type(email)}")
            return {}
            
        # Safely process attachments
        attachments = []
        if "Attachments" in email:
            # Ensure attachments is a list
            if not isinstance(email["Attachments"], list):
                logging.warning(f"extract_email_metadata - Attachments is not a list: {type(email['Attachments'])}")
                email["Attachments"] = []
            
            # Filter out non-dict attachments
            for attachment in email["Attachments"]:
                if isinstance(attachment, dict):
                    attachments.append(attachment)
                else:
                    logging.warning(f"extract_email_metadata - Skipping non-dict attachment: {type(attachment)}")
            
            logging.info(f"extract_email_metadata - Incoming attachments after filtering: {attachments}")
        else:
            logging.warning("extract_email_metadata - No Attachments field in input email")
        
        # Get subject safely
        subject = ""
        if "Subject" in email and isinstance(email["Subject"], str):
            subject = clean_text(email["Subject"])
        else:
            logging.warning(f"extract_email_metadata - Subject is not a string: {type(email.get('Subject', None))}")
            subject = "No Subject"
        
        # Process through Pydantic model with error handling
        if isinstance(email, dict):
            try:
                logging.info("extract_email_metadata - Converting to Pydantic model")
                email_data = EmailData(**email)
                logging.info(f"extract_email_metadata - Pydantic model attachments: {email_data.Attachments}")
                email = email_data.dict()
                logging.info(f"extract_email_metadata - After dict() conversion: {email.get('Attachments')}")
            except Exception as conversion_error:
                logging.error(f"extract_email_metadata - Pydantic conversion error: {str(conversion_error)}")
                # Skip the conversion if it fails
                logging.info("extract_email_metadata - Skipping Pydantic conversion due to error")
        
        # Build metadata with safe access to attributes
        meta = {
            "Subject": clean_text(email.get("Subject", "No Subject") if isinstance(email.get("Subject"), (str, type(None))) else "No Subject"),
            "WebLink": clean_text(email.get("WebLink", "") if isinstance(email.get("WebLink"), (str, type(None))) else ""),
            "SenderName": clean_text(email.get("SenderName", email.get("FromName", "")) if isinstance(email.get("SenderName"), (str, type(None))) else ""),
            "SenderEmail": clean_text(email.get("SenderEmail", email.get("FromEmail", "")) if isinstance(email.get("SenderEmail"), (str, type(None))) else ""),
            "FromName": clean_text(email.get("FromName", "") if isinstance(email.get("FromName"), (str, type(None))) else ""),
            "FromEmail": clean_text(email.get("FromEmail", "") if isinstance(email.get("FromEmail"), (str, type(None))) else ""),
            "ToRecipients": email.get("ToRecipients", []) if isinstance(email.get("ToRecipients"), list) else [],
            "CCRecipients": email.get("CCRecipients", []) if isinstance(email.get("CCRecipients"), list) else [],
            "PlainTextEmailBody": clean_text(email.get("PlainTextEmailBody", email.get("body", "")) if isinstance(email.get("PlainTextEmailBody"), (str, type(None))) else ""),
            "ID": clean_text(email.get("ID", "") if isinstance(email.get("ID"), (str, type(None))) else ""),
            "ConversationID": clean_text(email.get("ConversationID", "") if isinstance(email.get("ConversationID"), (str, type(None))) else ""),
            "HasAttachments": bool(email.get("HasAttachments", False)),
            "ReceivedDateTime": clean_text(email.get("ReceivedDateTime", "") if isinstance(email.get("ReceivedDateTime"), (str, type(None))) else ""),
            "Attachments": attachments
        }
        
        meta["Account"] = clean_text(email.get("Account", ""))
        logging.info(f"extract_email_metadata - Account field: '{meta['Account']}'")

        # Log outgoing attachments
        logging.info(f"extract_email_metadata - Outgoing attachments: {meta['Attachments']}")
        
        # --- Begin: Additional Mailbox Check ---
        # Define the mailbox addresses (in lower case for easier matching)
        target_mailboxes = {
            "accounts@geidi.com",
            "cebuaccounts@geidi.com",
            "accounts@justababy.com",
            "zanovaraccounts@geidi.com"
        }
        
        found_mailboxes = []
        # Check both ToRecipients and CCRecipients
        for field in ["ToRecipients", "CCRecipients"]:
            recipients = meta.get(field, [])
            if not isinstance(recipients, list):
                logging.warning(f"extract_email_metadata - {field} is not a list: {type(recipients)}")
                continue
                
            for recipient in recipients:
                if isinstance(recipient, str) and recipient.lower() in target_mailboxes:
                    found_mailboxes.append(recipient)
        
        # **NEW**: if we didn’t find any in To/CC, try the Account field
        if not found_mailboxes:
            # check both possible casings
            acct = None
            if isinstance(email.get("Account"), str):
                acct = email["Account"].lower()
            elif isinstance(email.get("account"), str):
                acct = email["account"].lower()

            if acct in target_mailboxes:
                logging.info(f"Falling back to Account field: {acct}")
                found_mailboxes.append(acct)
        
        # Store them in the metadata dictionary
        meta["MatchedMailboxes"] = found_mailboxes
        # --- End: Additional Mailbox Check ---

        return meta
    except Exception as e:
        logging.error(f"Metadata extraction error: {str(e)}")
        return {}

def check_reply_or_forward(subject: str) -> Dict[str, bool]:
    subject = subject.strip().lower()
    is_reply = subject.startswith("re:")
    is_forward = subject.startswith("fw:") or subject.startswith("fwd:") or "forwarded" in subject
    return {"is_reply": is_reply, "is_forward": is_forward}

def extract_yaml_conditions(yaml_data, category):
    """
    Extract subject/body keywords from YAML config for a given category.
    """
    keywords = {"subject_contains": [], "body_contains": []}
    category_data = yaml_data.get("finance_shared_mailbox", {}).get("categories", {}).get(category, {})
    for subcat, subdata in category_data.items():
        if not isinstance(subdata, dict):
            continue
        condition = subdata.get("condition", {})
        if isinstance(condition, dict):
            for key, value in condition.items():
                if key == "subject_contains" and isinstance(value, str):
                    keywords["subject_contains"].append(value.lower())
                elif key == "body_contains" and isinstance(value, str):
                    keywords["body_contains"].append(value.lower())
    return keywords


def check_other_emails_conditions(subject: str, body: str, email_data: dict) -> Tuple[bool, str, int]:
    """
    Check if the email matches any of the detailed conditions in the "other_emails" category.
    Returns (matched, subcategory, priority)
    """
    try:
        # Locate the 'other_emails' section in the YAML configuration
        other_email_config = state.yaml_data_cache.get("finance_shared_mailbox", {}) \
                                              .get("categories", {}) \
                                              .get("other_emails", {})
        if not other_email_config:
            logging.warning("other_emails configuration not found in YAML")
            return False, None, 999
    except Exception as e:
        logging.error(f"Error accessing YAML config: {str(e)}")
        return False, None, 999

    # Initialize tracking for best match
    best_match = None
    best_priority = 999
    subject_lower = subject.lower()
    body_lower = body.lower()
    
    logging.info(f"Checking 'other_emails' conditions for subject: '{subject_lower}'")

    # Iterate through subcategories with priority awareness
    for subcat, subdata in other_email_config.items():
        # Skip non-dict entries like "description"
        if not isinstance(subdata, dict):
            continue
            
        # Get condition and priority info
        condition = subdata.get("condition", {})
        priority = subdata.get("priority", 100)  # Default priority
        
        # Skip invalid conditions
        if not condition or not isinstance(condition, dict):
            continue
        
        logging.info(f"Checking subcategory '{subcat}' with priority {priority}")
        
        # Check subject_contains conditions
        subject_match = True
        if "subject_contains" in condition:
            value = condition["subject_contains"]
            if isinstance(value, list):
                # Check if ANY of the keywords are in the subject (OR condition)
                matches = [kw for kw in value if kw.lower() in subject_lower]
                subject_match = len(matches) > 0
                if matches:
                    logging.info(f"Subject matches for '{subcat}': {matches}")
            elif isinstance(value, str):
                subject_match = value.lower() in subject_lower
                if subject_match:
                    logging.info(f"Subject match for '{subcat}': {value}")
            
            # If subject doesn't match, skip this subcategory
            if not subject_match:
                continue
        
        # Check body_contains conditions
        body_match = True
        if "body_contains" in condition:
            value = condition["body_contains"]
            if isinstance(value, list):
                # Check if ANY of the keywords are in the body (OR condition)
                matches = [kw for kw in value if kw.lower() in body_lower]
                body_match = len(matches) > 0
                if matches:
                    logging.info(f"Body matches for '{subcat}': {matches}")
            elif isinstance(value, str):
                body_match = value.lower() in body_lower
                if body_match:
                    logging.info(f"Body match for '{subcat}': {value}")
            
            # If body doesn't match, skip this subcategory
            if not body_match:
                continue

        # Check must_not_include conditions (these are exclusion criteria)
        if "must_not_include" in condition:
            value = condition["must_not_include"]
            if isinstance(value, list):
                # If ANY of the forbidden keywords are found, skip this subcategory
                forbidden_matches = [kw for kw in value if kw.lower() in subject_lower or kw.lower() in body_lower]
                if forbidden_matches:
                    logging.info(f"Forbidden keywords found for '{subcat}': {forbidden_matches}")
                    continue
            elif isinstance(value, str):
                if value.lower() in subject_lower or value.lower() in body_lower:
                    logging.info(f"Forbidden keyword found for '{subcat}': {value}")
                    continue

        # Check contains conditions (generic contains in either subject or body)
        if "contains" in condition:
            value = condition["contains"]
            contains_match = False
            if isinstance(value, list):
                # If ANY of the required keywords are found, it's a match
                contains_matches = [kw for kw in value if kw.lower() in subject_lower or kw.lower() in body_lower]
                contains_match = len(contains_matches) > 0
                if contains_matches:
                    logging.info(f"Generic contains matches for '{subcat}': {contains_matches}")
            elif isinstance(value, str):
                contains_match = value.lower() in subject_lower or value.lower() in body_lower
                if contains_match:
                    logging.info(f"Generic contains match for '{subcat}': {value}")
            
            # If contains doesn't match, skip this subcategory
            if not contains_match:
                continue

        # If we got here, all conditions matched - check if this is higher priority than current best
        logging.info(f"Matched subcategory '{subcat}' with priority {priority}")
        if priority < best_priority:
            best_priority = priority
            best_match = subcat
    
    if best_match:
        logging.info(f"Best match in other_emails: '{best_match}' with priority {best_priority}")
        return True, best_match, best_priority
    
    return False, None, 999

import logging

def evaluate_yaml_condition(condition, subject, body, sender, meta):
    """
    Returns True if `condition` is satisfied by the given subject/body/sender/meta.
    Logs each check for full visibility.
    """
    # Build combined text once
    doc_text = meta.get("document", {}).get("text", "")
    combined = " ".join(filter(None, [subject, body, doc_text])).lower()
    logging.debug(f"Evaluating condition: {condition}")
    logging.debug(f"Subject: {subject!r}")
    logging.debug(f"Body: {body!r}")
    logging.debug(f"Document text: {doc_text!r}")

    # Bare-string → contains
    if isinstance(condition, str):
        hit = condition.lower() in combined
        logging.info(f"String condition '{condition}': {'✓' if hit else '✗'}")
        return hit

    if not isinstance(condition, dict):
        logging.warning(f"Ignoring non-dict condition: {condition}")
        return False

    # subject_contains
    if "subject_contains" in condition:
        terms = condition["subject_contains"]
        terms = [terms] if isinstance(terms, str) else terms
        hit = any(term.lower() in subject for term in terms)
        logging.info(f"subject_contains {terms}: {'✓' if hit else '✗'}")
        if not hit:
            return False

    # body_contains
    if "body_contains" in condition:
        terms = condition["body_contains"]
        terms = [terms] if isinstance(terms, str) else terms
        hit = any(term.lower() in body for term in terms)
        logging.info(f"body_contains {terms}: {'✓' if hit else '✗'}")
        if not hit:
            return False

    # contains (combined)
    if "contains" in condition:
        terms = condition["contains"]
        terms = [terms] if isinstance(terms, str) else terms
        matched = [t for t in terms if t.lower() in combined]
        logging.info(f"contains {terms}: matched {matched}")
        if not matched:
            return False

    # must_include (combined)
    if "must_include" in condition:
        terms = condition["must_include"]
        terms = [terms] if isinstance(terms, str) else terms
        matched = [t for t in terms if t.lower() in combined]
        logging.info(f"must_include {terms}: matched {matched}")
        if not matched:
            return False

    # must_not_include (combined)
    if "must_not_include" in condition:
        terms = condition["must_not_include"]
        terms = [terms] if isinstance(terms, str) else terms
        blocked = [t for t in terms if t.lower() in combined]
        logging.info(f"must_not_include {terms}: blocked {blocked}")
        if blocked:
            return False

    # sender
    if "sender" in condition:
        s_val = condition["sender"]
        expected = [s_val] if isinstance(s_val, str) else s_val
        matched = [s for s in expected if s.lower() in sender]
        logging.info(f"sender {expected}: matched {matched}")
        if not matched:
            return False

    logging.info("✅ Condition passed all checks")
    return True

def preprocess_attachments(emails):
    logging.info(f"Starting preprocess_attachments with {len(emails)} emails")
    
    for i, email in enumerate(emails):
        logging.info(f"Processing email {i} - Has Attachments key: {'Attachments' in email}")
        
        # Skip if email is not a dictionary
        if not isinstance(email, dict):
            logging.warning(f"Email {i} is not a dictionary: {type(email)}")
            continue
        
        # Initialize empty list if Attachments is not present or not a list
        if "Attachments" not in email or not isinstance(email["Attachments"], list):
            logging.warning(f"Email {i} has invalid Attachments: {type(email.get('Attachments', None))}")
            email["Attachments"] = []
            continue
            
        logging.info(f"Email {i} has {len(email['Attachments'])} attachments")
        
        processed_attachments = []
        for j, attachment in enumerate(email["Attachments"]):
            try:
                logging.info(f"Processing attachment {j} of email {i}: {attachment}")
                
                if isinstance(attachment, dict):
                    processed_attachment = {
                        "name": attachment.get("Name", ""),  # Use lowercase for Pydantic
                        "Name": attachment.get("Name", ""),  # Keep original for compatibility
                        "link": attachment.get("Link", ""),  # Use lowercase for Pydantic
                        "Link": attachment.get("Link", "")   # Keep original for compatibility
                    }
                    processed_attachments.append(processed_attachment)
                    logging.info(f"Processed attachment {j}: {processed_attachment}")
                else:
                    logging.warning(f"Attachment {j} is not a dict: {type(attachment)}")
                    # Skip this attachment instead of trying to process it
            except Exception as e:
                logging.error(f"Error processing attachment {j} of email {i}: {str(e)}")
                # Continue with next attachment rather than failing
        
        email["Attachments"] = processed_attachments
        logging.info(f"Email {i} now has {len(processed_attachments)} processed attachments")
    
    logging.info("Completed preprocess_attachments")
    return emails

def get_conditions_for_category(category: str, yaml_data: dict) -> str:
    """
    Extracts a summary of conditions and instructions for a given category from the YAML.
    """
    category_yaml = yaml_data.get("finance_shared_mailbox", {}).get("categories", {}).get(category, {})
    if not category_yaml:
        return ""
    
    conditions_prompt = f"Conditions for category '{category}':\n"
    for subcat, sub_details in category_yaml.items():
        if not isinstance(sub_details, dict):
            continue
        condition = sub_details.get("condition")
        if not condition:
            continue
        conditions_prompt += f"- Subcategory: {subcat}\n"
        if isinstance(condition, dict):
            for key, value in condition.items():
                if isinstance(value, list):
                    value_str = ", ".join(value)
                else:
                    value_str = str(value)
                conditions_prompt += f"  {key}: {value_str}\n"
        else:
            conditions_prompt += f"  condition: {condition}\n"
        if "detailed_task_instructions" in sub_details:
            instructions = sub_details["detailed_task_instructions"].replace("\n", " ")
            conditions_prompt += f"  Instructions: {instructions}\n"
    return conditions_prompt

# ===================================================
# 8. Crew and Agent Functions
# ===================================================
def generate_agents_from_yaml(yaml_data):
    from crewai import Crew, Agent, Task
    import logging
    agents = []
    
    finance_shared = yaml_data.get("finance_shared_mailbox", {})
    main_categories = finance_shared.get("main_categories", [])
    categories_data = finance_shared.get("categories", {})

    # Create a Classifier Agent.
    classifier_def = {
        "role": "Email Classifier",
        "goal": "Accurately categorize incoming emails",
        "backstory": "Expert in analyzing email content to determine the correct category.",
        "tasks": [{
            "description": (
                "Analyze the email content and metadata to determine its category.\n"
                "Categories: " + ', '.join(main_categories) + "\n"
                "Respond with a single category name."
            ),
            "expected_output": "|".join(main_categories)
        }]
    }
    agents.append(classifier_def)
    
    # Helper: process_action converts each action to a human-readable tuple.
    def process_action(action, actions_list):
        if isinstance(action, dict):
            action_type = action.get("action_type", "Unknown")
            if action_type == "file" and "folder" in action:
                folder = action["folder"]
                actions_list.append((action_type, f"File in: {folder}"))
            elif action_type == "create_jira_ticket":
                summary = action.get("summary", "[JIRA TICKET]")
                actions_list.append((action_type, f"Create Jira ticket: {summary}"))
            elif action_type == "forward":
                target = action.get("target", "Unknown Target")
                actions_list.append((action_type, f"Forward to: {target}"))
            elif action_type == "flag":
                actions_list.append((action_type, "Flag the email"))
            elif action_type == "notify_admin":
                message = action.get("message", "No message provided")
                actions_list.append((action_type, f"Notify admin: {message}"))
            else:
                actions_list.append((action_type, f"{action_type.title()} action"))
    
    # Helper: extract_actions searches for actions in category data.
    def extract_actions(category_data):
        actions_list = []
        if "actions" in category_data:
            for action in category_data["actions"]:
                process_action(action, actions_list)
        for key, value in category_data.items():
            if isinstance(value, dict) and "actions" in value:
                for action in value["actions"]:
                    process_action(action, actions_list)
        return actions_list

    # Create specialized agents for each main category.
    for category in main_categories:
        if category not in categories_data:
            logging.warning(f"Main category '{category}' not found in categories!")
            continue

        category_data = categories_data[category]
        if not isinstance(category_data, dict):
            continue
        
        display_name = category.replace('_', ' ').title()
        
        # Aggregate subcategory information.
        subcategories = []
        subcat_conditions = {}
        for key, value in category_data.items():
            if key in ['description', 'detailed_task_instructions', 'actions']:
                continue
            if isinstance(value, dict):
                if any(isinstance(sub_val, dict) for sub_val in value.values()):
                    nested_subcats = list(value.keys())
                    for sub in nested_subcats:
                        subcategories.append(f"{key}.{sub}")
                else:
                    subcategories.append(key)
                if "actions" in value:
                    for action in value["actions"]:
                        if "condition" in action:
                            subcat_conditions.setdefault(key, []).append(action["condition"])
        
        direct_conditions = []
        if 'condition' in category_data and isinstance(category_data['condition'], dict):
            for cond_key, cond_value in category_data['condition'].items():
                direct_conditions.append(f"{cond_key}: {cond_value}")
        
        actions = extract_actions(category_data)
        subcategories_text = "Available subcategories:\n" + "\n".join(f"- {sub}" for sub in subcategories) + "\n" if subcategories else ""
        direct_conditions_text = "Direct Conditions:\n" + "\n".join(f"- {cond}" for cond in direct_conditions) + "\n" if direct_conditions else ""
        actions_text = "Actions to apply:\n" + "\n".join(f"- {atype}: {adetail}" for atype, adetail in actions) + "\n" if actions else ""
        
        # Build the task description using YAML details.
        # In generate_agents_from_yaml(), within the loop for each main category

        # Build the task description using YAML details.
        task_desc = (
            f"IMPORTANT: You are processing emails classified as {display_name}.\n"
            f"{subcategories_text}"
            f"{direct_conditions_text}"
            f"{actions_text}\n"
            "Instructions:\n"
            "1. Extract key elements (subject, sender, body, attachments) from the email.\n"
            "2. Evaluate the email against the YAML file conditions:\n"
            "   - **Must Include:** Ensure that all keywords listed under 'must_include' are present in the email content (subject and/or body).\n"
            "   - **Must Not Include:** Ensure that none of the keywords under 'must_not_include' appear in the email content.\n"
            "   - Also, check for 'contains' conditions where at least one of the keywords should be present.\n"
            "3. Based on these condition evaluations, determine which subcategory the email falls into.\n"
            "4. Once the subcategory is identified, retrieve the corresponding next action(s) from the YAML file (e.g., file to a specific folder, forward, create Jira ticket, etc.).\n"
            "5. Provide a clear JSON output with keys: sub_category, condition, next_action, and rationale, where the rationale explains which YAML condition(s) were met.\n"
            "6. For emails that are replies or forwards, do not trigger new Jira tickets; simply annotate no further action is required.\n"
        )
        
        specialized_def = {
            "role": f"{display_name} Processor",
            "goal": f"Process {display_name.lower()} emails accurately",
            "backstory": f"Expert in handling {display_name.lower()} emails. Follow these instructions:\n{task_desc}",
            "tasks": [{
                "description": task_desc,
                "expected_output": (
                    '{"sub_category": "example", "condition": "Example Condition", '
                    '"next_action": "Example Action", "rationale": "Example Rationale"}'
                )
            }]
        }
        agents.append(specialized_def)
    
    return {"agents": agents}

def create_crew_system(yaml_data):
    from crewai import Crew, Agent, Task

    agent_defs = generate_agents_from_yaml(yaml_data)
    crews = {}

    for agent_def in agent_defs["agents"]:
        role = agent_def["role"]
        goal = agent_def["goal"]
        backstory = agent_def["backstory"]
        agent_key = f"{role}_{goal}"

        # ✅ Explicit model definition
        llm = ChatOpenAI(
            model="gpt-4o",  # You can swap this to "gpt-4o-mini" or "gpt-3.5-turbo"
            temperature=0.2,
            openai_api_key=OPENAI_API_KEY
        )

        if agent_key in state.persistent_agents:
            logging.info(f"Reusing existing agent: {role}")
            agent = state.persistent_agents[agent_key]
        else:
            agent = Agent(role=role, goal=goal, backstory=backstory, llm=llm)
            state.persistent_agents[agent_key] = agent
            logging.info(f"Created new agent: {role}")

        tasks = []
        for task_def in agent_def["tasks"]:
            task = Task(
                description=task_def["description"],
                agent=agent,
                expected_output=task_def["expected_output"]
            )
            tasks.append(task)

        key = "classification" if "Classifier" in role else role.split()[0].lower()
        crews[key] = Crew(agents=[agent], tasks=tasks, verbose=False)

    return crews


async def batch_process_emails(emails, batch_size, yaml_data, crews=None):
    """
    Process emails in batches, keeping domain classification but replacing AI crew with rule-based processing.
    
    Args:
        emails: List of email dictionaries
        batch_size: Number of emails to process in each batch
        yaml_data: YAML configuration with rules
        crews: Optional CrewAI crews (kept for backward compatibility)
        
    Returns:
        List of processed results
    """
    results = []
    total_batches = math.ceil(len(emails) / batch_size)
    supplier_mapping, client_mapping, internal_domains = await get_domain_config()
    supplier_names = list(supplier_mapping.keys())
    client_names = list(client_mapping.keys())
    initial_memory = get_memory_usage()
    logging.info(f"Starting batch processing with {initial_memory:.2f} MB memory usage")
    
    for batch in range(total_batches):
        batch_results = []
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(emails))
        logging.info(f"Processing batch {batch+1}/{total_batches} (emails {start+1}-{end})")
        batch_emails = emails[start:end]
        
        for email in batch_emails:
            try:
                # Check if email is a dictionary
                if not isinstance(email, dict):
                    logging.warning(f"Email in batch {batch+1} is not a dictionary: {type(email)}")
                    continue
                
                # Extract needed fields with type checking
                stripped_email = {}
                for field in ["Subject", "WebLink", "SenderName", "SenderEmail", "FromName", "FromEmail", 
                             "PlainTextEmailBody", "ToRecipients", "CCRecipients", "ID", "ConversationID", 
                             "HasAttachments", "ReceivedDateTime", "Attachments", "Account"]:
                    if field in email:
                        if field in ["ToRecipients", "CCRecipients", "Attachments"] and not isinstance(email[field], list):
                            stripped_email[field] = []
                        elif field in ["Subject", "WebLink", "SenderName", "SenderEmail", "FromName", "FromEmail", 
                                      "PlainTextEmailBody", "ID", "ConversationID", "ReceivedDateTime"] and not isinstance(email[field], (str, type(None))):
                            stripped_email[field] = str(email[field]) if email[field] is not None else ""
                        else:
                            stripped_email[field] = email[field]
                    else:
                        # Set default values based on field type
                        if field in ["ToRecipients", "CCRecipients", "Attachments"]:
                            stripped_email[field] = []
                        elif field == "HasAttachments":
                            stripped_email[field] = False
                        elif field == "Subject":
                            stripped_email[field] = "No Subject"
                        else:
                            stripped_email[field] = ""
                
                # Extract metadata
                meta = extract_email_metadata(stripped_email)
                
                # Skip emails with no content
                if (not meta.get("Subject") or meta["Subject"].lower() == "no subject") and not meta.get("PlainTextEmailBody"):
                    result = {
                        "email": meta,
                        "category": "no_content",
                        "processing_result": {
                            "sub_category": None,
                            "condition": None,
                            "next_action": "No action required",
                            "rationale": "Email has empty or null subject and body. Skipped classification.",
                            "raw_output": None,
                            "error": None
                        }
                    }
                    batch_results.append(result)
                    continue
                
                # Step 1: Use domain_classification to get the initial category
                category = domain_classification(meta, supplier_names, client_names, internal_domains)
                
                # Step 2: Process the email with rules instead of CrewAI
                result = await process_email_with_crew(
                    email_id=f"{meta['Subject']}_{meta['FromEmail']}",
                    meta=meta,
                    category=category
                )
                
                batch_results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing email in batch {batch+1}: {str(e)}")
                # Add a failure result
                batch_results.append({
                    "email": {"error": str(e)},
                    "category": "error",
                    "processing_result": {
                        "error": str(e),
                        "rationale": "Error during email processing",
                        "next_action": "Manual review required"
                    }
                })
        
        # Add batch results to final results
        results.extend(batch_results)
        del batch_results
        gc.collect()
        
        # Memory management
        current_memory = get_memory_usage()
        logging.info(f"Memory after batch {batch+1}: {current_memory:.2f} MB (+{current_memory - initial_memory:.2f} MB)")
        if current_memory > memory_threshold_mb * 0.8:
            logging.warning("Memory usage high, performing cleanup")
            optimize_memory()
        
        await asyncio.sleep(1)
    
    return results

GENERIC_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com", "aol.com"
}

def domain_classification(email_data, supplier_names, client_names, internal_domains):
    sender = email_data.get("FromEmail", "").lower()
    subject = email_data.get("Subject", "").lower()
    body = email_data.get("PlainTextEmailBody", "").lower() or email_data.get("body", "").lower()
    to_list = [recipient.lower() for recipient in email_data.get("ToRecipients", [])]
    sender_host = sender.split("@")[-1]

    # --- LOAD YOUR MAPPINGS FIRST ---
    if hasattr(state, 'domain_config'):
        supplier_mapping, client_mapping, _ = state.domain_config
    else:
        supplier_mapping, client_mapping, _ = {}, {}, []
        logging.warning("No domain_config found in state")
    
    # —— PAYMENT RELAY OVERRIDE ———
    # If Xero sent it, look inside for a client name before calling it a supplier
    if sender_host.endswith("xero.com"):
        for client in client_names:
            if client.lower() in subject or client.lower() in body:
                logging.info(f"📤 Detected client '{client}' in Xero email; reclassifying as client_emails")
                email_data["matched_name"] = client
                return "client_emails"
    # ——————————————————————————

    # Enhanced logging for debugging
    logging.info(f"Classification started - Subject: '{subject}', From: '{sender}'")
    logging.info(f"Supplier names count: {len(supplier_names)}")
    logging.info(f"Client names count: {len(client_names)}")
    
    # --- 0) If this is one of OUR shared inboxes, treat as internal, no further checks ---
    OUR_MAILBOXES = {
        "accounts@geidi.com",
        "cebuaccounts@geidi.com",
        "accounts@justababy.com",
        "zanovaraccounts@geidi.com",
    }
    if sender in OUR_MAILBOXES:
        logging.info(f"Sender {sender} is one of our shared mailboxes → internal_emails (no_action)")
        # tag it so downstream knows “no action”
        email_data["matched_subcategory"] = "no_action"
        return "internal_emails"
    
    # 1) If it’s a generic/free‐mail domain → do further, smarter checks
    if sender_host in GENERIC_DOMAINS:
        logging.info(f"Generic domain '{sender_host}' detected → applying extended matching")
        category, matched_name = match_supplier_or_client_extended(
            sender_email=sender,
            subject=subject,
            supplier_mapping=supplier_mapping,
            client_mapping=client_mapping,
            threshold=90
        )
        if category:
            email_data["matched_name"] = matched_name
            return category
        # if extended matching also fails, fall through to your default
        return "emails_from_other_entities"
        
    # Check if the email meets any of the "other_emails" conditions from YAML
    if state.yaml_data_cache:
        is_match, subcat, priority = check_other_emails_conditions(subject, body, email_data)
        if is_match and subcat:
            logging.info(f"Email matched 'other_emails.{subcat}' with priority {priority}")
            # Store subcategory in the email data for later use
            email_data["matched_subcategory"] = subcat
            return "other_emails"
       
    # Explicitly log the domain extraction for better debugging
    sender_domain = extract_domain_root(sender)
    logging.info(f"Explicit domain extraction: '{sender}' → '{sender_domain}'")

    # Check if the domain is in supplier_names or client_names directly
    if any(sender_domain.lower() == name.lower() for name in supplier_names):
        logging.info(f"Direct domain match in supplier_names: {sender_domain}")
    if any(sender_domain.lower() == name.lower() for name in client_names):
        logging.info(f"Direct domain match in client_names: {sender_domain}")

    # Check internal domain logic
    if hasattr(state, 'domain_config'):
        supplier_mapping, client_mapping, _ = state.domain_config
        # Log the mappings for debugging
        logging.info(f"Supplier mapping keys: {list(supplier_mapping.keys())[:5]}...")
        logging.info(f"Client mapping keys: {list(client_mapping.keys())[:5]}...")
    else:
        supplier_mapping, client_mapping, _ = {}, {}, []
        logging.warning("No domain_config found in state")

    # build a proper set of lowercase emails
    supplier_emails_set = set()
    for info in supplier_mapping.values():
        raw = info.get("email")
        if isinstance(raw, str):
            supplier_emails_set.add(raw.lower())
        elif isinstance(raw, list):
            for addr in raw:
                if isinstance(addr, str):
                    supplier_emails_set.add(addr.lower())
    # now this is safe:
    if sender.endswith("@geidi.com") and sender not in supplier_emails_set:
        return "internal_emails"

    # Rest of the classification logic...
    is_internal_sender = any(internal in sender for internal in internal_domains)
    logging.info(f"Sender: {sender} | is_internal_sender: {is_internal_sender}")
    
    # Continue with the rest of your domain_classification logic...
    if is_internal_sender:
        for recipient in to_list:
            if not any(internal in recipient for internal in internal_domains):
                supplier_match = match_against_names(recipient, supplier_names)
                if supplier_match:
                    logging.info(f"Internal email to supplier: {recipient}")
                    return "supplier_emails"
                client_match = match_against_names(recipient, client_names)
                if client_match:
                    logging.info(f"Internal email to client: {recipient}")
                    return "client_emails"
        return "internal_emails"

    # 5) **Your extended matching** (host/subdomain → fuzzy → local → subject)
    category, matched_name = match_supplier_or_client_extended(
        sender_email=sender,
        subject=subject,
        supplier_mapping=supplier_mapping,
        client_mapping=client_mapping,
        threshold=90
    )
    logging.info(f"Extended matching result: category={category}, matched_name={matched_name}")
    
    if category:
        email_data["matched_name"] = matched_name
        return category
    
    logging.info("No classification match found, defaulting to 'emails_from_other_entities'")
    return "emails_from_other_entities"

def find_best_subcategory(category_data, subject, body, sender, meta):
    best_match = {"full_subcategory": None, "priority": float('inf'), "condition": None, "actions": None}

    def search_subcats(data, prefix=""):
        nonlocal best_match
        for key, value in data.items():
            if key in ['description', 'actions']:
                continue

            if isinstance(value, dict):
                current_path = f"{prefix}.{key}" if prefix else key

                if 'condition' in value:
                    condition = value['condition']
                    logging.info(f"🔍 Evaluating subcategory '{current_path}' with condition: {condition}")

                    matched = evaluate_yaml_condition(condition, subject, body, sender, meta)
                    logging.info(f"✅ Condition {'matched' if matched else 'did NOT match'} for '{current_path}'")

                    if matched:
                        priority = value.get('priority', 99)
                        logging.info(f"📊 Current best priority: {best_match['priority']}, this match: {priority}")
                        if priority < best_match["priority"]:
                            best_match = {
                                "full_subcategory": current_path,
                                "condition": condition,
                                "actions": value.get("actions", []),
                                "priority": priority
                            }
                            logging.info(f"🏆 New best match: {current_path} (priority {priority})")

                else:
                    # If no direct condition, recursively go deeper
                    logging.info(f"📁 Entering nested path: '{current_path}'")
                    search_subcats(value, prefix=current_path)

    search_subcats(category_data)

    if best_match["full_subcategory"]:
        logging.info(f"✅ Final matched subcategory: {best_match['full_subcategory']}")
    else:
        logging.warning("⚠️ No matching subcategory found")

    return best_match["full_subcategory"], best_match["condition"], best_match["actions"]

async def process_email_with_crew(email_id, meta, category, crews=None):
    """
    Process an email with YAML rules after initial domain classification.
    
    Args:
        email_id: Unique identifier for the email
        meta: Dictionary with email metadata
        category: The category assigned by domain_classification()
        crews: Optional CrewAI crews (not used with rule-based approach)
        
    Returns:
        Dict with processing results
    """
    subject_lower = meta.get("Subject", "").lower()
    body_lower = meta.get("PlainTextEmailBody", "").lower()
    sender = meta.get("FromEmail", "").lower()
    
    # --- NEW LOGIC: internal‐thread replies to external recipients get no action ---
    reply_info = check_reply_or_forward(meta.get("Subject", ""))
    if category == "internal_emails" and reply_info.get("is_reply", False):
        all_recipients = [
            r.lower()
            for r in meta.get("ToRecipients", []) + meta.get("CCRecipients", [])
        ]
        TARGET_MAILBOXES = {
            "accounts@geidi.com",
            "cebuaccounts@geidi.com",
            "accounts@justababy.com",
            "zanovaraccounts@geidi.com"
        }
        external = [r for r in all_recipients if r not in TARGET_MAILBOXES]
        if external:
            logging.info(
                "Internal reply thread sent to external recipients; "
                "forcing no_action"
            )
            meta["matched_subcategory"] = "no_action"

    # ── OPTION A: SHORT-CIRCUIT FOR no_action ──
    if meta.get("matched_subcategory") == "no_action":
        return {
            "email": meta,
            "category": category,
            "processing_result": {
                "sub_category":    "no_action",
                "condition":       None,
                "next_action":     "file -> folder",
                "rationale":       "Internal reply sent externally; no further action.",
                "condition_validated": True,
                "confidence":      100
            }
        }
    # Skip known confirmation emails
    if "read:" in subject_lower and "was read" in body_lower:
        return {
            "email": meta,
            "category": category,
            "processing_result": {
                "sub_category": "read_confirmation",
                "condition": "subject_contains: read:, body_contains: was read",
                "next_action": "file -> junkmail",
                "rationale": "Email identified as a read confirmation; deleting email.",
                "condition_validated": True,
                "confidence": 100,
                "issue_type": "Task",
                "status": "AP Process",
                "entity": "Geidi Pty",
                "task_type": "Once off Task"
            }
        }

    # Initialize processing result with default values.
    result = {
        "sub_category": None,
        "condition": None,
        "next_action": "No specific action determined",
        "rationale": f"Email classified as {category} based on domain matching.",
        "condition_validated": False,
        "confidence": 80  # Default confidence
    }
    
    # Use the previously stored matched subcategory if present.
    if "matched_subcategory" in meta and meta["matched_subcategory"]:
        result["sub_category"] = meta["matched_subcategory"]
        result["condition_validated"] = True
        result["confidence"] = 90

    # For valid categories, use the new recursive function to find the best nested subcategory.
    if state.yaml_data_cache and category:
        finance_config = state.yaml_data_cache.get('finance_shared_mailbox', {})
        category_data = finance_config.get('categories', {}).get(category, {})
        
        if isinstance(category_data, dict):
            # Call the new recursive helper
            full_subcat, condition_used, actions_used = find_best_subcategory(
                category_data, subject_lower, body_lower, sender, meta
            )
            
            if full_subcat:
                result["sub_category"] = full_subcat  # e.g., "invoice_processing.paid"
                result["condition"] = str(condition_used)
                result["condition_validated"] = True
                result["confidence"] = 95
                result["rationale"] = f"Email matched conditions for {category}.{full_subcat} based on YAML rules."
                
                # Process actions if any are found.
                if actions_used:
                    matched_name = meta.get("matched_name")
                    context = build_substitution_context(meta, category, matched_name)
                    
                    computed_actions = []
                    for action in actions_used:
                        if isinstance(action, dict):
                            action_type = action.get("action_type", "unknown")
                            
                            if action_type == "file" and "folder" in action:
                                folder_template = action.get("folder", "[No Folder]")
                                folder = substitute_placeholders(folder_template, context)
                                computed_actions.append(f"file -> {folder}")
                            
                            elif action_type == "create_jira_ticket" and "summary" in action:
                                summary_template = action.get("summary", "[No Summary]")
                                summary = substitute_placeholders(summary_template, context)
                                computed_actions.append(f"create_jira_ticket -> {summary}")
                                
                                # --- Add Jira fields if needed ---
                                result["issue_type"] = action.get("issue_type", "Task")
                                result["status"]     = action.get("status", "AP Process")
                                result["entity"]     = action.get("entity", "Geidi Pty")
                                result["task_type"]  = action.get("task_type", "Once off Task")

                                mappings = state.jira_mappings or {}

                                # 1) Always assign to Ruth (from your jira.yaml user_mapping)
                                result["assignee"] = "Angeli Ildefonso"
                                # 2) Look up Ruth's accountId directly from jira.yaml
                                result["jira_assignee_id"] = mappings.get("user_ids", {}).get("angeli ildefonso")

                                # The rest of your Jira lookups remain unchanged:
                                result["jira_status_id"]     = mappings.get("status_ids",     {}).get(result["status"].lower())
                                result["jira_task_type_id"]  = mappings.get("task_type_ids", {}).get(result["task_type"].lower())
                                result["jira_entity_id"]     = mappings.get("company_ids",   {}).get(result["entity"].lower())
                                result["jira_transition_id"] = mappings.get("transition_ids",{}).get(result["status"].lower())
                                
                                subcat_key  = result.get("sub_category", "")
                                subcat_norm = subcat_key.lower().replace("_", " ")

                                # 1) If it's requires_action AND "invoice" in the subject → AP Process
                                if subcat_norm == "requires action" and "invoice" in subject_lower:
                                    result["status"]             = "AP Process"
                                    result["jira_status_id"]     = mappings["status_ids"].get("ap process")
                                    result["jira_transition_id"] = mappings["transition_ids"].get("ap process")

                                # 2) Otherwise, if it's requires_action or failed delivery → Upnext
                                elif subcat_norm in ("requires action", "failed delivery"):
                                    result["status"]             = "Upnext"
                                    result["jira_status_id"]     = mappings["status_ids"].get("upnext")
                                    result["jira_transition_id"] = mappings["transition_ids"].get("upnext")
                                    
                            elif action_type == "forward" and "target" in action:
                                target = substitute_placeholders(action.get("target", "Unknown"), context)
                                computed_actions.append(f"forward -> {target}")
                            
                            elif action_type == "flag":
                                computed_actions.append("flag -> Mark email as important")
                            
                            elif action_type == "delete":
                                computed_actions.append("delete -> Delete this email")
                            
                            elif action_type == "notify_admin" and "message" in action:
                                message = substitute_placeholders(action.get("message", "Notification"), context)
                                computed_actions.append(f"notify_admin -> {message}")
                    
                    if computed_actions:
                        result["next_action"] = ", ".join(computed_actions)
                        
                        # --- New: Extract summary from next_action ---
                        summary_texts = []
                        # Split the next_action string by comma to get individual actions.
                        for action_str in result["next_action"].split(", "):
                            if action_str.lower().startswith("create_jira_ticket"):
                                parts = action_str.split("->")
                                if len(parts) > 1:
                                    summary_texts.append(parts[1].strip())
                        if summary_texts:
                            result["summary"] = ", ".join(summary_texts)
                        else:
                            result["summary"] = "N/A"
    
    # Handle reply or forward adjustments.
    reply_info = check_reply_or_forward(meta.get("Subject", ""))
    if reply_info.get("is_reply", False):
        result["next_action"] = "file -> folder"
        result["rationale"] = result.get("rationale", "") + " Email identified as reply; skipping Jira ticket creation."
    
    # If this fell back to “emails_from_other_entities” and no sub_category was ever set,
    # force a “emails_from_other_categories” subcategory.
    if category == "emails_from_other_entities" and not result.get("sub_category"):
        result["sub_category"] = "emails_from_other_categories"
        result["rationale"] += " Defaulted sub_category to emails_from_other_categories as no YAML subcats exist."

    return {
        "email": meta,
        "category": category,
        "processing_result": result
    }
    
def ensure_promotional_category(subcategories, yaml_data):
    promotional_subcats = {"promotional_content", "marketing", "advertisement", "offer"}
    subcategories.update(promotional_subcats)
    if "promotional_emails" not in yaml_data and isinstance(yaml_data, dict):
        yaml_data["promotional_emails"] = {
            "description": "Marketing, promotional offers, and advertisements",
            "actions": [
                {
                    "condition": "Marketing or promotional material",
                    "action_steps": "No action required - promotional content",
                    "priority": "low"
                }
            ],
            "promotional_content": {
                "description": "Marketing materials and promotional offers",
                "actions": [
                    {
                        "condition": "Marketing or promotional material",
                        "action_steps": "No action required - promotional content",
                        "priority": "low"
                    }
                ]
            }
        }
    return subcategories, yaml_data

async def get_crews():
    new_config = await reload_finance_yaml_if_changed()
    
    if new_config or not state.crews_cache:
        if state.yaml_data_cache is None:
            finance_yaml, _ = await load_yaml_from_gitlab()
            state.yaml_data_cache = finance_yaml
        
        state.valid_subcategories = extract_valid_subcategories(state.yaml_data_cache)
        state.valid_subcategories, state.yaml_data_cache = ensure_promotional_category(state.valid_subcategories, state.yaml_data_cache)
        state.crews_cache = create_crew_system(state.yaml_data_cache)
        limit_cache_size()
        logging.info(f"Created new crews: {list(state.crews_cache.keys())}")
    else:
        logging.info(f"Reusing existing crews: {list(state.crews_cache.keys())}")
    
    return state.crews_cache

def extract_valid_subcategories(yaml_data):
    subcats = set()
    for category, data in yaml_data.items():
        if isinstance(data, dict):
            for key, value in data.items():
                if key not in ['description', 'actions']:
                    subcats.add(key)
                    if isinstance(value, dict) and "actions" in value:
                        for action in value["actions"]:
                            if "condition" in action:
                                subcats.add(action["condition"])
            if "actions" in data:
                for action in data["actions"]:
                    if "condition" in action:
                        subcats.add(action["condition"])
    logging.info(f"Extracted {len(subcats)} valid subcategories.")
    return subcats

# ===================================================
# 9. FastAPI Endpoints and Startup
# ===================================================
@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
    crews: Dict = Depends(get_crews)
):
    try:
        # 🚨 Check and reload if YAML files changed
        await reload_finance_yaml_if_changed()
        await reload_domain_yaml_if_changed()
        
        # Enhanced logging at the start
        logging.info(f"Classify request received with {len(request.emails)} emails")
        
        # Convert incoming items to a unified list of email dicts with robust error handling
        unified_emails = []
        
        for i, item in enumerate(request.emails):
            try:
                logging.info(f"Processing request item {i}, type: {type(item)}")
                
                if not isinstance(item, dict):
                    logging.warning(f"Item {i} is not a dictionary: {type(item)}")
                    continue
                    
                # If the current item is a dict that has an "emails" key and that value is a list,
                # use that list.
                if "emails" in item and isinstance(item["emails"], list):
                    for j, email in enumerate(item["emails"]):
                        if isinstance(email, dict):
                            unified_emails.append(email)
                        else:
                            logging.warning(f"Item {i}, nested email {j} is not a dictionary: {type(email)}")
                else:
                    # Otherwise, assume the item is already a single email dict.
                    unified_emails.append(item)
            except Exception as e:
                logging.error(f"Error processing request item {i}: {str(e)}")
                # Continue with next item rather than failing completely
        
        # Log and continue processing using unified_emails
        logging.info(f"Unified {len(unified_emails)} emails for processing")
        
        # Check each email for valid structure before processing
        valid_emails = []
        for idx, email in enumerate(unified_emails):
            try:
                if not isinstance(email, dict):
                    logging.warning(f"Email {idx} is not a dictionary: {type(email)}")
                    continue
                    
                # Log attachment info
                if "Attachments" in email:
                    logging.info(f"Email {idx} has attachments: {email.get('Attachments')}")
                
                valid_emails.append(email)
            except Exception as e:
                logging.error(f"Error validating email {idx}: {str(e)}")
        
        if not valid_emails:
            logging.warning("No valid emails to process after filtering")
            return ClassificationResponse(
                status="warning", 
                processed=0, 
                results=[]
            )
        
        # Continue with preprocessing and batch processing
        emails = preprocess_attachments(valid_emails)
        adaptive_batch_size = min(5, max(1, int(10 / math.sqrt(len(emails) + 1))))
        
        results = await batch_process_emails(
            emails, 
            batch_size=adaptive_batch_size, 
            yaml_data=state.yaml_data_cache, 
            crews=crews
        )
        
        background_tasks.add_task(gc.collect)
        return ClassificationResponse(
            status="success", 
            processed=len(results), 
            results=results
        )
    except Exception as e:
        logging.error(f"Error in /classify: {str(e)}")
        if "memory" in str(e).lower() or "allocation" in str(e).lower():
            background_tasks.add_task(optimize_memory)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-config", response_model=ReloadConfigResponse)
async def reload_config(
    background_tasks: BackgroundTasks,
    request: Request,
    force: bool = False
):
    try:
        if os.path.exists(LAST_AGENTS_FILE):
            os.remove(LAST_AGENTS_FILE)
        if force:
            logging.warning("Forcing complete reset of all agents and crews")
            state.crews_cache = {}
            state.persistent_agents = {}
            if hasattr(state, 'yaml_data_cache'):
                delattr(state, 'yaml_data_cache')
        else:
            state.crews_cache = {}
        background_tasks.add_task(gc.collect)
        yaml_data, _ = await load_yaml_from_gitlab()
        state.yaml_data_cache = yaml_data
        state.valid_subcategories = extract_valid_subcategories(yaml_data)
        state.valid_subcategories, yaml_data = ensure_promotional_category(state.valid_subcategories, yaml_data)
        crews = create_crew_system(yaml_data)
        state.crews_cache = crews
        agent_count = len(state.persistent_agents)
        return ReloadConfigResponse(
            status="success", 
            message="Configuration reloaded.",
            force_reset=force,
            persistent_agents_count=agent_count,
            crews_count=len(state.crews_cache)
        )
    except Exception as e:
        logging.error(f"Error reloading config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "timestamp": time.time(), 
        "crews_count": len(state.crews_cache),
        "uptime_seconds": int(time.time() - state.START_TIME)
    }

@app.get("/_ah/warmup")
async def warmup():
    logging.info("Warmup request received, returning immediately")
    return {"status": "initializing"}

@app.get("/memory", response_model=MemoryStatus)
async def memory_status():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        agent_count = len(state.persistent_agents)
        crew_memory_est = (memory_info.rss / (1024 * 1024)) / (len(state.crews_cache) + 1) if state.crews_cache else 0
        return MemoryStatus(
            status="ok",
            memory_usage_mb=memory_info.rss / (1024 * 1024),
            memory_threshold_mb=memory_threshold_mb,
            crews_cache_size=len(state.crews_cache),
            persistent_agents_count=agent_count,
            estimated_memory_per_crew_mb=crew_memory_est,
            max_cache_size=MAX_CACHE_SIZE,
            cached_crews=list(state.crews_cache.keys()),
            request_counter=state.request_counter,
            uptime_seconds=int(time.time() - state.START_TIME)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/optimize", response_model=OptimizeResponse)
async def force_memory_optimization():
    try:
        before = get_memory_usage()
        optimize_memory()
        after = get_memory_usage()
        return OptimizeResponse(
            status="success", 
            memory_before_mb=before,
            memory_after_mb=after,
            memory_saved_mb=before - after,
            crews_remaining=len(state.crews_cache)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logging.info("Starting up the application...")
    openai.api_key = OPENAI_API_KEY

    try:
        import psutil
    except ImportError:
        logging.warning("psutil not found, attempting to install...")
        import subprocess
        subprocess.check_call(["pip", "install", "psutil"])
        import psutil

    logging.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
    logging.info(f"Memory threshold set to: {memory_threshold_mb} MB")
    logging.info("Deferring CrewAI initialization until first request")
    gc.collect()

    # 🚀 Preload Jira YAML mappings
    try:
        jira_yaml = await load_jira_yaml_from_gitlab()
        state.jira_yaml_cache = jira_yaml
        state.jira_mappings = extract_jira_mappings(jira_yaml)

        if not state.jira_mappings:
            logging.warning("⚠️ Jira mappings are missing or not yet loaded.")
        else:
            logging.info(f"✅ Loaded Jira mappings: {list(state.jira_mappings.keys())}")
    except Exception as e:
        logging.error(f"Failed to preload Jira mappings: {str(e)}")

@functions_framework.http
def handler(request):
    from functions_framework import create_app
    from asgiref.wsgi import WsgiToAsgi
    asgi_app = WsgiToAsgi(app)
    return create_app(target=asgi_app)(request)

# ===================================================
# 10. Entry Point
# ===================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=2)
