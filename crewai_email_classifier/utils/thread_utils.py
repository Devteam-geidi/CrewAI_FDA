from utils.env_loader import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client
from postgrest.exceptions import APIError
import hashlib
from datetime import datetime

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def normalize_subject(subject: str) -> str:
    return subject.lower().replace("re:", "").replace("fwd:", "").strip()

def hash_subject(subject: str) -> str:
    return hashlib.sha256(normalize_subject(subject).encode()).hexdigest()

def find_existing_thread(subject_hash: str, in_reply_to: str | None):
    try:
        if in_reply_to:
            parent = supabase.table("emails").select("thread_id").eq("message_id", in_reply_to).execute()
            if parent and parent.data and len(parent.data) > 0:
                thread_id = parent.data[0].get("thread_id")
                if thread_id:
                    return thread_id

        existing = supabase.table("threads").select("id").eq("subject_hash", subject_hash).execute()
        if existing and existing.data and len(existing.data) > 0:
            return existing.data[0]["id"]

    except APIError as e:
        print(f"⚠️ Supabase APIError during thread lookup: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during thread lookup: {e}")

    return None

def create_new_thread(subject_hash: str):
    now = datetime.utcnow().isoformat()
    try:
        result = supabase.table("threads").insert({
            "subject_hash": subject_hash,
            "started_at": now,
            "updated_at": now
        }).execute()
        return result.data[0]["id"]
    except Exception as e:
        print(f"⚠️ Supabase APIError during thread insert: {e}")
        return None  # Allow the system to continue gracefully


def get_or_create_thread_id(subject: str, in_reply_to: str | None) -> str | None:
    subject_hash = hash_subject(subject)
    thread_id = find_existing_thread(subject_hash, in_reply_to)
    if thread_id:
        return thread_id
    return create_new_thread(subject_hash)
