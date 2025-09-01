from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env from root automatically if placed correctly

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_EMAILS_URL = os.getenv("SUPABASE_EMAILS_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEBHOOK_URL= os.getenv("WEBHOOK_URL")

if not SUPABASE_EMAILS_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Environment variables SUPABASE_URL or SUPABASE_API_KEY are missing!")
