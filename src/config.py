import os
from pathlib import Path
from dotenv import load_dotenv

# everything is relative to the project root so it works from the notebook, uvicorn and pytest
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_CSV = DATA_DIR / "raw" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
POLICY_DIR = DATA_DIR / "policies"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
FEEDBACK_DIR = DATA_DIR / "feedback"
CACHE_DIR = DATA_DIR / "llm_cache"

COMPANY = "ShopEase"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
MAX_RETRIES = 2
RESPONSES_PER_INTENT = 40   # bitext has ~1000 responses per intent, mostly rewordings

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"   # 1 = no gemini calls (CI)
