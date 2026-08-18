"""
config.py
=========
Loads the .env file once and exposes the settings and paths every other module
shares. Import this before anything that reads os.environ.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent

# override=False: a variable already exported in the shell wins over .env.
# That is what you want in CI or when temporarily testing a different key —
# the file is the default, not an override.
load_dotenv(REPO_ROOT / ".env", override=False)

# --- LLM -------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# --- Paths -----------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
ARTIFACT_DIR = REPO_ROOT / "ml_models" / "artifacts"
POLICY_INDEX_PATH = REPO_ROOT / "policies" / "index"
CUSTOMER_DB_PATH = REPO_ROOT / "database" / "customer_data.db"

# --- Model / policy --------------------------------------------------------
EMBEDDING_MODEL = os.getenv("POLICY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def require_groq_key() -> str:
    """Fail loudly where an LLM is genuinely required. Callers that can degrade
    gracefully should check GROQ_API_KEY directly instead of calling this."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env at the repo root "
            "(see .env.example) or export it in your shell."
        )
    return GROQ_API_KEY


def status() -> dict:
    """Small diagnostic used by `python config.py` so you can confirm the .env
    was actually picked up instead of guessing."""
    return {
        "groq_key_loaded": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL,
        "customer_db_exists": CUSTOMER_DB_PATH.exists(),
        "artifacts_exist": ARTIFACT_DIR.exists(),
        "splits_exist": SPLIT_DIR.exists(),
        "policy_index_exists": POLICY_INDEX_PATH.exists(),
    }


if __name__ == "__main__":
    for key, value in status().items():
        print(f"{key:<22}: {value}")
