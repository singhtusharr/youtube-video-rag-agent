"""Central config: loads env vars once, everything else imports from here."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Explicit path (not just load_dotenv()) so this works regardless of the
# process's current working directory — e.g. when uvicorn/streamlit are
# launched with --app-dir from outside this project folder.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
