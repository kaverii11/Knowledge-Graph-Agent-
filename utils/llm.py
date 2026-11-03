"""One place to configure the LLM provider.

Works with OpenAI by default, or any OpenAI-compatible API (Groq, Gemini, a local
vLLM/Ollama server) by setting LLM_BASE_URL and the model names in .env.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4o-mini")    # first-pass extraction, text-to-Cypher, answers
PRO_MODEL = os.getenv("PRO_MODEL", "gpt-4o")           # low-confidence fallback + scanned pages (needs vision)
QUERY_MODEL = os.getenv("QUERY_MODEL", FAST_MODEL)     # text-to-Cypher + answer writing
JUDGE_MODEL = os.getenv("JUDGE_MODEL", PRO_MODEL)      # LLM-as-a-judge in evals (ideally a different model family)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def has_llm_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def get_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY (or LLM_API_KEY + LLM_BASE_URL) in .env")
    return OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None)
