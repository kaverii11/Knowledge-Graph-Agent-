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


# The model that answers questions (single-shot and agent) can run on a second provider while the judge
# stays where it is, e.g. QUERY_PROVIDER=cloudflare to spread eval runs over two free tiers. Both modes of one
# eval run always use the same provider.
QUERY_PROVIDER = os.getenv("QUERY_PROVIDER", "default").lower()
if QUERY_PROVIDER == "cloudflare":
    QUERY_MODEL = os.getenv("CLOUDFLARE_QUERY_MODEL", "@cf/openai/gpt-oss-120b")
elif QUERY_PROVIDER == "openrouter":
    QUERY_MODEL = os.getenv("OPENROUTER_QUERY_MODEL", "openai/gpt-oss-120b")


def has_llm_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def get_client(role: str = "default") -> OpenAI:
    """role="query": the client for the answering model (honours QUERY_PROVIDER); anything else: the default."""
    if role == "query" and QUERY_PROVIDER == "cloudflare":
        account, token = os.getenv("CLOUDFLARE_ACCOUNT_ID"), os.getenv("CLOUDFLARE_API_TOKEN")
        if not account or not token:
            raise RuntimeError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env")
        return OpenAI(api_key=token, base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1")
    if role == "query" and QUERY_PROVIDER == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("Set OPENROUTER_API_KEY in .env")
        client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        host = os.getenv("OPENROUTER_PIN_PROVIDER", "groq")  # same host as direct Groq runs, no fallbacks
        return _Pinned(client, {"provider": {"order": [host], "allow_fallbacks": False}}) if host else client
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY (or LLM_API_KEY + LLM_BASE_URL) in .env")
    return OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None)


class _Pinned:
    """Adds OpenRouter routing options to every chat call, so a run never silently moves to another host."""

    def __init__(self, client: OpenAI, extra_body: dict):
        from types import SimpleNamespace
        self._client, self._extra = client, extra_body
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.beta = client.beta

    def _create(self, **kwargs):
        kwargs["extra_body"] = {**self._extra, **(kwargs.get("extra_body") or {})}
        return self._client.chat.completions.create(**kwargs)


class CountingClient:
    """Wraps an OpenAI client and records every chat call's tokens and time, so evals can report tokens
    and requests per question for any agent. Also paces calls to stay under a tokens-per-minute limit
    (Groq free tier: about 7,000 input tokens a minute per model), so rate-limit waits are not part of a
    question's latency."""

    def __init__(self, client: OpenAI, input_tpm: int | None = None):
        from types import SimpleNamespace
        self._client, self.input_tpm = client, input_tpm
        self.calls: list[dict] = []        # all calls ever made through this wrapper
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        beta = getattr(client, "beta", None)
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            parse=lambda **kw: self._count(beta.chat.completions.parse(**kw), kw)))) if beta else None

    def _create(self, **kwargs):
        return self._count(self._client.chat.completions.create(**kwargs), kwargs)

    def _count(self, r, kwargs):
        import time
        u = getattr(r, "usage", None)
        self.calls.append({"t": time.time(), "model": kwargs.get("model"),
                           "prompt": getattr(u, "prompt_tokens", 0) or 0,
                           "completion": getattr(u, "completion_tokens", 0) or 0})
        return r

    def mark(self) -> int:
        return len(self.calls)

    def since(self, mark: int) -> dict:
        calls = self.calls[mark:]
        return {"requests": len(calls), "prompt_tokens": sum(c["prompt"] for c in calls),
                "completion_tokens": sum(c["completion"] for c in calls),
                "tokens": sum(c["prompt"] + c["completion"] for c in calls)}

    def wait_for_budget(self, headroom: int):
        """Sleep (outside any timed section) until the last minute's input tokens leave `headroom` free."""
        import time
        if not self.input_tpm:
            return
        while True:
            now = time.time()
            used = sum(c["prompt"] for c in self.calls if now - c["t"] < 60)
            if used + headroom <= self.input_tpm:
                return
            oldest = min(c["t"] for c in self.calls if now - c["t"] < 60)
            time.sleep(max(1.0, 60 - (now - oldest) + 0.5))


def is_daily_limit(e: Exception) -> bool:
    """Groq's per-day token / request limit: waiting a minute won't help, so don't retry it."""
    return "per day" in str(e).lower()


def should_retry(e: Exception) -> bool:
    """Transient API errors (rate limit per minute, timeouts, 5xx) are retried; the daily limit is not."""
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    return isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)) \
        and not is_daily_limit(e)


def is_account_error(e: Exception) -> bool:
    """Out of credit (402) or a bad / unauthorised key (401, 403): not the agent's fault, and retrying won't help."""
    return getattr(e, "status_code", None) in (401, 402, 403)
