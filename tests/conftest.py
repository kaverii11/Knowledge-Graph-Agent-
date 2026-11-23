import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MLFLOW_TRACING", "0")
os.environ.setdefault("LLM_INPUT_TPM", "0")  # no rate-limit pacing with fake LLMs


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: calls the LLM API and the live graph (run with -m llm)")
