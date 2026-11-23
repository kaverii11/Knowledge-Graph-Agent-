"""MLflow tracing for the extraction graph, the Q&A agent and the MCP tools.

Every run records a trace: each LangGraph node, each LLM call (prompt, response,
token usage, latency) and each tool / database call. View them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://localhost:5000

Set MLFLOW_TRACING=0 to switch it off.
"""
import contextlib
import os

_enabled = False


def setup_tracing(experiment: str = "sebi-kg-agent") -> bool:
    global _enabled
    if _enabled:
        return True
    if os.getenv("MLFLOW_TRACING", "1") == "0":
        return False
    try:
        import mlflow
    except ImportError:
        return False
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlflow.db"))
    mlflow.set_experiment(experiment)
    # LangGraph nodes/edges come from the LangChain integration; our nodes call the
    # OpenAI SDK directly, so the OpenAI integration adds the LLM spans with token usage.
    mlflow.langchain.autolog()
    mlflow.openai.autolog()
    _enabled = True
    return True


def trace(span_type: str = "UNKNOWN", name: str | None = None):
    """@mlflow.trace when MLflow is installed, otherwise a no-op decorator."""
    try:
        import mlflow
    except ImportError:
        return lambda f: f
    kwargs = {"span_type": span_type}
    if name:
        kwargs["name"] = name
    return mlflow.trace(**kwargs)


class _NoSpan:
    def set_inputs(self, *a, **k): pass
    def set_outputs(self, *a, **k): pass
    def set_attributes(self, *a, **k): pass


@contextlib.contextmanager
def span(name: str, span_type: str = "UNKNOWN", **attributes):
    """A child span with attributes (tool, args, result size, tokens, latency) when tracing is on;
    a no-op otherwise, so unit tests and runs with MLFLOW_TRACING=0 record nothing."""
    if not _enabled:
        yield _NoSpan()
        return
    import mlflow
    with mlflow.start_span(name=name, span_type=span_type, attributes=attributes or None) as s:
        yield s
