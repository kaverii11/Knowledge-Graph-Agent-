"""Tool-selection eval: does an LLM pick the right MCP tool for a request?

The tool list (names, descriptions, JSON schemas) is read from the running MCP server, so this tests the
tool descriptions the way a real MCP client sees them. "none" means the model should refuse to call
any tool (e.g. a request to delete data: no tool can do it).

    python evals/tool_selection_eval.py [--save-baseline]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anyio
from mcp import Client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

import mcp_server
from utils.llm import QUERY_MODEL, get_client
from utils.tracing import setup_tracing, trace

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = ("You are an assistant connected to an MCP server for a SEBI circulars knowledge graph. "
          "Call exactly one tool that best serves the request. If no tool can do what is asked, do not call a tool; "
          "reply explaining why.")


async def mcp_tools_as_openai_functions():
    async with Client(mcp_server.server) as c:
        tools = (await c.list_tools()).tools
    return [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                              "parameters": t.input_schema}} for t in tools]


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True,
       retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)))
@trace(span_type="AGENT", name="select_tool")
def select_tool(client, tools, question) -> str:
    r = client.chat.completions.create(model=QUERY_MODEL, tools=tools, tool_choice="auto", temperature=0,
                                       messages=[{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": question}])
    calls = r.choices[0].message.tool_calls or []
    return calls[0].function.name if calls else "none"


def run(save_baseline=False):
    import mlflow
    setup_tracing()
    client = get_client()
    tools = anyio.run(mcp_tools_as_openai_functions)
    cases = json.load(open(os.path.join(HERE, "tool_selection.json")))
    results = []
    for case in cases:
        chosen = select_tool(client, tools, case["question"])
        ok = chosen in case["expected"]
        results.append({**case, "chosen": chosen, "ok": ok})
        print(f"{'OK ' if ok else 'XX '} {chosen:20s} <- {case['question']}")
    acc = sum(r["ok"] for r in results) / len(results)
    print(f"tool selection accuracy: {acc:.3f}")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"accuracy": acc, "model": QUERY_MODEL, "rows": results},
              open(os.path.join(HERE, "results", "tool_selection.json"), "w"), indent=2)
    with mlflow.start_run(run_name="tool-selection-eval"):
        mlflow.log_param("model", QUERY_MODEL)
        mlflow.log_metric("tool_selection_accuracy", acc)
    if save_baseline:
        path = os.path.join(HERE, "baseline.json")
        base = json.load(open(path)) if os.path.exists(path) else {}
        base["tool_selection"] = {"accuracy": acc}
        json.dump(base, open(path, "w"), indent=2)
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-baseline", action="store_true")
    run(ap.parse_args().save_baseline)
