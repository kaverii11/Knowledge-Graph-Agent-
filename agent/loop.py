"""Tool-using agent loop for Q&A over the SEBI knowledge graph.

The single-shot agent (query_graph.answer_question) writes one Cypher query and answers from its rows.
This loop lets the model choose tools step by step until it can answer:

    model ─► tool calls ─► tool results ─► model ─► ... ─► final answer ─► citation check

It calls the same read-only tool functions the MCP server exposes (in-process, no MCP round trip).
`ask` is left out: it is itself an LLM agent, so calling it from here would nest a second agent whose
tokens and steps the loop could not see or limit.

Guardrails:
- at most `max_steps` model calls and `token_budget` tokens per question; when the step limit is hit the
  model gets one last call without tools to answer from what it has, and says what is missing
- every tool argument is validated with Pydantic against the tool's own signature (same rules the MCP
  server enforces); unknown arguments are rejected
- tool errors (bad arguments, unknown circular, database error, rejected write query) go back to the model
  as a clear message so it can recover
- loop detection: repeating a call with the same arguments is not executed; the model gets a nudge, and a
  second repeat stops the loop
- the final answer must cite a circular number or document from the rows it used (one nudge, then it is
  flagged `uncited`)
- read-only: no tool can write; run_cypher rejects write clauses and runs in a read transaction. Write
  attempts are counted so unsafe requests can be scored.

Every step is traced in MLflow (tool, arguments, result size, tokens, latency).
"""
import inspect
import json
import os
import re
import time
from typing import Any, Literal, TypedDict

from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError
from pydantic import ConfigDict, ValidationError, create_model
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from utils.llm import is_account_error, should_retry

from langgraph.graph import END, StateGraph

from utils.circular_ids import find_circular_ids
from utils.tracing import span, trace

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))
# Set from the tuning split (8 tune questions, 2 runs, 16 question runs): the largest run that ended with an
# answer used 12,086 tokens (it hit the step limit); 1.5x that, rounded, leaves room for a harder question
# while stopping a runaway loop. Measured before the held-out questions were run.
TOKEN_BUDGET = int(os.getenv("AGENT_TOKEN_BUDGET", "18000"))
MAX_RESULT_CHARS = 6000  # a tool result longer than this is cut, and the model is told so

SYSTEM_PROMPT = """You answer questions about a knowledge graph of SEBI (India's securities regulator) circulars
and the acts, regulations and other circulars they reference or modify.

Graph: (:Circular {id, subject, date})-[:CONTAINS]->(:Clause {text, page_number})-[:REFERENCES|MODIFIES]->
(:Entity {name, type, number, aliases}). A Circular is a circular that was ingested; an Entity is a document
that circulars cite (an act, a regulation, or another circular, which may have a number and short names).

How to work:
- Use the tools to look things up; never answer from memory. Break multi-part questions into steps.
- Two kinds of thing: ingested circulars (list_circulars shows all of them with number, subject and date;
  find one by topic there) and the documents they cite (search_documents searches only cited documents by
  name, short name or number). circular_references lists what one circular cites; who_cites lists the
  circulars citing a document; run_cypher is for anything the other tools can't do (read-only).
- An empty result is an answer too: if a tool says nothing matches, don't keep rephrasing the same search.
- If a tool returns an error, read it and fix the call. Don't repeat a call you have already made.
- The graph is read-only. If asked to create, change or delete anything, refuse and call no tool.
- If the graph doesn't contain the answer, say so plainly. Don't guess.

Final answer: short and direct, and answer every part of the question. Cite the circular numbers (the last part, e.g. I/24814/2026, is enough) and
document names your answer rests on, and only state what the tool results show."""

UNKNOWN_TOOL = re.compile(r"attempted to call tool '([^']+)' which was not in request\.tools")

LOOP_NUDGE = ("You already called {tool} with these exact arguments; the result is earlier in this conversation. "
              "Use it, or call a different tool / different arguments, or give your final answer.")
CITE_NUDGE = ("Your answer doesn't cite where it comes from. Rewrite it citing the circular numbers and document "
              "names from the tool results that support it. If the results don't support an answer, say so.")
STEP_LIMIT_NUDGE = ("Step limit reached: no more tool calls. Give your final answer now from the results above, "
                    "and say clearly if something could not be found.")


# --- tools -------------------------------------------------------------------------------------------

def _tool_functions() -> dict:
    import mcp_server  # imported here: mcp_server -> query_graph -> agent.loop would otherwise be circular
    return {name: getattr(mcp_server, name)
            for name in ["list_circulars", "search_documents", "circular_references", "who_cites", "run_cypher"]}


def _arg_model(name: str, fn):
    """Pydantic model for a tool's arguments, built from the same Annotated signature the MCP server uses."""
    fields = {}
    for p in inspect.signature(fn).parameters.values():
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (p.annotation, default)
    return create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)


def build_tools() -> tuple[dict, dict, list]:
    """Returns (functions, argument models, OpenAI tool schemas)."""
    fns = _tool_functions()
    models = {n: _arg_model(n, f) for n, f in fns.items()}
    schemas = []
    for n, f in fns.items():
        params = models[n].model_json_schema()
        params.pop("title", None)
        schemas.append({"type": "function", "function": {
            "name": n, "description": inspect.getdoc(f) or "", "parameters": params}})
    return fns, models, schemas


# --- citations ---------------------------------------------------------------------------------------

def _short_id(cid: str) -> str:
    """'HO/19/34/14(10)2026-AFD-POD3/I/24814/2026' -> '24814/2026'; old format keeps its last 2 parts too."""
    return "/".join(cid.replace(" ", "").split("/")[-2:])


_DATE = re.compile(r"^\s*\w+\s+\d{1,2},\s*\d{4}\s*$|^\d{4}-\w{3}-\d{2}$")


def _looks_like_document_name(x: str) -> bool:
    """A name worth citing: 2+ words, not a date, not a label such as REFERENCES / MODIFIES / act."""
    return (8 <= len(x) <= 150 and len(x.split()) >= 2 and not _DATE.match(x) and not x.isupper()
            and not x.lstrip().upper().startswith(("MATCH", "RETURN")))


def _sources_in_rows(rows: list) -> tuple[set, set]:
    """Circular numbers and document names that appear in tool results."""
    ids, names = set(), set()
    for row in rows:
        for v in (row.values() if isinstance(row, dict) else [row]):
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if not isinstance(x, str):
                    continue
                for cid, _ in find_circular_ids(x):
                    ids.add(cid)
                if _looks_like_document_name(x):
                    names.add(x)
    return ids, names


def find_citations(answer: str, rows: list) -> list[str]:
    """Which circulars / documents from the rows the answer mentions."""
    ids, names = _sources_in_rows(rows)
    text = (answer or "").replace(" ", "").lower()
    lower = (answer or "").lower()
    cited = [cid for cid in sorted(ids) if _short_id(cid).lower() in text]
    cited += [n for n in sorted(names) if n.lower() in lower and n not in cited]
    return cited


NO_ANSWER = re.compile(r"\b(no (circular|document|result|record|information|match|regulation|act)s?|not (found|in the graph|"
                       r"available|present|contain|cited|recorded)|doesn'?t contain|does not contain|couldn'?t find|"
                       r"could not find|cannot|can'?t|unable|read-only|not able|refuse|not allowed|none of|there are none|"
                       r"isn'?t cited|is not cited|no circulars?)\b", re.I)


def is_no_answer(text: str) -> bool:
    """A refusal or 'nothing found' answer (curly apostrophes, as models print them, count too)."""
    return bool(NO_ANSWER.search((text or "").replace("\u2019", "'").replace("\u2018", "'")))


def _empty_result_hint(fns: dict, name: str, args: dict) -> str:
    """who_cites returns [] both when no document matches the words and when the document exists but no
    circular cites it that way; say which, so the model doesn't keep rephrasing a correct search."""
    if name != "who_cites":
        return "[] (no rows matched)"
    try:
        docs = fns["search_documents"](keyword=args["document"])
    except Exception:
        return "[] (no rows matched)"
    if not docs:
        return f"[] (no document in the graph matches '{args['document']}'; try search_documents with fewer words)"
    names = "; ".join(d["name"] for d in docs[:5])
    rel = args.get("relation", "any")
    return (f"[] (the graph has {len(docs)} matching document(s): {names}. No circular cites them with relation "
            f"'{rel}'. That is the answer for this relation.)")


# --- the graph ---------------------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    question: str
    messages: list
    steps: int                 # model calls made
    tokens: int                # prompt + completion tokens used
    tool_calls: list           # one entry per tool call: tool, args, ok, error, result_rows, result_chars, latency_s
    seen: dict                 # (tool, args) -> times called
    repeats: int
    rows: list                 # every row any tool returned, for grounding and citations
    nudged_cite: bool
    final_without_tools: bool  # step limit reached: next model call gets no tools
    bad_tool_retries: int      # provider-rejected calls to tools that don't exist
    answer: str
    stop_reason: str           # answered | step_limit | token_budget | loop | llm_error
    citations: list
    error: str | None


def _assistant_dict(msg) -> dict:
    d = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                           for c in msg.tool_calls]
    return d


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True,
       retry=retry_if_exception(should_retry))
def _chat(client, model, messages, tools):
    kwargs = {"model": model, "messages": messages, "temperature": 0.1}
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    return client.chat.completions.create(**kwargs)


def build_agent(client, model: str, max_steps: int = MAX_STEPS, token_budget: int = TOKEN_BUDGET,
                tools: tuple | None = None):
    fns, models, schemas = tools or build_tools()

    def call_model(state: AgentState) -> AgentState:
        offer_tools = None if state.get("final_without_tools") else schemas
        t0 = time.time()
        with span("model_step", "LLM", step=state["steps"] + 1, tools_offered=bool(offer_tools)) as s:
            try:
                r = _chat(client, model, state["messages"], offer_tools)
            except RateLimitError:
                raise  # out of quota is not an agent failure: the caller decides (wait, or stop and resume)
            except BadRequestError as e:
                bad = UNKNOWN_TOOL.search(str(e))
                if bad and state.get("bad_tool_retries", 0) < 2:
                    # The provider rejected a call to a tool that doesn't exist (gpt-oss sometimes calls its
                    # built-in tools, e.g. 'container.exec'): tell the model and let it try again
                    entry = {"step": state["steps"] + 1, "tool": bad.group(1), "args": "", "ok": False,
                             "error": f"Error: there is no tool called '{bad.group(1)}'.", "result_rows": 0,
                             "result_chars": 0, "latency_s": 0.0, "write_attempt": False}
                    nudge = (f"There is no tool called '{bad.group(1)}'. The only tools are: {', '.join(fns)}. "
                             "Call one of them, or give your final answer.")
                    return {"messages": state["messages"] + [{"role": "user", "content": nudge}],
                            "steps": state["steps"] + 1, "tool_calls": state["tool_calls"] + [entry],
                            "bad_tool_retries": state.get("bad_tool_retries", 0) + 1}
                return {"stop_reason": "llm_error", "error": f"{type(e).__name__}: {e}", "steps": state["steps"] + 1}
            except Exception as e:  # network errors after retries, anything unexpected
                if is_account_error(e):
                    raise  # out of credit (402) or a bad key: not an agent failure, the caller stops the run
                return {"stop_reason": "llm_error", "error": f"{type(e).__name__}: {e}",
                        "steps": state["steps"] + 1}
            usage = getattr(r, "usage", None)
            used = (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)
            msg = r.choices[0].message
            s.set_attributes({"tokens": used, "latency_s": round(time.time() - t0, 3),
                              "tool_calls": [c.function.name for c in (msg.tool_calls or [])]})
        return {"messages": state["messages"] + [_assistant_dict(msg)], "steps": state["steps"] + 1,
                "tokens": state["tokens"] + used}

    def run_tools(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        messages, log = list(state["messages"]), list(state["tool_calls"])
        seen, rows, repeats = dict(state["seen"]), list(state["rows"]), state["repeats"]
        stop = None
        for call in last.get("tool_calls", []):
            name, raw = call["function"]["name"], call["function"]["arguments"]
            entry = {"step": state["steps"], "tool": name, "args": raw, "ok": False, "error": None,
                     "result_rows": 0, "result_chars": 0, "latency_s": 0.0, "write_attempt": False}
            t0 = time.time()
            with span(f"tool:{name}", "TOOL", tool=name, args=raw) as s:
                content = None
                try:
                    args = json.loads(raw or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except ValueError as e:
                    content = f"Error: arguments are not valid JSON ({e}). Send a JSON object."
                if content is None and name not in fns:
                    content = f"Error: there is no tool called '{name}'. Available: {', '.join(fns)}."
                if content is None:
                    try:
                        args = models[name](**args).model_dump()
                    except ValidationError as e:
                        problems = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'arguments'}: {x['msg']}"
                                             for x in e.errors())
                        content = f"Error: invalid arguments for {name}: {problems}. Fix them and try again."
                if content is None:
                    key = name + json.dumps(args, sort_keys=True)
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] > 1:
                        repeats += 1
                        entry["error"] = "repeated call"
                        content = LOOP_NUDGE.format(tool=name)
                        if repeats >= 2:
                            stop = "loop"
                    else:
                        try:
                            result = fns[name](**args)
                            result_rows = result if isinstance(result, list) else [result]
                            rows.extend(result_rows)
                            text = json.dumps(result, default=str)
                            entry.update(ok=True, result_rows=len(result_rows), result_chars=len(text))
                            if len(text) > MAX_RESULT_CHARS:
                                text = text[:MAX_RESULT_CHARS] + " ... [truncated: narrow the query]"
                            content = text if result_rows else _empty_result_hint(fns, name, args)
                        except Exception as e:  # ToolError from the tool, or anything unexpected
                            msg = str(e) or type(e).__name__
                            entry["write_attempt"] = "only read queries" in msg
                            content = f"Error from {name}: {msg}"
                if entry["error"] is None and not entry["ok"]:
                    entry["error"] = content
                entry["latency_s"] = round(time.time() - t0, 3)
                s.set_attributes({"ok": entry["ok"], "result_rows": entry["result_rows"],
                                  "result_chars": entry["result_chars"], "latency_s": entry["latency_s"]})
            log.append(entry)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        out = {"messages": messages, "tool_calls": log, "seen": seen, "rows": rows, "repeats": repeats}
        if stop:
            out["stop_reason"] = stop
        return out

    def check_answer(state: AgentState) -> AgentState:
        answer = state["messages"][-1].get("content", "").strip()
        cited = find_citations(answer, state["rows"])
        nothing = is_no_answer(answer) or not state["rows"]
        if not cited and not nothing and not state.get("nudged_cite") and state["steps"] < max_steps:
            return {"messages": state["messages"] + [{"role": "user", "content": CITE_NUDGE}], "nudged_cite": True}
        return {"answer": answer, "citations": cited, "stop_reason": state.get("stop_reason") or "answered"}

    def force_final(state: AgentState) -> AgentState:
        return {"messages": state["messages"] + [{"role": "user", "content": STEP_LIMIT_NUDGE}],
                "final_without_tools": True, "stop_reason": "step_limit"}

    def after_model(state: AgentState) -> Literal["tools", "check", "model", "force_final", "end"]:
        if state.get("stop_reason") == "llm_error":
            return "end"
        if state["tokens"] >= token_budget:
            return "end"
        if state["messages"][-1]["role"] == "user":  # rejected call to a non-existent tool: try again
            return "force_final" if state["steps"] >= max_steps else "model"
        return "tools" if state["messages"][-1].get("tool_calls") else "check"

    def after_tools(state: AgentState) -> Literal["model", "force_final", "end"]:
        if state.get("stop_reason") == "loop" or state["tokens"] >= token_budget:
            return "end"
        return "force_final" if state["steps"] >= max_steps else "model"

    def after_check(state: AgentState) -> Literal["model", "end"]:
        return "end" if state.get("answer") is not None else "model"

    g = StateGraph(AgentState)
    g.add_node("model", call_model)
    g.add_node("tools", run_tools)
    g.add_node("check", check_answer)
    g.add_node("force_final", force_final)
    g.set_entry_point("model")
    g.add_conditional_edges("model", after_model, {"tools": "tools", "check": "check", "model": "model",
                                                   "force_final": "force_final", "end": END})
    g.add_conditional_edges("tools", after_tools, {"model": "model", "force_final": "force_final", "end": END})
    g.add_conditional_edges("check", after_check, {"model": "model", "end": END})
    g.add_edge("force_final", "model")
    return g.compile()


@trace(span_type="AGENT", name="agent_loop")
def run_agent(question: str, conn=None, client=None, model: str | None = None, max_steps: int = MAX_STEPS,
              token_budget: int = TOKEN_BUDGET, tools: tuple | None = None) -> dict:
    """Answer one question with the tool loop. Same keys as query_graph.answer_question, plus the trace."""
    from utils.llm import QUERY_MODEL, get_client
    if conn is not None:
        import mcp_server
        mcp_server.set_connection(conn)
    client = client or get_client("query")
    agent = build_agent(client, model or QUERY_MODEL, max_steps, token_budget, tools)
    t0 = time.time()
    state = agent.invoke({
        "question": question, "steps": 0, "tokens": 0, "tool_calls": [], "seen": {}, "repeats": 0, "rows": [],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
    }, {"recursion_limit": 4 * max_steps + 10})
    stop = state.get("stop_reason")
    if state.get("answer") is None:  # ended on budget, loop or error before a final answer
        stop = stop if stop in ("loop", "llm_error") else ("token_budget" if state["tokens"] >= token_budget else stop)
    cyphers = [json.loads(c["args"]).get("query", "") for c in state["tool_calls"]
               if c["tool"] == "run_cypher" and c["args"].strip().startswith("{")]
    return {
        "question": question,
        "answer": state.get("answer") or "",
        "results": state["rows"],
        "cypher": "\n;\n".join(q for q in cyphers if q),
        "error": state.get("error") or (None if state.get("answer") is not None else f"stopped: {stop}"),
        "mode": "agent",
        "steps": state["steps"],
        "tokens": state["tokens"],
        "tool_calls": state["tool_calls"],
        "stop_reason": stop,
        "citations": state.get("citations", []),
        "uncited": state.get("answer") is not None and not state.get("citations")
                   and not is_no_answer(state.get("answer")),
        "latency_s": round(time.time() - t0, 2),
    }
