"""Agent loop guardrails, with a scripted fake LLM and a fake graph: no API key or database needed.

The tools are the real MCP tool functions (real argument validation, real write-query rejection);
only the database connection and the LLM are fakes.
"""
import json
from types import SimpleNamespace as NS

import pytest

import mcp_server
from agent.loop import build_tools, find_citations, run_agent

KYC = "HO/19/34/14(10)2026-AFD-POD3/I/24814/2026"
POA = "HO/19/34/14(8)2026-AFD-POD2/I/19251/2026"


class FakeConn:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, params=None, read_only=False):
        self.calls.append((query, params, read_only))
        if "MATCH (c:Circular) OPTIONAL MATCH" in query:  # list_circulars
            return [{"circular_id": KYC, "subject": "KYC review for FPIs", "date": "October 30, 2026", "documents_cited": 3}]
        if "WHERE c.id = $id" in query:                     # circular_references
            if params["id"].endswith("24814/2026"):
                return [{"circular_id": KYC, "document": "FPI Master Circular", "type": "circular",
                         "relation": "REFERENCES", "page": 1, "evidence": "..."}]
            return []
        if "[r:MODIFIES]" in query:                         # who_cites, relation=modifies
            return [{"circular_id": POA, "subject": "Digitally signed PoA", "date": "August 20, 2026",
                     "document": "FPI Master Circular", "relation": "MODIFIES", "page": 2}]
        return []


def call(name, **args):
    return NS(id=f"call_{name}_{len(json.dumps(args))}", function=NS(name=name, arguments=json.dumps(args)))


def raw_call(name, arguments):
    return NS(id=f"call_{name}", function=NS(name=name, arguments=arguments))


def reply(content="", tool_calls=None, tokens=(100, 20)):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))],
              usage=NS(prompt_tokens=tokens[0], completion_tokens=tokens[1]))


class FakeLLM:
    """Returns the scripted replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.sent.append(kwargs)
        return self.replies.pop(0)


@pytest.fixture
def conn():
    c = FakeConn()
    mcp_server.set_connection(c)
    yield c
    mcp_server.set_connection(None)


def ask(llm, conn, **kw):
    return run_agent("Which circulars modify a document the KYC circular references?", conn, llm, model="fake", **kw)


def tool_messages(llm, i=-1):
    return [m["content"] for m in llm.sent[i]["messages"] if m["role"] == "tool"]


def test_multi_step_answer_with_citations(conn):
    llm = FakeLLM(
        reply(tool_calls=[call("circular_references", circular_id="I/24814/2026")]),
        reply(tool_calls=[call("who_cites", document="FPI Master Circular", relation="modifies")]),
        reply("The KYC circular references the FPI Master Circular, which is modified by I/19251/2026."),
    )
    out = ask(llm, conn)
    assert out["stop_reason"] == "answered" and out["error"] is None
    assert out["steps"] == 3 and out["tokens"] == 360
    assert [c["tool"] for c in out["tool_calls"]] == ["circular_references", "who_cites"]
    assert all(c["ok"] for c in out["tool_calls"])
    assert POA in out["citations"]
    assert all(read_only for _, _, read_only in conn.calls)


def test_tools_offered_are_the_five_read_only_tools_without_ask():
    _, _, schemas = build_tools()
    assert {s["function"]["name"] for s in schemas} == {
        "list_circulars", "search_documents", "circular_references", "who_cites", "run_cypher"}


def test_invalid_arguments_are_rejected_before_the_database_and_explained(conn):
    llm = FakeLLM(
        reply(tool_calls=[call("who_cites", document="FPI Master Circular", relation="deletes")]),
        reply(tool_calls=[call("who_cites", document="FPI Master Circular", relation="modifies")]),
        reply("I/19251/2026 modifies the FPI Master Circular."),
    )
    out = ask(llm, conn)
    first, second = out["tool_calls"]
    assert not first["ok"] and "invalid arguments for who_cites" in first["error"] and "relation" in first["error"]
    assert second["ok"]
    assert len(conn.calls) == 1                      # only the corrected call reached the database
    assert "invalid arguments" in tool_messages(llm, 1)[0]
    assert out["stop_reason"] == "answered"


@pytest.mark.parametrize("bad, expected", [
    (raw_call("who_cites", "{not json"), "not valid JSON"),
    (raw_call("delete_everything", "{}"), "no tool called"),
    (call("search_documents", keyword="FPI", limit=5), "Extra inputs are not permitted"),
    (call("search_documents", keyword="x"), "at least 2 characters"),
])
def test_malformed_calls_return_clear_errors(conn, bad, expected):
    llm = FakeLLM(reply(tool_calls=[bad]), reply("Nothing was found in the graph."))
    out = ask(llm, conn)
    assert expected in out["tool_calls"][0]["error"]
    assert conn.calls == []


def test_tool_error_goes_back_to_the_model_and_it_recovers(conn):
    llm = FakeLLM(
        reply(tool_calls=[call("circular_references", circular_id="I/0000/1999")]),
        reply(tool_calls=[call("list_circulars")]),
        reply(tool_calls=[call("circular_references", circular_id="I/24814/2026")]),
        reply("I/24814/2026 references the FPI Master Circular."),
    )
    out = ask(llm, conn)
    assert not out["tool_calls"][0]["ok"] and "list_circulars" in out["tool_calls"][0]["error"]
    assert "Error from circular_references" in tool_messages(llm, 1)[0]
    assert out["stop_reason"] == "answered" and out["citations"]


def test_repeated_call_is_not_run_again_and_a_second_repeat_stops_the_loop(conn):
    same = lambda: [call("who_cites", document="FPI Master Circular", relation="modifies")]
    llm = FakeLLM(reply(tool_calls=same()), reply(tool_calls=same()), reply(tool_calls=same()))
    out = ask(llm, conn)
    assert len(conn.calls) == 1
    assert "already called who_cites" in tool_messages(llm, 2)[-1]
    assert out["stop_reason"] == "loop" and out["answer"] == "" and out["error"] == "stopped: loop"


def test_step_limit_forces_a_final_answer_without_tools(conn):
    llm = FakeLLM(
        reply(tool_calls=[call("list_circulars")]),
        reply(tool_calls=[call("circular_references", circular_id="I/24814/2026")]),
        reply("I/24814/2026 references the FPI Master Circular; I could not check what modifies it."),
    )
    out = ask(llm, conn, max_steps=2)
    assert "tools" not in llm.sent[-1]                       # last call: no tools offered
    assert "Step limit reached" in llm.sent[-1]["messages"][-1]["content"]
    assert out["stop_reason"] == "step_limit" and out["answer"]


def test_token_budget_stops_the_loop(conn):
    llm = FakeLLM(reply(tool_calls=[call("list_circulars")], tokens=(900, 200)),
                  reply(tool_calls=[call("circular_references", circular_id="I/24814/2026")], tokens=(900, 200)))
    out = ask(llm, conn, token_budget=2000)
    assert out["stop_reason"] == "token_budget" and out["answer"] == "" and out["tokens"] == 2200
    assert len(llm.sent) == 2


def test_uncited_answer_gets_one_nudge(conn):
    llm = FakeLLM(
        reply(tool_calls=[call("who_cites", document="FPI Master Circular", relation="modifies")]),
        reply("One circular modifies it."),
        reply("One circular modifies it: I/19251/2026 (August 20, 2026)."),
    )
    out = ask(llm, conn)
    assert "doesn't cite" in llm.sent[-1]["messages"][-1]["content"]
    assert out["answer"].endswith("(August 20, 2026).") and POA in out["citations"] and not out["uncited"]


def test_answer_still_uncited_after_the_nudge_is_flagged(conn):
    llm = FakeLLM(
        reply(tool_calls=[call("who_cites", document="FPI Master Circular", relation="modifies")]),
        reply("One circular modifies it."),
        reply("Yes, one circular modifies it."),
    )
    out = ask(llm, conn)
    assert out["uncited"] and out["stop_reason"] == "answered"


def test_write_query_is_rejected_and_counted(conn):
    llm = FakeLLM(
        reply(tool_calls=[call("run_cypher", query="MATCH (c:Circular) DETACH DELETE c")]),
        reply("I can't do that: the graph is read-only."),
    )
    out = ask(llm, conn)
    assert out["tool_calls"][0]["write_attempt"] and conn.calls == []
    assert out["stop_reason"] == "answered"


def test_refusal_without_tool_calls(conn):
    llm = FakeLLM(reply("I can't delete circulars: the knowledge graph is read-only."))
    out = run_agent("Delete all circulars from the graph", conn, llm, model="fake")
    assert out["tool_calls"] == [] and conn.calls == [] and out["steps"] == 1 and not out["uncited"]


def test_find_citations_accepts_short_circular_numbers():
    rows = [{"circular_id": KYC, "document": "SEBI (Foreign Portfolio Investors) Regulations, 2019"}]
    assert find_citations("See I/24814/2026.", rows) == [KYC]
    assert find_citations("It cites the SEBI (Foreign Portfolio Investors) Regulations, 2019.", rows)
    assert find_citations("No idea.", rows) == []


def test_mode_flag_selects_baseline_or_agent(monkeypatch, conn):
    import agent.loop
    import query_graph
    monkeypatch.setattr(query_graph, "answer_question", lambda q, c, cl, s=None: {"answer": "single", "error": None})
    monkeypatch.setattr(agent.loop, "run_agent", lambda q, c, cl: {"answer": "agent", "mode": "agent"})
    monkeypatch.delenv("ASK_MODE", raising=False)
    assert query_graph.answer("q", conn, None)["answer"] == "single"
    assert query_graph.answer("q", conn, None, mode="agent")["answer"] == "agent"
    monkeypatch.setenv("ASK_MODE", "agent")
    assert query_graph.answer("q", conn, None)["answer"] == "agent"
    with pytest.raises(ValueError):
        query_graph.answer("q", conn, None, mode="nope")


def test_relation_labels_and_dates_are_not_citations():
    rows = [{"circular_id": POA, "date": "August 20, 2026", "relation": "MODIFIES", "type": "circular"}]
    assert find_citations("One circular modifies it, on August 20, 2026.", rows) == []


@pytest.mark.parametrize("text", ["No circulars in the knowledge graph are recorded as modifying it.",
                                  "I’m sorry, but I can’t modify or delete any data in the graph.",
                                  "The Companies Act, 2013 is not cited by any circular."])
def test_refusals_and_nothing_found_are_recognised(text):
    from agent.loop import is_no_answer
    assert is_no_answer(text)


def test_empty_who_cites_says_whether_the_document_exists(conn):
    from agent.loop import _empty_result_hint
    fns = {"search_documents": lambda keyword: [{"name": "SEBI (Debenture Trustees) Regulations, 1993"}]
           if "Trustees" in keyword else []}
    found = _empty_result_hint(fns, "who_cites", {"document": "Debenture Trustees", "relation": "modifies"})
    assert "1 matching document" in found and "That is the answer" in found
    missing = _empty_result_hint(fns, "who_cites", {"document": "Companies Act", "relation": "any"})
    assert "no document in the graph matches" in missing


def test_call_to_a_tool_that_does_not_exist_is_reported_back(conn):
    import httpx
    from openai import BadRequestError
    rejected = BadRequestError("Error code: 400 - Tool call validation failed: attempted to call tool "
                               "'container.exec' which was not in request.tools",
                               response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None)

    class Rejecting(FakeLLM):
        def create(self, **kwargs):
            if not self.sent:
                self.sent.append(kwargs)
                raise rejected
            return super().create(**kwargs)

    llm = Rejecting(reply(tool_calls=[call("who_cites", document="FPI Master Circular", relation="modifies")]),
                    reply("I/19251/2026 modifies the FPI Master Circular."))
    out = ask(llm, conn)
    assert out["stop_reason"] == "answered" and out["tool_calls"][0]["tool"] == "container.exec"
    assert "no tool called 'container.exec'" in llm.sent[1]["messages"][-1]["content"]


def test_daily_limit_is_not_turned_into_a_failed_answer(conn):
    import httpx
    from openai import RateLimitError
    limit = RateLimitError("Rate limit reached for model on tokens per day (TPD): Limit 200000",
                           response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)

    class OutOfQuota(FakeLLM):
        def create(self, **kwargs):
            raise limit

    with pytest.raises(RateLimitError):
        ask(OutOfQuota(), conn)
