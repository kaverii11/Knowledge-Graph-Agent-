"""The Q&A eval runner end to end with a fake LLM, fake judge and fake graph (no API key or database)."""
from types import SimpleNamespace as NS

from evals.run_agent_evals import Verdict, key_match, run, summarize, tool_call_validity
from tests.test_agent_loop import FakeConn, FakeLLM, call, reply


class FakeJudge:
    def __init__(self, verdict="correct"):
        self.beta = NS(chat=NS(completions=NS(parse=lambda **kw: NS(choices=[NS(message=NS(
            parsed=Verdict(correctness=verdict, grounded=True, reason="fake")))]))))
        self.chat = NS(completions=NS(create=None))


def test_agent_mode_run_scores_tokens_steps_and_key_match():
    llm = FakeLLM(
        reply(tool_calls=[call("circular_references", circular_id="I/24814/2026")], tokens=(500, 50)),
        reply(tool_calls=[call("who_cites", document="FPI Master Circular", relation="modifies")], tokens=(800, 50)),
        reply("Also modified by I/19251/2026 and I/20296/2026.", tokens=(900, 40)),
    )
    m = run(mode="agent", dataset="multihop", ids=["m01"], client=llm, judge_client=FakeJudge(), conn=FakeConn(), write=False)
    assert m["n"] == 1 and m["correctness"] == 1.0 and m["key_match_rate"] == 1.0
    assert m["steps_mean"] == 3 and m["tokens_mean"] == 2340 and m["tool_call_validity"] == 1.0


def test_unsafe_request_scores_refusal_and_no_writes():
    llm = FakeLLM(reply(tool_calls=[call("run_cypher", query="MATCH (c:Circular) DETACH DELETE c")]),
                  reply("I can't delete anything: the graph is read-only."))
    m = run(mode="agent", dataset="multihop", ids=["m18"], client=llm, judge_client=FakeJudge(), conn=FakeConn(), write=False)
    assert m["write_attempts"] == 1 and m["refusal_rate_unsafe"] == 0.0     # refused in words, but tried a write
    assert m["tool_call_validity"] == 0.0


def test_key_match_normalises_model_hyphens_and_handles_no_answer():
    item = {"answerable": True, "must_retrieve": ["HO/19/34/14(8)2026-AFD-POD2/I/19251/2026"]}
    assert key_match(item, {"answer": "It is HO/19/34/14(8)2026‑AFD‑POD2/I/19251/2026."})
    assert not key_match(item, {"answer": "It is I/20296/2026."})
    assert key_match({"answerable": False, "must_retrieve": []}, {"answer": "No circular in the graph modifies them."})
    assert not key_match({"answerable": False, "must_retrieve": []}, {"answer": "I/24467/2026 modifies them."})


def test_tool_call_validity_counts_malformed_and_write_calls():
    ok = {"write_attempt": False, "error": None}
    bad_args = {"write_attempt": False, "error": "Error: invalid arguments for who_cites: relation: ..."}
    tool_failed = {"write_attempt": False, "error": "Error from circular_references: No circular matching"}
    assert tool_call_validity({"mode": "agent", "tool_calls": [ok, bad_args, tool_failed, ok]}) == 0.75
    assert tool_call_validity({"mode": "single", "cypher": "MATCH (n) RETURN n", "error": None}) == 1.0
    assert tool_call_validity({"mode": "single", "cypher": "", "error": "no query generated"}) == 0.0


def test_pass_k_needs_every_run_correct():
    row = lambda i, v: {"id": i, "verdict": v, "correct": 1.0 if v == "correct" else 0.0, "key_match": True,
                        "grounded": True, "retrieval_hit": True, "tool_call_validity": 1.0, "steps": 1, "tokens": 1,
                        "latency_s": 1.0, "write_attempts": 0, "answer": "x", "unsafe": False}
    m = summarize([row("a", "correct"), row("b", "correct"), row("a", "correct"), row("b", "partial"),
                   row("a", "correct"), row("b", "correct")])
    assert m["pass^3"] == 0.5 and m["fully_correct_rate"] == round(5 / 6, 3)
