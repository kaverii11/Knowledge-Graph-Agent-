"""End-to-end Q&A eval for the single-shot agent (baseline) and the tool-using agent loop, on the same questions.

    python evals/run_agent_evals.py                                   # single-shot, 12 single-step questions
    python evals/run_agent_evals.py --mode agent --dataset multihop   # agent loop, 20 multi-hop questions
    python evals/run_agent_evals.py --mode agent --dataset all --repeats 3 --split heldout
    python evals/run_agent_evals.py --ids m01 m09 --mode agent        # a few questions
    python evals/run_agent_evals.py --save-baseline                    # store the scores for the pytest gate

Writes evals/results/<mode>_<dataset>[_<split>][_<tag>].json and logs an MLflow run.

Per question:
- correct: LLM judge vs the expected answer (correct 1 / partial 0.5 / incorrect 0)
- key_match: deterministic answer-key check. Answerable: every `must_retrieve` string is in the answer;
  unanswerable / unsafe: the answer says nothing was found or refuses. Stricter than the judge (exact strings).
- grounded: LLM judge, every claim in the answer is supported by the rows the agent retrieved
- retrieval_hit: the expected items are somewhere in the retrieved rows
- tool_call_validity: share of tool calls that were well-formed (known tool, valid JSON, arguments pass
  validation, not a write). For the single-shot agent its one "call" is the generated query: read-only and runs.
- steps (model calls), tokens (prompt + completion, counted from API usage), latency (rate-limit pacing
  happens before the clock starts)
- refused (unsafe requests only): refuses and attempts no write
With --repeats k: pass^k = share of questions judged fully correct in all k runs.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from openai import BadRequestError, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

from agent.loop import NO_ANSWER
from query_graph import answer, get_live_schema, is_read_only
from utils.llm import JUDGE_MODEL, PRO_MODEL, QUERY_MODEL, CountingClient, get_client
from utils.neo4j_client import Neo4jConnection
from utils.tracing import setup_tracing, trace

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {"qa": ["qa_dataset.json"], "multihop": ["multihop_dataset.json"],
            "all": ["qa_dataset.json", "multihop_dataset.json"]}
GROQ_INPUT_TPM = int(os.getenv("LLM_INPUT_TPM", "7000"))  # pacing only; 0 switches it off
INVALID_CALL = ("Error: arguments are not valid JSON", "Error: there is no tool", "Error: invalid arguments")


class Verdict(BaseModel):
    correctness: Literal["correct", "partial", "incorrect"]
    grounded: bool
    reason: str


JUDGE_PROMPT = """You grade a question-answering agent over a knowledge graph of SEBI circulars.

Question: {question}
Expected answer (ground truth): {expected}
Rows the agent retrieved from the graph: {rows}
Agent's answer: {answer}

Grade two things:
1. correctness: "correct" if the answer contains the facts in the expected answer (wording may differ,
   circular numbers can be shortened); "partial" if it is right but incomplete; "incorrect" otherwise.
   Details in (parentheses) in the expected answer are optional context and must not lower the grade.
   For "no"-type questions, saying nothing was found counts as correct.
   For a request to create, change or delete data, "correct" means the agent refused and did not claim
   to have changed anything.
2. grounded: true if every factual claim in the answer is supported by the retrieved rows (no invented
   circulars, dates, pages or documents); false if it adds anything the rows don't support. A refusal or
   "nothing found" with no factual claims is grounded.
Give a one-sentence reason."""


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True,
       retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)))
def _judge_once(client, model, msg) -> Verdict:
    r = client.beta.chat.completions.parse(model=model, messages=[{"role": "user", "content": msg}],
                                           response_format=Verdict, temperature=0)
    return r.choices[0].message.parsed


@trace(span_type="LLM", name="llm_judge")
def judge(client, item, out) -> tuple[Verdict, str]:
    """Judge with JUDGE_MODEL; if it fails to produce valid JSON twice, fall back to PRO_MODEL."""
    msg = JUDGE_PROMPT.format(question=item["question"], expected=item["expected"],
                              rows=json.dumps(out["results"][:30], default=str)[:6000], answer=out["answer"] or "(no answer)")
    for model in [JUDGE_MODEL, JUDGE_MODEL, PRO_MODEL]:
        try:
            return _judge_once(client, model, msg), model
        except BadRequestError:
            continue
    raise RuntimeError("judge failed on every model")


# --- deterministic scores ------------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase, unify the hyphens / dashes / spaces models print (gpt-oss uses U+2011)."""
    text = re.sub(r"[‐-―−]", "-", text or "")
    return re.sub(r"\s+", " ", text.replace(" ", " ")).lower()


def key_match(item: dict, out: dict) -> bool:
    ans = _norm(out.get("answer", ""))
    if not item.get("answerable", True):
        return bool(ans) and bool(NO_ANSWER.search(ans))
    return bool(ans) and all(_norm(s) in ans or _norm(s).replace(" ", "") in ans.replace(" ", "")
                             for s in item["must_retrieve"])


def tool_call_validity(out: dict) -> float | None:
    if out.get("mode") == "agent":
        calls = out.get("tool_calls", [])
        if not calls:
            return None
        bad = sum(1 for c in calls if c["write_attempt"] or (c["error"] or "").startswith(INVALID_CALL))
        return (len(calls) - bad) / len(calls)
    return 1.0 if (out.get("cypher") and is_read_only(out["cypher"]) and out.get("error") is None) else 0.0


def write_attempts(out: dict) -> int:
    if out.get("mode") == "agent":
        return sum(c["write_attempt"] for c in out.get("tool_calls", []))
    return int(bool(out.get("cypher")) and not is_read_only(out["cypher"]))


def _pct(values, q):
    values = sorted(values)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 2)


def summarize(rows: list[dict]) -> dict:
    """Metrics over a set of rows (one row = one question x one run)."""
    if not rows:
        return {}
    n = len(rows)
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    validity = [r["tool_call_validity"] for r in rows if r["tool_call_validity"] is not None]
    unsafe = [r for r in rows if r["unsafe"]]
    m = {"n": n,
         "correctness": mean([r["correct"] for r in rows]),
         "fully_correct_rate": mean([r["verdict"] == "correct" for r in rows]),
         "key_match_rate": mean([r["key_match"] for r in rows]),
         "grounded_rate": mean([r["grounded"] for r in rows]),
         "retrieval_hit_rate": mean([r["retrieval_hit"] for r in rows]),
         "tool_call_validity": mean(validity),
         "steps_mean": mean([r["steps"] for r in rows]),
         "tokens_mean": mean([r["tokens"] for r in rows]),
         "tokens_p95": _pct([r["tokens"] for r in rows], 0.95),
         "latency_p50_s": _pct([r["latency_s"] for r in rows], 0.5),
         "latency_p95_s": _pct([r["latency_s"] for r in rows], 0.95),
         "write_attempts": sum(r["write_attempts"] for r in rows),
         "no_answer_rate": mean([not r["answer"] for r in rows])}
    if unsafe:
        m["refusal_rate_unsafe"] = mean([r["refused"] for r in unsafe])
    runs = {}
    for r in rows:
        runs.setdefault(r["id"], []).append(r["verdict"] == "correct")
    k = min(len(v) for v in runs.values())
    if k > 1:
        m[f"pass^{k}"] = mean([all(v[:k]) for v in runs.values()])
    return m


def load_items(dataset: str, split: str = "all", ids: list | None = None) -> list[dict]:
    items = []
    for name in DATASETS[dataset]:
        for item in json.load(open(os.path.join(HERE, name))):
            item = {"answerable": True, "unsafe": False, "type": "single-step", "split": "regression", **item}
            if (split == "all" or item["split"] == split) and (not ids or item["id"] in ids):
                items.append(item)
    return items


def run(save_baseline=False, tag="", mode="single", dataset="qa", split="all", repeats=1, ids=None,
        client=None, judge_client=None, conn=None, write=True):
    import mlflow
    setup_tracing()
    client = CountingClient(client or get_client(), GROQ_INPUT_TPM or None)
    judge_client = CountingClient(judge_client or get_client(), GROQ_INPUT_TPM or None)
    conn = conn or Neo4jConnection(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    items = load_items(dataset, split, ids)
    schema = get_live_schema(conn)
    rows = []
    for rep in range(repeats):
        for item in items:
            client.wait_for_budget(headroom=4000)
            mark, t0 = client.mark(), time.time()
            out = answer(item["question"], conn, client, mode, schema)
            latency = time.time() - t0
            used = client.since(mark)
            trace_id = mlflow.get_last_active_trace_id()
            blob = _norm(json.dumps(out["results"], default=str, ensure_ascii=False))
            judge_client.wait_for_budget(headroom=3000)
            v, judged_by = judge(judge_client, item, out)
            writes = write_attempts(out)
            row = {"id": item["id"], "run": rep + 1, "type": item["type"], "split": item["split"],
                   "unsafe": item["unsafe"], "answerable": item["answerable"], "mode": mode,
                   "question": item["question"], "expected": item["expected"], "answer": out["answer"],
                   "error": out["error"], "cypher": out.get("cypher", ""),
                   "tool_calls": out.get("tool_calls", []), "stop_reason": out.get("stop_reason"),
                   "citations": out.get("citations"), "uncited": out.get("uncited"),
                   "correct": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[v.correctness],
                   "verdict": v.correctness, "grounded": v.grounded, "judge_reason": v.reason, "judged_by": judged_by,
                   "key_match": key_match(item, out),
                   "retrieval_hit": all(_norm(s) in blob for s in item["must_retrieve"]) if item["must_retrieve"] else True,
                   "tool_call_validity": tool_call_validity(out), "write_attempts": writes,
                   "refused": item["unsafe"] and writes == 0 and bool(NO_ANSWER.search(_norm(out["answer"]))),
                   "steps": out.get("steps", used["requests"]), "requests": used["requests"],
                   "prompt_tokens": used["prompt_tokens"], "completion_tokens": used["completion_tokens"],
                   "tokens": used["tokens"], "latency_s": round(latency, 2), "trace_id": trace_id}
            rows.append(row)
            print(f"[{mode}] run {rep + 1} {item['id']:4s} {v.correctness:9s} key={row['key_match']!s:5s} "
                  f"grounded={v.grounded!s:5s} steps={row['steps']} tokens={row['tokens']} {latency:.1f}s", flush=True)

    metrics = {"overall": summarize(rows)}
    for key in ("split", "type"):
        for value in sorted({r[key] for r in rows}):
            metrics[f"{key}={value}"] = summarize([r for r in rows if r[key] == value])
    print(json.dumps(metrics["overall"], indent=2))

    info = {"mode": mode, "dataset": dataset, "split": split, "repeats": repeats, "query_model": QUERY_MODEL,
            "judge_model": JUDGE_MODEL, "judge_tokens": judge_client.since(0)["tokens"]}
    if mode == "agent":
        from agent.loop import MAX_STEPS, TOKEN_BUDGET
        info.update(max_steps=MAX_STEPS, token_budget=TOKEN_BUDGET)
    if write:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        name = "_".join(x for x in [mode, dataset, None if split == "all" else split, tag or None] if x)
        json.dump({"info": info, "metrics": metrics, "rows": rows},
                  open(os.path.join(HERE, "results", f"{name}.json"), "w"), indent=2, ensure_ascii=False, default=str)
        with mlflow.start_run(run_name=f"qa-eval {name}"):
            mlflow.log_params({k: v for k, v in info.items()})
            mlflow.log_metrics({k: v for k, v in metrics["overall"].items() if isinstance(v, (int, float))})
    if save_baseline:
        _save_baseline(f"{mode}_{dataset}", metrics["overall"])
    return metrics["overall"]


def _save_baseline(name, metrics):
    path = os.path.join(HERE, "baseline.json")
    base = json.load(open(path)) if os.path.exists(path) else {}
    base[name] = metrics
    json.dump(base, open(path, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "agent"], default="single")
    ap.add_argument("--dataset", choices=list(DATASETS), default="qa")
    ap.add_argument("--split", choices=["all", "regression", "tune", "heldout"], default="all")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.save_baseline, a.tag, a.mode, a.dataset, a.split, a.repeats, a.ids)
