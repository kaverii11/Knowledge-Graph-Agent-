"""End-to-end agent eval: question -> generated Cypher -> rows -> answer, scored four ways.

    python evals/run_agent_evals.py                    # writes evals/results/agent_eval.json, logs to MLflow
    python evals/run_agent_evals.py --save-baseline    # also stores the scores as the regression baseline

Scores: query_ok (read-only and runs), retrieval_hit (expected items in the rows), and two LLM-as-a-judge
scores: correct (vs the expected answer: 1 / 0.5 / 0) and grounded (no claims beyond the rows).
"""
import argparse
import json
import os
import sys
import time
from typing import Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from openai import BadRequestError, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError

from query_graph import answer_question, get_live_schema, is_read_only
from utils.llm import JUDGE_MODEL, PRO_MODEL, get_client
from utils.neo4j_client import Neo4jConnection
from utils.tracing import setup_tracing, trace

HERE = os.path.dirname(os.path.abspath(__file__))


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
2. grounded: true if every factual claim in the answer is supported by the retrieved rows (no invented
   circulars, dates, pages or documents); false if it adds anything the rows don't support.
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


def run(save_baseline=False, tag=""):
    import mlflow
    setup_tracing()
    client = get_client()
    conn = Neo4jConnection(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    data = json.load(open(os.path.join(HERE, "qa_dataset.json")))
    schema = get_live_schema(conn)
    rows = []
    for item in data:
        t0 = time.time()
        out = answer_question(item["question"], conn, client, schema)
        latency = time.time() - t0
        trace_id = mlflow.get_last_active_trace_id()
        blob = json.dumps(out["results"], default=str)
        query_ok = bool(out["cypher"]) and is_read_only(out["cypher"]) and out["error"] is None
        retrieval_hit = all(s in blob for s in item["must_retrieve"]) if item["must_retrieve"] else True
        v, judged_by = judge(client, item, out)
        rows.append({"id": item["id"], "question": item["question"], "cypher": out["cypher"], "answer": out["answer"],
                     "error": out["error"], "query_ok": query_ok, "retrieval_hit": retrieval_hit,
                     "correct": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[v.correctness],
                     "verdict": v.correctness, "grounded": v.grounded, "judge_reason": v.reason,
                     "latency_s": round(latency, 2), "trace_id": trace_id, "judged_by": judged_by})
        print(f"{item['id']} query_ok={query_ok} hit={retrieval_hit} {v.correctness:9s} grounded={v.grounded}  {latency:.1f}s")

    n = len(rows)
    metrics = {"query_ok_rate": sum(r["query_ok"] for r in rows) / n,
               "retrieval_hit_rate": sum(r["retrieval_hit"] for r in rows) / n,
               "correctness": sum(r["correct"] for r in rows) / n,
               "grounded_rate": sum(r["grounded"] for r in rows) / n,
               "avg_latency_s": sum(r["latency_s"] for r in rows) / n}
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump({"metrics": metrics, "rows": rows}, open(os.path.join(HERE, "results", "agent_eval.json"), "w"), indent=2)
    with mlflow.start_run(run_name=f"agent-eval {tag}".strip()):
        mlflow.log_params({"judge_model": JUDGE_MODEL, "questions": n, "tag": tag})
        mlflow.log_metrics(metrics)
        mlflow.log_table({k: [r[k] for r in rows] for k in ["id", "question", "answer", "verdict", "grounded",
                                                             "retrieval_hit", "judge_reason"]}, "agent_eval.json")
    if save_baseline:
        _save_baseline("agent", metrics)
    return metrics


def _save_baseline(name, metrics):
    path = os.path.join(HERE, "baseline.json")
    base = json.load(open(path)) if os.path.exists(path) else {}
    base[name] = metrics
    json.dump(base, open(path, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(a.save_baseline, a.tag)
