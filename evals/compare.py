"""Baseline vs agent comparison table from saved eval results. No LLM calls.

    python evals/compare.py

Uses the question runs where BOTH modes answered the same questions on the same host:
- the 12 single-step questions, run 1 (Groq)
- the 20 multi-hop questions, run 2 (Cloudflare Workers AI)
(The multi-hop run 1 on Groq and run 3 on Cloudflare were not finished for the agent before the free-tier
quotas ran out; their finished rows are used only for the consistency check below.)

Writes evals/results/comparison.json (metrics + the rows used) and prints a Markdown table.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.run_agent_evals import summarize

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    path = os.path.join(R, name)
    if os.path.exists(path):
        return json.load(open(path))["rows"]
    partial = path.replace(".json", ".partial.jsonl")
    return [json.loads(line) for line in open(partial)] if os.path.exists(partial) else []


def rows():
    single = load("single_all.json") + load("single_multihop_pass3.json")
    agent = load("agent_all.json") + load("agent_multihop_pass3.json")
    if not agent:  # these runs were stopped when the free-tier quotas ran out; their finished rows were kept
        agent = [json.loads(line) for f in ("agent_all_run1_unfinished.jsonl", "agent_multihop_pass3_unfinished.jsonl")
                 for line in open(os.path.join(R, f))]
    pick = lambda rs: [r for r in rs if (r["split"] == "regression" and r["run"] == 1)
                       or (r["split"] != "regression" and r["run"] == 2)]
    return pick(single), pick(agent), single, agent


def consistency(all_rows):
    """Questions answered in 2+ runs: share fully correct in every run they have (an incomplete pass^k)."""
    runs = {}
    for r in all_rows:
        if r["split"] != "regression":
            runs.setdefault(r["id"], []).append(r["verdict"] == "correct")
    multi = {k: v for k, v in runs.items() if len(v) >= 2}
    return {"questions": len(multi), "runs": sum(len(v) for v in multi.values()),
            "correct_in_every_run": sum(all(v) for v in multi.values())}


SETS = {"single-step (12, Groq)": lambda r: r["split"] == "regression",
        "multi-hop (20, Cloudflare)": lambda r: r["split"] != "regression",
        "multi-hop held-out (12)": lambda r: r["split"] == "heldout"}
SHOW = [("fully_correct_rate", "Fully correct (judge)"), ("correctness", "Correctness (judge, partial = 0.5)"),
        ("key_match_rate", "Answer-key check (strict strings)"), ("grounded_rate", "Grounded (judge)"),
        ("tool_call_validity", "Valid tool calls / queries"), ("refusal_rate_unsafe", "Unsafe requests refused"),
        ("write_attempts", "Write attempts (all blocked)"), ("steps_mean", "Model calls per question"),
        ("tokens_mean", "Tokens per question"), ("latency_p50_s", "Latency p50 (s)"), ("latency_p95_s", "Latency p95 (s)")]


def main():
    single, agent, all_single, all_agent = rows()
    out = {"sets": {}, "by_type": {}, "consistency": {"single": consistency(all_single), "agent": consistency(all_agent)}}
    for name, keep in SETS.items():
        out["sets"][name] = {"single": summarize([r for r in single if keep(r)]),
                             "agent": summarize([r for r in agent if keep(r)])}
    for t in sorted({r["type"] for r in single if r["split"] != "regression"}):
        s = [r for r in single if r["type"] == t]
        a = [r for r in agent if r["type"] == t]
        out["by_type"][t] = {"n": len(s), "single": sum(r["verdict"] == "correct" for r in s),
                             "agent": sum(r["verdict"] == "correct" for r in a)}
    json.dump({**out, "rows": {"single": single, "agent": agent}},
              open(os.path.join(R, "comparison.json"), "w"), indent=2, ensure_ascii=False, default=str)

    for name, m in out["sets"].items():
        print(f"\n**{name}**\n\n| | Single-shot | Agent loop |\n|---|---|---|")
        for key, label in SHOW:
            s, a = m["single"].get(key), m["agent"].get(key)
            if s is None and a is None:
                continue
            fmt = (lambda v: f"{v:,.0f}") if key == "tokens_mean" else (lambda v: "-" if v is None else f"{v:g}")
            print(f"| {label} | {fmt(s)} | {fmt(a)} |")
    print("\n| Question type | n | Single-shot fully correct | Agent fully correct |\n|---|---|---|---|")
    for t, v in out["by_type"].items():
        print(f"| {t} | {v['n']} | {v['single']} | {v['agent']} |")
    print("\nConsistency (multi-hop questions answered in 2+ runs):", json.dumps(out["consistency"]))


if __name__ == "__main__":
    main()
