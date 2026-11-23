"""How often does the LLM judge agree with a human? No LLM calls.

1. Export a sample of judged answers to label by hand (the judge's verdict is NOT in the sheet):
       python evals/judge_agreement.py export            # samples the rows in results/comparison.json
   -> evals/results/judge_labels.csv  (fill in the `human` column: correct / partial / incorrect)
2. Score agreement:
       python evals/judge_agreement.py score
   -> exact agreement, agreement on correct vs not, Cohen's kappa, and every disagreement.

The sample is stratified: every question type and both modes are represented, chosen with a fixed seed.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "results", "judge_labels.csv")
LABELS = ("correct", "partial", "incorrect")


def _rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        r = json.load(open(p if os.path.isabs(p) else os.path.join(HERE, p)))["rows"]
        rows += [x for v in r.values() for x in v] if isinstance(r, dict) else r  # comparison.json: {mode: rows}
    return rows


def export(paths: list[str], n: int, seed: int = 7):
    rows = _rows(paths)
    rng = random.Random(seed)
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r["type"]), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    picked, keys = [], sorted(groups)
    while len(picked) < min(n, len(rows)):  # round-robin over (mode, type) so every stratum is represented
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    rng.shuffle(picked)
    os.makedirs(os.path.dirname(SHEET), exist_ok=True)
    with open(SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "question", "expected", "answer", "human"])
        for r in picked:
            w.writerow([f"{r['mode']}|{r['id']}|{r.get('run', 1)}", r["question"], r["expected"], r["answer"], ""])
    print(f"Wrote {len(picked)} rows to {SHEET}. Fill in 'human' with: {', '.join(LABELS)}")


def kappa(a: list, b: list) -> float | None:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def score(paths: list[str]):
    verdicts = {f"{r['mode']}|{r['id']}|{r.get('run', 1)}": r["verdict"] for r in _rows(paths)}
    labelled = [r for r in csv.DictReader(open(SHEET, encoding="utf-8")) if r["human"].strip()]
    bad = [r["human"] for r in labelled if r["human"].strip().lower() not in LABELS]
    if bad:
        raise SystemExit(f"Unknown labels {bad}; use {LABELS}")
    human = [r["human"].strip().lower() for r in labelled]
    judge = [verdicts[r["key"]] for r in labelled]
    n = len(human)
    binary = lambda xs: ["correct" if x == "correct" else "not" for x in xs]
    out = {"n": n,
           "exact_agreement": round(sum(h == j for h, j in zip(human, judge)) / n, 3),
           "kappa_3_labels": kappa(human, judge),
           "agreement_correct_vs_not": round(sum(h == j for h, j in zip(binary(human), binary(judge))) / n, 3),
           "kappa_correct_vs_not": kappa(binary(human), binary(judge)),
           "disagreements": [{"key": r["key"], "human": h, "judge": j}
                             for r, h, j in zip(labelled, human, judge) if h != j]}
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(HERE, "results", "judge_agreement.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "score"])
    ap.add_argument("results", nargs="*", default=["results/comparison.json"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    export(a.results, a.n) if a.action == "export" else score(a.results)
