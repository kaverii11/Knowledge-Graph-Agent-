"""Extraction eval: compares extracted references with a hand-labelled answer key.

    python evals/extraction_eval.py evals/ground_truth/fpi_kyc_review_2026.json data/fpi_kyc_review_2026.json
    python evals/extraction_eval.py --gt-dir evals/ground_truth --pred-dir data --log-mlflow --tag "after prompt fix"

See evals/README.md for the labelling rules.
"""
import argparse
import difflib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def is_match(agent_title: str, gt_title: str, threshold: float = 0.8) -> bool:
    """Exact (normalized), subset ("the sebi act, 1992" vs "sebi act, 1992") or fuzzy match."""
    a, g = normalize_text(agent_title), normalize_text(gt_title)
    if not a or not g:
        return False
    if a == g or a in g or g in a:
        return True
    return difflib.SequenceMatcher(None, a, g).ratio() >= threshold


def matches_item(agent_title: str, gt_item: dict) -> bool:
    return any(is_match(agent_title, t) for t in [gt_item["title"], *gt_item.get("aliases", [])])


def score(ground_truth: list, predictions: list) -> dict:
    matched = [False] * len(ground_truth)
    correct, duplicates, wrong = [], [], []
    relation_checks = []
    for p in predictions:
        page, title = p.get("page_number", -1), p.get("title", "")
        hits = [i for i, g in enumerate(ground_truth) if g["page_number"] == page and matches_item(title, g)]
        new = [i for i in hits if not matched[i]]
        if new:
            matched[new[0]] = True
            correct.append(p)
            if "relation" in ground_truth[new[0]]:
                relation_checks.append(p.get("relation", "references") == ground_truth[new[0]]["relation"])
        elif hits:
            duplicates.append(p)
        else:
            wrong.append(p)
    tp, fp, fn = len(correct), len(wrong), matched.count(False)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    avg = lambda xs: sum(x.get("confidence_score", 0.0) for x in xs) / len(xs) if xs else None
    return {
        "correct": tp, "wrong": fp, "missed": fn, "duplicates": len(duplicates),
        "precision": precision, "recall": recall, "f1": f1,
        "relation_right": sum(relation_checks), "relation_total": len(relation_checks),
        "avg_conf_correct": avg(correct), "avg_conf_wrong": avg(wrong),
        "wrong_titles": [f"p{x['page_number']}: {x['title']}" for x in wrong],
        "missed_titles": [f"p{g['page_number']}: {g['title']}" for g, m in zip(ground_truth, matched) if not m],
    }


def combine(per_file: dict) -> dict:
    tp = sum(r["correct"] for r in per_file.values())
    fp = sum(r["wrong"] for r in per_file.values())
    fn = sum(r["missed"] for r in per_file.values())
    dup = sum(r["duplicates"] for r in per_file.values())
    rel_ok = sum(r["relation_right"] for r in per_file.values())
    rel_n = sum(r["relation_total"] for r in per_file.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"correct": tp, "wrong": fp, "missed": fn, "duplicates": dup,
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "relation_accuracy": rel_ok / rel_n if rel_n else 0.0, "relation_right": rel_ok, "relation_total": rel_n}


def print_report(name: str, r: dict):
    print(f"\n== {name}")
    print(f"correct {r['correct']}  wrong {r['wrong']}  missed {r['missed']}  duplicates {r['duplicates']}")
    print(f"precision {r['precision']:.3f}  recall {r['recall']:.3f}  F1 {r['f1']:.3f}"
          f"  relation (references/modifies) right {r['relation_right']}/{r['relation_total']}")
    if r.get("avg_conf_correct") is not None or r.get("avg_conf_wrong") is not None:
        print(f"avg confidence: correct {r.get('avg_conf_correct')}  wrong {r.get('avg_conf_wrong')}")
    for t in r.get("wrong_titles", []):
        print("  wrong:", t)
    for t in r.get("missed_titles", []):
        print("  missed:", t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth", nargs="?")
    ap.add_argument("agent_output", nargs="?")
    ap.add_argument("--gt-dir")
    ap.add_argument("--pred-dir")
    ap.add_argument("--log-mlflow", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pairs = {}
    if args.gt_dir:
        for f in sorted(os.listdir(args.gt_dir)):
            if f.endswith(".json"):
                pairs[f[:-5]] = (os.path.join(args.gt_dir, f), os.path.join(args.pred_dir, f))
    else:
        pairs["single"] = (args.ground_truth, args.agent_output)

    per_file = {}
    for name, (gt_path, pred_path) in pairs.items():
        gt = json.load(open(gt_path, encoding="utf-8"))
        pred = json.load(open(pred_path, encoding="utf-8"))
        per_file[name] = score(gt, pred)
        print_report(name, per_file[name])
    overall = combine(per_file)
    print_report("OVERALL", overall)

    if args.log_mlflow:
        import mlflow
        from utils.tracing import setup_tracing
        setup_tracing()
        with mlflow.start_run(run_name=f"extraction-eval {args.tag}".strip()):
            mlflow.log_param("tag", args.tag)
            mlflow.log_metrics({k: v for k, v in overall.items() if isinstance(v, (int, float))})
            mlflow.log_dict(per_file, "per_file.json")
    return overall


if __name__ == "__main__":
    main()
