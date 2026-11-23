"""Failure taxonomy from the eval traces. No LLM calls.

    python evals/failure_taxonomy.py results/single_all.json results/agent_all.json [more result files]

Every question run that was not judged fully correct, or not grounded, gets one or more tags. Tags that a
rule can see are set automatically; the rest come from evals/failure_labels.json, written by reading the
saved answers / Cypher / tool calls (key "<mode>|<id>|<run>": ["tag", ...]; "-tag" removes an automatic tag
that reading the answer shows is wrong). Each tag says what went wrong:

  wrong_tool        called a tool that can't answer the question (e.g. searched cited documents for a circular)
  bad_arguments     a tool call failed validation, or the single-shot query had a syntax error / wrong label
  stopped_early     answered before it had looked up everything the question asks for
  looped            repeated a call, or ran out of steps / tokens
  hallucinated      claimed something the retrieved rows don't support (judge: not grounded)
  text_date_order   sorted dates as text ("September" > "October"), so newest / oldest came out wrong
  no_refusal        a write request got a write query instead of a refusal (blocked by the read-only guard)
  incomplete        right facts but left part of the question unanswered (judge: partial)
  judge_disagrees   the answer is right on reading it; the judge's grade looks wrong
"""
import argparse
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "failure_labels.json")


def auto_tags(r: dict) -> set:
    tags = set()
    calls = r.get("tool_calls") or []
    if any(c["error"] == "repeated call" for c in calls) or r.get("stop_reason") in ("loop", "step_limit", "token_budget"):
        tags.add("looped")
    if any((c["error"] or "").startswith(("Error: invalid arguments", "Error: arguments are not valid JSON",
                                         "Error: there is no tool")) for c in calls):
        tags.add("bad_arguments")
    if r["mode"] == "single" and r.get("error") and "SyntaxError" in r["error"]:
        tags.add("bad_arguments")
    if r["unsafe"] and r["write_attempts"] and not r.get("refused"):
        tags.add("no_refusal")
    if not r["grounded"]:
        tags.add("hallucinated")
    if r["verdict"] == "partial":
        tags.add("incomplete")
    return tags


def main(paths: list[str]):
    manual = json.load(open(LABELS)) if os.path.exists(LABELS) else {}
    by_mode, listing = {}, []
    for p in paths:
        rows = json.load(open(p if os.path.isabs(p) else os.path.join(HERE, p)))["rows"]
        for r in ([x for v in rows.values() for x in v] if isinstance(rows, dict) else rows):  # comparison.json: {mode: rows}
            if r["verdict"] == "correct" and r["grounded"]:
                continue
            key = f"{r['mode']}|{r['id']}|{r.get('run', 1)}"
            labels = manual.get(key, [])
            remove = {t[1:] for t in labels if t.startswith("-")}  # "-hallucinated": a rule's tag overruled by reading
            tags = sorted((auto_tags(r) | {t for t in labels if not t.startswith("-")}) - remove)
            by_mode.setdefault(r["mode"], Counter()).update(tags or ["untagged"])
            listing.append((key, r["verdict"], r["grounded"], tags, (r.get("judge_reason") or "")[:140]))
    for key, verdict, grounded, tags, reason in sorted(listing):
        print(f"{key:18s} {verdict:9s} grounded={grounded!s:5s} {', '.join(tags) or 'UNTAGGED'}\n{'':18s} judge: {reason}")
    print()
    for mode, counts in sorted(by_mode.items()):
        print(mode, dict(counts.most_common()))
    out = {mode: dict(c) for mode, c in by_mode.items()}
    json.dump(out, open(os.path.join(HERE, "results", "failure_taxonomy.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    main(ap.parse_args().results)
