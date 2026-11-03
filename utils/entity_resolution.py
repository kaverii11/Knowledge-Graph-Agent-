"""Alias resolution for circulars: the same document is written many ways.

    "Master Circular for FPIs, DDPs and Eligible Foreign Investors dated May 30, 2024"
    "Master Circular for Foreign Portfolio Investors, Designated Depository Participants and ..."
    "FPI Master Circular"
    "SEBI/HO/AFD/AFD-PoD2/P/CIR/P/2024/70"

String or embedding similarity can't link a circular number to its name, so circulars are matched on
evidence instead: the same circular number, the same date, or a short name made of the long name's
initials ("FPI" = Foreign Portfolio Investors). Merged documents keep every spelling in `aliases`.
"""
import re

from utils.circular_ids import find_circular_ids

DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})")
STOP = {"for", "of", "and", "the", "on", "in", "to", "a", "sebi", "vide", "dated", "no"}


def _date(text: str):
    m = DATE.search(text or "")
    return f"{m.group(3)}-{m.group(1)[:3]}-{int(m.group(2)):02d}" if m else None


def _kind(title: str) -> str:
    return "master" if "master circular" in title.lower() else "circular"


def _words(title: str) -> set:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in STOP}


def _acronyms(title: str) -> set:
    """'Foreign Portfolio Investors' -> 'fpi' (runs of 2+ capitalised words)."""
    out = set()
    for run in re.findall(r"(?:[A-Z][a-z]+\s+){1,5}[A-Z][a-z]+", title):
        out.add("".join(w[0] for w in run.split()).lower())
    return out


def _short_name_of(short: str, long: str) -> bool:
    """'FPI Master Circular' is a short name of 'Master Circular for Foreign Portfolio Investors, ...'."""
    s, l = _words(short), _words(long) | _acronyms(long)
    return len(s) >= 2 and s < l and bool(s - {"circular", "master"})


def evidence(ref: dict) -> dict:
    title, snippet = ref.get("title", ""), ref.get("context_snippet", "")
    ids_in_title = [c for c, _ in find_circular_ids(title)]
    ids_in_snip = list(dict.fromkeys(c for c, _ in find_circular_ids(snippet)))
    # Only trust a number from the snippet when it is the only one there
    number = ids_in_title[0] if ids_in_title else (ids_in_snip[0] if len(ids_in_snip) == 1 else None)
    date = _date(title) or (_date(snippet) if number or _kind(title) == "master" else None)
    # A bare circular number doesn't say whether it is a master circular, so it can match either
    kinds = {"master", "circular"} if ids_in_title and len(title) < 60 else {_kind(title)}
    return {"number": number, "date": date, "kinds": kinds}


def resolve_circulars(refs: list, existing: list | None = None) -> dict:
    """Returns {title: canonical_title} for every circular-type title in refs.

    existing: [{"name", "aliases", "number", "date"}] already in the graph, so documents seen in
    earlier circulars are reused instead of duplicated.
    """
    nodes = []  # each: {"titles": set, "numbers": set, "dates": set(kind,date), "existing_name": str|None}
    for e in existing or []:
        nodes.append({"titles": {e["name"], *(e.get("aliases") or [])},
                      "numbers": {e["number"]} if e.get("number") else set(),
                      "dates": {e["date"]} if e.get("date") else set(), "existing_name": e["name"]})
    for r in refs:
        if r.get("reference_type") != "circular":
            continue
        ev = evidence(r)
        nodes.append({"titles": {r["title"]}, "numbers": {ev["number"]} if ev["number"] else set(),
                      "dates": {(k, ev["date"]) for k in ev["kinds"]} if ev["date"] else set(), "existing_name": None})

    def same(a, b):
        if a["numbers"] & b["numbers"] or a["dates"] & b["dates"]:
            return True
        return any(_short_name_of(x, y) or _short_name_of(y, x) for x in a["titles"] for y in b["titles"])

    merged = True
    while merged:  # union until nothing changes
        merged = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if same(nodes[i], nodes[j]):
                    a, b = nodes[i], nodes.pop(j)
                    a["titles"] |= b["titles"]; a["numbers"] |= b["numbers"]; a["dates"] |= b["dates"]
                    a["existing_name"] = a["existing_name"] or b["existing_name"]
                    merged = True
                    break
            if merged:
                break

    mapping, meta = {}, {}
    for n in nodes:
        named = [t for t in n["titles"] if not list(find_circular_ids(t)) or len(t) > 60]
        canonical = n["existing_name"] or max(named or n["titles"], key=len)
        for t in n["titles"]:
            mapping[t] = canonical
        meta[canonical] = {"aliases": sorted(n["titles"] - {canonical}),
                           "number": next(iter(n["numbers"]), None),
                           "date": next(iter(n["dates"]), (None, None))[1]}
    return {"mapping": mapping, "meta": meta}
