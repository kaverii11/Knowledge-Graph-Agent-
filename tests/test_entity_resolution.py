from graph_db import lexical_similarity
from utils.entity_resolution import resolve_circulars

MC_NUMBER = "SEBI/HO/AFD/AFD-PoD2/P/CIR/P/2024/70"


def ref(title, snippet="", page=1):
    return {"title": title, "context_snippet": snippet, "reference_type": "circular", "page_number": page}


def test_name_number_and_short_name_merge_into_one_document():
    refs = [
        ref("Master Circular for FPIs, DDPs and Eligible Foreign Investors dated May 30, 2024"),
        ref(MC_NUMBER, f"... Eligible Foreign Investors” No. {MC_NUMBER} dated May 30, 2024 as amended ..."),
        ref("FPI Master Circular", "Para 9 (B)(iv) of Part B of the FPI Master Circular stands modified", page=2),
    ]
    out = resolve_circulars(refs)
    assert len(set(out["mapping"].values())) == 1
    meta = next(iter(out["meta"].values()))
    assert meta["number"] == MC_NUMBER and meta["date"] == "2024-May-30"


def test_matches_documents_already_in_graph():
    existing = [{"name": "Master Circular for FPIs, DDPs and Eligible Foreign Investors dated May 30, 2024",
                 "aliases": [], "number": None, "date": ("master", "2024-May-30")}]
    out = resolve_circulars([ref("FPI Master Circular")], existing)
    assert out["mapping"]["FPI Master Circular"] == existing[0]["name"]


def test_different_circulars_stay_separate():
    out = resolve_circulars([ref("SEBI Circular dated April 28, 2026"), ref("SEBI Circular dated November 25, 2025")])
    assert len(set(out["mapping"].values())) == 2


def test_lexical_fallback_never_merges_different_years():
    assert lexical_similarity("SEBI Act, 1992", "The SEBI Act 1992") > 0.9
    assert lexical_similarity("SEBI (ICDR) Regulations, 2018", "SEBI (LODR) Regulations, 2015") == 0.0
