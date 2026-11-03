import pytest

from evals.extraction_eval import score
from main import dedupe_per_page
from query_graph import is_read_only
from utils.circular_ids import detect_own_id, find_circular_ids


@pytest.mark.parametrize("q", [
    "MATCH (c:Circular) RETURN c.id",
    "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS 'set' RETURN e.name",  # 'set' inside a string is fine
])
def test_read_queries_allowed(q):
    assert is_read_only(q)


@pytest.mark.parametrize("q", [
    "MATCH (n) DETACH DELETE n",
    "CREATE (x:Entity {name: 'fake'})",
    "MATCH (e:Entity) SET e.name = 'x'",
    "merge (x:Circular {id: 1})",
    "MATCH (n) REMOVE n.name",
    "CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DELETE n', {})",
    "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
])
def test_write_queries_rejected(q):
    assert not is_read_only(q)


def test_circular_ids_old_and_new_format():
    text = ("No. SEBI/HO/AFD/AFD-PoD2/P/CIR/P/2024/70 dated May 30, 2024 and "
            "circular HO/17/11/12(3)2025-DDHS-POD1/ I/146/2025 dated November 25, 2025")
    assert [c for c, _ in find_circular_ids(text)] == [
        "SEBI/HO/AFD/AFD-PoD2/P/CIR/P/2024/70", "HO/17/11/12(3)2025-DDHS-POD1/I/146/2025"]


def test_own_number_detected_from_header():
    page1 = "CIRCULAR HO/(201)2026-DDHS-POD1/I/24467/2026 October 27, 2026 To, All Debenture Trustees ... HO/17/11/12(3)2025-DDHS-POD1/I/146/2025"
    assert detect_own_id(page1) == "HO/(201)2026-DDHS-POD1/I/24467/2026"


def test_dedupe_keeps_one_per_document_per_page():
    refs = [
        {"page_number": 1, "title": "SEBI Act, 1992", "confidence_score": 0.9},
        {"page_number": 1, "title": "the SEBI Act, 1992", "confidence_score": 0.8},
        {"page_number": 2, "title": "SEBI Act, 1992", "confidence_score": 0.9},
    ]
    out = dedupe_per_page(refs)
    assert len(out) == 2 and {r["page_number"] for r in out} == {1, 2}


def test_eval_scoring_counts_duplicates_separately():
    gt = [{"page_number": 1, "title": "SEBI Act, 1992", "aliases": ["Securities and Exchange Board of India Act, 1992"]}]
    pred = [{"page_number": 1, "title": "Securities and Exchange Board of India Act, 1992"},
            {"page_number": 1, "title": "SEBI Act, 1992"},
            {"page_number": 1, "title": "this circular"}]
    r = score(gt, pred)
    assert (r["correct"], r["duplicates"], r["wrong"], r["missed"]) == (1, 1, 1, 0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def test_write_hidden_after_a_string_is_still_caught():
    assert not is_read_only("MATCH (e) WHERE e.name = 'a' DELETE e")
    assert not is_read_only('MATCH (e) WHERE e.name = "it\'s" SET e.x = 1')
