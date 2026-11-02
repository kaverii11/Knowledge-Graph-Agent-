# SEBI Knowledge Graph Agent

Turns SEBI (Securities and Exchange Board of India) circulars, which are long and unstructured PDFs, into a **Neo4j knowledge graph** of circulars, clauses, and the acts, regulations, and other circulars they reference or modify. A compliance officer can then ask questions in plain English ("Which circulars modify the SEBI Act?") and get answers backed by the graph.

**Stack:** Python · LangGraph · OpenAI (GPT-4o-mini, GPT-4o, text-embedding-3-small) · Pydantic · Neo4j · PyMuPDF · Streamlit

## How it works

```
PDF ──► structural chunking ──► LangGraph extraction ──► entity resolution ──► Neo4j ──► text-to-Cypher Q&A
        (utils/pdf_parser.py)    (agent/graph.py)         (graph_db.py)                   (query_graph.py, app.py)
```

### 1. Structural chunking (`utils/pdf_parser.py`)
- Reads the PDF with PyMuPDF and splits it on the document's own structure (clause numbers like `1.`, `1.1`, `2.3.1.`, plus `Annexure`, `Part`, `Appendix` and Roman numerals), not on a fixed character count, so each chunk is a whole clause.
- Each chunk carries the text of the previous 2 chunks as context, so the model can resolve phrases like "the aforementioned Act".
- **Scanned PDFs:** if the first pages have no text layer, each page is rendered to an image and sent to GPT-4o vision instead.

### 2. Extraction pipeline (`agent/graph.py`)
A LangGraph `StateGraph` runs on every chunk:

| Step | What it does |
|---|---|
| `extract_regex` | Catches standard SEBI circular IDs (e.g. `SEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2023/71`) with a regex, at confidence 1.0 |
| `extract_fast_llm` | GPT-4o-mini extracts the remaining references (acts, regulations, non-standard references) as **Pydantic-validated structured output**, each with a confidence score |
| `validate_extraction` | Flags the chunk if any reference is below **0.7 confidence** or the call failed |
| `refine_pro` | Conditional edge: flagged chunks are re-extracted by **GPT-4o**, which replaces the fast model's results (regex hits are kept) |

Only transient OpenAI errors (rate limits, timeouts, 5xx) are retried, with exponential backoff; bad requests fail fast and go to the fallback.

### 3. Entity resolution + graph (`graph_db.py`)
- Different spellings of the same entity ("SEBI Act, 1992" and "Securities and Exchange Board of India Act, 1992") are merged by **embedding cosine similarity** (threshold 0.88). Strict SEBI circular IDs skip this step.
- Loaded into Neo4j with idempotent `MERGE` queries, so re-running ingestion doesn't create duplicates.
  - Nodes: `(:Circular)`, `(:Clause)`, `(:Entity)`
  - Edges: `[:CONTAINS]`, `[:REFERENCES]`, `[:MODIFIES]` (a clause `MODIFIES` an entity when its text uses amendment verbs like *amend*, *substitute*, *repeal*)

### 4. Question answering (`query_graph.py`, `app.py`)
- GPT-4o-mini translates the question into Cypher. The prompt includes the **live graph inventory** (relationship types, circular IDs, entity names), so the model searches for names that actually exist.
- Generated Cypher is **read-only**: write clauses are rejected and the query runs in a read transaction.
- GPT-4o-mini turns the query results into a short answer.
- Available as a terminal chat (`query_graph.py`) or a Streamlit chat app (`app.py`) that shows the generated Cypher.

### 5. Evaluation (`evals.py`)
Scores extraction against a hand-labelled ground-truth file: precision, recall, F1 (matching on page number plus a fuzzy title match), and the average confidence of correct vs. incorrect extractions.

## Setup

Requires Python 3.11+, an OpenAI API key, and a Neo4j database (local or [Neo4j Aura](https://neo4j.com/cloud/aura/)).

```bash
git clone https://github.com/kaverii11/Knowledge-Graph-Agent-.git
cd Knowledge-Graph-Agent-
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

## Usage

```bash
# 1. PDF -> structured references (JSON)
python main.py "data/<circular>.pdf" --output sebi_references.json

# 2. JSON -> Neo4j knowledge graph
python graph_db.py sebi_references.json --circular-id <circular_id>

# 3. Ask questions (terminal or web)
python query_graph.py
streamlit run app.py

# 4. Evaluate against a ground-truth file (optional)
python evals.py ground_truth.json sebi_references.json
```

## Project structure

```text
├── agent/graph.py          # LangGraph extraction pipeline (regex -> fast LLM -> validate -> pro fallback)
├── schemas/models.py       # Pydantic schemas for LLM output and stored references
├── utils/pdf_parser.py     # Structural chunking + scanned-PDF image path
├── utils/neo4j_client.py   # Shared Neo4j connection (reconnects on stale sessions, read-only mode)
├── main.py                 # CLI: PDF -> JSON
├── graph_db.py             # Entity resolution + Neo4j ingestion
├── query_graph.py          # Text-to-Cypher Q&A (terminal)
├── app.py                  # Streamlit chat UI
├── evals.py                # Precision / recall / F1
└── .env.example            # Required environment variables
```
