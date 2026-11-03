# SEBI Knowledge Graph Agent

Turns SEBI (Securities and Exchange Board of India) circulars, which are long and unstructured PDFs, into a **Neo4j knowledge graph** of circulars and the acts, regulations and other circulars they reference or modify. A compliance officer, or another AI agent over **MCP**, can then ask questions like "Which circulars modify the FPI Master Circular?" and get answers backed by the graph.

**Stack:** Python · LangGraph · OpenAI-compatible LLMs (OpenAI or Groq) · Pydantic · Neo4j · **MCP** · **MLflow tracing** · LLM-as-a-judge evals · pytest · Docker · Streamlit

```
PDF ─► page-safe chunking ─► LangGraph extraction ─► alias + entity resolution ─► Neo4j ─┬─► text-to-Cypher Q&A (CLI, Streamlit)
       (utils/pdf_parser)     (agent/graph)            (utils/entity_resolution)          └─► MCP server: 6 read-only tools
                    every step traced in MLflow · scored by 3 evals · regression-gated with pytest
```

## Results (3 real SEBI circulars, Aug-Oct 2026)

**Extraction** (`evals/extraction_eval.py`, 17 hand-labelled references):

| Version | Precision | Recall | F1 | Duplicates | references vs modifies |
|---|---|---|---|---|---|
| Before | 0.59 | 1.00 | 0.74 | 18 | 15/17 |
| After | **0.89** | **1.00** | **0.94** | 1 | **16/17** |

What the eval caught, and the fix for each:
- The model pulled references out of the *context window* (previous chunks) and tagged them with the current page → prompt rule: context is only for resolving short names.
- Some PDFs put a whole page in one text block, so clause markers were missed and page 2 references were labelled page 1 → chunker now works line by line and never crosses a page break.
- "This circular" and bare provision numbers counted as references → explicit extraction rules.
- SEBI's new circular-number format (`HO/19/34/14(10)2026-AFD-POD3/I/24814/2026`) wasn't recognised, and a circular's own number was counted as a reference to itself → new regex + own-number detection.
- `"as amended from time to time"` triggered false MODIFIES links → the model now labels each reference `references` / `modifies`.
- The model's self-reported confidence was 0.95-1.0 on right *and* wrong answers, so the confidence threshold alone can't be trusted: the eval tracks confidence on right vs wrong answers.

**Agent Q&A** (`evals/run_agent_evals.py`, 12 questions, LLM-as-a-judge):

| Version | Valid read-only query | Retrieved the right rows | Correct (judge) | Grounded (no hallucination) |
|---|---|---|---|---|
| v1 | 0.92 | 0.67 | 0.54 | 0.92 |
| v3 | **1.00** | **1.00** | **0.88** | 0.92 |

Changes: the query writer now sees each circular's subject and date and each document's short names and circular number; it matches on 1-3 keywords plus aliases instead of every word of a title; "SEBI Act" aliases are added for "Securities and Exchange Board of India Act"; LLM calls retry on rate limits. Note: between v1 and v3 the query model also moved from `gpt-oss-20b` to `gpt-oss-120b` and the judge to a different model family (Qwen, so the model isn't grading itself), so not all of the gain is from the prompt. One remaining miss is a data error, not an agent error (one MODIFIES link was mislabelled during extraction).

**Tool selection** (`evals/tool_selection_eval.py`): given the MCP server's real tool list, the LLM picks the expected tool for **10/12** requests, and calls no tool for "Delete all circulars".

These are small, hand-labelled sets: useful for catching regressions and comparing versions, not a benchmark.

## How it works

### 1. Chunking (`utils/pdf_parser.py`)
Splits each page line by line on the circular's own structure (`1.`, `2.1.`, `Annexure`, `Part`, Roman numerals). A chunk never crosses a page break, so every reference keeps its real page. Each chunk carries the previous 2 chunks as context. Scanned PDFs (no text layer) are sent page by page to a vision model.

### 2. Extraction graph (`agent/graph.py`)
A LangGraph `StateGraph` per chunk:

| Node | What it does |
|---|---|
| `extract_regex` | Circular numbers in the old (`SEBI/HO/.../CIR/2023/71`) and new (`HO/.../I/24814/2026`) formats, skipping the circular's own number |
| `extract_fast_llm` | Fast model returns **Pydantic-validated structured output**: title, type, `references`/`modifies`, evidence, confidence |
| `validate_extraction` | Flags the chunk if any reference is below 0.7 confidence or the call failed |
| `refine_pro` | Conditional edge: flagged chunks are re-extracted by the stronger model (regex hits are kept) |

Only transient errors (rate limits, timeouts, 5xx) are retried, with exponential backoff. Duplicates of the same document on a page are merged afterwards.

### 3. Entity resolution + graph (`utils/entity_resolution.py`, `graph_db.py`)
The same document is written many ways: *"Master Circular for FPIs, DDPs and Eligible Foreign Investors dated May 30, 2024"*, *"FPI Master Circular"*, *"SEBI/HO/AFD/AFD-PoD2/P/CIR/P/2024/70"*. Similarity scores can't link a circular number to a name, so circulars are matched on evidence: same circular number, same date, or a short name built from the long name's initials. This works across circulars already in the graph. Every spelling is kept in `aliases`. Acts and regulations are merged by embedding similarity, or by strict string similarity (never across different years) when the provider has no embedding model.

Graph: `(:Circular {id, subject, date})-[:CONTAINS]->(:Clause {text, page_number})-[:REFERENCES|MODIFIES]->(:Entity {name, type, number, aliases})`. Loading uses `MERGE`, so re-running never creates duplicates.

### 4. Q&A agent (`query_graph.py`, `app.py`)
The LLM writes Cypher from a live inventory of the graph (circulars with subjects and dates, documents with aliases and numbers). Two safety layers: write clauses are rejected (string literals and comments are ignored, so searching for the word "set" still works), and the query runs in a **read transaction** that Neo4j itself enforces. The answer is written only from the returned rows.

### 5. MCP server (`mcp_server.py`)
Exposes the graph to any MCP client (Claude Desktop, Cursor, an agent framework):

| Tool | Purpose |
|---|---|
| `list_circulars` | Every circular: number, subject, date, documents cited |
| `search_documents` | Find an act / regulation / circular by name, short name or number |
| `circular_references` | Everything one circular references or modifies, with page and evidence |
| `who_cites` | Which circulars reference or modify a document |
| `run_cypher` | Custom read-only Cypher (write queries rejected) |
| `ask` | Plain-English question → the text-to-Cypher agent |

All tools are annotated read-only, inputs are validated (lengths, allowed values), fixed queries use parameters (no string building), errors come back as clear tool errors, and every call is traced.

### 6. Observability (`utils/tracing.py`)
MLflow tracing records every LangGraph node, LLM call (prompt, response, token usage, latency), database query, MCP tool call and judge call. Eval runs are logged as MLflow runs with their metrics, so versions can be compared.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://localhost:5000
```

### 7. Evals and tests (`evals/`, `tests/`)
- `evals/extraction_eval.py`: precision / recall / F1 against hand-labelled answers, duplicates, `references` vs `modifies` accuracy, confidence on right vs wrong answers. Labelling rules in `evals/README.md`.
- `evals/run_agent_evals.py`: end-to-end Q&A with four scores, two of them from an LLM judge (correctness, grounded).
- `evals/tool_selection_eval.py`: does an LLM pick the right MCP tool?
- `pytest`: 23 unit tests, no API key or database needed (guards, ID parsing, entity resolution, eval scoring, the MCP server through a real MCP client with a fake database).
- `pytest -m llm`: regression gate. Re-runs the agent and tool-selection evals and fails if a score drops more than 0.10 below `evals/baseline.json`.

## Setup

Requires Python 3.11+, Docker, and an OpenAI key or a free Groq key.

```bash
git clone https://github.com/kaverii11/Knowledge-Graph-Agent-.git
cd Knowledge-Graph-Agent-
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in an LLM key and a Neo4j password
docker compose up -d            # Neo4j on bolt://localhost:7687, browser on http://localhost:7474
```

## Usage

```bash
# 1. PDF -> references (JSON) + detected circular number, subject, date
python main.py data/fpi_kyc_review_2026.pdf --output data/fpi_kyc_review_2026.json

# 2. JSON -> Neo4j (uses the detected circular number)
python graph_db.py data/fpi_kyc_review_2026.json

# 3. Ask questions
python query_graph.py            # terminal
streamlit run app.py             # web
python mcp_server.py             # MCP over stdio (or --http --port 8765)

# 4. Evals and tests
python evals/extraction_eval.py --gt-dir evals/ground_truth --pred-dir data
python evals/run_agent_evals.py
python evals/tool_selection_eval.py
pytest                           # unit tests
pytest -m llm                    # regression gate (needs the LLM key + graph)
```

**Claude Desktop:** add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sebi-knowledge-graph": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/Knowledge-Graph-Agent-/mcp_server.py"]
    }
  }
}
```

## Project structure

```text
├── agent/graph.py               # LangGraph extraction (regex -> fast LLM -> validate -> stronger-model fallback)
├── schemas/models.py            # Pydantic schemas for LLM output and stored references
├── utils/pdf_parser.py          # Page-safe structural chunking + scanned-PDF image path
├── utils/circular_ids.py        # Old + new SEBI circular-number formats, own-number detection
├── utils/entity_resolution.py   # Alias resolution for circulars (number / date / short-name evidence)
├── utils/llm.py                 # One place to configure the provider and models
├── utils/tracing.py             # MLflow tracing setup
├── utils/neo4j_client.py        # Shared Neo4j connection (reconnects, read-only mode)
├── main.py                      # CLI: PDF -> JSON
├── graph_db.py                  # Entity resolution + Neo4j ingestion
├── query_graph.py               # Text-to-Cypher agent (terminal)
├── mcp_server.py                # MCP server: 6 read-only tools
├── app.py                       # Streamlit chat UI
├── evals/                       # Answer keys, datasets, 3 eval scripts, results, baseline
├── tests/                       # Unit tests + LLM regression gate
├── data/                        # Extraction outputs (PDF download links in data/README.md)
└── docker-compose.yml           # Local Neo4j
```
