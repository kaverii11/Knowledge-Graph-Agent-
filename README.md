# SEBI Intelligence Engine: Knowledge Graph Agent

An enterprise-grade, evaluation-driven AI architecture designed for the compliance team of regulatory bodies and financial institutions. This system automates the ingestion, entity extraction, and knowledge harmonization of SEBI (Securities and Exchange Board of India) circulars into a queryable Neo4j Knowledge Graph.

## 🏗 System Architecture

The project implements a modern retrieval and orchestration flow using **LangGraph**, **Gemini**, and **Neo4j**.

1. **Intelligent Ingestion (`utils/pdf_parser.py`)**  
   Reads raw PDF circulars via PyMuPDF. It bypasses arbitrary character-count chunking in favor of **Structural Chunking**—identifying SEBI-domain markers (e.g., `1.1`, `Annexure A`) and isolating logical clauses while tracking precise page numbers and maintaining 2-hop contextual sliding windows to resolve co-references.

2. **Triad Extraction Orchestrator (`agent/graph.py`)**  
   A LangGraph `StateGraph` routes text chunks through a sequential three-tier extraction pipeline:
   - **Step A (Regex Filter)**: Extracts highly standard SEBI hierarchical identifiers (e.g., `SEBI/HO/MIRSD/...`).
   - **Step B (Fast LLM)**: Defers complex entity recognition to Gemini 1.5 Flash, strictly adhering to Pydantic definitions (Titles, Types, Context Snippets).
   - **Step C (Pro Fallback)**: A conditional edge that triggers if validation fails or confidence dips. Reroutes the chunk to Gemini 1.5 Pro for meticulous extraction.

3. **Graph Materialization (`graph_db.py`)**  
   Idempotently ingests the structured output into a **Neo4j** graph database using Cypher `MERGE` queries.
   - **Nodes**: `(Circular)`, `(Clause)`, `(Entity)`
   - **Edges**: `[:CONTAINS]`, `[:REFERENCES]`, `[:MODIFIES]` (heuristically evaluated from context snippets).

4. **Text-to-Cypher GraphRAG (`query_graph.py`)**  
   A natural language interface allowing Compliance Officers to interrogate the database. Gemini 1.5 Pro dynamically translates regulatory questions into Neo4j Cypher queries, executes them against the schema, and synthesizes the JSON response into a highly readable legal summary via Gemini 1.5 Flash.

5. **Evaluation-Driven Development (`evals.py`)**  
   A dedicated framework for calculating Precision, Recall, and F1-Scores against manually annotated ground-truth datasets, ensuring the extraction metrics remain continuously calibrated.

---

## 📂 Repository Structure

```text
├── agent/
│   └── graph.py             # LangGraph triad routing logic + prompts
├── schemas/
│   └── models.py            # Pydantic schemas (ExtractionResult, ReferenceType)
├── utils/
│   └── pdf_parser.py        # PyMuPDF structural chunking and sliding windows
├── main.py                  # CLI Entry point for PDF Ingestion -> JSON
├── graph_db.py              # Neo4j Graph populator
├── query_graph.py           # Text-to-Cypher GraphRAG app
├── evals.py                 # F1/Precision/Recall metric calculator
├── requirements.txt         # Project dependency definitions
└── .env                     # Secrets configuration (API Keys, Neo4j URIs)
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Ensure you are running **Python 3.11+**. Set up your virtual environment and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create a `.env` file at the repository root and provision your credentials:

```env
GEMINI_API_KEY=your_google_genai_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
```

### 3. Execution Flow

**Step 1: Ingest & Extract**  
Parse a SEBI PDF circular and export the structured references to JSON.
```bash
python main.py data/sample_sebi_circular.pdf --output sebi_references.json
```

**Step 2: Materialize the Knowledge Graph**  
Load the serialized JSON data into Neo4j.
```bash
python graph_db.py
```

**Step 3: Interrogate the Data (GraphRAG)**  
Launch the interactive compliance terminal.
```bash
python query_graph.py
```
*Example Question: "Which circulars modify the SEBI Act?"*

**Step 4: Evaluate Performance (Optional)**  
Run the pipeline against a human-annotated ground truth file.
```bash
python evals.py sample_ground_truth.json sebi_references.json
```

---
*Built for advanced AI regulatory and compliance workflows.*
