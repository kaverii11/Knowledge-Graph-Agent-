# 📋 Instructions — How to Start the Project

## Prerequisites

- **Python 3.11+**
- **Neo4j** database instance (local or [Neo4j Aura](https://neo4j.com/cloud/aura/))
- **Tesseract OCR** (required by `pytesseract`)
  ```bash
  # macOS
  brew install tesseract

  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr
  ```

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Knowledge-Graph-Agent-.git
cd Knowledge-Graph-Agent-
```

---

## 2. Create & Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file in the project root with the following keys:

```env
GEMINI_API_KEY=your_google_genai_key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=your_database
```

---

## 5. Run the Project

### Step 1 — Ingest & Extract (PDF → JSON)

Parse a SEBI circular PDF and export structured references:

```bash
python main.py data/<your_circular>.pdf --output sebi_references.json
```

**Example:**
```bash
python main.py "data/SEBI _ Guidelines for Custodians.pdf" --output sebi_references.json
```

### Step 2 — Materialize the Knowledge Graph (JSON → Neo4j)

Load the extracted JSON data into Neo4j:

```bash
python graph_db.py
```

### Step 3 — Query the Graph (GraphRAG)

Launch the interactive text-to-Cypher compliance terminal:

```bash
python query_graph.py
```

> **Example query:** *"Which circulars modify the SEBI Act?"*

### Step 4 — Evaluate Performance (Optional)

Run precision/recall/F1 evaluation against a ground-truth file:

```bash
python evals.py <ground_truth.json> <extracted_references.json>
```

**Example:**
```bash
python evals.py scores_references.json sebi_references.json
```

---

## Quick-Reference (All Commands)

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python main.py data/<circular>.pdf --output sebi_references.json
python graph_db.py
python query_graph.py

# Evaluate
python evals.py <ground_truth>.json <output>.json
```
