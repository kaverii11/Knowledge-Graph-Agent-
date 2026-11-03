# Evals

## Extraction eval (`extraction_eval.py`)
Hand-labelled answer keys for three real SEBI circulars (Aug-Oct 2026) live in `ground_truth/`.
Labelling rules:
- One entry per **document explicitly named on a page** (an act, regulation, circular or master circular).
- A provision inside an already-referenced document ("Regulation 2A of the DT Regulations") counts as that document.
- Not references: the circular itself ("this circular"), bare provision numbers with no document ("Regulation 9C"), people, websites.
- `aliases` hold other spellings of the same document (short names, circular numbers).

A prediction is **correct** if it matches an unmatched answer on the same page, a **duplicate** if it
matches an answer that was already found, and **wrong** otherwise. Precision = correct / (correct + wrong);
duplicates are reported separately.

## Agent eval (`run_agent_evals.py`)
Questions with expected answers about the knowledge graph (`qa_dataset.json`). For each question the
text-to-Cypher agent runs end to end and is scored on:
- **query_ok**: the generated Cypher is read-only and runs without error
- **retrieval_hit**: the rows returned contain the expected document/circular
- **correct**: an LLM judge compares the answer with the expected answer (correct / partial / incorrect)
- **grounded**: the LLM judge checks every claim in the answer is supported by the returned rows (hallucination check)

## Tool-selection eval (`tool_selection_eval.py`)
Gives an LLM the MCP server's tool list and a question, and checks it calls the expected tool.

`pytest -m llm` re-runs the agent and tool-selection evals and fails if scores drop below the saved baseline.
