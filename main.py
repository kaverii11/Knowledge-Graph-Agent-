import argparse
import json
import os
import warnings
from dotenv import load_dotenv

# Suppress the false-positive LangChain deprecation warning for Python 3.14
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

from utils.pdf_parser import parse_pdf_structural
from agent.graph import build_graph
from utils.tracing import setup_tracing
import re
from utils.circular_ids import detect_own_id
from utils.entity_resolution import DATE
from evals.extraction_eval import is_match


def dedupe_per_page(refs: list) -> list:
    """The same document is often mentioned in several chunks of one page: keep one entry per
    document per page (the most complete title, highest confidence)."""
    kept = []
    for r in sorted(refs, key=lambda x: (-len(x["title"]), -x["confidence_score"])):
        if not any(k["page_number"] == r["page_number"] and is_match(r["title"], k["title"]) for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda x: x["page_number"])

def process_pdf(file_path: str, output_path: str):
    print(f"Starting extraction for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    setup_tracing()
    # 1. Initialize Graph
    graph = build_graph()
    
    # 2. Parse PDF into structural chunks
    print("Parsing PDF...")
    chunks = list(parse_pdf_structural(file_path, context_window_size=2))
    
    print(f"Extracted {len(chunks)} structural chunks.")
    
    if len(chunks) == 0:
        print("Halting: the parser couldn't read the PDF structure.")
        return
        
    all_extracted_references = []
    first_page = " ".join(c["chunk_text"] for c in chunks if c["page_number"] == 1)
    own_id = detect_own_id(first_page)
    print(f"Circular number: {own_id or 'not found'}")
    # The subject line runs until the first numbered paragraph ("1. On October ...")
    subject = re.search(r"(?:Subject|Sub)\s*:\s*(.+?)(?=\s+\d{1,2}\.\s+[A-Z])", first_page, re.S)
    issued = DATE.search(first_page)
    meta = {"circular_id": own_id, "source_pdf": os.path.basename(file_path),
            "subject": " ".join(subject.group(1).split()) if subject else None,
            "date": issued.group(0) if issued else None}
    
    for i, chunk_data in enumerate(chunks):
        print(f"Processing chunk {i+1} on page {chunk_data['page_number']}...")
        
        # 3. Create initial state
        initial_state = {
            "chunk_text": chunk_data["chunk_text"],
            "chunk_image_bytes": chunk_data.get("chunk_image_bytes"),
            "context_text": chunk_data["context_text"],
            "page_number": chunk_data["page_number"],
            "references": [],
            "validation_status": "valid",
            "errors": [],
            "self_id": own_id,
        }
        
        # 4. Run LangGraph
        result_state = graph.invoke(initial_state)
        
        # 5. Collect resulting references
        refs = result_state.get("references", [])
        for ref in refs:
            # ref is an ExtractedReference Pydantic model
            all_extracted_references.append(ref.model_dump(mode="json"))
            
        if result_state.get("errors"):
            print(f"  Warning: Errors encountered during processing: {result_state['errors']}")
            
        print(f"  Extracted {len(refs)} references from this chunk.")
        
    # 6. Save final structured data
    raw_count = len(all_extracted_references)
    all_extracted_references = dedupe_per_page(all_extracted_references)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted_references, f, indent=2)
    with open(output_path.replace(".json", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Extraction complete! Saved {len(all_extracted_references)} references "
          f"({raw_count - len(all_extracted_references)} duplicates merged) to {output_path}")
    print(f"Next: python graph_db.py {output_path}")

if __name__ == "__main__":
    load_dotenv() # Load variables from .env
    
    parser = argparse.ArgumentParser(description="SEBI Circular Entity Extractor")
    parser.add_argument("pdf_path", type=str, help="Path to the SEBI circular PDF")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    process_pdf(args.pdf_path, args.output)
