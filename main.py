import argparse
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from utils.pdf_parser import parse_pdf_structural
from agent.graph import build_graph, GraphState

def process_pdf(file_path: str, output_path: str):
    print(f"Starting extraction for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    # 1. Initialize Graph
    graph = build_graph()
    
    # 2. Parse PDF into structural chunks
    chunks_iterator = parse_pdf_structural(file_path, context_window_size=2)
    
    all_extracted_references = []
    
    for i, chunk_data in enumerate(chunks_iterator):
        print(f"Processing chunk {i+1} on page {chunk_data['page_number']}...")
        
        # 3. Create initial state
        initial_state = {
            "chunk_text": chunk_data["chunk_text"],
            "context_text": chunk_data["context_text"],
            "page_number": chunk_data["page_number"],
            "references": [],
            "validation_status": "valid",
            "errors": []
        }
        
        # 4. Run LangGraph
        config = {"configurable": {"thread_id": str(i)}}
        result_state = graph.invoke(initial_state, config=config)
        
        # 5. Collect resulting references
        refs = result_state.get("references", [])
        for ref in refs:
            # ref is an ExtractedReference Pydantic model
            all_extracted_references.append(ref.model_dump())
            
        if result_state.get("errors"):
            print(f"  Warning: Errors encountered during processing: {result_state['errors']}")
            
        print(f"  Extracted {len(refs)} references from this chunk.")
        
    # 6. Save final structured data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted_references, f, indent=2)
        
    print(f"Extraction complete! Saved {len(all_extracted_references)} references to {output_path}")

if __name__ == "__main__":
    load_dotenv() # Load variables from .env
    
    parser = argparse.ArgumentParser(description="SEBI Circular Entity Extractor")
    parser.add_argument("pdf_path", type=str, help="Path to the SEBI circular PDF")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    process_pdf(args.pdf_path, args.output)
