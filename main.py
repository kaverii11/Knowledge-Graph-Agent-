import argparse
import json
import os
import warnings
from dotenv import load_dotenv

# Suppress the false-positive LangChain deprecation warning for Python 3.14
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

from utils.pdf_parser import parse_pdf_structural
from agent.graph import build_graph

def process_pdf(file_path: str, output_path: str):
    print(f"Starting extraction for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
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
            "errors": []
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
