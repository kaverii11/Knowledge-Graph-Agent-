import fitz  # PyMuPDF
import re
from typing import Iterator, Dict, Any, List

def is_structural_marker(text: str) -> bool:
    """
    Checks if a block of text starts with a SEBI structural marker.
    Examples: "1.", "1.1.", "1.1.1.", "Annexure A", "PART A"
    """
    text = text.strip()
    
    # Matches "1.", "1.1", "1.1.1", "1.1.1.", etc.
    if re.match(r"^\d+(\.\d+)*\.?\s", text):
        return True
        
    # Matches "Annexure A", "Annexure-I", "PART I", "Appendix A"
    if re.match(r"^(Annexure|Part|Appendix)\s+[A-Z0-9\-]+", text, re.IGNORECASE):
        return True
        
    # Matches Roman numerals: "I.", "II.", "IV."
    if re.match(r"^(X{0,3})(IX|IV|V?I{0,3})\.\s", text, re.IGNORECASE):
        return True
        
    return False

def parse_pdf_structural(file_path: str, context_window_size: int = 2) -> Iterator[Dict[str, Any]]:
    """
    Parses a PDF file into structural chunks based on SEBI markers.
    Maintains a sliding window of previous chunks for context resolution.
    
    Yields dicts with:
        - chunk_text: The main text of the structural chunk
        - context_text: Text from the previous N chunks
        - page_number: The starting page number of the chunk
    """
    current_chunk_text = ""
    current_chunk_start_page = 1
    chunks: List[Dict[str, Any]] = []

    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get blocks of text. block is a tuple: (x0, y0, x1, y1, "text", block_no, block_type)
            blocks = page.get_text("blocks")
            
            for block in blocks:
                # Check if block is text (block_type == 0)
                if len(block) >= 7 and block[6] == 0:
                    text = block[4].strip()
                    if not text:
                        continue
                    
                    # If we find a structural marker, we flush the current chunk
                    if is_structural_marker(text):
                        if current_chunk_text.strip():
                            chunks.append({
                                "chunk_text": current_chunk_text.strip(),
                                "page_number": current_chunk_start_page
                            })
                        # Start a new chunk
                        current_chunk_text = text + "\n"
                        current_chunk_start_page = page_num + 1 # 1-indexed
                    else:
                        # Append to current chunk
                        if not current_chunk_text:
                            current_chunk_start_page = page_num + 1
                        current_chunk_text += text + "\n"
                        
    # Flush the last chunk
    if current_chunk_text.strip():
        chunks.append({
            "chunk_text": current_chunk_text.strip(),
            "page_number": current_chunk_start_page
        })
        
    # Yield chunks with context window
    context_chunks_text = []
    for i, chunk in enumerate(chunks):
        context_text = "\n\n".join(context_chunks_text)
        
        yield {
            "chunk_text": chunk["chunk_text"],
            "context_text": context_text,
            "page_number": chunk["page_number"]
        }
        
        # Update context window
        context_chunks_text.append(chunk["chunk_text"])
        if len(context_chunks_text) > context_window_size:
            context_chunks_text.pop(0)
