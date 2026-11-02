import fitz  # PyMuPDF
import re
from typing import Iterator, Dict, Any, List

def is_structural_marker(text: str) -> bool:
    """
    Checks if a block of text starts with a SEBI structural marker.
    Examples: "1.", "1.1.", "1.1.1.", "Annexure A", "PART A"
    """
    text = text.strip()
    
    # Matches "1.", "1.1", "1.1.1", "1.1.1.", etc. A dot is required, so lines like "2023 ..." or "15 days" don't split chunks
    if re.match(r"^\d{1,3}\.(\d+\.?)*\s", text):
        return True
        
    # Matches "Annexure A", "Annexure-I", "PART I", "Appendix A"
    if re.match(r"^(Annexure|Part|Appendix)\s+[A-Z0-9\-]+", text, re.IGNORECASE):
        return True
        
    # Matches Roman numerals: "I.", "II.", "IV." (the lookahead stops an empty numeral from matching ". ")
    if re.match(r"^(?=[IVX])(X{0,3})(IX|IV|V?I{0,3})\.\s", text, re.IGNORECASE):
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
        # Check the first few pages for a text layer. If there's effectively none, it's a scanned PDF.
        full_doc_text = "".join([doc[i].get_text() for i in range(min(5, len(doc)))])
        is_scanned = len(full_doc_text.strip()) < 50

        if is_scanned:
            print("  [Parser] No native text layer detected. Sending page images to the vision model instead...")
            # Multimodal path: each page becomes an image chunk for GPT-4o vision
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150) # Render page as high-res image
                img_bytes = pix.tobytes("png")
                
                chunks.append({
                    "chunk_text": "",
                    "chunk_image_bytes": img_bytes,
                    "page_number": page_num + 1
                })
        else:
            # Native PyMuPDF Path (Block-based parsing)
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get blocks of text
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
            "chunk_image_bytes": chunk.get("chunk_image_bytes"),
            "context_text": context_text,
            "page_number": chunk["page_number"]
        }
        
        # Update context window only if there is text
        if chunk["chunk_text"]:
            context_chunks_text.append(chunk["chunk_text"])
            if len(context_chunks_text) > context_window_size:
                context_chunks_text.pop(0)
