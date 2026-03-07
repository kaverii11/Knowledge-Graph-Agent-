import re
import os
import json
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from schemas.models import ExtractedReference, ExtractionResult, ExtractionMethod, ReferenceType
from google import genai
from google.genai import types
from pydantic import ValidationError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

class GraphState(TypedDict):
    chunk_text: str
    chunk_image_bytes: Optional[bytes]
    context_text: str
    page_number: int
    references: List[ExtractedReference]
    validation_status: str
    errors: List[str]

def extract_regex(state: GraphState) -> GraphState:
    text = state["chunk_text"]
    references = state.get("references", [])
    
    if not text:
        return {"references": references}
        
    # Simple regex for SEBI circular formats like: SEBI/HO/MIRSD/MIRSD-PoD-1/P/CIR/2023/71
    sebi_pattern = r"(SEBI/[A-Z0-9/\-]+(?:CIR|Master)[/\w\-]+)"
    matches = re.finditer(sebi_pattern, text)
    
    for match in matches:
        ref_id = match.group(1)
        # Find some surrounding context
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        snippet = text[start:end].replace("\n", " ")
        
        # Avoid duplicate regex extractions
        if not any(r.title == ref_id for r in references):
            references.append(ExtractedReference(
                title=ref_id,
                reference_type=ReferenceType.CIRCULAR,
                page_number=state["page_number"],
                context_snippet=f"...{snippet}...",
                confidence_score=1.0,  # High confidence for regex matches
                extraction_method=ExtractionMethod.REGEX
            ))
            
    return {"references": references}

def get_llm_client():
    # Relies on GEMINI_API_KEY environment variable
    return genai.Client()

@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=15, max=70),
    retry=retry_if_exception_type(Exception)
)
def generate_content_with_retry(client, model, contents, config):
    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
    except Exception as e:
        print(f"DEBUG: Gemini API attempt failed. Error: {e}")
        raise

def extract_fast_llm(state: GraphState) -> GraphState:
    client = get_llm_client()
    text = state["chunk_text"]
    context = state["context_text"]
    image_bytes = state.get("chunk_image_bytes")
    references = state.get("references", [])
    
    prompt = f"""
    You are a legal AI assistant. Extract all mentions of other legal documents, circulars, acts, or regulations from the document.
    Do not duplicate standard SEBI circular IDs if you see them, but extract names of Acts, Regulations, or non-standard references.
    
    Context from previous sections (use for resolving pronouns like 'the aforementioned act'):
    {context}
    """
    
    if text:
        prompt += f"\n\nCurrent Text Chunk to process:\n{text}"
    else:
        prompt += "\n\nPlease extract references from the attached document image."
    
    contents = [prompt]
    if image_bytes:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
    
    try:
        response = generate_content_with_retry(
            client=client,
            model='gemini-1.5-flash', # Pivot to 1.5-flash to bypass 2.0/2.5 strict quotas
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.1,
            ),
        )
        
        if response.text:
            result_dict = json.loads(response.text)
            parsed_result = ExtractionResult(**result_dict)
            
            # Append fast LLM results, tagging them properly
            for ref in parsed_result.references:
                ref.extraction_method = ExtractionMethod.FAST_LLM
                ref.page_number = state["page_number"] # Ensure page number is correct
                references.append(ref)
                
    except Exception as e:
        return {"errors": [f"Fast LLM Error: {str(e)}"]}
        
    return {"references": references}

def validate_extraction(state: GraphState) -> GraphState:
    references = state.get("references", [])
    errors = state.get("errors", [])
    
    low_confidence_count = 0
    for ref in references:
        if ref.confidence_score < 0.7:
            low_confidence_count += 1
            errors.append(f"Low confidence ({ref.confidence_score}) for reference: {ref.title}")
            
    if low_confidence_count > 0 or len(errors) > 0:
        return {"validation_status": "invalid", "errors": errors}
        
    return {"validation_status": "valid"}

def refine_pro(state: GraphState) -> GraphState:
    # If the fast LLM failed or had low confidence, try with Pro model
    client = get_llm_client()
    text = state["chunk_text"]
    context = state["context_text"]
    image_bytes = state.get("chunk_image_bytes")
    
    prompt = f"""
    The previous extraction had validation issues or low confidence. Please carefully extract all legal references from this document.
    Ensure high accuracy.
    
    Context: {context}
    """
    
    if text:
        prompt += f"\n\nText: {text}"
    else:
        prompt += "\n\nPlease carefully extract references from the attached document image."
    
    contents = [prompt]
    if image_bytes:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
    
    try:
        response = generate_content_with_retry(
            client=client,
            model='gemini-1.5-pro', # Pivot to 1.5-pro for fallback
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.1,
            ),
        )
        
        if response.text:
            result_dict = json.loads(response.text)
            parsed_result = ExtractionResult(**result_dict)
            
            # We completely replace the non-regex references with Pro results
            # Filter out FAST_LLM references, keep REGEX ones
            new_refs = [r for r in state.get("references", []) if r.extraction_method == ExtractionMethod.REGEX]
            
            for ref in parsed_result.references:
                ref.extraction_method = ExtractionMethod.PRO_FALLBACK
                ref.page_number = state["page_number"]
                new_refs.append(ref)
                
            return {"references": new_refs, "validation_status": "valid", "errors": []}
            
    except Exception as e:
        # If it still fails, just return what we have and keep the error
        errors = state.get("errors", [])
        errors.append(f"Pro Fallback Error: {str(e)}")
        return {"errors": errors}
        
    return {}

def route_after_validation(state: GraphState) -> Literal["refine_pro", "__end__"]:
    if state.get("validation_status") == "invalid":
        return "refine_pro"
    return "__end__"

def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("extract_regex", extract_regex)
    workflow.add_node("extract_fast_llm", extract_fast_llm)
    workflow.add_node("validate_extraction", validate_extraction)
    workflow.add_node("refine_pro", refine_pro)
    
    # Build edges
    workflow.set_entry_point("extract_regex")
    workflow.add_edge("extract_regex", "extract_fast_llm")
    workflow.add_edge("extract_fast_llm", "validate_extraction")
    workflow.add_conditional_edges("validate_extraction", route_after_validation)
    workflow.add_edge("refine_pro", END)
    
    return workflow.compile()
